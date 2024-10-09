from sqlite3 import NotSupportedError
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForPreTraining, AutoModelForMaskedLM, 
    LlamaTokenizer, LlamaForCausalLM, OPTForCausalLM
    )
import numpy as np
import random
import argparse
import logging
import os
from tqdm import tqdm
import json
import sys
from dataset import WGDataset, BBQDataset
from utils import set_random_seed
from model_utils import get_params_map, get_all_model_grads, accumulate_grad
from consts import PAD_TOKEN, MASK_TOKEN
from partition_params import create_param_partition
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from accelerate import Accelerator, FullyShardedDataParallelPlugin


import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# NOTE: this is the most expensive operation here
def compute_similarities(param_partition, grads_1, grads_2): 
    '''

    :param param_partition: list of (param_name, index) pairs denoting the partition of the weights
    :param grads_1: dict from param_name to gradient tensor
    :param grads_2: dict from param_name to gradient tensor
    '''

    sims = [] # same indexing as param_partition
    for param_name, indices in param_partition: 
        grad_1 = grads_1[param_name]
        grad_2 = grads_2[param_name]

        if grad_1 is None or grad_2 is None: 
            sims.append(torch.Tensor([5]).squeeze()) # so that it should not be picked as a param to keep
            continue

        # index into the grads if this param is partitioned
        if indices is not None: 
            grad_1 = grad_1[indices]
            grad_2 = grad_2[indices]

        cosine_sim = F.cosine_similarity(grad_1, grad_2, dim=-1).detach().cpu()
        sims.append(cosine_sim)

    return sims

def find_parameters_to_keep(param_partition, similarities, k=10000, return_target_indices=False):
    # k is the number of the lowest similarities

    sim_stack = torch.stack(similarities)

    # k = k or sim_stack.shape[-1]
    top_k_result = sim_stack.topk(k, largest=False, sorted=True)

    target_indices = [ind.item() for ind in top_k_result[1]]
    params_to_keep = [param_partition[ind] for ind in target_indices]

    if return_target_indices: 
        return params_to_keep, target_indices
    else: 
        return params_to_keep

def _minimize_grads_2(grads_1, grads_2): 
    return grads_2

def _maximize_grads_1(grads_1, grads_2): 
    return -grads_1

def _maximize_grads(grads_1, grads_2): 
    return -(grads_1+grads_2)

def update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=_minimize_grads_2): 
    '''
    By convention, grads_1 should be the "pos grads" and grads_2 should be the "neg grads"
    For new_grad_calc, remember that optimizer steps in the direction of minimization (i.e., opposite direction of what new_grad_calc returns)
    '''
    # IMP [Depends on PyTorch version] - Change set_to_none=False, otherwise all grads are None
    optimizer.zero_grad(set_to_none=False) # so that any grad not for param in params_to_keep is zero
    # logger.debug("Inside update_model_param_grads")
    if params_to_keep is not None:
        # logger.debug(f"Length of params_to_keep: {len(params_to_keep)}")
        for param_name, indices in params_to_keep: 
            param = model_params_map[param_name]
            if indices is None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)
            else: 
                new_grad = new_grad_calc(grads_1[param_name][indices], grads_2[param_name][indices])
                param.grad[indices] = new_grad.to(device)
    else: 
        for param_name, param in model_params_map.items(): 
            if grads_1[param_name] is not None and grads_2[param_name] is not None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)

def take_optim_step(optimizer, model_params_map, param_partition, grads_1, grads_2, 
                    device, k=10000, new_grad_calc=_minimize_grads_2): 
    if k is not None: # do param partitioning
        similarities = compute_similarities(param_partition, grads_1, grads_2)
        # logger.debug("After compute_similarities")
        params_to_keep = find_parameters_to_keep(param_partition, similarities, k=k, return_target_indices=False)
        # logger.debug("After find_parameters_to_keep")
        update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=new_grad_calc)
    else: 
        update_model_param_grads(optimizer, model_params_map, None, grads_1, grads_2, device, new_grad_calc=new_grad_calc)
    # logger.debug("After update_model_param_grads")

    optimizer.step()

def do_an_mlm_backprop(model, input_ids, attention_mask, indices, target_tokens, vocab_size, device, model_name='', 
                        do_backprop=True, multiplier=None): 

    input_ids = input_ids.to(device)
    indices = indices.to(device)
    target_tokens = target_tokens.to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if 'roberta' not in model_name: 
        logits = outputs.prediction_logits
    else: 
        logits = outputs.logits

    indices = indices.unsqueeze(-1).repeat(1, vocab_size).unsqueeze(1)
    target_tokens = target_tokens.unsqueeze(-1)
    logits = logits.gather(1, indices)
    unsqueeze_later = logits.shape[0]==1
    logits = logits.squeeze()
    if unsqueeze_later: 
        logits = logits.unsqueeze(0)
    logits = logits.gather(1, target_tokens)

    if multiplier is not None: 
        logits = logits*multiplier

    final_output = logits.sum()

    if do_backprop: 
        final_output.backward()

    return final_output, logits

def do_an_lm_backprop(accelerator, model, input_ids, attention_mask, labels, device):

    # TODO - Replce device with when using accelerator.device
    if accelerator is not None:
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)
        labels = labels.to(accelerator.device)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

    # No need to shift labels because they are shifted internally: 
    # https://github.com/huggingface/transformers/pull/17987/commits/988fab92806dc8db0b0218018ee5a582f4545193 
    model.zero_grad()
    # logger.debug(f'input_ids shape: {input_ids.shape}')
    # logger.debug(f'attention_mask shape: {attention_mask.shape}')
    # logger.debug(f'labels shape: {labels.shape}')

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # logger.debug(f'loss: {outputs.loss}')
    # logger.debug(f'loss type: {type(outputs.loss)}')
    
    # TODO - Why negative?
    loss = -outputs.loss
    # logger.debug('Before loss.backward()')

    # Large models for eg. Llama-2-7b blows up here - Fixed using FSDP. Other options: 1. Offloading weights to CPU and 2. PEFT
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()
    # logger.debug('Passed loss.backward()')

    return loss

def _get_checkpoint_dir(base_dir, epoch, dedupe=''): 
    # TODO: make this better, for now it's usable (i.e., it should work better for restarting during a middle epoch)
    return f'{base_dir}/models/{dedupe}/model_{epoch}'

def save_model(model, tokenizer, epoch, base_dir, dedupe=''): 
    output_dir = _get_checkpoint_dir(base_dir, epoch, dedupe=dedupe)
    logger.info(f'Saving model at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

def retrain(accelerator, model, tokenizer, optimizer, device, dataloader: data.DataLoader, batch_size: int, 
            is_mlm: bool = True, k: int = 10000, num_epochs: int = 5, 
            sim_batch_size: int = -1, new_grad_calc=_minimize_grads_2, proportion_dev=0.5, 
            do_dynamic_gradient_selection: bool=False, 
            agg_dim=-1, start_at_epoch: int = 0, base_output_dir='', dedupe='', model_name=''):
    '''
    
    :param sim_batch_size: the number of examples to find the params to keep based off of. For simplicity, 
    this must be either a multiple of `batch_size`, -1 to denote that it should be same as batch_size, or None
    to denote using the full dataset (which needs not be a multiple of batch_size). 
    '''


    logger.info('retraining')

    if sim_batch_size == -1: 
        sim_batch_size = batch_size

    if do_dynamic_gradient_selection: 
        new_grad_calc = _maximize_grads

    if sim_batch_size is not None and (sim_batch_size<=0 or sim_batch_size%batch_size!=0): 
        raise ValueError(f'Batch size for computing similarity is invalid: {sim_batch_size}')

    vocab_size = len(tokenizer)
    params_map = get_params_map(model)
    # logger.debug("Passed get_params_map")
    param_partition = create_param_partition(params_map, dim_to_agg=agg_dim)
    # logger.debug("Passed create_param_partition")
    logger.info(f"model type: {type(model)}")
    logger.info(f"# weight vectors with {('input' if agg_dim==-1 else 'output')} agg: {len(param_partition)}")
    # ===== For OPT-350M ===== #
    # no. of weight vectors with input agg: 275275
    # ===== For Llama2 ===== #
    # no. of weight vectors with output agg: 1146945
    # no. of weight vectors with input agg: 1423937

    for epoch in range(start_at_epoch, num_epochs): 
        logger.info(f'On epoch {epoch+1}/{num_epochs}')
        model.train()

        curr_sim_batch_count = 0
        curr_disadvantaged_grads = None
        curr_advantaged_grads = None
        for (disadvantaged_seqs, advantaged_seqs), \
                (disadvantaged_attention_masks, advantaged_attention_masks), \
                inds, \
                (disadvantaged_target_tokens, advantaged_target_tokens), \
                (disadvantaged_labels, advantaged_labels) in tqdm(dataloader):

            optimizer.zero_grad()

            if is_mlm: 
                disadv_logits = do_an_mlm_backprop(model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    inds, 
                                    disadvantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=not do_dynamic_gradient_selection)[1]
            else:
                # logger.debug("Running LM backprop for disadv seqs") 
                _ = do_an_lm_backprop(accelerator,
                                    model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    disadvantaged_labels, 
                                    device)

            if not do_dynamic_gradient_selection:
                # logger.debug("Getting model grads for disdv seqs")
                disadvantaged_grads = get_all_model_grads(model)

            if is_mlm: 
                adv_logits = do_an_mlm_backprop(model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    inds, 
                                    advantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=not do_dynamic_gradient_selection)[1]
            else:
                # logger.debug("Running LM backprop for adv seqs")
                _ = do_an_lm_backprop(accelerator, 
                                    model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    advantaged_labels, 
                                    device)

            if not do_dynamic_gradient_selection:
                # logger.debug("Getting model grads for adv seqs")
                advantaged_grads = get_all_model_grads(model)

            if do_dynamic_gradient_selection: 
                disadvantaged_actually_disadvantaged = disadv_logits<adv_logits
                multiplier = disadvantaged_actually_disadvantaged.float()*2-1 # if actually disadvantaged, 1, else, -1
                do_an_mlm_backprop(model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    inds, 
                                    disadvantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=True, 
                                    multiplier=multiplier)
                disadvantaged_grads = get_all_model_grads(model) 
                do_an_mlm_backprop(model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    inds, 
                                    advantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=True, 
                                    multiplier=-multiplier)
                advantaged_grads = get_all_model_grads(model) 

            curr_disadvantaged_grads = accumulate_grad(curr_disadvantaged_grads, disadvantaged_grads)
            curr_advantaged_grads = accumulate_grad(curr_advantaged_grads, advantaged_grads)
            # logger.debug('After accumulating gradients')

            curr_sim_batch_count += batch_size
            
            if sim_batch_size is not None and curr_sim_batch_count>=sim_batch_size:
                logger.info(f"Taking optimizer step after {curr_sim_batch_count} examples") 
                take_optim_step(optimizer, params_map, param_partition, curr_disadvantaged_grads, curr_advantaged_grads, 
                                device, k=k, new_grad_calc=new_grad_calc)
                # logger.debug('After take_optim_step')
                curr_sim_batch_count = 0
                curr_disadvantaged_grads = None
                curr_advantaged_grads = None


        if sim_batch_size is None: 
            take_optim_step(optimizer, params_map, param_partition, curr_disadvantaged_grads, curr_advantaged_grads, 
                            device, k=k, new_grad_calc=new_grad_calc)

        if epoch in [4, 9]:
            saved_model_dir = save_model(model, tokenizer, epoch+1, base_output_dir, dedupe=dedupe)


def main(model_path_or_name: str, num_epochs: int = 5, is_mlm: bool = True, k: int = 10000, 
        sim_batch_size: int = -1, use_advantaged_for_grad: bool=True, agg_input: bool=True, 
        proportion_dev=0.75, do_dynamic_gradient_selection: bool=False, 
        lr: float = 1e-5, momentum: float = 0.9, batch_size: int = 16, seed: int = 89793, 
        num_workers: int = 4, start_at_epoch: int = 0, base_output_dir='', dedupe='', sample_data=False): 
    logger.info(f'Seed is {seed}')
    set_random_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Use model alias to determine whether to use accelerator for distributed fine-tuning
    model_alias = model_path_or_name.lower().split("/")[-1]
    logger.info(f"model alias: {model_alias}")
    assert model_alias != "", f"Empty model alias for {model_path_or_name}"
    use_accelerator = (model_alias in ["opt-2.7b", "opt-6.7b", "llama-2-7b-hf", "llama-2-13b-hf", "meta-llama-3-8b"])

    # TODO - Add a better condition for large models
    # if 'llama' in model_path_or_name.lower():
    if use_accelerator:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        logger.info(f"Accelerator: {accelerator.distributed_type}")
    else:
        # Set it None for single GPU models
        accelerator = None

    # When accelerator is None, used only in update_model_param_grads
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_mlm: 
        if 'roberta' in model_path_or_name: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, add_prefix_space=True)
        else: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = AutoModelForPreTraining.from_pretrained(model_path_or_name)
        model.resize_token_embeddings(len(tokenizer))
        model.train()
        model.to(device)
    else:
        if 'llama' in model_path_or_name.lower():
            if 'llama-3' in model_path_or_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
            else:
                tokenizer = LlamaTokenizer.from_pretrained(model_path_or_name) # pad_token=PAD_TOKEN, mask_token=MASK_TOKEN
            tokenizer.pad_token_id = tokenizer.eos_token_id

            model = LlamaForCausalLM.from_pretrained(
                model_path_or_name,
                # load_in_8bit=True
                device_map="balanced", # "auto"
                # use_cache=True,
            )
        elif 'opt' in model_path_or_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
            device_map = "balanced" if use_accelerator else "auto"
            model = OPTForCausalLM.from_pretrained(
                model_path_or_name, 
                device_map=device_map
                )
            model.resize_token_embeddings(len(tokenizer))
            if not use_accelerator:
                model.train()
                model.to(device)

    # TODO - Load BBQ dataset
    dataset = BBQDataset("../data/bbq_data", tokenizer, sample=sample_data)
    if sample_data:
        logger.warning('**** THIS IS A TEST RUN ****')
    logger.info(f'No. of BBQ sentence pairs: {len(dataset)}')

    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=BBQDataset.collate_batch_creator(tokenizer), 
                                num_workers=num_workers, 
                                )
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # TODO - Add a better condition for large models
    # if 'llama' in model_path_or_name.lower():
    if use_accelerator:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    agg_dim = -1 if agg_input else -2
    new_grad_calc = _minimize_grads_2 if use_advantaged_for_grad else _maximize_grads_1

    logger.info('Retraining now')
    logger.info(f'batch_size: {batch_size}')

    retrain(accelerator, model, tokenizer, optimizer, device, dataloader, batch_size, is_mlm=is_mlm, k=k, 
            num_epochs=num_epochs, sim_batch_size=sim_batch_size, new_grad_calc=new_grad_calc, 
            proportion_dev=proportion_dev, do_dynamic_gradient_selection=do_dynamic_gradient_selection, 
            agg_dim=agg_dim, start_at_epoch=start_at_epoch, base_output_dir=base_output_dir, dedupe=dedupe, model_name=model_path_or_name)

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'This script uses prompts to sample sentences from an LM')
    parser.add_argument('-m', type=str, required=True, dest='model_path_or_name', help='path to the model or name of the model')
    parser.add_argument('-l', type=float, default=1e-5, dest='lr', help='learning rate')
    parser.add_argument('-k', type=int, default=10000, dest='k', help='the k in top k')
    parser.add_argument('--use-full-grad', dest='k', action='store_const', const=None, help='to use the full gradient (rather than k parts)') # if this and also -k are used then whichever is rightmost argument wins
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-n', type=int, default=5, dest='num_epochs', help='number of epochs to train for (total)')
    parser.add_argument('-b', type=int, default=16, dest='batch_size', help='batch size')
    parser.add_argument('--start-at', type=int, default=0, help='start at checkpoint epoch number (e.g., 1, and if training 5 epochs then 4 more epochs will be done)')
    parser.add_argument('--dedupe', type=str, default='', help='dedupe string (basically just the name of the experiment), models will be saved to `sim_checkpoints/{dedupe}/model_{epoch}`')
    parser.add_argument('--output-agg', dest='aggregation', action='store_const', const='output', default='input', help='to use output aggregation (default: input aggregation)')
    # TODO - Use static (default), since dynamic is not useful based on the paper. Can experiment with it, if time permits.
    parser.add_argument('--dynamic_gradient_selection', dest='dynamic_gradient_selection', action='store_true', default=False, help='to choose disadvantaged and advantaged dynamically (default: static based on WG)')
    parser.add_argument('--use-disadvantaged', dest='use_advantaged_for_grad', action='store_false', default=True, help='to take gradient step to maximize disadvantaged (default: minimize advantaged)')
    # TODO - Investigate this option too
    # Explicitly alter sim_batch_size
    parser.add_argument('--sim-batch-size', type=int, default=-1, help='Update batch size (has to be multiple of batch_size)')
    # parser.add_argument('--use-same-params', dest='sim_batch_size', action='store_const', const=None, default=-1, help='to use the same params each epoch (default: picks params each batch)')
    # TEMP - Added new arg to sample (naive sampling) portion of data for test run
    parser.add_argument('--sample-data', action='store_true', default=False, help='If set to true, will sample 0.05 fraction data for test run')
    parser.add_argument('--model-output-dir', default='/scratch/ssd002/projects/opt_test/llm_bias/pcgu_models', help='Base directory to save all model checkpoints')

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    is_mlm = 'bert' in args.model_path_or_name
    sim_batch_size = args.sim_batch_size
    use_advantaged_for_grad = args.use_advantaged_for_grad
    agg_input = args.aggregation=='input'
    if args.dynamic_gradient_selection: 
        direction_selection = 'dynamic'
    elif args.use_advantaged_for_grad: 
        direction_selection = 'adv'
    else: 
        direction_selection = 'disadv'

    if args.k is None: 
        partition_usage = 'full_grad'
    elif args.sim_batch_size is None: 
        partition_usage = 'all'
    else: 
        partition_usage = 'notall'

    dedupe_model_name = args.model_path_or_name.split('/')[-1]
    actual_batch_size = args.batch_size if args.sim_batch_size == -1 else args.sim_batch_size
    dedupe = f"{dedupe_model_name}/{partition_usage}/{'inp' if agg_input else 'outp'}/{direction_selection}/{args.lr}/{actual_batch_size}/{args.k}"
    main(args.model_path_or_name, num_epochs=args.num_epochs, is_mlm=is_mlm, k=args.k, 
        proportion_dev=0.5, do_dynamic_gradient_selection=args.dynamic_gradient_selection, 
        sim_batch_size=sim_batch_size, use_advantaged_for_grad=use_advantaged_for_grad, agg_input=agg_input, 
        lr=args.lr, momentum=args.momentum, batch_size=args.batch_size, start_at_epoch=args.start_at, base_output_dir=args.model_output_dir, 
        dedupe=dedupe, sample_data=args.sample_data)
