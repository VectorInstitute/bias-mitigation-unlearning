
import argparse
import logging
import os
import torch
import argparse
import os
from peft import PeftConfig, PeftModel


# import numpy as np
# import torch
# import random
# import pandas as pd
# import pdb
from transformers import AutoModelForCausalLM

def adapter_negation(
    model,
    adapter_config,
    scale
    ):


    state_dict = model.state_dict()
    merged_keys = [mk for mk in state_dict.keys() if  ("lora_A" in mk)]
    print("lora_A:",len(merged_keys))
    merged_keys = [mk for mk in state_dict.keys() if  ("lora" in mk)]
    print("lora:",len(merged_keys))
    # breakpoint()
    if adapter_config=="lora_adapter":
        # neg_dict = {k:-v for k,v in state_dict.items() if "lora_A" in k}
        neg_dict = {k:-1*scale*v for k,v in state_dict.items() if "lora_A" in k}

    # ia3 (h+l*delta_h)-(h+delta_h)=(l-1)*delta_h h+delta_h-(l-1)*delta_h=h+(2-l)*delta_h
    elif adapter_config=="ia3_adapter":
        # neg_dict = {k:(torch.ones(v.shape)*2-v) for k,v in state_dict.items() if "lora" in k}
        neg_dict = {k:(torch.ones(v.shape)*(1+scale)-scale*v) for k,v in state_dict.items() if "lora" in k}
    else:
        raise ValueError(f"adapter_config {adapter_config} not supported") 

    state_dict.update(neg_dict)
    model.load_state_dict(state_dict)
    # model.set_active_adapters(["civil_comments"])
    # model.save_all_adapters(save_path)

    return model



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--finetuned_model_path', type=str, required=True)
    parser.add_argument('--scaling_coef', type=float, required=True)
    parser.add_argument('--model_save_path', type = str, required=True)
    args = parser.parse_args()

    pretrained_model= AutoModelForCausalLM.from_pretrained(args.pretrained_model_path)
    finetuned_model= PeftModel.from_pretrained(pretrained_model, args.finetuned_model_path)
    adapter_config=args.finetuned_model_path.split("-")[-1]#[:-1]
    print(args.finetuned_model_path.split("-")[-1][:-1])
    unbiased_model = adapter_negation(finetuned_model,adapter_config,args.scaling_coef)
    print("unbiased_model created")
    unbiased_model = unbiased_model.merge_and_unload()
    print("unbiased_model merged")
    unbiased_model.save_pretrained(args.model_save_path)
    print("unbiased_model saved")
        

