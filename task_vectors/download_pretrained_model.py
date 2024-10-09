import transformers
from datasets import load_dataset
import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, IA3Config, TaskType

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--pem', type=str, default=None)

args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map='auto',
)

if 'Llama' in args.model_name and args.pem is not None:
    if args.pem == 'lora_adapter':
        llama_peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_dropout=0.01,
        )
    elif args.pem == 'ia3_adapter':
        llama_peft_config = IA3Config(
            peft_type="IA3",
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError("Invalid PEM type for Llama model.")
    model = get_peft_model(model, llama_peft_config)
    model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if 'Llama' in args.model_name:
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.save_pretrained("./outputs/pretrained/" + args.model_name)
tokenizer.save_pretrained("./outputs/pretrained/" + args.model_name)
