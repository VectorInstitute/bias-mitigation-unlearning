import transformers
from datasets import load_dataset
import sys
sys.path.append('/h/ws_tyau/bias-lm-stream/bias_mitigation')
from get_dataset import CustomDataset
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import argparse
from peft import get_peft_model, LoraConfig, IA3Config, TaskType

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--pem', type=str, default=None)

args = parser.parse_args()

data = CustomDataset("stereo_dpo")
data = data.get_dataset()
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map='auto',
)

if args.pem:
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

data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

output_dir = args.output_path + args.model_name + "-" + str(args.pem)

if 'Llama' in args.model_name:
    trainer = transformers.Trainer(
        model=model, 
        train_dataset=data,
        eval_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2, 
            gradient_accumulation_steps=8,
            warmup_steps=100, 
            num_train_epochs=10, 
            learning_rate= 5e-4,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            metric_for_best_model="loss",
            output_dir=output_dir,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
elif 'opt' in args.model_name:
    trainer = transformers.Trainer(
        model=model, 
        train_dataset=data,
        eval_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4,
            warmup_steps=100, 
            num_train_epochs=10, 
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            metric_for_best_model="loss",
            output_dir=output_dir,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
else:
    raise ValueError("Invalid model name.")
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

