import json
from fastchat.conversation import Conversation, get_conv_template, conv_templates
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import SFTTrainer
from accelerate import PartialState

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device_string = PartialState().process_index

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map={'':device_string})

full = load_dataset("glue", "mrpc", split="train")

def formatting_prompts_func(example):
    output_texts = []
    for i in example['sentence1']:
        conv = get_conv_template('llama-2')
        conv.system_message = i
        conv.append_message(conv.roles[0], i)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        output_texts.append(prompt)
    return output_texts

trainer = SFTTrainer(
    model=model,
    train_dataset=full,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=20,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=20,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        output_dir="/groups/chiyuan/hf_ckpt",
        ddp_find_unused_parameters=False,
    ),
    max_seq_length=1024,
    formatting_func=formatting_prompts_func,
)

trainer.train()
