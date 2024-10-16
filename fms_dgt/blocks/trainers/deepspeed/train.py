# Standard
from argparse import ArgumentParser, Namespace
from typing import Dict
import asyncio
import json
import os
import sys

# Third Party
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
import deepspeed
import psutil
import torch
import uvloop

###
# Trainer itself
###


async def train(
    data_path: str,
    config_path: str,
    model_id_or_path: str,
    output_dir: str,
    local_rank: int,
    training_args: dict,
) -> str:
    def tokenize_fn(example: Dict):
        # this function assumes input will be a dictionary that matches TrainerData schema
        return tokenizer(example["input"], example["output"])

    print(training_args)
    input()
    dataset = load_from_disk(data_path).with_format("torch")

    is_distributed = False
    if is_distributed:
        deepspeed.init_distributed()

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD_TOK]"})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,  # return_tensors="pt"
    )

    training_args = _get_training_args(config_path, output_dir)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # Start the training
    trainer.train()


def _get_training_args(config_path: str, output_dir: str):

    with open(config_path, "r") as f:
        config = json.load(f)

    lr = config.get("optimizer", dict()).get("params", dict()).get("lr")
    fp16 = config.get("fp16", dict()).get("enabled", False)
    per_device_train_batch_size = config.get("train_micro_batch_size_per_gpu")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps")
    save_steps = config.get("save_steps")
    steps_per_print = config.get("steps_per_print")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        deepspeed=config_path,
        learning_rate=lr,
        fp16=fp16,
        logging_steps=steps_per_print,
        save_steps=save_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        # gradient_checkpointing=True,
    )
    return training_args


async def main(args: Namespace):
    """Runs the vllm server while checking that the original process is still alive"""
    pid, check_interval = args.pid, args.check_interval

    delattr(args, "pid")
    delattr(args, "check_interval")

    monitor_task = monitor(pid, check_interval)
    server_task = train(
        args.data_path,
        args.config_path,
        args.model_id_or_path,
        args.output_dir,
        args.local_rank,
        json.loads(args.training_args),
    )

    finished, unfinished = await asyncio.wait(
        [monitor_task, server_task], return_when=asyncio.FIRST_COMPLETED
    )
    for x in finished:
        result = x.result()
        if result:
            # cancel the other tasks, we have a result. We need to wait for the cancellations
            for task in unfinished:
                task.cancel()
            await asyncio.wait(unfinished)
            return result


async def monitor(parent_pid: int, check_interval: float):
    while True:
        await asyncio.sleep(check_interval)
        if not psutil.pid_exists(parent_pid):
            sys.exit()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--model-id-or-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--local_rank", required=True, type=int)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--check-interval", required=True, type=float)
    parser.add_argument("--training-args", required=True, type=str)
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    uvloop.run(main(args))
