# Standard
from argparse import ArgumentParser
from typing import Dict
import json

# Third Party
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
import deepspeed
import torch

###
# Trainer itself
###


def main(
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        deepspeed=config_path,
        **training_args,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # Start the training
    trainer.train()


class TrainingException(Exception):
    pass


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--model-id-or-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--local_rank", required=True, type=int)
    parser.add_argument("--training-args", required=True, type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        config_path=args.config_path,
        model_id_or_path=args.model_id_or_path,
        output_dir=args.output_dir,
        local_rank=args.local_rank,
        training_args=json.loads(args.training_args),
    )
