# Standard
from typing import Dict
import json
import os

# Third Party
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import deepspeed
import torch

###
# Trainer itself
###


def _train(
    data_path: str,
    config_path: str,
    model_id_or_path: str,
    model_dir: str,
) -> str:
    def collate_fn(batch):
        tokenized = [tokenizer(b).input_ids for b in batch]
        batch_data = {k: v.to("cuda") for k, v in data_collator(tokenized).items()}
        return batch_data

    def tokenize_fn(example: Dict):
        # this function assumes input will be a dictionary that matches TrainerData schema
        return tokenizer(example["input"], example["output"])

    dataset = load_from_disk(data_path).with_format("torch")

    is_distributed = False
    if is_distributed:
        deepspeed.init_distributed()

    with open(config_path, "r") as f:
        config = json.load(f)

    dschf = HfDeepSpeedConfig(config)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_engine, _, _, _ = deepspeed.initialize(
        config_params=config,
        model=model,
    )
    model_engine: deepspeed.DeepSpeedEngine

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD_TOK]"})
        model.resize_token_embeddings(len(tokenizer))

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=model_engine.train_batch_size(),
        shuffle=True,
        collate_fn=collate_fn,
    )

    model_engine.train()

    for step, batch in enumerate(dataloader):

        # forward() method
        loss = model_engine(**batch).loss

        # runs backpropagation
        model_engine.backward(loss)

        # weight update
        model_engine.step()

        # save checkpoint
        if step % dschf.get_value("save_steps") == 0:
            ckpt_id = f"chkpt_{step}"
            model_engine.save_checkpoint(model_dir, ckpt_id)

    final_model = os.path.join(model_dir, "last")
    model_engine.save_checkpoint(model_dir, "last")
