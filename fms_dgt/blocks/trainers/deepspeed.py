# Standard
from dataclasses import asdict
from typing import Dict, List
import gc
import json
import os
import shutil

# Third Party
from torch.utils.data import DataLoader
import deepspeed
import torch
import transformers

# Local
from fms_dgt.base.task import InputOutputData
from fms_dgt.blocks.trainers.trainer import BaseTrainer

###
# Trainer itself
###


class DeepspeedTrainer(BaseTrainer):
    def __init__(
        self,
        model_id_or_path: str,
        config_path: str,
        output_dir: str,
        data: List[InputOutputData],
        config_path: str,
        restart: bool = False,
    ):
        self._model_id_or_path = model_id_or_path
        self._config_path = config_path
        self._data = [d.input + d.output for d in data]

        self._output_dir = output_dir
        self._restart = restart
        if restart:
            shutil.rmtree(self._output_dir)

        self._is_distributed = False

        with open(self._config_path, "r") as f:
            config = json.load(f)
            self._save_steps = config["save_steps"]

    @property
    def trained_model_path(self):
        return os.path.join(self._output_dir, "last")

    def train(self):
        def collate_fn(batch):
            tokenized = [tokenizer(b).input_ids for b in batch]
            batch_data = {k: v.to("cuda") for k, v in data_collator(tokenized).items()}
            return batch_data

        if self._is_distributed:
            deepspeed.init_distributed()

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_id_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_id_or_path)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD_TOK]"})
            model.resize_token_embeddings(len(tokenizer))

        model_engine, _, _, _ = deepspeed.initialize(
            config=self._config_path, model=model, model_parameters=model.parameters()
        )
        model_engine: deepspeed.DeepSpeedEngine

        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, return_tensors="pt"
        )

        dataloader = DataLoader(
            self._data,
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
            if step % self._save_steps == 0:
                ckpt_id = f"chkpt_{self._save_steps}"
                model_engine.save_checkpoint(self._output_dir, ckpt_id)

        model_engine.save_checkpoint(*os.path.split(self.trained_model_path))

        # free model memory
        del model_engine
        gc.collect()
        torch.cuda.empty_cache()
