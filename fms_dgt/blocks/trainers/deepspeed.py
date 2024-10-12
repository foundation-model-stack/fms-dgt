# Standard
from typing import List
import gc
import json
import os

# Third Party
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)
import deepspeed
import torch

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import TrainerData, make_model_dir

###
# Trainer itself
###


@register_block("deepspeed")
class DeepspeedTrainerBlock(BaseTrainerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self._config_path, "r") as f:
            config = json.load(f)
            self._save_steps = config["save_steps"]

    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
    ) -> str:
        def collate_fn(batch: List[TrainerData]):
            # input will be batch of DICTIONARIES matching the TrainerData dataclass
            tokenized = [tokenizer(b["input"] + b["output"]).input_ids for b in batch]
            batch_data = {k: v.to("cuda") for k, v in data_collator(tokenized).items()}
            return batch_data

        data_dir = os.path.join(output_dir, "dataset")
        dataset = self.get_dataset(datastore, data_dir)

        model_dir = make_model_dir(output_dir)

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

        model_engine, _, _, _ = deepspeed.initialize(
            config=self._config_path, model=model, model_parameters=model.parameters()
        )
        model_engine: deepspeed.DeepSpeedEngine

        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=False, return_tensors="pt"
        )

        dataloader = DataLoader(
            dataset,
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
                model_engine.save_checkpoint(model_dir, ckpt_id)

        final_model = os.path.join(model_dir, "last")
        model_engine.save_checkpoint(final_model)

        # free model memory
        del model_engine
        gc.collect()
        torch.cuda.empty_cache()

        # return last model
        return final_model
