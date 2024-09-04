# Standard
from dataclasses import dataclass
from typing import Dict, Union
import gc
import shutil

# Third Party
from torch.utils.data import DataLoader
import deepspeed
import torch
import transformers

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.trainer import BaseTrainer
from fms_dgt.dataloaders.default import DefaultDataloader
from fms_dgt.utils import init_dataclass_from_dict

###
# Data classes with arguments go first
###


@dataclass
class SchedulerArgs:
    type: str = "WarmupLR"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.parms = {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            }


@dataclass
class OptimizationArgs:

    type: str = "Adam"
    params: dict = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            }


@dataclass
class FP16Args:
    enabled: bool = False


@dataclass
class BFloat16Args:
    enabled: bool = True


@dataclass
class OptimizerArgs:
    type: str = "Adam"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {"lr": 0.00015}


@dataclass
class TrainerArgs:
    train_micro_batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    optim_args: OptimizerArgs = None
    fp16_args: FP16Args = None
    bfloat16_args: BFloat16Args = None
    scheduler_args: SchedulerArgs = None
    zero_optimization: bool = True

    def __post_init__(self):
        self.optim_args = init_dataclass_from_dict(self.optim_args, OptimizationArgs)
        self.fp16_args = init_dataclass_from_dict(self.fp16_args, FP16Args)
        self.bfloat16_args = init_dataclass_from_dict(self.bfloat16_args, BFloat16Args)
        self.scheduler_args = init_dataclass_from_dict(
            self.scheduler_args, SchedulerArgs
        )


###
# Trainer itself
###


class DeepspeedTrainer(BaseTrainer):
    def __init__(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
        trainer_args: Union[TrainerArgs, Dict] = None,
        restart: bool = False,
    ):
        self._model_id_or_path = model_id_or_path
        print(self._model_id_or_path)
        input()
        self._trainer_args = init_dataclass_from_dict(trainer_args, TrainerArgs)
        self._datastore = datastore
        self._dataloader = DataLoader(
            self._datastore.load_data(),
            batch_size=trainer_args.train_micro_batch_size_per_gpu,
            shuffle=True,
        )

        self._output_dir = output_dir
        self._restart = restart
        if restart:
            shutil.rmtree(self._output_dir)

        self._is_distributed = False

    def _init_training(self):

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_id_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=self._trainer_args,
            model=model,
        )

        return model_engine, optimizer

    def train(self):

        if self._is_distributed:
            deepspeed.init_distributed()

        model_engine, optimizer = self._init_training()

        for step, batch in enumerate(self._dataloader):
            # forward() method
            loss = model_engine(batch)

            # runs backpropagation
            model_engine.backward(loss)

            # weight update
            model_engine.step()

            # save checkpoint
            if step % args.save_interval:
                client_sd["step"] = step
                ckpt_id = loss.item()
                model_engine.save_checkpoint(
                    args.save_dir, ckpt_id, client_sd=client_sd
                )
                self.save_dataloader_state()

    def init_model(self):
        self.model = LLM(**self.model_args)

    def release_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
