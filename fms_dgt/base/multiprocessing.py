# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

# Third Party
from ray.actor import ActorHandle
import ray

# Local
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.utils import init_dataclass_from_dict


@dataclass
class ParallelConfig:

    num_workers: int = 1
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: int = 0
    worker_configs: Optional[List[Dict]] = None


class ParallelBlock:
    """This class contains the functionality for turning a standard block into a parallelized block"""

    def __init__(
        self,
        block_class: Type,
        parallel_config: Dict,
        *args,
        **kwargs: Dict,
    ):
        parallel_config: ParallelConfig = init_dataclass_from_dict(
            parallel_config, ParallelConfig
        )

        worker_cfgs: Dict = {
            worker_idx: dict() for worker_idx in range(parallel_config.num_workers)
        }
        if parallel_config.worker_configs is not None and not isinstance(
            parallel_config.worker_configs, list
        ):
            raise ValueError(
                f"If [worker_configs] field is specified, it must be given as list"
            )
        for cfg in parallel_config.worker_configs:
            if not cfg.get("workers"):
                raise ValueError(f"Must identify list of worker ids in [workers] field")
            for worker_idx in cfg.pop("workers"):
                if type(worker_idx) != int:
                    raise ValueError(
                        f"Worker ids must be integers, not [{worker_idx}] which is of type {type(worker_idx)}"
                    )
                worker_cfgs[worker_idx] = cfg

        self._workers: List[ActorHandle] = []
        for worker_idx in range(parallel_config.num_workers):
            actor = ray.remote(
                num_cpus=parallel_config.num_cpus_per_worker,
                num_gpus=parallel_config.num_gpus_per_worker,
            )(block_class).remote(
                *args, **{**kwargs, **worker_cfgs.get(worker_idx, dict())}
            )
            self._workers.append(actor)

    @property
    def workers(self) -> List[ActorHandle]:
        """Returns the workers of the block

        Returns:
            List[ActorHandle]: List of workers for the block
        """
        return self._workers

    def close(self):
        while self._workers:
            worker = self._workers.pop()
            # must wait on each close call
            ray.get(worker.close.remote())
            ray.kill(worker)

    def generate(self, *args, **kwargs):  # for interfacing with IL
        return self(*args, **kwargs)

    def __call__(self, inputs: DATASET_TYPE, *args: Any, **kwargs: Any) -> DATASET_TYPE:
        """Distributes input list amongst workers according to parallel config

        Args:
            inputs (DATASET_TYPE): Input data to process

        Returns:
            DATASET_TYPE: Data after processing
        """
        # just return if empty input
        if len(inputs) == 0:
            return []

        # TODO: relax assumption that input is a list
        partition_size = max(len(inputs) // len(self._workers), 1)
        actor_results = []
        for worker_idx in range(len(self._workers)):
            if worker_idx * partition_size >= len(inputs):
                break
            actor_results.append(
                self._workers[worker_idx].__call__.remote(
                    (
                        inputs[worker_idx * partition_size :]
                        if (worker_idx == len(self._workers) - 1)
                        else inputs[
                            worker_idx
                            * partition_size : (worker_idx + 1)
                            * partition_size
                        ]
                    ),
                    *args,
                    **kwargs,
                )
            )
        generated_data = [d for gen_data in ray.get(actor_results) for d in gen_data]
        return generated_data
