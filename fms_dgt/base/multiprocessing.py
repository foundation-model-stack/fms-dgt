# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Type

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

        self._workers: List[ActorHandle] = []
        for _ in range(parallel_config.num_workers):
            actor = ray.remote(
                num_cpus=parallel_config.num_cpus_per_worker,
                num_gpus=parallel_config.num_gpus_per_worker,
            )(block_class).remote(*args, **kwargs)
            self._workers.append(actor)

    @property
    def workers(self) -> List[ActorHandle]:
        """Returns the workers of the block

        Returns:
            List[ActorHandle]: List of workers for the block
        """
        return self._workers

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
        for i in range(len(self._workers)):
            if i * partition_size >= len(inputs):
                continue
            actor_results.append(
                self._workers[i].__call__.remote(
                    (
                        inputs[i * partition_size :]
                        if (i == len(self._workers) - 1)
                        else inputs[i * partition_size : (i + 1) * partition_size]
                    ),
                    *args,
                    **kwargs,
                )
            )
        generated_data = [d for gen_data in ray.get(actor_results) for d in gen_data]
        return generated_data
