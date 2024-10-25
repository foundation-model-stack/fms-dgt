# Standard
from typing import Any, Dict, List, Type

# Third Party
from ray.actor import ActorHandle
import ray

# Local
from fms_dgt.constants import DATASET_TYPE


class ParallelBlock:
    """This class contains the functionality for turning a standard block into a parallelized block"""

    def __init__(
        self,
        block_class: Type,
        parallel_config: Dict,
        *args,
        **kwargs: Dict,
    ):

        worker_ct = parallel_config.get("worker_ct", 1)
        num_cpus = parallel_config.get("num_cpus", 1)
        num_gpus = parallel_config.get("num_gpus", 0)

        self._workers: List[ActorHandle] = []
        for _ in range(worker_ct):
            actor = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
                block_class
            ).remote(*args, **kwargs)
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
        # TODO: relax assumption that input is a list
        partition_size = len(inputs) // len(self._workers)
        actor_results = [
            self._workers[i].__call__.remote(
                inputs[i * partition_size : (i + 1) * partition_size], *args, **kwargs
            )
            for i in range(len(self._workers))
        ]
        generated_data = [d for gen_data in ray.get(actor_results) for d in gen_data]
        return generated_data
