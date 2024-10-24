# Standard
from typing import Any, Dict, List, Type

# Third Party
from ray.actor import ActorHandle
import ray

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.constants import DATASET_TYPE


class BaseParallelizableBlock(BaseBlock):
    """Tag class that allows one to identify when a block can be parallelized at the input level. Importantly,
    the assumption for all such blocks is that they will be executed with the `generate` method on an input of DATASET_TYPE
    """


class ParallelBlockWrapper:
    def __init__(
        self,
        parallel_config: Dict,
        block_class: Type[BaseBlock],
        block_kwargs: Dict,
    ):
        worker_ct = parallel_config.get("worker_ct", 1)
        num_cpus = parallel_config.get("num_cpus", 1)
        num_gpus = parallel_config.get("num_gpus", 0)

        self._workers: List[ActorHandle] = []
        for _ in range(worker_ct):
            actor = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
                block_class
            ).remote(**block_kwargs)
            self._workers.append(actor)

    def generate(
        self,
        inputs: DATASET_TYPE,
        *args: Any,
        **kwargs: Any,
    ) -> DATASET_TYPE:

        # TODO: relax assumption that input is a list
        partition_size = len(inputs) // len(self._workers)
        actor_results = [
            self._workers[i].generate.remote(
                inputs[i * partition_size : (i + 1) * partition_size], *args, **kwargs
            )
            for i in range(len(self._workers))
        ]
        generated_data = [d for gen_data in ray.get(actor_results) for d in gen_data]
        return generated_data
