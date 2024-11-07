# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

# Third Party
from ray.actor import ActorHandle
import ray

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.utils import init_dataclass_from_dict


@dataclass
class RayConfig:
    """Class containing the information needed to initialize ray actors for a particular block"""

    num_workers: int = 1
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: int = 0
    worker_configs: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.worker_configs is None:
            self.worker_configs = []


class RayBlock(BaseBlock):
    """This class contains the functionality for turning a standard block into a ray block"""

    def __init__(
        self,
        block_class: Type,
        ray_config: Dict,
        *args: Any,
        **kwargs: Dict,
    ):
        """RayBlock is a wrapper class that is initialized when the user provides a "ray_config" field to their block config. It operates by splitting
            the input list across ray actors, getting the results from each actor, then joining those results and returning them to the user

        Args:
            block_class (Type): Block type that should be sent to ray actors
            ray_config (Dict): Config used to init ray actors

        """
        # allow BaseBlock functions to be used
        super().__init__(*args, **kwargs)

        ray_config: RayConfig = init_dataclass_from_dict(ray_config, RayConfig)

        worker_cfgs: Dict = {
            worker_idx: dict() for worker_idx in range(ray_config.num_workers)
        }

        if not isinstance(ray_config.worker_configs, list):
            raise ValueError(
                f"If [worker_configs] field is specified, it must be given as list"
            )

        for cfg in ray_config.worker_configs:
            if not cfg.get("workers"):
                raise ValueError(f"Must identify list of worker ids in [workers] field")
            for worker_idx in cfg.pop("workers"):
                if type(worker_idx) != int:
                    raise ValueError(
                        f"Worker ids must be integers, not [{worker_idx}] which is of type {type(worker_idx)}"
                    )
                worker_cfgs[worker_idx] = cfg

        self._workers: List[ActorHandle] = []
        for worker_idx in range(ray_config.num_workers):
            actor = ray.remote(
                num_cpus=ray_config.num_cpus_per_worker,
                num_gpus=ray_config.num_gpus_per_worker,
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
        """Closes each actor then kills the actor at the ray-level"""
        # first workers
        for worker in self._workers:
            # must wait on each close call
            ray.get(worker.close.remote())
            ray.kill(worker)

        # now base block
        super().close()

    def execute(self, inputs: DATASET_TYPE, *args: Any, **kwargs: Any) -> DATASET_TYPE:
        """Distributes input list amongst workers according to ray config

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
                self._workers[worker_idx].execute.remote(
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
