# Standard
from typing import Any, List, TypeVar, Union
import json
import os

# Third Party
import pandas as pd

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_datastore

T = TypeVar("T")


@register_datastore("default")
class DefaultDatastore(BaseDatastore):
    """Base Class for all data stores"""

    def __init__(
        self,
        output_dir: str = None,
        task_name: str = None,
        output_format: str = ".jsonl",
        restart_generation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._output_dir = output_dir
        self._output_path = self._get_default_output_path(task_name, output_format)

        if restart_generation and os.path.exists(self.output_path):
            os.remove(self.output_path)

    @property
    def output_path(self) -> str:
        return self._output_path

    def _get_default_output_path(self, task_name: str, output_format: str):
        path_components = []
        path_components.append(self._output_dir)
        path_components.append(task_name)
        path_components.append("generated_instructions." + output_format)
        return os.path.join(*path_components)

    def save_data(
        self,
        new_data: List[T],
    ) -> None:

        output_format = os.path.splitext(self.output_path)[-1]

        if output_format == ".jsonl":
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "a") as f:
                for d in new_data:
                    f.write(json.dumps(d) + "\n")
        elif output_format == ".parquet":
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            pd.DataFrame(new_data).to_parquet(
                self.output_path,
                engine="fastparquet",
                append=os.path.isfile(self.output_path),
            )
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

    def load_data(self) -> List[T]:

        if not os.path.exists(self.output_path):
            return

        output_format = os.path.splitext(self.output_path)[-1]
        if output_format == ".jsonl":
            with open(self.output_path, "r") as f:
                try:
                    machine_data = [json.loads(l.strip()) for l in f.readlines()]
                except ValueError:
                    machine_data = []
        elif output_format == ".parquet":
            machine_data = (
                pd.read_parquet(self.output_path, engine="fastparquet")
                .apply(dict, axis=1)
                .to_list()
            )
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

        return machine_data

    def save_task(self) -> None:
        pass
