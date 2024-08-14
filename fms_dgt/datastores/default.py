# Standard
from typing import Any, List, TypeVar
import json
import os

# Third Party
import pandas as pd
import yaml

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
        seed_examples: List[T] = None,
        data_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._output_dir = self._get_default_output_dir(output_dir, task_name)
        self._output_path = os.path.join(
            self._output_dir, "generated_instructions." + output_format
        )
        self._instruction_output_path = os.path.join(
            self._output_dir, "formatted_instructions." + output_format
        )
        self._state_path = os.path.join(self._output_dir, "dataloader_state.txt")
        self._data_path = data_path
        self._seed_examples = seed_examples

        if restart_generation and os.path.exists(self.output_path):
            os.remove(self.output_path)

        # always delete instruction output path because it's regenerated from machine_data
        if os.path.exists(self._instruction_output_path):
            os.remove(self._instruction_output_path)

    @property
    def output_path(self) -> str:
        return self._output_path

    def _get_default_output_dir(self, output_dir: str, task_name: str):
        path_components = []
        path_components.append(output_dir)
        path_components.append(task_name)
        return os.path.join(*path_components)

    def save_data(
        self,
        new_data: List[T],
        output_path: str = None,
    ) -> None:

        output_path = output_path if output_path is not None else self.output_path
        output_format = os.path.splitext(output_path)[-1]

        if output_format == ".jsonl":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "a") as f:
                for d in new_data:
                    f.write(json.dumps(d) + "\n")
        elif output_format == ".parquet":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(new_data).to_parquet(
                output_path,
                engine="fastparquet",
                append=os.path.isfile(output_path),
            )
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

    def load_data(self, output_path: str = None) -> List[T]:

        output_path = output_path if output_path is not None else self.output_path

        if not os.path.exists(output_path):
            return

        output_format = os.path.splitext(output_path)[-1]
        if output_format == ".jsonl":
            with open(output_path, "r") as f:
                try:
                    machine_data = [json.loads(l.strip()) for l in f.readlines()]
                except ValueError:
                    machine_data = []
        elif output_format == ".parquet":
            machine_data = (
                pd.read_parquet(output_path, engine="fastparquet")
                .apply(dict, axis=1)
                .to_list()
            )
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

        return machine_data

    def load_dataset(self) -> List[T]:
        seed_examples = self._seed_examples

        if self._seed_examples is None:
            seed_examples = []

        if self._data_path is not None:
            if self._data_path.endswith(".json"):
                with open(self._data_path, "r") as f:
                    data = json.load(f)
            elif self._data_path.endswith(".yaml"):
                with open(self._data_path, "r") as f:
                    data = list(yaml.safe_load(f))

            assert (
                type(data) == list
            ), "Data used for default 'load_dataset' method must be a list!"

            seed_examples = seed_examples + data

        return seed_examples

    def save_task(self) -> None:
        pass

    def load_task(self) -> Any:
        pass

    def save_state(self, state: Any) -> None:
        with open(self._state_path, "w") as f:
            json.dump([state], f)

    def load_state(self) -> Any:
        if os.path.exists(self._state_path):
            with open(self._state_path, "r") as f:
                return json.load(f)[0]

    def save_instruction_data(self, new_data: List[T]) -> None:
        "Saves instruction data to specified location"
        self.save_data(new_data=new_data, output_path=self._instruction_output_path)
