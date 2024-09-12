# Standard
from typing import Any, List, TypeVar
import json
import os

# Third Party
import datasets
import pandas as pd
import yaml

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_datastore

T = TypeVar("T")


@register_datastore("default")
class DefaultDatastore(BaseDatastore):
    """Base Class for all data stores"""

    def __init__(
        self,
        output_dir: str = None,
        store_name: str = None,
        output_format: str = "jsonl",
        restart: bool = False,
        seed_examples: List[T] = None,
        data_path: str = None,
        hf_args_or_path: List = None,
        data_split: str = "train",
        **kwargs,
    ) -> None:
        super().__init__()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._output_dir = self._get_default_output_dir(output_dir, store_name)
        self._output_path = os.path.join(self._output_dir, "outputs." + output_format)
        self._final_output_path = os.path.join(
            self._output_dir, "final_outputs." + output_format
        )
        self._state_path = os.path.join(self._output_dir, "dataloader_state.txt")
        self._dataset_path = data_path
        self._hf_args = hf_args_or_path
        if type(self._hf_args) == str:
            self._hf_args = [self._hf_args]
        self._dataset_split = data_split
        self._seed_examples = seed_examples or []

        if self._dataset_path and self._hf_args:
            raise ValueError(
                "Cannot set both 'data_path' and 'hf_args' in datastore config"
            )

        if restart and os.path.exists(self.output_path):
            os.remove(self.output_path)

        # always delete instruction output path because it's regenerated from machine_data
        if os.path.exists(self._final_output_path):
            os.remove(self._final_output_path)

    @property
    def output_path(self) -> str:
        return self._output_path

    def _get_default_output_dir(self, output_dir: str, store_name: str):
        path_components = []
        path_components.append(output_dir)
        path_components.append(store_name.replace("->", "__"))
        return os.path.join(*path_components)

    def save_data(
        self,
        new_data: List[T],
        output_path: str = None,
    ) -> None:

        output_path = output_path if output_path is not None else self.output_path
        output_format = os.path.splitext(output_path)[-1]

        if output_format == ".jsonl":
            _write_json(new_data, output_path)
        elif output_format == ".parquet":
            _write_parquet(new_data, output_path)
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

    def load_data(self, output_path: str = None) -> List[T]:

        output_path = output_path if output_path is not None else self.output_path

        if not os.path.exists(output_path):
            return

        output_format = os.path.splitext(output_path)[-1]
        if output_format == ".jsonl":
            machine_data = _read_json(output_path)
        elif output_format == ".parquet":
            machine_data = _read_parquet(output_path)
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

        return machine_data

    def load_dataset(self) -> List[T]:

        seed_data = self._seed_examples
        data = []

        if self._dataset_path:
            if self._dataset_path.endswith(".json"):
                data = _read_json(self._dataset_path)
            elif self._dataset_path.endswith(".yaml"):
                data = _read_yaml(self._dataset_path)
            elif self._dataset_path.endswith(".parquet"):
                data = _read_parquet(self._dataset_path)
            elif os.path.isdir(self._dataset_path):
                data = _read_huggingface([self._dataset_path], self._dataset_split)
            else:
                raise ValueError(f"Unhandled data path input [{self._dataset_path}]")
        elif self._hf_args:
            data = _read_huggingface(self._hf_args, self._dataset_split)

        seed_data = _add_seed_data(data, seed_data)

        return seed_data

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
        self.save_data(new_data=new_data, output_path=self._final_output_path)

    def load_instruction_data(self) -> List[T]:
        "Loads instruction data from specified location"
        return self.load_data(output_path=self._instruction_output_path)


###
# Utilities
###


def _write_json(new_data: List[T], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")


def _write_parquet(new_data: List[T], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(new_data).to_parquet(
        output_path,
        engine="fastparquet",
        append=os.path.isfile(output_path),
    )


def _read_json(output_path: str):
    with open(output_path, "r") as f:
        try:
            machine_data = [json.loads(l.strip()) for l in f.readlines()]
        except ValueError:
            machine_data = []
    return machine_data


def _read_parquet(output_path: str):
    machine_data = (
        pd.read_parquet(output_path, engine="fastparquet").apply(dict, axis=1).to_list()
    )
    return machine_data


def _read_yaml(output_path: str):
    with open(output_path, "r") as f:
        machine_data = list(yaml.safe_load(f))
    return machine_data


def _read_huggingface(dataset_args: List[str], split: str):
    data = datasets.load_dataset(*dataset_args)
    if split:
        data = data[split]
    return data


def _add_seed_data(dataset: DATASET_TYPE, seed_data: List):
    if seed_data:
        if type(dataset) == list:
            dataset = dataset + seed_data
        elif type(dataset) == datasets.Dataset:
            seed_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=seed_data))
            dataset = datasets.concatenate_datasets([dataset, seed_dataset])
        else:
            raise ValueError(
                f"Data used for default 'load_dataset' method must be one of {DATASET_TYPE}!"
            )
    return dataset
