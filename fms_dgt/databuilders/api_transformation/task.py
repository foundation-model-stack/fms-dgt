# Standard
from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os

# Local
from fms_dgt.base.registry import register_datastore
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.datastores.default import DefaultDatastore
from fms_dgt.utils import sdg_logger


@dataclass
class ApiTransformData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    input: str
    output: str
    dialog_id: str
    speaker: str
    split: str


class ApiTransformTask(SdgTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiTransformData
    OUTPUT_DATA_TYPE = ApiTransformData


@register_datastore("api_datastore")
class ApiDatastore(DefaultDatastore):
    """Api transform datastore"""

    def __init__(
        self,
        data_path: str = None,
        splits: List[str] = None,
        restart_generation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._data_path = data_path
        self._splits = splits

        for split in self._splits:
            output_dir, output_file = os.path.split(self.output_path)
            output_path = os.path.join(output_dir, split, output_file)
            if restart_generation and os.path.exists(output_path):
                os.remove(output_path)

    def save_data(self, new_data: List[Dict]) -> None:
        for split in self._splits:
            split_data = [data for data in new_data if data["split"] == split]
            output_dir, output_file = os.path.split(self.output_path)
            output_path = os.path.join(output_dir, split, output_file)
            return super().save_data(split_data, output_path=output_path)

    def load_data(self) -> List[ApiTransformData]:
        machine_examples = []
        for split in self._splits:
            output_dir, output_file = os.path.split(self.output_path)
            output_path = os.path.join(output_dir, split, output_file)
            if os.path.exists(output_path):
                machine_examples.extend(super().load_data(output_path=output_path))
        return machine_examples

    def load_dataset(self) -> List[ApiTransformData]:
        raw_data = []
        for split in self._splits:
            sdg_logger.info("======= %s =======", split)
            data_dir = os.path.join(self._data_path, split)
            raw_data.extend(
                {"split": split, **d} for d in extract_raw_data(data_dir)
            )  # combine multiple dialog files
        return raw_data


def extract_raw_data(raw_data_dir):
    data_files = [
        item
        for item in os.listdir(raw_data_dir)
        if os.path.isfile(os.path.join(raw_data_dir, item))
        and not item == "schema.json"
    ]
    processed_data = []
    for file in data_files:  # tqdm(data_files):
        sdg_logger.info(file)
        data = json.load(open(os.path.join(raw_data_dir, file)))
        sdg_logger.info(len(data))
        for d in data:  # ):  # each dialog
            for t in d["turns"]:  # each turns
                if t["speaker"] == "USER":
                    turn_intent_slots = []
                    for f in t["frames"]:
                        if (
                            f["state"]["slot_values"]
                            and not f["state"]["active_intent"] == "NONE"
                        ):
                            turn_slots = []
                            for slot, values in f["state"]["slot_values"].items():
                                turn_slots.append(f"{slot} = {values[0]}")
                            slot_str = " ; ".join(turn_slots)
                            turn_intent_slots.append(
                                f"{f['state']['active_intent']}({slot_str})"
                            )
                    api_str = " [SEP] ".join(turn_intent_slots)
                    processed_data.append(
                        {
                            "dialog_id": d["dialogue_id"],
                            "speaker": "USER",
                            "input": t["utterance"],
                            "output": api_str,
                        }
                    )
                else:
                    processed_data.append(
                        {
                            "dialog_id": d["dialogue_id"],
                            "speaker": "BOT",
                            "input": t["utterance"],
                            "output": "",
                        }
                    )
    return processed_data
