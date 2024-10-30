# Standard
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
import csv
import json
import os
import random
import re

# Local
from fms_dgt.base.registry import register_datastore
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.datastores.default import DefaultDatastore
from fms_dgt.utils import sdg_logger


@dataclass
class ApiTransformData(SdgData):

    input: str
    output: str
    seed_api_group: str
    split: str


class ApiTransformTask(TransformTask):

    INPUT_DATA_TYPE = ApiTransformData
    OUTPUT_DATA_TYPE = ApiTransformData

    def __init__(
        self,
        *args,
        seed_api_group: str = None,
        api_specifications: Dict = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._seed_api_group = seed_api_group
        self._api_specifications = api_specifications

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            input=kwargs.pop("input", None),
            output=kwargs.pop("output", None),
            split=kwargs.pop("split", None),
            seed_api_group=self._seed_api_group,
            **kwargs,
        )

    def instantiate_instruction(self, data: ApiTransformData):
        api_specs = self._api_specifications.get(data.seed_api_group)
        if api_specs:
            positive_functions = [
                k["name"] for k in data.output
            ]  # {v for k, v in data.output.items() if k == "name"}
            keep_apis = (
                random.sample(
                    list(api_specs.keys()),
                    k=min(len(api_specs), 10),
                )
                + positive_functions
            )
            random.shuffle(keep_apis)
            api_specs = "\n".join([json.dumps(api_specs[k]) for k in keep_apis])
        else:
            api_specs = ""
        data_items = list(asdict(data).items())
        data_items.append(("api_specifications", api_specs))
        output = dict(self._instruction_format)
        for k in output.keys():
            for ds_k, ds_v in data_items:
                inp_key = "{{" + ds_k + "}}"
                if inp_key in output[k]:
                    output[k] = output[k].replace(inp_key, str(ds_v))
        return output


@dataclass
class ApiLlmTransformData(ApiTransformData):
    """This class is intended to hold the seed / machine generated instruction data"""

    dialog_id: str
    speaker: str


class ApiLlmTransformTask(ApiTransformTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiLlmTransformData


@dataclass
class ApiTopv2TransformData(ApiTransformData):
    """This class is intended to hold the seed / machine generated instruction data"""

    question: str
    input_string: str
    domain: str
    ontologies: Optional[List] = None


class ApiTopv2TransformTask(ApiTransformTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiTopv2TransformData


@dataclass
class ApiSnipsAtisTransformData(ApiTransformData):
    """This class is intended to hold the seed / machine generated instruction data"""

    text: str
    intents: str
    slots: List


class ApiSnipsAtisTransformTask(ApiTransformTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiSnipsAtisTransformData


###
# Defining datastores
###


class ApiTransformDatastore(DefaultDatastore):
    """Api transform datastore"""

    def __init__(
        self,
        data_path: str = None,
        splits: List[str] = None,
        extract_fn: Callable = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._data_path = data_path
        self._splits = splits
        self._extract_fn = extract_fn

    def load_data(self) -> List[Dict]:
        raw_data = []
        for split in self._splits:
            sdg_logger.info("======= %s =======", split)
            data_dir = os.path.join(self._data_path, split)
            raw_data.extend(
                {"split": split, **d} for d in self._extract_fn(data_dir)
            )  # combine multiple dialog files
        return raw_data


@register_datastore("api_llm_transform_datastore")
class ApiLlmTransformDatastore(ApiTransformDatastore):
    """Api transform datastore"""

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(extract_fn=extract_raw_llm_data, **kwargs)


@register_datastore("api_topv2_transform_datastore")
class ApiTopv2TransformDatastore(ApiTransformDatastore):
    """Api transform datastore"""

    def load_data(self) -> List[Dict]:
        raw_data = extract_raw_topv2_data(self._data_path)
        return raw_data


@register_datastore("api_snips_atis_transform_datastore")
class ApiSnipsAtisTransformDatastore(ApiTransformDatastore):
    """Api transform datastore"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(extract_fn=extract_raw_snips_atis_data, **kwargs)


def extract_raw_snips_atis_data(fpath: str):
    all_data = []
    texts, slots, intents = read_file(fpath + ".txt")
    for idx in range(len(texts)):
        all_data.append(
            {
                "text": texts[idx],
                "slots": slots[idx],
                "intents": intents[idx],
            }
        )
    return all_data


def read_file(file_path):
    """Read data file of given path.
    :param file_path: path of data file.
    :return: list of sentence, list of slot and list of intent.
    """
    texts, slots, intents = [], [], []
    text, slot = [], []
    with open(file_path, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                if "/" not in items[0]:
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])
                # clear buffer lists.
                text, slot = [], []
            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())
    return texts, slots, intents


def extract_raw_llm_data(raw_data_dir):
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


def extract_raw_topv2_data(data_dir: str):
    all_data = []
    for domain in [
        "navigation",
        "alarm",
        "event",
        "messaging",
        "music",
        "reminder",
        "timer",
        "weather",
    ]:
        for split in ["train", "eval", "test"]:
            json_data = []
            num_intents = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
            tsv_file_path = os.path.join(data_dir, f"{domain}_{split}.tsv")
            # Open and read the TSV file
            with open(tsv_file_path, "r", newline="") as file:
                # Create a CSV reader with tab as the delimiter
                reader = csv.reader(file, delimiter="\t")

                # Iterate through each row in the TSV file
                for idx, row in enumerate(reader):
                    if idx == 0:
                        continue
                    question = row[1]
                    input_string = row[2]
                    utterance_ontologies = get_ontologies(input_string)
                    ontologies = {}
                    ontologies = {
                        key: list(
                            set(
                                ontologies.get(key, [])
                                + utterance_ontologies.get(key, [])
                            )
                        )
                        for key in (ontologies.keys() | utterance_ontologies.keys())
                    }
                    data = {
                        "question": question,
                        "input_string": input_string,
                        "ontologies": ontologies,
                        "split": split,
                        "domain": domain,
                    }
                    all_data.append(data)
    return all_data


def get_ontologies(parsed_string):
    ontologies = {}
    text = re.sub(r"\s*\[\s*", " [ ", parsed_string)
    text = re.sub(r"\s*\]\s*", " ] ", text)
    left_sq_brackets = []
    intents = []
    for token in text.strip().split():
        if token == "[":
            if len(left_sq_brackets) > 0:
                left_sq_brackets[-1] += 1
        elif token == "]":
            left_sq_brackets[-1] -= 1
            if left_sq_brackets[-1] == 0:
                left_sq_brackets.pop()
                intents.pop()
        elif token.startswith("IN:"):
            if len(left_sq_brackets) > 0:
                left_sq_brackets[-1] -= 1
            left_sq_brackets.append(1)
            intent = token[len("IN:") :]
            intents.append(intent)
            ontologies[intent] = ontologies[intent] if intent in ontologies else []
        elif token.startswith("SL:"):
            slot = token[len("SL:") :]
            if slot not in ontologies[intents[-1]]:
                ontologies[intents[-1]].append(slot)
    return ontologies
