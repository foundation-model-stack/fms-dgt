# Standard
from typing import Any, Iterable, List, Optional

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.api_transformation.task import (
    ApiTransformData,
    ApiTransformTask,
)


@register_data_builder("transform_api_llm")
class ApiTransformDataBuilder(TransformationDataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiTransformTask = ApiTransformTask

    def __init__(
        self,
        *args: Any,
        num_prompt_instructions: Optional[int] = 3,
        num_base_examples: Optional[int] = 10,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._num_prompt_instructions = num_prompt_instructions
        self._num_base_examples = num_base_examples
        assert (
            self._num_prompt_instructions >= 1
        ), "Number of prompt examples must be at least 1"

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    def __call__(
        self,
        instruction_data: List[ApiTransformData],
    ) -> Iterable[ApiTransformData]:

        api_str_list, api_str_dialog_map, dialog_info = [], {}, {}
        for d in tqdm(instruction_data, desc="API transformation"):
            if d.output and "NONE(" not in d.output:
                api_str_list.append(d.output)
                api_str_dialog_map.setdefault(d.dialog_id, []).append(d.output)
                dialog_info[d.dialog_id] = (
                    d.split,
                    d.task_name,
                    d.speaker,
                )

        api_to_str = self.generate_llm_paraphrase(api_str_list)

        # reconstruct the data with llm-paraphrases
        for dialog_id, conv in api_str_dialog_map.items():
            split, task_name, speaker = dialog_info[dialog_id]
            input_list, output_list, api_list, intents = [], [], [], []
            for apis in conv:
                api_list.extend(apis.split("[SEP]"))
            for api in api_list:
                intent = api[: api.index("(")]
                if intent not in intents:
                    intents.append(intent)
            api_list = [
                max([api for api in api_list if api.startswith(intent)], key=len)
                for intent in intents
            ]  # take the longest string of intents (more slots).
            for api in api_list:
                if api in api_to_str.keys():
                    output_list.append(api)
                    api_str = api_to_str[api].lower()
                    api_str = api_str + "." if not api_str.endswith(".") else api_str
                    input_list.append(api_str)
            if input_list and output_list:
                yield ApiTransformData(
                    **{
                        "speaker": speaker,
                        "dialog_id": dialog_id,
                        "split": split,
                        "task_name": task_name,
                        "input": " ".join(input_list),
                        "output": " [SEP] ".join(output_list),
                    }
                )

    def generate_llm_paraphrase(self, api_str_list):
        single_api_str_list = [api.split("[SEP]") for api in api_str_list]
        single_api_str_list = [
            item.strip() for sublist in single_api_str_list for item in sublist
        ]  # flatten
        api_to_str = {}
        prompts = []
        for api in single_api_str_list:
            # output_string = f"intent: {api}"
            output_string = api
            prompt = (
                f"Convert the following intent and its parameters into an imperative sentence. Do not copy the API or its parameters as is in the output sentence.\n\nInput:\n"
                + output_string
                + "\nOutput:\n"
            )
            prompts.append({"prompt": prompt, "api": api})
        outputs = self.llm1.generate(prompts)
        for output in outputs:
            api_to_str[output["api"]] = output["result"].strip()
        return api_to_str
