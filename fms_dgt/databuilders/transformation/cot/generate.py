# Standard
from typing import Dict, Iterable, List
import copy

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators import BaseValidatorBlock
from fms_dgt.databuilders.transformation.cot.task import CotSdgData, CotTransformTask


@register_data_builder("cot_transform")
class CotTransformDataBuilder(TransformationDataBuilder):

    TASK_TYPE = CotTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # we are intentionally generic with val1 to maximize reuse
    val1: BaseValidatorBlock

    def __call__(self, input_data: List[CotSdgData]) -> Iterable[Dict]:

        llm_inputs = []
        for qa_pair in tqdm(input_data, desc="Data Transformation"):

            stop_seq = (
                [l for l in qa_pair.prompt.split("\n") if "{{input}}" in l][0]
                .split("{{input}}")[0]
                .strip()
            )

            new_inp = qa_pair.prompt.replace("{{input}}", qa_pair.input)
            llm_inputs.append(
                {"prompt": new_inp, "stop_sequences": [stop_seq], "data": qa_pair}
            )

        llm_outputs = self.llm1(
            llm_inputs,
            arg_fields=["prompt"],
            kwarg_fields=["stop_sequences"],
            result_field="llm_result",
        )

        val_outputs = self.val1(
            llm_outputs,
            arg_fields=["prompt", "llm_result"],
            result_field="val_result",
        )

        for output in val_outputs:
            response = output["llm_result"].strip()
            if response:
                new_qa: CotSdgData = copy.deepcopy(output["data"])
                new_qa.output = response
                yield new_qa
