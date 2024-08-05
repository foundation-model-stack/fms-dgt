# Standard
from typing import Any, Dict, List, Tuple
import copy

# First Party
from templates.databuilder.task import TemplateSdgData, TemplateSdgTask

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


@register_data_builder("data_builder_name")
class TemplateDataBuilder(DataBuilder):
    """Template data builder"""

    TASK_TYPE: SdgTask = TemplateSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # val1 is the validator which checks rouge score
    val1: RougeDedupValidator

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[TemplateSdgData],
    ) -> Tuple[List[TemplateSdgData], int]:

        # None of this code should work, you should replace it with your own SDG flow. However, it will illustrate the general process

        generator_inputs: List[Dict] = []
        for idata in instruction_data:
            # example of how to form an argument to the LLM generator
            prompt = idata.instruction + "\n\n" + idata.input
            inp = {"prompt": prompt, "seed": request_idx, "data": idata}
            generator_inputs.append(inp)

        llm_outputs = self.llm1(
            generator_inputs,
            arg_fields=["prompt"],
            kwarg_fields=["seed"],
            result_field="output",
        )

        validator_inputs = []
        for output in llm_outputs:
            # original input example
            orig_input: TemplateSdgData = output["data"]

            # getting output
            generator_output = output["output"]

            # assign output to instruction
            new_instruction = copy.copy(orig_input)
            new_instruction.instruction = generator_output

            inp = {"to_val": generator_output, "data": new_instruction}
            validator_inputs.append(inp)

        # filter rouge failed data
        outputs = [
            output["data"]
            for output in self.val1(
                validator_inputs, arg_fields=["to_val"], result_field="output"
            )
        ]

        discarded = len(validator_inputs) - len(outputs)

        return outputs, discarded
