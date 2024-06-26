# Standard
from typing import Any, List, Tuple
import copy

# First Party
from templates.databuilder.task import TemplateSdgData, TemplateSdgTask

# Local
from fms_sdg.base.databuilder import DataBuilder
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_data_builder
from fms_sdg.base.task import SdgTask
from fms_sdg.generators.llm import LMGenerator
from fms_sdg.validators.rouge import RougeValidator


@register_data_builder("data_builder_name")
class TemplateDataBuilder(DataBuilder):
    """Template data builder"""

    TASK_TYPE: SdgTask = TemplateSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # val1 is the validator which checks rouge score
    val1: RougeValidator

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

        generator_inputs: List[Instance] = []
        for idata in instruction_data:
            # example of how to form an argument to the LLM generator
            prompt = idata.instruction + "\n\n" + idata.input
            args = [prompt]
            kwargs = {"seed": request_idx}
            generator_inputs.append(Instance(args, kwargs, data=idata))

        self.llm1.generate_batch(generator_inputs)

        validator_inputs = []
        for generator_input in generator_inputs:
            # original input example
            orig_input: TemplateSdgData = generator_input.data

            # getting output
            generator_output = generator_input.result

            # assign output to instruction
            new_instruction = copy.copy(orig_input)
            new_instruction.instruction = generator_output

            args = [generator_output]
            validator_inputs.append(Instance(args, data=new_instruction))

        self.val1.validate_batch(validator_inputs)

        # filter rouge failed data
        outputs = [
            validator_input.data
            for validator_input in validator_inputs
            if validator_input.result
        ]

        discarded = len(validator_inputs) - len(outputs)

        return outputs, discarded
