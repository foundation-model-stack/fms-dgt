# Standard
from typing import Any, List, Optional, Tuple
import copy
import random
import time

# Third Party
import pandas as pd

# Local
from fms_sdg.base.databuilder import DataBuilder
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_data_builder
from fms_sdg.base.task import SdgTask, group_data_by_task
from fms_sdg.blocks.generators.llm import LMGeneratorBlock
from fms_sdg.blocks.validators.rouge import RougeValidator
from fms_sdg.databuilders.simple.task import InstructLabSdgData, InstructLabSdgTask
from fms_sdg.utils import sdg_logger
import fms_sdg.databuilders.simple.utils as utils


@register_data_builder("simple")
class SimpleInstructDataBuilder(DataBuilder):
    """Class for InstructLab"""

    TASK_TYPE: SdgTask = InstructLabSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGeneratorBlock

    # val1 is the validator which checks rouge score
    val1: RougeValidator

    def __init__(
        self,
        *args: Any,
        num_prompt_instructions: int = 2,
        prompt_file_path: str = "prompt.txt",
        request_batch_size: int = 5,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._prompt_template = utils.check_prompt_file(
            prompt_file_path, self.llm1.model_id_or_path
        )
        self._num_prompt_instructions = num_prompt_instructions
        self._request_batch_size = request_batch_size

    def _encode_prompt(self, prompt_instructions):
        # defining this as its own separate method allows us to overwrite it for subclasses
        prompt = utils.encode_prompt(prompt_instructions, self._prompt_template)
        return prompt

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[InstructLabSdgData],
    ) -> List[InstructLabSdgData]:

        print(pd.DataFrame([instruction_data[0]]).to_markdown())
        input("--")

        inputs: List[Instance] = []
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        for grouped_data in group_data_by_task(instruction_data):
            for i in range(0, len(grouped_data), self._num_prompt_instructions):
                prompt_instructions = grouped_data[
                    i : i + self._num_prompt_instructions
                ]
                prompt = self._encode_prompt(prompt_instructions)
                args = [prompt]
                kwargs = {"stop_sequences": [f"* Task {len(prompt_instructions)+2}"]}
                print(
                    pd.DataFrame(
                        [Instance(args, kwargs, data=prompt_instructions)]
                    ).to_markdown()
                )
                input("--")
                inputs.append(Instance(args, kwargs, data=prompt_instructions))

        request_start = time.time()
        self.llm1.generate_batch(inputs)
        request_duration = time.time() - request_start

        post_process_start = time.time()
        instruction_data = []
        for gen_inp in inputs:
            prompt_instructions: List[InstructLabSdgData] = gen_inp.data
            new_instruction_dicts, discarded = utils.post_process_gpt3_response(
                len(prompt_instructions),
                gen_inp.result,
            )
            # make sure the generated instruction carried over extra fields
            for new_ins_dict, orig_ins in zip(
                new_instruction_dicts, prompt_instructions
            ):
                new_ins = copy.copy(orig_ins)
                new_ins.instruction = new_ins_dict.get("instruction")
                new_ins.input = new_ins_dict.get("input")
                new_ins.output = new_ins_dict.get("output")
                instruction_data.append(new_ins)

        post_process_duration = time.time() - post_process_start
        sdg_logger.debug(
            "Request %s took %.2fs, post-processing took %.2fs",
            request_idx,
            request_duration,
            post_process_duration,
        )

        # now we assess and filter with rouge
        assess_start = time.time()
        all_instruction_tokens = self.val1.tokenize(
            [instr.instruction for instr in instruction_data]
        )

        val1_inputs: List[Instance] = []
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenized instructions
            new_instruction_tokens = self.val1.tokenize(
                instruction_data_entry.instruction
            )
            args = [new_instruction_tokens, all_instruction_tokens]
            val1_inputs.append(Instance(args, data=instruction_data_entry))
        self.val1.validate_batch(val1_inputs)

        # filter rouge failed data
        outputs = [val1_input.data for val1_input in val1_inputs if val1_input.result]
        discarded += len(val1_inputs) - len(outputs)

        assess_duration = time.time() - assess_start
        sdg_logger.debug(
            "Assessing generated samples took %.2fs, discarded %s instances",
            assess_duration,
            discarded,
        )

        return outputs
