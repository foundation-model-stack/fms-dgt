# Standard
from typing import Any, Dict, List
import copy
import random
import time

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask, group_data_by_task
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators import BaseValidatorBlock
from fms_dgt.databuilders.generation.simple.task import (
    InstructLabSdgData,
    InstructLabSdgTask,
)
from fms_dgt.utils import sdg_logger
import fms_dgt.databuilders.generation.simple.utils as utils


@register_data_builder("simple")
class SimpleInstructDataBuilder(DataBuilder):
    """Class for InstructLab"""

    TASK_TYPE: SdgTask = InstructLabSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # val1 is the validator, we leave this intentionally generic
    val1: BaseValidatorBlock

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

        inputs: List[Dict] = []
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        for grouped_data in group_data_by_task(instruction_data):
            for i in range(0, len(grouped_data), self._num_prompt_instructions):
                prompt_instructions = grouped_data[
                    i : i + self._num_prompt_instructions
                ]
                prompt = self._encode_prompt(prompt_instructions)
                inp = {
                    "prompt": prompt,
                    "gen_kwargs": {
                        "stop_sequences": [f"* Task {len(prompt_instructions)+2}"]
                    },
                    "data": prompt_instructions,
                }
                inputs.append(inp)

        request_start = time.time()

        llm_outputs = self.llm1(inputs, output_map={"result": "output"})
        request_duration = time.time() - request_start

        post_process_start = time.time()
        llm_data: List[InstructLabSdgData] = []
        for gen_inp in llm_outputs:
            prompt_instructions: List[InstructLabSdgData] = gen_inp["data"]
            new_instruction_dicts, discarded = utils.post_process_gpt3_response(
                len(prompt_instructions),
                gen_inp["output"],
            )
            # make sure the generated instruction carried over extra fields
            for new_ins_dict, orig_ins in zip(
                new_instruction_dicts, prompt_instructions
            ):
                new_ins = copy.copy(orig_ins)
                new_ins.instruction = new_ins_dict.get("instruction")
                new_ins.input = new_ins_dict.get("input")
                new_ins.output = new_ins_dict.get("output")
                llm_data.append(new_ins)

        post_process_duration = time.time() - post_process_start
        sdg_logger.info(
            "Request %s took %.2fs, post-processing took %.2fs",
            request_idx,
            request_duration,
            post_process_duration,
        )

        # now we assess and filter with val1
        assess_start = time.time()

        val_inputs: List[InstructLabSdgData] = []
        for instruction_data_entry in llm_data:
            # computing similarity with the pre-tokenized instructions
            inp = {
                "input": instruction_data_entry.instruction,
                "data": instruction_data_entry,
            }
            val_inputs.append(inp)

        # filter data
        outputs = [output["data"] for output in self.val1(val_inputs)]

        discarded += len(llm_data) - len(outputs)

        assess_duration = time.time() - assess_start
        sdg_logger.info(
            "Assessing generated samples took %.2fs, discarded %s instances",
            assess_duration,
            discarded,
        )

        return outputs
