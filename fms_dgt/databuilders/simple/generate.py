# Standard
from typing import Any, Dict, List, Optional, Tuple
import copy
import random
import time

# Third Party
import pandas as pd

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask, group_data_by_task
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.rouge import RougeValidator
from fms_dgt.blocks.postprocessors.fuzzy_dedup import FuzzyDedupPostprocessing
from fms_dgt.databuilders.simple.task import InstructLabSdgData, InstructLabSdgTask
from fms_dgt.utils import sdg_logger
import fms_dgt.databuilders.simple.utils as utils


@register_data_builder("simple")
class SimpleInstructDataBuilder(DataBuilder):
    """Class for InstructLab"""

    TASK_TYPE: SdgTask = InstructLabSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # val1 is the validator which checks rouge score
    val1: RougeValidator

    # postproc1 invokes fuzzy dedup postprocessing
    postproc1: FuzzyDedupPostprocessing

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
                    "stop_sequences": [f"* Task {len(prompt_instructions)+2}"],
                    "data": prompt_instructions,
                }
                inputs.append(inp)

        request_start = time.time()

        llm_outputs = self.llm1.generate(inputs)
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

        # now we assess and filter with rouge
        assess_start = time.time()
        all_instruction_tokens = self.val1.tokenize(
            [instr.instruction for instr in instruction_data]
        )

        outputs: List[InstructLabSdgData] = []
        for instruction_data_entry in llm_data:
            # computing similarity with the pre-tokenized instructions
            new_instruction_tokens = self.val1.tokenize(
                instruction_data_entry.instruction
            )
            inp = {
                "new_toks": new_instruction_tokens,
                "all_toks": all_instruction_tokens,
                "data": instruction_data_entry,
            }
            new_outputs = [output["data"] for output in self.val1.generate([inp])]
            if new_outputs:
                outputs.extend(new_outputs)
                all_instruction_tokens.append(new_instruction_tokens)

        # filter rouge failed data

        discarded += len(llm_data) - len(outputs)

        assess_duration = time.time() - assess_start
        sdg_logger.info(
            "Assessing generated samples took %.2fs, discarded %s instances",
            assess_duration,
            discarded,
        )

        return outputs

    def execute_postprocessing(self):
        # kick off fuzzy dedup
        self.postproc1.generate(inputs=None)
