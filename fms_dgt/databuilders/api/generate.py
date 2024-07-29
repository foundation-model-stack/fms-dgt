# Standard
from typing import Any, Dict, List, Optional
import random
import time

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import group_data_by_task
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation
from fms_dgt.blocks.validators.rouge import RougeValidator
from fms_dgt.databuilders.api.task import ApiSdgData, ApiSdgTask
from fms_dgt.utils import sdg_logger
import fms_dgt.databuilders.api.utils as api_utils


class ApiDataBuilder(DataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiSdgTask = ApiSdgTask

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
    val1: APIGenSpecValidator
    val2: RougeValidator

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[ApiSdgData],
    ) -> List[ApiSdgData]:

        # first generate new data
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        gen_inputs: List[Dict] = []
        for task_data in group_data_by_task(instruction_data):
            for _ in range(self._num_base_examples):
                prompt, new_instr = self._construct_new_data(task_data)
                inp = {"prompt": prompt, "stop_sequences": [f"API:"], "data": new_instr}
                gen_inputs.append(inp)

        request_start = time.time()
        llm_outputs = self.llm1.generate(
            gen_inputs,
            arg_fields=["prompt"],
            kwarg_fields=["stop_sequences"],
            result_field="output",
        )
        request_duration = time.time() - request_start

        # now begin filtering generated data
        post_process_start = time.time()

        outputs, wf_discarded = self._wf_filter_data(llm_outputs)

        outputs, rouge_discarded = self._rouge_filter_data(outputs, instruction_data)

        # return
        post_process_duration = time.time() - post_process_start
        sdg_logger.info(
            "Request %s took %.2fs, post-processing took %.2fs, discarded %s instances due to violated constraints, discarded %s instances due to rouge similarity",
            request_idx,
            request_duration,
            post_process_duration,
            wf_discarded,
            rouge_discarded,
        )

        return outputs

    def _wf_filter_data(self, data_to_filter: List[Dict]):
        # Well-formedness filtering
        val1_inputs: List[Dict] = []
        discarded = 0
        for gen_inp in data_to_filter:
            new_instr: ApiSdgData = gen_inp["data"]
            components = gen_inp["output"].split("A:")
            if len(components) == 2:
                question, answer = [x.strip() for x in components]
                new_instr.input = question
                new_instr.output = answer
                new_apis = {
                    pos_func: new_instr.api_specifications[new_instr.seed_api_group][
                        pos_func
                    ]
                    for pos_func in new_instr.positive_functions
                }

                # grab schema from input
                inp = {
                    "new_apis": new_apis,
                    "question": question,
                    "answer": answer,
                    "check_arg_question_overlap": new_instr.check_arg_question_overlap,
                    "intent_only": new_instr.intent_only,
                    "require_nested": new_instr.require_nested,
                    "min_ct": (
                        new_instr.func_count_bounds[0]
                        if new_instr.single_function
                        else len(new_instr.positive_functions)
                    ),
                    "max_ct": (
                        new_instr.func_count_bounds[1]
                        if new_instr.single_function
                        else len(new_instr.positive_functions)
                    ),
                    "data": new_instr,
                }

                val1_inputs.append(inp)
            else:
                discarded += 1

        # filter invalid data
        outputs = [
            output["data"]
            for output in self.val1.generate(
                val1_inputs,
                arg_fields=["new_apis", "question", "answer"],
                kwarg_fields=[
                    "check_arg_question_overlap",
                    "intent_only",
                    "require_nested",
                    "min_ct",
                    "max_ct",
                ],
                result_field="output",
            )
        ]

        discarded += len(val1_inputs) - len(outputs)

        return outputs, discarded

    def _rouge_filter_data(
        self, data_to_filter: List[ApiSdgData], orig_data: List[ApiSdgData]
    ):
        # Rouge filtering
        all_instruction_tokens = self.val2.tokenize(
            [instr.input for instr in orig_data]
        )

        outputs: List[Dict] = []
        for new_data in data_to_filter:
            # computing similarity with the pre-tokenized instructions
            new_instruction_tokens = self.val2.tokenize(new_data.input)
            inp = {
                "new_instruction_tokens": new_instruction_tokens,
                "all_instruction_tokens": all_instruction_tokens,
                "data": new_data,
            }
            new_outputs = [
                output["data"]
                for output in self.val2.generate(
                    [inp],
                    arg_fields=["new_instruction_tokens", "all_instruction_tokens"],
                    result_field="output",
                )
            ]
            if new_outputs:
                outputs.extend(new_outputs)
                all_instruction_tokens.append(new_instruction_tokens)

        # filter rouge failed data
        discarded = len(data_to_filter) - len(outputs)

        return outputs, discarded

    def _construct_new_data(self, task_data: List[ApiSdgData]):
        # gather ICL examples
        base_instr = task_data[0]
        groups = list(base_instr.api_specifications.keys())
        random.shuffle(groups)
        grouped_data: List[ApiSdgData] = []
        for group in groups:
            avail_data = [td for td in task_data if td.seed_api_group == group]
            if avail_data:
                grouped_data.append(random.choice(avail_data))

        grouped_data = grouped_data[: random.randint(1, self._num_prompt_instructions)]

        prompt_strings = [grouped_data[0].instruction]
        for instr in grouped_data:
            # TODO: cache string transform
            instr_api_specification = api_utils.api_spec_to_str(
                instr.api_specifications[instr.seed_api_group],
                instr.positive_functions,
                instr.task_name,
            )
            prompt_strings.append(
                f"API:\n{instr_api_specification}\nQ: {instr.input}\nA: {instr.output}"
            )

        # now build new example, we'll copy instr and clear its fields to be safe
        new_instr = instr.make_clear_copy()

        # just use last instruction to select new seed_api_group and positive_functions
        key_lst, key_weights = zip(
            *[
                (k, len(v))
                for k, v in new_instr.api_specifications.items()
                # if k not in [gd.seed_api_group for gd in grouped_data]
            ]
        )
        new_group = random.choices(key_lst, weights=key_weights, k=1)[0]
        new_pos_ct = random.randint(*new_instr.func_count_bounds)
        new_pos_apis = random.sample(
            list(new_instr.api_specifications[new_group]),
            k=new_pos_ct,
        )

        # when we have single function, we'll just take the first as the target
        if new_instr.single_function:
            new_pos_apis = [new_pos_apis[0]]

        new_api_specification = api_utils.api_spec_to_str(
            new_instr.api_specifications[new_group],
            new_pos_apis,
            new_instr.task_name,
        )

        new_instr.positive_functions = new_pos_apis
        new_instr.seed_api_group = new_group

        prompt_strings.append(f"API:\n{new_api_specification}\nQ:")
        prompt = "\n\n".join(prompt_strings)

        return prompt, new_instr


@register_data_builder("api_yes_no_detection")
class ApiYesNoDataBuilder(ApiDataBuilder):
    """Class for API Sequence task"""

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator
    val1: ApiGenSpecYesNoValidation


@register_data_builder("api_function_checking")
class ApiDetectionDataBuilder(ApiDataBuilder):
    """Class for API Sequence task"""

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator
    val1: APIGenSpecValidator
