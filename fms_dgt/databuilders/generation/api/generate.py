# Standard
from typing import Any, Dict, Iterable, List, Optional
import json
import random
import time

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import group_data_by_task
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation
from fms_dgt.blocks.validators.rouge import RougeDedupValidator
from fms_dgt.databuilders.generation.api.task import ApiSdgData, ApiSdgTask
from fms_dgt.utils import sdg_logger


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
    val2: RougeDedupValidator

    def call_with_task_list(
        self, request_idx: int, tasks: List[ApiSdgTask]
    ) -> Iterable[ApiSdgData]:

        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        task_api_specifications = dict(
            {task.name: task.all_api_specifications for task in tasks}
        )
        args = [request_idx, task_api_specifications, data_pool]
        kwargs = dict()
        return self(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        all_api_specification_groups: Dict[str, Dict],
        instruction_data: List[ApiSdgData],
    ) -> List[ApiSdgData]:

        # first generate new data
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        gen_inputs: List[Dict] = []
        for task_data in group_data_by_task(instruction_data):
            api_specification_groups = all_api_specification_groups[
                task_data[0].task_name
            ]
            for _ in range(self._num_base_examples):
                prompt, new_instr = self._construct_new_data(
                    api_specification_groups, task_data
                )
                inp = {
                    "prompt": prompt,
                    "gen_kwargs": {"stop_sequences": [f"API:"]},
                    "data": new_instr,
                }
                gen_inputs.append(inp)

        request_start = time.time()
        llm_outputs = self.llm1(gen_inputs, output_map={"result": "output"})
        request_duration = time.time() - request_start

        # now begin filtering generated data
        post_process_start = time.time()

        outputs, wf_discarded = self._wf_filter_data(llm_outputs)

        outputs, rouge_discarded = self._rouge_filter_data(outputs, instruction_data)

        # return
        post_process_duration = time.time() - post_process_start
        sdg_logger.info(
            "Request %s took %.2fs, validation took %.2fs, kept %s instances, discarded %s instances due to violated constraints, discarded %s instances due to rouge similarity",
            request_idx,
            request_duration,
            post_process_duration,
            len(outputs),
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
                    pos_func: new_instr.api_specifications[pos_func]
                    for pos_func in new_instr.positive_functions
                }

                # grab schema from input
                inp = {
                    "api_info": new_apis,
                    "question": question,
                    "answer": answer,
                    "check_arg_question_overlap": new_instr.check_arg_question_overlap,
                    "intent_only": new_instr.intent_only,
                    "require_nested": new_instr.require_nested,
                    "allow_subset": new_instr.allow_subset,
                    "multi_output": new_instr.func_count_bounds[0] > 1,
                    "data": new_instr,
                }

                val1_inputs.append(inp)
            else:
                discarded += 1

        # filter invalid data
        outputs = [output["data"] for output in self.val1(val1_inputs)]

        discarded += len(val1_inputs) - len(outputs)

        return outputs, discarded

    def _rouge_filter_data(
        self, data_to_filter: List[ApiSdgData], instruction_data: List[ApiSdgData]
    ):
        # Rouge filtering
        all_instructions = [instr.input for instr in instruction_data]

        val2_inputs: List[Dict] = []
        for new_data in data_to_filter:
            # computing similarity with the pre-tokenized instructions
            inp = {
                "input": new_data.input,
                "data": new_data,
            }
            val2_inputs.append(inp)

        # filter rouge data
        outputs = [
            output["data"]
            for output in self.val2(val2_inputs, context=all_instructions)
        ]

        discarded = len(val2_inputs) - len(outputs)

        return outputs, discarded

    def _construct_new_data(
        self, api_specification_groups: Dict, task_data: List[ApiSdgData]
    ):
        # gather ICL examples
        groups = list(api_specification_groups.keys())
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
            instr_api_specification = _api_spec_to_str(
                instr.api_specifications,
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
                for k, v in api_specification_groups.items()
                # if k not in [gd.seed_api_group for gd in grouped_data]
            ]
        )
        # TODO: hmm, do we want to weight these?
        # new_group = random.choices(key_lst, weights=key_weights, k=1)[0]
        new_group = random.choices(key_lst, k=1)[0]

        new_pos_ct = random.randint(*new_instr.func_count_bounds)
        new_pos_apis = random.sample(
            list(api_specification_groups[new_group]),
            k=new_pos_ct,
        )

        # when we have single function, we'll just take the first as the target
        if new_instr.single_function:
            new_pos_apis = [new_pos_apis[0]]

        new_api_specification = _api_spec_to_str(
            api_specification_groups[new_group],
            new_pos_apis,
            new_instr.task_name,
        )

        new_instr.api_specifications = api_specification_groups[new_group]
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


###
# Utilities
###


def _api_spec_to_str(
    api_group: Dict,
    pos_functions: List[str],
    task_name: str,
):
    api_infos = [api_group[api_id] for api_id in set(pos_functions)]
    if "parallel_single" in task_name:
        api_infos = [api_infos[0]]
    random.shuffle(api_infos)
    return "\n".join([json.dumps(api_info, indent=4) for api_info in api_infos])
