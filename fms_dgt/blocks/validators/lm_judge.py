# Standard
from typing import Any, Callable, Dict, List, Optional, Union

# Local
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators import BaseValidatorBlock
from fms_dgt.constants import DATASET_TYPE, TYPE_KEY


@register_block("llm_judge")
class LMJudgeValidator(BaseValidatorBlock):
    """LLM-based Validator"""

    def __init__(self, lm_config: Dict = None, **kwargs: Any):
        super().__init__(**kwargs)
        assert (
            TYPE_KEY in lm_config
        ), f"Must specify {TYPE_KEY} in 'lm' field of {self.name} block"

        self._llm_generator: LMGenerator = get_block(
            lm_config.get(TYPE_KEY), **lm_config
        )

        self._blocks.append(self._llm_generator)

    def execute(
        self,
        inputs: DATASET_TYPE,
        *,
        fields: Optional[Union[List, Dict]] = None,
        result_field: Optional[str] = None,
        lm_fields: Optional[Union[List, Dict]] = None,
        lm_result_field: Optional[str] = None,
        **kwargs,
    ):

        # simplify generation here
        llm_outputs = self._llm_generator(
            inputs,
            fields=lm_fields,
            result_field=lm_result_field,
            **kwargs,
        )

        judge_outputs, to_save = [], []
        for llm_output in llm_outputs:
            kwargs = self.get_args_kwargs(llm_output, fields=fields)
            success_func = list(kwargs.values())[0]

            lm_res = self._llm_generator.get_result(llm_output, lm_result_field)
            new_result = success_func(lm_res)
            if new_result or not self._filter_invalids:
                self.write_result(llm_output, new_result, result_field=result_field)
                judge_outputs.append(llm_output)

            if not new_result:
                to_save.append(
                    {
                        **kwargs,
                        result_field: new_result,
                    }
                )

        self.save_data(to_save)

        return judge_outputs

    def _validate(self, lm_output: str, success_func: Callable) -> bool:
        return success_func(lm_output)
