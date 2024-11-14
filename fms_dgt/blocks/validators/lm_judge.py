# Standard
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

# Local
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.blocks.generators.llm import LMBlockData, LMGenerator
from fms_dgt.blocks.validators import BaseValidatorBlock, BaseValidatorBlockData
from fms_dgt.constants import TYPE_KEY


@dataclass(kw_only=True)
class LMJudgeData(BaseValidatorBlockData, LMBlockData):
    success_func: Callable


@register_block("llm_judge")
class LMJudgeValidator(BaseValidatorBlock):
    """LLM-based Validator"""

    DATA_TYPE = LMJudgeData

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
        inputs: Iterable[LMJudgeData],
        **kwargs,
    ):

        # simplify generation here
        llm_outputs: List[LMJudgeData] = self._llm_generator(
            inputs,
            **kwargs,
        )

        judge_outputs, to_save = [], []
        for llm_output in llm_outputs:
            llm_output.is_valid = llm_output.success_func(llm_output.result)
            if llm_output.is_valid or not self._filter_invalids:
                judge_outputs.append(llm_output)

            if not llm_output.is_valid:
                to_save.append(llm_output)

        self.save_data(to_save)

        return judge_outputs

    def _validate(self, lm_output: str, success_func: Callable) -> bool:
        return success_func(lm_output)
