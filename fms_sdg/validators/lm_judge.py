# Standard
from typing import Any, Dict, List

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import get_generator, register_validator
from fms_sdg.base.validator import BaseValidator
from fms_sdg.generators.llm import LMGenerator

TYPE_KEY = "lm_type"


@register_validator("llm_judge")
class LMJudgeValidator(BaseValidator):
    """LLM-based Validator"""

    def __init__(self, name: str, config: Dict, **kwargs: Any):
        super().__init__(name, config, **kwargs)
        self._llm_generator: LMGenerator = get_generator(config[TYPE_KEY])(name, config)
        self._generators.append(self._llm_generator)

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        generator_inputs = [Instance([x.args[0]], x.kwargs) for x in inputs]
        self._llm_generator.generate_batch(generator_inputs)
        for gen_inp, inp in zip(generator_inputs, inputs):
            success_func = inp.args[1]
            inp.result = success_func(gen_inp.result)
