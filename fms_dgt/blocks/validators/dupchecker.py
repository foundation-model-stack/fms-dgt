# Standard
from functools import partial
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pandas import DataFrame

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block

try:
    # Third Party
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    pass

@register_block("dupchecker")
class DupCheckerValidator(BaseValidatorBlock):
    """Check for duplicates in the input"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def tokenize(self, inp: Union[List, str]):
        # MC points out that this doesn't even have to tokenize,
        # and that the input may not even be strings
        return inp

    def _validate(self, new_tokens: List, check_tokens: List[List]) -> bool:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""
        return new_tokens not in check_tokens
