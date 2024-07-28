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

        self._cache = dict()

        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def tokenize(self, inp: Union[List, str]):
        if type(inp) == list:
            tokenized = []
            for el in inp:
                if el not in self._cache:
                    self._cache[el] = self.scorer._tokenizer.tokenize(el)
                tokenized.append(self._cache[el])
            return tokenized
        else:
            if inp not in self._cache:
                self._cache[inp] = self.scorer._tokenizer.tokenize(inp)
            return self._cache[inp]

    def _validate(self, new_tokens: List[int], check_tokens: List[List[int]]) -> bool:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""
        print("DUPCHECK",new_tokens,check_tokens) ##
        return new_tokens not in check_tokens
