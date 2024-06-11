# Standard
from functools import partial
from typing import Any, Dict, List, Union

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_validator
from fms_sdg.base.validator import BaseValidator

try:
    # Third Party
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    pass


@register_validator("rouge_scorer")
class RougeValidator(BaseValidator):
    """Base Class for all Validators"""

    def __init__(self, name: str, config: Dict) -> None:
        super().__init__(name, config)
        self._threshold = config.get("threshold", None)
        if self._threshold <= 0:
            self._threshold = None

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

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Takes in a list of Instance objects (each containing their own arg / kwargs) and returns a list of tuples [[<is_true>, instance0], [<is_true>, instance1], ...]"""
        for x in inputs:
            x.result = self._validate(*x.args, **x.kwargs)

    def _validate(self, new_tokens: List[int], check_tokens: List[List[int]]) -> bool:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""

        if self._threshold is None:
            return True

        if new_tokens in check_tokens:
            # remove only first occurrence
            tok_ind = check_tokens.index(new_tokens)
            check_tokens = check_tokens[:tok_ind] + check_tokens[tok_ind + 1 :]

        if len(check_tokens) == 0:
            return True

        rouge_scores = map(
            partial(rouge_scorer._score_lcs, new_tokens),
            check_tokens,
        )

        return max(rouge_scores).fmeasure < self._threshold
