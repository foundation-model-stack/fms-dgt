# Standard
from functools import partial
from typing import Any, List, Optional, Union

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseValidatorBlock
from fms_dgt.base.registry import register_block

try:
    # Third Party
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    pass


@register_block("rouge_scorer")
class RougeDedupValidator(BaseValidatorBlock):
    """Base Class for all Validators"""

    def __init__(self, threshold: float = 1.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if threshold is None:
            # if threshold is set to None, we'll put it as an unreachably high value
            threshold = 1.1

        self._threshold = threshold

        self._cache = dict()

        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def tokenize(self, inp: Union[List, str]):
        if type(inp) == list:
            return [self.tokenize(el) for el in inp]
        else:
            if inp not in self._cache:
                self._cache[inp] = self.scorer._tokenizer.tokenize(inp)
            return self._cache[inp]

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        context: Optional[List[str]] = None,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ):
        """Deduplicator that removes elements of `inputs` that are too rouge-similar. By default it will pick the one that is maximally dissimilar from `context` to keep"""

        # tokenize context
        context = self.tokenize(context) if context else []

        # first score inputs by rouge similarity to context
        ranked_inputs = []
        for inp in inputs:
            (inp_str,), _ = self.get_args_kwargs(inp, arg_fields, kwarg_fields)
            new_tokens = self.tokenize(inp_str)

            worst_rouge_score = (
                max(
                    map(
                        partial(rouge_scorer._score_lcs, new_tokens),
                        context,
                    ),
                    key=lambda x: x.fmeasure,
                ).fmeasure
                if context and self._threshold <= 1
                else -1
            )

            # NOTE: there will be further rewrites in a second PR
            assert worst_rouge_score == self._validate_aux(new_tokens, context)
            is_valid_wrt_context = worst_rouge_score < self._threshold
            if is_valid_wrt_context or not self._filter_invalids:
                ranked_inputs.append(
                    (
                        worst_rouge_score,
                        new_tokens,
                        inp,
                    )
                )

        ranked_inputs.sort(key=lambda x: x[0])

        # now add, in order of increasing similarity to the context

        # only check against elements we've actually added
        check_against = []
        outputs = []
        for i, (worst_rouge_score, new_tokens, inp) in enumerate(ranked_inputs):
            is_valid_wrt_context = worst_rouge_score < self._threshold
            # only check against elements we've already added
            res = self._validate(new_tokens, check_against) and is_valid_wrt_context
            if res or not self._filter_invalids:
                self.write_result(inp, res, result_field)
                outputs.append(inp)
                if res:
                    check_against.append(new_tokens)

        return outputs

    def _validate_aux(
        self, new_tokens: List[int], check_tokens: List[List[int]]
    ) -> float:
        if (
            self._threshold > 1
        ):  # if threshold greater than 1, no need to bother computing this
            return -1

        if len(check_tokens) == 0:
            return -1

        rouge_scores = list(
            map(
                partial(rouge_scorer._score_lcs, new_tokens),
                check_tokens,
            )
        )

        return max(rouge_scores, key=lambda x: x.fmeasure).fmeasure

    def _validate(self, new_tokens: List[int], check_tokens: List[List[int]]) -> bool:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""
        return self._validate_aux(new_tokens, check_tokens) < self._threshold
