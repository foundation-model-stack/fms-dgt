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

    def __init__(self, threshold: float = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._threshold = threshold
        if self._threshold <= 0:
            self._threshold = None

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

        tokenized = []
        for inp in inputs:
            (inp_str,), _ = self.get_args_kwargs(inp, arg_fields, kwarg_fields)
            tokenized.append((self.tokenize(inp_str), inp))

        # first score inputs by rouge similarity to context
        ranked_inputs = []
        for new_tokens, inp in tokenized:
            worst_rouge_score = (
                max(
                    map(
                        partial(rouge_scorer._score_lcs, new_tokens),
                        context,
                    )
                ).fmeasure
                if context
                else 0.0
            )

            if worst_rouge_score < self._threshold or not self._filter_invalids:
                ranked_inputs.append(
                    (
                        worst_rouge_score,
                        worst_rouge_score < self._threshold,
                        new_tokens,
                        inp,
                    )
                )

        ranked_inputs.sort(key=lambda x: x[0])

        # now add
        all_tokens = []
        for _, _, new_tokens, inp in ranked_inputs:
            all_tokens.append(new_tokens)

        outputs = []
        for i, (_, is_valid_wrt_context, new_tokens, inp) in enumerate(ranked_inputs):
            # only check against elements we've already added
            check_against = all_tokens[:i]
            res = self._validate(new_tokens, check_against) and is_valid_wrt_context
            if res or not self._filter_invalids:
                self.write_result(inp, res, result_field)
                outputs.append(inp)

        return outputs

    def _validate(self, new_tokens: List[int], check_tokens: List[List[int]]) -> bool:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""

        if self._threshold is None:
            return True

        if len(check_tokens) == 0:
            return True

        rouge_scores = map(
            partial(rouge_scorer._score_lcs, new_tokens),
            check_tokens,
        )

        return max(rouge_scores, key=lambda x: x.fmeasure).fmeasure < self._threshold
