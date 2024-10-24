# Standard
from typing import Any, List
import copy

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.resources.watsonx import WatsonXResource
import fms_dgt.blocks.generators.utils as generator_utils

try:
    # Third Party
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models.schema import (
        TextGenParameters,
        ReturnOptionProperties,
    )
    from ibm_watsonx_ai.foundation_models import Model
except ModuleNotFoundError:
    pass


@register_block("watsonx")
class WatsonXAIGenerator(LMGenerator):
    """WatsonX AI Generator"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            # Third Party
            import ibm_watsonx_ai
        except ModuleNotFoundError as err:
            raise Exception(
                "attempted to use 'watsonx' LM type, but package `ibm_watsonx_ai` not installed. ",
                "please install these via `pip install -r fms_dgt[watsonx]`",
            ) from err

        # Load WatsonX Resource
        self._watsonx_resource: WatsonXResource = get_resource(
            "watsonx",
        )

        # Configure credentials for WatsonX AI service
        if self._watsonx_resource.token:
            self._credentials = Credentials(
                url=self._watsonx_resource.url, token=self._watsonx_resource.token
            )
        else:
            self._credentials = Credentials(
                url=self._watsonx_resource.url, api_key=self._watsonx_resource.key
            )

    def generate_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        # group requests by kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )

        for _, reqs in grouper.get_grouped().items():
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self._watsonx_resource.max_calls
            )

            for chunk in chunks:
                # Prepare inputs
                inputs = [instance.args[0] for instance in chunk]

                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                until = kwargs.get("stop_sequences", None)

                # Initialize generation parameters
                parameters = TextGenParameters(
                    return_options=ReturnOptionProperties(
                        input_text=True,
                    ),
                    **kwargs,
                )

                # Initialize model
                model = Model(
                    model_id=kwargs.pop("model_id_or_path", self.model_id_or_path),
                    credentials=self._credentials,
                    project_id=self._watsonx_resource.project_id,
                )

                # Execute generation routine
                responses = list(model.generate(prompt=inputs, params=parameters))

                # Process generated outputs
                for instance in chunk:
                    result = next(
                        resp.results[0]
                        for resp in responses
                        if instance.args[0] == resp.results[0].input_text
                    )

                    s = result.generated_text
                    self.update_instance_with_result(
                        "generate_batch",
                        s,
                        instance,
                        until,
                    )
                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        # group requests by kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood_batch requests",
        )

        for _, reqs in grouper.get_grouped().items():
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self._watsonx_resource.max_calls
            )

            for chunk in chunks:
                # Prepare inputs
                to_score = ["".join(instance.args) for instance in chunk]
                to_tokenize = [instance.args[-1] for instance in chunk]

                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params in self.config then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                # Initialize generation parameters
                score_params = TextGenParameters(
                    temperature=1.0,
                    decoding_method="greedy",
                    max_new_tokens=1,
                    min_new_tokens=0,
                    return_options=ReturnOptionProperties(
                        input_text=True,
                        generated_tokens=True,
                        input_tokens=True,
                        token_logprobs=True,
                    ),
                )

                # Initialize model
                model = Model(
                    model_id=kwargs.pop("model_id_or_path", self.model_id_or_path),
                    credentials=self._credentials,
                    project_id=self._watsonx_resource.project_id,
                )

                # Execute generation routine
                score_responses = list(
                    model.generate(prompt=to_score, params=score_params)
                )

                # Execute tokenization routine
                tok_responses = [
                    model.tokenize(prompt=entry, return_tokens=True)
                    for entry in to_tokenize
                ]

                for instance in chunk:
                    score_result = next(
                        resp.results[0]
                        for resp in score_responses
                        if "".join(instance.args) == resp.results[0].input_text
                    )
                    tok_count = next(
                        resp.token_count
                        for resp in tok_responses
                        if instance.args[-1] == resp.input_text
                    )

                    s = score_result.input_tokens
                    # tok_ct - 1 since first token in encoding is bos
                    s_toks = s[-(tok_count - 1) :]

                    answer = sum(
                        [tok.logprob for tok in s_toks if tok.logprob is not None]
                    )

                    self.update_instance_with_result(
                        "loglikelihood_batch", answer, instance
                    )
                    pbar.update(1)

        pbar.close()
