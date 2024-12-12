# Standard
from typing import Any, List
import copy
import logging

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.blocks.generators.llm import LMBlockData, LMGenerator
from fms_dgt.resources.watsonx import WatsonXResource
import fms_dgt.blocks.generators.utils as generator_utils

try:
    # Third Party
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference as Model
    from ibm_watsonx_ai.foundation_models.schema import (
        ReturnOptionProperties,
        TextGenParameters,
    )
except ModuleNotFoundError:
    pass


# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)


@register_block("watsonx")
class WatsonXAIGenerator(LMGenerator):
    """WatsonX AI Generator"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

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
        self, requests: List[LMBlockData], disable_tqdm: bool = False
    ) -> None:
        # group requests by kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.gen_kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )

        for _, reqs in grouper.get_grouped().items():
            chunks: List[List[LMBlockData]] = generator_utils.chunks(
                reqs, n=self._watsonx_resource.max_calls
            )

            for chunk in chunks:
                # Prepare inputs
                inputs = [instance.prompt for instance in chunk]

                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).gen_kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                until = kwargs.get("stop_sequences", None)

                # Initialize model
                model = Model(
                    model_id=kwargs.pop("model_id_or_path", self.model_id_or_path),
                    credentials=self._credentials,
                    project_id=self._watsonx_resource.project_id,
                    params=TextGenParameters(
                        **kwargs,
                    ),
                )

                # Execute generation routine
                responses = model.generate(prompt=inputs)

                # Process generated outputs
                for idx, instance in enumerate(chunk):
                    self.update_instance_with_result(
                        "generate_batch",
                        responses[idx]["results"][0]["generated_text"],
                        instance,
                        until,
                        additional={
                            {
                                "gen_token_count": responses[idx]["results"][0][
                                    "generated_token_count"
                                ]
                            }
                        },
                    )
                    pbar.update(1)

                # Clean up model object
                model = None
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

                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params in self.config then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                # Initialize model
                model = Model(
                    model_id=kwargs.pop("model_id_or_path", self.model_id_or_path),
                    credentials=self._credentials,
                    project_id=self._watsonx_resource.project_id,
                    params=TextGenParameters(
                        temperature=1.0,
                        decoding_method="greedy",
                        max_new_tokens=1,
                        min_new_tokens=0,
                        return_options=ReturnOptionProperties(
                            generated_tokens=True,
                            token_logprobs=True,
                        ),
                    ),
                )

                # Execute generation routine
                score_responses = model.generate(prompt=to_score)

                for idx, instance in enumerate(chunk):
                    score_result = score_responses[idx]["results"][0]

                    # tok_ct - 1 since first token in encoding is bos
                    generated_tokens = score_result["generated_tokens"][
                        -(score_result["generated_token_count"] - 1) :
                    ]

                    answer = sum(
                        [
                            tok["logprob"]
                            for tok in generated_tokens
                            if tok["logprob"] is not None
                        ]
                    )

                    self.update_instance_with_result(
                        "loglikelihood_batch", answer, instance
                    )
                    pbar.update(1)

                # Clean up model object
                model = None

        pbar.close()
