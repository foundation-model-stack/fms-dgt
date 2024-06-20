# Standard
from collections import defaultdict
from typing import Any, Dict, List
import copy
import os

# Third Party
from tqdm import tqdm

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import get_resource, register_generator
from fms_sdg.generators.llm import LMGenerator
from fms_sdg.resources.genai import GenAIKeyResource
import fms_sdg.generators.utils as generator_utils
import fms_sdg.utils as utils

try:
    # Third Party
    from dotenv import load_dotenv
    from genai import Client, Credentials
    from genai.schema import (
        TextGenerationParameters,
        TextGenerationReturnOptions,
        TextTokenizationParameters,
        TextTokenizationReturnOptions,
    )
except ModuleNotFoundError:
    pass


@register_generator("genai")
class GenAIGenerator(LMGenerator):
    """GenAI Generator"""

    def __init__(self, name: str, config: Dict, **kwargs: Any):
        super().__init__(name, config, **kwargs)

        try:
            # Third Party
            import genai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'genai' LM type, but package `genai` not installed. ",
                "please install these via `pip install -r fms_sdg[genai]`",
            )

        self._genai_resource: GenAIKeyResource = get_resource("genai", "GENAI_KEY")

        load_dotenv()
        credentials = Credentials(
            self._genai_resource.key, api_endpoint=os.getenv("GENAI_API", None)
        )
        self.client = Client(credentials=credentials)

    @property
    def eot_token_id(self):
        return ""

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        return context, continuation

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(
                generator_utils.chunks(
                    re_ord.get_reordered(), self._genai_resource.max_calls
                )
            ),
            disable=disable_tqdm,
        ):
            inps = []
            continuations = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = context_enc + continuation_enc
                inps.append(inp)
                continuations.append(continuation_enc)

            score_params = TextGenerationParameters(
                temperature=1.0,
                decoding_method="greedy",
                max_new_tokens=1,
                min_new_tokens=0,
                return_options=TextGenerationReturnOptions(
                    generated_tokens=True,
                    token_logprobs=True,
                    input_text=True,
                    input_tokens=True,
                ),
            )

            score_responses = list(
                self.client.text.generation.create(
                    model_id=self.model_id_or_path,
                    inputs=inps,
                    parameters=score_params,
                )
            )

            tok_responses = next(
                self.client.text.tokenization.create(
                    model_id=self.model_id_or_path,
                    input=continuations,
                    parameters=TextTokenizationParameters(
                        return_options=TextTokenizationReturnOptions(tokens=True)
                    ),
                )
            ).results

            for resp, tok_resp, (cache_key, context_enc, continuation_enc) in zip(
                score_responses, tok_responses, chunk
            ):
                s = resp.results[0].input_tokens
                # tok_ct - 1 since first token in encoding is bos
                s_toks = s[-(tok_resp.token_count - 1) :]
                answer = sum([tok.logprob for tok in s_toks if tok.logprob is not None])

                res.append(answer)

        return re_ord.get_original(res)

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

        for key, reqs in grouper.get_grouped().items():
            # n needs to be 1 because messages in
            # chat completion are not batch but
            # is regarded as a single conversation.
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self._genai_resource.max_calls
            )

            for chunk in chunks:
                inputs = [instance.args[0] for instance in chunk]
                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params in self.config then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                    until = kwargs.get("stop_sequences", None)
                    model_id = kwargs.pop("model_id", self.model_id_or_path)
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                parameters = TextGenerationParameters(
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                    ),
                    **kwargs,
                )

                responses = list(
                    self.client.text.generation.create(
                        model_id=model_id,
                        inputs=inputs,
                        parameters=parameters,
                    )
                )

                for instance in chunk:
                    result = next(
                        resp.results[0]
                        for resp in responses
                        if instance.args[0] == resp.results[0].input_text
                    )

                    s = result.generated_text
                    self.update_instance_with_result(s, instance, until)
                    pbar.update(1)

        pbar.close()
