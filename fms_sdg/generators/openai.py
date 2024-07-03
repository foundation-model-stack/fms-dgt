"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.util import find_spec
from typing import Any, Dict, List
import copy

# Third Party
from tqdm import tqdm

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import get_resource, register_generator
from fms_sdg.generators.llm import LMGenerator
from fms_sdg.resources.openai import OpenAIKeyResource
import fms_sdg.generators.utils as generator_utils

try:
    # Third Party
    from openai import OpenAI
except ModuleNotFoundError:
    pass


def oa_completion(client, chat: bool = False, **kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install fms_sdg[openai]`"
        )
    else:
        # Third Party
        import openai

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        # Standard
        import traceback

        traceback.print_exc()

    @generator_utils.retry_on_specific_exceptions(
        on_exceptions=[openai.OpenAIError],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        if chat:
            return client.chat.completions.create(**kwargs)
        else:
            return client.completions.create(**kwargs)

    return completion()


@register_generator("openai-chat", "local-chat-completions")
class OpenaiChatCompletionsLM(LMGenerator):
    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__(name, config, **kwargs)
        try:
            # Third Party
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install .[openai]`",
            )

        self.model_id_or_path: str = config.get(
            "model_id", "gpt-3.5-turbo"
        )  # GPT model or Local model using HuggingFace model paths
        self.base_url: str = config.get("base_url", None)
        self.truncate: bool = config.get("truncate", False)

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url)
        else:
            self._openai_resource: OpenAIKeyResource = get_resource(
                "openai", "OPENAI_API_KEY"
            )
            self.client = OpenAI(api_key=self._openai_resource.key)

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    def generate_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        # we group requests by their generation_kwargs,
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
            chunks: List[List[Instance]] = generator_utils.chunks(reqs, n=1)

            for chunk in chunks:
                inputs = [
                    {"role": "user", "content": instance.args[0]} for instance in chunk
                ]
                # all kwargs are identical
                gen_kwargs = next(iter(chunk)).kwargs

                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params in self.config then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                    model_id = kwargs.pop("model_id_or_path", self.model_id_or_path)
                    kwargs["stop"] = until
                    if "stop_sequences" in kwargs:
                        until = kwargs.pop("stop_sequences")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['stop_sequences']` to be of type Union[str,list] but got {until}"
                            )
                    if "max_new_tokens" in kwargs.keys():
                        kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
                    if "min_new_tokens" in kwargs:
                        kwargs.pop("min_new_tokens")
                    if "decoding_method" in kwargs:
                        kwargs.pop("decoding_method")
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                response = oa_completion(
                    client=self.client,
                    chat=True,
                    messages=inputs,
                    model=model_id,
                    **kwargs,
                )

                for resp, instance in zip(response.choices, chunk):
                    s = resp.message.content
                    self.update_instance_with_result(
                        "generate_batch", s, instance, until
                    )
                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")
