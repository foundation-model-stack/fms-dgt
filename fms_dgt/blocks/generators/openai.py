"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.util import find_spec
from typing import Any, Dict, List, Union
import copy
import logging

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.blocks.generators.llm import MODEL_ID_OR_PATH, LMBlockData, LMGenerator
from fms_dgt.resources.api import ApiKeyResource
from fms_dgt.utils import sdg_logger
import fms_dgt.blocks.generators.utils as generator_utils

try:
    # Third Party
    from openai import OpenAI
    import openai
except ModuleNotFoundError:
    pass

# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)


def oa_completion(
    client: OpenAI, chat: bool = False, **kwargs
) -> openai.types.completion.Completion:
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install fms_dgt[openai]`"
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


@register_block("openai", "vllm-remote", "rits")
class OpenaiCompletionsLM(LMGenerator):
    def __init__(
        self,
        api_key: str = "EMPTY",
        call_limit: int = 10,
        base_url: str = None,
        auto_chat_template: bool = False,
        **kwargs: Any,
    ):
        # only use auto_chat_template when model backend is vllm and auto_chat_template
        auto_chat_template = base_url and auto_chat_template
        super().__init__(auto_chat_template=auto_chat_template, **kwargs)

        try:
            # Third Party
            import openai  # noqa: E401
            import tiktoken
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"attempted to use '{self.block_type}' LM type, but package `openai` or `tiktoken` are not installed. please install these via `pip install .[openai]`",
            )

        self.base_url = base_url
        self._chat = False

        if self.batch_size is None:
            self._batch_size = 10

        # Set to EMPTY for local
        if self.base_url:
            # Read RITS Key from environment and change header to recognize RITS key
            if self.block_type == "rits":
                self._rits_resource: ApiKeyResource = get_resource(
                    "api", key_name="RITS_API_KEY", call_limit=self.batch_size
                )
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.base_url,
                    default_headers={"RITS_API_KEY": self._rits_resource.key},
                )
            else:
                self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            # Read from environment variable OPENAI_API_KEY
            self._openai_resource: ApiKeyResource = get_resource(
                "api", key_name="OPENAI_API_KEY", call_limit=call_limit
            )
            self.client = OpenAI(api_key=self._openai_resource.key)
            if auto_chat_template:
                sdg_logger.warning(f"auto_chat_template is disabled for OpenAI models")

    def _prepare_input(self, prompt: str):
        return prompt

    def _extract_output(self, resp) -> str:
        return resp.text

    def _extract_gen_token_count(self, resp) -> int:
        return resp.usage.completion_tokens

    def generate_batch(
        self, requests: List[LMBlockData], disable_tqdm: bool = False
    ) -> None:
        # we group requests by their generation_kwargs,
        grouper = generator_utils.Grouper(requests, lambda x: str(x.gen_kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )

        for key, reqs in grouper.get_grouped().items():
            chunks: List[List[LMBlockData]] = generator_utils.chunks(
                reqs, n=self.batch_size
            )

            for chunk in chunks:
                inputs = [self._prepare_input(instance.prompt) for instance in chunk]
                # all kwargs are identical
                gen_kwargs = next(iter(chunk)).gen_kwargs
                kwargs = self.modify_gen_kwargs(gen_kwargs)
                kwargs[("messages" if self._chat else "prompt")] = inputs

                model_id = kwargs.pop(MODEL_ID_OR_PATH, self.model_id_or_path)
                until = kwargs.get("stop", None)

                response = oa_completion(
                    client=self.client,
                    chat=self._chat,
                    model=model_id,
                    **kwargs,
                )

                gen_token_count = self._extract_gen_token_count(response)

                n = kwargs.get("n", 1)
                if len(response.choices) != n * len(chunk):
                    raise AssertionError(
                        f"Number of responses does not match number of inputs * n, [{len(response.choices)}, {n}, {len(chunk)}]"
                    )

                resp_groups = [
                    response.choices[i : i + n]
                    for i in range(0, len(response.choices), n)
                ]
                for resp_group, instance in zip(resp_groups, chunk):
                    n_s = []
                    for resp in resp_group:
                        s = self._extract_output(resp)
                        addtl = {"gen_token_count": gen_token_count}
                        n_s.append((s, addtl))

                    self.update_instance_with_result(
                        "generate_batch",
                        ([s for s, _ in n_s] if len(n_s) > 1 else s),
                        instance,
                        until,
                        ([addtl for _, addtl in n_s] if len(n_s) > 1 else addtl),
                    )

                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(self, *args, **kwargs):
        raise NotImplementedError("No support for logits.")

    def modify_gen_kwargs(self, gen_kwargs: dict) -> dict:
        # sampling_params
        if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
            # start with default params then overwrite with kwargs
            kwargs = {**self._base_kwargs, **kwargs}
            until = None
            if "stop_sequences" in kwargs:
                until = kwargs.pop("stop_sequences")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `kwargs['stop_sequences']` to be of type Union[str,list] but got {until}"
                    )
            kwargs["stop"] = until
            if "max_new_tokens" in kwargs.keys():
                kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
            if "min_new_tokens" in kwargs:
                kwargs.pop("min_new_tokens")
            if "decoding_method" in kwargs:
                kwargs.pop("decoding_method")

            kwargs.pop("random_seed", None)

        else:
            raise ValueError(
                f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
            )
        return kwargs


@register_block("openai-chat", "vllm-remote-chat")
class OpenaiChatCompletionsLM(OpenaiCompletionsLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chat = True
        if self.batch_size is None:
            self._batch_size = 10
        if self.block_type == "openai-chat":
            sdg_logger.warning(f"OpenAI Chat models only support batch size of 1")
            self._batch_size = 1

    def _prepare_input(self, prompt: Union[str, List[Dict]]):
        if type(prompt) == str:
            return {"role": "user", "content": prompt}
        return prompt

    def _extract_output(self, resp) -> str:
        return resp.message.content

    def generate_batch(self, *args: Any, **kwargs: Any) -> None:
        return super().generate_batch(*args, **kwargs)

    def loglikelihood_batch(self, *args, **kwargs):
        raise NotImplementedError("No support for logits.")
