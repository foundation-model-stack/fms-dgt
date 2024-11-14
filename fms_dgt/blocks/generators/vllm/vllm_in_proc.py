"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, List, Literal, Optional, Tuple
import copy

# Third Party
from dotenv import load_dotenv
from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.generators.llm import LMBlockData, LMGenerator
from fms_dgt.blocks.generators.utils import undistribute
from fms_dgt.utils import sdg_logger
import fms_dgt.blocks.generators.utils as generator_utils

try:
    # Third Party
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
    import ray
    import transformers
except ModuleNotFoundError:
    pass


# TODO: this can be made more efficient for our purposes by rewriting the async code ourselves
@register_block("vllm")
class vLLMGenerator(LMGenerator):
    """vLLM Generator"""

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        swap_space: int = None,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        load_dotenv()

        if not find_spec("vllm"):
            raise Exception(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install fms_dgt[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"

        if self.batch_size is None:
            self._batch_size = "auto"

        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        model_args = {
            "model": self.model_id_or_path,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": (
                int(self.max_length) if self.max_length is not None else None
            ),
            "swap_space": int(swap_space) if swap_space is not None else None,
            "quantization": quantization,
            "seed": int(self.random_seed) if self.random_seed is not None else None,
            # "distributed_executor_backend": (
            #     "ray" if self.tensor_parallel_size > 1 else "mp"
            # ),
        }
        self.model_args = {k: v for k, v in model_args.items() if v is not None}

        self._prefix_token_id = prefix_token_id

        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            assert parse_version(version("vllm")) < parse_version(
                "0.3.3"
            ), "data_parallel is only compatible with vllm < v0.3.3."
            sdg_logger.warning(
                "You might experience occasional issues with model weight downloading "
                "when data_parallel is in use. To ensure stable performance, run with "
                "data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"
            sdg_logger.info("Manual batching is not compatible with data parallelism.")

            # Third Party
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )

        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else self.model_id_or_path,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            sdg_logger.info(
                "Loglikelihood prefix token id used in evaluation: %s",
                self.prefix_token_id,
            )

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version(
                "0.3.0"
            ), "lora adapters only compatible with vllm > v0.3.0."
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def prefix_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            sampling_params = SamplingParams(stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1
            )

        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            @ray.remote
            def run_inference_one_model(
                model_args: dict, sampling_params, requests: List[List[int]]
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests, sampling_params=sampling_params
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)

        if self.lora_request is not None:
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
        else:
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )
        return outputs

    def generate_batch(
        self, requests: List[LMBlockData], disable_tqdm: bool = False
    ) -> None:
        # batch tokenize contexts
        context = [req.prompt for req in requests]
        context_encoding = self.tokenizer(context, add_special_tokens=False).input_ids
        request_list = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, requests)
        ]

        grouper = generator_utils.Grouper(request_list, lambda x: str(x[1].gen_kwargs))
        pbar = tqdm(
            total=len(request_list),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )

        for key, c_ce_reqs in grouper.get_grouped().items():

            chunks = generator_utils.chunks(
                c_ce_reqs,
                n=int(self.batch_size) if self.batch_size != "auto" else 0,
            )

            for chunk in chunks:
                context_and_encoding, chunk_instances = zip(*chunk)
                context, context_encoding = zip(*context_and_encoding)
                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk_instances)).gen_kwargs
                kwargs = self.modify_gen_kwargs(gen_kwargs)

                # add EOS token to stop sequences
                eos = self.tokenizer.decode(self.eot_token_id)
                until = kwargs.pop("stop_sequences", None)
                if not until:
                    until = [eos]
                else:
                    until.append(eos)

                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - kwargs["max_tokens"]
                context_encoding = [x[-max_ctx_len:] for x in context_encoding]

                # perform batched generation
                cont = self._model_generate(
                    requests=context_encoding,
                    generate=True,
                    stop=until,
                    **kwargs,
                )

                for output, instance in zip(cont, chunk_instances):
                    s = output.outputs[0].text
                    self.update_instance_with_result(
                        "generate_batch", s, instance, until
                    )
                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(self, requests, disable_tqdm: bool = False) -> None:
        new_reqs = []
        for req in requests:
            context, continuation = req.args
            if len(req.args) == 1:
                continuation_enc = self.tok_encode(req.args[0])
                context_enc = []
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append((context_enc, continuation_enc, req))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc_len = len(self.tok_encode(context))

            context_enc = whole_enc[:context_enc_len]
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        truncation=False,
    ):
        """ """
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding = self.tokenizer.encode(string, truncation=truncation)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[List[int], List[int], Instance]],
        disable_tqdm: bool = False,
    ) -> List[float]:
        grouper = generator_utils.Grouper(requests, lambda x: str(x[-1].gen_kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood_batch requests",
        )

        for key, c_c_reqs in grouper.get_grouped().items():

            chunks = generator_utils.chunks(
                c_c_reqs,
                n=int(self.batch_size) if self.batch_size != "auto" else 0,
            )

            for chunk in chunks:
                context_encs, continuation_encs, chunk_instances = zip(*chunk)
                inputs, ctxlens = [], []
                for context_enc, continuation_enc in zip(
                    context_encs, continuation_encs
                ):
                    inp = (context_enc + continuation_enc)[-(self.max_length) :]
                    ctxlen = len(context_enc) - max(
                        0, len(context_enc) + len(continuation_enc) - self.max_length
                    )
                    inputs.append(inp)
                    ctxlens.append(ctxlen)

                outputs = self._model_generate(requests=inputs, generate=False)

                for output, ctxlen, inp, instance in zip(
                    outputs, ctxlens, inputs, chunk_instances
                ):
                    answer = self._parse_logprobs(
                        tokens=inp,
                        outputs=output,
                        ctxlen=ctxlen,
                    )
                    self.update_instance_with_result(
                        "loglikelihood_batch", answer, instance
                    )
                    pbar.update(1)

        pbar.close()

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        continuation_logprobs_dicts = [
            (
                {
                    token: getattr(logprob, "logprob", logprob)
                    for token, logprob in logprob_dict.items()
                }
                if logprob_dict is not None
                else None
            )
            for logprob_dict in continuation_logprobs_dicts
        ]

        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        return continuation_logprobs

    def modify_gen_kwargs(self, gen_kwargs: dict) -> dict:
        # unpack our keyword arguments.
        until = None
        if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
            # start with default params then overwrite with kwargs
            kwargs = {**self._base_kwargs, **kwargs}
            if "stop_sequences" in kwargs:
                until = kwargs.get("stop_sequences")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `kwargs['stop_sequences']` to be of type Union[str,list] but got {until}"
                    )
            if "max_new_tokens" in kwargs.keys():
                kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
            if "min_new_tokens" in kwargs:
                kwargs["min_tokens"] = kwargs.pop("min_new_tokens")
            if "decoding_method" in kwargs:
                if kwargs.pop("decoding_method") == "sample":
                    kwargs["temperature"] = 0.0
            if "random_seed" in kwargs:
                kwargs["seed"] = kwargs.pop("random_seed")
            kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
            kwargs["spaces_between_special_tokens"] = kwargs.get(
                "spaces_between_special_tokens", False
            )
        else:
            raise ValueError(
                f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
            )
        return kwargs
