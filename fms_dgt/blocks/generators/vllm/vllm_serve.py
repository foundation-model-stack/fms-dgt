"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.util import find_spec
from typing import Any, Literal, Optional
import os
import subprocess
import uuid

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.generators.openai import OpenaiCompletionsLM
from fms_dgt.utils import sdg_logger

try:
    # Third Party
    from vllm import LLM
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
        check_interval: int = 10,
        lora_local_path: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                f"attempted to use '{self.block_type}' LM type, but package `vllm` is not installed. "
                f"Please install {self.block_type} via `pip install fms_dgt[vllm]`"
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

        pid = os.getpid()
        api_key = str(uuid.uuid4())
        host = "0.0.0.0"
        port = "9001"
        base_url = f"http://{host}:{port}/v1/"
        cmd = [
            (
                "python",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"),
            ),
            ("--pid", pid),
            ("--check-interval", check_interval),
            ("--api-key", api_key),
            ("--host", host),
            ("--port", port),
            ("--model", self.model_id_or_path),
            ("--tensor-parallel-size", tensor_parallel_size),
            ("--gpu-memory-utilization", gpu_memory_utilization),
        ]
        cmd = [str(x) for entry in cmd for x in entry]

        self._vllm_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._vllm = OpenaiCompletionsLM(api_key=api_key, base_url=base_url, **kwargs)

        while True:
            line = self._vllm_process.stdout.readline()
            if "Avg prompt throughput:" in line.decode("utf-8"):
                break
            elif self._vllm_process.poll() is not None:
                raise SystemError(f"Underlying vllm process has terminated!")

    def generate_batch(self, *args, **kwargs) -> None:
        return self._vllm.generate_batch(*args, **kwargs)

    def loglikelihood_batch(self, *args, **kwargs) -> None:
        return self._vllm.loglikelihood_batch(*args, **kwargs)

    def close(self):
        self._vllm_process.kill()
