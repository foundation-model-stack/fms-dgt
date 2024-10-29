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

# Third Party
from dotenv import load_dotenv
import psutil

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
@register_block("vllm-server")
class vLLMServerGenerator(LMGenerator):
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
        swap_space: int = 0,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        check_interval: int = 10,
        lora_local_path: str = None,
        host="0.0.0.0",
        port="8001",
        pid=None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        load_dotenv()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                f"attempted to use '{self.block_type}' LM type, but package `vllm` is not installed. "
                f"Please install {self.block_type} via `pip install fms_dgt[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"

        if self.batch_size is None:
            self._batch_size = "auto"

        self._check_interval = check_interval
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._data_parallel_size = int(data_parallel_size)
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._swap_space = int(swap_space)

        self._pid = pid if pid is not None else os.getpid()
        self._api_key = str(uuid.uuid4())

        self._host = host
        self._port = port
        self._base_url = f"http://{self._host}:{self._port}/v1/"
        self._vllm = OpenaiCompletionsLM(
            api_key=self._api_key, base_url=self._base_url, **kwargs
        )

        self._vllm_process = None
        self.init_model()

        # model_args = {
        #     "model": self.model_id_or_path,
        #     "gpu_memory_utilization": float(gpu_memory_utilization),
        #     "revision": revision,
        #     "dtype": dtype,
        #     "tokenizer": tokenizer,
        #     "tokenizer_mode": tokenizer_mode,
        #     "tokenizer_revision": tokenizer_revision,
        #     "trust_remote_code": trust_remote_code,
        #     "tensor_parallel_size": int(tensor_parallel_size),
        #     "max_model_len": (
        #         int(self.max_length) if self.max_length is not None else None
        #     ),
        #     "swap_space": int(swap_space) if swap_space is not None else None,
        #     "quantization": quantization,
        #     "seed": int(self.random_seed) if self.random_seed is not None else None,
        #     # "distributed_executor_backend": (
        #     #     "ray" if self.tensor_parallel_size > 1 else "mp"
        #     # ),
        # }
        # self._model_args = {k: v for k, v in model_args.items() if v is not None}

    def generate_batch(self, *args, **kwargs) -> None:
        return self._vllm.generate_batch(*args, **kwargs)

    def loglikelihood_batch(self, *args, **kwargs) -> None:
        return self._vllm.loglikelihood_batch(*args, **kwargs)

    def init_model(self, model_id_or_path: str = None):
        model_id_or_path = (
            self.model_id_or_path if model_id_or_path is None else model_id_or_path
        )
        cmd = [
            [
                "python",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"),
            ],
            ["--pid", self._pid],
            ["--check-interval", self._check_interval],
            ["--api-key", self._api_key],
            ["--host", self._host],
            ["--port", self._port],
            ["--model", model_id_or_path],
            ["--tensor-parallel-size", self._tensor_parallel_size],
            ["--gpu-memory-utilization", self._gpu_memory_utilization],
            ["--swap-space", self._swap_space],
            ["--disable-log-requests"],
            # ["--enable-prefix-caching"],
        ]
        cmd = [str(x) for entry in cmd for x in entry]

        sdg_logger.info(f"Starting vllm server with command:\n{' '.join(cmd)}")

        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        lines = []
        while True:
            # grab lines from both sdtout and stderr
            lines.append(
                "\n".join(
                    [
                        proc.readline().decode("utf-8").strip()
                        for proc in [
                            self._vllm_process.stdout,
                            self._vllm_process.stderr,
                        ]
                    ]
                ).strip()
            )

            # check for running process
            if any(
                [
                    t_str in lines[-1]
                    for t_str in ["Uvicorn running on socket", "Avg prompt throughput:"]
                ]
            ):
                sdg_logger.info(
                    "Server has been initialized, detailed log is provided below:\n\n"
                    + "*" * 50
                    + "\n".join([l for l in lines if l])
                    + "\n\n"
                    + "*" * 50
                )
                break
            elif self._vllm_process.poll() is not None:
                # if process has error'd out, kill it
                sdg_logger.error(
                    "Error in vllm server instance. The full traceback is provided below:\n\n"
                    + "*" * 50
                    + "\n".join([l for l in lines if l])
                    + "\n\n"
                    + "*" * 50
                )
                raise SystemError(f"Underlying vllm process has terminated!")

    def release_model(self):
        sdg_logger.info(f"Releasing model by killing process {self._vllm_process.pid}")
        base_proc = psutil.Process(self._vllm_process.pid)
        for child_proc in base_proc.children(recursive=True):
            child_proc.kill()
        base_proc.kill()

    def close(self):
        self.release_model()
