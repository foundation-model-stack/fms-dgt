"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from typing import Any, Dict, List, Optional, Union
import abc
import hashlib
import json
import os

# Third Party
from sqlitedict import SqliteDict
from tqdm import tqdm

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.instance import Instance
from fms_dgt.utils import sdg_logger

MODEL_ID_OR_PATH = "model_id_or_path"


class LMGenerator(BaseBlock):
    """Class for LLM Generators"""

    def __init__(
        self,
        model_id_or_path: str = None,
        decoding_method: str = "sample",
        max_new_tokens: int = None,
        min_new_tokens: int = None,
        random_seed: int = None,
        stop_sequences: List[str] = None,
        temperature: float = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._rank = 0
        self.cache_hook = CacheHook(None)

        self.model_id_or_path: str = model_id_or_path
        assert (
            self.model_id_or_path is not None
        ), f"Must specify model for Generator {self.name}"

        cfg_kwargs = dict()
        for k, v in {
            "decoding_method": decoding_method,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "random_seed": random_seed,
            "stop_sequences": stop_sequences,
            "temperature": temperature,
        }.items():
            if v is not None:
                cfg_kwargs[k] = v

        self._base_kwargs = cfg_kwargs

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    def update_instance_with_result(
        self,
        method: str,
        res: Any,
        instance: Instance,
        until: Optional[List[str]] = None,
    ):
        if until is not None and type(res) == str:
            for term in until:
                if len(term) > 0:
                    res = res.split(term)[0]
        instance.result = res
        self.cache_hook.add_partial(method, instance, res)

    @abc.abstractmethod
    def generate_batch(
        self, requests: List[Instance], **kwargs: Union[str, Dict]
    ) -> None:
        pass

    @abc.abstractmethod
    def loglikelihood_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        pass

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        method: str = "generate",
        **kwargs: Any,
    ):

        # simplify generation here
        instances: List[Instance] = []
        for inp in inputs:
            inp_args, inp_kwargs = self.get_args_kwargs(inp, arg_fields, kwarg_fields)
            instances.append(Instance(args=inp_args, kwargs=inp_kwargs, data=inp))

        if method == "generate":
            self.generate_batch(
                instances,
                **kwargs,
            )
        elif method == "loglikelihood":
            self.loglikelihood_batch(
                instances,
                **kwargs,
            )
        else:
            err_str = (
                f"Unhandled method type: {method}"
                if method is not None
                else "Must set 'method' kwarg to 'generate' or 'loglikelihood'"
            )
            raise ValueError(err_str)

        outputs = []
        for inst in instances:
            self.write_result(inst.data, inst.result, result_field)
            outputs.append(inst.data)

        return outputs


### SQLite-based caching of LM responses
def hash_args(attr, request):
    dat = json.dumps([attr] + [request.args, request.kwargs])
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict: SqliteDict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm: LMGenerator, cache_db) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

        self.dbdict

    def __getattr__(self, attr):
        lm_attr = getattr(self.lm, attr)
        if not callable(lm_attr):
            return lm_attr

        def fn(requests: List[Instance]):
            res = []
            remaining_reqs = []
            warned = False
            # figure out which ones are cached and which ones are new
            sdg_logger.info(
                "Loading '%s' responses from cache '%s' where possible...",
                attr,
                self.cache_db,
            )
            for req in tqdm(requests, desc="Checking cached requests"):
                hsh = hash_args(attr, req)
                if (
                    attr == "generate_batch"
                    and req.kwargs.get("decoding_method", None) == "sample"
                ):
                    # when we are doing non-greedy generation, don't use the cache
                    # (else every "randomly sampled" generation would be identical for repeats > 1).
                    if not warned:
                        sdg_logger.warning(
                            "Arguments to lm.generate_batch() '%s' include non-deterministic "
                            "sampling. Caching will not be performed for such requests.",
                            req.kwargs,
                        )
                        warned = True
                    res.append(None)
                    remaining_reqs.append(req)
                elif hsh in self.dbdict:
                    ob = self.dbdict[hsh]
                    assert ob is not None
                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            sdg_logger.info(
                "Cached requests: %s, Requests remaining: %s",
                len(requests) - len(remaining_reqs),
                len(remaining_reqs),
            )

            # actually run the LM on the requests that do not have cached results
            getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req in remaining_reqs:
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = req.result

                # caching
                hsh = hash_args(attr, req)
                self.dbdict[hsh] = req.result
            self.dbdict.commit()

            # now we store result
            for req, req_res in zip(requests, res):
                req.result = req_res

        return fn

    def generate(
        self,
        inputs: DATASET_TYPE,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        method: str = "generate",
        **kwargs: Any,
    ) -> None:

        # simplify generation here
        instances: List[Instance] = []
        for inp in inputs:
            inp_args, inp_kwargs = self.lm.get_args_kwargs(
                inp,
                arg_fields,
                kwarg_fields,
            )
            instances.append(Instance(args=inp_args, kwargs=inp_kwargs, data=inp))

        if method == "generate":
            self.generate_batch(
                instances,
                **kwargs,
            )
        elif method == "loglikelihood":
            self.loglikelihood_batch(
                instances,
                **kwargs,
            )
        else:
            err_str = (
                f"Unhandled method type: {method}"
                if method is not None
                else "Must set 'method' kwarg to 'generate' or 'loglikelihood'"
            )
            raise ValueError(err_str)

        outputs = []
        for inst in instances:
            self.lm.write_result(inst.data, inst.result, result_field)
            outputs.append(inst.data)

        return outputs

    def get_cache_hook(self):
        return CacheHook(self)
