# Standard
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os

# Third Party
from dotenv import load_dotenv

# Local
from fms_dgt.base.resource import BaseResource


class ApiKeyResource(BaseResource):
    def __init__(self, key_name: str, call_limit: int):
        super().__init__(key_name)

        load_dotenv()
        self._key = os.getenv(key_name, None)
        assert (
            self._key is not None
        ), f"Could not find API key {key_name} in config or environment!"

        self._max_calls = call_limit
        self._max_threads = call_limit

    @property
    def key(self):
        return self._key

    @property
    def max_calls(self):
        return self._max_calls

    @property
    def max_threads(self):
        return self._max_threads
