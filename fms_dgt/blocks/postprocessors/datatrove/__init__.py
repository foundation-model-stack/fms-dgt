# Standard
from typing import Any
import os

# Third Party
import pyarrow.parquet as pq

# Local
from fms_dgt.blocks.postprocessors import BaseDatastoreProcessingBlock
from fms_dgt.constants import DATASET_TYPE


class BaseDatatroveBlock(BaseDatastoreProcessingBlock):
    """Base Class for all Postprocessors"""

    def __init__(
        self,
        *args: Any,
        text_key: str = "text",
        id_key: str = "id",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text_key = text_key
        self._id_key = id_key

    @property
    def text_key(self):
        return self._text_key

    @property
    def id_key(self):
        return self._id_key

    def _save_data(self, batch_size: int | None = 10000) -> DATASET_TYPE:
        for f in os.listdir(self._output_dir):
            parquet_file = pq.ParquetFile(os.path.join(self.output_dir, f))
            for batch in parquet_file.iter_batches(batch_size):
                pp_data = batch.to_pylist()
                if pp_data:
                    for proc in pp_data:
                        base = proc["metadata"]
                        if self.text_key:
                            base[self.text_key] = proc["text"]
                        yield base
