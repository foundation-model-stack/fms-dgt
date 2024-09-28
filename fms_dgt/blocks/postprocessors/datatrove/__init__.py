# Standard
import os

# Third Party
import pandas as pd
import pyarrow.parquet as pq

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.blocks.postprocessors import BasePostProcessingBlock


class BaseDatatroveFilterDedupBlock(BasePostProcessingBlock):
    """Base Class for all Postprocessors"""

    def _save_data(
        self,
        file_name: str,
        to_datastore: BaseDatastore,
        batch_size: int | None = 10000,
    ) -> None:
        for f in os.listdir(self._output_dir):
            parquet_file = pq.ParquetFile(os.path.join(self.output_dir, f))
            for batch in parquet_file.iter_batches(batch_size):
                data = batch.to_pylist()
                if data and data[0]["id"].startswith(file_name):
                    to_datastore.save_data(batch.to_pylist())
