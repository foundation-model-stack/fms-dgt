# Standard
import os

# Third Party
import pandas as pd

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.blocks.postprocessors import BasePostProcessingBlock


class BaseDatatroveFilterDedupBlock(BasePostProcessingBlock):
    """Base Class for all Postprocessors"""

    def _get_data(self, inputs: DATASET_TYPE):
        ids = set()
        for f in os.listdir(self._output_dir):
            if f.endswith(self.data_filename):
                data_path = os.path.join(self._output_dir, f)
                proc_data = (
                    pd.read_parquet(data_path, engine="fastparquet")
                    .apply(dict, axis=1)
                    .to_list()
                )
                rem_ids = [
                    int(d_id[-1])
                    for d_id in [d["id"].split("/") for d in proc_data]
                    if d_id[0].startswith(self.data_filename)
                ]
                ids.update(rem_ids)

        # TODO: make this more efficient
        ret_data = [inp for i, inp in enumerate(inputs) if i in ids]
        return ret_data
