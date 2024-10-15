# Standard
import os

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import make_model_dir

###
# Trainer itself
###


@register_block("deepspeed")
class DeepspeedTrainerBlock(BaseTrainerBlock):
    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
    ) -> str:

        model_dir = make_model_dir(output_dir)

        data_path = os.path.join(output_dir, "dataset")
        self.set_dataset(datastore, data_path)

        # _train(
        #     data_path,
        #     self._config_path,
        #     model_id_or_path,
        #     model_dir,
        # )

        final_model = os.path.join(model_dir, "last")

        # return last model
        return final_model
