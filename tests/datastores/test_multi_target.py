# Standard
import os
import shutil

# Local
from fms_dgt.datastores.multi import MultiTargetDatastore


class TestMultiTarget:
    def test_multi_target(self):
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp_cache"
        )

        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        primary_cfg = {
            "type": "default",
            "output_dir": cache_path,
            "store_name": "primary",
        }
        additional_cfgs = [
            {
                "type": "default",
                "output_dir": cache_path,
                "store_name": "secondary_1",
            },
            {
                "type": "default",
                "output_dir": cache_path,
                "store_name": "secondary_2",
            },
        ]

        datastore = MultiTargetDatastore(
            type="multi_target", primary=primary_cfg, additional=additional_cfgs
        )

        saved_data = [{"a": 1, "b": 2}]
        datastore.save_data(saved_data)
        loaded_data = datastore.load_data()

        assert (
            saved_data == loaded_data
        ), f"Saved data must match loaded data for primary datastore {primary_cfg}, instead got {saved_data} and {loaded_data}"

        for ds in datastore.datastores:
            loaded_data = ds.load_data()
            assert (
                saved_data == loaded_data
            ), f"Saved data must match loaded data, instead got {saved_data} and {loaded_data}"

        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
