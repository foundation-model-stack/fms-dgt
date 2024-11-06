# Standard
from typing import Any
import os
import sys

# Third Party
from data_processing.utils import ParamsUtils
from data_processing_ray.runtime.ray import RayTransformLauncher
from doc_id_transform_ray import DocIDRayTransformRuntimeConfiguration
from fdedup_transform_ray import FdedupRayTransformConfiguration

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.postprocessors import BaseDatastoreProcessingBlock


@register_block("fuzzy_dedup")
class FuzzyDedupPostprocessing(BaseDatastoreProcessingBlock):
    """Base Class for all Postprocessors"""

    def __init__(
        self,
        num_permutations: int = 64,
        threshold: float = 0.8,
        shingles_size: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.shingles_size = shingles_size

    def doc_id(self, input_folder: str, output_folder: str, input_params: dict):

        local_conf = {
            "input_folder": input_folder,
            "output_folder": output_folder,
        }
        worker_options = {"num_cpus": 0.8}
        params = input_params | {
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            "run_locally": True,
            "runtime_worker_options": ParamsUtils.convert_to_ast(worker_options),
            "runtime_num_workers": 3,
            # doc id configuration
            "doc_id_doc_column": "output",
            "doc_id_hash_column": "hash_column",
            "doc_id_int_column": "int_id_column",
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = RayTransformLauncher(DocIDRayTransformRuntimeConfiguration())
        # launch
        launcher.launch()

    def fdedup(self, input_folder: str, output_folder: str, input_params: dict):

        local_conf = {
            "input_folder": input_folder,
            "output_folder": output_folder,
        }
        worker_options = {"num_cpus": 0.8}
        params = input_params | {
            # where to run
            "run_locally": True,
            # Data access. Only required parameters are specified
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            # Orchestration parameters
            "runtime_worker_options": ParamsUtils.convert_to_ast({"num_cpus": 0.8}),
            "runtime_num_workers": 1,
            "runtime_creation_delay": 0,
            # columns used
            "fdedup_doc_column": "output",
            "fdedup_id_column": "int_id_column",
            "fdedup_cluster_column": "cluster",
            # infrastructure
            "fdedup_bucket_cpu": 0.5,
            "fdedup_doc_cpu": 0.5,
            "fdedup_mhash_cpu": 0.5,
            "fdedup_num_doc_actors": 1,
            "fdedup_num_bucket_actors": 1,
            "fdedup_num_minhash_actors": 1,
            "fdedup_num_preprocessors": 2,
            # fuzzy parameters
            "fdedup_num_permutations": self.num_permutations,
            "fdedup_threshold": self.threshold,
            "fdedup_shingles_size": self.shingles_size,
            "fdedup_delimiters": " ",
            # Random delay between reads
            "fdedup_random_delay_limit": 5,
            # snapshotting
            "fdedup_snapshot_delay": 1,
            "fdedup_use_doc_snapshot": False,
            "fdedup_use_bucket_snapshot": False,
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = RayTransformLauncher(FdedupRayTransformConfiguration())
        # launch
        launcher.launch()

    def fdedup_embeddable(
        self,
        runtime_code_location: str = "{'github': 'github', 'commit_hash': '12345', 'path': 'path'}",
        runtime_pipeline_id: str = "pipeline_id",
        runtime_job_id: str = "job_id",
    ):
        args = {
            "runtime_code_location": runtime_code_location,
            "runtime_pipeline_id": runtime_pipeline_id,
            "runtime_job_id": runtime_job_id,
        }
        doc_id_output_folder = os.path.join(self.intermediate_dir, "doc_id")
        doc_id_task = self.doc_id(
            input_folder=self.input_dir,
            output_folder=doc_id_output_folder,
            input_params=args,
        )
        fdedup_task = self.fdedup(
            input_folder=doc_id_output_folder,
            output_folder=self.output_dir,
            input_params=args,
        )

    def _process(self, *args, **kwargs) -> None:
        self.fdedup_embeddable()
