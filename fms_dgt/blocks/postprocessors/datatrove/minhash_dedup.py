# Standard
from typing import List, Optional
import os

# Third Party
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages

# Local
from fms_dgt.blocks.postprocessors.datatrove import BaseDatatroveFilterDedupBlock


class MinHashDatatrove(BaseDatatroveFilterDedupBlock):
    """MinHash Datatrove Block"""

    def __init__(self, *args, total_tasks: int = 10, **kwargs):
        super().__init__(*args, **kwargs)

        # you can also change ngrams or the number of buckets and their size here
        self._minhash_config = MinhashConfig(
            hash_config=HashConfig(precision=64),
            num_buckets=14,
            hashes_per_bucket=8,
        )  # better precision -> fewer false positives (collisions)

        self._total_tasks = total_tasks

    def _process(self, *, arg_fields: Optional[List[str]] = None, **kwargs):

        arg_fields = arg_fields or self._arg_fields
        if len(arg_fields) != 1:
            raise ValueError(
                f"Must specify exactly one arg_field to be used as deduplication key"
            )
        text_key = arg_fields[0]

        # this is the original data that we want to deduplicate
        input_reader = ParquetReader(
            self._input_dir,
            text_key=text_key,
        )

        # stage 1 computes minhash signatures for each task (each task gets a set of files)
        stage1 = LocalPipelineExecutor(
            pipeline=[
                input_reader,
                MinhashDedupSignature(
                    output_folder=os.path.join(self._intermediate_dir, "signatures"),
                    config=self._minhash_config,
                    language=Languages.english,
                ),
            ],
            tasks=self._total_tasks,
            logging_dir=os.path.join(self._logging_dir, "signatures"),
        )

        # stage 2 finds matches between signatures in each bucket
        stage2 = LocalPipelineExecutor(
            pipeline=[
                MinhashDedupBuckets(
                    input_folder=os.path.join(self._intermediate_dir, "signatures"),
                    output_folder=os.path.join(self._intermediate_dir, "buckets"),
                    config=self._minhash_config,
                ),
            ],
            tasks=self._minhash_config.num_buckets,
            logging_dir=os.path.join(self._logging_dir, "buckets"),
            depends=stage1,
        )

        # stage 3 creates clusters of duplicates using the results from all buckets
        stage3 = LocalPipelineExecutor(
            pipeline=[
                MinhashDedupCluster(
                    input_folder=os.path.join(self._intermediate_dir, "buckets"),
                    output_folder=os.path.join(self._intermediate_dir, "remove_ids"),
                    config=self._minhash_config,
                ),
            ],
            tasks=1,
            logging_dir=os.path.join(self._logging_dir, "clusters"),
            depends=stage2,
        )

        # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
        # the data must match exactly stage 1, so number of tasks and the input source must be the same
        stage4 = LocalPipelineExecutor(
            pipeline=[
                input_reader,
                TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
                MinhashDedupFilter(
                    input_folder=os.path.join(self._intermediate_dir, "remove_ids"),
                    exclusion_writer=ParquetReader(
                        os.path.join(self._intermediate_dir, "removed"),
                    ),
                ),
                ParquetWriter(
                    output_folder=self._output_dir,
                    output_filename=self.data_filename,
                ),
            ],
            tasks=self._total_tasks,
            logging_dir=os.path.join(self._logging_dir, "filter"),
            depends=stage3,
        )

        stage4.run()
