# Standard
from dataclasses import asdict
from typing import Any, List, Set, Tuple
import time

# Local
from fms_sdg.base.databuilder import DataBuilder
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_data_builder
from fms_sdg.base.task import SdgTask
from fms_sdg.databuilders.nl2sql.sqlinstruct.models import (
    SQLDataGenerationSchema,
    SQLTriplet,
)
from fms_sdg.databuilders.nl2sql.sqlinstruct.pipeline import (
    SQLDataGenerationPromptingPipeline,
)
from fms_sdg.databuilders.nl2sql.sqlinstruct.prompts import PromptFactory
from fms_sdg.databuilders.nl2sql.task import SqlSdgData, SqlSdgTask
from fms_sdg.databuilders.simple.task import InstructLabSdgData
from fms_sdg.generators.llm import LMGenerator
from fms_sdg.utils import sdg_logger
from fms_sdg.validators.nl2sql.sql_syntax_validator import SQLSyntaxValidator


@register_data_builder("nl2sql")
class Nl2SqlDataBuilder(DataBuilder):
    """Class for InstructLab Taxonomy"""

    TASK_TYPE: SdgTask = SqlSdgTask

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    # llm1 is a code generator for the synthetic examples
    llm1: LMGenerator

    # val1 is the validator which checks SQL syntax
    val1: SQLSyntaxValidator

    def __call__(
        self,
        instruction_data: List[SqlSdgData],
    ) -> Tuple[List[InstructLabSdgData], int]:

        outputs: List[InstructLabSdgData] = []
        discarded: int = 0

        sdg_logger.info("Starting data generation...")
        for instruction_data_item in instruction_data:
            # NOTE: here we rely on the fact that all the relevant information is in the key "task_info".
            # We just need to add some "task_description" as it is redacted by data configuration loading.
            data_generation_schema_dict = asdict(instruction_data_item)
            data_generation_schema_dict[
                "task_description"
            ] = instruction_data_item.task_description
            data_generation_schema = SQLDataGenerationSchema(
                **data_generation_schema_dict
            )
            sdg_logger.info(
                f"Running generation pipeline with data configuration: {data_generation_schema.model_dump_json(indent=2)}"
            )
            prompting_pipeline = SQLDataGenerationPromptingPipeline()
            instances = prompting_pipeline.run(
                data_generation_schema=data_generation_schema
            )
            self.llm1.generate_batch(instances)

            sdg_logger.info("Post-processing generated data...")
            # NOTE: we process outputs in form of a tuple: schema, utterance, query to easily drop duplicates
            processed_outputs: Set[Tuple[str, str, str]] = set()
            for instance in instances:
                text = instance.args[0] + instance.result
                for prompt_class in PromptFactory.prompts.values():
                    if prompt_class.is_compatible(text):
                        entries = prompt_class.get_utterances_and_queries(text)
                        for entry in entries:
                            processed_outputs.add(
                                (
                                    instance.data["schema"],
                                    entry["utterance"],
                                    entry["query"],
                                )
                            )

            sdg_logger.info("Validating generated data...")
            instances_for_validation = [
                Instance(
                    kwargs={
                        "record": {
                            "sql_schema": sql_schema,
                            "utterance": utterance,
                            "sql_query": sql_query,
                        },
                        "sql_dialect": str(data_generation_schema.database_type.name),
                    },
                    data=SQLTriplet(
                        schema=sql_schema, utterances=[utterance], queries=[sql_query]
                    ).to_instruction(),
                )
                for sql_schema, utterance, sql_query in processed_outputs
            ]
            self.val1.validate_batch(inputs=instances_for_validation)

            sdg_logger.info("Converting to instructions...")
            for instance in instances_for_validation:
                # NOTE: we keep only valid ones
                if instance.result:
                    # NOTE: convert the generated instructions to a format compatible with fms_sdg.
                    converted_instruction = InstructLabSdgData(
                        # NOTE: coming from the package configuration
                        task_name=instruction_data_item.taxonomy_path,
                        # NOTE: info coming from taxonomy
                        taxonomy_path=instruction_data_item.taxonomy_path,
                        task_description=instruction_data_item.task_description,
                        # NOTE: info coming from generated entries
                        instruction=instance.data["user"],
                        input="",
                        document=None,
                        output=instance.data["assistant"],
                    )
                    outputs.append(converted_instruction)
                else:
                    discarded += 1
        sdg_logger.info("Data generation completed.")
        return outputs, discarded

    def call_with_task_list(self, request_idx: int, tasks: List[SdgTask]):
        # this data builder outputs data in a different format than the input, so only the original seed data should be used
        _ = request_idx
        data_pool = [e for task in tasks for e in task.seed_data]
        return self(data_pool)
