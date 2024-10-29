# Standard
from dataclasses import asdict
from typing import Any, Iterable, List, Set, Tuple

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.nl2sql.sql_execution_validator import (
    SQLExecutionValidator,
)
from fms_dgt.blocks.validators.nl2sql.sql_syntax_validator import SQLSyntaxValidator
from fms_dgt.databuilders.generation.nl2sql.sqlinstruct.models import (
    SQLDataGenerationSchema,
    SQLTriplet,
)
from fms_dgt.databuilders.generation.nl2sql.sqlinstruct.pipeline import (
    SQLDataGenerationPromptingPipeline,
)
from fms_dgt.databuilders.generation.nl2sql.sqlinstruct.prompts import PromptFactory
from fms_dgt.databuilders.generation.nl2sql.task import SqlSdgData, SqlSdgTask
from fms_dgt.databuilders.generation.simple.task import InstructLabSdgData
from fms_dgt.utils import sdg_logger


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

    # val2 is the validator which checks SQL execution
    val2: SQLExecutionValidator

    def __call__(
        self,
        instruction_data: List[SqlSdgData],
    ) -> List[InstructLabSdgData]:

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
                "Running generation pipeline with data configuration: %s",
                data_generation_schema.model_dump_json(indent=2),
            )
            prompting_pipeline = SQLDataGenerationPromptingPipeline()
            instances = prompting_pipeline.run(
                data_generation_schema=data_generation_schema
            )
            llm_outputs = self.llm1(
                instances, arg_fields=["prompt"], result_field="output"
            )

            sdg_logger.info("Post-processing generated data...")
            # NOTE: we process outputs in form of a tuple: schema, utterance, query to easily drop duplicates
            processed_outputs: Set[Tuple[str, str, str]] = set()
            for instance in llm_outputs:
                text = instance["prompt"] + instance["output"]
                for prompt_class in PromptFactory.prompts.values():
                    if prompt_class.is_compatible(text):
                        entries = prompt_class.get_utterances_and_queries(text)
                        for entry in entries:
                            processed_outputs.add(
                                (
                                    instance["data"]["schema"],
                                    entry["utterance"],
                                    entry["query"],
                                )
                            )

            sdg_logger.info("Validating generated data...")
            instances_for_validation = [
                {
                    "record": {
                        "sql_schema": sql_schema,
                        "utterance": utterance,
                        "sql_query": sql_query,
                    },
                    "sql_dialect": str(data_generation_schema.database_type.name),
                    "data": SQLTriplet(
                        schema=sql_schema,
                        utterances=[utterance],
                        queries=[sql_query],
                    ).to_instruction(),
                }
                for sql_schema, utterance, sql_query in processed_outputs
            ]
            filtered_output = self.val1(
                instances_for_validation,
                kwarg_fields=["record", "sql_dialect"],
                result_field="output",
            )
            filtered_output = self.val2(
                filtered_output,
                kwarg_fields=["record", "sql_dialect"],
                result_field="output",
            )

            sdg_logger.info("Converting to instructions...")
            for instance in filtered_output:
                # NOTE: convert the generated instructions to a format compatible with fms_dgt.
                converted_instruction = InstructLabSdgData(
                    # NOTE: coming from the package configuration
                    task_name=instruction_data_item.taxonomy_path,
                    # NOTE: info coming from taxonomy
                    taxonomy_path=instruction_data_item.taxonomy_path,
                    task_description=instruction_data_item.task_description,
                    # NOTE: info coming from generated entries
                    instruction=instance["data"]["user"],
                    input="",
                    document=None,
                    output=instance["data"]["assistant"],
                )
                outputs.append(converted_instruction)

            discarded += len(instances_for_validation) - len(filtered_output)

        sdg_logger.info("Data generation completed.")
        return outputs

    def call_with_task_list(self, request_idx: int, tasks: List[SdgTask]) -> Iterable:
        # this data builder outputs data in a different format than the input, so only the original seed data should be used
        _ = request_idx
        data_pool = [task.get_example() for task in tasks]
        return self(data_pool)
