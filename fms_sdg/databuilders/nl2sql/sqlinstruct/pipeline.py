"""Pipeline for SQL data generation prompting."""

# Standard
from copy import deepcopy
from typing import Dict, List

# Local
from .models import SQLDataGenerationSchema, SQLTriplet
from .prompts import PromptFactory
from .prompts.sql_prompts import (
    SchemaAndQueryToUtterancePrompt,
    SchemaToUtteranceAndQueryPrompt,
)


class SQLDataGenerationPromptingPipeline:
    def __init__(self) -> None:
        """Initialize SQLDataGenerationPromptingPipeline."""
        self.prompt_factory = PromptFactory()

    def run(self, data_generation_schema: SQLDataGenerationSchema) -> List[Dict]:
        """Run the data generation pipeline.

        Args:
            data_generation_schema: data generation schema.

        Yields:
            instances to be used for running an LLM generator.
        """
        base_sql_triplets = []
        # NOTE: extract complete SQL triplets from ground-truth.
        if data_generation_schema.ground_truth:
            # NOTE: single examples
            base_sql_triplets = [
                SQLTriplet(
                    schema=data_generation_schema.ddl_schema,
                    utterances=[ground_truth.utterance],
                    queries=[ground_truth.query],
                )
                for ground_truth in data_generation_schema.ground_truth
            ]
            # NOTE: aggregated examples
            base_sql_triplets.append(
                SQLTriplet(
                    schema=data_generation_schema.ddl_schema,
                    utterances=[
                        ground_truth.utterance
                        for ground_truth in data_generation_schema.ground_truth
                    ],
                    queries=[
                        ground_truth.query
                        for ground_truth in data_generation_schema.ground_truth
                    ],
                )
            )

        instances: List[Dict] = []
        # NOTE: this is trivially parallel
        for prompt_method_name in self.prompt_factory.prompts.keys():
            prompt_object = self.prompt_factory.build(
                method_name=prompt_method_name,
                # NOTE: currently unused
                config_dict=dict(),
            )
            if prompt_object:
                sql_triplets = deepcopy(base_sql_triplets)
                if isinstance(prompt_object, SchemaToUtteranceAndQueryPrompt):
                    # NOTE: generation using only the schema
                    sql_triplets.append(
                        SQLTriplet(
                            schema=data_generation_schema.ddl_schema,
                            utterances=[],
                            queries=[],
                        )
                    )
                elif isinstance(prompt_object, SchemaAndQueryToUtterancePrompt):
                    # NOTE: generation from schema and query logs
                    if data_generation_schema.query_logs:
                        for query in data_generation_schema.query_logs:
                            sql_triplets.append(
                                SQLTriplet(
                                    schema=data_generation_schema.ddl_schema,
                                    utterances=[],
                                    queries=[query],
                                )
                            )
                instances.extend(
                    [
                        {
                            "prompt": prompt_object.encode_prompt(
                                sql_triplet=sql_triplet
                            ),
                            "data": sql_triplet.model_dump(by_alias=True),
                        }
                        for sql_triplet in sql_triplets
                    ]
                )
        return instances
