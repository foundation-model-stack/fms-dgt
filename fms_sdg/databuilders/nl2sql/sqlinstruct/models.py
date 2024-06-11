"""Models used for data validation."""

# Standard
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third Party
from pydantic import BaseModel, ConfigDict, Field
import yaml

PathLike = Union[Path, str]


class DatabaseElement(BaseModel):
    description: Optional[str] = Field(None, description="Description of the element.")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata containing information on values, statistics, etc."
    )


class Column(DatabaseElement):
    column_name: str = Field(..., description="Name of the column.")
    column_expanded_name: Optional[str] = Field(
        None, description="LLM-friendly expanded name of the column."
    )
    column_type: str = Field(..., description="Type of the column.")
    skip: Optional[bool] = Field(
        False, description="Whether the column should be skipped/ignored."
    )


class Table(DatabaseElement):
    columns: List[Column] = Field(..., description="Columns of the table.")
    table_name: str = Field(..., description="Name of the table.")
    table_expanded_name: Optional[str] = Field(
        None, description="LLM-friendly expanded name of the table."
    )
    schema_name: Optional[str] = Field(
        None, description="LLM-friendly expanded name of the table."
    )
    skip: Optional[bool] = Field(
        False, description="Whether the table should be skipped/ignored."
    )


class Relationship(DatabaseElement):
    from_column: str = Field(..., description="Reference to the source column.")
    to_column: str = Field(..., description="Reference to the target column.")
    # TODO: remove optional once we use actual links between multiple tables abd schemas
    from_table: Optional[str] = Field(
        None, description="Reference to the source table."
    )
    to_table: Optional[str] = Field(None, description="Reference to the target table.")
    from_schema: Optional[str] = Field(
        None, description="Reference to the source schema."
    )
    to_schema: Optional[str] = Field(
        None, description="Reference to the target schema."
    )


class DatabaseType(str, Enum):
    postgres = "postgres"
    sqlite = "sqlite"


class DatabaseInformation(DatabaseElement):
    name: Optional[str] = Field(None, description="Name of the database.")
    schema_name: Optional[str] = Field(None, description="Name of the schema.")
    tables: Optional[List[Table]] = Field(
        ..., description="List of tables in the schema."
    )
    relationships: Optional[List[Relationship]] = Field(
        ..., description="List of tables in the schema."
    )
    data_source_id: Optional[str] = Field(
        None,
        description="Data source id from FlowPilot. Relevant in the case the data source has been ingested in FlowPilot.",
    )
    # TODO: add connection string validator
    database_connection_string: Optional[str] = Field(
        None,
        description="Database connection string. Relevant in the case the data source has not been ingested in FlowPilot.",
    )

    def to_ddl(self) -> str:
        """Render database schema as a DDL statement.

        Returns:
            a DDL statement defining the schema.
        """
        ddl_statement = ""
        if self.tables:
            for table in self.tables:
                ddl_columns_component = ",\n".join(
                    [
                        f"    {column.column_name} {column.column_type}"
                        for column in table.columns
                    ]
                )
                ddl_relationships_set = set()
                if self.relationships:
                    # TODO: extend to support for different relationships types (schema and table)
                    for relationship in self.relationships:
                        if relationship.from_column.startswith(f"{table.table_name}."):
                            table_name, column_name = relationship.from_column.split(
                                "."
                            )
                            (
                                referenced_table_name,
                                referenced_column_name,
                            ) = relationship.to_column.split(".")
                            ddl_relationships_set.add(
                                f"    CONSTRAINT {table_name}_{column_name}_fkey FOREIGN KEY ({column_name}) REFERENCES {f'{self.schema_name}.' if self.schema_name else ''}{referenced_table_name} ({referenced_column_name}) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION{f' -- {relationship.description}' if relationship.description else ''}"
                            )
                ddl_relationships_component = ",\n".join(sorted(ddl_relationships_set))
                ddl_relationships_component = (
                    f",\n{ddl_relationships_component}"
                    if ddl_relationships_component
                    else ""
                )
                ddl_statement += f"CREATE TABLE {f'{self.schema_name}.' if self.schema_name else ''}{table.table_name}\n(\n{ddl_columns_component}{ddl_relationships_component}\n);\n\n"
        return ddl_statement


class GroundTruthEntry(BaseModel):
    utterance: str = Field(..., description="Utterance in natural language.")
    # TODO: add SQL validator
    query: str = Field(..., description="Query in SQL format.")


class SQLDataGenerationSchema(BaseModel):
    # NOTE: from instruct-lab/taxonomy
    task_description: str = Field(
        "SQL data generation task", description="description of the task."
    )
    # NOTE: from instruct-lab/taxonomy
    created_by: str = Field(..., description="Creator of the configuration.")
    database_type: DatabaseType = Field(
        DatabaseType.postgres, description="Type of database"
    )
    # TODO: add SQL validator based on database type
    ddl_schema: str = Field(..., description="DDL statement")
    database_information: Optional[DatabaseInformation] = Field(
        None, description="Database information used for SQL data generation"
    )
    # TODO: add SQL validator based on database type for all provided queries
    ground_truth: Optional[List[GroundTruthEntry]] = Field(
        None, description="A list of ground-truth examples."
    )
    context: Optional[str] = Field(
        None,
        description="Context for data generation. Currently it is a free-form text field directly injected in the prompts.",
    )
    # TODO: add SQL validator based on database type for all provided queries
    query_logs: Optional[List[str]] = Field(
        None, description="A list of exemplar queries extracted from the query logs."
    )
    model_config = ConfigDict(use_enum_values=True)

    @staticmethod
    def from_yaml(filepath: PathLike) -> "SQLDataGenerationSchema":
        """Parse the SQL data generation schema from a .yaml file.

        Args:
            filepath: path to the .yaml file.

        Returns:
            a SQLDataGenerationSchema object.
        """
        with open(str(filepath), "rt") as fp:
            return SQLDataGenerationSchema(**yaml.safe_load(fp))

    def model_dump_yaml(  # type:ignore
        self, indent: Optional[int] = None, **kwargs
    ) -> str:
        """Dump the model to .yaml format.

        All key-word arguments are passed to model_dump (see pydantic docs for details).

        Args:
            indent: an optional indent for the indentation spaces.
                Defaults to None, a.k.a., use yaml.dump default.

        Returns:
            a .yaml formatted string.
        """
        return str(yaml.dump(self.model_dump(**kwargs), indent=indent))


class SQLTriplet(BaseModel):
    schema_field: str = Field(..., alias="schema")
    utterances: List[str]
    queries: List[str]

    def to_instruction(self) -> Dict[str, str]:
        """Build an instruct-lab data object from a SQL triplet object.

        Raises:
            ValueError: in case utterances and queries length do not match.

        Returns:
            an instruction with user and assistant fields.
        """
        if len(self.utterances) != len(self.queries):
            raise ValueError("Utterances and queries numbers do not match!")
        utterance = self.utterances[-1]
        query = self.queries[-1]
        conversation = ""
        for turn_utterance, turn_query in zip(self.utterances[:-1], self.queries[:-1]):
            conversation += turn_utterance + "\n" + "```sql\n" + turn_query + "\n```"
        context = "Given the following SQL schema:\n" + self.schema_field
        return dict(
            user=context
            + "\n"
            + (conversation + "\n" if conversation else conversation)
            + utterance
            + "\n",
            assistant="```sql\n" + query + "\n```",
        )
