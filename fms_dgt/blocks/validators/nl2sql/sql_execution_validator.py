# Standard
from dataclasses import dataclass
from typing import Any, Dict
import logging
import sqlite3

# Third Party
import sqlglot

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.validators import BaseValidatorBlock, BaseValidatorBlockData

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(kw_only=True)
class SQLValidatorData(BaseValidatorBlockData):
    record: Dict[str, str]
    sql_dialect: str = "postgres"


@register_block("sql_execution_validator")
class SQLExecutionValidator(BaseValidatorBlock):
    """SQL execution validator."""

    DATA_TYPE: SQLValidatorData = SQLValidatorData

    def _validate(self, input: SQLValidatorData, **kwargs: Any) -> bool:
        """Validate a record containing information on schema, query and utterance.

        Args:
            record: record containing the generated sample information.
        """
        is_valid = False
        try:
            connection = sqlite3.connect(":memory:")
            # NOTE: make sure the schema is compatible with SQLite
            parsed_schema = sqlglot.parse(input.record["sql_schema"])
            sqlite_schema = ";\n".join(
                [
                    parsed_element.sql(dialect="sqlite")
                    for parsed_element in parsed_schema
                    if parsed_element
                ]
            )
            # NOTE: here we might have multiple statements to define a schema
            cursor = connection.executescript(sqlite_schema)
            # TODO: consider enabling insertion of random records in tables
            # NOTE: make sure the query is compatible with SQLite
            parsed_query = sqlglot.parse_one(input.record["sql_query"])
            if parsed_query:
                sqlite_query = parsed_query.sql(dialect="sqlite")
                cursor.execute(sqlite_query)
                # TODO: consider using the result of the fetch
                # for further validation in the future
                _ = cursor.fetchall()
                is_valid = True
        except Exception:
            logger.warning(f"discarded generated record={input.record}")
        return is_valid
