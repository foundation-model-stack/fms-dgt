# Standard
from typing import Any
import logging

# Third Party
import sqlglot

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.validators import BaseValidatorBlock
from fms_dgt.blocks.validators.nl2sql.sql_execution_validator import SQLValidatorData

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@register_block("sql_syntax_validator")
class SQLSyntaxValidator(BaseValidatorBlock):
    """SQL syntax validator."""

    DATA_TYPE = SQLValidatorData

    def _validate(self, input: SQLValidatorData, **kwargs: Any) -> bool:
        """Validate a record containing information on schema, query and utterance.

        Args:
            record: record containing the generated sample information.
            sql_dialect: SQL dialect. Defaults to "postgres".
        """
        is_valid = False
        try:
            # NOTE: we only keep valid sql
            _ = sqlglot.parse(
                input.record["sql_schema"], dialect=str(input.sql_dialect)
            )
            _ = sqlglot.parse_one(
                input.record["sql_query"], dialect=str(input.sql_dialect)
            )
            is_valid = True
        except Exception:
            logger.warning(f"discarded generated record={input.record}")
        return is_valid
