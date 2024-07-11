# Standard
from typing import Any, Dict, List, Optional, Union
import logging

# Third Party
from datasets import Dataset
from pandas import DataFrame
import sqlglot

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@register_block("sql_syntax_validator")
class SQLSyntaxValidator(BaseValidatorBlock):
    """SQL syntax validator."""

    def _validate(
        self, record: Dict[str, str], sql_dialect: str = "postgres", **kwargs: Any
    ) -> bool:
        """Validate a record containing information on schema, query and utterance.

        Args:
            record: record containing the generated sample information.
            sql_dialect: SQL dialect. Defaults to "postgres".
        """
        is_valid = False
        try:
            # NOTE: we only keep valid sql
            _ = sqlglot.parse(record["sql_schema"], dialect=str(sql_dialect))
            _ = sqlglot.parse_one(record["sql_query"], dialect=str(sql_dialect))
            is_valid = True
        except Exception:
            logger.warning(f"discarded generated record={record}")
        return is_valid
