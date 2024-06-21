# Standard
from typing import Any, Dict, List
import logging

# Third Party
import sqlglot

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_validator
from fms_sdg.base.validator import BaseValidator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@register_validator("sql_syntax_validator")
class SQLSyntaxValidator(BaseValidator):
    """SQL syntax validator."""

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Validate a batch.

        Args:
            inputs: list of instances.
        """
        for x in inputs:
            x.result = self._validate(*x.args, **x.kwargs)

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
