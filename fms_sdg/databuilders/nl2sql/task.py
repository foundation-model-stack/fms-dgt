# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Local
from fms_sdg.base.task import SdgData, SdgTask


@dataclass
class SqlSdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    taxonomy_path: str
    task_description: str
    instruction: str
    input: str
    output: str
    document: str
    ddl_schema: str
    database_information: Optional[Dict[str, Any]]
    ground_truth: Optional[List[Dict[str, str]]]
    query_logs: Optional[List[str]]
    context: Optional[str]


class SqlSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    DATA_TYPE = SqlSdgData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._seed_data = [
            dict(
                ddl_schema=kwargs.get("ddl_schema", ""),
                database_information=kwargs.get("database_information", None),
                ground_truth=kwargs.get("ground_truth", None),
                query_logs=kwargs.get("query_logs", None),
                context=kwargs.get("context", None),
            )
        ]

    def instantiate_example(self, **kwargs: Any):
        return self.DATA_TYPE(
            task_name=self.name,
            taxonomy_path=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("question", kwargs.get("instruction")),
            input=kwargs.get("context", kwargs.get("input", "")),
            output=kwargs.get("answer", kwargs.get("output")),
            document=kwargs.get("document", None),
            ddl_schema=kwargs.get("ddl_schema", ""),
            database_information=kwargs.get("database_information", None),
            ground_truth=kwargs.get("ground_truth", None),
            query_logs=kwargs.get("query_logs", None),
            context=kwargs.get("context", None),
        )
