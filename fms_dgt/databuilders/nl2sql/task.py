# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Local
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.databuilders.simple.task import InstructLabSdgData


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

    INPUT_DATA_TYPE = SqlSdgData
    OUTPUT_DATA_TYPE = InstructLabSdgData

    def __init__(
        self,
        ddl_schema: str = "",
        database_information: str = None,
        database_type: str = "",
        ground_truth: str = None,
        query_logs: str = None,
        context: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._ddl_schema = ddl_schema
        self._db_info = database_information
        self._ground_truth = ground_truth
        self._query_logs = query_logs
        self._context = context

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
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

    def get_example(self) -> None:
        return self.instantiate_input_example(
            **dict(
                ddl_schema=self._ddl_schema,
                database_information=self._db_info,
                ground_truth=self._ground_truth,
                query_logs=self._query_logs,
                context=self._context,
            )
        )
