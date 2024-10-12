# Standard
from typing import Any, Dict, Iterable, Union

# Third Party
from datasets import Dataset
import pandas as pd

DATASET_ROW_TYPE = Union[Dict[str, Any], pd.Series]
DATASET_TYPE = Union[Iterable[DATASET_ROW_TYPE], pd.DataFrame, Dataset]

TYPE_KEY = "type"
NAME_KEY = "name"
TASK_NAME_KEY = "task_name"
