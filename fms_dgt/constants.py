# Standard
from typing import Any, Dict, Iterable, Union
import os

# Third Party
from datasets import Dataset
import pandas as pd

DATASET_ROW_TYPE = Union[Dict[str, Any], pd.Series]
DATASET_TYPE = Union[Iterable[DATASET_ROW_TYPE], pd.DataFrame, Dataset]

# these are variable names that we try to reuse across the codebase
TYPE_KEY = "type"
NAME_KEY = "name"
TASK_NAME_KEY = "task_name"

RAY_CONFIG_KEY = "ray_config"

DGT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
