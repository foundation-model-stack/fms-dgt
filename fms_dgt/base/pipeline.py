# Standard
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import json
import os
import time

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.registry import get_block
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.blocks.generators.llm import CachingLM, LMGenerator
from fms_dgt.utils import all_annotations, sdg_logger


@dataclass
class PipelineConfig(dict):
    # data builder naming/registry
    blocks: List[Dict] = None
    metadata: Optional[
        dict
    ] = None  # by default, not used in the code. allows for users to pass arbitrary info to data builders


TYPE_KEY = "type"


class Pipeline:
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Mapping = None,
        lm_cache: str = None,
        output_dir: str = None,
        restart_generation: bool = False,
        max_gen_requests: int = None,
        task_inits: dict = None,
        task_kwargs: dict = None,
        **kwargs: Any,
    ) -> None:
        """ """
