# Standard
from typing import Dict, List

# Local
from fms_dgt.base.task import SdgTask
from fms_dgt.constants import DATABUILDER_KEY


class StarTask(SdgTask):
    def __init__(
        self,
        *args,
        tasks: List[Dict],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        databuilders = list(set(task[DATABUILDER_KEY] for task in tasks))
        assert (
            len(databuilders) == 1
        ), f"Cannot have multiple tasks requiring different databuilders for STaR task"
        self._task_data_builder = databuilders[0]
        self._tasks = [{**kwargs, **task} for task in tasks]

    @property
    def tasks(self) -> List[Dict]:
        """Returns all tasks to be executed

        Returns:
            List[Dict]: List of tasks
        """
        return self._tasks

    @property
    def task_data_builder(self) -> List[Dict]:
        """Returns all tasks to be executed

        Returns:
            List[Dict]: List of tasks
        """
        return self._task_data_builder
