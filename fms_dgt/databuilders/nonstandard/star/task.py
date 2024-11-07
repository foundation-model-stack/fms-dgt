# Standard
from typing import Dict, List

# Local
from fms_dgt.base.task import SdgTask
from fms_dgt.constants import DATABUILDER_KEY, TASK_NAME_KEY


class StarTask(SdgTask):
    def __init__(
        self,
        *args,
        data_formatter_templates: List[Dict],
        tasks: List[Dict],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        for task in tasks:
            if task[TASK_NAME_KEY] not in data_formatter_templates:
                raise ValueError(
                    f"Task [{task[TASK_NAME_KEY]}] must have a data_formatter_template that maps its fields to input/output pairs"
                )
            formatter_keys = list(data_formatter_templates[task[TASK_NAME_KEY]])
            if set(formatter_keys) != set(["input", "output"]):
                raise ValueError(
                    f"Task [{task[TASK_NAME_KEY]}] data_formatter_template must map to input/output pairs, instead maps to {'/'.join(formatter_keys)}"
                )

        databuilders = list(set(task[DATABUILDER_KEY] for task in tasks))
        assert (
            len(databuilders) == 1
        ), f"Cannot have multiple tasks requiring different databuilders for STaR task"

        self._data_formatter_templates = data_formatter_templates
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

    @property
    def data_formatter_templates(self) -> Dict:
        """Returns data formatter templates for all tasks

        Returns:
            Dict: Dictionary of data formatter templates
        """
        return self._data_formatter_templates
