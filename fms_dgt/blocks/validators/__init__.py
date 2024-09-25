# Standard
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import dataclasses

# Third Party
import pandas as pd

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.datastore import DatastoreDataType
from fms_dgt.base.registry import get_datastore
from fms_dgt.blocks import TYPE_KEY

ARGS_SAVE_FIELD = "args"
KWARGS_SAVE_FIELD = "kwargs"


class BaseValidatorBlock(BaseBlock):
    def __init__(
        self,
        filter: Optional[bool] = False,
        datastore: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a block that validates (and possibly filters) its input.

        Parameters:
            filter (Optional[bool]): Whether to filter out invalid values from the list.
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._filter_invalids = filter
        # datastore params
        self._datastore = None
        if datastore is not None:
            canon_task_card = self._task_cards[0] if self._task_cards else None
            self._datastore = get_datastore(
                datastore.get(TYPE_KEY),
                **{
                    "task_card": canon_task_card,
                    "data_type": DatastoreDataType.VAL,
                    **datastore,
                },
            )

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ) -> DATASET_TYPE:
        """The generate function is the primary interface to a Block. For validator blocks, the implementation differs from BaseBlock in that the result is always a boolean value indicating whether the validation succeeded or failed. In addition, the validator block can optionally filter out invalid inputs that would return False instead of writing the result to the input.

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            arg_fields (Optional[List[str]]): Names of fields within the rows of
                the inputs that should be extracted and passed as positional
                args to the underlying implementation methods.
            kwarg_fields (Optional[List[str]]): Names of fields within the rows
                of the inputs that should be extracted and passed as keyword
                args to the underlying implementation methods.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function

        Returns:
            DATASET_TYPE: Input dataset with results added (possibly filtered to remove any invalid inputs)
        """
        outputs, filtered = [], []
        for x in inputs:
            inp_args, inp_kwargs = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            res = self._validate(*inp_args, **inp_kwargs)
            if res or not self._filter_invalids:
                self.write_result(x, res, result_field)
                outputs.append(x)
            if not res:
                filtered.append(
                    {ARGS_SAVE_FIELD: inp_args, KWARGS_SAVE_FIELD: inp_kwargs}
                )

        self.save_filtered(filtered)

        return outputs

    def save_filtered(self, filtered_data: DATASET_TYPE):
        def to_serializable(x):
            if isinstance(x, pd.Series):
                return to_serializable(x.to_dict())
            elif dataclasses.is_dataclass(x):
                return to_serializable(dataclasses.asdict(x))
            elif isinstance(x, dict):
                return {k: to_serializable(v) for k, v in x.items()}
            elif isinstance(x, (tuple, list)):
                return [to_serializable(y) for y in x]
            return x

        if filtered_data and self._datastore is not None:
            self._datastore.save_data([to_serializable(x) for x in filtered_data])

    @abstractmethod
    def _validate(self, *args: Any, **kwargs: Any) -> bool:
        """Derived validators must implement _validate with their core logic. This function should return True or False to reflect whether an input was valid or not"""
