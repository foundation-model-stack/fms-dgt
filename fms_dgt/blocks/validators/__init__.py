# Standard
from abc import abstractmethod
from typing import Any, List, Optional

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.constants import DATASET_TYPE


class BaseValidatorBlock(BaseBlock):
    def __init__(
        self,
        filter: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a block that validates (and possibly filters) its input.

        Parameters:
            filter (Optional[bool]): Whether to filter out invalid values from the list.
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._filter_invalids = filter

    def execute(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ) -> DATASET_TYPE:
        """The execute function is the primary interface to a Block. For validator blocks, the implementation differs from BaseBlock in that the result is always a boolean value indicating whether the validation succeeded or failed. In addition, the validator block can optionally filter out invalid inputs that would return False instead of writing the result to the input.

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
        outputs, to_save = [], []
        for x in inputs:
            inp_args, inp_kwargs = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            res = self._validate(*inp_args, **inp_kwargs)
            if res or not self._filter_invalids:
                self.write_result(x, res, result_field)
                outputs.append(x)
            if not res:
                iter_args = arg_fields or self._arg_fields or []
                to_save.append(
                    {
                        **dict(zip(iter_args, inp_args)),
                        **inp_kwargs,
                    }
                )

        self.save_data(to_save)

        return outputs

    @abstractmethod
    def _validate(self, *args: Any, **kwargs: Any) -> bool:
        """Derived validators must implement _validate with their core logic. This function should return True or False to reflect whether an input was valid or not"""
