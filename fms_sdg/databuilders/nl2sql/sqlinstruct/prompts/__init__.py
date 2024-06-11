"""Prompting utilities."""
# Standard
from typing import Any, Dict, List, Optional, Type
import logging

# Local
from .sql_prompts import (
    SchemaAndQueryToUtterancePrompt,
    SchemaAndUtteranceToQueryPrompt,
    SchemaToUtteranceAndQueryPrompt,
    SQLPrompt,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PromptFactory:
    prompts: Dict[str, Type[SQLPrompt]] = {
        SchemaToUtteranceAndQueryPrompt.__name__: SchemaToUtteranceAndQueryPrompt,
        SchemaAndQueryToUtterancePrompt.__name__: SchemaAndQueryToUtterancePrompt,
        SchemaAndUtteranceToQueryPrompt.__name__: SchemaAndUtteranceToQueryPrompt,
    }

    def __contains__(self, method_name: str) -> bool:
        """Check whether a prompt object is supported.

        Args:
            method_name: name of the method.

        Returns:
            whether the method is present.
        """
        return method_name in self.prompts

    def get_supported_methods(self) -> List[str]:
        """Get the list of supported prompts.

        Return:
            A list of string storing the names of the methods.
        """
        return sorted(self.prompts.keys())

    def build(
        self, method_name: str, config_dict: Dict[str, Any]
    ) -> Optional[SQLPrompt]:
        """Build a prompt object.

        If not available or instantiation fails it returns None.

        Args:
            method_name: name of the method.
            config_dict: configuration dictionary.

        Returns:
            a prompt object or None.
        """
        prompt = None
        if method_name in self:
            try:
                prompt = self.prompts[method_name](**config_dict)
            except Exception:
                logger.debug(
                    f"Failed to build prompt for method_name={method_name} using config_dict={config_dict}"
                )
        return prompt
