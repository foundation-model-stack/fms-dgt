"""Base prompting class."""
# Standard
from typing import List


class PromptTemplate:
    """Prompt template class."""

    def __init__(self, input_variables: List[str], template: str) -> None:
        """Instantiate a prompt template.

        Args:
            input_variables: input variables used in the template.
            template: template used for formatting.
        """
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs) -> str:
        """Format prompt given input.

        Returns:
            formatted prompt.
        """
        template_input = {
            input_variable: kwargs[input_variable]
            for input_variable in self.input_variables
        }
        return self.template.format(**template_input)
