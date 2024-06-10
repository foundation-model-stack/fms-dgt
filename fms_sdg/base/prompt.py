# Standard
from typing import Any, List
import re


class PromptTemplate:
    """Base Class for all Prompts"""

    """These are the input variables that need to be filled for the particular prompt"""
    input_variables: List[str]

    """The stop sequences needed to end generation for a particular prompt"""
    stop_sequences: List[str]

    def __init__(
        self,
        prompt_str: str = None,
        yaml_path: str = None,
        stop_sequences: List[str] = None,
    ):
        if yaml_path is not None:
            self._prompt = self.from_yaml(yaml_path)
        elif prompt_str is not None:
            self._prompt = prompt_str
        else:
            raise ValueError(f"Must specify prompt!")

        self.input_variables = re.findall("\{\{(.*?)\}\}", self._prompt)
        self.stop_sequences = stop_sequences

    @property
    def prompt(self):
        return self._prompt

    def format(self, *args: Any, **kwargs: Any) -> str:
        """Format the prompt to string with the optional variables and return"""
        string = self.prompt
        for k, v in kwargs.items():
            string = string.replace("{{" + k + "}}", v)
        return string

    def all_variables_matched(self, formatted_prompt: str):
        return all(
            [
                "{{" + inp_var + "}}" in formatted_prompt
                for inp_var in self.input_variables
            ]
        )

    def from_yaml(self, path: str, **kwargs: Any):
        """Load the corresponding yaml files and return the prompt"""
        raise NotImplementedError
