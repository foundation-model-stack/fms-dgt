"""SQL prompts."""
# Standard
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, List, Tuple

# Third Party
# Third-pary
import sqlglot

# Local
from ..models import SQLTriplet
from .prompt_template import PromptTemplate


def basic_seed_examples(
    schema: str, number_of_examples: int = 2
) -> Tuple[List[str], List[str]]:
    """Basic seed example based on the schema.

    Args:
        schema: DDL statement for schema.
        number_of_examples: number of examples to generate. Defaults to 2.

    Returns:
        a tuple containing a two lists of matching utterances and queries.
    """
    basic_seed_examples_count = 0
    utterances = []
    queries = []
    for parsed_component in sqlglot.parse(schema):
        if basic_seed_examples_count < number_of_examples:
            # NOTE: here we only consider create statements for tables
            if parsed_component.kind == "TABLE":
                table_name = parsed_component.this.this.name
                utterances.append(f"List first 10 entries for table {table_name}")
                queries.append(f"SELECT * FROM {table_name} LIMIT 10")
                # NOTE: we pick a random column
                a_column_name = parsed_component.this.expressions[0].name
                utterances.append(
                    f"Return the number of unique elements for column {a_column_name} in table {table_name}"
                )
                queries.append(
                    f"SELECT COUNT(DISTINCT {table_name}.{a_column_name}) FROM {table_name}"
                )
                basic_seed_examples_count += 1
        else:
            break
    return utterances[:number_of_examples], queries[:number_of_examples]


class SQLPrompt(ABC):
    """Base SQLPrompt."""

    prompt_template: PromptTemplate

    @staticmethod
    def render_examples(
        utterances: List[str], queries: List[str], contextualize_examples: bool = True
    ) -> str:
        """Render examples of utterances and queries.

        Args:
            utterances: a list of utterances.
            queries: a list of queries.
            contextualize_examples: whether to give context for the examples.
                Defaults to False.

        Returns:
            rendered examples.
        """
        examples = ""
        for index, (utterance, query) in enumerate(zip(utterances, queries)):
            if utterance.strip() and query.strip():
                if index == 0 and contextualize_examples:
                    # NOTE: introduce examples
                    examples += "\nConsidering the following examples:\n"
                if contextualize_examples:
                    examples += f"Example {index + 1}:\n"
                examples += f"utterance: {utterance}\n"
                examples += f"```sql\n{query}\n```\n"
        return examples

    @staticmethod
    @abstractmethod
    def is_compatible(text: str) -> bool:
        """Test whether the text is compatible with the prompt class.

        Args:
            text: prompt and generated text.

        Returns:
            compatibility with the prompt class.
        """

    @abstractmethod
    def encode_prompt(self, sql_triplet: SQLTriplet) -> str:
        """Encode SQLTriplet in a prompt.

        Args:
            sql_triplet: a SQL triplet.

        Returns:
            encoded prompt.
        """

    @staticmethod
    @abstractmethod
    def get_utterances_and_queries(text: str) -> List[Dict[str, str]]:
        """Parse utterances and queries from generated text.

        Args:
            text: prompt and generated text.

        Returns:
            a list of objects containing utterance and query.
        """


class SchemaToUtteranceAndQueryPrompt(SQLPrompt):
    """Prompt for schema to utterance and query task."""

    prompt_template: PromptTemplate = PromptTemplate(
        input_variables=["schema", "examples"],
        template="Given the following SQL schema:\n{schema}\nGenerate utterance and query pairs.\n{examples}utterance: ",
    )

    @staticmethod
    def is_compatible(text: str) -> bool:
        """Test whether the text is compatible with the prompt class.

        Args:
            text: prompt and generated text.

        Returns:
            compatibility with the prompt class.
        """
        return "Generate utterance and query pairs." in text

    def encode_prompt(self, sql_triplet: SQLTriplet) -> str:
        """Encode SQLTriplet in a prompt.

        Args:
            sql_triplet: a SQL triplet.

        Returns:
            encoded prompt.
        """
        schema = sql_triplet.schema_field
        utterances, queries = basic_seed_examples(schema=schema)
        examples = SQLPrompt.render_examples(
            utterances + sql_triplet.utterances,
            queries + sql_triplet.queries,
            contextualize_examples=False,
        )
        return self.prompt_template.format(schema=schema, examples=examples)

    @staticmethod
    def get_utterances_and_queries(text: str) -> List[Dict[str, str]]:
        """Parse utterances and queries from generated text.

        Args:
            text: prompt and generated text.

        Returns:
            a list of objects containing utterance and query.
        """
        entries = []
        splitted_text = text.split("Generate utterance and query pairs.")
        if len(splitted_text) == 2:
            text_to_process = splitted_text[1].strip()
            for splitted_text_to_process in text_to_process.split("```\nutterance"):
                elements = [
                    element.strip("`\n")
                    for element in splitted_text_to_process.replace("utterance: ", "")
                    .strip("`\n")
                    .split("sql")
                ]
                if len(elements) == 2:
                    entries.append({"utterance": elements[0], "query": elements[1]})
        return entries


class SchemaAndQueryToUtterancePrompt(SQLPrompt):
    """Prompt for schema and query to utterance task."""

    prompt_template: PromptTemplate = PromptTemplate(
        input_variables=["schema", "examples", "query"],
        template="Given the following SQL schema:\n{schema}\n{examples}Generate only a single utterance for query:\n```sql\n{query}\n```\nUse only natural language and no SQL code.\nutterance: ",
    )

    @staticmethod
    def is_compatible(text: str) -> bool:
        """Test whether the text is compatible with the prompt class.

        Args:
            text: prompt and generated text.

        Returns:
            compatibility with the prompt class.
        """
        return "Generate only a single utterance for query:" in text

    def encode_prompt(self, sql_triplet: SQLTriplet) -> str:
        """Encode SQLTriplet in a prompt.

        Args:
            sql_triplet: a SQL triplet.

        Returns:
            encoded prompt.
        """
        schema = sql_triplet.schema_field
        examples = ""
        utterances, queries = basic_seed_examples(schema=schema)
        if len(sql_triplet.utterances) > 1 and len(sql_triplet.queries) > 1:
            utterances.extend(sql_triplet.utterances[:-1])
            queries.extend(sql_triplet.queries[:-1])
        examples = SQLPrompt.render_examples(utterances, queries)
        return self.prompt_template.format(
            schema=schema.strip(),
            examples=examples,
            query=sql_triplet.queries[-1].strip(),
        )

    @staticmethod
    def get_utterances_and_queries(text: str) -> List[Dict[str, str]]:
        """Parse utterances and queries from generated text.

        Args:
            text: prompt and generated text.

        Returns:
            a list of objects containing utterance and query.
        """
        entries = []
        splitted_text = text.split(
            "Generate only a single utterance for query:\n```sql\n"
        )
        if len(splitted_text) == 2:
            text_to_process = splitted_text[1].strip()
            splitted_text_to_process = text_to_process.split(
                "```\nUse only natural language and no SQL code.\nutterance:"
            )
            if len(splitted_text_to_process) == 2:
                query, utterance = [
                    element.strip(" `\n") for element in splitted_text_to_process
                ]
                entries.append({"utterance": utterance, "query": query})
        return entries


class SchemaAndUtteranceToQueryPrompt(SQLPrompt):
    """Prompt for schema and utterance to query task."""

    prompt_template: PromptTemplate = PromptTemplate(
        input_variables=["schema", "examples", "utterance"],
        template="Given the following SQL schema:\n{schema}\n{examples}Generate only a single query for utterance:\n{utterance}\nFormatted as a JSON with a query key, do not generate natural language.",
    )

    @staticmethod
    def is_compatible(text: str) -> bool:
        """Test whether the text is compatible with the prompt class.

        Args:
            text: prompt and generated text.

        Returns:
            compatibility with the prompt class.
        """
        return "Generate only a single query for utterance:" in text

    def encode_prompt(self, sql_triplet: SQLTriplet) -> str:
        """Encode SQLTriplet in a prompt.

        Args:
            sql_triplet: a SQL triplet.

        Returns:
            encoded prompt.
        """
        schema = sql_triplet.schema_field
        examples = ""
        utterances, queries = basic_seed_examples(schema=schema)
        if len(sql_triplet.utterances) > 1 and len(sql_triplet.queries) > 1:
            utterances.extend(sql_triplet.utterances[:-1])
            queries.extend(sql_triplet.queries[:-1])
        examples = SQLPrompt.render_examples(utterances, queries)
        return self.prompt_template.format(
            schema=schema.strip(),
            examples=examples,
            utterance=sql_triplet.utterances[-1].strip(),
        )

    @staticmethod
    def get_utterances_and_queries(text: str) -> List[Dict[str, str]]:
        """Parse utterances and queries from generated text.

        Args:
            text: prompt and generated text.

        Returns:
            a list of objects containing utterance and query.
        """
        entries = []
        splitted_text = text.split("Generate only a single query for utterance:\n")
        if len(splitted_text) == 2:
            text_to_process = splitted_text[1].strip()
            splitted_text_to_process = text_to_process.split(
                "\nFormatted as a JSON with a query key, do not generate natural language."
            )
            if len(splitted_text_to_process) == 2:
                utterance, query_to_process = [
                    element.strip(" `\n") for element in splitted_text_to_process
                ]
                # Standard
                import json

                query = ""
                if query_to_process.startswith("json"):
                    try:
                        query_as_json = json.loads(
                            query_to_process.replace("json", "").strip()
                        )
                        query = query_as_json.get("query", query)
                    except:
                        pass
                if query:
                    entries.append({"utterance": utterance, "query": query})
        return entries
