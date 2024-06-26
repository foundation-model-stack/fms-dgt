# Third-party
# Third Party
import pytest

# Local
from fms_sdg.databuilders.nl2sql.sqlinstruct.models import SQLTriplet
from fms_sdg.databuilders.nl2sql.sqlinstruct.prompts.prompt_template import (
    PromptTemplate,
)
from fms_sdg.databuilders.nl2sql.sqlinstruct.prompts.sql_prompts import (
    SchemaAndQueryToUtterancePrompt,
    SchemaAndUtteranceToQueryPrompt,
    SchemaToUtteranceAndQueryPrompt,
    SQLPrompt,
)

SQL_TRIPLET = SQLTriplet(
    schema="CREATE TABLE PRODUCTS;",
    utterances=["List products", "List items"],
    queries=["SELECT * from PRODUCTS", "SELECT * from ITEMS"],
)


def test_prompt_template():
    prompt_template = PromptTemplate(
        input_variables=["a", "b"], template="A: {a}, B: {b}"
    )
    assert prompt_template.format(a=1, b=2) == "A: 1, B: 2"
    with pytest.raises(KeyError):
        prompt_template.format(c=2)


def test_sql_prompt_examples_rendering():
    rendered_examples = SQLPrompt.render_examples(
        SQL_TRIPLET.utterances, SQL_TRIPLET.queries
    )
    assert all(
        element in rendered_examples
        for element in (SQL_TRIPLET.utterances + SQL_TRIPLET.queries)
    )


def test_schema_to_utterance_and_query_prompt():
    prompt = SchemaToUtteranceAndQueryPrompt()
    encoded_prompt = prompt.encode_prompt(SQL_TRIPLET)
    assert isinstance(encoded_prompt, str)


def test_schema_and_query_to_utterance_prompt():
    prompt = SchemaAndQueryToUtterancePrompt()
    encoded_prompt = prompt.encode_prompt(SQL_TRIPLET)
    assert isinstance(encoded_prompt, str)


def test_schema_and_utterance_to_query_prompt():
    prompt = SchemaAndUtteranceToQueryPrompt()
    encoded_prompt = prompt.encode_prompt(SQL_TRIPLET)
    assert isinstance(encoded_prompt, str)
