# SQL Data Generator

Data builder for SQL data generation. It generate natural language utterances and related queries relying on various database information: schema, query logs, ground-truth samples and metadata.

## Data specification

This data builder supports generation defining the following parameters:

### Required

- `created_by`: creator of the task.
- `task_description`: description of the task.
- `data_builder`: nl2sql.
- `database_type`: type of the database (currently supported: postgres, sqlite)
- `ddl_schema`: DDL statement representing the database schema (comments to inform on supported values/metadata are supported)

### Optional

- `database_information`: database metadata, for supported format refer to class `DatabaseInformation` defined [here](./sqlinstruct/models.py)
- `ground_truth`: list of ground-truth samples formatted as objects with `utterance` and matching `query` keys.
- `context`: additional context and/or instructions to define custom rules for generating the data (this will be used in the prompts as-is).
- `query_logs`: list of SQL queries represented as DDL statements (strings) extracted from query logs.

An example can be found [here](../../../data/code/sql/nl2sql/orders/qna.yaml).

## Generators and validators

Default configuration for generators and validators used by the data builder is available [here](./nl2sql.yaml).

### Generators

- `ibm/granite-8b-code-instruct` via `ibm-generative-ai`.

### Validators

- `sql_syntax_validator`: validation of the generated data based on a SQL syntax checker running on generated queries and related schemas.

## Evaluation

TBD
