# Data builder title

Data builder used to ... (_describe your data builder_)

## Data specification

This data builder supports generation defining the following parameters:

### Required

- `created_by`: creator of the task.
- `task_description`: description of the task.
- `data_builder`: data builder name.
- ...

### Optional

- ...

An example can be found [here](../../data/code) (_point to an actual .yaml_).

## Generators and validators

Default configuration for generators and validators used by the data builder is available [here](./data_builder_name.yaml).

### Generators

- `mistralai/mixtral-8x7b-instruct-v01` via `ibm-generative-ai`.

### Validators

- `rouge_scorer`: validation of the generated data based on rouge.

## Evaluation

TBD
