# Notice

Code in this directory has been taken from https://github.com/IBM/API-BLEND/, which has been made available under an Apache-2.0 license

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# API Data Transformation

Data builder used to transform existing datasets to function calling data

<!-- - Simple Function: produce a query and function name, where the query is solvable with a call to the named function
- Multiple Function: produce a query and multiple function names, where the query is solvable with calls to the named functions
- Parallel Function: produce a query and multiple calls to the same function
- Parallel Multiple Function: produce a query and multiple calls to the different function
- Yes / No Detection: produce a query determine if query can be solved by provided function
 -->

## Data specification

### Raw Data:

Please download the raw data from the following links.

    - SGD: [Link](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
    - MultiWOZ: [Link](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)

The raw data should be stored as

`data/raw/dstc8-schema-guided-dialogue`
`data/raw/MultiWOZ_2.2`

This data builder supports generation defining the following parameters:

### Required

<!-- - `created_by`: creator of the task.
- `task_description`: description of the task.
- `data_builder`: simple
- `task_instruction`: general description of function-calling task that will be fed to model for each example
- `input`: instruction to be transformed into a one or more function calls
- `output`: function calls corresponding to instruction
- `positive_functions`: the names of each function that should be generated in the output
- `seed_api_group`: which group the functions produced in the output are drawn from
- `api_specifications`: a dictionary with keys being api groups and values being dictionaries containing all functions in a group -->

### Optional

<!-- - `min_func_count`: minimum number of functions that should be produced for a new example
- `max_func_count`: maximum number of functions that should be produced for a new example

An example can be found [here](../../../data/code/apis/glaive/sequencing/parallel_multiple/qna.yaml). -->

## Generators and validators

Default configuration for generators and validators used by the data builder is available [here](./transform_api.yaml).

### Generators

- `mistralai/mixtral-8x7b-instruct-v01` via `ibm-generative-ai`.

### Validators

<!-- - `api_single_intent`: validation for multi function task
- `api_multi_intent`: validation for simple function task
- `api_non_nested_sequence`: validation for parallel single/multi function calling task
- `api_yes_no`: validation for yes/no detection task
- `rouge_scorer`: validation of the generated data based on rouge. -->

## Evaluation

TBD
