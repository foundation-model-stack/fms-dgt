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

Data builder used to transform existing datasets into instruction-following function calling data. See original repository [here](https://github.com/IBM/API-BLEND/tree/main)

## Data specification

`bash python -m spacy download en_core_web_sm `

### Raw Data:

Please download the raw data from the following links.

    - SGD: [Link](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
    - MultiWOZ: [Link](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)

The raw data should be stored as

`data/raw/dstc8-schema-guided-dialogue`
`data/raw/MultiWOZ_2.2`

### Required

- `input`: instruction to convert to function call
- `output`: function call corresponding to instruction
- `dialog_id`: id of dialog
- `speaker`: speaker within dialog
- `split`: split that data originated from

## Generators and validators

Default configuration for generators and validators used by the data builder is available [here](./transform_api.yaml).

### Generators

- `mistralai/mixtral-8x7b-instruct-v01` via `ibm-generative-ai`.

### Validators

## Evaluation

TBD
