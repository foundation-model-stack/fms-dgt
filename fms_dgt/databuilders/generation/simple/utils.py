# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from typing import List
import random
import re
import string

# Third Party
from jinja2 import Template

# Local
from fms_dgt.databuilders.generation.simple.task import InstructLabSdgData
from fms_dgt.utils import sdg_logger

DEFAULT_PROMPT_TEMPLATE_MERLINITE = """\
You are asked to come up with a set of 5 diverse task instructions under {{taxonomy}}{{" for the task \\"%s\\""|format(task_description)  if task_description}}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
{% if not document -%}
3. The type of instructions should not have topic diversity. The list should follow the same topic and category.
{% else -%}
3. The type of instructions should be similar to provided examples. The generated instruction and the output should be grounded in the provided document.
{% endif -%}
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
{% if not document -%}
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when an instruction asks about some general information, "what is the highest peak in the world", it is not necessary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.
{% else -%}
7. The output should be an appropriate response to the input and the instruction. Long outputs are preferable.
{% endif %}

{% if not document -%}
List of 5 tasks:
{% else -%}
Based on below document provide a list of 5 tasks:

Document:
{{document}}

Here are some examples to help you understand the type of questions that are asked for this document:
{% endif -%}
"""

DEFAULT_PROMPT_TEMPLATE_MIXTRAL = """\
<s> [INST]You are a very knowledgeable AI Assistant that will faithfully assist the user with their task. You are asked to come up with a set of 5 diverse task instructions under {{taxonomy}}{{" for the task \\"%s\\""|format(task_description)  if task_description}}. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.
Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
{% if not document -%}
3. The type of instructions should not have topic diversity. The list should follow the same topic and category.
{% else -%}
3. The type of instructions should be similar to provided examples. The generated instruction and the output should be grounded in the provided document.
{% endif -%}
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
{% if not document -%}
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when an instruction asks about some general information, "what is the highest peak in the world", it is not necessary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.
{% else -%}
7. The output should be an appropriate response to the input and the instruction. Long outputs are preferable.
{% endif %}
{% if not document -%}
List of 5 tasks:
{% else -%}
Based on below document provide a list of 5 tasks:
Document:
{{document}}
Here are some examples to help you understand the type of questions that are asked for this document:
{% endif -%}[/INST]
"""

_WORD_DENYLIST = [
    "image",
    "images",
    "graph",
    "graphs",
    "picture",
    "pictures",
    "file",
    "files",
    "map",
    "maps",
    "draw",
    "plot",
    "go to",
    "video",
    "audio",
    "music",
    "flowchart",
    "diagram",
]


def check_prompt_file(prompt_file_path, model_id_or_path):
    """Check for prompt file."""
    try:
        with open(prompt_file_path, encoding="utf=8") as file:
            prompt_template = file.read()
    except FileNotFoundError as exc:
        print(
            f"Cannot find {prompt_file_path}. Using default prompt depending on model-family."
        )
        if "merlinite" in model_id_or_path:
            prompt_template = DEFAULT_PROMPT_TEMPLATE_MERLINITE
        elif "mixtral" in model_id_or_path or "llama" in model_id_or_path:
            prompt_template = DEFAULT_PROMPT_TEMPLATE_MIXTRAL
        else:
            raise ValueError(f"Unsupported model '{model_id_or_path}': {exc}") from exc
    prompt_template = prompt_template.strip() + "\n"
    return prompt_template


def encode_prompt(prompt_instructions: List[InstructLabSdgData], prompt: str):
    """Encode multiple prompt instructions into a single string.
    If documents exist, randomly select one."""
    idx = 0
    document = None
    document_list = prompt_instructions[0].document

    if document_list:
        document = random.choice(document_list)

    prompt = Template(prompt).render(
        taxonomy=prompt_instructions[0].taxonomy_path,
        task_description=prompt_instructions[0].task_description,
        document=document,
    )

    # pylint: disable=unused-variable
    for idx, task_obj in enumerate(prompt_instructions):
        (instruction, prompt_input, prompt_output, taxonomy_path,) = (
            task_obj.instruction,
            task_obj.input,
            task_obj.output,
            task_obj.taxonomy_path,
        )
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt_input = "<noinput>" if prompt_input.lower() == "" else prompt_input
        if prompt[-1] != "\n":
            prompt += "\n"
        prompt += f"* Task {idx + 1}\n"
        prompt += f"** Instruction\n{instruction}\n"
        prompt += f"** Input\n{prompt_input}\n"
        prompt += f"** Output\n{prompt_output}\n"
    prompt += f"* Task {idx + 2}\n"
    return prompt


def writeline2file(logfile, line):
    t = datetime.now().replace(microsecond=0).isoformat()
    with open(logfile, "a", encoding="utf-8") as fp:
        fp.write(f"{t} - {line}\n")


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return [], 0
    raw_instructions = f"* Task {num_prompt_instructions + 1}\n" + response
    raw_instructions = re.split(r"\* Task \d+", raw_instructions)
    instructions = []
    discarded = 0
    for inst in raw_instructions:
        if not inst.strip():
            continue

        splitted_data = re.split(r"\*\*\s+(Instruction|Input|Output):?", inst)
        if len(splitted_data) != 7:
            sdg_logger.info(
                "Discarded instruction (didn't match expected format): %s", repr(inst)
            )
            discarded += 1
            continue
        inst = splitted_data[2].strip()
        prompt_input = splitted_data[4].strip()
        prompt_input = "" if prompt_input.lower() == "<noinput>" else prompt_input
        prompt_output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            sdg_logger.info(
                "Discarded instruction (wrong number of words): %s", repr(splitted_data)
            )
            discarded += 1
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in _WORD_DENYLIST):
            sdg_logger.info(
                "Discarded instruction (contained a word from the denylist): %s",
                repr(splitted_data),
            )
            discarded += 1
            continue
        # We found that the model tends to add "write a program" to some existing instructions
        # which lead to a lot of such instructions and it's confusing whether the model needs
        # to write a program or directly output the result, so here we filter them out.
        # NOTE: this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            sdg_logger.info(
                "Discarded instruction (began with 'Write a program'): %s",
                repr(splitted_data),
            )
            discarded += 1
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            sdg_logger.info(
                "Discarded instruction (began with punctuation): %s",
                repr(splitted_data),
            )
            discarded += 1
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            sdg_logger.info(
                "Discarded instruction(began with non-ascii): %s", repr(splitted_data)
            )
            discarded += 1
            continue
        instructions.append(
            {"instruction": inst, "input": prompt_input, "output": prompt_output}
        )
    return instructions, discarded


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)
