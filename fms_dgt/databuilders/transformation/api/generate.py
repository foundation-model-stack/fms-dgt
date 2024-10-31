# Standard
from typing import Any, Iterable, List
import re

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.transformation.api.task import (
    ApiLlmTransformData,
    ApiLlmTransformTask,
    ApiSnipsAtisTransformData,
    ApiSnipsAtisTransformTask,
    ApiTopv2TransformData,
    ApiTopv2TransformTask,
    ApiTransformData,
)


@register_data_builder("transform_api_llm")
class ApiLlmTransformDataBuilder(TransformationDataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiLlmTransformTask = ApiLlmTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    def __call__(
        self, instruction_data: List[ApiLlmTransformData]
    ) -> Iterable[ApiTransformData]:

        api_str_list, api_str_dialog_map, dialog_info = [], {}, {}
        for d in tqdm(instruction_data, desc="API Transformation"):
            if d.output and "NONE(" not in d.output:
                api_str_list.append(d.output)
                api_str_dialog_map.setdefault(d.dialog_id, []).append(d.output)
                dialog_info[d.dialog_id] = (
                    d.split,
                    d.task_name,
                    d.seed_api_group,
                )

        api_to_str = self.generate_llm_paraphrase(api_str_list)

        outputs = []
        # reconstruct the data with llm-paraphrases
        for dialog_id, conv in api_str_dialog_map.items():
            split, task_name, seed_api_group = dialog_info[dialog_id]
            input_list, output_list, api_list, intents = [], [], [], []
            for apis in conv:
                api_list.extend(apis.split("[SEP]"))
            for api in api_list:
                intent = api[: api.index("(")]
                if intent not in intents:
                    intents.append(intent)
            api_list = [
                max([api for api in api_list if api.startswith(intent)], key=len)
                for intent in intents
            ]  # take the longest string of intents (more slots).
            for api in api_list:
                if api in api_to_str.keys():
                    api_dic = self.parse_function_call(api)
                    output_list.append(api_dic)
                    api_str = api_to_str[api]
                    api_str = api_str + "." if not api_str.endswith(".") else api_str
                    input_list.append(api_str)
            if input_list and output_list:
                outputs.append(
                    ApiTransformData(
                        **{
                            "split": split,
                            "task_name": task_name,
                            "input": " ".join(input_list),
                            "output": output_list,
                            "seed_api_group": seed_api_group,
                        }
                    )
                )
        return outputs

    def parse_function_call(self, function_call):
        pattern = re.compile(r"(\w+)\(([^)]*)\)")
        match = pattern.search(function_call)
        function_name = match.group(1)
        arguments_str = match.group(2)

        arguments = [arg.strip() for arg in arguments_str.split(";")]
        arguments_dict = {}
        for arg in arguments:
            key, value = map(str.strip, arg.split("=", 1))
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if value == "True":
                value = True
            elif value == "False":
                value = False
            arguments_dict[key] = value

        return {"name": function_name, "arguments": arguments_dict}

    def generate_llm_paraphrase(self, api_str_list):
        single_api_str_list = [api.split("[SEP]") for api in api_str_list]
        single_api_str_list = [
            item.strip() for sublist in single_api_str_list for item in sublist
        ]  # flatten
        api_to_str = {}
        prompts = []
        for api in set(single_api_str_list):
            # output_string = f"intent: {api}"
            output_string = api
            prompt = (
                f"Convert the following intent and its parameters into an imperative sentence. Do not copy the API or its parameters as is in the output sentence.\n\nInput:\n"
                + output_string
                + "\nOutput:\n"
            )
            prompts.append({"prompt": prompt, "api": api})
        outputs = self.llm1(prompts)
        for output in outputs:
            api_to_str[output["api"]] = output["result"].strip()
        return api_to_str


@register_data_builder("transform_api_snips_atis")
class ApiSnipsAtisTransformDataBuilder(TransformationDataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiSnipsAtisTransformTask = ApiSnipsAtisTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Third Party
        import spacy

        self._en = spacy.load("en_core_web_sm")

    def __call__(
        self, instruction_data: List[ApiSnipsAtisTransformData]
    ) -> Iterable[ApiTransformData]:
        outputs = []
        for data in tqdm(instruction_data, "snips_atis Transformation"):
            try:
                text = data.text
                intents = data.intents
                slots = data.slots
                task_name = data.task_name
                split = data.split
                seed_api_group = data.seed_api_group

                sentence = " ".join(text).strip()
                all_intents = intents[0].split("#")
                if len(all_intents) == 1:
                    clauses = [sentence]
                else:
                    clauses = split_string_on_delimiters(
                        sentence,
                        ["and also", "and then", ",", "and", "also"],
                        max_splits=len(all_intents),
                    )
                    if len(clauses) != len(all_intents):
                        clauses = split_string_on_delimiters(
                            sentence, [",", "and then"], max_splits=len(all_intents)
                        )
                    if len(clauses) != len(all_intents):
                        if len(clauses) != len(all_intents):
                            clauses = clause_parse(sentence, self._en)
                        if len(clauses) != len(all_intents):
                            new_clauses = []
                            for idx, c in enumerate(clauses):
                                id, clause = c
                                if len(new_clauses) == 0:
                                    new_clauses.append(clause)
                                elif new_clauses[-1].strip().endswith("and and"):
                                    new_clauses[-1] = (
                                        new_clauses[-1]
                                        .replace("and and", "and")
                                        .strip()
                                        + " "
                                        + clause
                                    )
                                else:
                                    new_clauses.append(clause)
                            clauses = new_clauses
                if len(clauses) > len(all_intents):
                    new_clauses = []
                    for c in clauses:
                        if len(new_clauses) > 0:
                            if len(new_clauses[-1].split(" ")) < 3:
                                new_clauses[-1] += " " + c
                                continue
                            elif len(c.split(" ")) < 5:
                                new_clauses[-1] += " " + c
                                continue
                        new_clauses.append(c)
                    clauses = new_clauses
                start = 0
                apis = []
                for i in range(len(clauses)):
                    if type(clauses[i]) == tuple:
                        num, clause = clauses[i]
                    else:
                        clause = clauses[i]
                    num_words = len(clause.split(" "))
                    slots_arr = slots[start : start + num_words]
                    tokens = text[start : start + num_words]
                    params = parse_IOB(tokens, slots_arr)
                    start += num_words
                    params_dic = {}
                    for val, name in params:
                        if name in params_dic:  # list
                            if type(params_dic[name]) == list:
                                params_dic[name].append(val)
                            else:  # str
                                prev_elem = params_dic[name]
                                params_dic[name] = [prev_elem]
                                params_dic[name].append(val)

                        else:
                            params_dic[name] = val
                    # params_lst = []
                    # for key, value in params_dic.items():
                    #     if len(value) == 1:
                    #         value = value[0]
                    #     if type(value) == str:
                    #         params_lst.append(f'{key} = "{value}"')
                    #     else:
                    #         params_lst.append(f'{key} = "{value}"')
                    api = {"name": all_intents[i], "arguments": params_dic}
                    # apis.append(f'{all_intents[i]}({", ".join(params_lst)})')
                    apis.append(api)

                outputs.append(
                    ApiTransformData(
                        **{
                            "task_name": task_name,
                            "split": split,
                            "input": sentence,
                            "output": apis,
                            "seed_api_group": seed_api_group,
                        }
                    )
                )
            except IndexError:
                pass


def split_string_on_delimiters(string, delimiters, max_splits=None):
    # Create a pattern for all delimiters
    pattern = "|".join(map(re.escape, delimiters))

    # Split the string using the pattern and max_splits
    return re.split(pattern, string, maxsplit=max_splits)


def clause_parse(text, en):
    doc = en(text)
    seen = set()
    chunks = []
    for sent in doc.sents:
        heads = [cc for cc in sent.root.children if cc.dep_ == "conj"]

        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = " ".join([ww.text for ww in words])
            chunks.append((head.i, chunk))

        unseen = [ww for ww in sent if ww not in seen]
        chunk = " ".join([ww.text for ww in unseen])
        chunks.append((sent.root.i, chunk))

    chunks = sorted(chunks, key=lambda x: x[0])
    return chunks


def parse_IOB(tokens, tags):
    # Third Party
    from nltk import pos_tag
    from nltk.chunk import conlltags2tree
    from nltk.tree import Tree

    # tag each token with pos
    pos_tags = [
        pos for token, pos in pos_tag(tokens)
    ]  # nltk.download('averaged_perceptron_tagger')
    # convert the BIO / IOB tags to tree
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)  # parse the tree to get our original text
    original_text = []
    for subtree in ne_tree:
        # checking for 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))
    return original_text


@register_data_builder("transform_api_topv2")
class ApiTopv2TransformDataBuilder(TransformationDataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiTopv2TransformTask = ApiTopv2TransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    def __call__(
        self,
        instruction_data: List[ApiTopv2TransformData],
    ) -> Iterable[ApiTransformData]:
        for data in tqdm(instruction_data, "Topv2 Transformation"):
            input_string = data.input_string
            ontologies = data.ontologies
            task_name = data.task_name
            split = data.split
            question = data.question
            seed_api_group = data.seed_api_group

            matches_2 = extract_slots(input_string)
            # Print the extracted slot information
            slot_info = {}
            for match in matches_2:
                parts = match.split(" ", maxsplit=1)
                slot_name = parts[0].replace("[", "").strip()
                slot_text = parts[1].replace("]", "").strip()
                if "[IN:" in slot_text:
                    pattern_intent = r"\[IN:([^]][^\s]*)"
                    matches_intent = re.findall(pattern_intent, slot_text)
                    for in_match in matches_intent:
                        slot_text = in_match
                        break
                if slot_name in slot_info:
                    slot_info[slot_name].append(slot_text)
                else:
                    slot_info[slot_name] = [slot_text]
            apis_seq = []

            for intent, slots in ontologies.items():
                if "UNSUPPORTED_" in intent:
                    continue
                api_slots = {}
                for slot in slots:
                    slt_val = slot_info["SL:" + slot][0]
                    slot_info["SL:" + slot] = slot_info["SL:" + slot][1:]
                    api_slots[slot] = slt_val
                api_slots_arr = [f'{slot} = "{val}"' for slot, val in api_slots.items()]
                # api = f'{intent}({", ".join(api_slots_arr)})'
                api = {"name": intent, "arguments": api_slots}
                apis_seq.append((api, input_string.index("IN:" + intent)))

            # Use re.search to find the pattern in the input string
            if len(apis_seq) > 0:
                ordered_seq = sorted(apis_seq, key=lambda tup: tup[1], reverse=True)
                only_apis = []
                for api in ordered_seq:
                    only_apis.append(api[0])

                yield ApiTransformData(
                    **{
                        "task_name": task_name,
                        "split": split,
                        "input": question,
                        "output": only_apis,
                        "seed_api_group": seed_api_group,
                    }
                )


def extract_slots(input_string):
    matches = []
    stack = []
    for i, char in enumerate(input_string):
        if char == "[":
            stack.append(i)
        elif char == "]":
            if stack:
                start = stack.pop()
                if input_string[start : start + 4] == "[SL:":
                    slot_info = input_string[start : i + 1]
                    matches.append(slot_info)

    return matches
