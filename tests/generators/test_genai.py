# Standard
import os

# Third Party
from dotenv import load_dotenv

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.generators.genai import GenAIGenerator

load_dotenv()


class Test_GenAIGenerator:

    config = {
        "type": "genai",
        "temperature": 0.5,
        "max_new_tokens": 512,
        "min_new_tokens": 1,
        "model_id_or_path": "mistralai/mixtral-8x7b-instruct-v01",
    }
    LM = GenAIGenerator(name="genai", config=config)
    prompt = "Give a one-sentence answer for the following question: Where are the Beatles from?"

    def test_generate_batch(self) -> None:
        inputs = []
        args = [self.prompt]
        inputs.append(Instance(args))
        self.LM.generate_batch(inputs)
        res = inputs[0].result
        assert "Liverpool" in res
