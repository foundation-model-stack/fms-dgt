# Standard
from typing import Dict, Iterable, List
import os
import shutil
import time

# Third Party
from tqdm import tqdm, trange

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import InputOutputData
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.transformation.star.task import (
    BootstrapInputData,
    BootstrapTransformTask,
)
from fms_dgt.trainers.deepspeed import DeepspeedTrainer
from fms_dgt.utils import sdg_logger

_QA_PROMPT = """You are an intelligent tutoring assistant that helps students with math homework. Given a question (indicated by "Question:"), explain how to solve the question step-by-step to achieve the answer. When you are explaining the answer to the student, please preface your explanation with "Let's think step-by-step." When you have finished your explanation, write down your answer with "Answer: "

Here are some examples:

Question: {{question}}
Explanation: Let's think step-by-step. """


@register_data_builder("star_transform")
class StarTransformDataBuilder(TransformationDataBuilder):

    TASK_TYPE: BootstrapTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # we are intentionally generic with val1 to maximize reuse
    val1: BaseValidatorBlock

    def __init__(
        self,
        task_kwargs: Dict,
        max_iters: int = 2,
        trainer_config_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "trainer_configs",
            "ds_config.json",
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._trainer_config_path = trainer_config_path
        self._task_kwargs = task_kwargs
        self._max_iters = max_iters

    def execute_tasks(self):
        """Main entry point for task execution."""

        # NOTE: here we are explicitly separating each task, i.e., we do not parallelize as we might in other databuilders
        for task_kwargs in tqdm(self._task_kwargs, desc="Running transformation tasks"):
            self.execute_single_task(task_kwargs)

        self.finalize_tasks(self._tasks)

    def execute_single_task(self, task_kwargs: BootstrapTransformTask):
        """Execute single task"""
        for iteration in trange(self._max_iters, desc="Bootstrap Iteration"):

            # initialize a fresh task
            task = BootstrapTransformTask(iteration=iteration, **task_kwargs)
            task.save_task()

            # initialize model
            if iteration == 0:
                model_id_or_path = self.llm1.model_id_or_path
                assert os.path.exists(model_id_or_path), f"Must use a local model!"
                if os.path.exists(task.prev_model):
                    shutil.rmtree(task.prev_model)
                shutil.copytree(model_id_or_path, task.prev_model)

            # annotation of dataset, resume if possible
            task.load_intermediate_data()
            task.load_dataloader_state()

            generate_start = time.time()

            new_data: List[InputOutputData] = []
            for generated_inst in self.call_with_task_list([task]):
                task.save_intermediate_data(generated_inst)
                new_data.append(generated_inst)
                task.save_dataloader_state()

            generate_duration = time.time() - generate_start
            sdg_logger.info(
                "Generation took %.2fs, generated %s data",
                generate_duration,
                len(task.machine_data),
            )

            task.save_final_data()

            # release model memory to allow for trainer
            self.llm1.release_model()

            # train model
            trainer = DeepspeedTrainer(
                model_id_or_path=task.prev_model,
                output_dir=task.curr_model_dir,
                data=task.load_final_data(),
                config_path=self._trainer_config_path,
            )
            trainer.train()

            # reload model with newly created
            if iteration != self._max_iters - 1:
                self.llm1.init_model(trainer.trained_model_path)

    def __call__(
        self,
        input_data: List[BootstrapInputData],
    ) -> Iterable[InputOutputData]:

        llm_inputs = []
        for qa_pair in tqdm(input_data, desc="Data Transformation"):
            # NOTE: since we have obtained this from huggingface, the actual answer is marked by "... #### <number>", so we'll extract that here

            new_inp = _QA_PROMPT.replace("{{question}}", qa_pair.question)
            llm_inputs.append(
                {"prompt": new_inp, "stop_sequences": ["Question:"], "data": qa_pair}
            )

        # NOTE: unlike in the other tutorials, we have provided 'arg_fields' / 'kwarg_fields' / 'result_field' in the data builder's config, thus we do not need to specify them here
        llm_outputs = self.llm1.generate(llm_inputs)

        for output in llm_outputs:
            orig_qa: BootstrapInputData = output["data"]
            # NOTE: we don't do any validation of the generated 'thought', however, in general that would be a good idea
            response = output["result"].strip()
            answer = self.correct_response(orig_qa.answer, response)
            # only save answers that are correct (indicated by them being 'not None')
            if answer is not None:
                # NOTE: here we yield from the data builder so that the data is saved immediately
                yield InputOutputData(
                    **{
                        "task_name": orig_qa.task_name,
                        "input": orig_qa.question,
                        "output": answer,
                    }
                )

    def correct_response(self, answer: str, response: str):
        # TODO: REMOVE THIS
        return answer
        ###
        if answer.strip() in response:
            return answer