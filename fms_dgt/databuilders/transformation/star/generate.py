# Standard
from typing import Dict, Iterable, List
import os
import shutil

# Third Party
from tqdm import tqdm, trange

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.transformation.star.task import (
    BootstrapInputData,
    BootstrapOutputData,
    BootstrapTransformTask,
)
from fms_dgt.trainers.deepspeed import DeepspeedTrainer

_PROMPT = """You are an intelligent tutoring assistant that helps students with math homework. Given a question and its answer, explain how to solve the question step-by-step to achieve the answer. When you are explaining the answer to the student, please preface your explanation with "Let's think step-by-step."

Here are some examples:

Question: 
Answer: 
Explanation: Let's think step-by-step. 
""".strip()


@register_data_builder("star_transform")
class StarTransformDataBuilder(TransformationDataBuilder):

    TASK_TYPE: BootstrapTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # we are intentionally generic with val1 to maximize reuse
    val1: BaseValidatorBlock

    def __init__(
        self, task_kwargs: Dict, max_iters: int = 2, trainer_cfg: dict = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._trainer_cfg = trainer_cfg
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
            task = BootstrapTransformTask(iteration=iteration, **task_kwargs)
            task.save_task()

            # initialize model
            if iteration == 0:
                model_id_or_path = self.llm1.model_id_or_path
                assert os.path.exists(model_id_or_path), f"Must use a local model!"
                if os.path.exists(task.prev_model):
                    shutil.rmtree(task.prev_model)
                shutil.copytree(model_id_or_path, task.prev_model)

            # annotation of dataset
            self._annotate(task)

            # train model
            trainer = DeepspeedTrainer(
                model_id_or_path=task.prev_model,
                output_dir=task.curr_model_dir,
                datastore=task._datastore,
                trainer_args=self._trainer_cfg,
            )
            trainer.train()
            trainer.release_model()

            # reload model with newly created
            if iteration != self._max_iters - 1:
                self.llm1.init_model(task.curr_model_dir)

    def _annotate(self, task: BootstrapTransformTask):

        # resume from annotation
        task.load_intermediate_data()

        # resume from annotation
        task.load_dataloader_state()

        # generate_start = time.time()

        # new_data: List[SdgData] = []
        # for generated_inst in self.call_with_task_list([task]):
        #     task.save_intermediate_data(generated_inst)
        #     new_data.append(generated_inst)
        #     task.save_dataloader_state()

        # generate_duration = time.time() - generate_start
        # sdg_logger.info(
        #     "Generation took %.2fs, generated %s data",
        #     generate_duration,
        #     len(task.machine_data),
        # )

        self.llm1.release_model()

    def __call__(
        self,
        input_data: List[BootstrapInputData],
    ) -> Iterable[BootstrapOutputData]:

        llm_inputs = []
        for qa_pair in tqdm(input_data, desc="Data Transformation"):
            # NOTE: since we have obtained this from huggingface, the actual answer is marked by "... #### <number>", so we'll extract that here

            new_inp = _PROMPT.replace("", qa_pair.question).replace("", qa_pair.answer)
            llm_inputs.append(
                {"prompt": new_inp, "stop_sequences": ["Question:"], "data": qa_pair}
            )

        # NOTE: unlike in the other tutorials, we have provided 'arg_fields' / 'kwarg_fields' / 'result_field' in the data builder's config, thus we do not need to specify them here
        llm_outputs = self.llm1.generate(llm_inputs)

        for output in llm_outputs:
            orig_qa: BootstrapInputData = output["data"]
            # NOTE: we don't do any validation of the generated 'thought', however, in general that would be a good idea
            thought = output["result"].strip()
            # NOTE: here we yield from the data builder so that the data is saved immediately
            yield BootstrapOutputData(
                **{
                    "task_name": orig_qa.task_name,
                    "input": orig_qa.question,
                    "output": orig_qa.answer,
                    "thought": thought,
                }
            )
