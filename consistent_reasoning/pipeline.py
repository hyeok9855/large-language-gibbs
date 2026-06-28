import asyncio
import logging
from collections import deque

from consistent_reasoning.models import ModelAPI
from consistent_reasoning.query_runner import QueryConfigBuilder, query_model

logger = logging.getLogger(__name__)


class Language:
    PYTHON = "python"


class Task:
    def __init__(self, name, func, dependencies=[]):
        self.name = name
        self.func = func
        self.index = None
        self.dependencies = dependencies
        self.dependents = []
        self.result = None
        for dep in dependencies:
            dep.dependents.append(self)

    async def execute(self, results):
        if self.result is None:
            dep_results = [results[dep.name] for dep in self.dependencies]
            if asyncio.iscoroutinefunction(self.func):
                self.result = await self.func(*dep_results, index=self.index)
            else:
                self.result = self.func(*dep_results, index=self.index)
        return self.result


class PipelineConfig:
    def __init__(
        self,
        name,
        anthropic_num_threads=2,
        openai_fraction_rate_limit=0.99,
        language=Language.PYTHON,
        num_problems=None,
        problem_ids=None,
        num_open_files=1000000,
        organization="NYU_ORG",
        print_prompt_and_response=False,
        api_base=None,
    ):
        self.name = name
        self.anthropic_num_threads = anthropic_num_threads
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        self.organization = organization
        self.print_prompt_and_response = print_prompt_and_response
        self.language = language
        self.num_problems = num_problems
        self.problem_ids = problem_ids
        self.num_open_files = num_open_files
        self.api_base = api_base
        self.play_sound = False


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.steps = []
        self.step_names = set()
        self.results = {}
        self.model_api = ModelAPI(
            self.config.anthropic_num_threads,
            self.config.openai_fraction_rate_limit,
            self.config.organization,
            self.config.print_prompt_and_response,
        )
        self.file_sem = asyncio.BoundedSemaphore(self.config.num_open_files)
        self.cost = {"red": 0, "blue": 0}

    def add_load_data_step(self, name, dataloader_fn, data_location, dependencies=[]):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        def call(*args, index):
            return dataloader_fn(
                data_location,
                num_problems=self.config.num_problems,
                problem_ids=self.config.problem_ids,
            )

        task = Task(name, call, dependencies)
        self.steps.append(task)
        return task

    def add_query_step(
        self,
        name,
        model,
        prompt_fn,
        parse_fn,
        dependencies=[],
        temperature=None,
        logprobs=None,
        team=None,
        max_tokens=4096,
        bon=1,
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        query_config_builder = (
            QueryConfigBuilder()
            .with_model_to_test(model)
            .with_prompt_fn(lambda x: prompt_fn(x))
            .with_parse_fn(lambda x: parse_fn(x))
            .with_num_problems(self.config.num_problems)
            .with_max_tokens(max_tokens)
            .with_temperature(temperature)
            .with_logprobs(logprobs)
            .with_bon(bon)
        )

        async def call(data, index):
            response_dict = await query_model(
                self.model_api,
                self.file_sem,
                query_config_builder.with_experiment_name(f"{self.config.name}/{index:02d}-{name}")
                .with_data(data)
                .build(),
            )
            self.add_cost_data(team, response_dict)
            return response_dict

        step = Task(name, call, dependencies)
        self.steps.append(step)
        return step

    def add_transformation_step(
        self,
        name,
        transformation_fn,
        dependencies=[],
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        async def call(*args, index):
            if asyncio.iscoroutinefunction(transformation_fn):
                return await transformation_fn(*args)
            return transformation_fn(*args)

        step = Task(name, call, dependencies)
        self.steps.append(step)
        return step

    def add_eval_step(
        self,
        name,
        eval_fn,
        dependencies=[],
    ):
        if name in self.step_names:
            raise ValueError(f"Step name {name} already exists")
        self.step_names.add(name)

        async def call(*args, index):
            return eval_fn(*args)

        step = Task(name, call, dependencies)
        self.steps.append(step)
        return step

    def topological_sort_tasks(self, tasks):
        in_degree = {task: len(task.dependencies) for task in tasks}

        queue = deque([task for task in tasks if in_degree[task] == 0])
        sorted_tasks = []
        task_order = {task: i for i, task in enumerate(tasks)}

        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)
            for dependent in task.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue = deque(sorted(queue, key=lambda t: task_order[t]))

        for i, task in enumerate(sorted_tasks):
            task.index = i
        return sorted_tasks

    def add_cost_data(self, team, response_dict):
        cost = sum([response["response"]["cost"] for response in response_dict.values()])
        if team is not None:
            if team not in self.cost:
                self.cost[team] = 0
            self.cost[team] += cost
            overall_team = team.split("_")[0]
            if overall_team != team:
                self.cost[overall_team] += cost

    async def run(self):
        steps = self.topological_sort_tasks(self.steps)
        for task in steps:
            logger.info(f"Starting step {task.index}: {task.name}")
            try:
                self.results[task.name] = await task.execute(self.results)
            except Exception as e:
                logger.error(f"Error in step {task.index}: {task.name}")
                logger.error(e)
                raise e
            logger.info(f"Finished step {task.index}: {task.name}")
        logger.info("Run complete!! Nice!! 🚀🚀")
        return self.results
