import os
import tiktoken
from consistent_reasoning.dataloaders import get_root_directory, get_default_results_directory

ROOT_DIR = get_root_directory()
DEFAULT_RESULTS_DIR = get_default_results_directory()


class QueryConfig:
    def __init__(
        self,
        experiment_name,
        model_to_test,
        dataloader_fn,
        data_location,
        data,
        prompt_fn,
        parse_fn=None,
        use_cache=False,
        num_problems=None,
        max_tokens=4096,
        results_dir=None,
        temperature=None,
        logprobs=None,
        bon=1,
    ):
        assert isinstance(model_to_test, str)
        self.experiment_name = experiment_name
        self.model_to_test = model_to_test
        self.dataloader_fn = dataloader_fn
        self.data_location = data_location
        self.data = data
        self.prompt_fn = prompt_fn
        self.use_cache = use_cache
        self.parse_fn = parse_fn
        self.num_problems = num_problems
        self.max_tokens = max_tokens
        self.results_dir = results_dir
        self.temperature = temperature if temperature is not None else 0.0
        self.logprobs = logprobs
        self.bon = bon

    def get_data(self):
        assert self.data is not None
        return self.data

    def __str__(self):
        return (
            f"QueryConfig("
            f"experiment_name={self.experiment_name}, "
            f"model_to_test={self.model_to_test}, "
            f"dataloader_fn={self.dataloader_fn}, "
            f"data_location={self.data_location}, "
            f"data={self.data}, "
            f"prompt_fn={self.prompt_fn}, "
            f"use_cache={self.use_cache}, "
            f"num_problems={self.num_problems}, "
            f"max_tokens={self.max_tokens}, "
            f"results_dir={self.results_dir}, "
            f"temperature={self.temperature}, "
            f"logprobs={self.logprobs}"
        )

    def __repr__(self):
        return self.__str__()


class QueryConfigBuilder:
    def __init__(self):
        self.experiment_name = None
        self.model_to_test = None
        self.dataloader_fn = None
        self.data_location = None
        self.data = None
        self.prompt_fn = None
        self.parse_fn = None
        self.use_cache = False
        self.num_problems = None
        self.max_tokens = 4096
        self.results_dir = None
        self.temperature = 0.0
        self.logprobs = None
        self.bon = 1

    def with_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name
        return self
 
    def with_bon(self, bon):
        self.bon = bon
        return self

    def with_model_to_test(self, model_to_test):
        assert isinstance(model_to_test, str)
        self.model_to_test = model_to_test
        return self

    def with_dataloader_fn(self, dataloader_fn):
        self.dataloader_fn = dataloader_fn
        return self

    def with_data_location(self, data_location):
        self.data_location = data_location
        return self

    def with_data(self, data):
        self.data = data
        return self

    def with_prompt_fn(self, prompt_fn):
        self.prompt_fn = prompt_fn
        return self

    def with_parse_fn(self, parse_fn):
        self.parse_fn = parse_fn
        return self

    def with_use_cache(self, use_cache):
        self.use_cache = use_cache
        return self

    def with_num_problems(self, num_problems):
        self.num_problems = num_problems
        return self

    def with_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        return self

    def with_results_dir(self, results_dir):
        self.results_dir = results_dir
        return self

    def with_temperature(self, temperature):
        self.temperature = temperature
        return self

    def with_logprobs(self, logprobs):
        self.logprobs = logprobs
        return self

    def build(self):
        assert self.experiment_name is not None, "Experiment name must be set"
        assert self.model_to_test is not None, "Model to test must be set"
        assert self.prompt_fn is not None, "Prompt function must be set"
        assert (self.data is not None) or (
            self.dataloader_fn is not None and self.data_location is not None
        ), "Data must be set"
        assert (self.data is None) or (
            self.dataloader_fn is None and self.data_location is None
        ), "Data and dataloader_fn/data_location cannot both be set"
        return QueryConfig(
            self.experiment_name,
            self.model_to_test,
            self.dataloader_fn,
            self.data_location,
            self.data,
            self.prompt_fn,
            self.parse_fn,
            self.use_cache,
            self.num_problems,
            self.max_tokens,
            self.results_dir,
            self.temperature,
            self.logprobs,
            self.bon,
        )


def _get_prompts(problems, prompt_fn):
    prompts = {}
    for problem_id, problem in problems.items():
        prompt = prompt_fn(problem)
        if prompt.text:
            prompts[problem_id] = prompt
    return prompts


def get_save_dir(query_config):
    results_dir = (
        ROOT_DIR / query_config.results_dir
        if query_config.results_dir is not None
        else DEFAULT_RESULTS_DIR
    )
    save_dir = results_dir / query_config.experiment_name / query_config.model_to_test
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def move_data_into_metadata(data):
    for data_id, value in data.items():
        filtered_val = {
            k: v for k, v in value.items() if k not in ("metadata", "prompt", "response")
        }
        metadata = value.get("metadata", {})
        new_metadata = metadata | filtered_val
        data[data_id]["metadata"] = new_metadata


def format_response(data, model_responses_map, query_config):
    model_responses_flattened = {}
    for data_id, response in model_responses_map.items():
        if query_config.bon == 1:
            model_responses_flattened[f"{data_id}"] = data[data_id] | response[0]
            continue
        for resp_id, resp in enumerate(response):
            model_responses_flattened[f"{data_id}-{resp_id}"] = data[data_id] | resp
    return model_responses_flattened


def tokenize_logit_bias(logit_bias, model):
    tokenizer = tiktoken.encoding_for_model(model)
    tokenized_bias = {}
    for k, v in logit_bias.items():
        tokenized = tokenizer.encode(k)
        assert len(tokenized) == 1, f"Tokenized bias key {k} is not a single token"
        tokenized_bias[tokenized[0]] = v
    return tokenized_bias


async def query_model(model_api, file_sem, query_config):
    import asyncio
    data = query_config.get_data()
    prompts = _get_prompts(data, query_config.prompt_fn)
    save_dir = get_save_dir(query_config)

    move_data_into_metadata(data)

    model_requests = [
        model_api(
            query_config.model_to_test,
            prompts[data_id].text,
            max_tokens=query_config.max_tokens,
            temperature=query_config.temperature,
            n=query_config.bon,
            top_p=1.0,
            logprobs=query_config.logprobs,
            use_cache=query_config.use_cache,
            metadata=data[data_id]["metadata"],
            parse_fn=query_config.parse_fn,
            file_sem=file_sem,
            **(
                {
                    "logit_bias": tokenize_logit_bias(
                        prompts[data_id].logit_bias, query_config.model_to_test
                    )
                }
                if prompts[data_id].logit_bias is not None
                else {}
            ),
        )
        for data_id in prompts.keys()
    ]

    model_responses = await asyncio.gather(*model_requests)

    model_responses_map = {
        data_id: response for data_id, response in zip(prompts.keys(), model_responses)
    }
    for key in data.keys():
        if key not in prompts:
            model_responses_map[key] = [data[key]]

    response = format_response(data, model_responses_map, query_config)

    return response
