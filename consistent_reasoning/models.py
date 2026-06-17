import os
import requests
import threading
import time
import asyncio
import logging
from typing import Any, Callable, Union, List, Dict

logger = logging.getLogger(__name__)
_thread_local = threading.local()

def setup_environment(logger_level="info"):
    level = getattr(logging, logger_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable logging from noisy libraries
    for logger_name in ["openai", "httpx", "urllib3", "requests", "matplotlib", "httpcore"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class OpenAICompatLLM:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 2,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        instruction_tuned: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.instruction_tuned = instruction_tuned

    def _session(self) -> requests.Session:
        session = getattr(_thread_local, "openai_compat_session", None)
        if session is None:
            session = requests.Session()
            _thread_local.openai_compat_session = session
        return session

    def generate(
        self,
        prompt: str,
        schema: list[str] | None = None,
        verbose: bool = False,
        max_trials: int = 10,
        history: list[dict] | None = None,
    ) -> str:
        if not isinstance(schema, list) or not all(isinstance(s, str) for s in schema):
            raise NotImplementedError(
                "OpenAICompatLLM only supports list[str] choice schemas; "
                f"got schema={schema!r}."
            )

        if self.instruction_tuned:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            if history:
                for demo in history:
                    messages.append({"role": "user", "content": demo["prompt"]})
                    messages.append(
                        {"role": "assistant", "content": "True" if demo["label"] else "False"}
                    )
            messages.append({"role": "user", "content": prompt})
            body = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "structured_outputs": {"choice": list(schema)},
            }
            url = f"{self.base_url}/chat/completions"
        else:
            full_prompt = f"{self.system_prompt}\n{prompt}" if self.system_prompt else prompt
            body = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "structured_outputs": {"choice": list(schema)},
            }
            url = f"{self.base_url}/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if verbose:
            print(f"[OpenAICompatLLM] POST {url} prompt={prompt[:80]!r}...")

        last_err: Exception | None = None
        for _ in range(max_trials):
            try:
                resp = self._session().post(url, headers=headers, json=body, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if self.instruction_tuned:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = data["choices"][0]["text"]

                if content == " Fals":
                    content = " False"
                if content == "Fals":
                    content = "False"
                if content in [" Tr", " Tru"]:
                    content = " True"
                if content in ["Tr", "Tru"]:
                    content = "True"

                if content not in schema:
                    raise ValueError(f"Response {content!r} is not in choice set {schema}")
                if verbose:
                    print(f"[OpenAICompatLLM] -> {content!r}")
                return content
            except Exception as e:
                last_err = e
                print(f"Error during generation: {e}. Retrying...")

        raise RuntimeError(
            f"Failed to generate a valid response after {max_trials} trials; "
            f"last error: {last_err!r}"
        )


class ReasoningOpenAICompatLLM(OpenAICompatLLM):
    def generate(
        self,
        prompt: str,
        schema: list[str] | dict[str, Any] | None = None,
        verbose: bool = False,
        max_trials: int = 10,
        history: list[dict] | None = None,
    ) -> str | dict[str, Any]:
        if isinstance(schema, list):
            return super().generate(prompt, schema, verbose, max_trials, history)
        
        if not isinstance(schema, dict):
            raise NotImplementedError("Only list[str] or dict schema supported.")

        if self.instruction_tuned:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            if history:
                for demo in history:
                    messages.append({"role": "user", "content": demo["prompt"]})
                    messages.append(
                        {"role": "assistant", "content": "True" if demo["label"] else "False"}
                    )
            messages.append({"role": "user", "content": prompt})
            body = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "structured_outputs": {"json": schema},
            }
            url = f"{self.base_url}/chat/completions"
        else:
            full_prompt = f"{self.system_prompt}\n{prompt}" if self.system_prompt else prompt
            body = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "structured_outputs": {"json": schema},
            }
            url = f"{self.base_url}/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if verbose:
            print(f"[ReasoningOpenAICompatLLM] POST {url} prompt={prompt[:80]!r}...")

        last_err: Exception | None = None
        for _ in range(max_trials):
            try:
                resp = self._session().post(url, headers=headers, json=body, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if self.instruction_tuned:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = data["choices"][0]["text"]
                
                if verbose:
                    print(f"[ReasoningOpenAICompatLLM] -> {content!r}")
                
                import json
                return json.loads(content)
            except Exception as e:
                last_err = e
                print(f"Error during generation: {e}. Retrying...")

        raise RuntimeError(
            f"Failed to generate a valid response after {max_trials} trials; "
            f"last error: {last_err!r}"
        )


class ModelAPI:
    def __init__(
        self,
        anthropic_num_threads: int = 2,
        openai_fraction_rate_limit: float = 0.99,
        organization: str = None,
        print_prompt_and_response: bool = False,
    ):
        self.print_prompt_and_response = print_prompt_and_response
        self.running_cost = 0.0

    def _session(self) -> requests.Session:
        session = getattr(_thread_local, "openai_compat_session", None)
        if session is None:
            session = requests.Session()
            _thread_local.openai_compat_session = session
        return session

    async def __call__(
        self,
        model_ids: Union[str, List[str]],
        prompt: Union[List[Dict[str, str]], str],
        max_tokens: int = 2000,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 50,
        num_candidates_per_completion: int = 1,
        parse_fn=None,
        use_cache: bool = True,
        file_sem=None,
        metadata=None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        model = model_ids[0] if isinstance(model_ids, list) else model_ids
        
        base_url = os.environ.get("LLAMA_API_BASE", "http://localhost:8000/v1").rstrip("/")
        api_key = os.environ.get("API_KEY", "EMPTY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        temperature = kwargs.get("temperature", 0.0)
        logprobs_val = kwargs.get("logprobs", None)
        
        if isinstance(prompt, list):
            url = f"{base_url}/chat/completions"
            body = {
                "model": model,
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": n
            }
            if logprobs_val is not None:
                body["logprobs"] = True
                body["top_logprobs"] = logprobs_val
        else:
            url = f"{base_url}/completions"
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": n
            }
            if logprobs_val is not None:
                body["logprobs"] = logprobs_val
                
        last_err = None
        response_data = None
        t_start = time.time()
        for attempt in range(max_attempts_per_api_call):
            try:
                def _post():
                    return self._session().post(url, headers=headers, json=body, timeout=120)
                resp = await asyncio.to_thread(_post)
                resp.raise_for_status()
                response_data = resp.json()
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.0 * (1.5 ** attempt))
                
        if response_data is None:
            raise RuntimeError(f"API call failed after {max_attempts_per_api_call} attempts. Last error: {last_err}")
            
        t_duration = time.time() - t_start
        
        results = []
        for choice in response_data["choices"]:
            completion_text = choice["message"]["content"] if "message" in choice else choice["text"]
            stop_reason = choice.get("finish_reason", "stop")
            
            logprobs_list = None
            if "logprobs" in choice and choice["logprobs"] is not None:
                raw_logprobs = choice["logprobs"]
                if "content" in raw_logprobs:
                    logprobs_list = []
                    for item in raw_logprobs["content"]:
                        top_logprob_dict = {}
                        if "top_logprobs" in item:
                            for top_logprob in item["top_logprobs"]:
                                top_logprob_dict[top_logprob["token"]] = top_logprob["logprob"]
                        logprobs_list.append(top_logprob_dict)
                elif "top_logprobs" in raw_logprobs:
                    logprobs_list = raw_logprobs["top_logprobs"]
            
            resp_dict = {
                "prompt": prompt,
                "metadata": metadata,
                "response": {
                    "model_id": model,
                    "completion": completion_text,
                    "stop_reason": stop_reason,
                    "duration": t_duration,
                    "api_duration": t_duration,
                    "cost": 0.0,
                    "logprobs": logprobs_list
                }
            }
            if parse_fn is not None:
                resp_dict = parse_fn(resp_dict)
            results.append(resp_dict)
            
        return results[:n]
