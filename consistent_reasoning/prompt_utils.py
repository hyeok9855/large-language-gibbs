import math
import logging

logger = logging.getLogger(__name__)

class Prompt:
    @staticmethod
    def empty():
        return Prompt([])

    def __init__(self, text, logit_bias=None):
        self.text = text
        self.logit_bias = logit_bias


def get_judge_prompt_fewshot(example, demonstrations=None, pipeline=True):
    if demonstrations is None:
        demonstrations = list(example["demonstration"].values())
    prompt = ""
    for i in demonstrations:
        prompt += i['prompt']
        prompt += "True" if i["label"] else "False"
        prompt += "\n\n"

    prompt += example['prompt']

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt


def get_yes_no(x):
    x = x.strip().lower()
    y = "true" in x
    n = "false" in x
    if y == n:
        return None
    return y


def get_yes_no_diff_logprobs(logprobs):
    eps = 1e-5
    prob_sums = {False: eps, True: eps}
    for k, v in logprobs.items():
        o = get_yes_no(k)
        if o is None:
            continue
        prob_sums[o] += math.exp(v)

    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])


def extract_claim_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response["score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(f"Problem {response['metadata']['uid']}: Error extracting judgment: {repr(e)}")
        response["score"] = 0
    return response


def extract_decision_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response["score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(f"Problem {response['metadata']['uid']}: Error extracting decision: {repr(e)}")
        response["score"] = 0
    return response


def _strip_one_trailing_space(prompt_or_text):
    if isinstance(prompt_or_text, str):
        return prompt_or_text[:-1] if prompt_or_text.endswith(" ") else prompt_or_text
    if isinstance(prompt_or_text, Prompt) and isinstance(prompt_or_text.text, str):
        if prompt_or_text.text.endswith(" "):
            prompt_or_text.text = prompt_or_text.text[:-1]
    return prompt_or_text


def _make_judge_prompt_creator(no_trailing_space, instruction_tuned=False, system_prompt=""):
    if instruction_tuned:
        assert not no_trailing_space

        def _wrapped_chat(example, demonstrations=None, pipeline=True):
            if demonstrations is None:
                demonstrations = list(example["demonstration"].values())
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for demo in demonstrations:
                messages.append({"role": "user", "content": demo["prompt"]})
                messages.append(
                    {"role": "assistant", "content": "True" if demo["label"] else "False"}
                )
            messages.append({"role": "user", "content": example["prompt"]})

            if pipeline:
                return Prompt(messages)
            else:
                return messages

        return _wrapped_chat

    if not no_trailing_space:
        return get_judge_prompt_fewshot

    def _wrapped(example, demonstrations=None, pipeline=True):
        return _strip_one_trailing_space(
            get_judge_prompt_fewshot(example, demonstrations, pipeline=pipeline)
        )

    return _wrapped
