import os
import json
import random
from pathlib import Path
from typing import Any

def get_root_directory() -> Path:
    return Path(__file__).resolve().parent.parent

def get_default_results_directory() -> Path:
    return Path("/tmp/cache/results/")


def load_processed_train(args):
    instruction_tuned = getattr(args, "instruction_tuned", False)
    root_dir = get_root_directory()

    if args.testbed == "alpaca":
        with open(root_dir / "data/train_alpaca.json") as f:
            train = json.load(f)
        if instruction_tuned:
            template = """Human: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 is more helpful and harmless than Response 2.
Is this claim True or False?"""
        else:
            template = """Human: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 is more helpful and harmless than Response 2.
I think this claim is """
        for i in train:
            i["source"] = "alpaca"
            i["prompt"] = template.format(
                question=i["question"],
                choice=i["choice"].replace("\n", ""),
                choice_2=i["choice_2"].replace("\n", ""),
            )

        args.GROUP_SIZE = 2

    elif args.testbed == "gsm8k":
        with open(root_dir / "data/train_gsm8k.json") as f:
            train = json.load(f)

        if instruction_tuned:
            template = """Question: {question}
Claim: {answer}
Is this claim True or False?"""
        else:
            template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i["source"] = "gsm8k"
            i["prompt"] = template.format(question=i["question"], answer=i["choice"])

        args.GROUP_SIZE = 4

    elif args.testbed == "truthfulQA":
        with open(root_dir / "data/train_truthfulqa.json") as f:
            train = json.load(f)

        if instruction_tuned:
            template = """Question: {question}
Claim: {answer}
Is this claim True or False?"""
        else:
            template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i["source"] = "truthfulQA"
            i["prompt"] = template.format(question=i["question"], answer=i["choice"])
        args.GROUP_SIZE = 4

    elif args.testbed == "truthfulQA-preference":
        with open(root_dir / "data/train_truthfulqa_preference.json") as f:
            train = json.load(f)

        if instruction_tuned:
            template = """Question: {question}
Answer 1: {choice}
Answer 2: {choice_2}
Claim: Answer 1 is more truthful than Answer 2.
Is this claim True or False?"""
        else:
            template = """Question: {question}
Answer 1: {choice}
Answer 2: {choice_2}
Claim: Answer 1 is more truthful than Answer 2.
I think this claim is """

        for i in train:
            i["source"] = "truthfulQA-preference"
            i["prompt"] = template.format(
                question=i["question"], choice=i["choice"], choice_2=i["choice_2"]
            )
        args.GROUP_SIZE = 2
    else:
        raise ValueError(f"Testbed {args.testbed} not supported")

    train_map = {}
    for i in train:
        if i["consistency_id"] not in train_map:
            train_map[i["consistency_id"]] = []
        train_map[i["consistency_id"]].append(i)

    out = []
    for key in train_map:
        out += train_map[key]
    return out


def _select_group_ids(args, num_groups: int) -> list[int]:
    context_size = getattr(args, "context_size", 0)
    group_index = getattr(args, "group_index", 0)
    if context_size > 0:
        return [(group_index + offset) % num_groups for offset in range(context_size)]

    group_size = args.GROUP_SIZE
    if args.batch_size % group_size != 0:
        raise ValueError(
            f"--batch_size must be divisible by group size {group_size} "
            f"for testbed {args.testbed!r}."
        )

    num_selected_groups = args.batch_size // group_size
    if num_selected_groups > num_groups:
        raise ValueError(
            f"Requested {num_selected_groups} groups, but only {num_groups} are available."
        )
    return random.sample(range(num_groups), num_selected_groups)


def load_data(args):
    train = load_processed_train(args)
    group_size = args.GROUP_SIZE
    num_groups = len(train) // group_size
    group_ids = _select_group_ids(args, num_groups)

    selected_ids = [
        group_id * group_size + offset for group_id in group_ids for offset in range(group_size)
    ]
    return train, selected_ids


def initialize(train, fewshot_ids, args):
    demonstrations = {}
    unlabeled_ids = []
    whole_ids = []
    seed_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"]
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:
            item["label"] = None
            item["type"] = "predict"
            unlabeled_ids.append(item["uid"])
        else:
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
            seed_ids.append(item["uid"])
        demonstrations[id] = item

    return demonstrations, unlabeled_ids, whole_ids, seed_ids


def load_eval_set(eval_set_path: Path) -> dict[str, Any]:
    with eval_set_path.open() as f:
        return json.load(f)


def build_partitions(
    consistency_ids: list[int],
    n_partitions: int,
    base_seed: int,
    chunk_size_cis: int,
) -> list[dict[str, Any]]:
    canonical = sorted(consistency_ids)
    if len(canonical) % chunk_size_cis != 0:
        raise ValueError(
            f"#CIs={len(canonical)} is not divisible by chunk_size_cis={chunk_size_cis}; "
            f"please pick a divisor of {len(canonical)}."
        )

    partitions: list[dict[str, Any]] = []
    for p in range(n_partitions):
        partition_seed = base_seed + p
        rng = random.Random(partition_seed)
        shuffled = list(canonical)
        rng.shuffle(shuffled)
        chunks = [shuffled[i : i + chunk_size_cis] for i in range(0, len(shuffled), chunk_size_cis)]
        partitions.append(
            {
                "partition_index": p,
                "partition_seed": partition_seed,
                "chunks": chunks,
            }
        )
    return partitions


def build_cid_index(train: list[dict[str, Any]]) -> dict[Any, list[int]]:
    index: dict[Any, list[int]] = {}
    for i, item in enumerate(train):
        index.setdefault(item["consistency_id"], []).append(i)
    return index


def select_items_for_chunk(
    train: list[dict[str, Any]],
    cid_index: dict[Any, list[int]],
    chunk_cids: list[int],
    expected_group_size: int,
) -> list[dict[str, Any]]:
    from copy import deepcopy
    items: list[dict[str, Any]] = []
    for cid in chunk_cids:
        if cid not in cid_index:
            raise KeyError(f"consistency_id={cid} not present in source train data")
        idxs = cid_index[cid]
        if len(idxs) != expected_group_size:
            raise ValueError(
                f"consistency_id={cid} has {len(idxs)} items, "
                f"expected group_size={expected_group_size}"
            )
        for idx in idxs:
            item = deepcopy(train[idx])
            item["_source_index"] = idx
            items.append(item)
    return items


def load_assignments(path, num_problems=None, problem_ids=None):
    return path


def save_to_cache(data, name, delete_existing=False, incoming_problem_ids=None):
    dir_path = get_default_results_directory() / name

    if delete_existing and os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    os.makedirs(dir_path, exist_ok=True)
    for k, v in data.items():
        if isinstance(v, list):
            to_write = [
                {key: value for key, value in item.items() if key not in ["prompt", "response"]}
                for item in v
            ]
        else:
            to_write = {key: value for key, value in v.items() if key not in ["prompt", "response"]}
        with open(dir_path / f"{k}.json", "w") as f:
            json.dump(to_write, f, indent=4)

    if incoming_problem_ids:
        with open(dir_path / "incoming_problem_ids.json", "w") as f:
            json.dump({"problem_ids": list(incoming_problem_ids)}, f, indent=4)


def read_from_cache(name):
    dir_path = get_default_results_directory() / name
    data = {}
    incoming_problem_ids = []

    for file in dir_path.glob("*.json"):
        if file.name == "incoming_problem_ids.json":
            with file.open("r") as f:
                incoming_problem_ids = json.load(f).get("problem_ids", [])
        else:
            with file.open("r") as f:
                value = json.load(f)
                if not value.get("metadata"):
                    value["metadata"] = {k: v for k, v in value.items()}
                data[file.stem] = value

    return data, incoming_problem_ids
