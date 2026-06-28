import json
import random
from pathlib import Path
from typing import Any


CONSISTENT_REASONING_DIR = Path(__file__).parent


def load_processed_train(args):
    instruction_tuned = getattr(args, "instruction_tuned", False)

    if args.testbed == "alpaca":
        with open(CONSISTENT_REASONING_DIR / "data/alpaca.json") as f:
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
I think this claim is"""
        for i in train:
            i["source"] = "alpaca"
            i["prompt"] = template.format(
                question=i["question"],
                choice=i["choice"].replace("\n", ""),
                choice_2=i["choice_2"].replace("\n", ""),
            )

        args.GROUP_SIZE = 2

    elif args.testbed == "gsm8k":
        with open(CONSISTENT_REASONING_DIR / "data/gsm8k.json") as f:
            train = json.load(f)

        if instruction_tuned:
            template = """Question: {question}
Claim: {answer}
Is this claim True or False?"""
        else:
            template = """Question: {question}
Claim: {answer}
I think this claim is"""

        for i in train:
            i["source"] = "gsm8k"
            i["prompt"] = template.format(question=i["question"], answer=i["choice"])

        args.GROUP_SIZE = 4

    elif args.testbed == "truthfulQA":
        with open(CONSISTENT_REASONING_DIR / "data/truthfulqa.json") as f:
            train = json.load(f)

        if instruction_tuned:
            template = """Question: {question}
Claim: {answer}
Is this claim True or False?"""
        else:
            template = """Question: {question}
Claim: {answer}
I think this claim is"""

        for i in train:
            i["source"] = "truthfulQA"
            i["prompt"] = template.format(question=i["question"], answer=i["choice"])
        args.GROUP_SIZE = 4

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
