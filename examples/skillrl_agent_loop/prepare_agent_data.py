import argparse
import os
import sys
from pathlib import Path

import datasets


def _add_skillrl_root(skillrl_root: str | None) -> str:
    if skillrl_root is None:
        skillrl_root = os.environ.get("SKILLRL_ROOT")
    if skillrl_root is None:
        skillrl_root = str(Path(__file__).resolve().parent)
    if skillrl_root not in sys.path:
        sys.path.insert(0, skillrl_root)
    return skillrl_root


def _count_alfworld(config_path: str, split: str, eval_dataset: str) -> int:
    import yaml

    _add_skillrl_root(None)
    alfworld_root = os.path.dirname(os.path.dirname(config_path))
    if alfworld_root not in sys.path:
        sys.path.insert(0, alfworld_root)
    from alfworld.info import ALFWORLD_DATA

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_path = os.path.expandvars(config["dataset"]["data_path"])
    if split != "train":
        if eval_dataset == "eval_out_of_distribution":
            key = "eval_ood_data_path"
        else:
            key = "eval_id_data_path"
        data_path = os.path.expandvars(config["dataset"].get(key, data_path))
    root = os.path.join(ALFWORLD_DATA, data_path)
    return sum(1 for _, _, files in os.walk(root) for name in files if name == "game.tw-pddl")


def _normalize_webshop_split(split: str) -> str:
    split = split.strip().lower()
    if split in {"val", "valid", "validation", "eval"}:
        return "valid"
    if split in {"test", "test_128", "train"}:
        return split
    raise ValueError(f"Unsupported WebShop split: {split!r}. Use train, valid/eval, test, or test_128.")


def _webshop_indices(split: str, size: int | None = None) -> list[int]:
    split = _normalize_webshop_split(split)
    if split == "test_128":
        indices = [round(i * 499 / 127) for i in range(128)]
    elif split == "test":
        indices = list(range(500))
    elif split == "valid":
        indices = list(range(500, 1500))
    elif split == "train":
        # The caller still needs _count_webshop to know the true train size.
        indices = []
    else:
        raise ValueError(f"Unsupported WebShop split: {split!r}.")

    if size is not None and size >= 0:
        indices = indices[:size]
    return indices


def _count_webshop(webshop_root: str, split: str) -> int:
    import json

    webshop_root_path = Path(webshop_root).expanduser()
    if (webshop_root_path / "data").exists():
        data_dir = webshop_root_path / "data"
    else:
        data_dir = webshop_root_path

    with open(data_dir / "items_shuffle_1000.json", encoding="utf-8") as f:
        products = json.load(f)
    with open(data_dir / "items_ins_v2_1000.json", encoding="utf-8") as f:
        attributes = json.load(f)

    goal_count = 0
    seen_asins = set()
    for item in products:
        asin = item.get("asin")
        if asin == "nan" or not asin or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        item_attributes = attributes.get(asin, {})
        if not item_attributes.get("instruction") or not item_attributes.get("instruction_attributes"):
            continue

        option_count = 1
        for option_contents in (item.get("customization_options") or {}).values():
            if option_contents:
                option_count *= len(option_contents)
        goal_count += option_count

    split = _normalize_webshop_split(split)
    if split == "test_128":
        return min(128, min(500, goal_count))
    if split == "test":
        return min(500, goal_count)
    if split == "valid":
        return max(0, min(1500, goal_count) - 500)
    return max(0, goal_count - 1500)


def _resolve_size(
    size: int,
    split: str,
    data_source: str,
    alfworld_config_path: str | None,
    alfworld_eval_dataset: str,
    webshop_root: str | None = None,
) -> int:
    if size >= 0:
        return size
    if data_source == "alfworld" and alfworld_config_path:
        return _count_alfworld(alfworld_config_path, split, alfworld_eval_dataset)
    if data_source == "webshop" and webshop_root:
        return _count_webshop(webshop_root, split)
    raise ValueError(f"{data_source} requires an explicit non-negative size for split={split}.")


def build_dataset(
    size: int,
    split: str,
    data_source: str,
    agent_name: str,
    indices: list[int] | None = None,
) -> datasets.Dataset:
    tool_selection = {
        "alfworld": ["take_action"],
        "webshop": ["search_action", "click_action"],
    }[data_source]
    features = datasets.Features(
        {
            "data_source": datasets.Value("string"),
            "agent_name": datasets.Value("string"),
            "prompt": [
                {
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ],
            "ability": datasets.Value("string"),
            "reward_model": {
                "style": datasets.Value("string"),
                "ground_truth": datasets.Value("string"),
            },
            "extra_info": {
                "split": datasets.Value("string"),
                "index": datasets.Value("int64"),
                "env_name": datasets.Value("string"),
                "tool_selection": [datasets.Value("string")],
            },
        }
    )
    rows = []
    if indices is None:
        indices = list(range(size))
    for idx in indices[:size]:
        rows.append(
            {
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "env_name": data_source,
                    "tool_selection": tool_selection,
                },
            }
        )
    if not rows:
        return datasets.Dataset.from_dict(
            {
                "data_source": [],
                "agent_name": [],
                "prompt": [],
                "ability": [],
                "reward_model": [],
                "extra_info": [],
            },
            features=features,
        )
    return datasets.Dataset.from_list(rows, features=features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--local_dir", default="/mnt/data/xts/data/skillrl-agent-loop")
    parser.add_argument("--agent_name", default="skillrl_env_agent")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=16)
    parser.add_argument("--alfworld_config_path", default=None)
    parser.add_argument("--alfworld_eval_dataset", default="eval_in_distribution")
    parser.add_argument("--webshop_root", default=None)
    parser.add_argument("--webshop_eval_dataset", default="valid")
    parser.add_argument("--skillrl_root", default=None)
    args = parser.parse_args()

    skillrl_root = _add_skillrl_root(args.skillrl_root)
    if args.webshop_root is None:
        args.webshop_root = str(
            Path(skillrl_root)
            / "agent_system"
            / "environments"
            / "env_package"
            / "webshop"
            / "webshop"
        )
    webshop_eval_dataset = _normalize_webshop_split(args.webshop_eval_dataset)
    output_dir = Path(args.local_dir).expanduser() / args.data_source
    output_dir.mkdir(parents=True, exist_ok=True)

    train_size = _resolve_size(
        args.train_data_size,
        "train",
        args.data_source,
        args.alfworld_config_path,
        args.alfworld_eval_dataset,
        args.webshop_root,
    )
    val_split = "test"
    if args.data_source == "webshop":
        val_split = webshop_eval_dataset
    val_size = _resolve_size(
        args.val_data_size,
        val_split,
        args.data_source,
        args.alfworld_config_path,
        args.alfworld_eval_dataset,
        args.webshop_root,
    )

    train_dataset = build_dataset(train_size, "train", args.data_source, args.agent_name)
    val_indices = None
    if args.data_source == "webshop" and val_split == "test_128":
        val_indices = _webshop_indices(val_split, val_size)
        val_size = len(val_indices)
    val_dataset = build_dataset(val_size, val_split, args.data_source, args.agent_name, indices=val_indices)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))
    print(f"Wrote {train_path} ({train_size}) and {val_path} ({val_size})")


if __name__ == "__main__":
    main()
