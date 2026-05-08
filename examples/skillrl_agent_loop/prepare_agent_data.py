import argparse
import os
import sys
from pathlib import Path

import datasets


def _add_skillrl_root(skillrl_root: str | None) -> str:
    if skillrl_root is None:
        skillrl_root = os.environ.get("SKILLRL_ROOT")
    if skillrl_root is None:
        skillrl_root = str(Path(__file__).resolve().parents[3] / "SkillRL")
    if skillrl_root not in sys.path:
        sys.path.insert(0, skillrl_root)
    return skillrl_root


def _count_alfworld(config_path: str, split: str) -> int:
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
        key = "eval_id_data_path"
        data_path = os.path.expandvars(config["dataset"].get(key, data_path))
    root = os.path.join(ALFWORLD_DATA, data_path)
    return sum(1 for _, _, files in os.walk(root) for name in files if name == "game.tw-pddl")


def _resolve_size(size: int, split: str, data_source: str, alfworld_config_path: str | None) -> int:
    if size >= 0:
        return size
    if data_source == "alfworld" and alfworld_config_path:
        return _count_alfworld(alfworld_config_path, split)
    raise ValueError(f"{data_source} requires an explicit non-negative size for split={split}.")


def build_dataset(size: int, split: str, data_source: str, agent_name: str) -> datasets.Dataset:
    tool_selection = {
        "alfworld": ["take_action"],
        "webshop": ["search_action", "click_action"],
    }[data_source]
    rows = []
    for idx in range(size):
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
    return datasets.Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--local_dir", default="/mnt/workspace/xts/data/skillrl-agent-loop")
    parser.add_argument("--agent_name", default="skillrl_env_agent")
    parser.add_argument("--train_data_size", type=int, default=16)
    parser.add_argument("--val_data_size", type=int, default=16)
    parser.add_argument("--alfworld_config_path", default=None)
    parser.add_argument("--skillrl_root", default=None)
    args = parser.parse_args()

    _add_skillrl_root(args.skillrl_root)
    output_dir = Path(args.local_dir).expanduser() / args.data_source
    output_dir.mkdir(parents=True, exist_ok=True)

    train_size = _resolve_size(args.train_data_size, "train", args.data_source, args.alfworld_config_path)
    val_size = _resolve_size(args.val_data_size, "test", args.data_source, args.alfworld_config_path)

    train_dataset = build_dataset(train_size, "train", args.data_source, args.agent_name)
    val_dataset = build_dataset(val_size, "test", args.data_source, args.agent_name)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))
    print(f"Wrote {train_path} ({train_size}) and {val_path} ({val_size})")


if __name__ == "__main__":
    main()

