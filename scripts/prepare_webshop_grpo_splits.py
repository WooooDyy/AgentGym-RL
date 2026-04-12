#!/usr/bin/env python3

import json
from pathlib import Path


def build_rows(start: int, stop: int) -> list[dict]:
    rows = []
    for local_idx, official_idx in enumerate(range(start, stop)):
        rows.append(
            {
                "item_id": f"webshop_{local_idx}",
                "conversations": [],
                "official_item_id": f"webshop_{official_idx}",
                "official_goal_idx": official_idx,
            }
        )
    return rows


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    human_goals_path = (
        root
        / "AgentGym"
        / "agentenv-webshop"
        / "webshop"
        / "baseline_models"
        / "data"
        / "human_goals.json"
    )
    with human_goals_path.open("r", encoding="utf-8") as f:
        human_goals = json.load(f)

    total = len(human_goals)
    test_end = 500
    eval_end = 1500
    if total != 12087:
        raise ValueError(f"Unexpected number of human goals: {total}")

    split_to_rows = {
        "train": build_rows(eval_end, total),
        "eval": build_rows(test_end, eval_end),
        "test": build_rows(0, test_end),
        "all": build_rows(0, total),
    }

    output_root = root / "AgentItemId"
    write_json(output_root / "train" / "webshop_train.json", split_to_rows["train"])
    write_json(output_root / "eval" / "webshop_eval.json", split_to_rows["eval"])
    write_json(output_root / "test" / "webshop_test.json", split_to_rows["test"])
    write_json(output_root / "all" / "webshop_all.json", split_to_rows["all"])

    write_json(output_root / "webshop_train.json", split_to_rows["train"])
    write_json(output_root / "webshop_eval.json", split_to_rows["eval"])
    write_json(output_root / "webshop_test.json", split_to_rows["test"])

    print("Prepared WebShop official human splits:")
    print(f"  train: {len(split_to_rows['train'])}")
    print(f"  eval:  {len(split_to_rows['eval'])}")
    print(f"  test:  {len(split_to_rows['test'])}")
    print(f"  all:   {len(split_to_rows['all'])}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
