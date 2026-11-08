"""Reject comparisons against paper tables from incompatible run configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PAPER_PROTOCOL = {
    "table1": ("bacon", 10),
    "table2": ("bacon", 10),
    "table4": ("simgcd", 10),
}


def verify(config: dict, table: str) -> None:
    expected_base, expected_imb = PAPER_PROTOCOL[table]
    observed_base = str(config["baseline"]["name"])
    observed_imb = int(config["data"]["protocol"]["imbalance_ratio"])
    if (observed_base, observed_imb) != (expected_base, expected_imb):
        raise ValueError(
            f"NOT comparable with paper {table}: "
            f"observed baseline={observed_base}, LT={observed_imb}; "
            f"expected baseline={expected_base}, LT={expected_imb}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-config", type=Path, required=True)
    parser.add_argument("--paper-table", choices=sorted(PAPER_PROTOCOL), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.saved_config.read_text(encoding="utf-8"))
    verify(config, args.paper_table)
    print("BASELINE + IMBALANCE RATIO MATCH. Now also verify split IDs, "
          "source commit, optimization settings, and evaluation backend.")
