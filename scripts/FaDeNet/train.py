from __future__ import annotations

import argparse
import sys

from scripts.common import train as base_train


def main() -> None:
    p = argparse.ArgumentParser(description="Train FaDeNet (wrapper over common trainer).")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML (e.g. configs/FaDeNet/prism_config.yaml)",
    )
    args, unknown = p.parse_known_args()

    sys.argv = [
        "scripts.common.train",
        "--config",
        args.config,
        "--model_pkg",
        "FaDeNet",
        *unknown,
    ]
    base_train.main()


if __name__ == "__main__":
    main()
