from __future__ import annotations

import argparse
import sys

from scripts.common import test as base_test


def main() -> None:
    p = argparse.ArgumentParser(description="Test FaDeNet (wrapper over common tester).")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML (e.g. configs/FaDeNet/prism_config.yaml)",
    )
    args, unknown = p.parse_known_args()

    sys.argv = [
        "scripts.common.test",
        "--config",
        args.config,
        "--model_pkg",
        "FaDeNet",
        *unknown,
    ]
    base_test.main()


if __name__ == "__main__":
    main()
