from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml


def _flatten_sections(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten config sections into one argparse-compatible dictionary."""
    flat: dict[str, Any] = {}
    for _, section in cfg.items():
        if isinstance(section, Mapping):
            flat.update(section)
    return flat


def load_args(argv: list[str] | None = None, *, default_config: str | Path | None = None) -> argparse.Namespace:
    """
    Parse `--config`, load a YAML config, and expose fields as an argparse.Namespace.
    """
    default_path = Path(default_config) if default_config is not None else Path("configs/FaDeNet/prism_config.yaml")
    parser = argparse.ArgumentParser(description="FaDeNet config")
    parser.add_argument("--config", type=str, default=str(default_path), help="Path to a YAML config file")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Invalid config format in {args.config!r} (expected a mapping at root)")

    for k, v in _flatten_sections(cfg).items():
        setattr(args, k, v)
    return args
