"""
utils.py — Shared helpers for the nuScenes VLM Phase 1 pipeline.

Provides:
  - Logging setup
  - Config loading (YAML)
  - JSONL read/write helpers
  - Path resolution
  - Reproducible-seed utilities
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console (and optionally file) handlers.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR.
        log_file: Optional path to write log output.

    Returns:
        The root logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (inherits root handler configuration)."""
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.  Defaults to
            ``<project_root>/configs/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "configs" / "config.yaml")

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


def load_prompt(cfg: Dict[str, Any]) -> str:
    """Load the fixed inference prompt from the path specified in the config.

    Args:
        cfg: Top-level configuration dictionary.

    Returns:
        Prompt string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_file = cfg.get("inference", {}).get("prompt_file", "configs/prompt.txt")
    prompt_path = _PROJECT_ROOT / prompt_file
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def resolve_output_path(cfg: Dict[str, Any], subdir: str, filename: str) -> Path:
    """Build an absolute output path from config.

    Args:
        cfg: Top-level config dictionary.
        subdir: Key in ``cfg['outputs']`` (e.g. ``"predictions"``).
        filename: File name to append.

    Returns:
        Absolute Path, with parent directories created.
    """
    outputs_root = _PROJECT_ROOT / cfg["outputs"]["root"]
    sub = cfg["outputs"].get(subdir, subdir)
    target = outputs_root / sub / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ─────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed JSON objects.  Empty list if the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logging.getLogger(__name__).warning(
                        "Skipping malformed JSONL line in %s: %s", path, exc
                    )
    return records


def append_jsonl(path: Path | str, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file (atomic-ish incremental write).

    Args:
        path: Destination ``.jsonl`` file (created if absent).
        record: JSON-serialisable dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str))
        fh.write("\n")


def load_completed_tokens(predictions_path: Path | str) -> set:
    """Return the set of sample_token values already saved in a predictions file.

    Used to resume interrupted inference runs without reprocessing.

    Args:
        predictions_path: Path to an existing ``.jsonl`` predictions file.

    Returns:
        Set of sample token strings already present.
    """
    records = load_jsonl(predictions_path)
    return {r["sample_token"] for r in records if "sample_token" in r}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager that measures elapsed wall-clock time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────────

def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a one-level nested list."""
    return [item for sublist in nested for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
