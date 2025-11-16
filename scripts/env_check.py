#!/usr/bin/env python
"""Quick checks to validate the Python environment for LLM Observatory."""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List

REQUIRED_MODULES: Dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "peft": "peft",
    "sentencepiece": "sentencepiece",
    "faiss": "faiss-cpu",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "wandb": "wandb",
    "tensorboard": "tensorboard",
}

PYTHON_VERSION_ERROR = "".join(
    [
        "Python 3.11 is required, but the current version is ",
        "{version}",
    ]
)
NUMPY_VERSION_ERROR = "".join(
    [
        "numpy 1.26.x is required, but found ",
        "numpy=={version}",
    ]
)
SUCCESS_MESSAGE = " ".join(
    [
        "Environment validation succeeded.",
        "All required dependencies are available.",
    ]
)


def check_python_version(errors: List[str]) -> None:
    """Ensure we are running on Python 3.11."""
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 11):
        version = f"{major}.{minor}"
        errors.append(PYTHON_VERSION_ERROR.format(version=version))


def check_required_modules(errors: List[str]) -> None:
    """Import each dependency to confirm it is installed."""
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - best effort logging
            errors.append(f"Failed to import {package_name}: {exc}")


def check_numpy_version(errors: List[str]) -> None:
    """Make sure numpy is pinned to the 1.26.x range."""
    import numpy as np

    if not np.__version__.startswith("1.26."):
        errors.append(NUMPY_VERSION_ERROR.format(version=np.__version__))


def main() -> None:
    errors: List[str] = []
    check_python_version(errors)
    check_required_modules(errors)
    check_numpy_version(errors)

    if errors:
        print("Environment validation failed:")
        for message in errors:
            print(f" - {message}")
        raise SystemExit(1)

    print(SUCCESS_MESSAGE)


if __name__ == "__main__":
    main()
