"""Runtime dependency checks shared by public modules."""

from __future__ import annotations

import importlib
import shutil


def require_executable(name: str, install_hint: str | None = None) -> str:
    """Return an executable path or raise an actionable error."""
    executable = shutil.which(name)
    if executable:
        return executable

    hint = f" {install_hint}" if install_hint else ""
    raise RuntimeError(
        f"Required executable '{name}' was not found on PATH.{hint}"
    )


def require_module(module_name: str, package_hint: str | None = None):
    """Import a module or raise an actionable ImportError."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise ImportError(
            f"Required Python dependency '{module_name}' is not installed. "
            f"Install it with '{hint}' or install emsaplibraries with its "
            "declared dependencies."
        ) from exc
