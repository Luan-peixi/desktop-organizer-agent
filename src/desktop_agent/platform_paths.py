"""Portable local-directory discovery for the desktop web application."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

DEFAULT_DIRECTORY_ENV = "DESKTOP_AGENT_DEFAULT_DIRECTORY"


def find_desktop_directory(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> Path:
    """Find the current user's desktop without embedding a username."""

    user_home = (home or Path.home()).expanduser()
    env = environment if environment is not None else os.environ
    system = platform or sys.platform

    explicit = env.get(DEFAULT_DIRECTORY_ENV)
    if explicit:
        return Path(explicit).expanduser()

    candidates: list[Path] = []
    if system.startswith("win"):
        for variable in ("OneDrive", "OneDriveConsumer", "USERPROFILE"):
            root = env.get(variable)
            if root:
                candidates.append(Path(root) / "Desktop")
    else:
        xdg_desktop = _read_xdg_desktop(user_home, env)
        if xdg_desktop is not None:
            candidates.append(xdg_desktop)
        candidates.append(user_home / "Desktop")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    # Keep the conventional path visible and editable if a desktop directory
    # has not been created yet or the operating system denies access.
    return candidates[0] if candidates else user_home / "Desktop"


def _read_xdg_desktop(
    home: Path,
    environment: Mapping[str, str],
) -> Path | None:
    """Read Linux's configured desktop directory without executing shell text."""

    config_home = Path(
        environment.get("XDG_CONFIG_HOME", str(home / ".config"))
    )
    config = config_home / "user-dirs.dirs"
    try:
        content = config.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r'^XDG_DESKTOP_DIR="([^"\n]+)"$', content, re.MULTILINE)
    if match is None:
        return None
    raw_path = match.group(1)
    if "$" in raw_path.replace("$HOME", ""):
        return None
    return Path(raw_path.replace("$HOME", str(home), 1)).expanduser()
