"""Tests for portable desktop-directory discovery."""

from pathlib import Path

from desktop_agent.platform_paths import find_desktop_directory


def test_macos_uses_current_users_desktop(tmp_path: Path) -> None:
    """The default must derive from the current home, not a fixed username."""

    desktop = tmp_path / "Desktop"
    desktop.mkdir()

    result = find_desktop_directory(
        home=tmp_path,
        environment={},
        platform="darwin",
    )

    assert result == desktop


def test_windows_prefers_existing_onedrive_desktop(tmp_path: Path) -> None:
    """A redirected Windows desktop should be found when it exists."""

    profile = tmp_path / "profile"
    onedrive = tmp_path / "OneDrive"
    desktop = onedrive / "Desktop"
    desktop.mkdir(parents=True)

    result = find_desktop_directory(
        home=profile,
        environment={
            "USERPROFILE": str(profile),
            "OneDrive": str(onedrive),
        },
        platform="win32",
    )

    assert result == desktop


def test_linux_reads_xdg_desktop_configuration(tmp_path: Path) -> None:
    """Linux localized/custom desktop paths should follow XDG settings."""

    configured = tmp_path / "桌面"
    configured.mkdir()
    config = tmp_path / ".config"
    config.mkdir()
    (config / "user-dirs.dirs").write_text(
        'XDG_DESKTOP_DIR="$HOME/桌面"\n',
        encoding="utf-8",
    )

    result = find_desktop_directory(
        home=tmp_path,
        environment={},
        platform="linux",
    )

    assert result == configured


def test_explicit_default_directory_override_is_supported(
    tmp_path: Path,
) -> None:
    """Packaged deployments may provide an explicit default when necessary."""

    custom = tmp_path / "incoming"

    result = find_desktop_directory(
        home=tmp_path,
        environment={"DESKTOP_AGENT_DEFAULT_DIRECTORY": str(custom)},
        platform="darwin",
    )

    assert result == custom
