# cinemastream/tests/test_smoke.py
# Smoke tests: verify core scripts load without errors.
# These run on every push via .github/workflows/ci.yml (created in Ch 017a).

import sys
import importlib
from pathlib import Path

# Make cinemastream/scripts importable during CI
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


def test_scripts_directory_exists():
    """cinemastream/scripts/ must exist for the lint step to pass."""
    assert scripts_dir.exists(), f"Expected {scripts_dir} to exist"


def test_logging_setup_importable():
    """logging_setup (from Ch 013a) should import without crashing."""
    try:
        mod = importlib.import_module("logging_setup")
        assert hasattr(mod, "get_logger"), "get_logger() must be exported"
    except ModuleNotFoundError:
        import pytest
        pytest.skip("logging_setup.py not yet in scripts/ — skipping")


def test_no_hardcoded_secrets_in_scripts():
    """
    Basic guard: no script should contain literal credential strings.
    Catches the most common accidental commit pattern.
    """
    scripts = list(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
    bad_patterns = ["SECRET_KEY", "password ="]
    violations = []
    for script in scripts:
        content = script.read_text(encoding="utf-8")
        for pattern in bad_patterns:
            if pattern in content:
                violations.append(f"{script.name}: found '{pattern}'")
    assert not violations, "Potential hardcoded secrets:\n" + "\n".join(violations)
