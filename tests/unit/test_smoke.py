"""Verify the application skeleton."""

from pathlib import Path

import app
import app.core


def test_imports() -> None:
    """Import this project's package and expose its initial version."""
    assert app.__version__ == "0.1.0"
    assert app.__file__ is not None
    assert app.core.__file__ is not None
    root = Path(__file__).resolve().parents[2]
    assert Path(app.__file__).resolve() == root / "app" / "__init__.py"
    assert Path(app.core.__file__).resolve() == root / "app" / "core" / "__init__.py"
