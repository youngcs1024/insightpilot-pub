"""The accepted parent and data node interfaces stay fixed during memory integration."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_node_signature_changed_since_phase_4() -> None:
    snapshot = json.loads((ROOT / "tests/node_signatures.json").read_text())
    assert snapshot
    for key, expected in snapshot.items():
        path, name = key.split(":")
        nodes = ast.parse((ROOT / path).read_text()).body
        node = next(item for item in nodes if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
        assert ast.unparse(node.args) == expected, key
