"""Small AST rules enforcing import/call boundaries without matching comments or prose."""

import ast
from pathlib import PurePosixPath


class RuleVisitor(ast.NodeVisitor):
    """Resolve ordinary and aliased imports, including imports inside functions."""

    def __init__(self, path: str) -> None:
        self.path = PurePosixPath(path)
        self.aliases: dict[str, str] = {}
        self.violations: dict[str, list[int]] = {
            "no_bare_create_task": [],
            "no_detail_str_e": [],
            "no_sync_engine_in_app": [],
            "nodes_do_no_io": [],
        }

    def resolve(self, node: ast.expr) -> str:
        """Resolve a name or dotted attribute through the local import map."""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.resolve(node.value)}.{node.attr}"
        return ""

    def check_import(self, module: str, line: int) -> None:
        forbidden = (
            "app.db",
            "app.clients",
            "app.repositories",
            "sqlalchemy",
            "asyncpg",
            "psycopg",
            "psycopg_pool",
            "httpx",
            "httpx2",
            "requests",
            "aiohttp",
            "urllib.request",
            "socket",
            "openai",
            "langchain_openai",
            "langchain_core.language_models",
            "langchain.chat_models",
        )
        if self.path.is_relative_to("app/agents/nodes") and any(
            module == prefix or module.startswith(prefix + ".") for prefix in forbidden
        ):
            self.violations["nodes_do_no_io"].append(line)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )
            self.check_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            package = self.path.parent.parts
            module = ".".join((*package[: len(package) - node.level + 1], module)).rstrip(".")
        for alias in node.names:
            qualified = f"{module}.{alias.name}"
            self.aliases[alias.asname or alias.name] = qualified
            self.check_import(qualified, node.lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            isinstance(node.value, ast.Call)
            and self.resolve(node.value.func) == "structlog.get_logger"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = "structlog.BoundLogger"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve(node.func)
        if name == "asyncio.create_task" and self.path != PurePosixPath("app/core/background.py"):
            self.violations["no_bare_create_task"].append(node.lineno)
        if name in {
            "sqlalchemy.create_engine",
            "sqlalchemy.engine.create_engine",
            "sqlalchemy.engine.create.create_engine",
        }:
            self.violations["no_sync_engine_in_app"].append(node.lineno)
        for keyword in node.keywords:
            if (
                keyword.arg == "detail"
                and name != "structlog.BoundLogger.exception"
                and isinstance(keyword.value, ast.Call)
                and self.resolve(keyword.value.func) in {"str", "builtins.str"}
            ):
                self.violations["no_detail_str_e"].append(node.lineno)
        self.generic_visit(node)


def violations(source: str, path: str) -> dict[str, list[int]]:
    """Inspect Python source; strings, comments and TaskGroup calls are not violations."""
    visitor = RuleVisitor(path)
    visitor.visit(ast.parse(source))
    return visitor.violations
