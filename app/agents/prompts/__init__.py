"""Versioned prompt assets are loaded once, never inside a node invocation."""

from pathlib import Path

_ROOT = Path(__file__).parent
SYSTEM = (_ROOT / "system.md").read_text()
SQL_GENERATE = (_ROOT / "sql_generate.md").read_text()
FORMAT_ANSWER = (_ROOT / "format_answer.md").read_text()
SCHEMA = (_ROOT / "schema.md").read_text()
METRIC_INTENT = (_ROOT / "metric_intent.md").read_text()
SQL_GENERATE_PHASE1 = (_ROOT / "sql_generate_phase1.md").read_text()
SQL_CORRECT = (_ROOT / "sql_correct.md").read_text()
REWRITE_QUESTION = (_ROOT / "rewrite_question.md").read_text()
KNOWLEDGE_SYSTEM = (_ROOT / "knowledge_system.md").read_text()
KNOWLEDGE_REWRITE = (_ROOT / "knowledge_rewrite.md").read_text()
KNOWLEDGE_CITATION_REPAIR = (_ROOT / "knowledge_citation_repair.md").read_text()
ROUTER = (_ROOT / "router.md").read_text()
