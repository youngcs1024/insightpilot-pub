"""Program-owned qualifications rendered from committed observations only."""

from app.agents.contracts import MAX_ANSWER_CHARS
from app.schemas.sanity import SanityFlag

CAVEATS: dict[SanityFlag, str] = {
    SanityFlag.EMPTY_RESULT: "查询成功，但未返回任何行；不会因空结果放宽条件或重新查询。",
    SanityFlag.ALL_NULL: "返回结果中存在全 NULL 列，不能据此推断该列的数值。",
    SanityFlag.SINGLE_NULL_SCALAR: "查询返回单个 NULL，无法据此得到数值；NULL 不等于零。",
    SanityFlag.TRUNCATED: "查询结果已截断；统计仅覆盖返回行，不能代表完整总体。",
    SanityFlag.SUSPICIOUS_ZERO: "明确预期非零的列中出现零值，请核对业务预期。",
    SanityFlag.EXTREME_MAGNITUDE: "返回数值超过配置的极值阈值，请核对量级；这不证明数据错误。",
    SanityFlag.NEGATIVE_MONEY: "明确配置的金额列中出现负数，请结合业务口径解释。",
    SanityFlag.CARDINALITY_SPIKE: "返回行数超过配置的预期上限，请核对结果粒度。",
}


def render_caveats(flags: list[SanityFlag]) -> str:
    """Make every stored advisory visible even if generated prose omits it."""
    if not flags:
        return ""
    lines = [f"- {CAVEATS[flag]}" for flag in dict.fromkeys(flags)]
    return "\n\n结果提示:\n\n" + "\n".join(lines)


def append_caveats(markdown: str, flags: list[SanityFlag]) -> str:
    """Reserve room for all advisories within the existing answer size contract."""
    caveats = render_caveats(flags)
    available = MAX_ANSWER_CHARS - len(caveats)
    if len(markdown) > available:
        note = "\n\n[回答正文已截短，以保留完整结果提示。]"
        markdown = markdown[: available - len(note)] + note
    return markdown + caveats
