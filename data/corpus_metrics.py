"""Compare the normative memo body with published metric definitions."""

from app.core.errors import CorpusValidationError
from app.schemas.corpus import CorpusDocument
from app.schemas.metrics import MetricDefinition

DEFINITION_HEADING = "## 规范定义\n"
EXPLANATION_HEADING = "## 使用说明\n"


def metric_contract_text(definition: MetricDefinition) -> str:
    """Render the authoritative facts as readable prose and explicit field lists."""
    return "\n\n".join(
        [
            f"指标\uff1a{definition.key}；版本\uff1a{definition.version}；名称\uff1a{definition.display_name}",
            definition.description,
            "默认日期字段\uff1a" + definition.default_date_field.value,
            "必需过滤条件\uff1a\n" + "\n".join("- " + item for item in definition.required_filters),
            "支持粒度\uff1a" + "、".join(grain.value for grain in definition.supported_grains),
            "来源表\uff1a" + "、".join(definition.base_tables),
        ]
    )


def normative_body(text: str) -> str:
    """Require one visible definition section bounded by its business explanation."""
    if text.count(DEFINITION_HEADING) != 1 or text.count(EXPLANATION_HEADING) != 1:
        raise CorpusValidationError("Memo needs exactly one normative section.")
    _, _, remainder = text.partition(DEFINITION_HEADING)
    body, separator, _ = remainder.partition(EXPLANATION_HEADING)
    if not separator:
        raise CorpusValidationError("Memo section order is invalid.")
    return body.strip()


def validate_metric_memos(
    documents: list[CorpusDocument], definitions: list[MetricDefinition]
) -> None:
    """Check real memo text, including all catalog dates, filters and grains."""
    expected = {definition.key: definition for definition in definitions}
    memos = [document for document in documents if document.metadata.metric_key is not None]
    keys = [document.metadata.metric_key for document in memos]
    if not expected or len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise CorpusValidationError("Metric memos do not cover the published catalog exactly.")
    for document in memos:
        key = document.metadata.metric_key
        if key is None:
            raise CorpusValidationError("Missing metric key.")
        if normative_body(document.text) != metric_contract_text(expected[key]):
            raise CorpusValidationError(
                "Metric memo body differs from catalog.", path=document.entry.path
            )
