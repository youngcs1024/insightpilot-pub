"""Retain actual server tokens for ten fixed representative strings."""

import asyncio
import sys

from pydantic import ValidationError

from app.core.errors import InsightPilotError, RetrievalConfigurationError
from app.retrieval.config import MilvusSettings
from app.retrieval.milvus_repo import AnalyzerReport, AnalyzerRequest, MilvusRepository
from scripts.milvus_init import MilvusOperatorSettings

SAMPLES = (
    "七天无理由退货政策",
    "SKU-A1023 退款",
    "618大促规则",
    "SKU-B2048 七天退货",
    "SKU-A1023，七天退货。",
    "退款率 GMV KPI",
    "华东地区八月退款政策",
    "促销折扣不叠加",
    "订单取消与售后退款",
    "SKU-A1023 SKU-B2048 对比",
)


def require_sku_tokens(report: AnalyzerReport) -> None:
    """A fragment, concatenation or dropped hyphen never substitutes for a SKU token."""
    for sample in report.samples:
        for sku in ("SKU-A1023", "SKU-B2048"):
            if sku in sample.text and sku not in sample.tokens:
                raise RetrievalConfigurationError(reason="sku_not_preserved")


async def smoke(settings: MilvusSettings) -> int:
    """Print even failing analyzer evidence before returning a nonzero status."""
    async with MilvusRepository(settings) as repository:
        report = await repository.analyze(AnalyzerRequest(texts=list(SAMPLES)))
        print(report.model_dump_json(indent=2))
        require_sku_tokens(report)
    return 0


def main() -> int:
    """Use exactly the same bounded environment configuration as milvus_init."""
    try:
        return asyncio.run(smoke(MilvusOperatorSettings().retrieval.milvus))
    except ValidationError:
        print("Invalid Milvus configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(f"{exc.code}: {exc.user_message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
