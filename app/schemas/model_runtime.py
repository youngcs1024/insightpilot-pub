"""Version-one model HTTP contracts, importable without any ML dependency."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeInt

EMBED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
Text = Annotated[str, Field(min_length=1, max_length=32_000)]
Dense = Annotated[list[FiniteFloat], Field(min_length=1024, max_length=1024)]
Sparse = Annotated[dict[NonNegativeInt, Annotated[FiniteFloat, Field(ge=0)]], Field(min_length=1)]
Score = Annotated[FiniteFloat, Field(ge=0, le=1)]


class Contract(BaseModel):
    """Reject silently extended or incompatible wire contracts."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    schema_version: Literal[1] = 1


class EmbedMode(StrEnum):
    """Modes share tokenization, truncation and normalization semantics."""

    DOCUMENT = "document"
    QUERY = "query"


class EmbedRequest(Contract):
    """One bounded model-server request, before future client-side splitting."""

    texts: Annotated[list[Text], Field(min_length=1, max_length=16)]
    mode: EmbedMode


class RerankRequest(Contract):
    """All retrieval candidates or compression sentences in one network call."""

    query: Text
    passages: Annotated[list[Text], Field(min_length=1, max_length=256)]
    max_length: int = Field(default=320, ge=1, le=320)


class ModelMetadata(Contract):
    """Exact identities and effective settings accompany every model result."""

    embed_model: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    rerank_model: Literal["BAAI/bge-reranker-v2-m3"] = "BAAI/bge-reranker-v2-m3"
    embed_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    rerank_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    precision: Literal["fp16", "fp32"]
    embed_batch: int = Field(ge=1, le=16)
    rerank_batch: int = Field(ge=1, le=16)
    embed_max_length: int = Field(ge=1, le=512)
    rerank_max_length: int = Field(ge=1, le=320)
    dense_normalized: Literal[True] = True
    score_normalization: Literal["sigmoid"] = "sigmoid"


class ModelResponse(Contract):
    """Server duration includes queueing; it is not client wall time."""

    request_id: str
    ms: NonNegativeInt
    queue_ms: NonNegativeInt
    inference_ms: NonNegativeInt
    metadata: ModelMetadata


class EmbedResult(ModelResponse):
    """Both retrieval vector arms are mandatory and validated by the client."""

    dense: list[Dense]
    sparse: list[Sparse]


class RerankResult(ModelResponse):
    """Normalized scores retain passage order."""

    scores: list[Score]


class ReadyResult(Contract):
    """Successful readiness attests both model warmups and their identity."""

    request_id: str
    ready: Literal[True] = True
    metadata: ModelMetadata


class ModelFailureKind(StrEnum):
    """Retry decisions never inspect prose or infer retryability from HTTP status."""

    UNAVAILABLE = "MODEL_RUNTIME_UNAVAILABLE"
    CONTRACT = "MODEL_RUNTIME_CONTRACT"
    OOM = "MODEL_RUNTIME_OOM"
    QUEUE_FULL = "MODEL_RUNTIME_QUEUE_FULL"
    DEADLINE = "MODEL_RUNTIME_DEADLINE"
    AUTH = "MODEL_RUNTIME_AUTH"
    INPUT = "MODEL_RUNTIME_INPUT"
    INTERNAL = "MODEL_RUNTIME_INTERNAL"


class ModelFailure(Contract):
    """Public errors contain fixed messages, never input or backend exception text."""

    code: ModelFailureKind
    message: str
    request_id: str
    retryable: bool
