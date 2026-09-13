"""Import every application model to populate Alembic's target metadata."""

from app.db.models.chunk import Chunk
from app.db.models.conversation import Conversation
from app.db.models.document import CorpusManifestRecord, Document
from app.db.models.evidence import DataEvidenceRecord
from app.db.models.metric_definition import MetricDefinitionRecord
from app.db.models.refresh_token import RefreshToken
from app.db.models.schema_metadata import SchemaMetadataRecord, SchemaTableRecord
from app.db.models.turn import Turn, TurnRole, TurnStatus
from app.db.models.user import User

__all__ = [
    "Chunk",
    "Conversation",
    "CorpusManifestRecord",
    "DataEvidenceRecord",
    "Document",
    "MetricDefinitionRecord",
    "RefreshToken",
    "SchemaMetadataRecord",
    "SchemaTableRecord",
    "Turn",
    "TurnRole",
    "TurnStatus",
    "User",
]
