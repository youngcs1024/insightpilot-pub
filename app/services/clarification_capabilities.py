"""Fresh capability menus from committed application reference catalogs."""

from app.core.config_models import DatabaseSettings
from app.core.deadline import Deadline
from app.core.retry import run_operation
from app.db.session import Database
from app.repositories.document import DocumentRepository
from app.repositories.metric import MetricRepository
from app.schemas.clarification import AvailableMetric, ClarificationCapabilities
from app.schemas.ingestion import DocumentStatus


class ClarificationCapabilityService:
    """Own bounded reads; never advertise uncommitted or deleted corpus members."""

    def __init__(self, database: Database, settings: DatabaseSettings) -> None:
        self._database = database
        self._timeout_s = settings.command_timeout_s

    async def read(self, *, deadline: Deadline) -> ClarificationCapabilities:
        """Read within the caller deadline and the existing typed DB retry policy."""

        async def read() -> ClarificationCapabilities:
            async with self._database.session() as session, session.begin():
                metrics = await MetricRepository(session).list_active()
                documents = DocumentRepository(session)
                manifest = await documents.manifest()
                members = (
                    {
                        (item.document_id, item.document_version, item.chunking_version)
                        for item in manifest.members
                    }
                    if manifest
                    else set()
                )
                registered = await documents.list_documents() if members else []
                categories = {
                    item.metadata.doc_type
                    for item in registered
                    if item.status is DocumentStatus.ACTIVE
                    and item.chunk_count > 0
                    and (item.document_id, item.document_version, item.chunking_version) in members
                }
                return ClarificationCapabilities(
                    metrics=[
                        AvailableMetric(key=item.key, display_name=item.display_name)
                        for item in metrics
                    ],
                    document_categories=sorted(categories),
                )

        return await run_operation(
            read, deadline=deadline, timeout_s=self._timeout_s, name="clarification_capabilities"
        )
