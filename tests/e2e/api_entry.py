"""Production application with local observation capture, only in the test image."""

from asgi_correlation_id import correlation_id
from fastapi import FastAPI, Header

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import AuthenticationError
from app.core.observability import Observability, Observation, TraceMetadata
from tests.e2e.contracts import Observations, SpanRecord


class RecordingObservability(Observability):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.records: list[tuple[str, str, Observation]] = []

    def begin(
        self,
        name: str,
        metadata: TraceMetadata,
        *,
        generation: bool = False,
        trace_id: str | None = None,
        parent: Observation | None = None,
    ) -> Observation:
        observation = super().begin(
            name, metadata, generation=generation, trace_id=trace_id, parent=parent
        )
        self.records.append((correlation_id.get() or "", name, observation))
        return observation


def application() -> FastAPI:
    settings = Settings(_env_file=None)
    observer = RecordingObservability(settings)
    app = create_app(settings, observability=observer)

    @app.get("/_e2e/observations", response_model=Observations)
    async def observations(authorization: str = Header()) -> Observations:
        if authorization != "Bearer " + settings.mcp.auth_token.get_secret_value():
            raise AuthenticationError()
        return Observations(
            spans=[
                SpanRecord(request_id=identifier, name=name, metadata=observation.metadata)
                for identifier, name, observation in observer.records
            ]
        )

    return app
