"""Deterministic LangChain model and application structured-output substitute."""

from collections import deque
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, PrivateAttr

from app.core.deadline import Deadline
from app.core.llm_config import ModelRole


class FakeCall(BaseModel):
    """Snapshot of a model invocation, independent of subsequent prompt edits."""

    messages: list[BaseMessage]
    role: ModelRole | None = None
    schema_name: str | None = None


class FakeChatModel(BaseChatModel):
    """Consume scripted responses once, recording calls without any network I/O."""

    model_config = ConfigDict(extra="allow")  # Test-local method instrumentation is supported.

    _responses: deque[BaseModel | str | Exception] = PrivateAttr()
    _calls: list[FakeCall] = PrivateAttr(default_factory=list)

    def __init__(self, responses: Sequence[BaseModel | str | Exception]) -> None:
        super().__init__(cache=False)
        self._responses = deque(
            value.model_copy(deep=True) if isinstance(value, BaseModel) else value
            for value in responses
        )

    def enqueue(self, *responses: BaseModel | str | Exception) -> None:
        """Append scripted responses to a function-scoped fake fixture."""
        self._responses.extend(
            value.model_copy(deep=True) if isinstance(value, BaseModel) else value
            for value in responses
        )

    @property
    def _llm_type(self) -> str:
        return "insightpilot-scripted-fake"

    @property
    def calls(self) -> list[FakeCall]:
        """Return detached records so assertions cannot mutate the call history."""
        return [call.model_copy(deep=True) for call in self._calls]

    def _next(self, call: FakeCall) -> BaseModel | str:
        self._calls.append(call.model_copy(deep=True))
        if not self._responses:
            raise AssertionError("FakeChatModel response queue exhausted")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 -- LangChain override signature.
    ) -> ChatResult:
        response = self._next(FakeCall(messages=messages))
        content = response.model_dump_json() if isinstance(response, BaseModel) else response
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 -- LangChain override signature.
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)

    async def generate_structured[T: BaseModel](
        self, role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline
    ) -> T:
        """Respect deadlines and validate the response using the requested schema."""
        deadline.check("fake_llm")
        response = self._next(FakeCall(messages=messages, role=role, schema_name=schema.__name__))
        if isinstance(response, str):
            return schema.model_validate_json(response)
        return schema.model_validate(response.model_dump())
