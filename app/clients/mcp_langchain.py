"""Convert the locked MCP server's input schemas into validated async LangChain tools."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Mapping, Never, cast

from langchain_core.tools import BaseTool, ToolException
from mcp.types import Tool
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    create_model,
)

from app.clients.mcp_client import McpClient
from app.core.deadline import current_deadline
from app.core.errors import McpPolicyRejected, McpResultError, McpToolSchemaError

_ANNOTATIONS = {"title", "description", "default"}
_STRING = {"minLength", "maxLength", "pattern"}
_ARRAY = {"minItems", "maxItems"}
_NUMERIC = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
_FIELD_NAMES = {
    "minLength": "min_length",
    "maxLength": "max_length",
    "minItems": "min_length",
    "maxItems": "max_length",
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "pattern": "pattern",
}
_KNOWN_KEYWORDS = (
    _ANNOTATIONS
    | set(_FIELD_NAMES)
    | {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "anyOf",
        "$ref",
        "$defs",
        "additionalProperties",
        "format",
    }
)


def _fail(tool: str, path: str, construct: str) -> Never:
    raise McpToolSchemaError(tool, f"{path}: {construct}")


def _mapping(value: object, tool: str, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(tool, path, "expected object schema")
    return cast("Mapping[str, object]", value)


def _check_keywords(schema: Mapping[str, object], allowed: set[str], tool: str, path: str) -> None:
    unsupported = set(schema) - _ANNOTATIONS - allowed
    if unsupported:
        _fail(tool, path, sorted(unsupported)[0])


def _field(schema: Mapping[str, object], required: bool) -> Any:
    options: dict[str, Any] = {
        key: schema[key] for key in ("description", "title") if key in schema
    }
    if required:
        return Field(..., **options)
    if "default" in schema:
        return Field(schema["default"], **options)
    # A default factory makes this field omittable without advertising a false
    # JSON-Schema default. exclude_unset keeps its placeholder off the wire.
    return Field(default_factory=lambda: None, **options)


def _constrained(annotation: Any, schema: Mapping[str, object]) -> Any:
    options: dict[str, Any] = {
        field: schema[wire] for wire, field in _FIELD_NAMES.items() if wire in schema
    }
    return Annotated[annotation, Field(**options)] if options else annotation


def _object_model(  # noqa: PLR0913, PLR0917 -- recursive schema context is explicit.
    schema: Mapping[str, object],
    name: str,
    tool: str,
    path: str,
    definitions: Mapping[str, object],
    ref_depth: int,
) -> type[BaseModel]:
    raw_properties = schema.get("properties", {})
    properties = _mapping(raw_properties, tool, f"{path}.properties")
    raw_required = schema.get("required", [])
    if not isinstance(raw_required, list) or not all(
        isinstance(item, str) for item in raw_required
    ):
        _fail(tool, f"{path}.required", "expected string array")
    required = set(raw_required)
    if required - set(properties):
        _fail(tool, f"{path}.required", "unknown field")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        _fail(tool, f"{path}.additionalProperties", "schema-valued extras")
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, raw_field in properties.items():
        field_path = f"{path}.properties.{field_name}"
        field_schema = _mapping(raw_field, tool, field_path)
        annotation = _annotation(
            field_schema, f"{name}_{field_name}", tool, field_path, definitions, ref_depth
        )
        fields[field_name] = (
            annotation,
            _field(field_schema, field_name in required),
        )
    return create_model(
        name,
        __config__=ConfigDict(extra="allow" if additional else "forbid"),
        **fields,
    )


def _annotation(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915, PLR0917 -- bounded subset.
    schema: Mapping[str, object],
    name: str,
    tool: str,
    path: str,
    definitions: Mapping[str, object],
    ref_depth: int,
) -> Any:
    unknown = set(schema) - _KNOWN_KEYWORDS
    if unknown:
        _fail(tool, path, sorted(unknown)[0])
    if "$defs" in schema and path != "inputSchema":
        _fail(tool, path, "nested $defs")
    if "$ref" in schema:
        _check_keywords(schema, {"$ref"}, tool, path)
        reference = schema["$ref"]
        if ref_depth or not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            _fail(tool, path, "$ref depth or target")
        target = reference.removeprefix("#/$defs/")
        if target not in definitions:
            _fail(tool, path, f"missing $defs/{target}")
        return _annotation(
            _mapping(definitions[target], tool, f"$defs.{target}"),
            f"{name}_{target}",
            tool,
            f"$defs.{target}",
            definitions,
            ref_depth + 1,
        )
    if "anyOf" in schema:
        _check_keywords(schema, {"anyOf"}, tool, path)
        choices = schema["anyOf"]
        if not isinstance(choices, list) or len(choices) != 2:
            _fail(tool, path, "anyOf shape")
        variants = [_mapping(item, tool, f"{path}.anyOf") for item in choices]
        nulls = [item for item in variants if item.get("type") == "null"]
        if len(nulls) != 1:
            _fail(tool, path, "anyOf must contain one null")
        _annotation(nulls[0], f"{name}_Null", tool, f"{path}.anyOf.null", definitions, ref_depth)
        value_schema = next(item for item in variants if item.get("type") != "null")
        return _annotation(value_schema, name, tool, f"{path}.anyOf", definitions, ref_depth) | None
    kind = schema.get("type")
    if not isinstance(kind, str):
        _fail(tool, path, f"type={kind}")
    allowed = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items"} | _ARRAY,
        "string": {"format"} | _STRING,
        "integer": _NUMERIC,
        "number": _NUMERIC,
        "boolean": set(),
        "null": set(),
    }.get(kind)
    if allowed is None:
        _fail(tool, path, f"type={kind}")
    allowed = allowed | {"type", "enum"}
    if path == "inputSchema":
        allowed.add("$defs")
    _check_keywords(schema, allowed, tool, path)
    if "enum" in schema:
        if kind not in {"string", "integer", "number", "boolean", "null"}:
            _fail(tool, path, "enum type")
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            _fail(tool, path, "enum values")
        return _constrained(Literal.__getitem__(tuple(values)), schema)
    if kind == "object":
        return _object_model(schema, name, tool, path, definitions, ref_depth)
    if kind == "array":
        if "items" not in schema:
            _fail(tool, path, "array.items")
        item = _annotation(
            _mapping(schema["items"], tool, f"{path}.items"),
            f"{name}_Item",
            tool,
            f"{path}.items",
            definitions,
            ref_depth,
        )
        return _constrained(list[item], schema)
    if kind == "string":
        fmt = schema.get("format")
        if fmt is None:
            return _constrained(StrictStr, schema)
        if fmt == "date-time":
            if _STRING & set(schema):
                _fail(tool, path, "date-time length or pattern")
            return AwareDatetime
        _fail(tool, path, f"format={fmt}")
    if kind == "integer":
        return _constrained(StrictInt, schema)
    if kind == "number":
        return _constrained(StrictFloat, schema)
    if kind == "boolean":
        return StrictBool
    if kind == "null":
        return type(None)
    _fail(tool, path, f"type={kind}")


def json_schema_to_pydantic(
    schema: Mapping[str, object],
    *,
    name: str,
    tool_name: str,
) -> type[BaseModel]:
    """Convert supported MCP input schemas before a tool becomes visible."""
    definitions = _mapping(schema.get("$defs", {}), tool_name, "inputSchema.$defs")
    if schema.get("type") != "object":
        _fail(tool_name, "inputSchema", "root must be object")
    return cast(
        "type[BaseModel]",
        _annotation(schema, name, tool_name, "inputSchema", definitions, 0),
    )


class McpLangChainTool(BaseTool):
    """One discovered descriptor backed by the lifespan-owned MCP client."""

    args_schema: type[BaseModel]
    _client: McpClient = PrivateAttr()

    def __init__(self, *, client: McpClient, **data: Any) -> None:
        super().__init__(**data)
        self._client = client

    def _parse_input(
        self,
        tool_input: str | dict[str, Any],
        tool_call_id: str | None,
    ) -> str | dict[str, Any]:
        if not isinstance(tool_input, dict):
            raise McpResultError()
        return self.args_schema.model_validate(tool_input).model_dump(exclude_unset=True)

    async def _arun(self, **kwargs: Any) -> str:
        arguments = self.args_schema.model_validate(kwargs)
        try:
            result = await self._client.invoke_discovered_tool(
                self.name, arguments, deadline=current_deadline()
            )
        except McpPolicyRejected as exc:
            refusal = {"code": exc.code, "reasons": [reason.value for reason in exc.reasons]}
            raise ToolException(json.dumps(refusal, sort_keys=True)) from exc
        return result.text

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("MCP LangChain tools are async only")


def mcp_tool_to_langchain(descriptor: Tool, client: McpClient) -> BaseTool:
    """Build one executable BaseTool with a local Pydantic argument model."""
    model = json_schema_to_pydantic(
        descriptor.input_schema, name=f"{descriptor.name}_Args", tool_name=descriptor.name
    )
    return McpLangChainTool(
        client=client,
        name=descriptor.name,
        description=descriptor.description or "",
        args_schema=model,
        handle_tool_error=True,
    )
