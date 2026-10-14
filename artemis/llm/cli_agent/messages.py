# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import binascii
import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from artemis.llm.cli_agent.types import ImageSegment, Segment, TextSegment

HEADER_USER = "### USER"
HEADER_ASSISTANT = "### ASSISTANT"
TOOL_RESULT_PREFIX = "### TOOL RESULT name="
TOOL_RESULT_ID_PREFIX = " id="
MESSAGE_SEPARATOR = "\n\n"
LINE_BREAK = "\n"
TOOL_CALLS_PREFIX = "tool_calls: "
IMAGE_URL_PREFIX = "[image url: "
IMAGE_URL_SUFFIX = "]"
INVALID_DATA_URI_TEXT = "[image url: invalid data uri]"
IGNORED_TOOL_PREFIX = "[ignored unknown tool: "
IGNORED_TOOL_SUFFIX = "]"
DATA_URI_PREFIX = "data:"
BASE64_TOKEN = "base64"
BLOCK_TEXT = "text"
BLOCK_IMAGE_URL = "image_url"
BLOCK_IMAGE = "image"
SOURCE_BASE64 = "base64"
DEFAULT_MIME = "application/octet-stream"
DEFAULT_EXTENSION = "bin"
JPEG_SUBTYPE = "jpeg"
JPEG_EXTENSION = "jpg"
FIRST_IMAGE_INDEX = 1
IMAGE_INDEX_WIDTH = 3
URLSAFE_TO_STANDARD = str.maketrans("-_", "+/")
TOOL_CALL_ID_PREFIX = "call_"
TOOL_CALL_ID_HEX_LENGTH = 12
TOOL_CALL_TYPE = "tool_call"
MAX_EXTENSION_LENGTH = 10
MIME_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/x-icon": "ico",
    "image/svg+xml": "svg",
    "image/heic": "heic",
    "image/heif": "heif",
}
CHOICE_ANY = "any"
CHOICE_REQUIRED = "required"
CHOICE_AUTO = "auto"
CHOICE_NONE = "none"
AT_LEAST_ONE_CHOICES = frozenset({CHOICE_ANY, CHOICE_REQUIRED})
NON_NAME_CHOICES = frozenset({CHOICE_ANY, CHOICE_REQUIRED, CHOICE_AUTO, CHOICE_NONE})
RULE_ENVELOPE_ONLY = "Return only the JSON envelope. Do not write an explanation."
RULE_ENVELOPE_SHAPE = (
    'The envelope is an object with a string field "content" and an array field "tool_calls".'
)
RULE_TOOL_CALL_SHAPE = (
    'Each tool call is an object with a string field "name" and an object field "args".'
)
RULE_MULTIPLE_CALLS = "Multiple tool calls in one turn are allowed."
RULE_AT_LEAST_ONE = "You must call at least one tool."
RULE_OPTIONAL = "Calling tools is optional."
RULE_NONE = "tool_calls must be empty."
RULE_NAMED_PREFIX = 'You must call the tool "'
RULE_NAMED_SUFFIX = '".'
TOOLS_HEADING = "Tools:"
Piece = str | ImageSegment


def build_request_parts(
    messages: Sequence[BaseMessage], workdir: Path
) -> tuple[str, list[Segment]]:
    system_parts: list[str] = []
    pieces: list[Piece] = []
    image_index = FIRST_IMAGE_INDEX
    for message in messages:
        if _is_system(message):
            text = _text_only(message.content)
            if text:
                system_parts.append(text)
            continue
        if pieces:
            _push_text(pieces, MESSAGE_SEPARATOR)
        _push_text(pieces, _header(message) + LINE_BREAK)
        image_index = _append_content(pieces, message.content, workdir, image_index)
        _append_tool_calls(pieces, message)
    return MESSAGE_SEPARATOR.join(system_parts), _to_segments(pieces)


def build_envelope_schema(tool_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": tool_names},
                        "args": {"type": "object"},
                    },
                    "required": ["name", "args"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }


def render_tool_instructions(tools: list[dict], tool_choice: Any) -> str:
    lines = [
        RULE_ENVELOPE_ONLY,
        RULE_ENVELOPE_SHAPE,
        RULE_TOOL_CALL_SHAPE,
        RULE_MULTIPLE_CALLS,
        "",
        TOOLS_HEADING,
    ]
    for tool in tools:
        lines.extend(_tool_lines(tool))
    lines.append("")
    lines.append(_tool_choice_rule(tool_choice))
    return "\n".join(lines)


def parse_envelope(text: str, structured: dict | None, tool_names: set[str]) -> AIMessage:
    payload = _envelope_dict(structured)
    if payload is None:
        payload = _last_json_object(text, _envelope_dict)
    if payload is None:
        payload = _bare_single_tool_envelope(structured, text, tool_names)
    if payload is None:
        return AIMessage(content=text)
    return _message_from_envelope(payload, tool_names)


def _bare_single_tool_envelope(
    structured: dict | None, text: str, tool_names: set[str]
) -> dict[str, Any] | None:
    if len(tool_names) != 1:
        return None
    args = structured if isinstance(structured, dict) else _last_json_object(text, _plain_dict)
    if args is None:
        return None
    return {"content": "", "tool_calls": [{"name": next(iter(tool_names)), "args": args}]}


def _plain_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _is_system(message: BaseMessage) -> bool:
    if isinstance(message, SystemMessage):
        return True
    return getattr(message, "type", "") == "system"


def _header(message: BaseMessage) -> str:
    if isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool":
        name = getattr(message, "name", None) or ""
        call_id = getattr(message, "tool_call_id", None) or ""
        return f"{TOOL_RESULT_PREFIX}{name}{TOOL_RESULT_ID_PREFIX}{call_id}"
    if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
        return HEADER_ASSISTANT
    if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
        return HEADER_USER
    role = getattr(message, "type", "message")
    return f"### {str(role).upper()}"


def _text_only(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
            continue
        if not isinstance(block, dict) or block.get("type") != BLOCK_TEXT:
            continue
        text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _push_text(pieces: list[Piece], text: str) -> None:
    if not text:
        return
    if pieces and isinstance(pieces[-1], str):
        pieces[-1] = pieces[-1] + text
        return
    pieces.append(text)


def _to_segments(pieces: list[Piece]) -> list[Segment]:
    segments: list[Segment] = []
    for piece in pieces:
        if isinstance(piece, str):
            segments.append(TextSegment(piece))
            continue
        segments.append(piece)
    return segments


def _append_content(pieces: list[Piece], content: Any, workdir: Path, index: int) -> int:
    if isinstance(content, str):
        _push_text(pieces, content)
        return index
    if content is None:
        return index
    if not isinstance(content, list):
        _push_text(pieces, str(content))
        return index
    for block in content:
        index = _append_block(pieces, block, workdir, index)
    return index


def _append_block(pieces: list[Piece], block: Any, workdir: Path, index: int) -> int:
    if isinstance(block, str):
        _push_text(pieces, block)
        return index
    if not isinstance(block, dict):
        return index
    block_type = block.get("type")
    if block_type == BLOCK_TEXT:
        text = block.get("text")
        if isinstance(text, str):
            _push_text(pieces, text)
        return index
    if block_type == BLOCK_IMAGE_URL:
        return _append_image_reference(pieces, _image_url_string(block), workdir, index)
    if block_type == BLOCK_IMAGE:
        return _append_image_block(pieces, block, workdir, index)
    return index


def _image_url_string(block: dict) -> str | None:
    image_url = block.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    return None


def _append_image_block(pieces: list[Piece], block: dict, workdir: Path, index: int) -> int:
    payload = _image_base64_payload(block)
    if isinstance(payload, str):
        if payload.startswith(DATA_URI_PREFIX):
            return _append_image_reference(pieces, payload, workdir, index)
        segment = _write_image(workdir, index, _block_mime(block), payload)
        if segment is None:
            _push_text(pieces, INVALID_DATA_URI_TEXT)
            return index
        pieces.append(segment)
        return index + 1
    url = _image_block_url(block)
    if url is None:
        return index
    return _append_image_reference(pieces, url, workdir, index)


def _image_base64_payload(block: dict) -> str | None:
    if block.get("source_type") == SOURCE_BASE64 and isinstance(block.get("data"), str):
        return block["data"]
    if isinstance(block.get("base64"), str):
        return block["base64"]
    data = block.get("data")
    if isinstance(data, str) and data.startswith(DATA_URI_PREFIX):
        return data
    return None


def _image_block_url(block: dict) -> str | None:
    url = block.get("url")
    if isinstance(url, str):
        return url
    return _image_url_string(block)


def _block_mime(block: dict) -> str:
    mime = block.get("mime_type")
    if not isinstance(mime, str) or not mime.strip():
        return DEFAULT_MIME
    return _bare_mime(mime)


def _append_image_reference(pieces: list[Piece], url: str | None, workdir: Path, index: int) -> int:
    if not isinstance(url, str) or not url:
        return index
    parsed = _parse_data_uri(url)
    if parsed is None:
        if url.startswith(DATA_URI_PREFIX):
            _push_text(pieces, INVALID_DATA_URI_TEXT)
            return index
        _push_text(pieces, f"{IMAGE_URL_PREFIX}{url}{IMAGE_URL_SUFFIX}")
        return index
    mime, payload = parsed
    segment = _write_image(workdir, index, mime, payload)
    if segment is None:
        _push_text(pieces, INVALID_DATA_URI_TEXT)
        return index
    pieces.append(segment)
    return index + 1


def _parse_data_uri(url: str) -> tuple[str, str] | None:
    if not url.startswith(DATA_URI_PREFIX):
        return None
    header, separator, payload = url.partition(",")
    if separator != ",":
        return None
    meta = header[len(DATA_URI_PREFIX) :]
    if BASE64_TOKEN not in meta.lower():
        return None
    mime = _bare_mime(meta.split(";")[0]) if meta.split(";")[0].strip() else DEFAULT_MIME
    return mime, payload


def _bare_mime(mime: str) -> str:
    cleaned = mime.split(";")[0].strip().lower()
    if not cleaned:
        return DEFAULT_MIME
    return cleaned


def _write_image(workdir: Path, index: int, mime: str, payload: str) -> ImageSegment | None:
    try:
        raw, cleaned = _decode_base64(payload)
    except (binascii.Error, ValueError):
        return None
    safe_mime = _bare_mime(mime)
    path = workdir / f"img_{index:0{IMAGE_INDEX_WIDTH}d}.{_extension_for(safe_mime)}"
    path.write_bytes(raw)
    return ImageSegment(path=path, media_type=safe_mime, base64_data=cleaned)


def _decode_base64(payload: str) -> tuple[bytes, str]:
    cleaned = "".join(payload.split())
    if not cleaned:
        raise ValueError("empty base64")
    standard = cleaned.translate(URLSAFE_TO_STANDARD)
    padding = (-len(standard)) % 4
    padded = standard + ("=" * padding)
    raw = base64.b64decode(padded, validate=True)
    if not raw:
        raise ValueError("empty image payload")
    return raw, padded


def _extension_for(mime: str) -> str:
    mapped = MIME_EXTENSIONS.get(mime)
    if mapped:
        return mapped
    if not mime.startswith("image/"):
        return DEFAULT_EXTENSION
    subtype = mime.removeprefix("image/").split("+")[0]
    cleaned = "".join(char for char in subtype if char.isalnum())
    if not cleaned or len(cleaned) > MAX_EXTENSION_LENGTH:
        return DEFAULT_EXTENSION
    if cleaned == JPEG_SUBTYPE:
        return JPEG_EXTENSION
    return cleaned


def _append_tool_calls(pieces: list[Piece], message: BaseMessage) -> None:
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list) or not tool_calls:
        return
    encoded = json.dumps(tool_calls, ensure_ascii=False, default=str)
    line = f"{TOOL_CALLS_PREFIX}{encoded}"
    if pieces and isinstance(pieces[-1], str) and not pieces[-1].endswith(LINE_BREAK):
        _push_text(pieces, LINE_BREAK + line)
        return
    _push_text(pieces, line)


def _tool_lines(tool: Any) -> list[str]:
    if not isinstance(tool, dict):
        return []
    function = tool.get("function")
    if not isinstance(function, dict):
        return []
    name = function.get("name")
    if not isinstance(name, str):
        return []
    description = function.get("description")
    if not isinstance(description, str):
        description = ""
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    schema = json.dumps(parameters, ensure_ascii=False)
    return [
        f"- name: {name}",
        f"  description: {description}",
        f"  parameters: {schema}",
    ]


def _tool_choice_rule(tool_choice: Any) -> str:
    if tool_choice is True or (
        isinstance(tool_choice, str) and tool_choice in AT_LEAST_ONE_CHOICES
    ):
        return RULE_AT_LEAST_ONE
    if tool_choice == CHOICE_NONE:
        return RULE_NONE
    name = _named_tool_choice(tool_choice)
    if name:
        return f"{RULE_NAMED_PREFIX}{name}{RULE_NAMED_SUFFIX}"
    return RULE_OPTIONAL


def _named_tool_choice(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, str):
        if tool_choice in NON_NAME_CHOICES:
            return None
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _envelope_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if "content" not in value and "tool_calls" not in value:
        return None
    return value


def _last_json_object(
    text: str, accept: Callable[[Any], dict[str, Any] | None]
) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    index = 0
    length = len(text)
    while index < length:
        brace = text.find("{", index)
        if brace < 0:
            return found
        try:
            obj, end = decoder.raw_decode(text, brace)
        except (json.JSONDecodeError, RecursionError):
            index = brace + 1
            continue
        accepted = accept(obj)
        if accepted is not None:
            found = accepted
        if end <= brace:
            index = brace + 1
            continue
        index = end
    return found


def _message_from_envelope(payload: dict[str, Any], tool_names: set[str]) -> AIMessage:
    content = _content_text(payload.get("content"))
    calls, ignored = _tool_calls_from_envelope(payload.get("tool_calls"), tool_names)
    if ignored:
        notes = LINE_BREAK.join(ignored)
        content = f"{content}{LINE_BREAK}{notes}" if content else notes
    if not calls:
        return AIMessage(content=content)
    return AIMessage(content=content, tool_calls=calls)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tool_calls_from_envelope(
    raw_calls: Any, tool_names: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw_calls, list):
        return [], []
    calls: list[dict[str, Any]] = []
    ignored: list[str] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or name not in tool_names:
            if isinstance(name, str) and name:
                ignored.append(f"{IGNORED_TOOL_PREFIX}{name}{IGNORED_TOOL_SUFFIX}")
            continue
        calls.append(
            {
                "name": name,
                "args": _tool_args(item.get("args")),
                "id": f"{TOOL_CALL_ID_PREFIX}{uuid.uuid4().hex[:TOOL_CALL_ID_HEX_LENGTH]}",
                "type": TOOL_CALL_TYPE,
            }
        )
    return calls, ignored


def _tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}
