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

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from artemis.llm.cli_agent.messages import (
    build_envelope_schema,
    build_request_parts,
    parse_envelope,
    render_tool_instructions,
)
from artemis.llm.cli_agent.types import CliRequest, CliResponse, ImageSegment, Segment, TextSegment

TEMP_DIR_PREFIX = "artemis-cli-"
DEFAULT_TIMEOUT_SECONDS = 120.0
THREAD_POOL_WORKERS = 1
INSTRUCTION_SEPARATOR = "\n\n"
OMITTED_SCREENSHOT_TEXT = "[earlier screenshot omitted]"
RETRY_ENVELOPE_TEXT = (
    "Your previous reply was not a valid JSON envelope with a tool call. "
    "Reply again with only the JSON envelope."
)
CHOICE_ANY = "any"
CHOICE_REQUIRED = "required"
CHOICE_AUTO = "auto"
CHOICE_NONE = "none"
AT_LEAST_ONE_CHOICES = frozenset({CHOICE_ANY, CHOICE_REQUIRED})
NON_NAME_CHOICES = frozenset({CHOICE_ANY, CHOICE_REQUIRED, CHOICE_AUTO, CHOICE_NONE})


class CliChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Any
    model_name: str | None = None
    effort: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    keep_last_images: int | None = None

    @property
    def _llm_type(self) -> str:
        return f"cli-{self.backend.name}"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as directory:
            workdir = Path(directory)
            system_prompt, segments = build_request_parts(messages, workdir)
            segments = _limit_images(segments, self.keep_last_images)
            json_schema = None
            names: list[str] = []
            if tools:
                names = _tool_names(tools)
                system_prompt += INSTRUCTION_SEPARATOR + render_tool_instructions(
                    tools, tool_choice
                )
                json_schema = build_envelope_schema(names)
            request = _make_request(
                self,
                system_prompt=system_prompt,
                segments=segments,
                workdir=workdir,
                json_schema=json_schema,
            )
            response = await self.backend.run(request)
            message = _message_from_response(response, tools, set(names))
            if tools and _tool_choice_requires_call(tool_choice) and not message.tool_calls:
                retry = _make_request(
                    self,
                    system_prompt=system_prompt,
                    segments=[*segments, TextSegment(RETRY_ENVELOPE_TEXT)],
                    workdir=workdir,
                    json_schema=json_schema,
                )
                response = await self.backend.run(retry)
                message = _message_from_response(response, tools, set(names))
            _attach_metadata(message, response, self)
            return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not _event_loop_running():
            return asyncio.run(
                self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        def _invoke() -> ChatResult:
            return asyncio.run(
                self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
            return executor.submit(_invoke).result()


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _make_request(
    model: CliChatModel,
    *,
    system_prompt: str,
    segments: list[Segment],
    workdir: Path,
    json_schema: dict[str, Any] | None,
) -> CliRequest:
    return CliRequest(
        system_prompt=system_prompt,
        segments=segments,
        workdir=workdir,
        timeout_seconds=model.timeout_seconds,
        model=model.model_name,
        effort=model.effort,
        json_schema=json_schema,
    )


def _message_from_response(
    response: CliResponse, tools: list[dict] | None, tool_names: set[str]
) -> AIMessage:
    if not tools:
        return AIMessage(content=response.text)
    return parse_envelope(response.text, response.structured, tool_names)


def _attach_metadata(message: AIMessage, response: CliResponse, model: CliChatModel) -> None:
    message.usage_metadata = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.input_tokens + response.output_tokens,
    }
    message.response_metadata = {
        "model_name": response.model_name or model.model_name,
        "backend": model.backend.name,
    }


def _tool_names(tools: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        name = _one_tool_name(tool)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _one_tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    if isinstance(name, str):
        return name
    return None


def _tool_choice_requires_call(tool_choice: Any) -> bool:
    if tool_choice is True or (
        isinstance(tool_choice, str) and tool_choice in AT_LEAST_ONE_CHOICES
    ):
        return True
    if tool_choice is None or tool_choice == CHOICE_AUTO or tool_choice == CHOICE_NONE:
        return False
    if isinstance(tool_choice, str):
        return tool_choice not in NON_NAME_CHOICES
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        return isinstance(function, dict) and isinstance(function.get("name"), str)
    return False


def _limit_images(segments: list[Segment], keep_last: int | None) -> list[Segment]:
    if keep_last is None:
        return segments
    keep = keep_last if keep_last > 0 else 0
    image_indexes = [
        index for index, segment in enumerate(segments) if isinstance(segment, ImageSegment)
    ]
    if len(image_indexes) <= keep:
        return segments
    drop = set(image_indexes if keep == 0 else image_indexes[:-keep])
    merged: list[Segment] = []
    for index, segment in enumerate(segments):
        piece: Segment = TextSegment(OMITTED_SCREENSHOT_TEXT) if index in drop else segment
        if isinstance(piece, TextSegment) and merged and isinstance(merged[-1], TextSegment):
            merged[-1] = TextSegment(_join_text(merged[-1].text, piece.text))
            continue
        merged.append(piece)
    return merged


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith("\n") or right.startswith("\n"):
        return left + right
    return left + "\n" + right
