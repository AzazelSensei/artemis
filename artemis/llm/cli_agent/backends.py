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

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Any

from artemis.llm.cli_agent.types import (
    CliBackend,
    CliBackendError,
    CliRequest,
    CliResponse,
    ImageSegment,
    Segment,
    TextSegment,
)

_STDERR_TAIL = 800
_CLAUDE_QUIET_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
}
_HELP_TIMEOUT_SECONDS = 30.0
_KILL_WAIT_SECONDS = 5.0
_CLAUDE_SETTINGS = '{"disableAllHooks":true}'
_CODEX_LAST_NAME = "last.txt"
_IMAGE_READ_NOTE = (
    "Read the attached image files listed as [IMAGE k: file] from the workspace before answering."
)
_SCHEMA_REPLY_PREFIX = "Reply with only JSON matching this schema: "
_CLAUDE_BIN_ENV = "ARTEMIS_CLAUDE_BIN"
_CODEX_BIN_ENV = "ARTEMIS_CODEX_BIN"
_CURSOR_BIN_ENV = "ARTEMIS_CURSOR_BIN"
_GROK_BIN_ENV = "ARTEMIS_GROK_BIN"
_CLAUDE_DROPPED_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_CODEX_DROPPED_KEYS = ("OPENAI_API_KEY", "CODEX_API_KEY")
_CURSOR_DROPPED_KEYS = ("CURSOR_API_KEY",)
_GROK_DROPPED_KEYS = ("XAI_API_KEY", "GROK_CODE_XAI_API_KEY")
_CURSOR_PROGRAMS = ("cursor-agent",)
_FORBIDDEN_ARGS = frozenset({"--bare", "--force", "-f", "--yolo", "--worktree", "-w"})
_END_OF_OPTIONS = "--"
_GROK_PROMPT_NAME = "prompt.txt"
_CLAUDE_SUCCESS_SUBTYPE = "success"
_ERROR_SUBTYPE_PREFIX = "error"
_CODEX_TOTAL_RE = re.compile(r"tokens used\s*:?\s*([\d.,]+)", re.IGNORECASE)
_FORBIDDEN_PREFIX = "--dangerously-"
_TEXT_KEYS = ("result", "text", "output", "response", "message", "content")
_STRUCTURED_KEYS = ("structured_output", "structured", "structuredOutput")
_LONG_INPUT_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
_SHORT_INPUT_KEYS = ("input", "cache_creation", "cache_read")
_CODEX_INPUT_RE = re.compile(
    r'"(?:input_tokens|inputTokens)"\s*:\s*(\d+)'
    r"|(?:input[_ -]*tokens?|tokens?[_ -]*input)\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)
_CODEX_OUTPUT_RE = re.compile(
    r'"(?:output_tokens|outputTokens)"\s*:\s*(\d+)'
    r"|(?:output[_ -]*tokens?|tokens?[_ -]*output)\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)
_CODEX_IO_RE = re.compile(
    r"input\s*=\s*(\d+).{0,80}?output\s*=\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_HELP_CACHE: dict[tuple[str, ...], bool] = {}


def create_backend(provider: str) -> CliBackend:
    if provider == "claude_code":
        return ClaudeCodeBackend()
    if provider == "codex_cli":
        return CodexBackend()
    if provider == "cursor_cli":
        return CursorBackend()
    if provider == "grok_cli":
        return GrokBackend()
    raise ValueError(f"unknown cli provider: {provider}")


class ClaudeCodeBackend:
    name = "claude_code"

    async def run(self, request: CliRequest) -> CliResponse:
        workdir = _require_workdir(request.workdir)
        _require_timeout(request.timeout_seconds)
        binary = _resolve_bin(_CLAUDE_BIN_ENV, "claude")
        env = _claude_env()
        options: list[str] = []
        model = _blank_to_none(request.model)
        if model is not None:
            options.extend(["--model", model])
        effort = _blank_to_none(request.effort)
        if effort is not None and await _help_contains(binary, [], "--effort", workdir, env):
            options.extend(["--effort", effort])
        if request.json_schema is not None:
            options.extend(["--json-schema", _dump_schema(request.json_schema)])
        argv = [*_claude_base_argv(binary, request.system_prompt), *options]
        _assert_argv_allowed(argv)
        stdout, _stderr = await _run_process(
            argv,
            workdir,
            request.timeout_seconds,
            stdin_bytes=_claude_stdin(_claude_content(request.segments)),
            env=env,
        )
        return _claude_response(stdout)


class CodexBackend:
    name = "codex_cli"

    async def run(self, request: CliRequest) -> CliResponse:
        workdir = _require_workdir(request.workdir)
        _require_timeout(request.timeout_seconds)
        binary = _resolve_bin(_CODEX_BIN_ENV, "codex")
        env = _env_without(_CODEX_DROPPED_KEYS)
        prompt, images = _render_prompt(request.system_prompt, request.segments)
        output_path = workdir / _CODEX_LAST_NAME
        output_path.unlink(missing_ok=True)
        argv = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-o",
            str(output_path),
        ]
        model = _blank_to_none(request.model)
        if model is not None:
            argv.extend(["-m", model])
        effort = _blank_to_none(request.effort)
        if effort is not None:
            argv.extend(["-c", f'model_reasoning_effort="{effort}"'])
        for image in images:
            argv.extend(["-i", str(image)])
        if request.json_schema is not None:
            prompt = f"{prompt}\n\n{_SCHEMA_REPLY_PREFIX}{_dump_schema(request.json_schema)}"
        argv.extend(["--", prompt])
        _assert_argv_allowed(argv)
        _stdout, stderr = await _run_process(
            argv,
            workdir,
            request.timeout_seconds,
            env=env,
        )
        if not output_path.is_file():
            raise CliBackendError("codex did not write last.txt")
        try:
            text = output_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CliBackendError(f"failed to read last.txt: {exc}") from exc
        if not text.strip():
            raise CliBackendError("codex wrote an empty last.txt")
        input_tokens, output_tokens = _codex_tokens(stderr.decode("utf-8", errors="replace"))
        return CliResponse(
            text=text,
            structured=_as_object(text.strip()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw={"stderr": stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:]},
        )


class CursorBackend:
    name = "cursor_cli"

    async def run(self, request: CliRequest) -> CliResponse:
        workdir = _require_workdir(request.workdir)
        _require_timeout(request.timeout_seconds)
        binary = _resolve_cursor_bin()
        env = _env_without(_CURSOR_DROPPED_KEYS)
        prompt = _prompt_with_workspace_images(request)
        argv = [
            binary,
            "-p",
            "--trust",
            "--mode",
            "ask",
            "--workspace",
            str(workdir),
            "--output-format",
            "json",
        ]
        model = _blank_to_none(request.model)
        if model is not None:
            argv.extend(["--model", model])
        argv.extend([_END_OF_OPTIONS, prompt])
        _assert_argv_allowed(argv)
        stdout, _stderr = await _run_process(
            argv,
            workdir,
            request.timeout_seconds,
            env=env,
        )
        return _json_cli_response(stdout)


class GrokBackend:
    name = "grok_cli"

    async def run(self, request: CliRequest) -> CliResponse:
        workdir = _require_workdir(request.workdir)
        _require_timeout(request.timeout_seconds)
        binary = _resolve_bin(_GROK_BIN_ENV, "grok")
        env = _env_without(_GROK_DROPPED_KEYS)
        prompt = _prompt_with_workspace_images(request, include_schema=False)
        prompt_path = workdir / _GROK_PROMPT_NAME
        try:
            prompt_path.write_text(prompt, encoding="utf-8")
        except OSError as exc:
            raise CliBackendError(f"failed to write {prompt_path.name}: {exc}") from exc
        argv = [binary, "--prompt-file", str(prompt_path)]
        model = _blank_to_none(request.model)
        if model is not None:
            argv.extend(["-m", model])
        if request.json_schema is not None:
            argv.extend(["--json-schema", _dump_schema(request.json_schema)])
        else:
            argv.extend(["--output-format", "json"])
        _assert_argv_allowed(argv)
        stdout, _stderr = await _run_process(
            argv,
            workdir,
            request.timeout_seconds,
            env=env,
        )
        return _json_cli_response(stdout)


def _require_workdir(workdir: Path) -> Path:
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    if not workdir.is_dir():
        raise CliBackendError(f"workdir does not exist: {workdir}")
    return workdir


def _require_timeout(timeout: float) -> None:
    if timeout <= 0:
        raise CliBackendError(f"invalid timeout: {timeout}")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _lookup_program(program: str) -> str | None:
    found = shutil.which(program)
    if found:
        return found
    for root in (Path.home() / ".local" / "bin", Path("/opt/homebrew/bin")):
        candidate = root / program
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_bin(env_key: str, program: str) -> str:
    override = os.environ.get(env_key, "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(override)
        if found:
            return found
        raise CliBackendError(f"{program} binary not found at {override}; check {env_key}")
    found = _lookup_program(program)
    if not found:
        raise CliBackendError(f"{program} binary not found; set {env_key}")
    return found


def _resolve_cursor_bin() -> str:
    override = os.environ.get(_CURSOR_BIN_ENV, "").strip()
    if override:
        return _resolve_bin(_CURSOR_BIN_ENV, "cursor")
    for program in _CURSOR_PROGRAMS:
        found = _lookup_program(program)
        if found:
            return found
    raise CliBackendError(f"cursor binary not found; set {_CURSOR_BIN_ENV}")


def _env_without(keys: tuple[str, ...]) -> dict[str, str]:
    env = dict(os.environ)
    for key in keys:
        env.pop(key, None)
    return env


def _assert_argv_allowed(argv: list[str]) -> None:
    for arg in argv:
        if arg == _END_OF_OPTIONS:
            return
        flag = arg.split("=", 1)[0]
        if flag in _FORBIDDEN_ARGS or flag.startswith(_FORBIDDEN_PREFIX):
            raise CliBackendError(f"refusing forbidden flag: {arg}")


def _dump_schema(schema: dict[str, Any]) -> str:
    try:
        return json.dumps(schema)
    except TypeError as exc:
        raise CliBackendError(f"json_schema is not serializable: {exc}") from exc


def _require_image(segment: ImageSegment) -> Path:
    if not segment.path.is_file():
        raise CliBackendError(f"image file not found: {segment.path}")
    return segment.path


def _claude_base_argv(binary: str, system_prompt: str) -> list[str]:
    return [
        binary,
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--settings",
        _CLAUDE_SETTINGS,
        "--system-prompt",
        system_prompt,
    ]


def _claude_stdin(content: list[dict[str, Any]]) -> bytes:
    payload = {"type": "user", "message": {"role": "user", "content": content}}
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _claude_content(segments: list[Segment]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, TextSegment):
            content.append({"type": "text", "text": segment.text})
            continue
        if isinstance(segment, ImageSegment):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": segment.media_type,
                        "data": segment.base64_data,
                    },
                }
            )
            continue
        raise CliBackendError(f"unsupported segment: {type(segment).__name__}")
    return content


def _render_prompt(system_prompt: str, segments: list[Segment]) -> tuple[str, list[Path]]:
    chunks: list[str] = []
    images: list[Path] = []
    for segment in segments:
        if isinstance(segment, TextSegment):
            chunks.append(segment.text)
            continue
        if isinstance(segment, ImageSegment):
            path = _require_image(segment)
            images.append(path)
            chunks.append(f"[IMAGE {len(images)}: {path.name}]")
            continue
        raise CliBackendError(f"unsupported segment: {type(segment).__name__}")
    return _join_prompt(system_prompt, chunks), images


def _prompt_with_workspace_images(request: CliRequest, *, include_schema: bool = True) -> str:
    chunks: list[str] = []
    image_count = 0
    for segment in request.segments:
        if isinstance(segment, TextSegment):
            chunks.append(segment.text)
            continue
        if isinstance(segment, ImageSegment):
            image_count += 1
            placed = _place_image(segment, request.workdir, image_count)
            chunks.append(f"[IMAGE {image_count}: {placed.name}]")
            continue
        raise CliBackendError(f"unsupported segment: {type(segment).__name__}")
    prompt = _join_prompt(request.system_prompt, chunks)
    if image_count:
        prompt = f"{prompt}\n\n{_IMAGE_READ_NOTE}"
    if include_schema and request.json_schema is not None:
        prompt = f"{prompt}\n\n{_SCHEMA_REPLY_PREFIX}{_dump_schema(request.json_schema)}"
    return prompt


def _place_image(segment: ImageSegment, workdir: Path, index: int) -> Path:
    path = _require_image(segment)
    root = workdir.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        destination = root / f"img_{index:03d}{path.suffix.lower()}"
        try:
            shutil.copyfile(resolved, destination)
        except OSError as exc:
            raise CliBackendError(f"failed to copy image: {exc}") from exc
        return destination
    return resolved


def _join_prompt(system_prompt: str, chunks: list[str]) -> str:
    body = "\n".join(chunks)
    if system_prompt and body:
        return f"{system_prompt}\n\n{body}"
    if system_prompt:
        return system_prompt
    return body


def _seconds_label(timeout: float) -> str:
    if float(timeout).is_integer():
        return str(int(timeout))
    return str(timeout)


def _program_name(argv: list[str]) -> str:
    if not argv:
        return "cli"
    return Path(argv[0]).name


async def _run_process(
    argv: list[str],
    cwd: Path,
    timeout: float,
    stdin_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bytes, bytes]:
    return_code, stdout, stderr = await _capture_process(
        argv,
        cwd,
        timeout,
        stdin_bytes=stdin_bytes,
        env=env,
    )
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:]
        if not detail.strip():
            detail = stdout.decode("utf-8", errors="replace")[-_STDERR_TAIL:]
        raise CliBackendError(f"{_program_name(argv)} exited {return_code}: {detail}")
    return stdout, stderr


async def _capture_process(
    argv: list[str],
    cwd: Path,
    timeout: float,
    stdin_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    if not argv:
        raise CliBackendError("empty argv")
    stdin_mode = asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=stdin_mode,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            _communicate(process, stdin_bytes),
            timeout=timeout,
        )
    except TimeoutError:
        await _kill_and_wait(process)
        raise CliBackendError(
            f"{_program_name(argv)} timed out after {_seconds_label(timeout)}s"
        ) from None
    return_code = process.returncode
    if return_code is None:
        raise CliBackendError(f"{_program_name(argv)} exited without a return code")
    return return_code, stdout or b"", stderr or b""


async def _kill_and_wait(process: asyncio.subprocess.Process) -> None:
    _kill_process_tree(process)
    try:
        await asyncio.wait_for(process.wait(), timeout=_KILL_WAIT_SECONDS)
    except TimeoutError:
        process.kill()


def _claude_env() -> dict[str, str]:
    env = _env_without(_CLAUDE_DROPPED_KEYS)
    for key, value in _CLAUDE_QUIET_ENV.items():
        env.setdefault(key, value)
    return env


async def _communicate(
    process: asyncio.subprocess.Process,
    stdin_bytes: bytes | None,
) -> tuple[bytes, bytes]:
    if stdin_bytes is None:
        return await process.communicate()
    return await process.communicate(stdin_bytes)


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.pid is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        process.kill()


async def _help_contains(
    binary: str,
    prefix: list[str],
    flag: str,
    cwd: Path,
    env: dict[str, str],
) -> bool:
    argv = (binary, *prefix, "--help")
    cache_key = (*argv, flag)
    cached = _HELP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        _return_code, stdout, stderr = await _capture_process(
            list(argv),
            cwd,
            _HELP_TIMEOUT_SECONDS,
            env=env,
        )
    except CliBackendError as exc:
        if "timed out after" not in str(exc):
            raise
        _HELP_CACHE[cache_key] = False
        return False
    text = (
        stdout.decode("utf-8", errors="replace") + "\n" + stderr.decode("utf-8", errors="replace")
    )
    found = flag in text
    _HELP_CACHE[cache_key] = found
    return found


def _claude_response(stdout: bytes) -> CliResponse:
    result_event: dict[str, Any] | None = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result_event = event
    if result_event is None:
        raise CliBackendError("claude stream-json produced no result event")
    if result_event.get("is_error") is True:
        raise CliBackendError(f"claude result error: {result_event.get('result')}")
    subtype = result_event.get("subtype")
    if subtype is not None and subtype != _CLAUDE_SUCCESS_SUBTYPE:
        raise CliBackendError(f"claude result subtype: {subtype}")
    result_text = result_event.get("result")
    text = result_text if isinstance(result_text, str) else ""
    structured = None
    if "structured_output" in result_event:
        structured = _as_object(result_event.get("structured_output"))
    usage = result_event.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_tokens, output_tokens = _usage_tokens(usage)
    model_usage = result_event.get("modelUsage")
    model_name = None
    if isinstance(model_usage, dict) and model_usage:
        model_name = next(iter(model_usage))
    if not text.strip() and structured is None:
        raise CliBackendError("claude returned an empty result")
    return CliResponse(
        text=text,
        structured=structured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=model_name if isinstance(model_name, str) else None,
        raw=result_event,
    )


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    if any(key in usage for key in _LONG_INPUT_KEYS):
        incoming = sum(_as_int(usage.get(key)) for key in _LONG_INPUT_KEYS)
    else:
        incoming = sum(_as_int(usage.get(key)) for key in _SHORT_INPUT_KEYS)
    if "output_tokens" in usage:
        outgoing = _as_int(usage.get("output_tokens"))
    else:
        outgoing = _as_int(usage.get("output"))
    return incoming, outgoing


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _codex_tokens(stderr: str) -> tuple[int, int]:
    paired = list(_CODEX_IO_RE.finditer(stderr))
    if paired:
        last = paired[-1]
        return int(last.group(1)), int(last.group(2))
    incoming = _last_match_int(_CODEX_INPUT_RE, stderr)
    outgoing = _last_match_int(_CODEX_OUTPUT_RE, stderr)
    if incoming is not None or outgoing is not None:
        return incoming or 0, outgoing or 0
    totals = _CODEX_TOTAL_RE.findall(stderr)
    if not totals:
        return 0, 0
    digits = "".join(char for char in totals[-1] if char.isdigit())
    return (int(digits) if digits else 0), 0


def _last_match_int(pattern: re.Pattern[str], text: str) -> int | None:
    found: int | None = None
    for match in pattern.finditer(text):
        for group in match.groups():
            if group is not None and group.isdigit():
                found = int(group)
                break
    return found


def _json_cli_response(stdout: bytes) -> CliResponse:
    raw_text = stdout.decode("utf-8", errors="replace").strip()
    if not raw_text:
        raise CliBackendError("cli returned empty output")
    payload = _prefer_result(_parse_cli_json(raw_text))
    if not isinstance(payload, dict):
        raise CliBackendError(f"cli output is not a json object: {raw_text[-_STDERR_TAIL:]}")
    subtype = payload.get("subtype")
    if isinstance(subtype, str) and subtype.startswith(_ERROR_SUBTYPE_PREFIX):
        raise CliBackendError(f"cli result subtype: {subtype}")
    if payload.get("is_error") is True:
        detail = payload.get("result")
        if not isinstance(detail, str):
            detail = payload.get("text")
        raise CliBackendError(f"cli result error: {detail}")
    structured = _structured_field(payload)
    text = _first_text(payload)
    if text is None and structured is None:
        raise CliBackendError(f"cli output has no result text: {raw_text[-_STDERR_TAIL:]}")
    if text is None:
        text = json.dumps(structured, ensure_ascii=False)
    input_tokens, output_tokens = _payload_tokens(payload)
    return CliResponse(
        text=text,
        structured=structured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_name=_model_name(payload),
        raw=payload,
    )


def _parse_cli_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    events: list[Any] = []
    for line in raw.splitlines():
        piece = line.strip()
        if not piece:
            continue
        try:
            events.append(json.loads(piece))
        except json.JSONDecodeError:
            continue
    if len(events) == 1:
        return events[0]
    if events:
        return events
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise CliBackendError(f"cli output is not json: {raw[-_STDERR_TAIL:]}") from exc
    raise CliBackendError(f"cli output is not json: {raw[-_STDERR_TAIL:]}")


def _prefer_result(payload: Any) -> Any:
    if not isinstance(payload, list):
        return payload
    for item in reversed(payload):
        if isinstance(item, dict) and item.get("type") == "result":
            return item
    parts: list[str] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        data = item.get("data")
        if isinstance(data, str):
            parts.append(data)
    if parts:
        merged: dict[str, Any] = {"text": "".join(parts)}
        for item in reversed(payload):
            if not isinstance(item, dict):
                continue
            for key in _STRUCTURED_KEYS:
                if key in item:
                    merged[key] = item[key]
            if isinstance(item.get("usage"), dict):
                merged["usage"] = item["usage"]
        return merged
    for item in reversed(payload):
        if isinstance(item, dict):
            return item
    return payload


def _structured_field(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in _STRUCTURED_KEYS:
        if key not in payload:
            continue
        parsed = _as_object(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _as_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _first_text(payload: dict[str, Any]) -> str | None:
    for key in _TEXT_KEYS:
        if key not in payload:
            continue
        coerced = _coerce_text(payload.get(key))
        if coerced is not None and coerced.strip():
            return coerced
    return None


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        bits: list[str] = []
        for item in value:
            if isinstance(item, str):
                bits.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    bits.append(text)
        if bits:
            return "\n".join(bits)
        return None
    if isinstance(value, dict):
        for key in ("text", "content", "result"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
    return None


def _payload_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return _usage_tokens(usage)
    return _as_int(payload.get("input_tokens")), _as_int(payload.get("output_tokens"))


def _model_name(payload: dict[str, Any]) -> str | None:
    for key in ("model", "model_name", "modelName"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        first = next(iter(model_usage))
        if isinstance(first, str):
            return first
    return None
