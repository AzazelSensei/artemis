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
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artemis.desktop.actions import AxElement, AxError, Observation

SCRIPT_ENV = "ARTEMIS_DESKTOP_AX_SCRIPT"
DEFAULT_SCRIPT = Path.home() / ".claude/skills/native-desktop-use/scripts/desktop_control.sh"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_TREE_LIMIT = 400
APP_FOLDERS = (
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)
APP_SUFFIX = ".app"
SCROLL_DIRECTIONS = frozenset({"down", "up"})
SET_TEXT_FALLBACK_CODES = frozenset({"unsupported", "not_editable"})
SIGNATURE_FIELD_SEPARATOR = "\x1f"
FOCUS_PREFIX = "focus"
SCROLL_AREA_SELECTOR = {"role": "AXScrollArea"}
KNOWN_STATES = frozenset(
    {"on", "off", "mixed", "disabled", "focused", "selected", "expanded", "collapsed"}
)
DECORATIVE_ROLES = frozenset(
    {"group", "image", "text", "unknown", "splitter", "split", "row", "cell", "column"}
)

QUOTED = r'"((?:[^"\\]|\\.)*)"'
LINE_PATTERN = re.compile(r"^\s*\[(?P<ref>\d+|-)\]\s+(?P<role>\S+)(?P<rest>.*)$")
FOCUS_PATTERN = re.compile(r"^" + FOCUS_PREFIX + r"\s+(?P<line>\[(?P<ref>\d+|-)\].*)$")
LABEL_PATTERN = re.compile(r"^\s+" + QUOTED)
SUMMARY_PATTERN = re.compile(r"^\s+~" + QUOTED)
VALUE_PATTERN = re.compile(r"^\s+=(?:" + QUOTED + r"|(?P<bare>\S+))")
PLACEHOLDER_PATTERN = re.compile(r"^\s+ph=" + QUOTED)
ACTIONS_PATTERN = re.compile(r"\s+\((?P<actions>[A-Za-z0-9_,]+)\)$")
IDENTIFIER_PATTERN = re.compile(r"(?:^|\s+)#(?P<identifier>.+)$")
APP_LINE_PATTERN = re.compile(
    r"^" + QUOTED + r"(?:\s+(?P<bundle>\S+))?\s+pid:(?P<pid>\d+)(?P<flags>.*)$"
)
ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
ESCAPE_PATTERN = re.compile(r"\\(.)")


def _unescape(text: str) -> str:
    return ESCAPE_PATTERN.sub(lambda match: ESCAPES.get(match.group(1), match.group(1)), text)


@dataclass(frozen=True)
class _ParsedLine:
    ref: str
    role: str
    label: str
    value: str | None
    states: tuple[str, ...]
    identifier: str | None
    actions: tuple[str, ...]


def _consume(pattern: re.Pattern[str], rest: str) -> tuple[re.Match[str] | None, str]:
    match = pattern.match(rest)
    if match is None:
        return None, rest
    return match, rest[match.end() :]


def _parse_line(line: str) -> _ParsedLine | None:
    head = LINE_PATTERN.match(line)
    if head is None or head.group("ref") == "-":
        return None
    rest = head.group("rest")
    label_match, rest = _consume(LABEL_PATTERN, rest)
    summary_match, rest = _consume(SUMMARY_PATTERN, rest)
    value_match, rest = _consume(VALUE_PATTERN, rest)
    placeholder_match, rest = _consume(PLACEHOLDER_PATTERN, rest)
    actions: tuple[str, ...] = ()
    actions_match = ACTIONS_PATTERN.search(rest)
    if actions_match is not None:
        actions = tuple(actions_match.group("actions").split(","))
        rest = rest[: actions_match.start()]
    identifier: str | None = None
    identifier_match = IDENTIFIER_PATTERN.search(rest)
    if identifier_match is not None:
        identifier = identifier_match.group("identifier").strip()
        rest = rest[: identifier_match.start()]
    label = ""
    if label_match is not None:
        label = _unescape(label_match.group(1))
    elif summary_match is not None:
        label = _unescape(summary_match.group(1))
    elif placeholder_match is not None:
        label = _unescape(placeholder_match.group(1))
    value: str | None = None
    if value_match is not None:
        quoted = value_match.group(1)
        value = _unescape(quoted) if quoted is not None else value_match.group("bare")
    return _ParsedLine(
        ref=head.group("ref"),
        role=head.group("role"),
        label=label,
        value=value,
        states=tuple(token for token in rest.split() if token in KNOWN_STATES),
        identifier=identifier,
        actions=actions,
    )


def _is_decorative(parsed: _ParsedLine) -> bool:
    if parsed.label or parsed.value is not None or parsed.identifier is not None:
        return False
    if parsed.states or parsed.actions:
        return False
    base, _, subrole = parsed.role.partition(":")
    return not subrole and base in DECORATIVE_ROLES


def _ref_number(ref: str) -> int:
    if not (ref.isascii() and ref.isdigit()):
        raise AxError("invalid_ref", f"ref must be a positive integer: {ref!r}")
    return int(ref)


def _signature(app_name: str, window_title: str | None, elements: tuple[AxElement, ...]) -> str:
    parts = [app_name, window_title or ""]
    parts.extend(
        f"{element.role}{SIGNATURE_FIELD_SEPARATOR}{element.label}"
        f"{SIGNATURE_FIELD_SEPARATOR}{element.value or ''}"
        for element in elements
    )
    return hashlib.sha1("\n".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()


def _build_observation(response: dict[str, Any], bundle_id: str | None) -> Observation:
    lines = [str(line) for line in response.get("lines", [])]
    elements: list[AxElement] = []
    index_by_ref: dict[str, int] = {}
    focused_ref: str | None = None
    for line in lines:
        focus_match = FOCUS_PATTERN.match(line)
        if focus_match is not None:
            focused_ref = focus_match.group("ref")
            line = focus_match.group("line")
        parsed = _parse_line(line)
        if parsed is None or _is_decorative(parsed) or parsed.ref in index_by_ref:
            continue
        index = len(elements) + 1
        index_by_ref[parsed.ref] = index
        elements.append(
            AxElement(
                index=index,
                ref=parsed.ref,
                role=parsed.role,
                label=parsed.label,
                value=parsed.value,
                states=parsed.states,
                identifier=parsed.identifier,
                actions=parsed.actions,
            )
        )
    app_name = str(response.get("app") or "")
    window_title = str(response.get("window") or "") or None
    frozen = tuple(elements)
    return Observation(
        app_name=app_name,
        bundle_id=bundle_id,
        window_title=window_title,
        elements=frozen,
        focused_index=index_by_ref.get(focused_ref) if focused_ref else None,
        truncated=bool(response.get("truncated", False)),
        signature=_signature(app_name, window_title, frozen),
    )


def _resolve_script(script: Path | None) -> Path:
    if script is None:
        configured = os.environ.get(SCRIPT_ENV)
        script = Path(configured).expanduser() if configured else DEFAULT_SCRIPT
    if not script.is_file():
        raise AxError("not_found", f"desktop control script not found: {script}")
    if not os.access(script, os.X_OK):
        raise AxError("not_executable", f"desktop control script is not executable: {script}")
    return script


def _scan_installed_apps() -> list[str]:
    names: set[str] = set()
    for folder in APP_FOLDERS:
        if not folder.is_dir():
            continue
        names.update(
            entry.name.removesuffix(APP_SUFFIX)
            for entry in folder.iterdir()
            if entry.name.endswith(APP_SUFFIX)
        )
    return sorted(names)


class AxClient:
    def __init__(self, script: Path | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._script = _resolve_script(script)
        self._timeout_s = timeout_s
        self._installed: list[str] | None = None
        self._bundles: dict[str, str] = {}

    async def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({**request, "format": "json"}, ensure_ascii=False).encode("utf-8")
        try:
            process = await asyncio.create_subprocess_exec(
                str(self._script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AxError("exec_failed", f"{self._script}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=self._timeout_s
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise AxError("timeout", f"{request.get('op')} exceeded {self._timeout_s}s") from exc
        return self._decode(stdout, stderr)

    @staticmethod
    def _decode(stdout: bytes, stderr: bytes) -> dict[str, Any]:
        text = stdout.decode("utf-8", errors="replace").strip()
        try:
            response = json.loads(text)
        except json.JSONDecodeError as exc:
            detail = text or stderr.decode("utf-8", errors="replace").strip()
            raise AxError("bad_response", detail[:500]) from exc
        if not isinstance(response, dict):
            raise AxError("bad_response", text[:500])
        if not response.get("ok", False):
            raise AxError(str(response.get("error") or "unknown"), response.get("message"))
        return response

    async def status(self) -> dict[str, Any]:
        return await self._call({"op": "status"})

    async def _running_apps(self) -> list[tuple[str, str | None, bool]]:
        response = await self._call({"op": "apps"})
        apps: list[tuple[str, str | None, bool]] = []
        for line in response.get("lines", []):
            match = APP_LINE_PATTERN.match(str(line))
            if match is None:
                continue
            bundle = match.group("bundle")
            apps.append(
                (
                    _unescape(match.group(1)),
                    None if bundle in (None, "-") else bundle,
                    "active" in match.group("flags").split(),
                )
            )
        return apps

    async def frontmost(self) -> tuple[str, str | None]:
        apps = await self._running_apps()
        for name, bundle, is_active in apps:
            if is_active:
                return name, bundle
        status = await self.status()
        name = str(status.get("frontmost") or "")
        if not name:
            raise AxError("not_found", "no frontmost application")
        bundle = next((bundle for app, bundle, _ in apps if app == name), None)
        return name, bundle

    async def _bundle_for(self, app: str) -> str | None:
        if "." in app and " " not in app:
            return app
        cached = self._bundles.get(app)
        if cached is not None:
            return cached
        for name, bundle, _ in await self._running_apps():
            if bundle is not None:
                self._bundles[name] = bundle
        return self._bundles.get(app)

    async def observe(self, app: str | None = None, limit: int = DEFAULT_TREE_LIMIT) -> Observation:
        if app is None:
            name, bundle = await self.frontmost()
            target = bundle or name
        else:
            target = app
            bundle = None
        response = await self._call({"op": "tree", "app": target, "limit": limit})
        if bundle is None:
            bundle = await self._bundle_for(str(response.get("app") or target))
        return _build_observation(response, bundle)

    async def click(self, app: str, ref: str) -> dict[str, Any]:
        return await self._call({"op": "click", "app": app, "ref": _ref_number(ref)})

    async def set_text(self, app: str, ref: str, text: str) -> dict[str, Any]:
        try:
            return await self._call(
                {"op": "set_text", "app": app, "ref": _ref_number(ref), "text": text}
            )
        except AxError as error:
            if error.code not in SET_TEXT_FALLBACK_CODES:
                raise
        return await self._call(
            {
                "op": "type_text",
                "app": app,
                "ref": _ref_number(ref),
                "text": text,
                "mode": "replace",
            }
        )

    async def key(self, app: str, key: str) -> dict[str, Any]:
        return await self._call({"op": "key", "app": app, "keys": key})

    async def scroll(self, app: str, ref: str | None, direction: str) -> dict[str, Any]:
        if direction not in SCROLL_DIRECTIONS:
            raise AxError("invalid_direction", f"unsupported scroll direction: {direction}")
        target = ref if ref is not None else await self._first_scroll_ref(app)
        return await self._call(
            {"op": "scroll", "app": app, "ref": _ref_number(target), "direction": direction}
        )

    async def _first_scroll_ref(self, app: str) -> str:
        response = await self._call({"op": "find", "app": app, "selector": SCROLL_AREA_SELECTOR})
        for line in response.get("lines", []):
            parsed = _parse_line(str(line))
            if parsed is not None:
                return parsed.ref
        raise AxError("no_scroll_target", f"no scroll area found in {app}")

    async def launch(self, app: str) -> dict[str, Any]:
        return await self._call({"op": "launch", "app": app})

    async def installed_apps(self) -> list[str]:
        if self._installed is None:
            self._installed = await asyncio.to_thread(_scan_installed_apps)
        running = [name for name, _, _ in await self._running_apps()]
        return sorted({*self._installed, *running})
