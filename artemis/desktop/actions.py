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

from dataclasses import dataclass
from enum import StrEnum

EDITABLE_ROLES = frozenset({"textfield", "textarea", "searchfield", "combobox"})
EDITABLE_SUBROLES = frozenset({"search"})
SETTABLE_STATE = "settable"
DISPLAY_ESCAPES = (("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"), ("\r", "\\r"), ("\t", "\\t"))


def escape_display(text: str) -> str:
    for raw, escaped in DISPLAY_ESCAPES:
        text = text.replace(raw, escaped)
    return text


class DesktopOp(StrEnum):
    CLICK = "click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL_DOWN = "scroll_down"
    SCROLL_UP = "scroll_up"
    OPEN_APP = "open_app"
    WAIT = "wait"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class DesktopAction:
    op: DesktopOp
    index: int | None = None
    text: str | None = None
    key: str | None = None
    app: str | None = None
    explanation: str | None = None
    source: str = "jev"
    confidence: float | None = None


@dataclass(frozen=True)
class AxElement:
    index: int
    ref: str
    role: str
    label: str
    value: str | None
    states: tuple[str, ...]
    identifier: str | None
    actions: tuple[str, ...]

    @property
    def is_editable(self) -> bool:
        base, _, subrole = self.role.partition(":")
        if base in EDITABLE_ROLES or subrole in EDITABLE_SUBROLES:
            return True
        return SETTABLE_STATE in self.states

    @property
    def display(self) -> str:
        parts = [f"[{self.index}]", self.role, f'"{escape_display(self.label)}"']
        if self.value is not None:
            parts.append(f'="{escape_display(self.value)}"')
        parts.extend(self.states)
        return " ".join(parts)


@dataclass(frozen=True)
class Observation:
    app_name: str
    bundle_id: str | None
    window_title: str | None
    elements: tuple[AxElement, ...]
    focused_index: int | None
    truncated: bool
    signature: str

    def element(self, index: int) -> AxElement | None:
        if index < 1 or index > len(self.elements):
            return None
        return self.elements[index - 1]

    @property
    def text_list(self) -> str:
        return "\n".join(element.display for element in self.elements)


class AxError(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or code
