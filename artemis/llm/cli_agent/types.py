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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class CliBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextSegment:
    text: str


@dataclass(frozen=True)
class ImageSegment:
    path: Path
    media_type: str
    base64_data: str


Segment = TextSegment | ImageSegment


@dataclass
class CliRequest:
    system_prompt: str
    segments: list[Segment]
    workdir: Path
    timeout_seconds: float
    model: str | None = None
    effort: str | None = None
    json_schema: dict[str, Any] | None = None


@dataclass
class CliResponse:
    text: str
    structured: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class CliBackend(Protocol):
    name: str

    async def run(self, request: CliRequest) -> CliResponse: ...
