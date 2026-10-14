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
import math
import os
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

JEV_MODEL = "jev-1.13.0"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_TIMEOUT_S = 10.0
MAX_CHOICE_OPTIONS = 255
PROBABILITY_SUM_TOLERANCE = 0.025
RETRY_DELAY_S = 0.3
MAX_RETRIES = 1
QUESTION_TYPE_CHOICE = "choice"
QUESTION_TYPE_NOUL = "noul"
SUPPORTED_QUESTION_TYPES = frozenset({QUESTION_TYPE_CHOICE, QUESTION_TYPE_NOUL})
AUTH_STATUS_CODES = frozenset({401, 403})
REQUEST_STATUS_CODES = frozenset({400, 404, 413, 422})
RATE_LIMIT_STATUS = 429
OVERLOADED_STATUS = 529
TRANSIENT_SERVER_STATUS = 499
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 599
SUCCESS_STATUS = 200
ERROR_BODY_PREVIEW_CHARS = 200


class JevError(RuntimeError):
    pass


class JevConfigError(JevError):
    pass


class JevAuthError(JevError):
    pass


class JevRequestError(JevError):
    pass


class JevRateLimitError(JevError):
    pass


class JevServerError(JevError):
    pass


class JevTimeoutError(JevError):
    pass


class JevResponseError(JevError):
    pass


@dataclass(frozen=True)
class JevChoice:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class JevNoul:
    value: float


@dataclass(frozen=True)
class JevResult:
    answers: dict[str, JevChoice | JevNoul]
    input_tokens: int
    latency_s: float
    model: str


def _is_unit_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _is_retryable_status(status: int) -> bool:
    if status in (RATE_LIMIT_STATUS, OVERLOADED_STATUS, TRANSIENT_SERVER_STATUS):
        return True
    return SERVER_ERROR_MIN <= status <= SERVER_ERROR_MAX


def _error_for_status(status: int, body_preview: str) -> JevError:
    message = f"Jev API returned HTTP {status}: {body_preview}"
    if status in AUTH_STATUS_CODES:
        return JevAuthError(f"Jev API authentication failed (HTTP {status})")
    if status == RATE_LIMIT_STATUS:
        return JevRateLimitError(message)
    if (
        status in (OVERLOADED_STATUS, TRANSIENT_SERVER_STATUS)
        or SERVER_ERROR_MIN <= status <= SERVER_ERROR_MAX
    ):
        return JevServerError(message)
    if status in REQUEST_STATUS_CODES:
        return JevRequestError(message)
    return JevError(message)


def _validate_questions(questions: dict[str, dict[str, Any]]) -> None:
    if not questions:
        raise JevRequestError("At least one question is required")
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise JevRequestError(f"Question '{name}' must be an object")
        question_type = question.get("type")
        if question_type not in SUPPORTED_QUESTION_TYPES:
            raise JevRequestError(f"Question '{name}' has unsupported type {question_type!r}")
        if question_type != QUESTION_TYPE_CHOICE:
            continue
        criteria = question.get("criteria")
        if not isinstance(criteria, dict):
            raise JevRequestError(f"Choice question '{name}' requires a criteria object")
        option_count = len(criteria)
        if option_count < 1 or option_count > MAX_CHOICE_OPTIONS:
            raise JevRequestError(
                f"Choice question '{name}' has {option_count} options; "
                f"allowed range is 1..{MAX_CHOICE_OPTIONS}"
            )


def _parse_choice(name: str, answer: dict[str, Any], criteria: dict[str, Any]) -> JevChoice:
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in criteria:
        raise JevResponseError(f"answers.{name}.choice is not one of the criteria keys")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevResponseError(f"answers.{name}.probabilities must be an object")
    if set(probabilities) != set(criteria):
        raise JevResponseError(f"answers.{name}.probabilities keys do not match criteria keys")
    for key, probability in probabilities.items():
        if not _is_unit_number(probability):
            raise JevResponseError(f"answers.{name}.probabilities.{key} is not a number in [0, 1]")
    total = math.fsum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise JevResponseError(f"answers.{name}.probabilities sum to {total:.4f}, expected 1")
    if probabilities[choice] < max(probabilities.values()):
        raise JevResponseError(f"answers.{name}.choice is not the highest-probability option")
    confidence = answer.get("confidence")
    if not _is_unit_number(confidence):
        raise JevResponseError(f"answers.{name}.confidence is not a number in [0, 1]")
    return JevChoice(
        choice=choice,
        probabilities={key: float(value) for key, value in probabilities.items()},
        confidence=float(confidence),
    )


def _parse_noul(name: str, answer: dict[str, Any]) -> JevNoul:
    value = answer.get("noul")
    if not _is_unit_number(value):
        raise JevResponseError(f"answers.{name}.noul is not a number in [0, 1]")
    return JevNoul(value=float(value))


def _parse_answers(
    payload: dict[str, Any], questions: dict[str, dict[str, Any]]
) -> dict[str, JevChoice | JevNoul]:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevResponseError("answers must be an object")
    parsed: dict[str, JevChoice | JevNoul] = {}
    for name, question in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, dict):
            raise JevResponseError(f"answers.{name} is missing")
        question_type = question["type"]
        answer_type = answer.get("type")
        if answer_type is not None and answer_type != question_type:
            raise JevResponseError(f"answers.{name}.type {answer_type!r} != {question_type!r}")
        if question_type == QUESTION_TYPE_CHOICE:
            parsed[name] = _parse_choice(name, answer, question["criteria"])
            continue
        parsed[name] = _parse_noul(name, answer)
    return parsed


def _parse_input_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    tokens = usage.get("input_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        return 0
    return tokens


class JevClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = JEV_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise JevConfigError(f"Jev API key is empty; set {JEV_KEY_ENV}")
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_env(cls) -> JevClient:
        api_key = os.environ.get(JEV_KEY_ENV, "").strip()
        if not api_key:
            raise JevConfigError(f"Environment variable {JEV_KEY_ENV} is not set")
        return cls(api_key)

    async def ask(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> JevResult:
        _validate_questions(questions)
        body = {"model": self._model, "state": state, "questions": questions}
        started = time.monotonic()
        response = await self._post_with_retry(body)
        latency_s = time.monotonic() - started
        try:
            payload = response.json()
        except ValueError as exc:
            raise JevResponseError("Response body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise JevResponseError("Response body must be a JSON object")
        answers = _parse_answers(payload, questions)
        model = payload.get("model")
        return JevResult(
            answers=answers,
            input_tokens=_parse_input_tokens(payload),
            latency_s=latency_s,
            model=model if isinstance(model, str) and model else self._model,
        )

    async def _post_with_retry(self, body: dict[str, Any]) -> httpx.Response:
        attempt = 0
        while True:
            response = await self._post_once(body)
            status = response.status_code
            if status == SUCCESS_STATUS:
                return response
            if _is_retryable_status(status) and attempt < MAX_RETRIES:
                attempt += 1
                await asyncio.sleep(RETRY_DELAY_S)
                continue
            raise _error_for_status(status, response.text[:ERROR_BODY_PREVIEW_CHARS])

    async def _post_once(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(JEV_ENDPOINT, json=body)
        except httpx.TimeoutException as exc:
            raise JevTimeoutError("Jev API request timed out") from exc
        except httpx.TransportError as exc:
            raise JevServerError(f"Jev API transport error: {type(exc).__name__}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
