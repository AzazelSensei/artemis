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
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from artemis.desktop.actions import AxError, DesktopAction, DesktopOp, Observation
from artemis.desktop.ax import DEFAULT_TREE_LIMIT, AxClient
from artemis.desktop.fallback import FallbackDecider, describe_action
from artemis.desktop.policy import (
    OP_DONE,
    SKILL_KEY_BY_OPTION,
    DesktopContext,
    RecentAction,
    is_blocked_app,
)
from artemis.desktop.policy import decide as decide_with_jev
from artemis.agents.jev.policy import _read_min_confidence
from artemis.llm.jev import JevClient, JevConfigError
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_MAX_STEPS = "max_steps"
STATUS_ERROR = "error"
DECIDED_BY_JEV = "jev"
DECIDED_BY_FALLBACK = "fallback"
DEFAULT_MAX_STEPS = 25
WAIT_S = 1.5
REPEAT_EXCLUDE_THRESHOLD = 2
STUCK_THRESHOLD = 3
TIME_PRECISION = 3
SECURITY_CODES = frozenset({"blocked_app", "read_only_app", "secure_field"})
TERMINAL_OPS = frozenset({DesktopOp.DONE, DesktopOp.FAILED})
SCROLL_DIRECTIONS = {DesktopOp.SCROLL_DOWN: "down", DesktopOp.SCROLL_UP: "up"}
INVALID_ACTION_CODE = "invalid_action"
APP_NOT_ALLOWED_CODE = "app_not_allowed"
UNKNOWN_REJECTION = "unknown reason"

JEV_DONE_NOTE = "Jev believes the goal is achieved; verify and report"
JEV_DEFERRED_NOTE = "Jev deferred the decision: {}"
STUCK_NOTE = (
    "stuck: the screen did not change after the last {} actions; try a different approach or report"
)
NOT_TRUSTED_MESSAGE = (
    "Accessibility permission is missing for the process running Artemis; the user must"
    " grant it in System Settings > Privacy & Security > Accessibility"
)
SCREEN_LOCKED_MESSAGE = "The screen is locked; desktop actions are not possible"
SECURE_INPUT_MESSAGE = (
    "Secure keyboard input is active (a password field has focus); desktop actions are paused"
)
SECURITY_MESSAGE = (
    "Stopped for safety: {code} ({message}). This app or field must be handled by the user."
)
BLOCKED_OPEN_MESSAGE = "opening {!r} is not allowed by the desktop policy"
APP_NOT_ALLOWED_MESSAGE = (
    "opening {!r} is not allowed: this task is limited to {!r} and apps named in the goal"
)
JEV_FAILURE_NOTE = "jev error: {}: {}"
MAX_STEPS_MESSAGE = "Reached the step limit ({}) without a final report"
INVALID_MAX_STEPS_MESSAGE = "max_steps must be at least 1, got {}"
DRY_RUN_MESSAGE = "Dry run: decided {} without executing it"
DEFAULT_DONE_MESSAGE = "The fallback model reported the goal as completed"
DEFAULT_FAILED_MESSAGE = "The fallback model reported the goal as failed"
AX_ERROR_MESSAGE = "Desktop control error: {}"
FALLBACK_ERROR_MESSAGE = "Fallback model {!r} failed: {}: {}"
TARGET_SUFFIX = " on {}"
ELEMENT_TEXT = '{} "{}"'
CHANGED_SUFFIX = " -> screen changed"
UNCHANGED_SUFFIX = " -> screen unchanged"
ERROR_SUFFIX = " -> error {}"


class DesktopAx(Protocol):
    async def status(self) -> dict[str, Any]: ...

    async def observe(
        self, app: str | None = None, limit: int = DEFAULT_TREE_LIMIT
    ) -> Observation: ...

    async def click(self, app: str, ref: str) -> dict[str, Any]: ...

    async def set_text(self, app: str, ref: str, text: str) -> dict[str, Any]: ...

    async def key(self, app: str, key: str) -> dict[str, Any]: ...

    async def scroll(self, app: str, ref: str | None, direction: str) -> dict[str, Any]: ...

    async def launch(self, app: str) -> dict[str, Any]: ...

    async def installed_apps(self) -> list[str]: ...


class DesktopFallback(Protocol):
    async def decide(
        self,
        goal: str,
        observation: Observation,
        recent: Sequence[DesktopAction | str],
        note: str | None = None,
    ) -> DesktopAction: ...


class FallbackUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DesktopRunConfig:
    goal: str
    app: str | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    fallback_preset: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class DesktopRunDeps:
    ax: DesktopAx | None = None
    jev: JevClient | None = None
    fallback: DesktopFallback | None = None
    use_jev: bool = True


@dataclass
class DesktopStep:
    number: int
    action: DesktopAction
    decided_by: str
    decision_s: float
    action_s: float
    screen_changed: bool | None
    error: str | None


@dataclass
class DesktopResult:
    status: str
    explanation: str
    steps: list[DesktopStep] = field(default_factory=list)
    total_s: float = 0.0
    jev_steps: int = 0
    fallback_steps: int = 0


@dataclass(frozen=True)
class _Outcome:
    status: str
    explanation: str


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, TIME_PRECISION)


def _skill_key(key: str) -> str:
    return SKILL_KEY_BY_OPTION.get(key, key)


def _target_value(action: DesktopAction) -> str | None:
    if action.op == DesktopOp.PRESS_KEY and action.key:
        return _skill_key(action.key)
    if action.op == DesktopOp.OPEN_APP and action.app:
        return action.app
    return None


def _jev_note(info: dict[str, Any]) -> str:
    confidence = info.get("confidence")
    is_confident = isinstance(confidence, (int, float)) and confidence >= _read_min_confidence()
    if info.get("operation") == OP_DONE and is_confident:
        return JEV_DONE_NOTE
    return JEV_DEFERRED_NOTE.format(info.get("rejection") or UNKNOWN_REJECTION)


@dataclass(frozen=True)
class _StepTrace:
    action: DesktopAction
    changed: bool
    error: AxError | None


@dataclass(frozen=True)
class _RunParts:
    ax: DesktopAx
    jev: JevClient | None
    fallback: DesktopFallback | None


def _history_line(trace: _StepTrace, label: str | None) -> str:
    line = describe_action(trace.action)
    if label:
        line += TARGET_SUFFIX.format(label)
    if trace.error is not None:
        return line + ERROR_SUFFIX.format(trace.error)
    return line + (CHANGED_SUFFIX if trace.changed else UNCHANGED_SUFFIX)


def _is_named_in(app: str, *texts: str | None) -> bool:
    needle = app.strip().casefold()
    if not needle:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    return any(text and pattern.search(text.casefold()) for text in texts)


def _terminal_outcome(action: DesktopAction) -> _Outcome:
    if action.op == DesktopOp.DONE:
        return _Outcome(STATUS_COMPLETED, action.explanation or DEFAULT_DONE_MESSAGE)
    return _Outcome(STATUS_FAILED, action.explanation or DEFAULT_FAILED_MESSAGE)


def _security_outcome(error: AxError) -> _Outcome:
    return _Outcome(STATUS_FAILED, SECURITY_MESSAGE.format(code=error.code, message=error.message))


class _DesktopRun:
    def __init__(self, config: DesktopRunConfig, parts: _RunParts) -> None:
        self._config = config
        self._ax = parts.ax
        self._jev = parts.jev
        self._fallback = parts.fallback
        self._target = config.app
        self._apps: tuple[str, ...] = ()
        self._steps: list[DesktopStep] = []
        self._recent: list[RecentAction] = []
        self._history: list[str] = []
        self._excluded: set[tuple[DesktopOp, int | None]] = set()
        self._excluded_targets: set[tuple[DesktopOp, str]] = set()
        self._repeats: Counter[tuple[DesktopOp, int | None, str | None]] = Counter()
        self._unchanged = 0

    async def run(self, started: float) -> DesktopResult:
        outcome = await self._run_guarded()
        jev_steps = sum(1 for step in self._steps if step.decided_by == DECIDED_BY_JEV)
        return DesktopResult(
            status=outcome.status,
            explanation=outcome.explanation,
            steps=self._steps,
            total_s=_elapsed(started),
            jev_steps=jev_steps,
            fallback_steps=len(self._steps) - jev_steps,
        )

    async def _run_guarded(self) -> _Outcome:
        try:
            return await self._run_loop()
        except AxError as error:
            if error.code in SECURITY_CODES:
                return _security_outcome(error)
            return _Outcome(STATUS_ERROR, AX_ERROR_MESSAGE.format(error))
        except FallbackUnavailableError as error:
            return _Outcome(STATUS_ERROR, str(error))

    async def _run_loop(self) -> _Outcome:
        problem = await self._status_problem()
        if problem is not None:
            return _Outcome(STATUS_ERROR, problem)
        if self._config.app:
            await self._ax.launch(self._config.app)
        if self._jev is not None:
            self._apps = await self._load_apps()
        observation = await self._ax.observe(self._target)
        for number in range(1, self._config.max_steps + 1):
            outcome, observation = await self._step(number, observation)
            if outcome is not None:
                return outcome
        return _Outcome(STATUS_MAX_STEPS, MAX_STEPS_MESSAGE.format(self._config.max_steps))

    async def _status_problem(self) -> str | None:
        status = await self._ax.status()
        if not status.get("trusted"):
            return NOT_TRUSTED_MESSAGE
        if status.get("screenLocked"):
            return SCREEN_LOCKED_MESSAGE
        if status.get("secureInput"):
            return SECURE_INPUT_MESSAGE
        return None

    async def _load_apps(self) -> tuple[str, ...]:
        try:
            return tuple(await self._ax.installed_apps())
        except AxError as error:
            logger.info(f"Desktop app list unavailable for Jev: {error}")
            return ()

    async def _step(
        self, number: int, observation: Observation
    ) -> tuple[_Outcome | None, Observation]:
        decision_started = time.monotonic()
        action, decided_by = await self._decide(observation)
        decision_s = _elapsed(decision_started)
        if action.op in TERMINAL_OPS:
            self._record(DesktopStep(number, action, decided_by, decision_s, 0.0, None, None))
            return _terminal_outcome(action), observation
        if self._config.dry_run:
            self._record(DesktopStep(number, action, decided_by, decision_s, 0.0, None, None))
            message = DRY_RUN_MESSAGE.format(describe_action(action))
            return _Outcome(STATUS_COMPLETED, message), observation
        action_started = time.monotonic()
        error = await self._execute(action, observation)
        action_s = _elapsed(action_started)
        if error is not None and error.code in SECURITY_CODES:
            step = DesktopStep(number, action, decided_by, decision_s, action_s, None, str(error))
            self._record(step)
            return _security_outcome(error), observation
        try:
            current = await self._ax.observe(self._target)
        except AxError as observe_error:
            failure = str(error or observe_error)
            step = DesktopStep(number, action, decided_by, decision_s, action_s, None, failure)
            self._record(step)
            raise
        changed = current.signature != observation.signature
        self._track(_StepTrace(action, changed, error), observation)
        error_text = str(error) if error is not None else None
        step = DesktopStep(number, action, decided_by, decision_s, action_s, changed, error_text)
        self._record(step)
        return None, current

    async def _decide(self, observation: Observation) -> tuple[DesktopAction, str]:
        note: str | None = None
        if self._unchanged >= STUCK_THRESHOLD:
            note = STUCK_NOTE.format(self._unchanged)
        elif self._jev is not None:
            action, note = await self._ask_jev(self._jev, observation)
            if action is not None:
                return action, DECIDED_BY_JEV
        return await self._ask_fallback(observation, note), DECIDED_BY_FALLBACK

    async def _ask_jev(
        self, client: JevClient, observation: Observation
    ) -> tuple[DesktopAction | None, str]:
        ctx = DesktopContext(
            goal=self._config.goal,
            observation=observation,
            recent=tuple(self._recent),
            excluded=frozenset(self._excluded),
            apps=self._apps,
            excluded_targets=frozenset(self._excluded_targets),
        )
        try:
            action, info = await decide_with_jev(client, ctx)
        except Exception as exc:
            logger.info(f"Jev failed, deferring to the fallback model: {type(exc).__name__}: {exc}")
            return None, JEV_FAILURE_NOTE.format(type(exc).__name__, exc)
        if action is not None and action.op not in TERMINAL_OPS:
            return action, ""
        logger.info(f"Jev deferred to the fallback model: {info}")
        return None, _jev_note(info)

    def _fallback_decider(self) -> DesktopFallback:
        if self._fallback is None:
            self._fallback = FallbackDecider(self._config.fallback_preset)
        return self._fallback

    async def _ask_fallback(self, observation: Observation, note: str | None) -> DesktopAction:
        preset = self._config.fallback_preset
        try:
            decider = self._fallback_decider()
            return await decider.decide(self._config.goal, observation, list(self._history), note)
        except Exception as exc:
            message = FALLBACK_ERROR_MESSAGE.format(preset, type(exc).__name__, exc)
            raise FallbackUnavailableError(message) from exc

    async def _execute(self, action: DesktopAction, observation: Observation) -> AxError | None:
        try:
            await self._dispatch(action, observation)
        except AxError as error:
            return error
        return None

    def _action_app(self, observation: Observation) -> str:
        return self._target or observation.bundle_id or observation.app_name

    @staticmethod
    def _ref(action: DesktopAction, observation: Observation) -> str:
        element = observation.element(action.index) if action.index is not None else None
        if element is None:
            raise AxError(INVALID_ACTION_CODE, f"no element with index {action.index}")
        return element.ref

    async def _dispatch(self, action: DesktopAction, observation: Observation) -> None:
        app = self._action_app(observation)
        match action.op:
            case DesktopOp.CLICK:
                await self._ax.click(app, self._ref(action, observation))
            case DesktopOp.TYPE_TEXT:
                if action.text is None:
                    raise AxError(INVALID_ACTION_CODE, "type_text needs text")
                await self._ax.set_text(app, self._ref(action, observation), action.text)
            case DesktopOp.PRESS_KEY:
                if not action.key:
                    raise AxError(INVALID_ACTION_CODE, "press_key needs a key")
                await self._ax.key(app, _skill_key(action.key))
            case DesktopOp.SCROLL_DOWN | DesktopOp.SCROLL_UP:
                await self._ax.scroll(app, None, SCROLL_DIRECTIONS[action.op])
            case DesktopOp.OPEN_APP:
                await self._open_app(action)
            case DesktopOp.WAIT:
                await asyncio.sleep(WAIT_S)
            case _:
                raise AxError(INVALID_ACTION_CODE, f"unsupported operation {action.op}")

    async def _open_app(self, action: DesktopAction) -> None:
        if not action.app:
            raise AxError(INVALID_ACTION_CODE, "open_app needs an app")
        if is_blocked_app(action.app):
            raise AxError("blocked_app", BLOCKED_OPEN_MESSAGE.format(action.app))
        if not self._is_allowed_app(action.app):
            message = APP_NOT_ALLOWED_MESSAGE.format(action.app, self._config.app)
            raise AxError(APP_NOT_ALLOWED_CODE, message)
        await self._ax.launch(action.app)
        self._target = action.app

    def _is_allowed_app(self, app: str) -> bool:
        allowed = self._config.app
        if not allowed:
            return True
        if app.strip().casefold() == allowed.strip().casefold():
            return True
        return _is_named_in(app, self._config.goal)

    @staticmethod
    def _history_target(action: DesktopAction, observation: Observation) -> str | None:
        if action.index is None:
            return _target_value(action)
        element = observation.element(action.index)
        if element is None:
            return None
        return ELEMENT_TEXT.format(element.role, element.label)

    @staticmethod
    def _target_label(action: DesktopAction, observation: Observation) -> str | None:
        if action.index is not None:
            element = observation.element(action.index)
            return element.display if element else None
        return _target_value(action)

    def _track(self, trace: _StepTrace, observation: Observation) -> None:
        action, changed, error = trace.action, trace.changed, trace.error
        label = self._target_label(action, observation)
        self._recent.append(RecentAction(action=action, screen_changed=changed, target=label))
        target = self._history_target(action, observation)
        self._history.append(_history_line(trace, target))
        if changed:
            self._excluded.clear()
            self._excluded_targets.clear()
            self._repeats.clear()
            self._unchanged = 0
            return
        self._unchanged += 1
        repeat_key = (action.op, action.index, _target_value(action))
        self._repeats[repeat_key] += 1
        if error is None and self._repeats[repeat_key] < REPEAT_EXCLUDE_THRESHOLD:
            return
        value = _target_value(action)
        if value is not None:
            self._excluded_targets.add((action.op, value))
            return
        self._excluded.add((action.op, action.index))

    def _record(self, step: DesktopStep) -> None:
        self._steps.append(step)
        logger.info(
            f"Desktop step {step.number}: {describe_action(step.action)}"
            f" decided_by={step.decided_by} decision={step.decision_s:.2f}s"
            f" action={step.action_s:.2f}s changed={step.screen_changed} error={step.error}"
        )


def _jev_from_env() -> JevClient | None:
    try:
        return JevClient.from_env()
    except JevConfigError as error:
        logger.info(f"Jev disabled, using only the fallback model: {error}")
        return None


async def run_desktop_task(
    config: DesktopRunConfig, deps: DesktopRunDeps | None = None
) -> DesktopResult:
    started = time.monotonic()
    deps = deps or DesktopRunDeps()
    if config.max_steps < 1:
        message = INVALID_MAX_STEPS_MESSAGE.format(config.max_steps)
        return DesktopResult(STATUS_ERROR, message, total_s=_elapsed(started))
    try:
        ax = deps.ax or AxClient()
    except AxError as error:
        return DesktopResult(
            STATUS_ERROR, AX_ERROR_MESSAGE.format(error), total_s=_elapsed(started)
        )
    owned_jev = _jev_from_env() if deps.jev is None and deps.use_jev else None
    jev = deps.jev if deps.use_jev and deps.jev is not None else owned_jev
    try:
        return await _DesktopRun(config, _RunParts(ax, jev, deps.fallback)).run(started)
    finally:
        if owned_jev is not None:
            await owned_jev.aclose()
