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
from dataclasses import dataclass
from typing import Any

from artemis.agents.flash.decision import PreTurnDecision
from artemis.agents.flash.runner import FlashRunner
from artemis.agents.jev.policy import (
    MAX_RECENT_ACTIONS,
    OP_BACK,
    OP_ENTER,
    OP_OPEN_APP,
    OP_SCROLL_DOWN,
    OP_SCROLL_UP,
    OP_TAP,
    OP_TYPE_TEXT,
    OP_WAIT,
    ActionRecord,
    JevVerdict,
    PolicyContext,
    ScreenElement,
    build_questions,
    build_state,
    decision_from_verdict,
    describe_target,
    extract_elements,
    extract_text_spans,
    interpret,
    select_app_candidates,
)
from artemis.context import ArtemisContext
from artemis.graph.state import State
from artemis.llm.jev import JevClient, JevError
from third_party.mobile_use.controllers.platform_specific_commands_controller import (
    get_adb_device,
    list_packages_async,
)
from third_party.mobile_use.tools.mobile.launch_app import (
    LAUNCHER_ACTIVITY_QUERY,
    parse_launchable_packages,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

MAX_STALE_JEV_TURNS = 3
MAX_EXCLUDED_ACTIONS = 32
OP_BY_TOOL = {
    "click": OP_TAP,
    "input_text": OP_TYPE_TEXT,
    "wait_for_delay": OP_WAIT,
    "manage_app": OP_OPEN_APP,
}
OP_BY_KEY = {"ENTER": OP_ENTER, "BACK": OP_BACK}
OP_BY_SWIPE_DIRECTION = {"up": OP_SCROLL_DOWN, "down": OP_SCROLL_UP}
FINAL_REPORT_INDEX = 0
NEXT_XML_INDEX = 2


@dataclass(frozen=True)
class _ExcludedAction:
    screen: str
    record: ActionRecord


@dataclass(frozen=True)
class _ObservedTurn:
    xml_list: str | None
    elements: dict[str, ScreenElement]
    jev_call_id: str | None
    verdict: JevVerdict | None


class JevRunner(FlashRunner):
    def __init__(self, ctx: ArtemisContext, goal: str, max_turns: int | None = None):
        self.jev_client = JevClient.from_env()
        super().__init__(ctx, goal, max_turns)
        self._packages: list[str] | None = None
        self._recent_actions: list[ActionRecord] = []
        self._excluded: list[_ExcludedAction] = []
        self._last_action_xml: str | None = None
        self._stale_jev_turns = 0
        self._observed = _ObservedTurn(xml_list=None, elements={}, jev_call_id=None, verdict=None)

    async def run(self, state: State) -> dict:
        try:
            return await super().run(state)
        finally:
            await self.jev_client.aclose()

    async def _load_packages(self) -> list[str]:
        if self._packages is not None:
            return self._packages
        try:
            self._packages = sorted(await self._query_launchable_packages())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"Jev could not list launchable packages; OPEN_APP disabled: {exc!r}")
            self._packages = []
        logger.info(f"Jev loaded {len(self._packages)} launchable packages")
        return self._packages

    async def _query_launchable_packages(self) -> set[str]:
        device = get_adb_device(self.ctx)
        if device is None:
            return set()
        raw = await list_packages_async(self.ctx)
        package_set = {p.strip() for p in str(raw).splitlines() if p.strip()}
        query_output = await asyncio.to_thread(device.shell, LAUNCHER_ACTIVITY_QUERY)
        return parse_launchable_packages(str(query_output), package_set)

    def _excluded_on(self, xml_list: str | None) -> tuple[ActionRecord, ...]:
        screen = xml_list or ""
        return tuple(item.record for item in self._excluded if item.screen == screen)

    def _exclude(self, xml_list: str | None, record: ActionRecord) -> None:
        screen = xml_list or ""
        if any(
            i.screen == screen and i.record.signature == record.signature for i in self._excluded
        ):
            return
        self._excluded.append(_ExcludedAction(screen=screen, record=record))
        self._excluded = self._excluded[-MAX_EXCLUDED_ACTIONS:]

    def _is_repeat(self, verdict: JevVerdict, xml_list: str | None) -> bool:
        if not self._recent_actions or xml_list != self._last_action_xml:
            return False
        return self._recent_actions[-1].signature == verdict.signature

    async def _build_context(self, elements: list[ScreenElement]) -> PolicyContext:
        packages = await self._load_packages()
        return PolicyContext(
            goal=self.goal,
            elements=tuple(elements),
            text_spans=tuple(extract_text_spans(self.goal)),
            apps=tuple(select_app_candidates(self.goal, packages)),
            recent_actions=tuple(self._recent_actions[-MAX_RECENT_ACTIONS:]),
            excluded_actions=self._excluded_on(self._observed.xml_list),
        )

    async def _pre_turn_decision(
        self, state: State, turn: int, xml_list: str
    ) -> PreTurnDecision | None:
        elements = extract_elements(getattr(state, "indexed_elements", None))
        self._observed = _ObservedTurn(
            xml_list=xml_list,
            elements={e.key: e for e in elements},
            jev_call_id=None,
            verdict=None,
        )
        if self._stale_jev_turns >= MAX_STALE_JEV_TURNS:
            logger.info(
                f"Jev turn {turn}: screen unchanged after {self._stale_jev_turns}"
                " consecutive Jev actions; deferring to the model"
            )
            return None
        ctx = await self._build_context(elements)
        try:
            result = await self.jev_client.ask(build_state(ctx), build_questions(ctx))
        except JevError as exc:
            logger.warning(f"Jev turn {turn}: request failed, deferring to the model: {exc!r}")
            return None
        verdict = interpret(result, ctx)
        logger.info(
            f"Jev turn {turn}: op={verdict.operation} target={describe_target(verdict)}"
            f" conf={verdict.confidence:.2f} latency={result.latency_s:.2f}s"
            f" goal_achieved={verdict.goal_achieved} rejection={verdict.rejection}"
        )
        if verdict.tool_call is None:
            return None
        if self._is_repeat(verdict, xml_list):
            logger.info(
                f"Jev turn {turn}: repeated {verdict.signature} on an unchanged screen;"
                " excluding it and deferring to the model"
            )
            self._exclude(xml_list, verdict.to_record())
            return None
        decision = decision_from_verdict(verdict, result)
        if decision is None:
            return None
        self._observed = _ObservedTurn(
            xml_list=xml_list,
            elements=self._observed.elements,
            jev_call_id=verdict.tool_call["id"],
            verdict=verdict,
        )
        return decision

    async def _process_tool_calls(self, tool_calls: list, state: State, *args: Any, **kwargs: Any):
        observed = self._observed
        outcome = await super()._process_tool_calls(tool_calls, state, *args, **kwargs)
        if outcome[FINAL_REPORT_INDEX] is not None:
            return outcome
        next_xml = outcome[NEXT_XML_INDEX]
        screen_changed = None if observed.xml_list is None else next_xml != observed.xml_list
        is_jev_turn = observed.jev_call_id is not None and any(
            isinstance(tc, dict) and tc.get("id") == observed.jev_call_id for tc in tool_calls
        )
        if is_jev_turn and observed.verdict is not None:
            self._recent_actions.append(observed.verdict.to_record(screen_changed))
        else:
            self._recent_actions.extend(
                self._records_from_model_turn(tool_calls, observed, screen_changed)
            )
        self._recent_actions = self._recent_actions[-MAX_RECENT_ACTIONS:]
        self._last_action_xml = observed.xml_list
        self._stale_jev_turns = (
            self._stale_jev_turns + 1 if is_jev_turn and not screen_changed else 0
        )
        return outcome

    def _records_from_model_turn(
        self, tool_calls: list, observed: _ObservedTurn, screen_changed: bool | None
    ) -> list[ActionRecord]:
        action_names = frozenset(self.executor.action_tool_names)
        records = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or self._tool_name(tool_call) not in action_names:
                continue
            records.append(self._record_from_tool_call(tool_call, observed, screen_changed))
        return records

    @staticmethod
    def _tool_name(tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("name") or "").split(":")[-1]

    @staticmethod
    def _operation_for(name: str, args: dict[str, Any]) -> str:
        if name == "press_key":
            return OP_BY_KEY.get(str(args.get("key", "")).upper(), f"KEY_{args.get('key')}")
        if name == "swipe":
            direction = str(args.get("direction") or args.get("gesture") or "").lower()
            return OP_BY_SWIPE_DIRECTION.get(direction, f"SWIPE_{direction or 'GESTURE'}".upper())
        return OP_BY_TOOL.get(name, name.upper())

    @classmethod
    def _record_from_tool_call(
        cls, tool_call: dict[str, Any], observed: _ObservedTurn, screen_changed: bool | None
    ) -> ActionRecord:
        args = tool_call.get("args") or {}
        op = cls._operation_for(cls._tool_name(tool_call), args)
        target = args.get("target")
        element = observed.elements.get(str(target)) if isinstance(target, int) else None
        if op == OP_OPEN_APP:
            target_key = str(args.get("app_name") or "") or None
            target_label = target_key
        else:
            target_key = element.key if element else None
            target_label = element.display if element else args.get("target_description")
        text = args.get("text")
        return ActionRecord(
            op=op,
            target_key=target_key,
            target_label=target_label,
            text=text if isinstance(text, str) else None,
            screen_changed=screen_changed,
        )
