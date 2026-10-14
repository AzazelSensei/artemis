# Copyright 2025-2026 Minitap, Inc.
# Modifications Copyright 2026 Google LLC
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
#
# Originally from mobile-use (https://github.com/minitap-ai/mobile-use).
# See third_party/mobile_use/METADATA for the upstream source and local modifications.

"""Resolve a natural-language app name to an installed package via the Hopper agent."""

import asyncio
import re
import unicodedata

from artemis.context import ArtemisContext
from third_party.mobile_use.agents.hopper.hopper import HopperOutput, hopper
from third_party.mobile_use.controllers.platform_specific_commands_controller import (
    get_adb_device,
    list_packages_async,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


LAUNCH_APP_DOCSTRING = (
    "[ACTION] Finds and launches an application on the device using its natural language name."
)

TURKISH_ASCII_FOLD = str.maketrans(
    {
        "İ": "i",
        "I": "i",
        "ı": "i",
        "Ş": "s",
        "ş": "s",
        "Ç": "c",
        "ç": "c",
        "Ğ": "g",
        "ğ": "g",
        "Ü": "u",
        "ü": "u",
        "Ö": "o",
        "ö": "o",
    }
)
TOKEN_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
PACKAGE_SEGMENT_SEPARATOR = "."
FILLER_TOKENS = frozenset(
    {"app", "apps", "application", "uygulama", "uygulamasi", "uygulamasini", "the"}
)
VENDOR_SEGMENTS = frozenset(
    {"com", "org", "net", "android", "apps", "app", "google", "samsung", "sec"}
)
MISLEADING_PACKAGES = frozenset(
    {"com.google.android.apps.docs", "com.google.android.videos", "com.samsung.android.forest"}
)
PLURAL_SUFFIX = "s"
GENERIC_APP_TOKENS = frozenset(
    {
        "browser",
        "calculator",
        "calendar",
        "camera",
        "clock",
        "contacts",
        "email",
        "files",
        "fitness",
        "gallery",
        "health",
        "mail",
        "maps",
        "messages",
        "messaging",
        "music",
        "notes",
        "phone",
        "photos",
        "player",
        "radio",
        "settings",
        "video",
        "weather",
    }
)
NON_LAUNCHABLE_SEGMENTS = frozenset(
    {"provider", "providers", "server", "service", "services", "overlay", "overlays"}
)
LAUNCHER_ACTIVITY_QUERY = (
    "cmd package query-activities --brief"
    " -a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
)
COMPONENT_SEPARATOR = "/"


def _normalize_app_name_tokens(app_name: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", app_name.translate(TURKISH_ASCII_FOLD).lower())
    ascii_name = "".join(ch for ch in folded if not unicodedata.combining(ch))
    tokens = [token for token in TOKEN_SPLIT_PATTERN.split(ascii_name) if token]
    return [token for token in tokens if token not in FILLER_TOKENS]


def _find_token_span(token: str, segments: list[str]) -> range | None:
    for start in range(len(segments)):
        joined = ""
        for end in range(start, len(segments)):
            joined += segments[end]
            if joined == token:
                return range(start, end + 1)
            if not token.startswith(joined):
                break
    return None


def _count_extra_segments(tokens: list[str], package: str) -> int | None:
    segments = package.lower().split(PACKAGE_SEGMENT_SEPARATOR)
    if NON_LAUNCHABLE_SEGMENTS.intersection(segments):
        return None
    covered: set[int] = set()
    for token in tokens:
        span = _find_token_span(token, segments)
        if span is None:
            return None
        covered.update(span)
    if all(segments[index] in VENDOR_SEGMENTS for index in covered):
        return None
    return sum(
        1
        for index, segment in enumerate(segments)
        if index not in covered and segment not in VENDOR_SEGMENTS
    )


def _select_unique_candidate(tokens: list[str], package_set: set[str]) -> tuple[str | None, int]:
    extras_by_package = {}
    for package in package_set:
        extra_count = _count_extra_segments(tokens, package)
        if extra_count is not None:
            extras_by_package[package] = extra_count
    if not extras_by_package:
        return None, 0
    candidate_count = len(extras_by_package)
    min_extra = min(extras_by_package.values())
    if min_extra > 0:
        return None, candidate_count
    best = [package for package, extra in extras_by_package.items() if extra == min_extra]
    if len(best) != 1:
        return None, candidate_count
    winner = best[0]
    if _is_namespace_parent(winner, extras_by_package):
        return None, candidate_count
    return winner, candidate_count


def _is_namespace_parent(package: str, candidates: dict[str, int]) -> bool:
    child_prefix = package + PACKAGE_SEGMENT_SEPARATOR
    return any(other.startswith(child_prefix) for other in candidates)


def _is_generic_token(token: str) -> bool:
    return token in GENERIC_APP_TOKENS or token.removesuffix(PLURAL_SUFFIX) in GENERIC_APP_TOKENS


def match_package_deterministically(app_name: str, package_set: set[str]) -> str | None:
    tokens = _normalize_app_name_tokens(app_name)
    if not tokens:
        return None
    if all(_is_generic_token(token) for token in tokens):
        return None
    package_set = package_set - MISLEADING_PACKAGES
    query_variants = [tokens]
    if len(tokens) > 1:
        query_variants.append(["".join(tokens)])
    for variant in query_variants:
        selected, candidate_count = _select_unique_candidate(variant, package_set)
        if candidate_count == 0:
            continue
        if selected is None:
            logger.info(
                f"No exact deterministic match for '{app_name}' "
                f"({candidate_count} partial candidates); deferring to hopper"
            )
        return selected
    return None


def parse_launchable_packages(query_output: str, package_set: set[str]) -> set[str]:
    launchable = set()
    for line in query_output.splitlines():
        component = line.strip()
        if COMPONENT_SEPARATOR not in component:
            continue
        package = component.split(COMPONENT_SEPARATOR, 1)[0].strip()
        if package in package_set:
            launchable.add(package)
    return launchable


async def _list_launchable_packages(ctx: ArtemisContext, package_set: set[str]) -> set[str]:
    try:
        device = get_adb_device(ctx)
        if device is None:
            return set()
        query_output = await asyncio.to_thread(device.shell, LAUNCHER_ACTIVITY_QUERY)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Launcher activity query failed, skipping deterministic match: {e}")
        return set()
    return parse_launchable_packages(str(query_output), package_set)


async def find_package(ctx: ArtemisContext, app_name: str, use_fallback: bool = True) -> str | None:
    """Finds the package name for a given application name.

    Returns None if package not found or on error.
    """
    package_cache = getattr(ctx, "package_cache", None)
    if package_cache is None or not isinstance(package_cache, dict):
        try:
            ctx.package_cache = {}
            package_cache = ctx.package_cache
        except Exception:  # pylint: disable=broad-exception-caught
            package_cache = {}

    if isinstance(package_cache, dict) and app_name in package_cache:
        logger.info(f"Cache hit for app '{app_name}': {package_cache[app_name]}")
        return package_cache[app_name]

    try:
        all_packages = await list_packages_async(ctx=ctx)
        package_set = {p.strip() for p in all_packages.split("\n") if p.strip()}

        # Fast path: If app_name is already directly an installed package name
        if app_name in package_set:
            if isinstance(package_cache, dict):
                package_cache[app_name] = app_name
            return app_name

        launchable_packages = await _list_launchable_packages(ctx, package_set)
        if not launchable_packages:
            logger.info(f"No launchable package list for '{app_name}', falling back to hopper")
        else:
            deterministic_package = match_package_deterministically(app_name, launchable_packages)
            if deterministic_package is not None:
                logger.info(
                    f"Resolved app '{app_name}' to '{deterministic_package}'"
                    " via deterministic match"
                )
                if isinstance(package_cache, dict):
                    package_cache[app_name] = deterministic_package
                return deterministic_package
            logger.info(
                f"No unique deterministic match for app '{app_name}', falling back to hopper"
            )

        hopper_output: HopperOutput = await hopper(
            ctx=ctx,
            request=(f"I'm looking for the package name of the following app: '{app_name}'"),
            data=all_packages,
            use_fallback=use_fallback,
        )
        if not hopper_output.found or not hopper_output.output:
            if isinstance(package_cache, dict):
                package_cache[app_name] = None
            return None

        package_name = hopper_output.output.strip()
        if package_name not in package_set:
            logger.warning(
                f"Hopper returned package '{package_name}' for '{app_name}', "
                "but it is NOT physically installed on the device!"
            )
            if isinstance(package_cache, dict):
                package_cache[app_name] = None
            return None

        if isinstance(package_cache, dict):
            package_cache[app_name] = package_name
        return package_name
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to find package for '{app_name}': {e}")
        return None
