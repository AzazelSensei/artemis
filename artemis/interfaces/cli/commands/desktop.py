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
from typing import Annotated

from artemis.desktop.fallback import DEFAULT_PRESET, PRESET_ENV
from artemis.desktop.runner import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    DesktopResult,
    DesktopRunConfig,
    run_desktop_task,
)
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
import typer

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
SUCCESS_STYLE = "green"
FAILURE_STYLE = "red"
PANEL_TITLE = "Artemis Desktop"
FALLBACK_HELP = f"Fallback model preset name (default: {DEFAULT_PRESET}, or {PRESET_ENV})."


def _build_summary(result: DesktopResult) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Status", escape(result.status))
    table.add_row("Explanation", escape(result.explanation))
    table.add_row("Total time", f"{result.total_s:.2f}s")
    table.add_row("Steps", str(len(result.steps)))
    table.add_row("Jev steps", str(result.jev_steps))
    table.add_row("Fallback steps", str(result.fallback_steps))
    return table


def desktop_command(
    goal: Annotated[str, typer.Argument(help="Goal to achieve on the Mac desktop.")],
    app: Annotated[
        str | None,
        typer.Option("--app", "-a", help="Target application name (e.g. Calculator)."),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Maximum number of steps.")
    ] = DEFAULT_MAX_STEPS,
    fallback: Annotated[str | None, typer.Option("--fallback", help=FALLBACK_HELP)] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Decide a single step without executing it.")
    ] = False,
) -> None:
    config = DesktopRunConfig(
        goal=goal, app=app, max_steps=max_steps, fallback_preset=fallback, dry_run=dry_run
    )
    try:
        result = asyncio.run(run_desktop_task(config))
    except Exception as error:  # pylint: disable=broad-exception-caught
        Console().print(
            Panel(
                escape(f"{type(error).__name__}: {error}"),
                title=PANEL_TITLE,
                border_style=FAILURE_STYLE,
            )
        )
        raise typer.Exit(EXIT_FAILURE) from error
    is_completed = result.status == STATUS_COMPLETED
    style = SUCCESS_STYLE if is_completed else FAILURE_STYLE
    Console().print(Panel(_build_summary(result), title=PANEL_TITLE, border_style=style))
    raise typer.Exit(EXIT_SUCCESS if is_completed else EXIT_FAILURE)
