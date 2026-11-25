"""Apply database migrations: `python -m app.db.migrate` (compose runs this once)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "app" / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def upgrade(url: str) -> None:
    command.upgrade(alembic_config(url), "head")


async def upgrade_async(url: str) -> None:
    # Alembic's env runs its own event loop, so give it a thread without one.
    await asyncio.to_thread(upgrade, url)


if __name__ == "__main__":
    from app.core.config import get_settings

    upgrade(get_settings().database_url)
    print("migrations applied")
