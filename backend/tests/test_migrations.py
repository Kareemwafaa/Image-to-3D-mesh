"""The Alembic migrations must produce exactly the schema the models describe."""

from __future__ import annotations

import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext

from app.db.migrate import upgrade
from app.db.models import Base


def test_migrations_match_models(tmp_path):
    db = tmp_path / "m.db"
    upgrade(f"sqlite+aiosqlite:///{db}")
    engine = sa.create_engine(f"sqlite:///{db}")
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": False})
        diff = compare_metadata(ctx, Base.metadata)
    assert diff == [], diff
