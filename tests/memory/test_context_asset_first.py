from univa.memory.context import build_memory_context
from univa.memory.service import ProjectMemoryService


def test_context_prefers_asset_search(tmp_path):
    svc = ProjectMemoryService.open("test", db_path=tmp_path / "mem.db")
    svc.store.upsert_asset_index(kind="image", path="/tmp/a.png", caption="blue house")
    ctx = build_memory_context(project_id="test", query="blue house", t_start=0.0, t_end=1.0, db_path=tmp_path / "mem.db")
    assert "assets" in ctx and len(ctx["assets"]) >= 1
