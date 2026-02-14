from univa.memory.service import ProjectMemoryService


def test_caption_updates_asset_index(tmp_path):
    svc = ProjectMemoryService.open("test", db_path=tmp_path / "mem.db")
    asset_id = svc.store.upsert_asset_index(kind="image", path="/tmp/a.png", prompt="x")
    svc.store.update_asset_caption(asset_id, "a red car", entity_summary="car:red")
    rows = svc.store.search_assets("red", limit=5)
    assert len(rows) >= 1
