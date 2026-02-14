from univa.memory.service import ProjectMemoryService


def test_backfill_indexes_existing_artifacts(tmp_path):
    svc = ProjectMemoryService.open("test", db_path=tmp_path / "mem.db")
    seg = svc.upsert_segment(0.0, 1.0, "source")
    svc.add_artifact(kind="video", path="/tmp/x.mp4", segment_id=seg)
    svc.backfill_asset_index()
    rows = svc.store.search_assets("/tmp/x.mp4", limit=5)
    assert len(rows) >= 1
