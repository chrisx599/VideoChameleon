from univa.memory.service import ProjectMemoryService


def test_asset_index_on_clip_save(tmp_path):
    svc = ProjectMemoryService.open("test", db_path=tmp_path / "mem.db")
    seg = svc.upsert_segment(0.0, 1.0, "source")
    svc.save_clip_take(segment_id=seg, output_path="/tmp/out.mp4", prompt="dog", model="m")
    rows = svc.store.search_assets("dog", limit=5)
    assert len(rows) >= 1
