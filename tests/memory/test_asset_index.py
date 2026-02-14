from univa.memory.store import ProjectMemoryStore


def test_asset_index_table_exists(tmp_path):
    store = ProjectMemoryStore.open(project_id="test", db_path=tmp_path / "mem.db")
    cur = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_index'")
    row = cur.fetchone()
    assert row is not None


def test_asset_index_upsert_and_fts(tmp_path):
    store = ProjectMemoryStore.open(project_id="test", db_path=tmp_path / "mem.db")
    asset_id = store.upsert_asset_index(
        kind="video",
        path="/tmp/out.mp4",
        prompt="a cat on a bike",
        caption="orange cat riding",
        entity_summary="cat: orange",
        tags="cat,bike",
    )
    cur = store.conn.execute("SELECT asset_id FROM asset_index WHERE asset_id=?", (asset_id,))
    assert cur.fetchone() is not None
    cur = store.conn.execute("SELECT rowid FROM asset_index_fts WHERE asset_index_fts MATCH 'orange' LIMIT 1")
    assert cur.fetchone() is not None


def test_asset_search(tmp_path):
    store = ProjectMemoryStore.open(project_id="test", db_path=tmp_path / "mem.db")
    store.upsert_asset_index(kind="image", path="/tmp/a.png", caption="blue house")
    rows = store.search_assets("blue", limit=5)
    assert len(rows) >= 1
