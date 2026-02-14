from univa.memory.store import ProjectMemoryStore


def test_asset_index_table_exists(tmp_path):
    store = ProjectMemoryStore.open(project_id="test", db_path=tmp_path / "mem.db")
    cur = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asset_index'")
    row = cur.fetchone()
    assert row is not None
