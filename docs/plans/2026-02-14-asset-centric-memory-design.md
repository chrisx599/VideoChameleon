# Asset-Centric Project Memory Design

Date: 2026-02-14

## Summary
Move project memory to an asset-first retrieval model while keeping per-project SQLite storage. Core assets (video/image/last_frame + prompt) are always persisted. When memory would be incomplete, automatically run vision-to-text to enrich captions and entity states. Retrieval is primarily semantic (SQLite FTS), then augments with entities and minimal time context.

## Goals
- Project-bound memory that persists across sessions (user selects project at startup).
- Asset-centric storage and retrieval for generation context.
- Rich memory: captions + entity states when information is insufficient.
- Semantic search via SQLite FTS within the project DB.

## Non-Goals
- External vector DB or cloud services.
- Full event-sourcing or audit log.
- Long-term retention policies beyond per-project permanence.

## Requirements
- Per-project DB: `projects/<project_id>/memory.db` (existing).
- Core assets: generated video/image + last_frame + prompt/negative_prompt.
- Auto vision2text only when memory details are insufficient for persistence.
- Entity states for characters/locations when available.
- Retrieval prioritizes semantic search, then fills in time context if needed.

## Proposed Approach (Recommended)
Introduce an asset-centric index layer while keeping current tables intact.

### Storage
- Keep existing tables: `timeline_segments`, `clips`, `artifacts`, `entity_states`, `beats`.
- Add `asset_index` materialized table to aggregate key asset fields.
- Add `asset_index_fts` (FTS5) to index prompt/caption/entity_summary/tags.

### Indexing
- On asset creation: write `artifacts` and update `asset_index`.
- After vision2text: store raw caption as `artifact(kind=caption)` and update `asset_index`.
- On entity_state updates: refresh the linked asset's `entity_summary` in `asset_index`.

### Retrieval
- `memory_search_assets(project_id, query)`
  - FTS search against `asset_index_fts`.
  - Return top-k assets with `clip_id`/`segment_id` links.
- `build_memory_context`
  - Use assets as anchor.
  - Add relevant `entity_states` and latest `last_frame`.
  - Only pull minimal time window data if needed for continuity.

## Data Flow
### Write Path
1. Generation produces output + prompt.
2. Persist `artifacts` (kind `video`/`image`/`last_frame`).
3. If memory details are insufficient, run `vision2text`.
4. Persist caption as `artifact(kind=caption)`.
5. Extract and store entity states (characters/locations).
6. Update `asset_index` and FTS.

### Read Path
1. Query FTS with prompt/entity hints.
2. Retrieve asset list (top-k).
3. Augment with entity states and last_frame.
4. Build minimal `MEMORY_CONTEXT` for generation.

## Error Handling
- vision2text failure: still store asset; record a beat for failure.
- FTS update failure: mark `asset_index` row `needs_reindex` and continue.
- Entity extraction failure: keep caption only.

## Testing Scope
- Unit tests
  - Asset write updates `asset_index` and is searchable via FTS.
  - vision2text triggering logic for insufficient memory.
  - entity_summary refresh on entity_state changes.
- E2E
  - New session selects project, generates asset, then retrieves it for context.

## Rollout
- Schema migration to add `asset_index` and `asset_index_fts`.
- Backfill existing assets to index.
- Switch retrieval to FTS-first.
