# WaveSpeed Universal Runner Design

Date: 2026-02-15

## Goal
Support all new WaveSpeed models dynamically without code changes by adding a single “universal runner” that:
- Discovers models and schemas at runtime.
- Validates parameters against `api_schema` when available.
- Auto-uploads local media files and replaces fields with URLs.
- Submits jobs and polls results.

This preserves existing code paths but establishes a new, preferred entry point for WaveSpeed usage.

## Background / Current State
The repo currently has multiple WaveSpeed-specific helpers and tools:
- `univa/utils/wavespeed_api.py` contains many model-specific functions (text/image/video, frame-to-frame, audio).
- `univa/mcp_tools/image_gen.py` and `univa/mcp_tools/video_gen.py` expose fixed tool entry points that select hardcoded models based on config.
- Config selects models by name (`config/mcp_tools_config/config.yaml` or example), requiring changes for new models.

This structure does not scale for frequent model additions.

## Requirements
- Single generic entry point for WaveSpeed model execution.
- Automatically support new models from `/api/v3/models` without code changes.
- Allow local file inputs to be uploaded automatically.
- Keep output compatible with `ToolResponse` and existing tool expectations.
- Avoid breaking existing tools; maintain backward compatibility.

## Non-Goals
- Removing existing model-specific tools or APIs.
- Building a full UI or CLI for model discovery.
- Adding a complex preset system for popular models (may be added later).

## Approach (Recommended)
### Schema-driven Universal Runner
Use the WaveSpeed model list (`GET /api/v3/models`) to discover available models and their `api_schema`.

Pipeline:
1. Fetch/cached model list.
2. Validate inputs against `api_schema` if provided.
3. Detect local media file paths in parameters and upload them.
4. Submit task via `POST /api/v3/{model-id}`.
5. Poll results via `GET /api/v3/predictions/{id}/result` (preferred) or `GET /api/v3/predictions/{id}`.
6. Download outputs when present, returning local path + URL.

This satisfies the “support all new models” requirement with minimal future maintenance.

## Design Details
### Components
- **WaveSpeedClient**
  - `list_models()` -> fetch + cache `/api/v3/models`.
  - `upload_media()` -> upload local files; prefer `/api/v3/media/upload/binary`, fallback to `/api/v3/upload`.
  - `run_model(model_id, payload)` -> submit tasks.
  - `poll_result(task_id, result_url_hint=None)` -> wait for completion.

- **SchemaAdapter**
  - Parse `api_schema` from model definitions.
  - Validate required fields and basic type constraints.
  - Identify media inputs for upload (file path or list of file paths).

- **UniversalRunner**
  - Public entry: `wavespeed_run(model_id, params, save_path=None, timeout_sec=...)`.
  - Orchestrates model discovery, validation, upload, submit, and polling.
  - Returns `ToolResponse` with `output_url`, `output_path`, and raw `content`.

### Data Flow
1. Caller provides `model_id` + `params`.
2. Universal runner loads cached model definitions (or fetches `/models`).
3. `SchemaAdapter` validates params and identifies local file paths.
4. `WaveSpeedClient.upload_media()` uploads files and replaces params with URLs.
5. `WaveSpeedClient.run_model()` submits the request.
6. `WaveSpeedClient.poll_result()` waits for completion.
7. Outputs are downloaded (optional) and returned in `ToolResponse`.

### Error Handling
- **Model not found**: return a clear error, suggest refreshing models.
- **Schema mismatch**: indicate missing/invalid fields with field names.
- **Upload failure**: include failed field name and API error details.
- **Timeout**: return task id and timeout message.
- **Task failure**: propagate WaveSpeed error payload.

### Configuration
Add an optional `wavespeed` section to config:
- `cache_ttl_sec` (default: 180)
- `poll_interval_sec` (default: 2)
- `timeout_sec` (default: 300)
- `download_outputs` (default: true)

Existing model selection config remains but is no longer required by the new tool.

## Testing
- **Unit tests** with mocked HTTP:
  - Model list parsing and caching.
  - Schema validation for required fields and basic types.
  - Upload path: local file -> upload -> URL replacement.
  - Upload fallback behavior.
  - Submit + polling for both `/predictions/{id}` and `/result`.

- **Integration test** (optional):
  - Run a real lightweight model with a minimal prompt to verify full pipeline.

## Migration Plan
- Add new tool entry `wavespeed_run` and keep existing tools unchanged.
- Update docs/examples to prefer the universal runner.
- Over time, route internal new usage to the universal runner to reduce drift.

## Open Questions
- Confirm acceptable timeout defaults per environment.
- Decide whether to auto-download outputs by default for all models or make it opt-in.

