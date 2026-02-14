# Logging + CLI Interaction Optimization Design

Date: 2026-02-14

## Summary
Unify backend logging into a single, consistent log format and file across CLI, server, MCP tools, and utils. Make CLI console output user-facing only, while detailed tool execution information moves to logs. Improve CLI progress feedback with dynamic status rendering.

## Goals
- Single log format and single log file across all backend components.
- Log format includes full fields and uses `|` separators.
- CLI console shows only user-visible output; tool execution details go to logs.
- CLI progress feels more dynamic and informative.

## Non-goals
- Switching to external logging stacks (ELK, OpenTelemetry) or new dependencies.
- Log rotation or retention policies (single file only).
- Reworking core business logic or tool execution semantics.

## Architecture
- Centralize logging configuration in a shared module used by CLI and server.
- Use `contextvars` to inject `session_id`, `project_id`, and `tool` into log records.
- Make all modules use `logging.getLogger(__name__)` and inherit the root configuration.

## Log Format
- Format: `timestamp | level | logger | session_id | project_id | tool | message`
- Missing fields are filled with `-` to avoid formatter errors.
- Single log file at `logs/app.log` (configurable via function args if needed).

## Components
- `univa/utils/logging_setup.py`
  - `configure_logging(log_file, level, enable_console=False)`
  - `ContextFilter` to add context fields
  - Optional helper to set context (wrapping `contextvars`)
- `univa/univa_agent.py`
  - Replace CLI logging block with `configure_logging`
  - Set contextvars for `session_id` and `project_id` during CLI loop
  - Remove tool panels from console; keep them in logs
  - Add a dynamic progress status line (spinner + phase text)
- `univa/univa_server.py`
  - Call `configure_logging` at startup
  - Set per-request contextvars when handling a request
- `univa/mcp_tools/*` and `univa/utils/*`
  - Remove local `basicConfig` usage
  - Use module loggers only; rely on global setup

## Data Flow
1. Entry points (CLI/Server) call `configure_logging` once at startup.
2. Each request or CLI turn sets contextvars (`session_id`, `project_id`, `tool`).
3. Any module logging uses the global configuration and receives injected fields.
4. Console output is user-focused; detailed execution logs are written to file.

## Error Handling
- If log directory/file cannot be created, fall back to `stderr` handler.
- Log context injection is defensive: missing values become `-`.
- CLI errors display brief summaries; full tracebacks go to log file.

## Testing / Verification
- Manual CLI run: verify dynamic progress, no tool panels in console.
- Verify `logs/app.log` exists and lines match the target format.
- Optional unit test: configure logging, emit a record, assert formatted fields.

## Rollout
- Update CLI and server to use shared setup.
- Remove/replace local logger configs in tools/utils.
- Validate log output and console UX.
