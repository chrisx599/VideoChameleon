# Remove Auth Feature Design

Date: 2026-02-14

## Summary
Remove all authentication functionality from backend and web. The system will operate as a single-user local deployment. All auth-related code, configs, env vars, pages, dependencies, and endpoints are deleted.

## Goals
- Remove all auth logic and configuration from backend and web.
- Keep core API behavior intact while eliminating access-code checks and auth middleware.
- Simplify runtime configuration by removing auth-related settings.

## Non-Goals
- Replacing auth with another mechanism.
- Introducing new rate limiting or multi-user controls.
- Changing unrelated product behavior.

## Architecture
- Backend no longer initializes auth services or middleware.
- All API routes are public; no access-code or session checks.
- Web app removes login/signup flows and auth client usage.
- The auth package and all related dependencies are removed.

## Components & Data Flow
- Delete `univa/auth/*` and any imports/usage from `univa/univa_server.py`.
- Remove auth configuration from `univa/config/config.py` and `univa/config/config.toml`.
- Remove auth-related admin endpoints and CLI commands.
- Remove `packages/auth` and `apps/web` auth pages, hooks, and env keys.
- Eliminate any request/response fields tied to access codes or sessions.

## Error Handling
- Remove auth-specific errors (401 Unauthorized) and related logs.
- Existing error handling remains unchanged otherwise.

## Testing
- Remove or update tests that depend on auth.
- Validate service startup.
- Validate key API endpoints still respond normally without auth.

## Rollout Notes
- This change is destructive to auth functionality by design.
- Any environment variables related to auth can be removed from deployment configs.
