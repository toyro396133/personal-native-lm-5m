# COMPASS V9.12b — Local approval UI, incremental engineering add-on

**Announced after V9.12 acceptance, before V9.12b UI tests.** This is not part of the pre-registered V9.12 engineering/science gates and cannot be counted retrospectively as a new learning achievement.

Goal: make the persistent COMPASS core usable through a **local browser** on low-resource Windows (Python stdlib only, no Node/Docker/GPU), while retaining explicit approval and separate per-project/personal memory.

Requirements:
1. HTTP server binds exclusively to `127.0.0.1`; reject untrusted Host values and cross-origin state changes. Process-lifetime cryptographically random anti-CSRF token checked via a custom header on all POST requests. No unauthenticated GET can mutate principles.
2. RTL Hebrew accessible static interface; list personal/project principles, explain rationale/source/revision/conditions, show pending proposals, accept or reject with written reason, propose user/project changes, inspect filtered resolution and audit status.
3. Never interpolate untrusted principle strings as HTML; JS must render with `textContent`. Escape and bound JSON at interface edges, reject oversized bodies, reject unsupported HTTP content types.
4. No automatic approvals, no uncontrolled project-scope promotion, stale version still rejected by the store; error feedback visible. All POST operations pass through the already tested CompassStore API; do NOT write SQL directly to change memory.
5. Browser interaction tested via a real loopback HTTP client: load root and token, GET state, POST proposal, verify still pending, reject missing/wrong token/Host and content type, approve with reason, GET active project principle, restart server and verify SQLite persistence.
6. Security limitations: this is a single-user LOCAL DEVELOPMENT server, not authenticated remote access, not multi-user, not a public website. Browser origin and session token are mitigations, not real OS/account identity. Never bind 0.0.0.0 by default.

**Engineering pass:** all V9.12b HTTP tests plus original V9.12 regression suite pass Python 3.11,3.12,3.13. **No learning claim:** `NOT_YET_LEARNED` continues unchanged.
