# HebbMesh — Repository Agent Guidelines

Zero-inference 4D synaptic memory over plain Markdown (`v1.7`). Pure Node.js (`>= 22.5`, native `node:sqlite` WAL), **0 npm dependencies**, **0 external network/CDN calls**.

## Core CLI Commands
- Query memory (text): `node src/engine_v1.7.mjs spark ./demo/memory "<query>"`
- Query memory (JSON): `node src/engine_v1.7.mjs spark ./demo/memory "<query>" --json`
- 3D A* concept bridge: `node src/engine_v1.7.mjs bridge ./demo/memory "<topicA>" "<topicB>"`
- Legacy v1.2 compile/spark/bridge: `npm run compile`, `npm run spark`, `npm run bridge`
- Offline 3D Viewer (`127.0.0.1:4322`): `node demo.mjs`

## Benchmarks & Verification
- Primary v1.7 suite (Suite B + F-01 Privacy Firewall + 20-case CMI Red Team): `npm run bench:v1.7`
- Full LoCoMo-10 evaluation: `node bench/benchmark_v1.7.mjs --locomo /path/to/locomo10.json`
- Historical blind baselines: `node bench/benchmark_v1.2.mjs` and `node bench/benchmark.mjs`

## Security & Architecture Constraints
- **Privacy Firewall (`F-01`):** `private_vault.md` (`PRIV-xx`) nodes MUST stay isolated before edge weaving, 1-hop dendrite diffusion, spreading activation, 3D A* pathfinding, and HTTP viewer APIs (`0/16` leaks).
- **Viewer Hardening (`src/viewer-server.mjs`, `viewer/index.html`):** Bind strictly to `127.0.0.1`, no wildcard CORS, keep strict CSP and `Sec-Fetch-Site: cross-site` rejection, never expose host OS paths, and use **0 `.innerHTML` sinks**.
- **SQLite Safety (`src/storage.mjs`):** Use parameterized prepared statements (`?`) only; never interpolate untrusted input into SQL.
- **Zero-Bloat Rule:** Do not introduce npm dependencies, embedding models, or remote API calls.
- **Benchmark Integrity:** Do not hand-edit frozen result snapshots (`bench/results_*.json`); regenerate only via benchmark scripts.

## Drop-In Integrations
End-user integration templates live in `integrations/` (`agent-rules.md`, `claude-settings.example.json`, `antigravity-hooks.example.json`). Keep them strictly additive and non-overwriting.
