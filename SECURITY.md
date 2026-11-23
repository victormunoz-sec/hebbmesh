# Security Policy — HebbMesh

**Maintainer:** Victor Muñoz Jara (Independent AI Systems & Security Researcher)

## 1. Trust Boundary & Zero-Supply-Chain Posture
- **Zero External Dependencies (`0` npm packages):** HebbMesh runs purely on native Node.js built-ins (`node:sqlite`, `node:http`, `fs`, `path`), eliminating third-party npm supply-chain risk.
- **100% Offline & Local-First:** Zero external telemetry, zero cloud APIs, and zero third-party CDNs in the 3D Viewer.
- **Strict Localhost Binding:** The 3D Viewer binds strictly to `127.0.0.1:4322` with `Content-Security-Policy: default-src 'self' 'unsafe-inline'; connect-src 'self'`, `X-Content-Type-Options: nosniff`, `Cache-Control: no-store`, zero wildcard CORS headers, and `Sec-Fetch-Site: cross-site` rejection on `/api/*`. Do **not** expose port `4322` to `0.0.0.0` or a public LAN without an authenticated TLS reverse proxy.

## 2. Privacy Firewall & LLM Context Hygiene
- **Graph-Level Isolation (`private_vault.md`):** Nodes marked `private: true` are strictly excluded from public edges, 1-hop dendrite diffusion (preventing lateral private-tag leakage), spreading activation, 3D A\* pathfinding (with uniform non-oracle error codes), and HTTP viewer serialization.
- **DOM-XSS & SQLi Immunity:** All SQLite operations use parameterized prepared statements (`?`). The offline 3D Viewer uses exclusively `.textContent` / `createElement` with **0 `.innerHTML` sinks**.
- **Indirect Prompt Injection Guidance:** Because Markdown files are injected into LLM prompts, treat external web text copied into notes as untrusted data and wrap retrieved memories inside explicit delimiters (e.g., `<memory_data>...</memory_data>`).

## 3. Coordinated Vulnerability Disclosure (CVD) & SLA
Please **do not** open public GitHub issues for suspected security vulnerabilities.
- **Private Reporting:** Use GitHub Private Vulnerability Reporting or email `victormunoz.research@gmail.com` with subject `[SECURITY][HebbMesh]`.
- **Response SLA:** Acknowledgement `< 24h` · CVSS v3.1 triage `< 72h` · Patch target `< 7 days`.

## 4. Scope & Safe Harbor
Good-faith security research against `v1.0.x` is welcomed under Safe Harbor.
- **In-Scope:** Privacy Firewall bypasses (activation, A\* bridge, or lateral dendrite leakage), DOM-XSS in `viewer/index.html`, SQLi, Path Traversal, or cross-site API bypasses.
- **Out-of-Scope:** Self-DoS via manual corruption of local Markdown/SQLite files, or user-forced binding to public network interfaces.
