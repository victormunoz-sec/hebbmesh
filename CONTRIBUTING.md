# Contributing to HebbMesh

Thank you for your interest in improving **HebbMesh**!

## Engineering Principles
1. **Zero-Bloat First:** Do not introduce heavy runtime dependencies, Docker requirements, or 100+ MB local embedding runtimes into the core engine.
2. **Markdown as Source of Truth:** SQLite (`node:sqlite` WAL) is strictly a derived, rebuildable index.
3. **Measure Before Optimizing:** Any retrieval or latency change must include before/after results from `npm run bench`.

## Contributor License Agreement (Dual-License Notice)
Because HebbMesh is distributed under a dual-licensing model (`AGPL-3.0` + Commercial License), by submitting a Pull Request or patch to this repository, you confirm that:
1. You have the legal right to submit the contribution.
2. You grant **Victor Muñoz Jara** a perpetual, worldwide, non-exclusive, royalty-free license to use, modify, sublicense, and distribute your contribution under both `AGPL-3.0` and HebbMesh's Commercial License.
