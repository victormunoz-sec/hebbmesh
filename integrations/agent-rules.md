## HebbMesh Memory Recall (Additive — Does Not Override User Rules)
- When recalling past decisions, bugs, or project context from your Markdown memory folder, run `node ./hebbmesh/src/engine_v1.7.mjs spark <memory-dir> "<query>" --json` before scanning raw notes.
- To uncover indirect links between two topics, run `node ./hebbmesh/src/engine_v1.7.mjs bridge <memory-dir> "<topicA>" "<topicB>"`.
- When appending a new memory entry to Markdown (`lessons.md`, `milestones.md`, `graph.md`), include 3–5 synonyms via `| sem: term1, term2, term3` for zero-embedding semantic recall.
- Respect Privacy Firewall (`F-01`): never query, bypass, or copy contents from `private_vault.md` into shared prompts or public outputs.
- Keep these instructions strictly additive alongside your existing git, coding-style, and architecture rules.
