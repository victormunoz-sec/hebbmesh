# HebbMesh

**Zero-Bloat 4D Synaptic Memory over Plain Markdown.**  
*Zero Docker · Zero Vector DBs · Zero npm Dependencies · Native `node:sqlite` WAL · AMPA/NMDA + GABA Gating (v1.2).*  
**Technical Report (v1.2 · Zenodo DOI):** [10.5281/zenodo.24473999](https://doi.org/10.5281/zenodo.24473999) *(supersedes v1.1: [10.5281/zenodo.24451147](https://doi.org/10.5281/zenodo.24451147))*

> **Why HebbMesh?** Most AI agent memory stacks either dump raw Markdown into your context window (burning tokens) or force you to maintain Docker, Postgres/Qdrant, and heavy local embedding runtimes. **HebbMesh** treats plain Markdown files as your 20-year source of truth and compiles them into a 4D/3D associative synaptic graph in **~39 ms** with **0 external dependencies**.

![HebbMesh 4D Synaptic Observatory — Live Zero-Embedding Spark and 3D A* Bridge](./assets/hebbmesh-demo.gif)

---

## 60-Second Quickstart

Requires **Node.js >= 22.5** (for native `node:sqlite`). No `npm install` needed.

```bash
git clone https://github.com/victormunoz-sec/hebbmesh.git
cd hebbmesh

# 1. Run the live 3D Brain Viewer + synthetic demo (100% offline, 127.0.0.1:4322)
node demo.mjs

# 2. Run the reproducible head-to-head benchmarks vs SQLite FTS5 (BM25)
node bench/benchmark.mjs       # v1.1 baseline suite (80 blind queries, 33 clusters)
node bench/ablation_v1.2.mjs   # v1.2 M0-M4 ablation suite on dev_set
node bench/benchmark_v1.2.mjs  # v1.2 preregistered blind suite (80 queries: Sonnet 5.5 + DeepSeek V4 Flash)
```

---

## How to Use HebbMesh in Your Own Agent (3 Ways)

### 1. Plug into Claude Code, OpenCode, Cursor, or Antigravity (CLI / Session Hook)
Stop dumping 40 KB of stale history into `CLAUDE.md` or `AGENTS.md`. Point HebbMesh at your own Markdown folder and inject only the **5–6 relevant neurons (~150 tokens)** in ~3 ms:

```bash
# Human-readable context injection (great for SessionStart hooks)
node src/engine.mjs spark ./my-memory "stripe webhook duplicate charge"

# Machine-readable JSON (for Python / Node / MCP wrappers)
node src/engine.mjs spark ./my-memory "stripe webhook duplicate charge" --json

# Discover hidden 3D A* bridge between two separate topics
node src/engine.mjs bridge ./my-memory "stripe webhook" "cloudflare waf"
```

### 2. Embed in Any Custom Node.js Bot or Agent (3 Lines)
No Docker, no Qdrant, no OpenAI embedding calls:

```js
import { activateNetwork, findBridge } from './src/engine.mjs';

// Call right before sending the prompt to Claude, Gemini, DeepSeek, or Ollama:
const mem = activateNetwork('./my-memory', userMessage, { topK: 5 });
const context = mem.winners.map((w) => `[${w.code}] ${w.summary}`).join('\n');
```

### 3. Visualize Your Own Markdown Second Brain Offline
Drop plain `.md` files (`lessons.md`, `graph.md`, `milestones.md`, `profile.md`, `inbox.md`, `knowledge/*.md`, `synthesis/YYYY-Wnn.md`) into any folder and launch the 3D Viewer directly on your data:

```bash
node demo.mjs ./my-memory
# -> Opens 100% offline 3D Synaptic Observatory at http://127.0.0.1:4322
```

**Tip — Write Memories with Zero-Embedding Seeds (`| sem:`):**
When your agent saves a new lesson or note, append 3–5 synonyms or metaphors inline:
```markdown
- **A-01 `[V]` Idempotent Stripe Webhooks:** Persist `event.id` before mutating balances. | sem: double charge, duplicate payment, billing bug
```
HebbMesh automatically compiles those seeds into high-weight dendrites (`0.80`) with **0 MB** of embedding models.

---

## Preregistered Blind Benchmarks (`v1.2` vs `v1.1`)

### v1.2 Preregistered Blind Validation (Dual-LLM External Set: Sonnet 5.5 + DeepSeek V4 Flash)

In `v1.1`, unnormalized 2-hop spreading activation degraded zero-overlap `Semantic@3` by `-20.0 pts` (`43.3% -> 23.3%`) due to supercritical gain (`~3.3x/hop`) and unpruned high-DF cliques. **`v1.2` (`src/engine.mjs`, `v1.2.0-HebbMesh-4D`)** resolves this with five neuro-computational zero-embedding mechanisms:
- **M0 (Semantic Synaptogenesis & Clique Pruning):** weaves `| sem:` (`0.85`) and `concepts` (`0.90`) into edges while tightening `dfMax` from `13%` (`df<=24`) to `<=5%` (`df<=max(6, round(0.05*N))`) with quadratic IDF.
- **M1 (Sparse Out-Fan + Sub-Stochastic Normalization + Refractory Period):** emits only to `top 3` strong neighbors (`w >= 0.58`), normalized by `max(1.8, sumTopM)` to guarantee spectral radius `< 1` without underflow.
- **M2 (AMPA vs. NMDA Coincidence Detection + Parental Dominance):** direct hits (`AMPA`) get bounded tie-break boost without dethroning their sole parent; non-direct neighbors (`NMDA`) require `>= 2` distinct seeds (`kParents >= 2`, modeling the `Mg2+` block) to eliminate orphan lexical drift.
- **M3 (Competitive Lateral GABA Inhibition — k-WTA):** only the `top 4` direct seeds above `0.45 * maxDirect` emit.
- **M4 (4D Inter-Lobe Firewall):** attenuates spurious technical-to-affective cross-lobe jumps (`0.55x`).

Evaluated in a **single pre-registered run** (`PROTOCOLO_PREREGISTRO_v1.2.md`, 5/5 SHA-256 hashes verified pre-execution) on a fresh **80-query blind set** (`demo/queries_blind_v1.2.json`: 40 queries from **Claude Sonnet 5.5** + 40 queries from **DeepSeek V4 Flash**, 80 clusters, `0` overlap with `v1.1`, Hebbian learning OFF, 10,000x cluster bootstrap).

Run `node bench/benchmark_v1.2.mjs` to reproduce:

#### Condition A (Principal — Blind Regenerated Seeds `seeds_regenerated.json`, `N = 80`)

| Engine / Variant | Direct @1/@3 | Semantic @1/@3 | MultiHop @1/@3 | Episodic @1/@3 | Overall @1/@3 | Mean / p95 Latency | Ep. Violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. SQLite FTS5 (BM25 pure)** | 90% / 95% | 26.7% / 43.3% | 75% / 90% | 70% / 80% | 60.0% / 72.5% | 1.04 / 1.44 ms | 0/4 |
| **2. SQLite FTS5 + \| sem:** | 90% / 100% | 50.0% / 66.7% | 85% / 95% | 80% / 90% | 72.5% / 85.0% | 1.02 / 1.43 ms | 0/4 |
| **3. HebbMesh v1.2 sin \| sem:** | 90% / 100% | 40.0% / 60.0% | 75% / 95% | 100% / 100% | 68.8% / 83.8% | 6.33 / 10.90 ms | 0/4 |
| **4. HebbMesh v1.2 sin propagacion (0 hops)** | 85% / 100% | 56.7% / 76.7% | 90% / 100% | 100% / 100% | 77.5% / 91.3% | 5.79 / 8.76 ms | 0/4 |
| **5. HebbMesh v1.2 Full (`A4`: M0–M4)** | **85% / 100%** | **60.0% / 76.7%** | **90% / 100%** | **100% / 100%** | **78.8% / 91.3%** | **5.50 / 7.78 ms** | **0/4** |
| **6. HebbMesh v1.2 Full + GABA suave** | 85% / 100% | 60.0% / 76.7% | 90% / 100% | 100% / 100% | 78.8% / 91.3% | 5.33 / 7.91 ms | 0/4 |

- **Condition B (Comparative — Inline Corpus `| sem:`):** `HebbMesh v1.2 (A4)` reaches **87.5% Overall@3** (`4.17 ms` mean; `3.88 ms` with `GABA suave`), outperforming `SQLite FTS5 + | sem:` (`76.3%`, **+11.2 pts**) and `no_prop` (`86.3%`, +1.2 pts, with `Episodic@1` rising `70% -> 80%`).
- **Source-Model Split (Condition A):** On the stricter **Sonnet 5.5** partition (`20.0%` zero-overlap semantic, `1.27` mean token overlap), `HebbMesh v1.2 (A4)` reaches **90.0% Overall@3** (`73.3% Semantic@3`, `100% Episodic@3`) vs `82.5%` for `FTS5 + | sem:` (`60.0% Semantic@3`, `80% Episodic@3`). On **DeepSeek V4 Flash** (`6.7%` zero-overlap, `2.73` mean overlap), `A4` reaches **92.5% Overall@3 / 85.0% Overall@1** vs `87.5% / 72.5%` for `FTS5 + | sem:`.
- **Statistical & Honesty Notes (10,000x Cluster Bootstrap, 95% CI, `±5 pts` margin):**
  - `A4 − no_prop`: **exact parity `[0.0, 0.0]`** across all 4 categories on `@3` (the `-20 pts` propagation regression from `v1.1` is fully eliminated), with `+3.3 pts` on `Semantic@1` and `+1.3 pts` on `Overall@1`.
  - `A4 − FTS5+|sem:`: strong directional lead (`+6.3 pts Overall@3`, CI `[0.0, +12.5]`; `+10.0 pts Semantic@3`, CI `[-3.3, +23.3]`; `+10.0 pts Episodic@3`, CI `[0.0, +30.0]`), classified conservatively as `INCONCLUSO` at 95% CI due to `N = 80` sample size.
  - **Second Domain & Scale Stress Test (`bench/scale_v1.2.mjs`, Clinical/Biomedical, `N = 1k–10k`):** On `2,020` clinical neurons (`200` queries), `v1.2 (A4)` reaches **100% MultiHop@3 and 100% Episodic@3** vs **0%** for `FTS5 + | sem:` and `no_prop` (**+50.0 pts Overall@3**, 95% CI `[+43, +57]`, `SUPERIOR`, `0/20` privacy leaks). Profiling `10,000` neurons revealed NMDA propagation takes only **`0.93 ms`**; adding an in-memory inverted index + `dfMax <= 40` in experimental `src/engine_v1.3.mjs` (alongside Turrigiano homeostasis & quantum phase interference previewed for upcoming Paper 3) brings `10,000`-neuron compilation from `9.33 s` down to **`432 ms`** (`6.68 MB` SQLite) and query latency down to **`~12.8 ms`** (`p95 21.1 ms < 25 ms`) with `0.0` regression on real blind sets.
  - **Privacy Firewall (F-01):** `16/16` Positive Control (Firewall OFF) vs **`0/16` leaks** (Firewall ON) and `0` lateral dendrite leaks.

---

### v1.1 Historical Baseline (`demo/queries_blind.json`, 80 queries, 33 clusters — now `dev_set`)

Evaluated under `PROTOCOLO_PREREGISTRO.md` (`bench/benchmark.mjs`, `bench/results_v1.1.json`) on the same 187-neuron corpus with 30 adversarial 0%-stem-overlap semantic queries (`dev_set` for `v1.2` ablation `bench/ablation_v1.2.mjs`, where `v1.2 A4` restored `Direct@1` `75% -> 95%`, `Semantic@3` `23.3% -> 43.3%`, and `MultiHop@1` `70% -> 95%` in `4.27 ms`):

#### Condition A (`v1.1` Baseline — Blind Regenerated Seeds `seeds_regenerated.json`)

| Engine / Variant | Direct@3 | Semantic@3 | MultiHop@3 | Episodic@3 | Mean Latency | p95 Latency | Ep. Violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. SQLite FTS5 (BM25 pure)** | 100% | 0.0% | 100% | 90% | 0.80 ms | 1.35 ms | 0/4 |
| **2. SQLite FTS5 + \| sem:** | 100% | 33.3% | 100% | 90% | 0.81 ms | 1.34 ms | 0/4 |
| **3. HebbMesh sin \| sem:** | 90% | 6.7% | 95% | 80% | 5.34 ms | 10.87 ms | 0/4 |
| **4. HebbMesh sin propagacion (0 hops)** | 100% | 43.3% | 100% | 70% | 4.42 ms | 6.98 ms | 0/4 |
| **5. HebbMesh sin tope episodico (maxEp=Inf)** | 100% | 23.3% | 95% | 90% | 4.24 ms | 6.79 ms | 1/4 |
| **6. HebbMesh Full 4D (`v1.1` unnormalized)** | 100% | 23.3% | 95% | 80% | 3.52 ms | 5.04 ms | 0/4 |

- **Hebbian Drift Stability (`v1.1`, 50 repeated queries across 2 active clusters):** Edge Gini coefficient moved from 0.1141 to 0.1183 (+0.0042 delta, zero rich-get-richer runaway); 0.0 pts hit@3 degradation across all 4 categories.

**Footprint:** `456 KB` SQLite WAL index (`.hebbmesh.sqlite`) · `~43 ms` compile · `128.9 MB` RSS · `0` external npm dependencies · `0` network calls.

---

## Five Core Mechanisms

1. **Zero-Embedding Seeding (`| sem:`):** Instead of bundling 100+ MB ONNX embedding models, the LLM capturing a memory appends 3–5 synonyms or idioms inline (`| sem: furry friend, dog, companion`). HebbMesh compiles them as high-weight dendrites (`0.80`), lifting semantic/metaphor retrieval from **0.0% (BM25 pure)** to **33.3%–43.3%** with **0 MB runtime overhead**.
2. **O(1) Hebbian Persistence over `node:sqlite` WAL:** Co-activated neurons strengthen their edges incrementally in a single SQLite transaction without rewriting full JSON graphs, while allowing concurrent reads from the 3D Viewer.
3. **Weekly Episodic Consolidation with Triple Anti-Super-Hub Throttling (`S-Wnn-F1..F4`):** Weekly syntheses are partitioned into 4 functional lobes (`Daily`, `Exploration`, `Execution`, `Support`), capped at backing weight (`z=0.62`), and throttled to `maxEpisodic = 1` per query so broad recaps never crowd out surgical engineering lessons.
4. **3D A\* Synaptic GPS Bridge (`findBridge`):** Finds non-obvious shortest paths between disjoint concepts across 3D `(X, Y, Z)` space (e.g., `"stripe webhook" -> "cloudflare waf"` resolves `A-01 -> HITO-1 -> HITO-6 -> A-04` in `<5 ms`) with **anti-pareidolia hub tolls** and a **strict Privacy Firewall**.
5. **Privacy Firewall by Construction:** Nodes in `private_vault.md` are excluded before spreading activation, before A\* pathfinding (preventing graph triangulation leaks), and at the HTTP viewer boundary (`127.0.0.1` only, zero wildcard CORS).

---

## Architecture

```text
  Plain Markdown Files (20-Year Source of Truth)
  ├── graph.md           (G-xx temporal [V]/[S] edges)
  ├── lessons.md         (A-xx engineering rules + | sem: + | eco:)
  ├── milestones.md      (HITO-xx project timeline)
  ├── profile.md         (PERFIL-xx routines & preferences)
  ├── knowledge/*.md     (DOC-xx technical library -> Anti-Redundancy Map)
  ├── synthesis/Wnn.md   (S-Wnn-F1..F4 episodic chapters)
  └── private_vault.md   (PRIV-xx isolated by Privacy Firewall)
           │
           ▼  compile (~39 ms, 0 deps)
  .hebbmesh.sqlite (WAL · O(1) Hebbian Updates · ~456 KB)
           │
     ┌─────┴──────────────────────────┐
     ▼                                ▼
  CLI / Agent Hook               Offline 3D Viewer
  (spark + 3D A* bridge)         (127.0.0.1:4322 · 0 CDN deps)
```

---

## Security by Construction

Most open-source AI memory prototypes ignore security boundaries. **HebbMesh** is built with adversarial AppSec rigor ([`SECURITY.md`](./SECURITY.md)):

- **Zero Supply-Chain Risk:** `0` npm dependencies and `0` external CDNs — runs 100% offline on native Node.js built-ins.
- **Graph-Layer Privacy Firewall (Zero Lateral Leakage):** `private_vault.md` nodes are isolated before edge weaving, before 1-hop dendrite diffusion (preventing private tags from leaking into public neighbors), before spreading activation, and before 3D A\* pathfinding (with uniform non-oracle responses).
- **Hardened Offline 3D Viewer:** Strictly bound to `127.0.0.1:4322`, zero wildcard CORS, strict `Content-Security-Policy`, `Sec-Fetch-Site: cross-site` rejection on `/api/*`, no local OS paths leaked in API JSON, and **0 `.innerHTML` sinks** (DOM-XSS immune).
- **Zero SQL Injection:** 100% parameterized prepared statements (`?`) across `node:sqlite` WAL storage.

---

## Dual Licensing (Open Source + Commercial)

- **Open Source:** Licensed under **GNU AGPL-3.0-or-later** ([`LICENSE`](./LICENSE)). Free to inspect, study, benchmark, and self-host.
- **Commercial License:** Embedding HebbMesh inside proprietary SaaS platforms, closed-source agent fleets, or commercial products without AGPL source-disclosure obligations requires a Commercial License ([`COMMERCIAL_LICENSE.md`](./COMMERCIAL_LICENSE.md)).

---

## Turnkey Production Setup & Custom Agent Architecture

- **Iris Starter Kit (Coming Soon on Gumroad):** Pre-wired hooks for Claude Code, OpenCode, Cursor, and Antigravity + automated nocturnal REM consolidation + 4-phase weekly synthesis + guided Specialist Agent Factory + `firmas.mjs` token-diet suite.
- **Consulting, Evals & Commercial Licensing:** Building AI agents, model evals (SLM ontological load / context isolation), or low-cost multi-agent harnesses? Contact **Victor Muñoz Jara** (Independent AI Systems & Security Researcher) at `victormunoz.research@gmail.com`.

---

## Citation

If you reference HebbMesh, its pre-registered ablation protocol, Zero-Embedding Seeding (`| sem:`), or AMPA/NMDA + GABA gated propagation (`v1.2`), please cite the Technical Report:

```bibtex
@techreport{munozjara2026hebbmesh,
  author      = {Mu{\~n}oz Jara, Victor},
  title       = {Zero-Bloat Synaptic Memory over Markdown: A Pre-Registered Empirical Study of Write-Time Semantic Seeding, Neuro-Gated Graph Propagation (AMPA/NMDA + GABA), and Privacy Firewalls for Local-First LLM Agents},
  institution = {Independent AI Systems \& Security Research},
  year        = {2026},
  month       = nov,
  version     = {1.2},
  doi         = {10.5281/zenodo.24473999},
  url         = {https://doi.org/10.5281/zenodo.24473999}
}
```
