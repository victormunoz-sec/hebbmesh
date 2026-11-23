import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { DatabaseSync } from 'node:sqlite';
import {
  parseMemoryDirectory,
  compileNetwork,
  activateNetworkInMemory,
  findBridge,
  tokenize,
} from '../src/engine.mjs';
import { getActiveStoragePath } from '../src/storage.mjs';

/**
 * HebbMesh Protocolo Preregistro v1.1 — Reproducible Blind Benchmark Suite (0 npm deps)
 *
 * Evaluates 80 blind external queries (`demo/queries_blind.json`, 33 thematic clusters)
 * with Hebbian learning OFF (fresh/immutable network per query) across 2 seed conditions:
 *   - Condition A (PRINCIPAL): Blind regenerated seeds (`demo/seeds_regenerated.json`)
 *   - Condition B (COMPARATIVE): Original inline `| sem:` corpus seeds
 *
 * Across 6 preregistered variants:
 *   1. SQLite FTS5 (BM25 pure)
 *   2. SQLite FTS5 + | sem: (doc2query-style document expansion)
 *   3. HebbMesh without | sem:
 *   4. HebbMesh without propagation (0 hops)
 *   5. HebbMesh without episodic cap (maxEpisodic = Infinity)
 *   6. HebbMesh Full 4D
 *
 * Plus:
 *   - Cluster bootstrap (10,000 resamples, fixed PRNG seed, 95% CI, equivalence margin +-5 pts)
 *   - Separate Hebbian Drift experiment (50 repeated queries on 2 clusters, Gini + hit@3 delta)
 *   - 16-probe Adversarial Privacy Firewall check with mandatory Positive Control (Firewall OFF vs ON)
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const labDir = path.resolve(rootDir, '..');
const memoryDir = path.resolve(rootDir, 'demo', 'memory');

const queriesPath = path.resolve(rootDir, 'demo', 'queries_blind.json');
const seedsRegenPath = path.resolve(rootDir, 'demo', 'seeds_regenerated.json');
const probesPath = path.resolve(rootDir, 'demo', 'privacy_probes.json');
const protocolPath = path.resolve(labDir, 'PROTOCOLO_PREREGISTRO.md');
const enginePath = path.resolve(rootDir, 'src', 'engine.mjs');
const outJsonPath = path.resolve(__dirname, 'results_v1.1.json');

const queries = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
const seedsRegen = JSON.parse(fs.readFileSync(seedsRegenPath, 'utf8'));
const probes = JSON.parse(fs.readFileSync(probesPath, 'utf8'));

const CATS = ['direct', 'semantic', 'multihop', 'episodic'];
const BOOTSTRAP_ITERS = 10000;
const BOOTSTRAP_SEED = 20261122;

function sha256(p) {
  return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex').toUpperCase();
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

function round2(n) {
  return Math.round(n * 100) / 100;
}

function round4(n) {
  return Math.round(n * 10000) / 10000;
}

function percentile(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor((p / 100) * (s.length - 1))));
  return s[idx];
}

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Deterministic Mulberry32 PRNG for reproducible cluster bootstrap
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildFts5Baseline(neuronsMap, includeSeeds = false) {
  const db = new DatabaseSync(':memory:');
  db.exec(`
    CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, code UNINDEXED, type UNINDEXED, content);
  `);
  const ins = db.prepare('INSERT INTO docs (id, code, type, content) VALUES (?, ?, ?, ?)');
  for (const n of neuronsMap.values()) {
    if (n.private) continue;
    const semText = includeSeeds && Array.isArray(n.sem) ? ' ' + n.sem.join(' ') : '';
    ins.run(n.id, n.code, n.type, `${n.code} ${n.title} ${n.summary}${semText}`);
  }
  return db;
}

function queryFts5(db, query, topK = 5) {
  const t0 = performance.now();
  const toks = tokenize(query);
  if (!toks.length) return { codes: [], types: [], ms: 0 };
  const ftsQuery = toks.map((t) => `"${t.replace(/"/g, '')}"`).join(' OR ');
  const rows = db.prepare(`
    SELECT code, type, bm25(docs) AS score
    FROM docs
    WHERE docs MATCH ?
    ORDER BY score ASC
    LIMIT ?
  `).all(ftsQuery, topK);
  return {
    codes: rows.map((r) => r.code),
    types: rows.map((r) => r.type),
    ms: performance.now() - t0,
  };
}

function evaluateVariant(id, label, runner, queryList = queries) {
  const perQuery = [];
  const latencies = [];
  const cats = {};
  for (const c of CATS) cats[c] = { n: 0, h1: 0, h3: 0, h5: 0 };

  let epCtrlN = 0;
  let epCtrlViolations = 0;
  let epCtrlHit3 = 0;
  let epCtrlEpTop3Sum = 0;
  let epCtrlEpTop5Sum = 0;

  for (const q of queryList) {
    const res = runner(q.query);
    latencies.push(res.ms);

    const exp = new Set(q.expected);
    const in1 = res.codes.slice(0, 1).some((c) => exp.has(c)) ? 1 : 0;
    const in3 = res.codes.slice(0, 3).some((c) => exp.has(c)) ? 1 : 0;
    const in5 = res.codes.slice(0, 5).some((c) => exp.has(c)) ? 1 : 0;

    if (!cats[q.cat]) cats[q.cat] = { n: 0, h1: 0, h3: 0, h5: 0 };
    cats[q.cat].n++;
    cats[q.cat].h1 += in1;
    cats[q.cat].h3 += in3;
    cats[q.cat].h5 += in5;

    const epTop3 = res.types.slice(0, 3).filter((t) => t === 'episodic').length;
    const epTop5 = res.types.slice(0, 5).filter((t) => t === 'episodic').length;

    if (q.maxEpisodicAllowed !== undefined) {
      epCtrlN++;
      if (epTop5 > q.maxEpisodicAllowed) epCtrlViolations++;
      if (in3) epCtrlHit3++;
      epCtrlEpTop3Sum += epTop3;
      epCtrlEpTop5Sum += epTop5;
    }

    perQuery.push({
      id: q.id,
      cat: q.cat,
      cluster: q.cluster,
      in1,
      in3,
      in5,
      epTop3,
      epTop5,
    });
  }

  const byCat = {};
  for (const c of CATS) {
    const st = cats[c];
    byCat[c] = {
      n: st.n,
      hit1Pct: st.n ? round1((st.h1 / st.n) * 100) : 0,
      hit3Pct: st.n ? round1((st.h3 / st.n) * 100) : 0,
      hit5Pct: st.n ? round1((st.h5 / st.n) * 100) : 0,
    };
  }

  return {
    id,
    label,
    n: queryList.length,
    byCat,
    exploratoryOverall: {
      hit1Pct: round1((perQuery.reduce((a, x) => a + x.in1, 0) / queryList.length) * 100),
      hit3Pct: round1((perQuery.reduce((a, x) => a + x.in3, 0) / queryList.length) * 100),
      hit5Pct: round1((perQuery.reduce((a, x) => a + x.in5, 0) / queryList.length) * 100),
    },
    episodicControl: {
      n: epCtrlN,
      superHubViolations: epCtrlViolations,
      hit3Pct: epCtrlN ? round1((epCtrlHit3 / epCtrlN) * 100) : 0,
      avgEpisodicInTop3: epCtrlN ? round2(epCtrlEpTop3Sum / epCtrlN) : 0,
      avgEpisodicInTop5: epCtrlN ? round2(epCtrlEpTop5Sum / epCtrlN) : 0,
    },
    latencyMs: {
      mean: round2(mean(latencies)),
      p95: round2(percentile(latencies, 95)),
    },
    perQuery,
  };
}

function classifyDelta(ciLow, ciHigh) {
  if (ciLow > 0) return 'SUPERIOR';
  if (ciHigh < 0) return 'INFERIOR';
  if (ciLow >= -5.0 && ciHigh <= 5.0) return 'PARIDAD_+-5';
  return 'INCONCLUSO';
}

function clusterBootstrapCategory(perQueryA, perQueryB = null, cat, iters = BOOTSTRAP_ITERS, seed = BOOTSTRAP_SEED) {
  const rowsA = perQueryA.filter((r) => r.cat === cat);
  const rowsB = perQueryB ? perQueryB.filter((r) => r.cat === cat) : null;

  const clusterMap = new Map();
  for (let i = 0; i < rowsA.length; i++) {
    const cl = rowsA[i].cluster;
    if (!clusterMap.has(cl)) clusterMap.set(cl, []);
    clusterMap.get(cl).push({ a: rowsA[i].in3, b: rowsB ? rowsB[i].in3 : 0 });
  }
  const clusters = [...clusterMap.values()];
  const k = clusters.length;
  const rng = mulberry32(seed);
  const samples = new Array(iters);

  for (let it = 0; it < iters; it++) {
    let sumA = 0, sumB = 0, cnt = 0;
    for (let j = 0; j < k; j++) {
      const picked = clusters[Math.floor(rng() * k)];
      for (let m = 0; m < picked.length; m++) {
        sumA += picked[m].a;
        sumB += picked[m].b;
        cnt++;
      }
    }
    const rateA = (sumA / cnt) * 100;
    const rateB = (sumB / cnt) * 100;
    samples[it] = rowsB ? (rateA - rateB) : rateA;
  }

  samples.sort((x, y) => x - y);
  const ciLow = round1(samples[Math.floor(0.025 * iters)]);
  const ciHigh = round1(samples[Math.floor(0.975 * iters)]);
  const obsA = (rowsA.reduce((s, r) => s + r.in3, 0) / rowsA.length) * 100;
  const obsB = rowsB ? (rowsB.reduce((s, r) => s + r.in3, 0) / rowsB.length) * 100 : 0;
  const point = round1(rowsB ? (obsA - obsB) : obsA);

  return {
    clusters: k,
    n: rowsA.length,
    point,
    ci95: [ciLow, ciHigh],
    ...(rowsB ? { verdict: classifyDelta(ciLow, ciHigh) } : {}),
  };
}

function runCondition(condName, seedsOverride = null) {
  const parseOpts = seedsOverride
    ? { useSemanticSeeds: true, seedsOverride }
    : { useSemanticSeeds: true };

  // Raw parsed neurons map (before weave deletes n.sem) for FTS5 + | sem:
  const rawMap = parseMemoryDirectory(memoryDir, parseOpts);
  const ftsPureDb = buildFts5Baseline(rawMap, false);
  const ftsSemDb = buildFts5Baseline(rawMap, true);

  // Compile fresh immutable networks
  const netNoSem = compileNetwork(memoryDir, { fresh: true, useSemanticSeeds: false, noPersist: true });
  const netFull = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    ...(seedsOverride ? { seedsOverride } : {}),
    noPersist: condName !== 'regenerated_blind', // Persist Condition A once to measure SQLite WAL footprint
  });

  const v1 = evaluateVariant('fts5_pure', '1. SQLite FTS5 (BM25 pure)', (q) => queryFts5(ftsPureDb, q, 5));
  const v2 = evaluateVariant('fts5_plus_sem', '2. SQLite FTS5 + | sem:', (q) => queryFts5(ftsSemDb, q, 5));
  const v3 = evaluateVariant('hebbmesh_no_sem', '3. HebbMesh sin | sem:', (q) => {
    const r = activateNetworkInMemory(netNoSem, q, { topK: 5, maxHops: 2, maxEpisodic: 1 });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v4 = evaluateVariant('hebbmesh_no_prop', '4. HebbMesh sin propagacion', (q) => {
    const r = activateNetworkInMemory(netFull, q, { topK: 5, disablePropagation: true, maxEpisodic: 1 });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v5 = evaluateVariant('hebbmesh_no_ep_cap', '5. HebbMesh sin tope episodico', (q) => {
    const r = activateNetworkInMemory(netFull, q, { topK: 5, maxHops: 2, maxEpisodic: Infinity });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v6 = evaluateVariant('hebbmesh_full_4d', '6. HebbMesh Full 4D', (q) => {
    const r = activateNetworkInMemory(netFull, q, { topK: 5, maxHops: 2, maxEpisodic: 1 });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });

  const variants = [v1, v2, v3, v4, v5, v6];

  // Cluster bootstrap (10,000 resamples) per category for every variant + paired deltas
  const bootstrap = {
    iterations: BOOTSTRAP_ITERS,
    seed: BOOTSTRAP_SEED,
    perVariant: {},
    pairedDeltas: {
      full4d_minus_fts5Sem: {},
      full4d_minus_noProp: {},
      full4d_minus_noEpCap: {},
    },
  };

  for (let vi = 0; vi < variants.length; vi++) {
    const v = variants[vi];
    bootstrap.perVariant[v.id] = {};
    for (let ci = 0; ci < CATS.length; ci++) {
      const cat = CATS[ci];
      bootstrap.perVariant[v.id][cat] = clusterBootstrapCategory(
        v.perQuery,
        null,
        cat,
        BOOTSTRAP_ITERS,
        BOOTSTRAP_SEED + vi * 100 + ci
      );
    }
  }

  for (let ci = 0; ci < CATS.length; ci++) {
    const cat = CATS[ci];
    bootstrap.pairedDeltas.full4d_minus_fts5Sem[cat] = clusterBootstrapCategory(
      v6.perQuery, v2.perQuery, cat, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 1000 + ci
    );
    bootstrap.pairedDeltas.full4d_minus_noProp[cat] = clusterBootstrapCategory(
      v6.perQuery, v4.perQuery, cat, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 2000 + ci
    );
    bootstrap.pairedDeltas.full4d_minus_noEpCap[cat] = clusterBootstrapCategory(
      v6.perQuery, v5.perQuery, cat, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 3000 + ci
    );
  }

  return {
    condition: condName,
    compileMs: netFull.metrics.compileMs,
    totalNeurons: netFull.metrics.totalNeurons,
    totalEdges: netFull.metrics.totalEdges,
    variants: variants.map(({ perQuery, ...summary }) => summary),
    bootstrap,
    netFull,
    v6PerQuery: v6.perQuery,
  };
}

// Gini coefficient over undirected/directed edge weights
function computeEdgeGini(net) {
  const weights = [];
  for (const n of Object.values(net.neurons)) {
    for (const w of Object.values(n.edges || {})) {
      weights.push(Number(w));
    }
  }
  if (!weights.length) return 0;
  weights.sort((a, b) => a - b);
  const n = weights.length;
  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum === 0) return 0;
  let weightedSum = 0;
  for (let i = 0; i < n; i++) {
    weightedSum += (2 * (i + 1) - n - 1) * weights[i];
  }
  return round4(weightedSum / (n * sum));
}

function runHebbianDriftExperiment(baseNet) {
  // Deep clone network so evaluation stays isolated
  const net = structuredClone(baseNet);
  const giniBefore = computeEdgeGini(net);
  const evalBefore = evaluateVariant('drift_before', 'Before 50 Hebbian queries', (q) => {
    const r = activateNetworkInMemory(net, q, { topK: 5, maxHops: 2, maxEpisodic: 1 });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });

  // Select top 2 clusters by query frequency in queries_blind.json
  const clCounts = new Map();
  for (const q of queries) clCounts.set(q.cluster, (clCounts.get(q.cluster) || 0) + 1);
  const top2Clusters = [...clCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 2)
    .map(([c]) => c);
  const driftPool = queries.filter((q) => top2Clusters.includes(q.cluster));

  // Apply 50 repeated Hebbian queries on those 2 clusters using engine.mjs exact rule (+0.04, cap 1.0)
  for (let step = 0; step < 50; step++) {
    const q = driftPool[step % driftPool.length];
    const r = activateNetworkInMemory(net, q.query, { topK: 5, maxHops: 2, maxEpisodic: 1 });
    if (r.winners.length >= 2) {
      const top = r.winners.slice(0, 3);
      for (const w of top) {
        net.neurons[w.id].activations = Number(net.neurons[w.id].activations || 0) + 1;
      }
      for (let i = 0; i < top.length; i++) {
        for (let j = i + 1; j < top.length; j++) {
          const a = net.neurons[top[i].id];
          const b = net.neurons[top[j].id];
          const nextW = round2(Math.min(1.0, Number(a.edges[b.id] || 0.35) + 0.04));
          a.edges[b.id] = nextW;
          b.edges[a.id] = nextW;
        }
      }
    }
  }

  const giniAfter = computeEdgeGini(net);
  const giniDelta = round4(giniAfter - giniBefore);
  const evalAfter = evaluateVariant('drift_after', 'After 50 Hebbian queries', (q) => {
    const r = activateNetworkInMemory(net, q, { topK: 5, maxHops: 2, maxEpisodic: 1 });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });

  const deltaByCat = {};
  let degradationAlert = false;
  for (const c of CATS) {
    const d = round1(evalAfter.byCat[c].hit3Pct - evalBefore.byCat[c].hit3Pct);
    deltaByCat[c] = {
      before: evalBefore.byCat[c].hit3Pct,
      after: evalAfter.byCat[c].hit3Pct,
      deltaPts: d,
    };
    if (d <= -3.0) degradationAlert = true;
  }

  return {
    repeatedQueries: 50,
    targetClusters: top2Clusters,
    poolSize: driftPool.length,
    giniBefore,
    giniAfter,
    giniDelta,
    richGetRicherAlert: giniAfter > 0.45 || giniDelta >= 0.10,
    degradationAlert,
    deltaByCat,
  };
}

function runPrivacyProbes(netProtected) {
  // Compile a separate control-positive network with Privacy Firewall DISABLED
  const netUnsafe = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    disablePrivacyFirewall: true,
    noPersist: true,
  });

  const details = [];
  let positiveControlTriggered = 0;
  let firewallOnLeaks = 0;

  for (const p of probes) {
    let posHit = false;
    let safeLeak = false;
    let posObserved = [];
    let safeObserved = [];

    if (p.mode === 'activate') {
      const rPos = activateNetworkInMemory(netUnsafe, p.query, { topK: 5, maxHops: 2, disablePrivacyFirewall: true });
      posObserved = rPos.winners.map((w) => w.code);
      posHit = posObserved.some((c) => c.startsWith('PRIV-'));

      const rSafe = activateNetworkInMemory(netProtected, p.query, { topK: 5, maxHops: 2, disablePrivacyFirewall: false });
      safeObserved = rSafe.winners.map((w) => w.code);
      safeLeak = safeObserved.some((c) => c.startsWith('PRIV-'));
    } else if (p.mode === 'bridge') {
      const bPos = findBridge(netUnsafe, p.queryA, p.queryB, { disablePrivacyFirewall: true, minWeight: 0.30 });
      posObserved = bPos.found ? bPos.path.map((x) => x.code) : [bPos.reason];
      posHit = bPos.found && bPos.path.some((x) => x.code.startsWith('PRIV-'));

      const bSafe = findBridge(netProtected, p.queryA, p.queryB, { disablePrivacyFirewall: false });
      safeObserved = bSafe.found ? bSafe.path.map((x) => x.code) : [bSafe.reason];
      safeLeak = bSafe.found && bSafe.path.some((x) => x.code.startsWith('PRIV-'));
    }

    if (posHit) positiveControlTriggered++;
    if (safeLeak) firewallOnLeaks++;

    details.push({
      id: p.id,
      vector: p.vector,
      mode: p.mode,
      target: p.target,
      positiveControlHit: posHit,
      firewallOnLeak: safeLeak,
      posObserved: posObserved.slice(0, 5),
      safeObserved: safeObserved.slice(0, 5),
    });
  }

  // Lateral dendrite isolation check (F-01)
  const privOnlyTokens = new Set([
    'confidential_token_alpha', 'confidential_token_beta',
    'confidential_token_gamma', 'confidential_token_delta',
    'imposter', 'heartbreak', 'breakup',
  ]);
  let lateralDendriteLeaks = 0;
  for (const n of Object.values(netProtected.neurons)) {
    if (n.private) continue;
    for (const dTok of Object.keys(n.dendrites || {})) {
      if (privOnlyTokens.has(dTok)) lateralDendriteLeaks++;
    }
  }

  return {
    totalProbes: probes.length,
    positiveControlTriggered,
    firewallOnLeaks,
    lateralDendriteLeaks,
    pass: positiveControlTriggered === probes.length && firewallOnLeaks === 0 && lateralDendriteLeaks === 0,
    details,
  };
}

// === EXECUTE PROTOCOLO PREREGISTRO v1.1 ===
const condA = runCondition('regenerated_blind', seedsRegen);
const condB = runCondition('original_corpus', null);
const drift = runHebbianDriftExperiment(condA.netFull);
const privacy = runPrivacyProbes(condA.netFull);

const storagePath = getActiveStoragePath(memoryDir);
const diskKb = fs.existsSync(storagePath) ? round1(fs.statSync(storagePath).size / 1024) : 0;
const ramMb = round1(process.memoryUsage().rss / (1024 * 1024));

const report = {
  protocol: 'PROTOCOLO_PREREGISTRO_v1.1',
  timestamp: new Date().toISOString(),
  hashes: {
    PROTOCOLO_PREREGISTRO_md: sha256(protocolPath),
    engine_mjs: sha256(enginePath),
    seeds_regenerated_json: sha256(seedsRegenPath),
    queries_blind_json: sha256(queriesPath),
    privacy_probes_json: sha256(probesPath),
  },
  dataset: {
    totalNeurons: condA.totalNeurons,
    publicNeurons: Object.values(condA.netFull.neurons).filter((n) => !n.private).length,
    privateNeurons: Object.values(condA.netFull.neurons).filter((n) => n.private).length,
    totalEdges: condA.totalEdges,
    queriesTotal: queries.length,
    clustersTotal: new Set(queries.map((q) => q.cluster)).size,
  },
  footprint: {
    compileMs: condA.compileMs,
    sqliteDiskKb: diskKb,
    sqliteFile: path.basename(storagePath),
    processRssMb: ramMb,
    externalNpmDeps: 0,
  },
  conditionA_regenerated_blind: {
    variants: condA.variants,
    bootstrap: condA.bootstrap,
  },
  conditionB_original_corpus: {
    variants: condB.variants,
    bootstrap: condB.bootstrap,
  },
  hebbianDrift: drift,
  privacyRedTeam: privacy,
};

fs.writeFileSync(outJsonPath, JSON.stringify(report, null, 2), 'utf8');

// === CONSOLE SUMMARY ===
function printConditionTable(title, cond) {
  console.log(`\n=== ${title} ===`);
  console.log('| Variant                        | Direct@3 | Semantic@3 | MultiHop@3 | Episodic@3 | EpCtrl@3 (Viol) | Mean ms | P95 ms |');
  console.log('|--------------------------------|----------|------------|------------|------------|-----------------|---------|--------|');
  for (const v of cond.variants) {
    const epCtrlStr = `${v.episodicControl.hit3Pct}% (${v.episodicControl.superHubViolations}/${v.episodicControl.n})`;
    console.log(
      `| ${v.label.padEnd(30)} | ${String(v.byCat.direct.hit3Pct + '%').padStart(8)} | ${String(v.byCat.semantic.hit3Pct + '%').padStart(10)} | ${String(v.byCat.multihop.hit3Pct + '%').padStart(10)} | ${String(v.byCat.episodic.hit3Pct + '%').padStart(10)} | ${epCtrlStr.padStart(15)} | ${String(v.latencyMs.mean).padStart(7)} | ${String(v.latencyMs.p95).padStart(6)} |`
    );
  }
  console.log('\n[Paired Cluster Bootstrap 10,000x: HebbMesh Full 4D - FTS5+|sem:]');
  for (const c of CATS) {
    const d = cond.bootstrap.pairedDeltas.full4d_minus_fts5Sem[c];
    console.log(`  ${c.padEnd(9)}: delta=${d.point >= 0 ? '+' : ''}${d.point} pts | 95% CI [${d.ci95[0]}, ${d.ci95[1]}] | clusters=${d.clusters} | verdict=${d.verdict}`);
  }
}

console.log('=========================================================================');
console.log('  HEBBMESH v1.1 — PREREGISTERED BLIND BENCHMARK (80 QUERIES, 33 CLUSTERS)');
console.log('=========================================================================');
console.log(`Footprint: SQLite=${diskKb} KB | Compile=${condA.compileMs} ms | RSS=${ramMb} MB | Deps=0`);
printConditionTable('CONDITION A (PRINCIPAL): Blind Regenerated Seeds', condA);
printConditionTable('CONDITION B (COMPARATIVE): Original Corpus Seeds', condB);

console.log('\n[Hebbian Drift Experiment: 50 repeated queries on 2 clusters]');
console.log(`  Clusters: ${drift.targetClusters.join(', ')} (pool=${drift.poolSize})`);
console.log(`  Edge Gini: ${drift.giniBefore} -> ${drift.giniAfter} (delta=${drift.giniDelta >= 0 ? '+' : ''}${drift.giniDelta}) | richGetRicherAlert=${drift.richGetRicherAlert}`);
console.log(`  hit@3 Delta: Direct=${drift.deltaByCat.direct.deltaPts} | Semantic=${drift.deltaByCat.semantic.deltaPts} | MultiHop=${drift.deltaByCat.multihop.deltaPts} | Episodic=${drift.deltaByCat.episodic.deltaPts} | degradationAlert=${drift.degradationAlert}`);

console.log('\n[Privacy Red-Team: 16 Adversarial Probes (Mandatory Positive Control)]');
console.log(`  Control Positivo (Firewall OFF): ${privacy.positiveControlTriggered}/${privacy.totalProbes} triggered PRIV-*`);
console.log(`  Firewall ON:                     ${privacy.firewallOnLeaks}/${privacy.positiveControlTriggered} leaks | lateralDendriteLeaks=${privacy.lateralDendriteLeaks} | PASS=${privacy.pass}`);
console.log(`Saved JSON: ${outJsonPath}\n`);
