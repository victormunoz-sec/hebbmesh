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

/**
 * HebbMesh Protocolo Preregistro v1.2 -- Reproducible Blind Benchmark Suite (0 npm deps)
 *
 * Evaluates 80 blind external queries (`demo/queries_blind_v1.2.json`, 80 thematic clusters)
 * across 2 source models (sonnet-5.5 n=40 vs deepseek-v4-flash n=40)
 * with Hebbian learning OFF (fresh/immutable network per query, noPersist: true) across 2 conditions:
 *   - Condition A (PRINCIPAL): Blind regenerated seeds (`demo/seeds_regenerated.json`)
 *   - Condition B (COMPARATIVE): Original inline `| sem:` corpus seeds
 *
 * Across 6 preregistered variants:
 *   1. fts5_pure: SQLite FTS5 (BM25 pure)
 *   2. fts5_sem: SQLite FTS5 + | sem:
 *   3. hebb_no_sem: HebbMesh v1.2 without | sem:
 *   4. hebb_no_prop: HebbMesh v1.2 without propagation (0 hops)
 *   5. hebb_v12_a4: HebbMesh v1.2 A4 (topK:5, maxHops:2, maxEpisodic:1)
 *   6. hebb_v12_a4_gaba: HebbMesh v1.2 A4 + GABA Soft
 *
 * Plus:
 *   - Pre-execution SHA-256 verification (abort with process.exit(1) on mismatch)
 *   - Lexical gap analysis on 30 semantic queries
 *   - Source-model breakdown on Condition A for all 6 variants
 *   - Cluster bootstrap (10,000 resamples, Mulberry32 seed=20261123, 95% CI, margin +-5 pts)
 *   - 16-probe Adversarial Privacy Firewall check with mandatory Positive Control
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const memoryDir = path.resolve(rootDir, 'demo', 'memory');

const queriesPath = path.resolve(rootDir, 'demo', 'queries_blind_v1.2.json');
const seedsRegenPath = path.resolve(rootDir, 'demo', 'seeds_regenerated.json');
const probesPath = path.resolve(rootDir, 'demo', 'privacy_probes.json');
const outJsonPath = path.resolve(__dirname, 'results_v1.2.json');

const CATS = ['direct', 'semantic', 'multihop', 'episodic'];
const BOOT_CATS = ['direct', 'semantic', 'multihop', 'episodic', 'overall'];
const BOOTSTRAP_ITERS = 10000;
const BOOTSTRAP_SEED = 20261123;

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex').toUpperCase();
}

const EXPECTED_HASHES = {
  'src/engine.mjs': '01F40FC52A7EE83C7626E9791F0CCA7B8D9ACB99665C3F5921071573DFB678AE',
  'src/storage.mjs': '07F491BB838E4B4F1FFBABD6F9766B87A8A6BC4B3889810A1F5216A7BE113BFD',
  'demo/queries_blind_v1.2.json': 'F0460CA41AEF16F97F558328A1F5EC95FA5C468F2FD9D1C4EF70131C32E309FA',
  'demo/seeds_regenerated.json': '2FFB17B31C84C5356D63749D6642F32BB085BDC083813BC82021B8DA538EC0A6',
  'demo/privacy_probes.json': '9C664E018589998D0AA5E2A3E384EB0A81EA48BF00699F83D6C8D79F23D4F0DA',
};

function verifyPreExecutionHashes() {
  const details = {};
  let allPass = true;
  for (const [relPath, expected] of Object.entries(EXPECTED_HASHES)) {
    const fullPath = path.resolve(rootDir, relPath);
    if (!fs.existsSync(fullPath)) {
      console.error(`[FATAL] Missing file for pre-execution SHA check: ${relPath}`);
      process.exit(1);
    }
    const actual = sha256(fullPath);
    const match = actual === expected;
    details[relPath] = { expected, actual, match };
    if (!match) {
      console.error(`[FATAL] SHA-256 mismatch for ${relPath}: expected ${expected}, got ${actual}`);
      allPass = false;
    }
  }
  if (!allPass) {
    process.exit(1);
  }
  return { pass: true, details };
}

// 1. Run Pre-execution SHA verification immediately
const shaVerification = verifyPreExecutionHashes();

const queries = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
const seedsRegen = JSON.parse(fs.readFileSync(seedsRegenPath, 'utf8'));
const probes = JSON.parse(fs.readFileSync(probesPath, 'utf8'));

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
      source_model: q.source_model,
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

  const overall = {
    hit1Pct: round1((perQuery.reduce((a, x) => a + x.in1, 0) / queryList.length) * 100),
    hit3Pct: round1((perQuery.reduce((a, x) => a + x.in3, 0) / queryList.length) * 100),
    hit5Pct: round1((perQuery.reduce((a, x) => a + x.in5, 0) / queryList.length) * 100),
  };

  return {
    id,
    label,
    n: queryList.length,
    byCat,
    overall,
    exploratoryOverall: overall,
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

function clusterBootstrap(perQueryA, perQueryB = null, catOrOverall = 'overall', iters = BOOTSTRAP_ITERS, seed = BOOTSTRAP_SEED) {
  const isOverall = !catOrOverall || catOrOverall === 'overall';
  const rowsA = isOverall ? perQueryA : perQueryA.filter((r) => r.cat === catOrOverall);
  const rowsB = perQueryB ? (isOverall ? perQueryB : perQueryB.filter((r) => r.cat === catOrOverall)) : null;

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

function computeSourceModelBreakdown(cond) {
  const models = ['sonnet-5.5', 'deepseek-v4-flash'];
  const breakdown = {};

  for (const v of cond.variants) {
    breakdown[v.id] = { label: v.label };
    for (const sm of models) {
      const rows = v.perQuery.filter((r) => r.source_model === sm);
      const totalN = rows.length;
      const byCat = {};
      for (const c of CATS) {
        const catRows = rows.filter((r) => r.cat === c);
        const cn = catRows.length;
        byCat[c] = {
          n: cn,
          hit1Pct: cn ? round1((catRows.reduce((a, x) => a + x.in1, 0) / cn) * 100) : 0,
          hit3Pct: cn ? round1((catRows.reduce((a, x) => a + x.in3, 0) / cn) * 100) : 0,
        };
      }
      const overall = {
        n: totalN,
        hit1Pct: totalN ? round1((rows.reduce((a, x) => a + x.in1, 0) / totalN) * 100) : 0,
        hit3Pct: totalN ? round1((rows.reduce((a, x) => a + x.in3, 0) / totalN) * 100) : 0,
      };
      breakdown[v.id][sm] = {
        n: totalN,
        byCat,
        overall,
      };
    }
  }
  return breakdown;
}

function computeLexicalGap(rawMap, queryList) {
  const byCode = new Map();
  for (const n of rawMap.values()) byCode.set(n.code, n);

  function tokensOverlap(tokA, tokB) {
    if (tokA === tokB) return true;
    const lenA = tokA.length;
    const lenB = tokB.length;
    const minLen = Math.min(lenA, lenB);
    if (minLen >= 4) {
      if (tokA.startsWith(tokB) || tokB.startsWith(tokA)) return true;
      const targetPrefixLen = Math.min(5, lenA, lenB);
      if (targetPrefixLen >= 4) {
        let p = 0;
        while (p < targetPrefixLen && tokA[p] === tokB[p]) p++;
        if (p >= targetPrefixLen) return true;
      }
    }
    return false;
  }

  const semQueries = queryList.filter((q) => q.cat === 'semantic');
  const details = [];
  let zeroOverlapCount = 0;
  let totalOverlapTokensSum = 0;
  const byModel = {};

  for (const q of semQueries) {
    const qToks = tokenize(q.query);
    const expToks = [];
    for (const code of q.expected) {
      const n = byCode.get(code);
      if (n) {
        expToks.push(...tokenize(`${n.title} ${n.summary}`));
      }
    }
    const matchingTokens = qToks.filter((qt) => expToks.some((et) => tokensOverlap(qt, et)));
    const overlapTokensCount = matchingTokens.length;
    const isZero = overlapTokensCount === 0;
    if (isZero) zeroOverlapCount++;
    totalOverlapTokensSum += overlapTokensCount;

    if (!byModel[q.source_model]) {
      byModel[q.source_model] = {
        totalSemantic: 0,
        zeroOverlapCount: 0,
        zeroOverlapPct: 0,
        totalOverlapTokens: 0,
        meanOverlapTokens: 0,
      };
    }
    const smStats = byModel[q.source_model];
    smStats.totalSemantic++;
    if (isZero) smStats.zeroOverlapCount++;
    smStats.totalOverlapTokens += overlapTokensCount;

    details.push({
      id: q.id,
      cluster: q.cluster,
      source_model: q.source_model,
      expected: q.expected,
      qTokensCount: qToks.length,
      overlapTokensCount,
      matchingTokens,
    });
  }

  for (const sm of Object.keys(byModel)) {
    const smStats = byModel[sm];
    smStats.zeroOverlapPct = smStats.totalSemantic
      ? round1((smStats.zeroOverlapCount / smStats.totalSemantic) * 100)
      : 0;
    smStats.meanOverlapTokens = smStats.totalSemantic
      ? round2(smStats.totalOverlapTokens / smStats.totalSemantic)
      : 0;
  }

  return {
    totalSemantic: semQueries.length,
    zeroOverlapCount,
    zeroOverlapPct: semQueries.length ? round1((zeroOverlapCount / semQueries.length) * 100) : 0,
    meanOverlapTokens: semQueries.length ? round2(totalOverlapTokensSum / semQueries.length) : 0,
    bySourceModel: byModel,
    details,
  };
}

function runCondition(condName, seedsOverride = null) {
  const parseOpts = seedsOverride
    ? { useSemanticSeeds: true, seedsOverride }
    : { useSemanticSeeds: true };

  const rawMap = parseMemoryDirectory(memoryDir, parseOpts);
  const ftsPureDb = buildFts5Baseline(rawMap, false);
  const ftsSemDb = buildFts5Baseline(rawMap, true);

  const netNoSem = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: false,
    m0Topology: true,
    noPersist: true,
  });

  const netM0 = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    m0Topology: true,
    ...(seedsOverride ? { seedsOverride } : {}),
    noPersist: true,
  });

  const v1 = evaluateVariant('fts5_pure', '1. SQLite FTS5 (BM25 pure)', (q) => queryFts5(ftsPureDb, q, 5));
  const v2 = evaluateVariant('fts5_sem', '2. SQLite FTS5 + | sem:', (q) => queryFts5(ftsSemDb, q, 5));
  const v3 = evaluateVariant('hebb_no_sem', '3. HebbMesh v1.2 sin | sem:', (q) => {
    const r = activateNetworkInMemory(netNoSem, q, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v4 = evaluateVariant('hebb_no_prop', '4. HebbMesh v1.2 sin propagacion', (q) => {
    const r = activateNetworkInMemory(netM0, q, { topK: 5, disablePropagation: true, maxEpisodic: 1, reinforce: false });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v5 = evaluateVariant('hebb_v12_a4', '5. HebbMesh v1.2 A4', (q) => {
    const r = activateNetworkInMemory(netM0, q, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });
  const v6 = evaluateVariant('hebb_v12_a4_gaba', '6. HebbMesh v1.2 A4 + GABA Soft', (q) => {
    const r = activateNetworkInMemory(netM0, q, { topK: 5, maxHops: 2, useGabaSoft: true, maxEpisodic: 1, reinforce: false });
    return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
  });

  const variants = [v1, v2, v3, v4, v5, v6];

  const bootstrap = {
    iterations: BOOTSTRAP_ITERS,
    seed: BOOTSTRAP_SEED,
    perVariant: {},
    pairedDeltas: {
      a4_minus_noProp: {},
      a4_minus_fts5Sem: {},
      full4d_minus_noProp: {},
      full4d_minus_fts5Sem: {},
    },
  };

  for (let vi = 0; vi < variants.length; vi++) {
    const v = variants[vi];
    bootstrap.perVariant[v.id] = {};
    for (let ci = 0; ci < BOOT_CATS.length; ci++) {
      const cat = BOOT_CATS[ci];
      bootstrap.perVariant[v.id][cat] = clusterBootstrap(
        v.perQuery,
        null,
        cat,
        BOOTSTRAP_ITERS,
        BOOTSTRAP_SEED + vi * 100 + ci
      );
    }
  }

  for (let ci = 0; ci < BOOT_CATS.length; ci++) {
    const cat = BOOT_CATS[ci];
    const dNoProp = clusterBootstrap(
      v5.perQuery, v4.perQuery, cat, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 1000 + ci
    );
    const dFts5Sem = clusterBootstrap(
      v5.perQuery, v2.perQuery, cat, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 2000 + ci
    );
    bootstrap.pairedDeltas.a4_minus_noProp[cat] = dNoProp;
    bootstrap.pairedDeltas.full4d_minus_noProp[cat] = dNoProp;
    bootstrap.pairedDeltas.a4_minus_fts5Sem[cat] = dFts5Sem;
    bootstrap.pairedDeltas.full4d_minus_fts5Sem[cat] = dFts5Sem;
  }

  return {
    condition: condName,
    compileMs: netM0.metrics.compileMs,
    totalNeurons: netM0.metrics.totalNeurons,
    totalEdges: netM0.metrics.totalEdges,
    variants: variants.map(({ perQuery, ...summary }) => summary),
    fullVariants: variants,
    bootstrap,
    netM0,
  };
}

function runPrivacyProbes(netProtected) {
  const netUnsafe = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    m0Topology: true,
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

// === EXECUTE BENCHMARK SUITE v1.2 ===
const rawMapWithoutSeeds = parseMemoryDirectory(memoryDir, { useSemanticSeeds: false });
const lexicalGap = computeLexicalGap(rawMapWithoutSeeds, queries);

const condA = runCondition('regenerated_blind', seedsRegen);
const condB = runCondition('inline_corpus', null);

const smBreakdown = computeSourceModelBreakdown({ variants: condA.fullVariants });
const privacy = runPrivacyProbes(condA.netM0);

const report = {
  protocol: 'PROTOCOLO_PREREGISTRO_v1.2',
  timestamp: new Date().toISOString(),
  shaVerification,
  dataset: {
    totalNeurons: condA.totalNeurons,
    publicNeurons: Object.values(condA.netM0.neurons).filter((n) => !n.private).length,
    privateNeurons: Object.values(condA.netM0.neurons).filter((n) => n.private).length,
    totalEdges: condA.totalEdges,
    queriesTotal: queries.length,
    clustersTotal: new Set(queries.map((q) => q.cluster)).size,
    byCategory: {
      direct: queries.filter((q) => q.cat === 'direct').length,
      semantic: queries.filter((q) => q.cat === 'semantic').length,
      multihop: queries.filter((q) => q.cat === 'multihop').length,
      episodic: queries.filter((q) => q.cat === 'episodic').length,
    },
    bySourceModel: {
      'sonnet-5.5': queries.filter((q) => q.source_model === 'sonnet-5.5').length,
      'deepseek-v4-flash': queries.filter((q) => q.source_model === 'deepseek-v4-flash').length,
    },
  },
  lexicalGap: {
    totalSemantic: lexicalGap.totalSemantic,
    zeroOverlapCount: lexicalGap.zeroOverlapCount,
    zeroOverlapPct: lexicalGap.zeroOverlapPct,
    meanOverlapTokens: lexicalGap.meanOverlapTokens,
    bySourceModel: lexicalGap.bySourceModel,
    details: lexicalGap.details,
  },
  conditionA: {
    name: 'regenerated_blind',
    compileMs: condA.compileMs,
    totalNeurons: condA.totalNeurons,
    totalEdges: condA.totalEdges,
    variants: condA.variants,
    bootstrap: condA.bootstrap,
  },
  conditionB: {
    name: 'inline_corpus',
    compileMs: condB.compileMs,
    totalNeurons: condB.totalNeurons,
    totalEdges: condB.totalEdges,
    variants: condB.variants,
    bootstrap: condB.bootstrap,
  },
  sourceModelBreakdown: smBreakdown,
  bootstrap: condA.bootstrap,
  privacyFirewall: privacy,
};

fs.writeFileSync(outJsonPath, JSON.stringify(report, null, 2), 'utf8');

// === CONSOLE ASCII/MARKDOWN SUMMARY ===
function printConditionTable(title, cond) {
  console.log(`\n=== ${title} ===`);
  console.log('| Variant                             | Direct@3 | Semantic@3 | MultiHop@3 | Episodic@3 | Overall@3 | EpCtrl@3 (Viol) | Mean ms | P95 ms |');
  console.log('|-------------------------------------|----------|------------|------------|------------|-----------|-----------------|---------|--------|');
  for (const v of cond.variants) {
    const epCtrlStr = `${v.episodicControl.hit3Pct}% (${v.episodicControl.superHubViolations}/${v.episodicControl.n})`;
    console.log(
      `| ${v.label.padEnd(35)} | ${String(v.byCat.direct.hit3Pct + '%').padStart(8)} | ${String(v.byCat.semantic.hit3Pct + '%').padStart(10)} | ${String(v.byCat.multihop.hit3Pct + '%').padStart(10)} | ${String(v.byCat.episodic.hit3Pct + '%').padStart(10)} | ${String(v.overall.hit3Pct + '%').padStart(9)} | ${epCtrlStr.padStart(15)} | ${String(v.latencyMs.mean).padStart(7)} | ${String(v.latencyMs.p95).padStart(6)} |`
    );
  }
}

function printSourceModelTable(title, breakdown) {
  console.log(`\n=== ${title} ===`);
  console.log('| Variant                             | Model             | Direct@3 | Semantic@3 | MultiHop@3 | Episodic@3 | Overall@1 | Overall@3 |');
  console.log('|-------------------------------------|-------------------|----------|------------|------------|------------|-----------|-----------|');
  for (const [vid, entry] of Object.entries(breakdown)) {
    for (const sm of ['sonnet-5.5', 'deepseek-v4-flash']) {
      const stats = entry[sm];
      console.log(
        `| ${entry.label.padEnd(35)} | ${sm.padEnd(17)} | ${String(stats.byCat.direct.hit3Pct + '%').padStart(8)} | ${String(stats.byCat.semantic.hit3Pct + '%').padStart(10)} | ${String(stats.byCat.multihop.hit3Pct + '%').padStart(10)} | ${String(stats.byCat.episodic.hit3Pct + '%').padStart(10)} | ${String(stats.overall.hit1Pct + '%').padStart(9)} | ${String(stats.overall.hit3Pct + '%').padStart(9)} |`
      );
    }
  }
}

console.log('================================================================================');
console.log('  HEBBMESH v1.2 -- PREREGISTERED BLIND BENCHMARK (N=80 QUERIES, 80 CLUSTERS)');
console.log('================================================================================');
console.log('Pre-execution SHA-256 verification: PASS (5/5 files verified)');
console.log(`Dataset: ${report.dataset.totalNeurons} neurons (${report.dataset.publicNeurons} public, ${report.dataset.privateNeurons} private), ${report.dataset.totalEdges} edges`);
console.log(`Queries: ${report.dataset.queriesTotal} total (Direct: ${report.dataset.byCategory.direct}, Semantic: ${report.dataset.byCategory.semantic}, MultiHop: ${report.dataset.byCategory.multihop}, Episodic: ${report.dataset.byCategory.episodic})`);
console.log(`Models: Sonnet-5.5 (n=${report.dataset.bySourceModel['sonnet-5.5']}), DeepSeek-v4-Flash (n=${report.dataset.bySourceModel['deepseek-v4-flash']})`);

console.log('\n[Lexical Gap Analysis (Semantic queries n=30, rawMap without seeds)]');
console.log(`  Zero-overlap queries: ${lexicalGap.zeroOverlapCount}/${lexicalGap.totalSemantic} (${lexicalGap.zeroOverlapPct}%)`);
console.log(`  Mean overlap tokens:  ${lexicalGap.meanOverlapTokens}`);
for (const [sm, smStats] of Object.entries(lexicalGap.bySourceModel)) {
  console.log(`    - ${sm}: zeroOverlap=${smStats.zeroOverlapCount}/${smStats.totalSemantic} (${smStats.zeroOverlapPct}%), meanOverlap=${smStats.meanOverlapTokens}`);
}

printConditionTable('CONDITION A (PRINCIPAL): Blind Regenerated Seeds', condA);
printSourceModelTable('CONDITION A: SOURCE-MODEL BREAKDOWN', smBreakdown);

console.log('\n[Paired Cluster Bootstrap 10,000x on Condition A (Mulberry32 seed=20261123, 95% CI)]');
console.log('-- Comparison: HebbMesh v1.2 A4 vs HebbMesh sin propagacion (a4_minus_noProp) --');
for (const c of BOOT_CATS) {
  const d = condA.bootstrap.pairedDeltas.a4_minus_noProp[c];
  console.log(`  ${c.padEnd(9)}: delta=${d.point >= 0 ? '+' : ''}${d.point} pts | 95% CI [${d.ci95[0]}, ${d.ci95[1]}] | clusters=${d.clusters} | verdict=${d.verdict}`);
}
console.log('-- Comparison: HebbMesh v1.2 A4 vs SQLite FTS5 + | sem: (a4_minus_fts5Sem) --');
for (const c of BOOT_CATS) {
  const d = condA.bootstrap.pairedDeltas.a4_minus_fts5Sem[c];
  console.log(`  ${c.padEnd(9)}: delta=${d.point >= 0 ? '+' : ''}${d.point} pts | 95% CI [${d.ci95[0]}, ${d.ci95[1]}] | clusters=${d.clusters} | verdict=${d.verdict}`);
}

printConditionTable('CONDITION B (COMPARATIVE): Inline Corpus Seeds', condB);

console.log('\n[F-01 Privacy Firewall: 16 Adversarial Probes (Mandatory Positive Control)]');
console.log(`  Control Positivo (Firewall OFF): ${privacy.positiveControlTriggered}/${privacy.totalProbes} triggered PRIV-*`);
console.log(`  Firewall ON:                     ${privacy.firewallOnLeaks}/${privacy.positiveControlTriggered} leaks | lateralDendriteLeaks=${privacy.lateralDendriteLeaks} | PASS=${privacy.pass}`);
console.log(`Saved JSON report: ${outJsonPath}\n`);
