import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  compileNetwork,
  activateNetworkInMemory,
  findBridge,
} from '../src/engine.mjs';

/**
 * HebbMesh Protocolo Ablacion v1.2 — Reproducible Synaptic Ablation Suite (0 npm deps)
 *
 * Evaluates 80 blind external queries (demo/queries_blind.json, 4 categories)
 * across 2 seed conditions:
 *   - Condition A: Regenerated blind seeds (demo/seeds_regenerated.json)
 *   - Condition B: Inline corpus seeds (| sem:, no override)
 *
 * Across 6 ablation variants:
 *   - A0: netLegacy (m0Topology: false, disablePropagation: true, 0-hop)
 *   - A1: netM0     (m0Topology: true, disablePropagation: true, 0-hop)
 *   - A2: netM0     (2-hop, raw: no M2/M3/M4 gating/kWTA/firewall)
 *   - A3: netM0     (2-hop, +M2 valence gating)
 *   - A4: netM0     (2-hop, +M2 gating + M3 kWTA + M4 inter-lobe firewall)
 *   - A4_GABA: netM0 (+M2 + M3 + M4 + GABA soft braking on high degree)
 *
 * Plus:
 *   - F-01 Privacy Firewall check on Condition A netM0 (16 adversarial probes)
 *   - Mandatory Positive Control (Firewall OFF vs Firewall ON)
 *   - Lateral dendrite token isolation check
 *
 * CLI flags:
 *   --json   Outputs complete structured JSON report to stdout
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const memoryDir = path.resolve(rootDir, 'demo', 'memory');

const queriesPath = path.resolve(rootDir, 'demo', 'queries_blind.json');
const seedsRegenPath = path.resolve(rootDir, 'demo', 'seeds_regenerated.json');
const probesPath = path.resolve(rootDir, 'demo', 'privacy_probes.json');

const queries = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
const seedsRegen = JSON.parse(fs.readFileSync(seedsRegenPath, 'utf8'));
const probes = JSON.parse(fs.readFileSync(probesPath, 'utf8'));

const CATS = ['direct', 'semantic', 'multihop', 'episodic'];

function round1(n) {
  return Math.round(n * 10) / 10;
}

function round2(n) {
  return Math.round(n * 100) / 100;
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

const VARIANT_SPECS = [
  {
    id: 'A0',
    label: 'A0: netLegacy (0-hop)',
    netKey: 'netLegacy',
    opts: { topK: 5, disablePropagation: true, maxEpisodic: 1 },
  },
  {
    id: 'A1',
    label: 'A1: netM0 (0-hop)',
    netKey: 'netM0',
    opts: { topK: 5, disablePropagation: true, maxEpisodic: 1 },
  },
  {
    id: 'A2',
    label: 'A2: netM0 (2-hop, raw)',
    netKey: 'netM0',
    opts: { topK: 5, maxHops: 2, useM2Gating: false, useM3KWta: false, useM4Firewall: false, maxEpisodic: 1 },
  },
  {
    id: 'A3',
    label: 'A3: netM0 (2-hop, +M2)',
    netKey: 'netM0',
    opts: { topK: 5, maxHops: 2, useM2Gating: true, useM3KWta: false, useM4Firewall: false, maxEpisodic: 1 },
  },
  {
    id: 'A4',
    label: 'A4: netM0 (2-hop, +M2+M3+M4)',
    netKey: 'netM0',
    opts: { topK: 5, maxHops: 2, useM2Gating: true, useM3KWta: true, useM4Firewall: true, maxEpisodic: 1 },
  },
  {
    id: 'A4_GABA',
    label: 'A4_GABA: netM0 (+M2+M3+M4+GABA)',
    netKey: 'netM0',
    opts: { topK: 5, maxHops: 2, useM2Gating: true, useM3KWta: true, useM4Firewall: true, useGabaSoft: true, maxEpisodic: 1 },
  },
];

function evaluateVariant(id, label, net, runOpts, queryList = queries) {
  const perQuery = [];
  const latencies = [];
  const cats = {
    direct: { n: 0, h1: 0, h3: 0, h5: 0 },
    semantic: { n: 0, h1: 0, h3: 0, h5: 0 },
    multihop: { n: 0, h1: 0, h3: 0, h5: 0 },
    episodic: { n: 0, h1: 0, h3: 0, h5: 0 },
  };

  for (const q of queryList) {
    const res = activateNetworkInMemory(net, q.query, runOpts);
    latencies.push(res.ms);
    const codes = res.winners.map((w) => w.code);
    const exp = new Set(q.expected);
    const in1 = codes.slice(0, 1).some((c) => exp.has(c)) ? 1 : 0;
    const in3 = codes.slice(0, 3).some((c) => exp.has(c)) ? 1 : 0;
    const in5 = codes.slice(0, 5).some((c) => exp.has(c)) ? 1 : 0;

    if (!cats[q.cat]) cats[q.cat] = { n: 0, h1: 0, h3: 0, h5: 0 };
    cats[q.cat].n++;
    cats[q.cat].h1 += in1;
    cats[q.cat].h3 += in3;
    cats[q.cat].h5 += in5;

    perQuery.push({
      id: q.id,
      cat: q.cat,
      cluster: q.cluster,
      in1,
      in3,
      in5,
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
    compileMs: net.metrics.compileMs,
    totalNeurons: net.metrics.totalNeurons,
    totalEdges: net.metrics.totalEdges,
    byCat,
    overall,
    latencyMs: {
      mean: round2(mean(latencies)),
      p95: round2(percentile(latencies, 95)),
    },
    perQuery,
  };
}

function runCondition(condName, seedsOverride = null) {
  const netLegacy = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    m0Topology: false,
    noPersist: true,
    ...(seedsOverride ? { seedsOverride } : {}),
  });

  const netM0 = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    m0Topology: true,
    noPersist: true,
    ...(seedsOverride ? { seedsOverride } : {}),
  });

  const networks = { netLegacy, netM0 };

  const variants = VARIANT_SPECS.map((spec) => {
    const net = networks[spec.netKey];
    return evaluateVariant(spec.id, spec.label, net, spec.opts, queries);
  });

  return {
    name: condName,
    netLegacy,
    netM0,
    variants,
  };
}

function runPrivacyProbes(netProtected, seedsOverride = null) {
  const netUnsafe = compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    m0Topology: true,
    disablePrivacyFirewall: true,
    noPersist: true,
    ...(seedsOverride ? { seedsOverride } : {}),
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

function formatHit(h1, h3) {
  return `${h1}% / ${h3}%`;
}

function printTable(title, variants) {
  console.log(`\n=== ${title} ===`);
  const headers = [
    'Variant',
    'Direct @1/@3',
    'Semantic @1/@3',
    'MultiHop @1/@3',
    'Episodic @1/@3',
    'Overall @1/@3',
    'Latency mean/p95',
    'Comp (ms)',
    'Graph (N/E)',
  ];
  const colWidths = [32, 14, 16, 16, 16, 15, 18, 11, 13];

  const headerLine = '| ' + headers.map((h, i) => h.padEnd(colWidths[i])).join(' | ') + ' |';
  const sepLine = '|-' + colWidths.map((w) => '-'.repeat(w)).join('-|-') + '-|';
  console.log(headerLine);
  console.log(sepLine);

  for (const v of variants) {
    const direct = formatHit(v.byCat.direct.hit1Pct, v.byCat.direct.hit3Pct);
    const semantic = formatHit(v.byCat.semantic.hit1Pct, v.byCat.semantic.hit3Pct);
    const multihop = formatHit(v.byCat.multihop.hit1Pct, v.byCat.multihop.hit3Pct);
    const episodic = formatHit(v.byCat.episodic.hit1Pct, v.byCat.episodic.hit3Pct);
    const overall = formatHit(v.overall.hit1Pct, v.overall.hit3Pct);
    const lat = `${v.latencyMs.mean} / ${v.latencyMs.p95} ms`;
    const comp = `${v.compileMs} ms`;
    const graph = `${v.totalNeurons} / ${v.totalEdges}`;

    const row = [
      v.label.padEnd(colWidths[0]),
      direct.padStart(colWidths[1]),
      semantic.padStart(colWidths[2]),
      multihop.padStart(colWidths[3]),
      episodic.padStart(colWidths[4]),
      overall.padStart(colWidths[5]),
      lat.padStart(colWidths[6]),
      comp.padStart(colWidths[7]),
      graph.padStart(colWidths[8]),
    ];
    console.log('| ' + row.join(' | ') + ' |');
  }
}

function printPrivacySummary(privacy) {
  console.log('\n=== F-01 PRIVACY FIREWALL VERIFICATION (CONDITION A netM0) ===');
  console.log(`Total Probes Tested:               ${privacy.totalProbes}`);
  console.log(`Positive Control (Firewall OFF):   ${privacy.positiveControlTriggered}/${privacy.totalProbes} triggered PRIV-* (Expected: ${privacy.totalProbes}/${privacy.totalProbes})`);
  console.log(`Firewall ON Leaks:                 ${privacy.firewallOnLeaks}/${privacy.totalProbes} (Expected: 0)`);
  console.log(`Lateral Dendrite Leaks:            ${privacy.lateralDendriteLeaks} (Expected: 0)`);
  console.log(`Firewall Verification Verdict:     ${privacy.pass ? 'PASS (CONFIRMED ZERO LEAKS)' : 'FAIL'}`);
  console.log('\n| Probe | Vector              | Mode     | Target | Pos Control | Firewall ON | Result |');
  console.log('|-------|---------------------|----------|--------|-------------|-------------|--------|');
  for (const d of privacy.details) {
    const posStr = d.positiveControlHit ? 'TRIGGERED' : 'MISSED';
    const safeStr = d.firewallOnLeak ? 'LEAK' : 'BLOCKED';
    const resStr = (d.positiveControlHit && !d.firewallOnLeak) ? 'PASS' : 'FAIL';
    console.log(`| ${d.id.padEnd(5)} | ${d.vector.padEnd(19)} | ${d.mode.padEnd(8)} | ${d.target.padEnd(6)} | ${posStr.padStart(11)} | ${safeStr.padStart(11)} | ${resStr.padStart(6)} |`);
  }
}

// === RUN SUITE ===
const isJsonMode = process.argv.includes('--json');

const condA = runCondition('regenerated_blind', seedsRegen);
const condB = runCondition('inline_corpus', null);
const privacy = runPrivacyProbes(condA.netM0, seedsRegen);

const report = {
  protocol: 'HEBBMESH_ABLATION_v1.2',
  timestamp: new Date().toISOString(),
  dataset: {
    queriesTotal: queries.length,
    probesTotal: probes.length,
    memoryDir,
  },
  conditionA_regenerated_blind: {
    compile: {
      netLegacy: condA.netLegacy.metrics,
      netM0: condA.netM0.metrics,
    },
    variants: condA.variants.map((v) => ({
      id: v.id,
      label: v.label,
      compileMs: v.compileMs,
      totalNeurons: v.totalNeurons,
      totalEdges: v.totalEdges,
      byCat: v.byCat,
      overall: v.overall,
      latencyMs: v.latencyMs,
    })),
  },
  conditionB_inline_corpus: {
    compile: {
      netLegacy: condB.netLegacy.metrics,
      netM0: condB.netM0.metrics,
    },
    variants: condB.variants.map((v) => ({
      id: v.id,
      label: v.label,
      compileMs: v.compileMs,
      totalNeurons: v.totalNeurons,
      totalEdges: v.totalEdges,
      byCat: v.byCat,
      overall: v.overall,
      latencyMs: v.latencyMs,
    })),
  },
  privacyFirewall: {
    totalProbes: privacy.totalProbes,
    positiveControlTriggered: privacy.positiveControlTriggered,
    firewallOnLeaks: privacy.firewallOnLeaks,
    lateralDendriteLeaks: privacy.lateralDendriteLeaks,
    pass: privacy.pass,
    details: privacy.details,
  },
};

if (isJsonMode) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log('=========================================================================================================');
  console.log('  HEBBMESH v1.2 — SYNAPTIC ABLATION SUITE (80 BLIND QUERIES, 6 VARIANTS, F-01 PRIVACY VERIFICATION)');
  console.log('=========================================================================================================');
  printTable('CONDITION A: Blind Regenerated Seeds (demo/seeds_regenerated.json)', condA.variants);
  printTable('CONDITION B: Inline Corpus Seeds (| sem: no override)', condB.variants);
  printPrivacySummary(privacy);
  console.log('=========================================================================================================\n');
}
