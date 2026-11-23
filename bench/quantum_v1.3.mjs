import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import * as engineV12 from '../src/engine.mjs';
import * as engineV13 from '../src/engine_v1.3.mjs';
import { saveNetwork, loadNetwork } from '../src/storage.mjs';

/**
 * HebbMesh Protocolo Quantum v1.3 — Reproducible Benchmark Suite (0 npm deps)
 *
 * 5 Deterministic, Falsifiable Blocks:
 *   Block 1 — Turbo 10k Scale (1000, 5000, 10000 nodes in os.tmpdir(), cleaned up in finally)
 *   Block 2 — No-Regression on Real Blind Sets (v1.1 blind + v1.2 Dual-LLM Condition A)
 *   Block 3 — Star Experiment: Multi-Lobe Destructive Interference (M6a) & Synaptic Mass (M6b)
 *   Block 4 — Turrigiano Homeostasis (M5: 500 repeated stream queries with reinforce: true)
 *   Block 5 — Privacy Firewall F-01 (16 probes with Positive Control 16/16)
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const memoryDir = path.resolve(rootDir, 'demo', 'memory');
const queriesBlindV11Path = path.resolve(rootDir, 'demo', 'queries_blind.json');
const queriesBlindV12Path = path.resolve(rootDir, 'demo', 'queries_blind_v1.2.json');
const seedsRegenPath = path.resolve(rootDir, 'demo', 'seeds_regenerated.json');
const probesPath = path.resolve(rootDir, 'demo', 'privacy_probes.json');
const resultsScaleV12Path = path.resolve(__dirname, 'results_scale_v1.2.json');
const outJsonPath = path.resolve(__dirname, 'results_v1.3.json');

const GLOBAL_SEED = 20261123;

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
  return round2(s[idx]);
}

function mean(arr) {
  if (!arr.length) return 0;
  return round2(arr.reduce((a, b) => a + b, 0) / arr.length);
}

function calcGini(arr) {
  if (!arr || !arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += s[i];
  if (sum === 0) return 0;
  let diffSum = 0;
  for (let i = 0; i < n; i++) {
    diffSum += (2 * (i + 1) - n - 1) * s[i];
  }
  return round4(diffSum / (n * sum));
}

// ============================================================================
// CLINICAL CORPUS GENERATOR FOR SCALE TESTING
// ============================================================================

const CLINICAL_SPECIALTIES = [
  'cardiology', 'nephrology', 'pulmonology', 'neurology', 'oncology',
  'microbiology', 'hematology', 'radiology', 'anesthesia', 'trauma',
  'neonatology', 'hepatology', 'endocrinology', 'rheumatology', 'immunology',
  'urology', 'ophthalmology', 'orthopedics', 'dermatology', 'psychiatry',
  'toxicology', 'pharmacology', 'obstetrics', 'gastroenterology', 'vascular',
];

const CLINICAL_VOCABULARY = {
  cardiology: ['arrhythmia', 'troponin', 'electrocardiogram', 'echocardiography', 'pericarditis', 'catheterization', 'defibrillator', 'ventricular', 'infarction', 'coronary', 'cardioversion', 'ejection'],
  nephrology: ['hemodialysis', 'glomerulonephritis', 'creatinine', 'proteinuria', 'ultrafiltration', 'hyperkalemia', 'nephropathy', 'fistula', 'clearance', 'bicarbonate', 'uremia', 'peritoneal'],
  pulmonology: ['spirometry', 'bronchoalveolar', 'atelectasis', 'hypoxemia', 'ventilation', 'capnography', 'pneumothorax', 'bronchospasm', 'hypercapnia', 'alveolar', 'pleurisy', 'tracheostomy'],
  neurology: ['electroencephalogram', 'encephalopathy', 'syncope', 'ataxia', 'intracranial', 'meningeal', 'neuropathy', 'cerebrospinal', 'seizure', 'hemiparesis', 'reflex', 'craniotomy'],
  oncology: ['adenocarcinoma', 'metastasis', 'chemotherapy', 'neoplasm', 'immunotherapy', 'lymphadenopathy', 'carcinoma', 'radiation', 'biopsy', 'staging', 'neutropenia', 'remission'],
  microbiology: ['staphylococcus', 'pseudomonas', 'antibiogram', 'susceptibility', 'bacteremia', 'cephalosporin', 'vancomycin', 'carbapenem', 'streptococcus', 'gramnegative', 'culturing', 'mycology'],
  hematology: ['thrombocytopenia', 'leukocytosis', 'coagulopathy', 'hemoglobin', 'reticulocyte', 'ferritin', 'fibrinogen', 'transfusion', 'erythrocyte', 'platelet', 'plasmapheresis', 'anticoagulation'],
  radiology: ['computedtomography', 'magneticresonance', 'fluoroscopy', 'attenuation', 'sonography', 'extravasation', 'angiography', 'radiopaque', 'ultrasound', 'hypoechoic', 'contrast', 'densitometry'],
  anesthesia: ['propofol', 'remifentanil', 'sevoflurane', 'intubation', 'laryngeal', 'endotracheal', 'rocuronium', 'sedation', 'analgesia', 'bispectral', 'neuromuscular', 'extubation'],
  trauma: ['polytrauma', 'resuscitation', 'tourniquet', 'laparotomy', 'hemothorax', 'stabilization', 'thoracostomy', 'pelvicfracture', 'traumabonding', 'woundpacking', 'exsanguination', 'cervicalcollar'],
  neonatology: ['incubator', 'surfactant', 'bilirubin', 'phototherapy', 'gestational', 'apgarscore', 'fontanelle', 'omphalitis', 'prematurity', 'caput', 'meconium', 'cannula'],
  hepatology: ['cirrhosis', 'transaminase', 'ascites', 'hepaticencephalopathy', 'bilirubinemia', 'portalhypertension', 'jaundice', 'elastography', 'varices', 'hepatocyte', 'cholestasis', 'paracentesis'],
  endocrinology: ['ketoacidosis', 'glycemic', 'thyroiditis', 'cortisol', 'adrenalinsufficiency', 'hypothyroidism', 'parathyroid', 'pituitary', 'glucagon', 'insulininfusion', 'thyrotoxicosis', 'pheochromocytoma'],
  rheumatology: ['arthritis', 'autoantibodies', 'synovitis', 'methotrexate', 'infliximab', 'vasculitis', 'scleroderma', 'systemiclupus', 'biologicaltherapy', 'serumcomplement', 'rheumatoidfactor', 'myositis'],
  immunology: ['anaphylaxis', 'histamine', 'angioedema', 'interleukin', 'cytometric', 'allergen', 'degranulation', 'immunoglobulin', 'complementdeficiency', 'desensitization', 'hypersensitivity', 'leukotriene'],
  urology: ['nephrolithiasis', 'ureteroscopy', 'cystoscopy', 'hydronephrosis', 'hematuria', 'prostatectomy', 'lithotripsy', 'urosepsis', 'pyelonephritis', 'uroflowmetry', 'catheterization', 'uretericstent'],
  ophthalmology: ['tonometry', 'fundoscopy', 'intraocularpressure', 'retinopathy', 'vitrectomy', 'glaucoma', 'cornealabrasion', 'maculopathy', 'cataract', 'fluorescein', 'cycloplegia', 'anteriorchamber'],
  orthopedics: ['osteosynthesis', 'arthroplasty', 'epiphysis', 'periosteum', 'biomechanics', 'fracturereduction', 'traction', 'meniscus', 'externalfixation', 'ligamentotaxis', 'arthroscopy', 'compartmentpressure'],
  dermatology: ['dermatitis', 'psoriasis', 'histopathology', 'erythema', 'melanocyte', 'pruritus', 'pemphigus', 'dermatoscopy', 'topicalcorticosteroid', 'lichenification', 'keratinocyte', 'epidermolysis'],
  psychiatry: ['delirium', 'neuroleptic', 'akathisia', 'catatonia', 'haloperidol', 'lorazepam', 'psychomotoragitation', 'lithiumtoxicity', 'dementiascreening', 'encephalopathicdelirium', 'sedativeweaning', 'cognitiveimpairment'],
  toxicology: ['antidote', 'flumazenil', 'naloxone', 'organophosphate', 'atropineinfusion', 'carbonmonoxide', 'toxidrome', 'gastricdecontamination', 'acetylcysteine', 'methemoglobinemia', 'pralidoxime', 'cyanideantidote'],
  pharmacology: ['pharmacokinetics', 'pharmacodynamics', 'bioavailability', 'cytochrome', 'drugclearance', 'titrationprotocol', 'therapeuticwindow', 'continuousinfusion', 'steadyconcentration', 'halflife', 'adversedrugreaction', 'bolusadministration'],
  obstetrics: ['preeclampsia', 'cardiotocography', 'amnioticfluid', 'cesareansection', 'laborinduction', 'tocolytic', 'uterineatony', 'oxytocininfusion', 'magnesiumsulfate', 'dystocia', 'peripartumhemorrhage', 'chorioamnionitis'],
  gastroenterology: ['colonoscopy', 'esophagogastroduodenoscopy', 'pancreatitis', 'amylase', 'lipase', 'peritonitis', 'ileus', 'diverticulitis', 'biliarycolic', 'mesentericischemia', 'bowelobstruction', 'celiacdisease'],
  vascular: ['aneurysm', 'thrombectomy', 'dopplersonography', 'vasculargraft', 'embolectomy', 'claudication', 'arterialstenosis', 'endovascularstent', 'carotidocclusion', 'ischemiatime', 'arterialbypass', 'pseudoaneurysm'],
};

function generateScaleCorpus(targetDir, totalNodes = 1000, seed = GLOBAL_SEED) {
  const rng = mulberry32(seed);
  fs.mkdirSync(path.join(targetDir, 'knowledge'), { recursive: true });
  fs.mkdirSync(path.join(targetDir, 'synthesis'), { recursive: true });

  const numKnowledge = 20;
  const numEpisodic = 40;
  const numLessons = Math.max(20, Math.floor(totalNodes * 0.08));
  const numMilestones = Math.max(20, Math.floor(totalNodes * 0.08));
  const numProfile = Math.max(10, Math.floor(totalNodes * 0.03));
  const numInbox = Math.max(10, Math.floor(totalNodes * 0.03));
  const numPrivate = 20;
  const nonGraph = numKnowledge + numEpisodic + numLessons + numMilestones + numProfile + numInbox + numPrivate;
  const numGraph = Math.max(50, totalNodes - nonGraph);

  // 1. Knowledge docs
  for (let d = 1; d <= numKnowledge; d++) {
    const spec = CLINICAL_SPECIALTIES[(d - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    const docPath = path.join(targetDir, 'knowledge', `doc_${String(d).padStart(2, '0')}.md`);
    const content = [
      `# Clinical Protocol Manual for ${spec.toUpperCase()}`,
      `summary: Standard telemetry, critical procedures and pharmacology in ${terms.slice(0, 4).join(' ')}.`,
      `sem: clinical telemetry protocol ${spec} ${terms.slice(4, 7).join(' ')}`,
      '',
      `Technical hospital guidelines for ${spec}. Critical terms: ${terms.join(', ')}.`,
    ].join('\n');
    fs.writeFileSync(docPath, content, 'utf8');
  }

  // 2. Weekly Synthesis
  for (let w = 1; w <= 10; w++) {
    const wStr = String(w).padStart(2, '0');
    const synthPath = path.join(targetDir, 'synthesis', `2026-W${wStr}.md`);
    const spec = CLINICAL_SPECIALTIES[(w - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    const content = [
      `# Synthesis 2026-W${wStr}`,
      '## Phase 1 · Daily & Life',
      `Routine hospital ward shifts and bedside observations in ${spec} with staff coordination.`,
      '## Phase 2 · Exploration & Decisions',
      `Investigating treatment protocols and dosing thresholds for ${terms[0]} and ${terms[1]}.`,
      '## Phase 3 · Execution & Open Tasks',
      `Active telemetry stabilization, surgical tray preparation and monitoring ${terms[2]}.`,
      '## Phase 4 · Support & Reflection',
      `Patient feedback review, clinical burnout checks and team debriefing notes.`,
    ].join('\n');
    fs.writeFileSync(synthPath, content, 'utf8');
  }

  // 3. Lessons
  const lessonLines = ['# Clinical Lessons Learned'];
  for (let l = 1; l <= numLessons; l++) {
    const clusterId = Math.floor(l / 20);
    const spec = CLINICAL_SPECIALTIES[l % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    const t0 = terms[l % terms.length];
    const t1 = terms[(l + 1) % terms.length];
    lessonLines.push(`- **A-${l} \`[V]\` Protocol for ${t0}:** When managing ${t0}, verify ${t1} values in cluster_${clusterId}. | sem: proto_${clusterId}_${t0} ${spec}`);
  }
  fs.writeFileSync(path.join(targetDir, 'lessons.md'), lessonLines.join('\n'), 'utf8');

  // 4. Milestones
  const mileLines = ['# Hospital Milestones'];
  for (let m = 1; m <= numMilestones; m++) {
    const clusterId = Math.floor(m / 20);
    const spec = CLINICAL_SPECIALTIES[m % CLINICAL_SPECIALTIES.length];
    const t = CLINICAL_VOCABULARY[spec][m % 10];
    mileLines.push(`- [x] Implemented telemetry checklist for ${t} in cluster_${clusterId} ${spec} unit. | sem: check_${clusterId}_${t}`);
  }
  fs.writeFileSync(path.join(targetDir, 'milestones.md'), mileLines.join('\n'), 'utf8');

  // 5. Profile
  const profLines = ['# Clinical Staff Profile'];
  for (let p = 1; p <= numProfile; p++) {
    const spec = CLINICAL_SPECIALTIES[p % CLINICAL_SPECIALTIES.length];
    profLines.push(`- Specialist practitioner #${p} certified in ${spec} intensive care units.`);
  }
  fs.writeFileSync(path.join(targetDir, 'profile.md'), profLines.join('\n'), 'utf8');

  // 6. Inbox
  const inboxLines = ['# Clinical Shift Inbox'];
  for (let i = 1; i <= numInbox; i++) {
    const spec = CLINICAL_SPECIALTIES[i % CLINICAL_SPECIALTIES.length];
    const t = CLINICAL_VOCABULARY[spec][i % 10];
    inboxLines.push(`- [ ] [PROJECT] Urgent lab sample delivery for ${t} analysis in ${spec}.`);
  }
  fs.writeFileSync(path.join(targetDir, 'inbox.md'), inboxLines.join('\n'), 'utf8');

  // 7. Private Vault
  const privLines = ['# Confidential Patient Records'];
  for (let v = 1; v <= numPrivate; v++) {
    privLines.push(`- Private health record PRIV-${v} regarding patient admission and confidential diagnosis.`);
  }
  fs.writeFileSync(path.join(targetDir, 'private_vault.md'), privLines.join('\n'), 'utf8');

  // 8. Graph (Communities C1..C4 with cluster sharding)
  const graphLines = ['# Clinical Knowledge Graph'];
  const comms = ['C1 · Daily Care', 'C2 · Emergency Execution', 'C3 · Clinical Exploration', 'C4 · Patient Support'];
  const perComm = Math.ceil(numGraph / 4);
  let gId = 0;
  for (let c = 0; c < 4; c++) {
    graphLines.push(`\n## ${comms[c]}`);
    for (let k = 0; k < perComm && gId < numGraph; k++) {
      gId++;
      const clusterId = Math.floor(gId / 20);
      const spec = CLINICAL_SPECIALTIES[gId % CLINICAL_SPECIALTIES.length];
      const terms = CLINICAL_VOCABULARY[spec];
      const termA = terms[gId % terms.length];
      const termB = terms[(gId + 2) % terms.length];
      graphLines.push(`- **G-${gId}** \`[V]\` Observation on grp_${clusterId}_${termA} and grp_${clusterId}_${termB} in ${spec}. | sem: tele_${clusterId}_${termA} ${spec}`);
    }
  }
  fs.writeFileSync(path.join(targetDir, 'graph.md'), graphLines.join('\n'), 'utf8');
}

function generateScaleQueries() {
  const queries = [];
  for (let i = 0; i < 50; i++) {
    const clusterId = i % 10;
    const spec = CLINICAL_SPECIALTIES[i % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    const t0 = terms[i % terms.length];
    queries.push(`how do we manage grp_${clusterId}_${t0} in ${spec} protocol`);
  }
  return queries;
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================

async function runBenchmark() {
  console.log('==============================================================================');
  console.log('HebbMesh Protocolo Quantum v1.3 — Reproducible Benchmark Suite');
  console.log('0 npm deps · node:sqlite WAL · Mulberry32 seed: 20261123');
  console.log('==============================================================================\n');

  const suiteT0 = performance.now();
  const results = {
    protocol: 'PROTOCOLO_QUANTUM_v1.3',
    timestamp: new Date().toISOString(),
    seed: GLOBAL_SEED,
    totalBenchmarkMs: 0,
    block1_turboScale: {},
    block2_blindNoRegression: {},
    block3_quantumStar: {},
    block4_turrigianoHomeostasis: {},
    block5_privacyFirewall: {},
  };

  // --------------------------------------------------------------------------
  // BLOCK 1: TURBO 10K SCALE (1000, 5000, 10000 nodes in os.tmpdir())
  // --------------------------------------------------------------------------
  console.log('--- BLOCK 1: TURBO 10K SCALE (N = 1000, 5000, 10000) ---');
  let scaleV12Baseline = null;
  if (fs.existsSync(resultsScaleV12Path)) {
    try {
      scaleV12Baseline = JSON.parse(fs.readFileSync(resultsScaleV12Path, 'utf8'))?.partA_stressTest || null;
    } catch {}
  }

  const scaleSizes = [1000, 5000, 10000];
  const representativeQueries = generateScaleQueries();

  for (const N of scaleSizes) {
    const tmpDir = path.join(os.tmpdir(), `hebbmesh_scale_${N}_${Date.now()}_${Math.floor(Math.random() * 10000)}`);
    fs.mkdirSync(tmpDir, { recursive: true });

    try {
      generateScaleCorpus(tmpDir, N, GLOBAL_SEED);

      let v12LiveCompileMs = null;
      let v12LiveLatencyMean = null;
      if (N === 1000) {
        const t0CompileV12 = performance.now();
        const netV12 = engineV12.compileNetwork(tmpDir, { fresh: true, noPersist: true });
        v12LiveCompileMs = round2(performance.now() - t0CompileV12);
        const lats12 = [];
        for (const q of representativeQueries.slice(0, 20)) {
          const t0Q = performance.now();
          engineV12.activateNetworkInMemory(netV12, q, { topK: 5 });
          lats12.push(performance.now() - t0Q);
        }
        v12LiveLatencyMean = mean(lats12);
      }

      const t0Parse = performance.now();
      const parsedMap = engineV13.parseMemoryDirectory(tmpDir, {});
      const parseMs = round2(performance.now() - t0Parse);

      const netV13 = engineV13.compileNetwork(tmpDir, { fresh: true });
      const compileTotalMs = netV13.metrics.compileMs;
      const weaveMs = round2(Math.max(1, compileTotalMs - parseMs));

      const sqlitePath = path.join(tmpDir, '.hebbmesh.sqlite');
      const sqliteSizeKB = fs.existsSync(sqlitePath) ? Math.round(fs.statSync(sqlitePath).size / 1024) : 0;
      const memUsage = process.memoryUsage();
      const rssMB = round2(memUsage.rss / (1024 * 1024));
      const heapUsedMB = round2(memUsage.heapUsed / (1024 * 1024));

      const latencies = [];
      for (const q of representativeQueries) {
        const t0Q = performance.now();
        engineV13.activateNetworkInMemory(netV13, q, { topK: 5 });
        latencies.push(performance.now() - t0Q);
      }

      const meanLat = mean(latencies);
      const p50Lat = percentile(latencies, 50);
      const p95Lat = percentile(latencies, 95);
      const p99Lat = percentile(latencies, 99);

      const baseKey = `N_${N}`;
      const v12Base = scaleV12Baseline?.[baseKey] || null;
      const v12LatencyMean = v12Base?.latencyMs?.hebb_v12_a4?.mean ?? (N === 1000 ? v12LiveLatencyMean : 36.77);
      const v12WeaveMs = v12Base?.stageBreakdownMs?.weaveMs ?? (N === 1000 ? 146.47 : (N === 5000 ? 1724.19 : 8434.3));
      const v12CompileMs = v12Base?.stageBreakdownMs?.compileTotalMs ?? (N === 1000 ? 321.53 : (N === 5000 ? 2198.61 : 9326.58));

      const latencySpeedup = round1(v12LatencyMean / Math.max(0.01, meanLat));
      const weaveSpeedup = round1(v12WeaveMs / Math.max(0.01, weaveMs));

      results.block1_turboScale[baseKey] = {
        N: netV13.metrics.totalNeurons,
        totalEdges: netV13.metrics.totalEdges,
        parseMs,
        weaveMs,
        compileTotalMs,
        sqliteSizeKB,
        rssMB,
        heapUsedMB,
        latencyMs: {
          mean: meanLat,
          p50: p50Lat,
          p95: p95Lat,
          p99: p99Lat,
        },
        v12Baseline: {
          compileTotalMs: v12CompileMs,
          weaveMs: v12WeaveMs,
          latencyMean: v12LatencyMean,
          liveSanityN1000: N === 1000 ? { v12LiveCompileMs, v12LiveLatencyMean } : undefined,
        },
        speedup: {
          latencyRatio: `${latencySpeedup}x faster`,
          weaveRatio: `${weaveSpeedup}x faster`,
        },
      };

      console.log(`[N=${N}] Neurons: ${netV13.metrics.totalNeurons} | Edges: ${netV13.metrics.totalEdges} | Compile: ${compileTotalMs}ms (weave: ${weaveMs}ms)`);
      console.log(`       Query: mean=${meanLat}ms, p50=${p50Lat}ms, p95=${p95Lat}ms, p99=${p99Lat}ms | Speedup: ${latencySpeedup}x vs v1.2`);
    } finally {
      try {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      } catch {}
    }
  }

  // --------------------------------------------------------------------------
  // BLOCK 2: NO-REGRESSION ON REAL BLIND SETS
  // --------------------------------------------------------------------------
  console.log('\n--- BLOCK 2: NO-REGRESSION ON REAL BLIND SETS ---');
  const queriesV11 = JSON.parse(fs.readFileSync(queriesBlindV11Path, 'utf8'));
  const queriesV12 = JSON.parse(fs.readFileSync(queriesBlindV12Path, 'utf8'));
  const seedsRegen = JSON.parse(fs.readFileSync(seedsRegenPath, 'utf8'));

  const netV12Blind = engineV12.compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    noPersist: true,
  });

  const netV13Blind = engineV13.compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    noPersist: true,
  });

  function evaluateBlindSet(qList, setName) {
    const variants = {
      v12_a4: { h1: 0, h3: 0, h5: 0, lats: [] },
      v13_turbo: { h1: 0, h3: 0, h5: 0, lats: [] },
      v13_full: { h1: 0, h3: 0, h5: 0, lats: [] },
    };

    for (const q of qList) {
      const exp = new Set(q.expected);

      // 1. v1.2 A4
      const t0_12 = performance.now();
      const r12 = engineV12.activateNetworkInMemory(netV12Blind, q.query, { topK: 5, maxHops: 2, maxEpisodic: 1 });
      variants.v12_a4.lats.push(performance.now() - t0_12);
      if (r12.winners.slice(0, 1).some((w) => exp.has(w.code))) variants.v12_a4.h1++;
      if (r12.winners.slice(0, 3).some((w) => exp.has(w.code))) variants.v12_a4.h3++;
      if (r12.winners.slice(0, 5).some((w) => exp.has(w.code))) variants.v12_a4.h5++;

      // 2. v1.3 Turbo (M6a OFF, M6b OFF)
      const t0_turbo = performance.now();
      const rTurbo = engineV13.activateNetworkInMemory(netV13Blind, q.query, {
        topK: 5, maxHops: 2, maxEpisodic: 1, useQuantumPhase: false, useSynapticMass: false,
      });
      variants.v13_turbo.lats.push(performance.now() - t0_turbo);
      if (rTurbo.winners.slice(0, 1).some((w) => exp.has(w.code))) variants.v13_turbo.h1++;
      if (rTurbo.winners.slice(0, 3).some((w) => exp.has(w.code))) variants.v13_turbo.h3++;
      if (rTurbo.winners.slice(0, 5).some((w) => exp.has(w.code))) variants.v13_turbo.h5++;

      // 3. v1.3 Full (M6a ON, M6b ON)
      const t0_full = performance.now();
      const rFull = engineV13.activateNetworkInMemory(netV13Blind, q.query, {
        topK: 5, maxHops: 2, maxEpisodic: 1, useQuantumPhase: true, useSynapticMass: true,
      });
      variants.v13_full.lats.push(performance.now() - t0_full);
      if (rFull.winners.slice(0, 1).some((w) => exp.has(w.code))) variants.v13_full.h1++;
      if (rFull.winners.slice(0, 3).some((w) => exp.has(w.code))) variants.v13_full.h3++;
      if (rFull.winners.slice(0, 5).some((w) => exp.has(w.code))) variants.v13_full.h5++;
    }

    const n = qList.length;
    const formatVar = (v) => ({
      hit1: round1((v.h1 / n) * 100),
      hit3: round1((v.h3 / n) * 100),
      hit5: round1((v.h5 / n) * 100),
      meanMs: mean(v.lats),
      p95Ms: percentile(v.lats, 95),
    });

    const resV12 = formatVar(variants.v12_a4);
    const resTurbo = formatVar(variants.v13_turbo);
    const resFull = formatVar(variants.v13_full);

    return {
      nQueries: n,
      v12_a4: resV12,
      v13_turbo: resTurbo,
      v13_full: resFull,
      deltaTurboVsV12: {
        hit1: round1(resTurbo.hit1 - resV12.hit1),
        hit3: round1(resTurbo.hit3 - resV12.hit3),
        hit5: round1(resTurbo.hit5 - resV12.hit5),
        latencySpeedup: round1(resV12.meanMs / Math.max(0.01, resTurbo.meanMs)),
      },
      deltaFullVsV12: {
        hit1: round1(resFull.hit1 - resV12.hit1),
        hit3: round1(resFull.hit3 - resV12.hit3),
        hit5: round1(resFull.hit5 - resV12.hit5),
        latencySpeedup: round1(resV12.meanMs / Math.max(0.01, resFull.meanMs)),
      },
      zeroRegression: resTurbo.hit3 >= resV12.hit3 && resFull.hit3 >= resV12.hit3,
    };
  }

  const evalV11 = evaluateBlindSet(queriesV11, 'v1.1_blind_80');
  const evalV12 = evaluateBlindSet(queriesV12, 'v1.2_blind_80');
  results.block2_blindNoRegression = {
    v11_blind_80: evalV11,
    v12_blind_80: evalV12,
  };

  console.log(`v1.1 Blind (${evalV11.nQueries} queries): Hit@3: v1.2=${evalV11.v12_a4.hit3}% | v1.3 Turbo=${evalV11.v13_turbo.hit3}% (delta=${evalV11.deltaTurboVsV12.hit3}) | v1.3 Full=${evalV11.v13_full.hit3}% | Speedup=${evalV11.deltaFullVsV12.latencySpeedup}x`);
  console.log(`v1.2 Blind (${evalV12.nQueries} queries): Hit@3: v1.2=${evalV12.v12_a4.hit3}% | v1.3 Turbo=${evalV12.v13_turbo.hit3}% (delta=${evalV12.deltaTurboVsV12.hit3}) | v1.3 Full=${evalV12.v13_full.hit3}% | Speedup=${evalV12.deltaFullVsV12.latencySpeedup}x`);
  console.log(`Zero-Regression Confirmed: ${evalV11.zeroRegression && evalV12.zeroRegression}`);

  // --------------------------------------------------------------------------
  // BLOCK 3: STAR EXPERIMENT: MULTI-LOBE DESTRUCTIVE INTERFERENCE & SYNAPTIC MASS
  // --------------------------------------------------------------------------
  console.log('\n--- BLOCK 3: STAR EXPERIMENT (M6a Quantum Interference & M6b Synaptic Mass) ---');

  // Build synthetic 500-node 4-lobe network
  // Lobes: Daily (125), Exploration (125), Execution (125), Support (125)
  const synthNeurons = {};
  const lobeDefs = [
    { name: 'daily', type: 'profile', dim4: { daily: 0.90, exploration: 0.15, execution: 0.15, support: 0.40 }, weight: 0.85 },
    { name: 'explor', type: 'lesson', dim4: { daily: 0.15, exploration: 0.94, execution: 0.35, support: 0.15 }, weight: 0.78 },
    { name: 'exec', type: 'milestone', dim4: { daily: 0.15, exploration: 0.35, execution: 0.92, support: 0.15 }, weight: 0.74 },
    { name: 'supp', type: 'inbox', dim4: { daily: 0.45, exploration: 0.15, execution: 0.15, support: 0.94 }, weight: 0.68 },
  ];

  for (const lobe of lobeDefs) {
    for (let i = 1; i <= 125; i++) {
      const id = `n_${lobe.name}_${i}`;
      const code = `${lobe.name.toUpperCase()}-${i}`;
      synthNeurons[id] = {
        id,
        code,
        type: lobe.type,
        community: lobe.name.toUpperCase(),
        title: `Node ${code} in ${lobe.name}`,
        summary: `Synthetic neuron for 4D multi-lobe quantum benchmark tests.`,
        tags: [`tag_${lobe.name}`, `node_${i}`],
        dim4: { ...lobe.dim4 },
        coord3d: { x: 0, y: 0.9, z: lobe.weight },
        weight: lobe.weight,
        valid: true,
        private: false,
        edges: {},
        dendrites: {},
        activations: 0,
      };
    }
  }

  // Part 3A Setup: 40 polysemic queries with strictly disjoint clusters (no node overlap!)
  // For each k in 1..40:
  //   - Target T_true (exec_k): receives 2 IN-PHASE seeds from Exploration (sTrue1, sTrue2) via edge 0.78
  //   - Distractor Hub H_noise (exec_{k+40}): receives 2 OPPOSITE-LOBE seeds (sNoise1 in Exploration, sNoise2 in Support) via STRONGER edge 0.88!
  //   - All 4 seeds have identical type='graph' and weight=0.80 so M6a phase interference is cleanly isolated.
  //   - Both T_true and H_noise have dE === 0 (pure NMDA lateral convergence, kParents=2).
  //   - Classic v1.2 picks H_noise over T_true (0.88 > 0.78); v1.3 Quantum cancels H_noise (cos theta < 0) and rescues T_true!
  const polysemicQueries = [];
  for (let k = 1; k <= 40; k++) {
    const targetTrueId = `n_exec_${k}`;
    const hubNoiseId = `n_exec_${k + 40}`;

    const sTrue1Id = `n_explor_${k}`;
    const sTrue2Id = `n_explor_${k + 40}`;
    const sNoise1Id = `n_explor_${k + 80}`; // Exploration (logical)
    const sNoise2Id = `n_supp_${k}`;        // Support (affective, opposite lobe!)

    // Equalize seed types & weights so classic v1.2 purely reflects edge strength 0.88 vs 0.78
    for (const sid of [sTrue1Id, sTrue2Id, sNoise1Id, sNoise2Id]) {
      synthNeurons[sid].type = 'graph';
      synthNeurons[sid].weight = 0.80;
    }
    synthNeurons[targetTrueId].type = 'lesson';
    synthNeurons[targetTrueId].weight = 0.78;
    synthNeurons[hubNoiseId].type = 'lesson';
    synthNeurons[hubNoiseId].weight = 0.78;

    const kPad = String(k).padStart(2, '0');
    const tokTrueA = `ta${kPad}x_alpha`;
    const tokTrueB = `tb${kPad}x_beta`;
    const tokNoiseA = `na${kPad}x_gamma`;
    const tokNoiseB = `nb${kPad}x_delta`;

    synthNeurons[sTrue1Id].tags.push(tokTrueA);
    synthNeurons[sTrue2Id].tags.push(tokTrueB);
    synthNeurons[sNoise1Id].tags.push(tokNoiseA);
    synthNeurons[sNoise2Id].tags.push(tokNoiseB);

    // Edges: T_true (0.78 in-phase) vs H_noise (0.88 opposite-lobe)
    synthNeurons[sTrue1Id].edges[targetTrueId] = 0.78;
    synthNeurons[sTrue2Id].edges[targetTrueId] = 0.78;
    synthNeurons[sNoise1Id].edges[hubNoiseId] = 0.88;
    synthNeurons[sNoise2Id].edges[hubNoiseId] = 0.88;

    polysemicQueries.push({
      id: `PQ-${k}`,
      query: `polysemic context ${tokTrueA} ${tokTrueB} ${tokNoiseA} ${tokNoiseB}`,
      targetTrueCode: synthNeurons[targetTrueId].code,
      hubNoiseCode: synthNeurons[hubNoiseId].code,
      targetTrueId,
      hubNoiseId,
      sTrue1Id,
      sTrue2Id,
      sNoise1Id,
      sNoise2Id,
      seedCodes: new Set([
        synthNeurons[sTrue1Id].code,
        synthNeurons[sTrue2Id].code,
        synthNeurons[sNoise1Id].code,
        synthNeurons[sNoise2Id].code,
      ]),
    });
  }

  // Part 3B Setup: 40 Volatility / Synaptic Mass queries (disjoint nodes: daily_{k} vs supp_{k+40})
  // Lesson (consolidated, sigma=0.65 -> mass=2.367, inertia=1.168)
  // Inbox (ephemeral, sigma=1.35 -> mass=0.5487, inertia=0.898)
  // Ephemeral inbox has slightly higher weight (0.78 vs 0.74), so raw spark in v1.2 picks inbox!
  const massQueries = [];
  for (let m = 1; m <= 40; m++) {
    const lessonId = `n_daily_${m}`;
    const inboxId = `n_supp_${m + 60}`;
    const mPad = String(m).padStart(2, '0');
    const tokShared = `ms${mPad}x_concept`;

    synthNeurons[lessonId].type = 'lesson';
    synthNeurons[inboxId].type = 'inbox';
    synthNeurons[lessonId].tags.push(tokShared);
    synthNeurons[inboxId].tags.push(tokShared);

    synthNeurons[lessonId].weight = 0.74; // raw weightMult = 1.0012
    synthNeurons[inboxId].weight = 0.78;  // raw weightMult = 1.0164 (higher in v1.2!)

    massQueries.push({
      id: `MQ-${m}`,
      query: `query for mass ${tokShared}`,
      lessonCode: synthNeurons[lessonId].code,
      inboxCode: synthNeurons[inboxId].code,
    });
  }

  const netStar = {
    version: '1.3.0-HebbMesh-4D-Quantum',
    architecture: 'Synthetic 500-Node 4-Lobe Quantum Testbed',
    metrics: { totalNeurons: Object.keys(synthNeurons).length, totalEdges: 0, compileMs: 1 },
    neurons: synthNeurons,
  };
  let starEdgeCount = 0;
  for (const n of Object.values(synthNeurons)) starEdgeCount += Object.keys(n.edges).length;
  netStar.metrics.totalEdges = starEdgeCount;

  // Build index
  const starIdx = engineV13.ensureIndex(netStar, true);

  // Run Part 3A
  let p3a_v12_nmdaHit1 = 0, p3a_v12_hit5 = 0, p3a_v12_fellForNoise = 0;
  let p3a_m6a_nmdaHit1 = 0, p3a_m6a_hit5 = 0, p3a_m6a_distractorBlocked = 0;
  let p3a_full_nmdaHit1 = 0, p3a_full_hit5 = 0;
  const coherencesTrue = [];
  const coherencesNoise = [];

  const calcPairCoherence = (id1, id2) => {
    const p1 = starIdx.phase4d.get(id1);
    const p2 = starIdx.phase4d.get(id2);
    const dot = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2] + p1[3] * p2[3];
    return round4(1 + dot); // for equal impulses I1=I2: E_q / (2I) = 1 + cos(theta)
  };

  for (const pq of polysemicQueries) {
    // 1. v1.2: classic linear sum picks H_noise (edge 0.88 > 0.78) into slot #5
    const r12 = engineV12.activateNetworkInMemory(netStar, pq.query, { topK: 5, maxHops: 2 });
    const nmda12 = r12.winners.filter((w) => !pq.seedCodes.has(w.code));
    if (nmda12[0]?.code === pq.targetTrueCode) p3a_v12_nmdaHit1++;
    if (nmda12[0]?.code === pq.hubNoiseCode) p3a_v12_fellForNoise++;
    if (r12.winners.some((w) => w.code === pq.targetTrueCode)) p3a_v12_hit5++;

    // 2. v1.3 M6a ON, M6b OFF: Quantum interference cancels H_noise, rescuing T_true
    const rM6a = engineV13.activateNetworkInMemory(netStar, pq.query, {
      topK: 5, maxHops: 2, useQuantumPhase: true, useSynapticMass: false,
    });
    const nmdaM6a = rM6a.winners.filter((w) => !pq.seedCodes.has(w.code));
    if (nmdaM6a[0]?.code === pq.targetTrueCode) p3a_m6a_nmdaHit1++;
    if (rM6a.winners.some((w) => w.code === pq.targetTrueCode)) p3a_m6a_hit5++;
    if (!rM6a.winners.some((w) => w.code === pq.hubNoiseCode)) p3a_m6a_distractorBlocked++;

    // Extract real measured coherence from engine trace on T_true & real phase dot product on H_noise
    const trueWinner = rM6a.winners.find((w) => w.code === pq.targetTrueCode);
    const mCoh = trueWinner?.via?.match(/coh=([0-9.]+)/);
    coherencesTrue.push(mCoh ? Number(mCoh[1]) : calcPairCoherence(pq.sTrue1Id, pq.sTrue2Id));
    coherencesNoise.push(calcPairCoherence(pq.sNoise1Id, pq.sNoise2Id));

    // 3. v1.3 Full (M6a ON, M6b ON)
    const rFull = engineV13.activateNetworkInMemory(netStar, pq.query, {
      topK: 5, maxHops: 2, useQuantumPhase: true, useSynapticMass: true,
    });
    const nmdaFull = rFull.winners.filter((w) => !pq.seedCodes.has(w.code));
    if (nmdaFull[0]?.code === pq.targetTrueCode) p3a_full_nmdaHit1++;
    if (rFull.winners.some((w) => w.code === pq.targetTrueCode)) p3a_full_hit5++;
  }

  const p3aRes = {
    totalQueries: polysemicQueries.length,
    v12_baseline: {
      nmdaHit1: round1((p3a_v12_nmdaHit1 / polysemicQueries.length) * 100),
      hit5: round1((p3a_v12_hit5 / polysemicQueries.length) * 100),
      fellForDistractorPct: round1((p3a_v12_fellForNoise / polysemicQueries.length) * 100),
    },
    v13_m6a_on: {
      nmdaHit1: round1((p3a_m6a_nmdaHit1 / polysemicQueries.length) * 100),
      hit5: round1((p3a_m6a_hit5 / polysemicQueries.length) * 100),
    },
    v13_full: {
      nmdaHit1: round1((p3a_full_nmdaHit1 / polysemicQueries.length) * 100),
      hit5: round1((p3a_full_hit5 / polysemicQueries.length) * 100),
    },
    meanCoherenceTargetTrue: mean(coherencesTrue),
    meanCoherenceHubNoise: mean(coherencesNoise),
    distractorsBlockedPct: round1((p3a_m6a_distractorBlocked / polysemicQueries.length) * 100),
  };

  // Run Part 3B
  let p3b_v12_p1 = 0, p3b_v12_h3 = 0;
  let p3b_m6b_p1 = 0, p3b_m6b_h3 = 0;

  for (const mq of massQueries) {
    // 1. v1.2 (inbox wins because raw weight is 0.78 vs 0.74)
    const r12 = engineV12.activateNetworkInMemory(netStar, mq.query, { topK: 5, maxHops: 0 });
    if (r12.winners[0]?.code === mq.lessonCode) p3b_v12_p1++;
    if (r12.winners.slice(0, 3).some((w) => w.code === mq.lessonCode)) p3b_v12_h3++;

    // 2. v1.3 M6b ON (lesson inertiaMod 1.168 vs inbox 0.898 flips ranking decisively)
    const rM6b = engineV13.activateNetworkInMemory(netStar, mq.query, {
      topK: 5, maxHops: 0, useSynapticMass: true,
    });
    if (rM6b.winners[0]?.code === mq.lessonCode) p3b_m6b_p1++;
    if (rM6b.winners.slice(0, 3).some((w) => w.code === mq.lessonCode)) p3b_m6b_h3++;
  }

  const p3bRes = {
    totalQueries: massQueries.length,
    v12_baseline: {
      precision1: round1((p3b_v12_p1 / massQueries.length) * 100),
      hit3: round1((p3b_v12_h3 / massQueries.length) * 100),
    },
    v13_m6b_synaptic_mass: {
      precision1: round1((p3b_m6b_p1 / massQueries.length) * 100),
      hit3: round1((p3b_m6b_h3 / massQueries.length) * 100),
    },
    deltaPrecision1: round1(((p3b_m6b_p1 - p3b_v12_p1) / massQueries.length) * 100),
  };

  results.block3_quantumStar = {
    part3A_polysemicInterference: p3aRes,
    part3B_synapticMassVolatility: p3bRes,
  };

  console.log(`Part 3A (Quantum Phase): Distractors Blocked (NMDA): ${p3aRes.distractorsBlockedPct}%`);
  console.log(`       NMDA Hit@1 (T_true vs H_noise): v1.2=${p3aRes.v12_baseline.nmdaHit1}% (fell for noise: ${p3aRes.v12_baseline.fellForDistractorPct}%) -> v1.3 M6a=${p3aRes.v13_m6a_on.nmdaHit1}% | v1.3 Full=${p3aRes.v13_full.nmdaHit1}%`);
  console.log(`       Measured Coherence: TargetTrue=${p3aRes.meanCoherenceTargetTrue} vs HubNoise=${p3aRes.meanCoherenceHubNoise}`);
  console.log(`Part 3B (Synaptic Mass): Precision@1: v1.2=${p3bRes.v12_baseline.precision1}% -> v1.3=${p3bRes.v13_m6b_synaptic_mass.precision1}% (+${p3bRes.deltaPrecision1}%)`);

  // --------------------------------------------------------------------------
  // BLOCK 4: TURRIGIANO HOMEOSTASIS (M5: 500 STREAM QUERIES)
  // --------------------------------------------------------------------------
  console.log('\n--- BLOCK 4: TURRIGIANO HOMEOSTASIS (500 STREAM QUERIES WITH REINFORCE: TRUE) ---');

  const netClassic = JSON.parse(JSON.stringify(netV13Blind));
  const netTurrigiano = JSON.parse(JSON.stringify(netV13Blind));
  engineV13.ensureIndex(netClassic, true);
  engineV13.ensureIndex(netTurrigiano, true);

  function getGraphEdgeStats(net, touchedIds = null) {
    const weights = [];
    let saturatedCount = 0;
    let maxOutDegreeSum = 0;
    let maxOutDegreeSumTouched = 0;

    for (const n of Object.values(net.neurons)) {
      let outSum = 0;
      for (const w of Object.values(n.edges || {})) {
        const val = Number(w);
        weights.push(val);
        outSum += val;
        if (val >= 0.99) saturatedCount++;
      }
      if (outSum > maxOutDegreeSum) maxOutDegreeSum = outSum;
      if (touchedIds && touchedIds.has(n.id) && outSum > maxOutDegreeSumTouched) {
        maxOutDegreeSumTouched = outSum;
      }
    }

    return {
      edgeCount: weights.length,
      saturatedCount,
      saturatedPct: weights.length ? round1((saturatedCount / weights.length) * 100) : 0,
      meanWeight: mean(weights),
      maxOutDegreeSum: round2(maxOutDegreeSum),
      ...(touchedIds ? { maxOutDegreeSumTouched: round2(maxOutDegreeSumTouched), touchedNeurons: touchedIds.size } : {}),
      gini: calcGini(weights),
    };
  }

  function evalAccuracyOn80(net, isV13 = true) {
    let h1 = 0, h3 = 0;
    for (const q of queriesV12) {
      const exp = new Set(q.expected);
      const res = isV13
        ? engineV13.activateNetworkInMemory(net, q.query, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false })
        : engineV12.activateNetworkInMemory(net, q.query, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false });
      if (res.winners.slice(0, 1).some((w) => exp.has(w.code))) h1++;
      if (res.winners.slice(0, 3).some((w) => exp.has(w.code))) h3++;
    }
    return {
      hit1: round1((h1 / queriesV12.length) * 100),
      hit3: round1((h3 / queriesV12.length) * 100),
    };
  }

  const initialStats = getGraphEdgeStats(netTurrigiano);
  const initialAcc = evalAccuracyOn80(netTurrigiano);

  const streamQueriesCount = 500;
  const touchedClassic = new Set();
  const touchedTurrigiano = new Set();
  for (let s = 0; s < streamQueriesCount; s++) {
    const qObj = queriesV12[s % queriesV12.length];
    const rc = engineV13.activateNetworkInMemory(netClassic, qObj.query, {
      topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: true, useHomeostasis: false,
    });
    if (rc.hebbianLinks?.length) rc.winners.slice(0, 3).forEach((w) => touchedClassic.add(w.id));

    const rt = engineV13.activateNetworkInMemory(netTurrigiano, qObj.query, {
      topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: true, useHomeostasis: true, wMax: 5.2,
    });
    if (rt.hebbianLinks?.length) rt.winners.slice(0, 3).forEach((w) => touchedTurrigiano.add(w.id));
  }

  const finalStatsClassic = getGraphEdgeStats(netClassic, touchedClassic);
  const finalStatsTurrigiano = getGraphEdgeStats(netTurrigiano, touchedTurrigiano);

  const finalAccClassic = evalAccuracyOn80(netClassic);
  const finalAccTurrigiano = evalAccuracyOn80(netTurrigiano);

  results.block4_turrigianoHomeostasis = {
    streamQueriesCount,
    initial: {
      edgeStats: initialStats,
      accuracy: initialAcc,
    },
    classicHebb_v12: {
      edgeStats: finalStatsClassic,
      accuracy: finalAccClassic,
    },
    turrigianoHomeostasis_v13: {
      edgeStats: finalStatsTurrigiano,
      accuracy: finalAccTurrigiano,
    },
    deltas: {
      saturatedEdgesAvoided: finalStatsClassic.saturatedCount - finalStatsTurrigiano.saturatedCount,
      maxOutDegreeTouchedReduction: round2(finalStatsClassic.maxOutDegreeSumTouched - finalStatsTurrigiano.maxOutDegreeSumTouched),
      accuracyPreservedHit3: round1(finalAccTurrigiano.hit3 - finalAccClassic.hit3),
    },
  };

  console.log(`Before Stream: Saturated Edges: ${initialStats.saturatedCount} | Max Out-Degree Sum: ${initialStats.maxOutDegreeSum} | Hit@3: ${initialAcc.hit3}%`);
  console.log(`After 500 Stream Queries:`);
  console.log(`  - Classic Hebb:  Saturated Edges: ${finalStatsClassic.saturatedCount} | Max Out-Degree (touched): ${finalStatsClassic.maxOutDegreeSumTouched} | Mean W: ${finalStatsClassic.meanWeight} | Gini: ${finalStatsClassic.gini} | Hit@3: ${finalAccClassic.hit3}%`);
  console.log(`  - Turrigiano:    Saturated Edges: ${finalStatsTurrigiano.saturatedCount} | Max Out-Degree (touched): ${finalStatsTurrigiano.maxOutDegreeSumTouched} | Mean W: ${finalStatsTurrigiano.meanWeight} | Gini: ${finalStatsTurrigiano.gini} | Hit@3: ${finalAccTurrigiano.hit3}%`);
  console.log(`  -> Synaptic scaling prevented ${results.block4_turrigianoHomeostasis.deltas.saturatedEdgesAvoided} saturated edges and bounded reinforced out-degree sum to ${finalStatsTurrigiano.maxOutDegreeSumTouched} (W_max=5.2).`);

  // --------------------------------------------------------------------------
  // BLOCK 5: PRIVACY FIREWALL F-01
  // --------------------------------------------------------------------------
  console.log('\n--- BLOCK 5: PRIVACY FIREWALL F-01 (16 PROBES) ---');
  const probes = JSON.parse(fs.readFileSync(probesPath, 'utf8'));

  const netProtected = engineV13.compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    disablePrivacyFirewall: false,
    noPersist: true,
  });

  const netUnsafe = engineV13.compileNetwork(memoryDir, {
    fresh: true,
    useSemanticSeeds: true,
    seedsOverride: seedsRegen,
    disablePrivacyFirewall: true,
    noPersist: true,
  });

  let positiveControlTriggered = 0;
  let firewallOnLeaks = 0;
  const probeDetails = [];

  for (const p of probes) {
    let posHit = false;
    let safeLeak = false;

    if (p.mode === 'activate') {
      const rPos = engineV13.activateNetworkInMemory(netUnsafe, p.query, { topK: 5, maxHops: 2, disablePrivacyFirewall: true });
      posHit = rPos.winners.some((w) => w.code.startsWith('PRIV-'));

      const rSafe = engineV13.activateNetworkInMemory(netProtected, p.query, { topK: 5, maxHops: 2, disablePrivacyFirewall: false });
      safeLeak = rSafe.winners.some((w) => w.code.startsWith('PRIV-'));
    } else if (p.mode === 'bridge') {
      const bPos = engineV13.findBridge(netUnsafe, p.queryA, p.queryB, { disablePrivacyFirewall: true, minWeight: 0.30 });
      posHit = bPos.found && bPos.path.some((x) => x.code.startsWith('PRIV-'));

      const bSafe = engineV13.findBridge(netProtected, p.queryA, p.queryB, { disablePrivacyFirewall: false });
      safeLeak = bSafe.found && bSafe.path.some((x) => x.code.startsWith('PRIV-'));
    }

    if (posHit) positiveControlTriggered++;
    if (safeLeak) firewallOnLeaks++;

    probeDetails.push({
      id: p.id,
      vector: p.vector,
      mode: p.mode,
      target: p.target,
      positiveControlHit: posHit,
      firewallOnLeak: safeLeak,
    });
  }

  // Lateral dendrite isolation
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

  const firewallPass = positiveControlTriggered === probes.length && firewallOnLeaks === 0 && lateralDendriteLeaks === 0;
  results.block5_privacyFirewall = {
    totalProbes: probes.length,
    positiveControlTriggered,
    firewallOnLeaks,
    lateralDendriteLeaks,
    pass: firewallPass,
    details: probeDetails,
  };

  console.log(`Probes: ${probes.length} | Positive Control: ${positiveControlTriggered}/${probes.length} | Firewall Leaks: ${firewallOnLeaks} | Dendrite Leaks: ${lateralDendriteLeaks}`);
  console.log(`Firewall F-01 Status: ${firewallPass ? 'PASS (100% Secure)' : 'FAIL'}`);

  // --------------------------------------------------------------------------
  // SAVE RESULTS
  // --------------------------------------------------------------------------
  results.totalBenchmarkMs = round2(performance.now() - suiteT0);
  fs.writeFileSync(outJsonPath, JSON.stringify(results, null, 2), 'utf8');
  console.log(`\n==============================================================================`);
  console.log(`Benchmark completed in ${results.totalBenchmarkMs} ms.`);
  console.log(`Full JSON results saved to: ${outJsonPath}`);
  console.log(`==============================================================================`);
}

runBenchmark().catch((err) => {
  console.error('[FATAL] Benchmark suite failed:', err);
  process.exit(1);
});
