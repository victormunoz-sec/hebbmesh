import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import {
  parseMemoryDirectory,
  compileNetwork,
  activateNetworkInMemory,
  findBridge,
  tokenize,
} from '../src/engine.mjs';
import {
  saveNetwork,
  loadNetwork,
  getSqlitePath,
} from '../src/storage.mjs';

/**
 * HebbMesh v1.2 -- Benchmark de Escalabilidad y Segundo Dominio Clinico (0 npm deps)
 *
 * Part A: Stress Test de Escala (N = 1000, 5000, 10000 neuronas)
 *   - Desglose estricto de tiempos: Parse, Weave, Persist, Total Compile, Cold Load.
 *   - Metricas de recursos: SQLite DB size (KB), RSS (MB), Heap Used (MB).
 *   - Topologia: Neuronas, Aristas, Grado Promedio (avgDegree).
 *   - Latencia de consulta (50 queries representativas): mean, p50, p95, p99 en ms
 *     para fts5_sem, hebb_no_prop, hebb_v12_a4.
 *   - Deteccion de cuellos de botella (compileTotal > 60s o p95 > 25ms).
 *
 * Part B: Segundo Dominio Clinico (N_nodes = 2020; N_queries = 200 en 4 categorias de 50)
 *   - 0% Dev/SaaS. Dominio: Operaciones Hospitalarias y Telemetria Biomedica.
 *   - 5 Variantes: fts5_pure, fts5_sem, hebb_no_prop, hebb_v12_a4, hebb_v12_a4_gaba.
 *   - 4 Categorias (50 cada una, 200 clusters tematicos c_001..c_200):
 *       1. direct (50 queries)
 *       2. semantic (50 queries con 100% gap lexico verificado)
 *       3. multihop_convergent (50 queries NMDA kParents >= 2)
 *       4. episodic_convergent (50 queries incidentes semanales S-Wnn-F3)
 *   - Control super-hub episodico (maxEpisodicAllowed: 1 en 10 queries directas).
 *   - Cluster Bootstrap (10,000 resamples, Mulberry32 seed=20261123, 95% CI)
 *     para a4_minus_noProp y a4_minus_fts5Sem.
 *   - Privacy Firewall F-01 (20 sondas: 14 activacion + 6 bridge A* con control positivo 20/20).
 *
 * Salida: bench/results_scale_v1.2.json y resumen ASCII/Markdown en consola.
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outJsonPath = path.resolve(__dirname, 'results_scale_v1.2.json');

const BOOTSTRAP_ITERS = 10000;
const GLOBAL_SEED = 20261123;
const CATS_PART_B = ['direct', 'semantic', 'multihop_convergent', 'episodic_convergent'];
const BOOT_CATS_PART_B = ['direct', 'semantic', 'multihop_convergent', 'episodic_convergent', 'overall'];

// === FUNCIONES MATEMATICAS Y PRNG DETERMINISTA ===

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

// === BASELINE FTS5 IN-MEMORY ===

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

// === EVALUACION Y BOOTSTRAP ===

function evaluateVariant(id, label, runner, queryList) {
  const perQuery = [];
  const latencies = [];
  const cats = {};
  for (const c of CATS_PART_B) cats[c] = { n: 0, h1: 0, h3: 0, h5: 0 };

  let epCtrlN = 0;
  let epCtrlViolations = 0;
  let epCtrlHit3 = 0;
  let epCtrlEpTop3Sum = 0;
  let epCtrlEpTop5Sum = 0;

  for (const q of queryList) {
    const t0 = performance.now();
    const res = runner(q.query);
    const elapsed = performance.now() - t0;
    latencies.push(elapsed);

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
  for (const c of CATS_PART_B) {
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
    episodicControl: {
      n: epCtrlN,
      superHubViolations: epCtrlViolations,
      hit3Pct: epCtrlN ? round1((epCtrlHit3 / epCtrlN) * 100) : 0,
      avgEpisodicInTop3: epCtrlN ? round2(epCtrlEpTop3Sum / epCtrlN) : 0,
      avgEpisodicInTop5: epCtrlN ? round2(epCtrlEpTop5Sum / epCtrlN) : 0,
    },
    latencyMs: {
      mean: round2(mean(latencies)),
      p50: round2(percentile(latencies, 50)),
      p95: round2(percentile(latencies, 95)),
      p99: round2(percentile(latencies, 99)),
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

function clusterBootstrap(perQueryA, perQueryB = null, catOrOverall = 'overall', iters = BOOTSTRAP_ITERS, seed = GLOBAL_SEED) {
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

function computeLexicalGap(rawMap, queryList) {
  const byCode = new Map();
  for (const n of rawMap.values()) byCode.set(n.code, n);

  const semQueries = queryList.filter((q) => q.cat === 'semantic');
  const details = [];
  let zeroOverlapCount = 0;
  let totalOverlapTokensSum = 0;

  for (const q of semQueries) {
    const qToks = tokenize(q.query);
    const expToks = [];
    for (const code of q.expected) {
      const n = byCode.get(code);
      if (n) expToks.push(...tokenize(`${n.title} ${n.summary}`));
    }
    const matchingTokens = qToks.filter((qt) => expToks.some((et) => tokensOverlap(qt, et)));
    const overlapTokensCount = matchingTokens.length;
    const isZero = overlapTokensCount === 0;
    if (isZero) zeroOverlapCount++;
    totalOverlapTokensSum += overlapTokensCount;

    details.push({
      id: q.id,
      cluster: q.cluster,
      expected: q.expected,
      qTokensCount: qToks.length,
      overlapTokensCount,
      matchingTokens,
    });
  }

  return {
    totalSemantic: semQueries.length,
    zeroOverlapCount,
    zeroOverlapPct: semQueries.length ? round1((zeroOverlapCount / semQueries.length) * 100) : 0,
    meanOverlapTokens: semQueries.length ? round2(totalOverlapTokensSum / semQueries.length) : 0,
    details,
  };
}

// === GENERADOR DETERMINISTA DE SEGUNDO DOMINIO CLINICO ===

const CLINICAL_SPECIALTIES = [
  'cardiology', 'nephrology', 'pulmonology', 'neurology', 'oncology',
  'microbiology', 'hematology', 'radiology', 'anesthesia', 'trauma',
  'neonatology', 'hepatology', 'endocrinology', 'rheumatology', 'immunology',
  'urology', 'ophthalmology', 'orthopedics', 'dermatology', 'psychiatry',
  'toxicology', 'pharmacology', 'obstetrics', 'gastroenterology', 'vascular'
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
  vascular: ['aneurysm', 'thrombectomy', 'dopplersonography', 'vasculargraft', 'embolectomy', 'claudication', 'arterialstenosis', 'endovascularstent', 'carotidocclusion', 'ischemiatime', 'arterialbypass', 'pseudoaneurysm']
};

const CLINICAL_WARDS = [
  'icu_alpha_ward', 'icu_beta_ward', 'telemetry_central_unit', 'or_suite_cardiac',
  'burn_specialized_bay', 'hemodialysis_bay_a', 'pacu_recovery_north', 'trauma_bay_zero',
  'nicu_high_acuity', 'ccu_coronary_care', 'endoscopy_suite_two', 'stepdown_ward_south'
];

const CLINICAL_DEVICES = [
  'ventilator_servo_u', 'infusionpump_alaris_t', 'ecmo_cardiohelp_sys', 'hemofilter_prismaflex',
  'defibrillator_zoll_x', 'telemetry_mindray_bv', 'capnograph_microstream', 'bis_monitor_vista'
];

/**
 * 50 pares semanticos curados para Part B (100% gap lexico verificado):
 * Ingles (query y | sem:) vs Espanol/Latin neutro en titulo/resumen sin prefijo compartido de >=4 letras.
 */
const SEMANTIC_PAIRS_50 = [
  { en: 'myocardial heart attack coronary blockage chest pain', es: 'registro necrosis celula musculo lesion isquemia pared' },
  { en: 'pulmonary blood clot leg deep venous thrombosis', es: 'obstruccion rama pleura disminucion oxigeno saturacion baja' },
  { en: 'brain stroke weakness drooping face slurred speech', es: 'deficiencia motora asimetria reflejo corteza craneal aguda' },
  { en: 'diabetic blood sugar surge ketone crisis thirst', es: 'alteracion glucemia descompensada acidosis suero deshidratacion severa' },
  { en: 'septic shock low blood pressure fever chill', es: 'colapso vascular perfusion tisular lactato elevado respuesta inmune' },
  { en: 'breathing distress air hunger oxygen mask cannula', es: 'falla intercambio gaseoso alveolo esfuerzo toracico ventilatorio agudo' },
  { en: 'erratic heartbeat electric storm shock paddles rhythm', es: 'despolarizacion ectopica sincrona paleta descarga miocardio rapida' },
  { en: 'brain bleed sudden worst headache ruptured vessel', es: 'fuga licuor cefalorraquideo cisterna sangrado subaracnoideo agudo' },
  { en: 'upper belly pain enzyme surge swollen gland', es: 'proceso inflamatorio retroperitoneal necrosis amilasa lipasa elevada' },
  { en: 'collapsed lung trapped air needle chest puncture', es: 'acumulacion cavidad pleural desplazamiento mediastino trocar drenaje' },
  { en: 'allergic hives throat swelling bee sting shot', es: 'reaccion cutanea urticaria cierre glotis adrenalina ampolla inmediata' },
  { en: 'unstoppable shaking fits convulsion brain storm drip', es: 'descarga corticoespinal motriz crisis repetitiva benzodiacepina bolo' },
  { en: 'fluid around heart muffled sounds neck veins', es: 'derrame seroso saco fibroso elevacion venosa yugular grave' },
  { en: 'ripping back pain torn main artery surgery', es: 'desgarro tunica intima vaso troncal bifurcacion quirofano urgente' },
  { en: 'yellow skin sudden liver shutdown confusion coma', es: 'atrofia hepatocito ictericia bilirrubina estupor amonio elevado' },
  { en: 'high potassium peaked wave heart rhythm danger', es: 'elevacion cation serico conduccion auriculoventricular ensanchamiento trazado' },
  { en: 'infected heart valve fever murmur blood culture', es: 'vegetacion endocardio proliferacion endotelio soplo nuevo bacteriemia' },
  { en: 'smoke fumes headache dizzy high pressure tank', es: 'inhalacion gases combustion carboxihemoglobina camara hiperbarica forzada' },
  { en: 'tight swollen limb calf pain blade release', es: 'tension intersticial aponeurosis isquemia corte longitudinal descompresion' },
  { en: 'operating room high fever stiff jaw antidote', es: 'respuesta anestesica temperatura extrema espasmo receptor rianodina dantroleno' },
  { en: 'vomiting dark blood stomach ulcer scope clip', es: 'egreso hematinico mucosa gastrica erosion hemostasia mecanica urgente' },
  { en: 'gut blood loss belly cramps dead bowel', es: 'hipoperfusion arteria digestiva asas congestion necrosis pared intestinal' },
  { en: 'blood bag reaction wet lungs low oxygen', es: 'distres agudo secundario hemocomponente neutrofilos infiltrado bilateral' },
  { en: 'tiny clots low platelets plasma swap wash', es: 'microangiopatia esquistocitos deficit proteasa recambio afaresis urgente' },
  { en: 'peeling skin painful blisters drug rash burn', es: 'desprendimiento epitelio mucocutaneo eritema ampolla superficie corporal' },
  { en: 'kidney shutdown zero urine fluid buildup pill', es: 'falla filtrado glomerular excrecion nula uremia sobrecarga hidrica' },
  { en: 'weakness salt craving fainting hormone shot rescue', es: 'hipofuncion corteza suprarrenal corticoide reposicion choque refractario' },
  { en: 'racing pulse hot sweat neck gland crisis', es: 'hipermetabolismo fiebre taquisfigmia bloqueo bocio agudo severo' },
  { en: 'droopy eyelids weak breathing swallow trouble drip', es: 'fatigabilidad placa motora debilidad diafragma anticuerpos receptores' },
  { en: 'lead pipe stiffness high fever psych med', es: 'reaccion psicotropico hipertermia extrapiramidalismo bloqueo dopamina agudo' },
  { en: 'red painful eye halos blurry sight drop', es: 'camara anterior estrecha bloqueo pupilar midriasis tonometria alta' },
  { en: 'no pulse flat monitor chest push rescue', es: 'disociacion electromecanica ausencia latido maniobra toracica protocolo' },
  { en: 'sleepy shallow breaths trapped gas face mask', es: 'retencion acido narcosis somnolencia soporte ventilatorio positivo' },
  { en: 'broken thigh bone skin spots confusion breath', es: 'globulo lipidico diafisis femoral petequia torax desaturacion subita' },
  { en: 'gas bubble neck line sudden gasp chest', es: 'ingreso atmosferico cateter yugular aspiracion auricula derecha urgente' },
  { en: 'farm spray drooling tiny pupils twitching rescue', es: 'inhibicion colinesterasa miosis sialorrea fasciculaciones oxima dosis' },
  { en: 'bounding pulse leaky main valve short breath', es: 'insuficiencia retorno valvular reflujo diastolico dilatacion camara' },
  { en: 'stiff neck bright light fever back tap', es: 'signo nuca fotofobia puncion lumbar polimorfonucleares liquido' },
  { en: 'back ache numb legs weak feet steroid', es: 'aplastamiento canal medular deficit sensitivo dexametasona descompresiva' },
  { en: 'oozing gums widespread bruises low clot factor', es: 'coagulopatia consumo trombosis microvascular sangrado mucoso fibrinopenia' },
  { en: 'fast spreading limb rot gas under skin', es: 'gangrena tejido subcutaneo crepitacion fascia reseccion bisturi urgente' },
  { en: 'bloody stool pale child bruised kidney shut', es: 'toxina verotoxina anemia plaquetopenia enteropatia falla renal aguda' },
  { en: 'sky high blood pressure headache blurry vision', es: 'cifras tensionales maximas papiledema vasodilatador goteo continuo' },
  { en: 'birth shock sudden blue lips heavy bleed', es: 'paso material fetal torrente venoso materno coagulopatia paro' },
  { en: 'swollen throat veins blood vomit rubber band', es: 'dilatacion submucosa esofago ligadura elastica hemostasia urgente' },
  { en: 'blue skin failing lungs bypass machine blood', es: 'hipoxia refractaria circuito extracorporeo membrana intercambio soporte' },
  { en: 'tight chest wheezing silent lungs mist spray', es: 'hiperreactividad bronquial roncus sibilancia aerosol dilatador continuo' },
  { en: 'wet skin tearing eyes slow heart poison', es: 'hiperactividad parasimpatica diaforesis lagrimeo bradicardia antimuscarinico' },
  { en: 'cancer chemo kidney jam high salt crystals', es: 'destruccion neoplasica purinas hiperfosfatemia quelante metabolico agudo' },
  { en: 'house fire cherry skin bitter smell shot', es: 'asfixia quimica mitocondria citocromo oxidasa antidoto hidroxocobalamina' },
];

/**
 * Generador de Corpus Clinico.
 * targetDir: Directorio temporal.
 * totalPublic: Cantidad exacta de neuronas publicas.
 * totalPrivate: Cantidad exacta de neuronas privadas.
 * withEvalGroundTruth: true genera las 200 queries y 20 sondas de privacidad de Part B.
 */
function generateClinicalCorpus(targetDir, { totalPublic, totalPrivate = 20, withEvalGroundTruth = false, seed = GLOBAL_SEED }) {
  const rng = mulberry32(seed);

  fs.mkdirSync(path.join(targetDir, 'knowledge'), { recursive: true });
  fs.mkdirSync(path.join(targetDir, 'synthesis'), { recursive: true });

  const numKnowledge = 20; // doc_01.md .. doc_20.md -> DOC-1..DOC-20
  const numEpisodic = 50;  // 2026-W01.md .. W50.md (solo F3) -> S-W01-F3 .. S-W50-F3
  const numLessons = withEvalGroundTruth ? 100 : Math.max(20, Math.min(100, Math.floor(totalPublic * 0.05)));
  const numMilestones = withEvalGroundTruth ? 100 : Math.max(20, Math.min(100, Math.floor(totalPublic * 0.05)));
  const numProfile = withEvalGroundTruth ? 50 : Math.max(10, Math.min(50, Math.floor(totalPublic * 0.025)));
  const numInbox = withEvalGroundTruth ? 50 : Math.max(10, Math.min(50, Math.floor(totalPublic * 0.025)));

  const fixedNonGraph = numKnowledge + numEpisodic + numLessons + numMilestones + numProfile + numInbox;
  if (totalPublic < fixedNonGraph + 50) {
    throw new Error(`totalPublic (${totalPublic}) es demasiado pequeno para la estructura requerida (min ${fixedNonGraph + 50})`);
  }
  const numGraph = totalPublic - fixedNonGraph;

  const queries = [];
  const privacyProbes = [];

  // 1. KNOWLEDGE (20 documentos clinicos)
  for (let d = 1; d <= numKnowledge; d++) {
    const spec = CLINICAL_SPECIALTIES[(d - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    const docPath = path.join(targetDir, 'knowledge', `doc_${String(d).padStart(2, '0')}.md`);
    const content = [
      `# Manual de Protocolos Clinicos en ${spec.toUpperCase()}`,
      `summary: Compendio estandarizado de telemetria procedimientos y farmacologia en ${terms.slice(0, 4).join(' ')}.`,
      `sem: clinical telemetry protocol ${spec} ${terms.slice(4, 7).join(' ')}`,
      '',
      `Documento tecnico de referencia hospitalaria en servicio de ${spec}.`,
      `Parametros de atencion critica: ${terms.join(', ')}.`,
      `Equipamiento asignado: ${CLINICAL_DEVICES[(d - 1) % CLINICAL_DEVICES.length]}.`,
      `Unidad operativa: ${CLINICAL_WARDS[(d - 1) % CLINICAL_WARDS.length]}.`,
    ].join('\n');
    fs.writeFileSync(docPath, content, 'utf8');
  }

  // 2. SYNTHESIS (50 semanas, con seccion F3 para neuronas S-Wnn-F3)
  for (let w = 1; w <= numEpisodic; w++) {
    const wStr = String(w).padStart(2, '0');
    const synthPath = path.join(targetDir, 'synthesis', `2026-W${wStr}.md`);
    const spec = CLINICAL_SPECIALTIES[(w - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];

    let f3Lines = [
      `Registro hospitalario guardia semana ${wStr} unidad ${CLINICAL_WARDS[(w - 1) % CLINICAL_WARDS.length]} ${terms[0]}.`,
    ];

    if (withEvalGroundTruth) {
      // Part B Category d: episodic_convergent
      // Neurona S-W{w}-F3 contiene 2 pares puente con prefijos unicos (primeros 5 chars distintos para evitar stem-collision)
      f3Lines.push(`Sintesis clinica e${wStr}a1_bridge e${wStr}a2_bridge e${wStr}b1_bridge e${wStr}b2_bridge ${spec}.`);
    }

    const content = [
      `# Resumen Semanal 2026-W${wStr}`,
      `## Phase 3 Execution & Open Tasks`,
      ...f3Lines,
    ].join('\n');
    fs.writeFileSync(synthPath, content, 'utf8');
  }

  // 3. MILESTONES (HITO-1 .. HITO-numMilestones)
  const milestoneLines = [];
  for (let m = 1; m <= numMilestones; m++) {
    const spec = CLINICAL_SPECIALTIES[(m - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    if (withEvalGroundTruth && m <= 50) {
      const wStr = String(m).padStart(2, '0');
      // Clue 1 para episodic_convergent w=m (prefijos unicos w01a1/w01a2, df>=9 en fillers)
      milestoneLines.push(
        `- [x] Registro clinico unidad hospitalaria ${m}: w${wStr}a1_clue w${wStr}a2_clue e${wStr}a1_bridge e${wStr}a2_bridge monitoreo.`
      );
    } else {
      milestoneLines.push(
        `- [x] Auditoria clinica ${m}: revision de ${terms.slice(0, 3).join(' ')} en ${spec} unidad hospitalaria.`
      );
    }
  }
  fs.writeFileSync(path.join(targetDir, 'milestones.md'), milestoneLines.join('\n'), 'utf8');

  // 4. PROFILE (PERFIL-1 .. PERFIL-numProfile)
  const profileLines = [];
  for (let p = 1; p <= numProfile; p++) {
    const spec = CLINICAL_SPECIALTIES[(p - 1) % CLINICAL_SPECIALTIES.length];
    profileLines.push(
      `- Especialista medico ${p}: jefe de seccion en ${spec} con experiencia en cuidados criticos y telemetria avanzada.`
    );
  }
  fs.writeFileSync(path.join(targetDir, 'profile.md'), profileLines.join('\n'), 'utf8');

  // 5. INBOX (INBOX-1 .. INBOX-numInbox, todos publicos con [CLINICAL])
  const inboxLines = [];
  for (let i = 1; i <= numInbox; i++) {
    const spec = CLINICAL_SPECIALTIES[(i - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    inboxLines.push(
      `- [ ] [CLINICAL] Notificacion de guardia ${i}: ingreso de interconsulta para ${spec} requiriendo evaluacion de ${terms[0]} y ${terms[1]}.`
    );
  }
  fs.writeFileSync(path.join(targetDir, 'inbox.md'), inboxLines.join('\n'), 'utf8');

  // 6. LESSONS (A-1 .. A-numLessons)
  const lessonLines = [];
  for (let l = 1; l <= numLessons; l++) {
    const spec = CLINICAL_SPECIALTIES[(l - 1) % CLINICAL_SPECIALTIES.length];
    const terms = CLINICAL_VOCABULARY[spec];
    let body = `Manejo estandarizado en ${spec}: correlacion diagnostica de ${terms.slice(0, 3).join(' ')} en guardia.`;
    if (withEvalGroundTruth && l <= 50) {
      const cStr = String(l).padStart(2, '0');
      // Part B Category c: Target C para multihop_convergent
      // Contiene 3+3 tokens puente con prefijos unicos (df=2) + vocabulario de alta frecuencia (> dfMax), ZERO query tokens, ZERO sem.
      body = `Registro clinico unidad hospitalaria m${cStr}a1_path m${cStr}a2_path m${cStr}a3_path m${cStr}b1_path m${cStr}b2_path m${cStr}b3_path monitoreo.`;
    }
    lessonLines.push(
      `- **A-${l} \`[V]\` Registro clinico ${l}:** ${body}`
    );
  }
  fs.writeFileSync(path.join(targetDir, 'lessons.md'), lessonLines.join('\n'), 'utf8');

  // 7. GRAPH.MD (G-1 .. G-numGraph distribuidos en C1, C2, C3, C4)
  // Plan de asignacion para Part B cuando withEvalGroundTruth = true:
  //   G-1..G-50: Direct targets (Category a)
  //   G-51..G-100: Semantic targets (Category b)
  //   G-101..G-150: Seed A para multihop_convergent (Category c)
  //   G-151..G-200: Seed B para multihop_convergent (Category c)
  //   G-201..G-250: Clue 2 para episodic_convergent (Category d)
  //   G-301..G-750: Background filler neurons para tokens sintoma y clue (asegurar df >= 8, > 6)
  //   G-801..G-820: Anchor neurons publicos para sondas bridge de privacidad (G-815..G-820)
  //   Resto: Nodos de soporte y relleno clinico regular estructurados para avgDegree 3-7.

  const graphLines = [];
  let currentLobe = '';

  const setLobe = (lobeName) => {
    if (currentLobe !== lobeName) {
      currentLobe = lobeName;
      graphLines.push(`\n## ${lobeName}\n`);
    }
  };

  // Preparar tokens de relleno para garantizar df >= 8 (df > 6) en Part B
  // Para cada i=1..50, sym_a1_i, sym_a2_i, sym_b1_i, sym_b2_i, clue_a1_i, clue_a2_i, clue_b1_i, clue_b2_i
  // se distribuyen en 8 neuronas filler distintas.
  const fillerTokensMap = new Map(); // gIdx -> array of tokens
  if (withEvalGroundTruth) {
    const fillerStart = 301;
    const fillerCount = 450;
    for (let i = 1; i <= 50; i++) {
      const cStr = String(i).padStart(2, '0');
      const symList = [
        `c${cStr}a1_sym`, `c${cStr}a2_sym`, `c${cStr}b1_sym`, `c${cStr}b2_sym`,
        `w${cStr}a1_clue`, `w${cStr}a2_clue`, `w${cStr}b1_clue`, `w${cStr}b2_clue`
      ];
      for (let sIdx = 0; sIdx < symList.length; sIdx++) {
        const tok = symList[sIdx];
        // Asignar tok a 8 neuronas filler distintas sin co-ocurrencia entre pares del mismo caso i
        for (let rep = 0; rep < 8; rep++) {
          const slot = fillerStart + ((i * 13 + sIdx * 55 + rep * 6) % fillerCount);
          if (!fillerTokensMap.has(slot)) fillerTokensMap.set(slot, []);
          fillerTokensMap.get(slot).push(tok);
        }
      }
    }
  }

  for (let g = 1; g <= numGraph; g++) {
    // Asignar seccion/comunidad
    let comm = 'C1';
    let lobeHeader = 'C1 · Daily Care & Wards';
    if (withEvalGroundTruth) {
      if (g >= 101 && g <= 200) {
        comm = 'C3'; lobeHeader = 'C3 · Exploration & Clinical Diagnostic Hypotheses';
      } else if (g >= 201 && g <= 250) {
        comm = 'C2'; lobeHeader = 'C2 · Execution & Critical Protocols';
      } else if (g % 4 === 1) {
        comm = 'C1'; lobeHeader = 'C1 · Daily Care & Wards';
      } else if (g % 4 === 2) {
        comm = 'C2'; lobeHeader = 'C2 · Execution & Critical Protocols';
      } else if (g % 4 === 3) {
        comm = 'C3'; lobeHeader = 'C3 · Exploration & Clinical Diagnostic Hypotheses';
      } else {
        comm = 'C4'; lobeHeader = 'C4 · Support & Bioethics';
      }
    } else {
      if (g % 4 === 1) { comm = 'C1'; lobeHeader = 'C1 · Daily Care & Wards'; }
      else if (g % 4 === 2) { comm = 'C2'; lobeHeader = 'C2 · Execution & Critical Protocols'; }
      else if (g % 4 === 3) { comm = 'C3'; lobeHeader = 'C3 · Exploration & Clinical Diagnostic Hypotheses'; }
      else { comm = 'C4'; lobeHeader = 'C4 · Support & Bioethics'; }
    }
    setLobe(lobeHeader);

    const specIdx = (g - 1) % CLINICAL_SPECIALTIES.length;
    const spec = CLINICAL_SPECIALTIES[specIdx];
    const terms = CLINICAL_VOCABULARY[spec];
    const subtopicIdx = Math.floor((g - 1) / CLINICAL_SPECIALTIES.length) % 12;

    // Conectividad controlada (avgDegree 3-7): agrupar neuronas por modulo de 6 para compartir 2 tags especificos
    const clusterMod = Math.floor(g / 6);
    const sharedTag1 = `clintag_${spec}_${clusterMod % 20}`;
    const sharedTag2 = `subtopic_${spec}_${subtopicIdx}`;

    let lineText = '';

    if (withEvalGroundTruth && g <= 50) {
      // Category a: Direct Target
      const directTopic = terms[subtopicIdx % terms.length];
      const directWord2 = terms[(subtopicIdx + 1) % terms.length];
      const protCode = `prot_clin_${String(g).padStart(2, '0')}`;
      lineText = `- **G-${g}** \`[V]\` Protocolo de intervencion ${protCode} en ${spec} tratamiento especifico ${directTopic} ${directWord2} reporte de unidad.`;
    } else if (withEvalGroundTruth && g >= 51 && g <= 100) {
      // Category b: Semantic Target (100% gap lexico verificado)
      const pair = SEMANTIC_PAIRS_50[g - 51];
      lineText = `- **G-${g}** \`[V]\` Procedimiento estandar ${pair.es} caso clinico codigo ${g} | sem: ${pair.en}`;
    } else if (withEvalGroundTruth && g >= 101 && g <= 150) {
      // Category c: Seed A para multihop_convergent (prefijos unicos c01a1/c01a2)
      const i = g - 100;
      const cStr = String(i).padStart(2, '0');
      lineText = `- **G-${g}** \`[V]\` Registro clinico unidad hospitalaria c${cStr}a1_sym c${cStr}a2_sym m${cStr}a1_path m${cStr}a2_path m${cStr}a3_path monitoreo.`;
    } else if (withEvalGroundTruth && g >= 151 && g <= 200) {
      // Category c: Seed B para multihop_convergent
      const i = g - 150;
      const cStr = String(i).padStart(2, '0');
      lineText = `- **G-${g}** \`[V]\` Registro clinico unidad hospitalaria c${cStr}b1_sym c${cStr}b2_sym m${cStr}b1_path m${cStr}b2_path m${cStr}b3_path monitoreo.`;
    } else if (withEvalGroundTruth && g >= 201 && g <= 250) {
      // Category d: Clue 2 para episodic_convergent
      const w = g - 200;
      const wStr = String(w).padStart(2, '0');
      lineText = `- **G-${g}** \`[V]\` Registro clinico unidad hospitalaria w${wStr}b1_clue w${wStr}b2_clue e${wStr}b1_bridge e${wStr}b2_bridge monitoreo.`;
    } else if (withEvalGroundTruth && g >= 815 && g <= 820) {
      // Bridge public anchor para sondas de privacidad F-01 (PRIV-15..PRIV-20)
      const k = g - 800; // 15..20
      lineText = `- **G-${g}** \`[V]\` Comite hospitalario evaluacion etica publica caso bridge_priv_tag_${k} bridge_priv_tag2_${k} dictamen general.`;
    } else {
      // Relleno regular o filler para background tokens
      const fillerToks = fillerTokensMap.get(g);
      const fillerStr = fillerToks && fillerToks.length ? ` ${fillerToks.join(' ')}` : '';
      lineText = `- **G-${g}** \`[V]\` Registro clinico en ${spec} monitoreo de ${terms[subtopicIdx % terms.length]} ${sharedTag1} ${sharedTag2}${fillerStr} unidad hospitalaria.`;
    }

    graphLines.push(lineText);
  }

  fs.writeFileSync(path.join(targetDir, 'graph.md'), graphLines.join('\n'), 'utf8');

  // 8. PRIVATE VAULT (PRIV-1 .. PRIV-totalPrivate)
  const privLines = [];
  for (let v = 1; v <= totalPrivate; v++) {
    if (withEvalGroundTruth && v <= 14) {
      // Sondas de activacion (P01..P14)
      privLines.push(
        `- PRIV-${v} Registro confidencial historial clinico paciente confidential_clin_token_${v} expediente reservado secreto medico.`
      );
    } else if (withEvalGroundTruth && v >= 15 && v <= 20) {
      // Sondas de bridge (P15..P20) - comparten 2 tags con G-(800+v) + token privado exclusivo
      privLines.push(
        `- PRIV-${v} Protocolo de atencion privada auditoria reservada bridge_priv_tag_${v} bridge_priv_tag2_${v} secret_vault_bridge_${v} comite reservado.`
      );
    } else {
      privLines.push(
        `- PRIV-${v} Datos confidenciales de paciente reservado codigo clinico privado vault secreto medico.`
      );
    }
  }
  fs.writeFileSync(path.join(targetDir, 'private_vault.md'), privLines.join('\n'), 'utf8');

  // 9. CONSTRUCCION DE GROUND TRUTH (Part B queries y privacidad)
  if (withEvalGroundTruth) {
    // 50 direct queries (Category a)
    for (let i = 1; i <= 50; i++) {
      const specIdx = (i - 1) % CLINICAL_SPECIALTIES.length;
      const spec = CLINICAL_SPECIALTIES[specIdx];
      const terms = CLINICAL_VOCABULARY[spec];
      const subtopicIdx = Math.floor((i - 1) / CLINICAL_SPECIALTIES.length) % 12;
      const directTopic = terms[subtopicIdx % terms.length];
      const directWord2 = terms[(subtopicIdx + 1) % terms.length];

      const protCode = `prot_clin_${String(i).padStart(2, '0')}`;
      const qObj = {
        id: `q_direct_${String(i).padStart(3, '0')}`,
        cat: 'direct',
        cluster: `c_${String(i).padStart(3, '0')}`,
        query: `${protCode} ${directTopic} ${directWord2} protocolo intervencion ${spec}`,
        expected: [`G-${i}`],
      };

      // 10 queries directas con maxEpisodicAllowed: 1 para control de super-hubs
      if (i <= 10) {
        qObj.maxEpisodicAllowed = 1;
        qObj.query += ' guardia medica';
      }

      queries.push(qObj);
    }

    // 50 semantic queries (Category b)
    for (let i = 1; i <= 50; i++) {
      const pair = SEMANTIC_PAIRS_50[i - 1];
      queries.push({
        id: `q_sem_${String(i).padStart(3, '0')}`,
        cat: 'semantic',
        cluster: `c_${String(50 + i).padStart(3, '0')}`,
        query: pair.en,
        expected: [`G-${50 + i}`],
      });
    }

    // 50 multihop_convergent queries (Category c) - Pure NMDA kParents >= 2
    for (let i = 1; i <= 50; i++) {
      const cStr = String(i).padStart(2, '0');
      queries.push({
        id: `q_mhop_${String(i).padStart(3, '0')}`,
        cat: 'multihop_convergent',
        cluster: `c_${String(100 + i).padStart(3, '0')}`,
        query: `c${cStr}a1_sym c${cStr}a2_sym c${cStr}b1_sym c${cStr}b2_sym`,
        expected: [`A-${i}`],
      });
    }

    // 50 episodic_convergent queries (Category d) - Pure NMDA kParents >= 2 onto S-W{w}-F3
    for (let w = 1; w <= 50; w++) {
      const wStr = String(w).padStart(2, '0');
      queries.push({
        id: `q_epis_${String(w).padStart(3, '0')}`,
        cat: 'episodic_convergent',
        cluster: `c_${String(150 + w).padStart(3, '0')}`,
        query: `w${wStr}a1_clue w${wStr}a2_clue w${wStr}b1_clue w${wStr}b2_clue`,
        expected: [`S-W${wStr}-F3`],
      });
    }

    // 20 Privacy Probes (14 activate + 6 bridge)
    for (let k = 1; k <= 14; k++) {
      privacyProbes.push({
        id: `P${String(k).padStart(2, '0')}`,
        vector: 'literal_token_activate',
        mode: 'activate',
        target: `PRIV-${k}`,
        query: `confidential_clin_token_${k} PRIV-${k}`,
      });
    }
    for (let k = 15; k <= 20; k++) {
      privacyProbes.push({
        id: `P${String(k).padStart(2, '0')}`,
        vector: 'astar_bridge_priv_to_public',
        mode: 'bridge',
        target: `PRIV-${k}`,
        queryA: `n_PRIV_${k}`,
        queryB: `n_G_${800 + k}`,
      });
    }
  }

  return { queries, privacyProbes };
}

// === RUNNER PART A: STRESS TEST DE ESCALA (N = 1000, 5000, 10000) ===

function runPartA() {
  const scaleTargets = [1000, 5000, 10000];
  const results = {};

  for (const N of scaleTargets) {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), `hebbmesh-scale-N${N}-`));
    try {
      const totalPrivate = 20;
      const totalPublic = N - totalPrivate;

      generateClinicalCorpus(tmpDir, {
        totalPublic,
        totalPrivate,
        withEvalGroundTruth: false,
        seed: GLOBAL_SEED + N,
      });

      // 1. Parse Directory
      const tParse0 = performance.now();
      const rawMap = parseMemoryDirectory(tmpDir, { useSemanticSeeds: true });
      const parseMs = round2(performance.now() - tParse0);

      // 2. Compile Network (fresh, m0Topology, noPersist)
      const tComp0 = performance.now();
      const net = compileNetwork(tmpDir, {
        fresh: true,
        useSemanticSeeds: true,
        m0Topology: true,
        noPersist: true,
      });
      const compileNoPersistMs = performance.now() - tComp0;
      const weaveMs = round2(Math.max(0, compileNoPersistMs - parseMs));

      // 3. Persist SQLite
      const tPers0 = performance.now();
      saveNetwork(tmpDir, net);
      const persistMs = round2(performance.now() - tPers0);
      const compileTotalMs = round2(parseMs + weaveMs + persistMs);

      // 4. Cold Load from SQLite
      const tCold0 = performance.now();
      loadNetwork(tmpDir);
      const coldLoadMs = round2(performance.now() - tCold0);

      // 5. DB File Size (sqlite + wal + shm)
      const dbPath = getSqlitePath(tmpDir);
      let sqliteBytes = 0;
      for (const p of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
        try {
          if (fs.existsSync(p)) sqliteBytes += fs.statSync(p).size;
        } catch { }
      }
      const sqliteSizeKB = round2(sqliteBytes / 1024);

      // 6. Memory usage
      const mem = process.memoryUsage();
      const rssMB = round2(mem.rss / (1024 * 1024));
      const heapUsedMB = round2(mem.heapUsed / (1024 * 1024));

      // 7. Graph Topology
      const totalNeurons = net.metrics.totalNeurons;
      const totalEdges = net.metrics.totalEdges;
      const avgDegree = round2(totalEdges / totalNeurons);

      // 8. 50 Representative Clinical Queries Latency Benchmark
      const ftsSemDb = buildFts5Baseline(rawMap, true);
      const testQueries = [];
      const specList = CLINICAL_SPECIALTIES;
      for (let qi = 0; qi < 50; qi++) {
        const sp = specList[qi % specList.length];
        const vList = CLINICAL_VOCABULARY[sp];
        const w1 = vList[qi % vList.length];
        const w2 = vList[(qi + 2) % vList.length];
        testQueries.push(`${w1} ${w2} protocolo intervencion ${sp}`);
      }

      const latsFts = [];
      const latsNoProp = [];
      const latsA4 = [];

      for (const q of testQueries) {
        const t0F = performance.now();
        queryFts5(ftsSemDb, q, 5);
        latsFts.push(performance.now() - t0F);

        const t0NP = performance.now();
        activateNetworkInMemory(net, q, { topK: 5, disablePropagation: true, maxEpisodic: 1, reinforce: false });
        latsNoProp.push(performance.now() - t0NP);

        const t0A = performance.now();
        activateNetworkInMemory(net, q, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false });
        latsA4.push(performance.now() - t0A);
      }

      const compileBottleneck = compileTotalMs > 60000;
      const a4P95 = round2(percentile(latsA4, 95));
      const latencyBottleneck = a4P95 > 25.0;

      results[`N_${N}`] = {
        scaleN: N,
        totalNeurons,
        totalEdges,
        avgDegree,
        stageBreakdownMs: {
          parseMs,
          weaveMs,
          persistMs,
          compileTotalMs,
          coldLoadMs,
        },
        resourceMetrics: {
          sqliteSizeKB,
          rssMB,
          heapUsedMB,
        },
        latencyMs: {
          fts5_sem: {
            mean: round2(mean(latsFts)),
            p50: round2(percentile(latsFts, 50)),
            p95: round2(percentile(latsFts, 95)),
            p99: round2(percentile(latsFts, 99)),
          },
          hebb_no_prop: {
            mean: round2(mean(latsNoProp)),
            p50: round2(percentile(latsNoProp, 50)),
            p95: round2(percentile(latsNoProp, 95)),
            p99: round2(percentile(latsNoProp, 99)),
          },
          hebb_v12_a4: {
            mean: round2(mean(latsA4)),
            p50: round2(percentile(latsA4, 50)),
            p95: a4P95,
            p99: round2(percentile(latsA4, 99)),
          },
        },
        bottlenecks: {
          compileBottleneck,
          latencyBottleneck,
        },
      };
    } finally {
      try {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      } catch { }
    }
  }

  return results;
}

// === RUNNER PART B: SEGUNDO DOMINIO MULTI-HOP Y PODER ESTADISTICO ===

function runPartB() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hebbmesh-scale-partB-'));
  try {
    const totalPublic = 2000;
    const totalPrivate = 20;

    const { queries, privacyProbes } = generateClinicalCorpus(tmpDir, {
      totalPublic,
      totalPrivate,
      withEvalGroundTruth: true,
      seed: GLOBAL_SEED,
    });

    // 1. Lexical Gap Analysis (mapa raw sin seeds vs queries semanticas)
    const rawMapWithoutSeeds = parseMemoryDirectory(tmpDir, { useSemanticSeeds: false });
    const lexicalGap = computeLexicalGap(rawMapWithoutSeeds, queries);

    // 2. Compilar Red Protegida (Firewall ON) y Mapa con Seeds para FTS5
    const netProtected = compileNetwork(tmpDir, {
      fresh: true,
      useSemanticSeeds: true,
      m0Topology: true,
      disablePrivacyFirewall: false,
      noPersist: true,
    });

    const rawMapWithSeeds = parseMemoryDirectory(tmpDir, { useSemanticSeeds: true });
    const ftsPureDb = buildFts5Baseline(rawMapWithSeeds, false);
    const ftsSemDb = buildFts5Baseline(rawMapWithSeeds, true);

    // 3. Evaluar 5 Variantes (Hebbian OFF, reinforce:false, topK:5, maxEpisodic:1)
    const v1 = evaluateVariant('fts5_pure', '1. SQLite FTS5 (BM25 pure)', (q) => queryFts5(ftsPureDb, q, 5), queries);
    const v2 = evaluateVariant('fts5_sem', '2. SQLite FTS5 + | sem:', (q) => queryFts5(ftsSemDb, q, 5), queries);
    const v3 = evaluateVariant('hebb_no_prop', '3. HebbMesh v1.2 sin propagacion', (q) => {
      const r = activateNetworkInMemory(netProtected, q, { topK: 5, disablePropagation: true, maxEpisodic: 1, reinforce: false });
      return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
    }, queries);
    const v4 = evaluateVariant('hebb_v12_a4', '4. HebbMesh v1.2 A4', (q) => {
      const r = activateNetworkInMemory(netProtected, q, { topK: 5, maxHops: 2, maxEpisodic: 1, reinforce: false });
      return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
    }, queries);
    const v5 = evaluateVariant('hebb_v12_a4_gaba', '5. HebbMesh v1.2 A4 + GABA Soft', (q) => {
      const r = activateNetworkInMemory(netProtected, q, { topK: 5, maxHops: 2, useGabaSoft: true, maxEpisodic: 1, reinforce: false });
      return { codes: r.winners.map((w) => w.code), types: r.winners.map((w) => w.type), ms: r.ms };
    }, queries);

    const variants = [v1, v2, v3, v4, v5];

    // 4. Cluster Bootstrap (10,000 iters, Mulberry32 seed=20261123, 95% CI)
    const bootstrap = {
      iterations: BOOTSTRAP_ITERS,
      seed: GLOBAL_SEED,
      perVariant: {},
      pairedDeltas: {
        a4_minus_noProp: {},
        a4_minus_fts5Sem: {},
      },
    };

    for (let vi = 0; vi < variants.length; vi++) {
      const v = variants[vi];
      bootstrap.perVariant[v.id] = {};
      for (let ci = 0; ci < BOOT_CATS_PART_B.length; ci++) {
        const cat = BOOT_CATS_PART_B[ci];
        bootstrap.perVariant[v.id][cat] = clusterBootstrap(
          v.perQuery,
          null,
          cat,
          BOOTSTRAP_ITERS,
          GLOBAL_SEED + vi * 100 + ci
        );
      }
    }

    for (let ci = 0; ci < BOOT_CATS_PART_B.length; ci++) {
      const cat = BOOT_CATS_PART_B[ci];
      bootstrap.pairedDeltas.a4_minus_noProp[cat] = clusterBootstrap(
        v4.perQuery,
        v3.perQuery,
        cat,
        BOOTSTRAP_ITERS,
        GLOBAL_SEED + 1000 + ci
      );
      bootstrap.pairedDeltas.a4_minus_fts5Sem[cat] = clusterBootstrap(
        v4.perQuery,
        v2.perQuery,
        cat,
        BOOTSTRAP_ITERS,
        GLOBAL_SEED + 2000 + ci
      );
    }

    // 5. Privacy Firewall F-01 (20 Sondas Adversarias + Control Positivo Mandatorio)
    const netUnsafe = compileNetwork(tmpDir, {
      fresh: true,
      useSemanticSeeds: true,
      m0Topology: true,
      disablePrivacyFirewall: true,
      noPersist: true,
    });

    const probeDetails = [];
    let positiveControlTriggered = 0;
    let firewallOnLeaks = 0;

    for (const p of privacyProbes) {
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

      probeDetails.push({
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

    // Aislamiento lateral de dendritas (F-01)
    let lateralDendriteLeaks = 0;
    for (const n of Object.values(netProtected.neurons)) {
      if (n.private) continue;
      for (const dTok of Object.keys(n.dendrites || {})) {
        if (dTok.startsWith('confidential_clin_') || dTok.startsWith('secret_vault_bridge_')) {
          lateralDendriteLeaks++;
        }
      }
    }

    const privacyResults = {
      totalProbes: privacyProbes.length,
      positiveControlTriggered,
      firewallOnLeaks,
      lateralDendriteLeaks,
      pass: positiveControlTriggered === privacyProbes.length && firewallOnLeaks === 0 && lateralDendriteLeaks === 0,
      details: probeDetails,
    };

    return {
      dataset: {
        totalNeurons: netProtected.metrics.totalNeurons,
        publicNeurons: Object.values(netProtected.neurons).filter((n) => !n.private).length,
        privateNeurons: Object.values(netProtected.neurons).filter((n) => n.private).length,
        totalEdges: netProtected.metrics.totalEdges,
        avgDegree: round2(netProtected.metrics.totalEdges / netProtected.metrics.totalNeurons),
        queriesTotal: queries.length,
        clustersTotal: new Set(queries.map((q) => q.cluster)).size,
        byCategory: {
          direct: queries.filter((q) => q.cat === 'direct').length,
          semantic: queries.filter((q) => q.cat === 'semantic').length,
          multihop_convergent: queries.filter((q) => q.cat === 'multihop_convergent').length,
          episodic_convergent: queries.filter((q) => q.cat === 'episodic_convergent').length,
        },
      },
      lexicalGap: {
        totalSemantic: lexicalGap.totalSemantic,
        zeroOverlapCount: lexicalGap.zeroOverlapCount,
        zeroOverlapPct: lexicalGap.zeroOverlapPct,
        meanOverlapTokens: lexicalGap.meanOverlapTokens,
        details: lexicalGap.details,
      },
      variants: variants.map(({ perQuery, ...summary }) => summary),
      bootstrap,
      privacyFirewall: privacyResults,
    };
  } finally {
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } catch { }
  }
}

// === EJECUCION GENERAL Y REPORTE ===

console.log('Iniciando Benchmark de Escalabilidad y Segundo Dominio Clinico (v1.2)...');
const tGlobal0 = performance.now();

const partAResults = runPartA();
const partBResults = runPartB();

const totalBenchmarkMs = round2(performance.now() - tGlobal0);

const fullReport = {
  protocol: 'PROTOCOLO_ESCALABILIDAD_SEGUNDO_DOMINIO_v1.2',
  timestamp: new Date().toISOString(),
  seed: GLOBAL_SEED,
  totalBenchmarkMs,
  partA_stressTest: partAResults,
  partB_secondDomain: partBResults,
};

fs.writeFileSync(outJsonPath, JSON.stringify(fullReport, null, 2), 'utf8');

// === RESUMEN ASCII / MARKDOWN EN CONSOLA ===

console.log('\n================================================================================');
console.log('  HEBBMESH v1.2 -- REPORTE DE ESCALABILIDAD Y SEGUNDO DOMINIO CLINICO');
console.log('================================================================================');
console.log(`Timestamp: ${fullReport.timestamp} | Semilla Mulberry32: ${GLOBAL_SEED}`);
console.log(`Duracion total ejecucion suite: ${totalBenchmarkMs} ms\n`);

console.log('=== PART A: STRESS TEST DE ESCALA (N = 1000, 5000, 10000) ===');
console.log('| Target N | Neurons | Edges | AvgDeg | Parse ms | Weave ms | Persist ms | Compile ms | ColdLoad ms | DB KB | RAM MB | FTS5 P95 | HebbP95 | Bottleneck |');
console.log('|----------|---------|-------|--------|----------|----------|------------|------------|-------------|-------|--------|----------|---------|------------|');
for (const [key, item] of Object.entries(partAResults)) {
  const sb = item.stageBreakdownMs;
  const rm = item.resourceMetrics;
  const lm = item.latencyMs;
  const bn = (item.bottlenecks.compileBottleneck || item.bottlenecks.latencyBottleneck) ? 'ALERTA' : '0 BOTTLENECK';
  console.log(
    `| ${String(item.scaleN).padStart(8)} | ${String(item.totalNeurons).padStart(7)} | ${String(item.totalEdges).padStart(5)} | ${String(item.avgDegree).padStart(6)} | ${String(sb.parseMs).padStart(8)} | ${String(sb.weaveMs).padStart(8)} | ${String(sb.persistMs).padStart(10)} | ${String(sb.compileTotalMs).padStart(10)} | ${String(sb.coldLoadMs).padStart(11)} | ${String(rm.sqliteSizeKB).padStart(5)} | ${String(rm.rssMB).padStart(6)} | ${String(lm.fts5_sem.p95).padStart(8)} | ${String(lm.hebb_v12_a4.p95).padStart(7)} | ${bn.padStart(10)} |`
  );
}

console.log('\n=== PART B: SEGUNDO DOMINIO CLINICO (N = 2020 NEURONAS, 200 QUERIES EN 4 CATEGORIAS) ===');
console.log(`Dataset: ${partBResults.dataset.totalNeurons} neuronas (${partBResults.dataset.publicNeurons} publicas, ${partBResults.dataset.privateNeurons} privadas), ${partBResults.dataset.totalEdges} aristas (AvgDeg: ${partBResults.dataset.avgDegree})`);
console.log(`Gap Lexico Semantico: ${partBResults.lexicalGap.zeroOverlapCount}/${partBResults.lexicalGap.totalSemantic} con cero solapamiento (${partBResults.lexicalGap.zeroOverlapPct}%) | Media tokens solapados: ${partBResults.lexicalGap.meanOverlapTokens}`);

console.log('\n| Variante                             | Direct@3 | Semantic@3 | MultiHop@3 | Episodic@3 | Overall@3 | EpCtrl@3 (Viol) | Mean ms | P95 ms |');
console.log('|--------------------------------------|----------|------------|------------|------------|-----------|-----------------|---------|--------|');
for (const v of partBResults.variants) {
  const epCtrlStr = `${v.episodicControl.hit3Pct}% (${v.episodicControl.superHubViolations}/${v.episodicControl.n})`;
  console.log(
    `| ${v.label.padEnd(36)} | ${String(v.byCat.direct.hit3Pct + '%').padStart(8)} | ${String(v.byCat.semantic.hit3Pct + '%').padStart(10)} | ${String(v.byCat.multihop_convergent.hit3Pct + '%').padStart(10)} | ${String(v.byCat.episodic_convergent.hit3Pct + '%').padStart(10)} | ${String(v.overall.hit3Pct + '%').padStart(9)} | ${epCtrlStr.padStart(15)} | ${String(v.latencyMs.mean).padStart(7)} | ${String(v.latencyMs.p95).padStart(6)} |`
  );
}

console.log('\n[Paired Cluster Bootstrap 10,000x en Part B (Mulberry32 seed=20261123, 95% CI)]');
console.log('-- Comparacion: HebbMesh v1.2 A4 vs HebbMesh sin propagacion (a4_minus_noProp) --');
for (const c of BOOT_CATS_PART_B) {
  const d = partBResults.bootstrap.pairedDeltas.a4_minus_noProp[c];
  console.log(`  ${c.padEnd(20)}: delta=${d.point >= 0 ? '+' : ''}${d.point} pts | 95% CI [${d.ci95[0]}, ${d.ci95[1]}] | clusters=${d.clusters} | verdict=${d.verdict}`);
}
console.log('-- Comparacion: HebbMesh v1.2 A4 vs SQLite FTS5 + | sem: (a4_minus_fts5Sem) --');
for (const c of BOOT_CATS_PART_B) {
  const d = partBResults.bootstrap.pairedDeltas.a4_minus_fts5Sem[c];
  console.log(`  ${c.padEnd(20)}: delta=${d.point >= 0 ? '+' : ''}${d.point} pts | 95% CI [${d.ci95[0]}, ${d.ci95[1]}] | clusters=${d.clusters} | verdict=${d.verdict}`);
}

console.log('\n[F-01 Privacy Firewall: 20 Sondas Adversarias (14 Activacion + 6 Bridge A*)]');
console.log(`  Control Positivo (Firewall OFF): ${partBResults.privacyFirewall.positiveControlTriggered}/${partBResults.privacyFirewall.totalProbes} activaron PRIV-*`);
console.log(`  Firewall ON:                     ${partBResults.privacyFirewall.firewallOnLeaks}/${partBResults.privacyFirewall.positiveControlTriggered} fugas | lateralDendriteLeaks=${partBResults.privacyFirewall.lateralDendriteLeaks} | PASS=${partBResults.privacyFirewall.pass}`);

console.log(`\nReporte JSON persistido exitosamente en: ${outJsonPath}\n`);
