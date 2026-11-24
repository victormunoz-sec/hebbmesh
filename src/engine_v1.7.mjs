import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import {
  loadNetwork,
  saveNetwork,
  applyHebbianDelta,
  getStorageMtime,
  getActiveStoragePath,
} from './storage.mjs';

/**
 * HebbMesh Engine v1.7.0-HebbMesh-4D-ZeroInference
 * Zero-Bloat 4D Synaptic Memory over Markdown (0 npm deps, 0 embeddings, 0 LLM)
 *
 * v1.7 Enhancements over v1.3:
 *  - F2b_025: Extended lexical capacity (coreTags + weighted extraTags at 0.25)
 *  - F4: Working Memory temporal NEXT bidirectional propagation + question polarity
 *  - U1: Same-speaker episodic continuity (t-2 rare-token context + SameSpk edges)
 *  - U2: Speaker-author vs third-party mention weighting (mentionMult = 0.60)
 *  - U3: Session turn sequential tokenization (up to 48 tokens, caption boilerplate filtered)
 *  - U4: Conversation query pronoun/adverb noise filtering (EXTRA_STD_STOP)
 *  - S1_WB: Word-boundary title matching
 *  - PMI_Safe: Lazy-cached positive-PMI co-occurrence expansion with F-01 isolation and pos=0 shield
 *  - P1b_PureCov: Pauli Exclusion / Synaptic Refraction top-K coverage reranking
 *  - CMI_ZERO_REG: Anti-poisoning immunity (obsolete propagation brake, recency, anti-stuffing)
 */

export const VERSION = '1.7.0-HebbMesh-4D-ZeroInference';

const STOPWORDS = new Set([
  // English
  'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'are', 'was', 'were',
  'been', 'being', 'will', 'would', 'could', 'should', 'into', 'over', 'after', 'before',
  'about', 'between', 'under', 'above', 'what', 'when', 'where', 'which', 'while', 'how',
  'why', 'who', 'all', 'any', 'both', 'each', 'more', 'most', 'other', 'some', 'such',
  'only', 'own', 'same', 'than', 'too', 'very', 'just', 'not', 'but', 'can', 'did', 'does',
  'has', 'had', 'its', 'our', 'out', 'use', 'using', 'used', 'via', 'new', 'also', 'one',
  // Spanish
  'que', 'con', 'por', 'para', 'una', 'uno', 'unos', 'unas', 'del', 'las', 'los', 'como',
  'pero', 'mas', 'sin', 'sobre', 'entre', 'esta', 'este', 'esto', 'estos', 'estas', 'ese',
  'eso', 'esa', 'esos', 'esas', 'aqui', 'alli', 'donde', 'cuando', 'quien', 'cual', 'todo',
  'toda', 'todos', 'todas', 'muy', 'bien', 'mal', 'hoy', 'ayer', 'algo', 'nada', 'solo',
  'ahora', 'antes', 'despues', 'hacer', 'tiene', 'tener', 'puede', 'poder', 'desde', 'hasta',
  'hacia', 'cada', 'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'porque', 'aunque',
  'siempre', 'nunca', 'paso', 'habia', 'estan', 'son', 'fue', 'ser', 'hay',
]);

const EXTRA_STD_STOP = new Set([
  'her', 'his', 'she', 'him', 'they', 'them', 'their', 'you', 'your', 'yours',
  'many', 'much', 'might', 'likely', 'still', 'really',
]);

const IMAGE_BOILERPLATE = new Set([
  'image', 'photo', 'picture', 'showing', 'shows', 'depicting',
]);

const WM_NEXT_CFG = {
  maxAnchorSeeds: 3,
  minAnchorRel: 0.55,
  fwdWeight: 0.50,
  fwdQuestionWeight: 0.55,
  bwdWeight: 0.42,
  bwdQuestionWeight: 0.48,
  directBoostScale: 0.25,
};

const CMI_ZERO_REG_DEFAULTS = {
  validFalseMult: 0.30,
  brakeObsoleteProp: true,
  obsoleteEmitFactor: 0.35,
  obsoleteDstFactor: 0.20,
  resetObsoleteMass: true,
  recencyWeight: 0.38,
  antiStuffing: true,
  stuffInboxOnly: false,
  stuffTagThreshold: 15,
  stuffFloor: 0.85,
  authority: false,
};

const PMI_SAFE_DEFAULTS = {
  weight: 0.22,
  maxAssoc: 3,
  minQIdf: 1.5,
  maxDfRatio: 0.12,
  capNode: 1.6,
};

const U1_DEFAULTS = {
  wPrev: 0.40,
  wSameSpk: 0.25,
  sameSpkFwd: 0.42,
  sameSpkBwd: 0.35,
};

const U2_DEFAULTS = {
  mentionMult: 0.60,
};

const PAULI_DEFAULTS = {
  gamma: 0.65,
  alpha: 0.08,
  beta: 0.22,
  sessDecay: 1.0,
  poolSize: 40,
  minInfoIdf: 1.10,
  startPos: 1,
  exemptNext: true,
  gateUncoveredOnly: false,
  onlyWithSessions: false,
  minInfoToks: 0,
};

export function norm(s = '') {
  return String(s || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

function round2(n) {
  return Math.round(n * 100) / 100;
}

function clamp01(n) {
  return Math.max(0, Math.min(1, Number(n) || 0));
}

function readSafe(p) {
  try {
    return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
  } catch {
    return '';
  }
}

function getSessionKey(n) {
  const sid = n?.sessionId ?? n?.session;
  if (sid === undefined || sid === null) return '';
  return String(sid).trim();
}

function buildP4Map(tags) {
  const byP4 = new Map();
  for (let i = 0; i < tags.length; i++) {
    const t = tags[i];
    if (t.length < 4) continue;
    const p4 = t.slice(0, 4);
    let arr = byP4.get(p4);
    if (!arr) { arr = []; byP4.set(p4, arr); }
    arr.push(t);
  }
  return byP4;
}

export function tokenize(text = '') {
  return norm(text)
    .replace(/[^a-z0-9_\-\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w) && !/^\d+$/.test(w));
}

export function extractTags(text, max = 18) {
  const freq = new Map();
  for (const w of tokenize(text)) {
    freq.set(w, (freq.get(w) || 0) + 1);
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .slice(0, max)
    .map(([w]) => w);
}

function extractSessionTurnTags(text, max = 48) {
  const toks = tokenize(text);
  const seen = new Set();
  const out = [];
  for (let i = 0; i < toks.length && out.length < max; i++) {
    const w = toks[i];
    if (IMAGE_BOILERPLATE.has(w) || seen.has(w)) continue;
    seen.add(w);
    out.push(w);
  }
  return out;
}

/**
 * Extracts inline Zero-Embedding Seeds (`| sem: a, b, c`), Affective Echo (`| eco: ...`),
 * and Session ID (`| session: ...`).
 */
export function parseMetadata(lineRaw = '') {
  let clean = String(lineRaw || '').replace(/<!--.*?-->/g, '').trim();
  let echo = '';
  let sem = [];
  let sessionId = '';

  const codeRanges = [...clean.matchAll(/`[^`]+`/g)].map((m) => [m.index, m.index + m[0].length]);
  const inCode = (idx) => codeRanges.some(([s, e]) => idx >= s && idx < e);
  const pipeIdxs = [];
  for (let i = 0; i < clean.length; i++) {
    if (clean[i] === '|' && !inCode(i)) pipeIdxs.push(i);
  }

  for (let k = pipeIdxs.length - 1; k >= 0; k--) {
    const idx = pipeIdxs[k];
    const tail = clean.slice(idx);
    const mEco = !echo && tail.match(/^\|\s*eco:\s*(.+)$/i);
    if (mEco) {
      echo = mEco[1].trim();
      clean = clean.slice(0, idx).trim();
      continue;
    }
    const mSem = !sem.length && tail.match(/^\|\s*sem:\s*(.+)$/i);
    if (mSem) {
      sem = mSem[1].split(/[,;]+/).map((s) => s.trim()).filter((s) => s.length >= 2);
      clean = clean.slice(0, idx).trim();
      continue;
    }
    const mSess = !sessionId && tail.match(/^\|\s*session(?:id)?:\s*(.+)$/i);
    if (mSess) {
      sessionId = mSess[1].trim();
      clean = clean.slice(0, idx).trim();
      continue;
    }
    break;
  }

  if (!echo) {
    const mEco = clean.match(/_\(eco:\s*([^)]+)\)_/i);
    if (mEco && !inCode(mEco.index)) {
      echo = mEco[1].trim();
      clean = clean.replace(mEco[0], '').trim();
    }
  }
  if (!sem.length) {
    const mSem = clean.match(/_\(sem:\s*([^)]+)\)_/i);
    if (mSem && !inCode(mSem.index)) {
      sem = mSem[1].split(/[,;]+/).map((s) => s.trim()).filter((s) => s.length >= 2);
      clean = clean.replace(mSem[0], '').trim();
    }
  }

  return { clean: clean.replace(/\|\s*$/, '').trim(), echo, sem, sessionId };
}

export function extractKeyConcepts(textRaw = '', max = 10) {
  const txt = String(textRaw || '').replace(/<!--.*?-->/g, ' ');
  const found = new Map();
  const add = (raw) => {
    const c = raw.trim().replace(/^['"`(]+|['"`.,;:)]+$/g, '');
    if (c.length < 3 || c.length > 36) return;
    if (/^\d+$/.test(c) || /\.(?:md|mjs|js|ts|json|html|css|py|sql)$/i.test(c)) return;
    const k = norm(c);
    if (STOPWORDS.has(k)) return;
    if (!found.has(k)) found.set(k, c);
  };

  for (const m of txt.matchAll(/`([A-Za-z0-9_./:@+-]{3,36})`/g)) add(m[1]);
  for (const m of txt.matchAll(/\b([A-Z][a-zA-Z0-9]+(?:-[A-Za-z0-9]+)+|[A-Z]{2,}[A-Za-z0-9-]{1,})\b/g)) {
    if (!/^(?:UTF-8|ISO|JSON|HTML|HTTP|REST|API|TODO|NOTE|PASS|FAIL)$/i.test(m[1])) add(m[1]);
  }
  return [...found.values()].slice(0, max);
}

export function compute4DCoords(type, subtype, text, valid = true) {
  const t = norm(text);
  const dim4 = { daily: 0.15, exploration: 0.15, execution: 0.15, support: 0.15 };

  if (type === 'graph') {
    if (subtype === 'C1' || subtype === 'DAILY') {
      dim4.daily = 0.88; dim4.support = 0.45;
    } else if (subtype === 'C2' || subtype === 'EXECUTION') {
      dim4.execution = 0.92; dim4.exploration = 0.35;
    } else if (subtype === 'C3' || subtype === 'EXPLORATION') {
      dim4.exploration = 0.94; dim4.execution = 0.42;
    } else {
      dim4.support = 0.94; dim4.daily = 0.48;
    }
  } else if (type === 'lesson' || type === 'knowledge') {
    dim4.exploration = 0.88; dim4.execution = 0.46;
  } else if (type === 'milestone') {
    dim4.execution = 0.90; dim4.exploration = 0.48;
  } else if (type === 'profile') {
    dim4.daily = 0.90; dim4.support = 0.45;
  } else if (type === 'private') {
    dim4.support = 0.98; dim4.daily = 0.40;
  } else if (type === 'episodic') {
    if (subtype === 'F1') { dim4.daily = 0.90; dim4.support = 0.35; }
    else if (subtype === 'F2') { dim4.exploration = 0.90; dim4.execution = 0.38; }
    else if (subtype === 'F3') { dim4.execution = 0.90; dim4.exploration = 0.35; }
    else { dim4.support = 0.92; dim4.daily = 0.40; }
  } else if (type === 'inbox') {
    if (subtype === 'SUPPORT' || subtype === 'PRIVATE') { dim4.support = 0.94; dim4.daily = 0.45; }
    else if (subtype === 'PROFILE') { dim4.daily = 0.88; dim4.support = 0.40; }
    else if (subtype === 'PROJECT' || subtype === 'TECH') { dim4.execution = 0.88; dim4.exploration = 0.52; }
    else { dim4.exploration = 0.88; dim4.execution = 0.42; }
  }

  if (/(architecture|paper|benchmark|graph|synapse|hebb|algorithm|eval|model|research|sqlite|rag)/.test(t)) {
    dim4.exploration = Math.min(1, dim4.exploration + 0.22);
  }
  if (/(deploy|production|bug|latency|stripe|webhook|ci|docker|postgres|cloudflare|incident|fix|release|billing)/.test(t)) {
    dim4.execution = Math.min(1, dim4.execution + 0.22);
  }
  if (/(burnout|stress|anxiety|fatigue|calm|family|rest|overwhelmed|worry|relief|gratitude|trust)/.test(t)) {
    dim4.support = Math.min(1, dim4.support + 0.25);
  }
  if (/(coffee|tea|music|guitar|cat|dog|running|cooking|ramen|gaming|routine|weekend)/.test(t)) {
    dim4.daily = Math.min(1, dim4.daily + 0.22);
  }

  for (const k of Object.keys(dim4)) dim4[k] = round2(dim4[k]);

  const logicalPole = Math.max(dim4.exploration, dim4.execution);
  const affectivePole = Math.max(dim4.support, dim4.daily * 0.85);
  const x = round2(Math.max(-1, Math.min(1, logicalPole - affectivePole)));
  const y = valid ? 0.90 : 0.25;

  let z = 0.68;
  if (type === 'episodic') z = 0.62;
  else if (type === 'graph' || type === 'private') z = 0.86;
  else if (type === 'lesson' || type === 'knowledge') z = 0.78;
  else if (type === 'milestone') z = 0.74;

  return { dim4, coord3d: { x, y: round2(y), z: round2(z) }, weight: round2(z) };
}

function collectMarkdownSources(memoryDir) {
  const files = [];
  for (const rel of ['graph.md', 'lessons.md', 'milestones.md', 'profile.md', 'inbox.md', 'private_vault.md']) {
    const p = path.resolve(memoryDir, rel);
    if (fs.existsSync(p)) files.push(p);
  }
  for (const sub of ['knowledge', 'synthesis']) {
    const d = path.resolve(memoryDir, sub);
    if (fs.existsSync(d)) {
      for (const f of fs.readdirSync(d)) {
        if (f.endsWith('.md')) files.push(path.join(d, f));
      }
    }
  }
  return files;
}

function buildNodeTagsAndM0(full, type, sessionId) {
  const m0Tags = extractTags(full, type === 'episodic' ? 15 : 18);
  if (sessionId) {
    return {
      tags: extractSessionTurnTags(full, 48),
      _m0Tags: m0Tags,
      hasQuestion: full.includes('?'),
    };
  }
  return {
    tags: extractTags(full, type === 'episodic' ? 15 : 128),
    _m0Tags: m0Tags,
    hasQuestion: false,
  };
}

export function parseMemoryDirectory(memoryDir, options = {}) {
  const disableSeeds = Boolean(options.disableSeeds) || options.useSemanticSeeds === false;
  const neurons = new Map();

  // 1. Temporal Knowledge Graph (`graph.md` -> G-xx)
  const graphRaw = readSafe(path.resolve(memoryDir, 'graph.md'));
  let community = 'C1';
  for (const line of graphRaw.split(/\r?\n/)) {
    const mH = line.match(/^##+\s+(C\d+)\s*[·:-]/i);
    if (mH) { community = mH[1].toUpperCase(); continue; }
    const mG = line.match(/^\s*-\s+\*\*(G-\d+)\*\*\s*`\[(V|S)\]`\s*(.+)/i);
    if (!mG) continue;
    const code = mG[1].toUpperCase();
    const valid = mG[2].toUpperCase() === 'V';
    const { clean, echo, sem, sessionId } = parseMetadata(mG[3]);
    const id = `n_${code.replace('-', '_')}`;
    const { dim4, coord3d, weight } = compute4DCoords('graph', community, clean, valid);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(clean, 'graph', sessionId);
    neurons.set(id, {
      id, code, type: 'graph', community,
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid, private: false,
      ptr: `graph.md#${code}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 2. Lessons (`lessons.md` -> A-xx)
  const lessonsRaw = readSafe(path.resolve(memoryDir, 'lessons.md'));
  for (const line of lessonsRaw.split(/\r?\n/)) {
    const mA = line.match(/^\s*-\s+\*\*(A-\d+)\s*`\[(V|S)\]`\s*([^:*]+):\*\*\s*(.+)/i);
    if (!mA) continue;
    const code = mA[1].toUpperCase();
    const valid = mA[2].toUpperCase() === 'V';
    const title = mA[3].trim();
    const { clean, echo, sem, sessionId } = parseMetadata(mA[4]);
    const full = `${title}: ${clean}`;
    const id = `n_${code.replace('-', '_')}`;
    const { dim4, coord3d, weight } = compute4DCoords('lesson', 'EXPLORATION', full, valid);
    const concepts = extractKeyConcepts(full, 8);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(full, 'lesson', sessionId);
    neurons.set(id, {
      id, code, type: 'lesson', community: 'EXPLORATION',
      title: title.slice(0, 90),
      summary: full.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(concepts.length ? { concepts } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid, private: false,
      ptr: `lessons.md#${code}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 3. Milestones (`milestones.md` -> HITO-xx)
  const milesRaw = readSafe(path.resolve(memoryDir, 'milestones.md'));
  let idxM = 0;
  for (const line of milesRaw.split(/\r?\n/)) {
    if (!/^\s*-\s+\[/.test(line)) continue;
    idxM++;
    const { clean, echo, sem, sessionId } = parseMetadata(line.replace(/^\s*-\s+\[[^\]]+\]\s*/, ''));
    if (!clean) continue;
    const code = `HITO-${idxM}`;
    const id = `n_HITO_${idxM}`;
    const { dim4, coord3d, weight } = compute4DCoords('milestone', 'EXECUTION', clean, true);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(clean, 'milestone', sessionId);
    neurons.set(id, {
      id, code, type: 'milestone', community: 'EXECUTION',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid: true, private: false,
      ptr: `milestones.md#L${idxM}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 4. Profile (`profile.md` -> PERFIL-xx)
  const profRaw = readSafe(path.resolve(memoryDir, 'profile.md'));
  let idxP = 0;
  for (const line of profRaw.split(/\r?\n/)) {
    if (!/^\s*-\s+/.test(line)) continue;
    idxP++;
    const { clean, echo, sem, sessionId } = parseMetadata(line.replace(/^\s*-\s+/, ''));
    if (clean.length < 8) continue;
    const code = `PERFIL-${idxP}`;
    const id = `n_PERFIL_${idxP}`;
    const { dim4, coord3d, weight } = compute4DCoords('profile', 'DAILY', clean, true);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(clean, 'profile', sessionId);
    neurons.set(id, {
      id, code, type: 'profile', community: 'DAILY',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid: true, private: false,
      ptr: `profile.md#L${idxP}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 5. Inbox (`inbox.md` -> INBOX-xx)
  const inboxRaw = readSafe(path.resolve(memoryDir, 'inbox.md'));
  let idxI = 0;
  for (const line of inboxRaw.split(/\r?\n/)) {
    const mI = line.match(/^\s*-\s+\[[^\]]+\]\s*\[([A-Z]+)\]\s*(.+)/);
    if (!mI) continue;
    idxI++;
    const sub = mI[1].toUpperCase();
    const { clean, echo, sem, sessionId } = parseMetadata(mI[2]);
    const code = `INBOX-${idxI}`;
    const id = `n_INBOX_${idxI}`;
    const isPriv = sub === 'PRIVATE' || sub === 'INTIMO';
    const { dim4, coord3d, weight } = compute4DCoords('inbox', sub, clean, true);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(clean, 'inbox', sessionId);
    neurons.set(id, {
      id, code, type: 'inbox', community: sub,
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid: true, private: isPriv,
      ptr: `inbox.md#L${idxI}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 6. Private Vault (`private_vault.md` -> PRIV-xx, protected by Privacy Firewall)
  const privRaw = readSafe(path.resolve(memoryDir, 'private_vault.md'));
  let idxV = 0;
  for (const line of privRaw.split(/\r?\n/)) {
    if (!/^\s*-\s+/.test(line)) continue;
    idxV++;
    const { clean, echo, sem, sessionId } = parseMetadata(line.replace(/^\s*-\s+/, ''));
    if (clean.length < 8) continue;
    const code = `PRIV-${idxV}`;
    const id = `n_PRIV_${idxV}`;
    const { dim4, coord3d, weight } = compute4DCoords('private', 'PRIVATE', clean, true);
    const { tags, _m0Tags, hasQuestion } = buildNodeTagsAndM0(clean, 'private', sessionId);
    neurons.set(id, {
      id, code, type: 'private', community: 'PRIVATE',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(hasQuestion ? { hasQuestion: true } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags,
      _m0Tags,
      dim4, coord3d, weight, valid: true, private: true,
      ptr: `private_vault.md#L${idxV}`,
      edges: {}, dendrites: {}, activations: 0,
    });
  }

  // 7. Knowledge Library (`knowledge/*.md` -> DOC-xx)
  const knowDir = path.resolve(memoryDir, 'knowledge');
  if (fs.existsSync(knowDir)) {
    const files = fs.readdirSync(knowDir).filter((f) => f.endsWith('.md')).sort();
    let idxD = 0;
    for (const f of files) {
      const raw = readSafe(path.join(knowDir, f));
      if (!raw) continue;
      idxD++;
      const mTit = raw.match(/^#\s+(.+)/m);
      const mSum = raw.match(/^summary:\s*(.+)/im) || raw.match(/^resumen:\s*(.+)/im);
      const mSem = raw.match(/^sem:\s*(.+)/im);
      const title = (mTit ? mTit[1] : f.replace(/\.md$/, '')).trim();
      const summary = mSum
        ? `${title}: ${mSum[1].trim()}`
        : `${title} - ${raw.replace(/^---[\s\S]*?---/, '').replace(/\s+/g, ' ').trim().slice(0, 140)}`;
      const sem = !disableSeeds && mSem ? mSem[1].split(/[,;]+/).map((s) => s.trim()).filter(Boolean) : [];
      const concepts = extractKeyConcepts(raw, 12);
      const code = `DOC-${idxD}`;
      const id = `n_DOC_${idxD}`;
      const full = `${title} ${summary} ${raw.slice(0, 900)}`;
      const { dim4, coord3d, weight } = compute4DCoords('knowledge', 'KNOWLEDGE', `${title} ${summary}`, true);
      const allTags = extractTags(full, 128);
      neurons.set(id, {
        id, code, type: 'knowledge', community: 'KNOWLEDGE',
        title: title.slice(0, 90),
        summary: summary.slice(0, 180),
        ...(concepts.length ? { concepts } : {}),
        ...(sem.length ? { sem } : {}),
        tags: allTags,
        _m0Tags: allTags.slice(0, 18),
        dim4, coord3d, weight, valid: true, private: false,
        ptr: `knowledge/${f}`,
        edges: {}, dendrites: {}, activations: 0,
      });
    }
  }

  // 8. Weekly Episodic Syntheses (`synthesis/AAAA-Wnn.md` -> S-Wnn-F1..F4)
  const synthDir = path.resolve(memoryDir, 'synthesis');
  if (fs.existsSync(synthDir)) {
    const labels = {
      F1: 'Daily & Life',
      F2: 'Exploration & Decisions',
      F3: 'Execution & Open Tasks',
      F4: 'Support & Reflection',
    };
    const classifyH2 = (h2) => {
      const h = norm(h2);
      if (h.includes('phase 1') || h.includes('fase 1') || h.includes('daily') || h.includes('cotidiano')) return 'F1';
      if (h.includes('phase 2') || h.includes('fase 2') || h.includes('explor') || h.includes('decision')) return 'F2';
      if (h.includes('phase 3') || h.includes('fase 3') || h.includes('execut') || h.includes('ejecu')) return 'F3';
      return 'F4';
    };
    const files = fs.readdirSync(synthDir).filter((f) => /^\d{4}-W\d{2}\.md$/i.test(f)).sort();
    for (const f of files) {
      const raw = readSafe(path.join(synthDir, f));
      if (!raw) continue;
      const semCode = f.replace(/\.md$/i, '').toUpperCase();
      const wShort = (semCode.match(/W\d{2}/i)?.[0] || semCode).toUpperCase();
      const buckets = { F1: [], F2: [], F3: [], F4: [] };
      let activePhase = null;
      for (const l of raw.split(/\r?\n/)) {
        const mH2 = l.match(/^##\s+(.+)/);
        if (mH2) { activePhase = classifyH2(mH2[1]); continue; }
        if (!activePhase) continue;
        const tr = l.trim();
        if (!tr || tr === '---' || tr.startsWith('### ')) continue;
        buckets[activePhase].push(tr.replace(/^[-*>\s]+/, '').replace(/\*\*/g, ''));
      }
      for (const [fKey, lines] of Object.entries(buckets)) {
        if (!lines.length) continue;
        const body = lines.join(' ').replace(/\s+/g, ' ').trim();
        if (body.length < 25) continue;
        const id = `n_S_${semCode.replace(/[^A-Za-z0-9]/g, '_')}_${fKey}`;
        const { dim4, coord3d, weight } = compute4DCoords('episodic', fKey, body, true);
        const baseTags = extractTags(body, 12);
        const tags = [...new Set([...baseTags, wShort.toLowerCase(), 'week', 'weekly', 'synthesis'])].slice(0, 15);
        neurons.set(id, {
          id, code: `S-${wShort}-${fKey}`, type: 'episodic', community: fKey,
          title: `Week ${semCode} - ${labels[fKey]}`,
          summary: body.slice(0, 180),
          tags,
          _m0Tags: [...tags],
          dim4, coord3d, weight, valid: true, private: false,
          ptr: `synthesis/${f}`,
          edges: {}, dendrites: {}, activations: 0,
        });
      }
    }
  }

  if (!disableSeeds && options.seedsOverride && typeof options.seedsOverride === 'object') {
    for (const n of neurons.values()) {
      if (Object.prototype.hasOwnProperty.call(options.seedsOverride, n.code)) {
        const ov = options.seedsOverride[n.code];
        if (Array.isArray(ov) && ov.length) {
          n.sem = ov.map((s) => String(s).trim()).filter((s) => s.length >= 2);
        } else {
          delete n.sem;
        }
      }
    }
  }

  return neurons;
}

function weaveEdgesAndDendrites(neurons, prevNet = null, decay = false, disablePrivacyFirewall = false, options = {}) {
  const m0Topology = options.m0Topology !== false;
  const list = [...neurons.values()];
  const getM0Tags = (n) => n._m0Tags || n.tags || [];

  const tagToIds = new Map();
  for (const n of list) {
    for (const tag of getM0Tags(n)) {
      if (!tagToIds.has(tag)) tagToIds.set(tag, []);
      tagToIds.get(tag).push(n.id);
    }
  }

  const N = Math.max(1, list.length);
  const pairScores = new Map();

  if (m0Topology) {
    const dfMax = Math.min(40, Math.max(6, Math.round(N * 0.05)));
    const tokToEntries = new Map();
    const registerTok = (tok, id, factor) => {
      let m = tokToEntries.get(tok);
      if (!m) { m = new Map(); tokToEntries.set(tok, m); }
      const prev = m.get(id) || 0;
      if (factor > prev) m.set(id, factor);
    };

    for (const n of list) {
      for (const tag of getM0Tags(n)) registerTok(tag, n.id, 1.0);
      if (Array.isArray(n.concepts)) {
        for (const c of n.concepts) {
          for (const tok of tokenize(c)) registerTok(tok, n.id, 0.90);
        }
      }
      if (Array.isArray(n.sem)) {
        for (const phrase of n.sem) {
          for (const tok of tokenize(phrase)) registerTok(tok, n.id, 0.85);
        }
      }
    }

    for (const [, entryMap] of tokToEntries.entries()) {
      const df = entryMap.size;
      if (df < 2 || df > dfMax) continue;
      const idf = Math.min(1, Math.log10(N / df) / 1.4);
      const idfSq = idf * idf;
      const entries = [...entryMap.entries()];
      for (let i = 0; i < entries.length; i++) {
        const [a, fa] = entries[i];
        const privA = Boolean(neurons.get(a)?.private);
        for (let j = i + 1; j < entries.length; j++) {
          const [b, fb] = entries[j];
          if (!disablePrivacyFirewall && privA !== Boolean(neurons.get(b)?.private)) continue;
          const key = a < b ? `${a}|${b}` : `${b}|${a}`;
          pairScores.set(key, (pairScores.get(key) || 0) + 0.30 * idfSq * Math.sqrt(fa * fb));
        }
      }
    }
  } else {
    for (const [, ids] of tagToIds.entries()) {
      if (ids.length < 2 || ids.length > 24) continue;
      const idf = Math.min(1, Math.log10(N / ids.length) / 1.4);
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          const a = ids[i], b = ids[j];
          const key = a < b ? `${a}|${b}` : `${b}|${a}`;
          pairScores.set(key, (pairScores.get(key) || 0) + 0.30 * idf);
        }
      }
    }
  }

  const addEdge = (a, b, w) => {
    const nA = neurons.get(a);
    if (!nA || a === b) return;
    const cur = Number(nA.edges[b] || 0);
    if (w > cur) nA.edges[b] = round2(Math.min(1.0, w));
  };

  for (const [key, score] of pairScores.entries()) {
    if (score < 0.18) continue;
    const [a, b] = key.split('|');
    if (!disablePrivacyFirewall && Boolean(neurons.get(a)?.private) !== Boolean(neurons.get(b)?.private)) continue;
    const w = round2(Math.min(0.92, 0.34 + score * 0.62));
    addEdge(a, b, w);
    addEdge(b, a, w);
  }

  if (prevNet?.neurons) {
    for (const [id, curN] of neurons.entries()) {
      const prevN = prevNet.neurons[id];
      if (!prevN) continue;
      curN.activations = Number(prevN.activations || 0);
      for (const [destId, val] of Object.entries(prevN.edges || {})) {
        if (!neurons.has(destId)) continue;
        if (Boolean(curN.private) !== Boolean(neurons.get(destId)?.private)) continue;
        const wPrev = decay ? Math.max(0.25, round2(Number(val) - 0.01)) : Number(val);
        if (wPrev > Number(curN.edges[destId] || 0)) curN.edges[destId] = round2(wPrev);
      }
    }
  }

  for (const n of list) {
    n.edges = Object.fromEntries(
      Object.entries(n.edges).sort((a, b) => b[1] - a[1]).slice(0, 8)
    );
  }

  for (const n of list) {
    const ownTags = new Set(getM0Tags(n));
    const dend = {};

    if (Array.isArray(n.sem)) {
      for (const phrase of n.sem) {
        for (const tok of tokenize(phrase)) {
          if (ownTags.has(tok)) continue;
          dend[tok] = Math.max(dend[tok] || 0, 0.80);
        }
      }
      delete n.sem;
    }

    for (const [destId, edgeW] of Object.entries(n.edges)) {
      if (edgeW < 0.52) continue;
      const nb = neurons.get(destId);
      if (!nb || (!disablePrivacyFirewall && !n.private && nb.private)) continue;
      for (const tag of getM0Tags(nb)) {
        if (ownTags.has(tag)) continue;
        const df = (tagToIds.get(tag) || []).length;
        if (df < 1 || df > 6) continue;
        const spec = Math.min(1, Math.log10(N / df) / 1.5);
        dend[tag] = round2(Math.min(0.90, (dend[tag] || 0) + edgeW * spec * 0.39));
      }
    }

    n.dendrites = Object.fromEntries(
      Object.entries(dend)
        .filter(([, w]) => w >= 0.22)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 16)
    );
  }

  const linkTemporalNext = options.linkTemporalNext !== false;
  const temporalNextWeight = round2(Number(options.temporalNextWeight ?? 0.72));
  if (linkTemporalNext && temporalNextWeight > 0) {
    const lastBySession = new Map();
    for (const n of list) {
      const key = getSessionKey(n);
      if (!key) continue;
      const prev = lastBySession.get(key);
      if (prev && prev.id !== n.id) {
        if (disablePrivacyFirewall || Boolean(prev.private) === Boolean(n.private)) {
          if ((prev.edges[n.id] || 0) < temporalNextWeight) prev.edges[n.id] = temporalNextWeight;
          if ((n.edges[prev.id] || 0) < temporalNextWeight) n.edges[prev.id] = temporalNextWeight;
        }
      }
      lastBySession.set(key, n);
    }
  }

  for (const n of list) {
    delete n._m0Tags;
  }
}

function buildPmiTopTableSafe({
  groups,
  coreTagsById,
  extraTagsById,
  dfMap,
  N,
  includePrivate,
}) {
  const isEligible = (tok) => {
    if (!tok || tok.length < 4 || STOPWORDS.has(tok) || /^\d+$/.test(tok)) return false;
    const d = dfMap.get(tok) || 0;
    return d >= 2 && (d / N) <= 0.12;
  };

  const winCount = new Map();
  const cooc = new Map();
  let totalWin = 0;

  for (let gi = 0; gi < groups.length; gi++) {
    const seq = groups[gi];
    for (let i = 0; i < seq.length; i++) {
      const center = seq[i];
      if (center.valid === false) continue;
      if (!includePrivate && center.private) continue;

      totalWin++;
      const winSet = new Set();
      const lo = Math.max(0, i - 2);
      const hi = Math.min(seq.length - 1, i + 2);
      const centerPriv = Boolean(center.private);

      const addTags = (tags) => {
        if (!tags) return;
        for (let ti = 0; ti < tags.length; ti++) {
          if (winSet.size >= 40) return;
          const tok = tags[ti];
          if (isEligible(tok)) winSet.add(tok);
        }
      };

      addTags(coreTagsById.get(center.id));
      for (let j = lo; j <= hi && winSet.size < 40; j++) {
        if (j === i) continue;
        const nb = seq[j];
        if (nb.valid === false || (!includePrivate && nb.private)) continue;
        if (Boolean(nb.private) !== centerPriv) continue;
        addTags(coreTagsById.get(nb.id));
      }
      if (winSet.size < 40) addTags(extraTagsById.get(center.id));
      for (let j = lo; j <= hi && winSet.size < 40; j++) {
        if (j === i) continue;
        const nb = seq[j];
        if (nb.valid === false || (!includePrivate && nb.private)) continue;
        if (Boolean(nb.private) !== centerPriv) continue;
        addTags(extraTagsById.get(nb.id));
      }

      if (winSet.size === 0) continue;
      const winToks = [...winSet];
      for (let a = 0; a < winToks.length; a++) {
        const ta = winToks[a];
        winCount.set(ta, (winCount.get(ta) || 0) + 1);
      }
      for (let a = 0; a < winToks.length; a++) {
        const ta = winToks[a];
        for (let b = a + 1; b < winToks.length; b++) {
          const tb = winToks[b];
          const key = ta < tb ? `${ta}|${tb}` : `${tb}|${ta}`;
          cooc.set(key, (cooc.get(key) || 0) + 1);
        }
      }
    }
  }

  const pmiTop = new Map();
  if (totalWin === 0) return pmiTop;

  const adj = new Map();
  for (const [key, c] of cooc.entries()) {
    if (c < 2) continue;
    const sep = key.indexOf('|');
    const a = key.slice(0, sep);
    const b = key.slice(sep + 1);
    const wa = winCount.get(a) || 0;
    const wb = winCount.get(b) || 0;
    if (wa === 0 || wb === 0) continue;
    const pmi = Math.log2((c * totalWin) / (wa * wb));
    if (pmi < 1.5) continue;
    const score = pmi * Math.log2(1 + c);
    const normW = Math.min(1.0, pmi / 6.0);

    let listA = adj.get(a);
    if (!listA) { listA = []; adj.set(a, listA); }
    listA.push({ tok: b, normW, score });

    let listB = adj.get(b);
    if (!listB) { listB = []; adj.set(b, listB); }
    listB.push({ tok: a, normW, score });
  }

  for (const [tok, arr] of adj.entries()) {
    arr.sort((x, y) => y.score - x.score || (x.tok < y.tok ? -1 : x.tok > y.tok ? 1 : 0));
    pmiTop.set(tok, arr.slice(0, 3).map((x) => [x.tok, x.normW]));
  }

  return pmiTop;
}

/**
 * Lazy non-enumerable net._index builder and accessor (v1.7 O(1) Set/P4 + MergedCtx + Lazy PMI).
 */
export function ensureIndex(net, forceRebuild = false) {
  const force = typeof forceRebuild === 'object' && forceRebuild !== null
    ? Boolean(forceRebuild.forceRebuild)
    : Boolean(forceRebuild);
  const buildPmiNow = typeof forceRebuild === 'object' && forceRebuild !== null
    ? Boolean(forceRebuild.buildPmi)
    : false;

  const neurons = net.neurons || {};
  const neuronList = Object.values(neurons);
  const N = neuronList.length;

  if (!force && net._index && net._index.N === N) {
    if (buildPmiNow && !net._index.pmiTopPublic) {
      net._index.pmiTopPublic = buildPmiTopTableSafe({
        groups: net._index.sessionGroups,
        coreTagsById: net._index.coreTagsById,
        extraTagsById: net._index.extraTagsById,
        dfMap: net._index.dfPublic,
        N,
        includePrivate: false,
      });
      net._index.pmiTopAll = buildPmiTopTableSafe({
        groups: net._index.sessionGroups,
        coreTagsById: net._index.coreTagsById,
        extraTagsById: net._index.extraTagsById,
        dfMap: net._index.dfAll,
        N,
        includePrivate: true,
      });
    }
    return net._index;
  }

  const codeMap = new Map();
  const normCodes = new Map();
  const order = new Map();
  const dendEntries = new Map();
  const titleNorms = new Map();
  const titleWordToIds = new Map();
  const titleWordsById = new Map();
  const mass = new Map();
  const hasQuestionById = new Map();
  const sessionById = new Map();
  const authorKeyById = new Map();
  const authorTokSetById = new Map();
  const spkAuthorCounts = new Map();

  let sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  let sessionNodesCount = 0;

  for (let idxPos = 0; idxPos < neuronList.length; idxPos++) {
    const n = neuronList[idxPos];
    order.set(n.id, idxPos);
    dendEntries.set(n.id, Object.entries(n.dendrites || {}));

    const d4 = n.dim4 || { daily: 0.15, exploration: 0.15, execution: 0.15, support: 0.15 };
    sum0 += Number(d4.daily || 0);
    sum1 += Number(d4.exploration || 0);
    sum2 += Number(d4.execution || 0);
    sum3 += Number(d4.support || 0);

    let sigma = 0.85;
    if (n.type === 'lesson' || n.type === 'graph' || n.type === 'knowledge') sigma = 0.65;
    else if (n.type === 'milestone' || n.type === 'profile') sigma = 0.68;
    else if (n.type === 'episodic') sigma = 0.90;
    else if (n.type === 'inbox') sigma = 1.35;
    mass.set(n.id, 1 / (sigma * sigma));

    const nc = norm(n.code || '');
    normCodes.set(n.id, nc);
    if (nc) {
      codeMap.set(nc, n.id);
      const unhyphen = nc.replace(/[^a-z0-9]/g, '');
      if (unhyphen && !codeMap.has(unhyphen)) {
        codeMap.set(unhyphen, n.id);
      }
    }

    const tNorm = norm(n.title || '');
    titleNorms.set(n.id, tNorm);
    const words = tNorm.split(/[^a-z0-9_-]+/).filter((w) => w.length >= 3);
    const seenWords = new Set(words);
    titleWordsById.set(n.id, [...seenWords]);
    for (const w of seenWords) {
      let wArr = titleWordToIds.get(w);
      if (!wArr) { wArr = []; titleWordToIds.set(w, wArr); }
      wArr.push(n.id);
    }

    const sumStr = n.summary || '';
    const titStr = n.title || '';
    hasQuestionById.set(n.id, Boolean(n.hasQuestion) || sumStr.includes('?') || titStr.includes('?'));

    const sk = getSessionKey(n);
    sessionById.set(n.id, sk);
    if (sk) sessionNodesCount++;

    const spkSource = sumStr || titStr;
    const mSpk = spkSource.match(/^\s*(?:\([^)]+\)\s*)?([^:]{1,40}):/);
    const spkToks = mSpk ? tokenize(mSpk[1]) : [];
    authorKeyById.set(n.id, spkToks.join('_'));
    authorTokSetById.set(n.id, new Set(spkToks));
    for (let si = 0; si < spkToks.length; si++) {
      const st = spkToks[si];
      spkAuthorCounts.set(st, (spkAuthorCounts.get(st) || 0) + 1);
    }
  }

  const hasSessions = sessionNodesCount >= Math.max(4, Math.floor(N * 0.2));
  const knownSpeakerToks = new Set();
  for (const [tok, cnt] of spkAuthorCounts.entries()) {
    if (cnt >= 3) knownSpeakerToks.add(tok);
  }

  const coreTagsById = new Map();
  const extraTagsById = new Map();
  const coreSetById = new Map();
  const coreByP4ById = new Map();
  const extraSetById = new Map();
  const extraByP4ById = new Map();
  const dfPublic = new Map();
  const dfAll = new Map();
  const exactTokToIds = new Map();

  for (let i = 0; i < neuronList.length; i++) {
    const n = neuronList[i];
    const origTags = Array.isArray(n.tags) ? n.tags : [];
    let coreTags;
    let extraTags;

    if (hasSessions) {
      const seenCore = new Set();
      coreTags = [];
      for (let ti = 0; ti < origTags.length && coreTags.length < 48; ti++) {
        const w = origTags[ti];
        if (IMAGE_BOILERPLATE.has(w) || seenCore.has(w)) continue;
        seenCore.add(w);
        coreTags.push(w);
      }
      extraTags = [];
    } else if (n.type === 'episodic') {
      coreTags = origTags;
      extraTags = [];
    } else {
      coreTags = origTags.slice(0, 18);
      extraTags = origTags.slice(18, 128);
    }

    coreTagsById.set(n.id, coreTags);
    extraTagsById.set(n.id, extraTags);
    coreSetById.set(n.id, new Set(coreTags));
    coreByP4ById.set(n.id, buildP4Map(coreTags));
    if (extraTags.length > 0) {
      extraSetById.set(n.id, new Set(extraTags));
      extraByP4ById.set(n.id, buildP4Map(extraTags));
    }

    const seenAll = new Set([...coreTags, ...extraTags, ...Object.keys(n.dendrites || {})]);
    for (const tok of seenAll) {
      dfAll.set(tok, (dfAll.get(tok) || 0) + 1);
      if (!n.private) dfPublic.set(tok, (dfPublic.get(tok) || 0) + 1);
      let arr = exactTokToIds.get(tok);
      if (!arr) { arr = []; exactTokToIds.set(tok, arr); }
      arr.push(n.id);
    }
  }

  const prefix4ToToks = new Map();
  for (const tok of exactTokToIds.keys()) {
    if (tok.length >= 4) {
      const p4 = tok.slice(0, 4);
      let arr = prefix4ToToks.get(p4);
      if (!arr) { arr = []; prefix4ToToks.set(p4, arr); }
      arr.push(tok);
    }
  }

  const titlePrefix4ToWords = new Map();
  for (const w of titleWordToIds.keys()) {
    if (w.length >= 4) {
      const p4 = w.slice(0, 4);
      let arr = titlePrefix4ToWords.get(p4);
      if (!arr) { arr = []; titlePrefix4ToWords.set(p4, arr); }
      arr.push(w);
    }
  }

  const meanDim4 = N > 0 ? [sum0 / N, sum1 / N, sum2 / N, sum3 / N] : [0.25, 0.25, 0.25, 0.25];
  const phase4d = new Map();
  for (const n of neuronList) {
    const d4 = n.dim4 || { daily: 0.15, exploration: 0.15, execution: 0.15, support: 0.15 };
    const day = Number(d4.daily || 0);
    const exp = Number(d4.exploration || 0);
    const exe = Number(d4.execution || 0);
    const sup = Number(d4.support || 0);
    const v0 = (exp + exe) - (day + sup);
    const v1 = 0.55 * (exp - exe);
    const v2 = 0.55 * (day - sup);
    const mag = Math.hypot(v0, v1, v2);
    if (mag < 1e-6) phase4d.set(n.id, [0, 0, 0, 1]);
    else phase4d.set(n.id, [v0 / mag, v1 / mag, v2 / mag, 0]);
  }

  const sessionGroups = [];
  if (hasSessions) {
    const bySess = new Map();
    const noSess = [];
    for (let i = 0; i < neuronList.length; i++) {
      const n = neuronList[i];
      const sk = sessionById.get(n.id);
      if (sk) {
        let arr = bySess.get(sk);
        if (!arr) { arr = []; bySess.set(sk, arr); }
        arr.push(n);
      } else {
        noSess.push(n);
      }
    }
    for (const arr of bySess.values()) sessionGroups.push(arr);
    for (let i = 0; i < noSess.length; i++) sessionGroups.push([noSess[i]]);
  } else {
    for (let i = 0; i < neuronList.length; i++) sessionGroups.push([neuronList[i]]);
  }

  const nextNeighbors = new Map();
  const nextPairSet = new Set();
  const lastBySession = new Map();
  for (let i = 0; i < neuronList.length; i++) {
    const n = neuronList[i];
    const key = sessionById.get(n.id);
    if (!key) continue;
    const prev = lastBySession.get(key);
    if (prev && prev.id !== n.id && Boolean(prev.private) === Boolean(n.private)) {
      if (!nextNeighbors.has(prev.id)) nextNeighbors.set(prev.id, []);
      if (!nextNeighbors.has(n.id)) nextNeighbors.set(n.id, []);
      nextNeighbors.get(prev.id).push({ dstId: n.id, dir: 'fwd' });
      nextNeighbors.get(n.id).push({ dstId: prev.id, dir: 'bwd' });
      nextPairSet.add(`${prev.id}|${n.id}`);
      nextPairSet.add(`${n.id}|${prev.id}`);
    }
    lastBySession.set(key, n);
  }

  // U1: Pre-merged prevCtx (t-1, wPrev=0.40) + prevSameSpkCtx (t-2, wSameSpk=0.25) + sameSpkNeighbors
  const mergedCtxMapById = new Map();
  const mergedCtxByP4ById = new Map();
  const mergedCtxTokToIds = new Map();
  const sameSpkNeighbors = new Map();
  const wPrevCfg = U1_DEFAULTS.wPrev;
  const wSameSpkCfg = U1_DEFAULTS.wSameSpk;
  const maxMergedCtxW = Math.max(wPrevCfg, wSameSpkCfg);

  if (hasSessions) {
    const rareToksCache = new Map();
    const getRareToks = (n) => {
      if (rareToksCache.has(n.id)) return rareToksCache.get(n.id);
      const seen = new Set();
      const out = [];
      const core = coreTagsById.get(n.id) || [];
      const extra = extraTagsById.get(n.id) || [];
      const checkAndAdd = (tok) => {
        if (!tok || tok.length < 4 || STOPWORDS.has(tok) || seen.has(tok)) return;
        seen.add(tok);
        const freq = dfPublic.get(tok) || 0;
        if (freq === 0 || (freq / N) > 0.15) return;
        const idf = Math.max(0.4, Math.log10((N + 1) / freq) + 0.5);
        if (idf >= 1.4) out.push(tok);
      };
      for (let k = 0; k < core.length; k++) checkAndAdd(core[k]);
      for (let k = 0; k < extra.length; k++) checkAndAdd(extra[k]);
      rareToksCache.set(n.id, out);
      return out;
    };

    for (let gi = 0; gi < sessionGroups.length; gi++) {
      const seq = sessionGroups[gi];
      if (seq.length < 2) continue;
      const lastByAuthor = new Map();

      for (let i = 0; i < seq.length; i++) {
        const cur = seq[i];
        if (!sessionById.get(cur.id)) continue;
        const ownCoreSet = coreSetById.get(cur.id);
        const ownExtraSet = extraSetById.get(cur.id);
        const mergedMap = new Map();

        if (i > 0) {
          const prevN = seq[i - 1];
          if (Boolean(prevN.private) === Boolean(cur.private)) {
            const pRare = getRareToks(prevN);
            for (let k = 0; k < pRare.length; k++) {
              const t = pRare[k];
              if (ownCoreSet.has(t) || (ownExtraSet && ownExtraSet.has(t))) continue;
              const prevW = mergedMap.get(t) || 0;
              if (wPrevCfg > prevW) mergedMap.set(t, wPrevCfg);
            }
          }
        }

        const aKey = authorKeyById.get(cur.id);
        if (aKey) {
          const prevSame = lastByAuthor.get(aKey);
          if (prevSame && prevSame.id !== cur.id && Boolean(prevSame.private) === Boolean(cur.private)) {
            if (!sameSpkNeighbors.has(prevSame.id)) sameSpkNeighbors.set(prevSame.id, []);
            if (!sameSpkNeighbors.has(cur.id)) sameSpkNeighbors.set(cur.id, []);
            sameSpkNeighbors.get(prevSame.id).push({ dstId: cur.id, dir: 'fwd' });
            sameSpkNeighbors.get(cur.id).push({ dstId: prevSame.id, dir: 'bwd' });

            const sRare = getRareToks(prevSame);
            for (let k = 0; k < sRare.length; k++) {
              const t = sRare[k];
              if (ownCoreSet.has(t) || (ownExtraSet && ownExtraSet.has(t))) continue;
              const prevW = mergedMap.get(t) || 0;
              if (wSameSpkCfg > prevW) mergedMap.set(t, wSameSpkCfg);
            }
          }
          lastByAuthor.set(aKey, cur);
        }

        if (mergedMap.size > 0) {
          mergedCtxMapById.set(cur.id, mergedMap);
          const byP4 = new Map();
          for (const [tok, w] of mergedMap.entries()) {
            let arr = mergedCtxTokToIds.get(tok);
            if (!arr) { arr = []; mergedCtxTokToIds.set(tok, arr); }
            arr.push(cur.id);
            if (tok.length >= 4) {
              const p4 = tok.slice(0, 4);
              let pArr = byP4.get(p4);
              if (!pArr) { pArr = []; byP4.set(p4, pArr); }
              pArr.push([tok, w]);
            }
          }
          if (byP4.size > 0) mergedCtxByP4ById.set(cur.id, byP4);
        }
      }
    }
  }

  let pmiTopPublic = null;
  let pmiTopAll = null;
  if (buildPmiNow) {
    pmiTopPublic = buildPmiTopTableSafe({
      groups: sessionGroups, coreTagsById, extraTagsById, dfMap: dfPublic, N, includePrivate: false,
    });
    pmiTopAll = buildPmiTopTableSafe({
      groups: sessionGroups, coreTagsById, extraTagsById, dfMap: dfAll, N, includePrivate: true,
    });
  }

  const idx = {
    N,
    hasSessions,
    dfPublic,
    dfAll,
    exactTokToIds,
    prefix4ToToks,
    codeMap,
    normCodes,
    order,
    dendEntries,
    titleNorms,
    titleWordToIds,
    titleWordsById,
    titlePrefix4ToWords,
    meanDim4,
    phase4d,
    mass,
    coreTagsById,
    extraTagsById,
    coreSetById,
    coreByP4ById,
    extraSetById,
    extraByP4ById,
    hasQuestionById,
    sessionById,
    authorTokSetById,
    knownSpeakerToks,
    sessionGroups,
    nextNeighbors,
    nextPairSet,
    mergedCtxMapById,
    mergedCtxByP4ById,
    mergedCtxTokToIds,
    maxMergedCtxW,
    sameSpkNeighbors,
    pmiTopPublic,
    pmiTopAll,
  };

  Object.defineProperty(net, '_index', {
    value: idx,
    writable: true,
    configurable: true,
    enumerable: false,
  });

  return idx;
}

export function compileNetwork(memoryDir, options = {}) {
  const t0 = performance.now();
  const prevNet = options.fresh ? null : loadNetwork(memoryDir);
  const map = parseMemoryDirectory(memoryDir, options);
  weaveEdgesAndDendrites(map, prevNet, Boolean(options.decay), Boolean(options.disablePrivacyFirewall), options);

  const neuronsObj = Object.fromEntries(map.entries());
  let totalEdges = 0;
  for (const n of map.values()) totalEdges += Object.keys(n.edges).length;

  const net = {
    version: VERSION,
    architecture: 'HebbMesh 4D/Zero-Inference Synaptic Memory + Pauli/CMI/PMI + O(1) SQLite WAL',
    updatedAt: new Date().toISOString(),
    metrics: {
      totalNeurons: map.size,
      totalEdges,
      compileMs: round2(performance.now() - t0),
    },
    neurons: neuronsObj,
  };

  ensureIndex(net, { forceRebuild: true, buildPmi: Boolean(options.buildPmi) });

  if (!options.noPersist) saveNetwork(memoryDir, net);
  return net;
}

export function loadOrCompile(memoryDir, options = {}) {
  if (options.forceCompile) return compileNetwork(memoryDir, options);
  const existing = loadNetwork(memoryDir);
  if (!existing || existing.version !== VERSION) return compileNetwork(memoryDir, options);
  const netMtime = getStorageMtime(memoryDir);
  for (const f of collectMarkdownSources(memoryDir)) {
    try {
      if (fs.statSync(f).mtimeMs > netMtime + 300) return compileNetwork(memoryDir, options);
    } catch { }
  }
  return existing;
}

function tokenMatchScore(qTok, targetTok) {
  if (qTok === targetTok) return 1.0;
  if (qTok.length >= 4 && targetTok.length >= 4) {
    if (targetTok.startsWith(qTok) || qTok.startsWith(targetTok)) return 0.78;
    const stemLen = Math.min(5, Math.min(qTok.length, targetTok.length));
    if (stemLen >= 4 && qTok.slice(0, stemLen) === targetTok.slice(0, stemLen)) return 0.62;
  }
  return 0;
}

/**
 * 3D A* Synaptic Bridge between two disjoint concepts with Anti-Pareidolia & Privacy Firewall.
 */
export function findBridge(net, queryA, queryB, options = {}) {
  const t0 = performance.now();
  const allowPrivate = Boolean(options.allowPrivate) || Boolean(options.disablePrivacyFirewall);
  const minWeight = options.minWeight ?? 0.55;
  const maxHops = options.maxHops ?? 5;
  const hubPenalty = options.hubPenalty ?? 0.08;

  const pickBestNode = (q) => {
    const res = activateNetworkInMemory(net, q, { topK: 1, reinforce: false, allowPrivate });
    return res.winners[0]?.id || null;
  };

  const resolveSafeId = (q) => {
    const direct = net.neurons[q];
    if (direct) return (direct.private && !allowPrivate) ? null : q;
    return pickBestNode(q);
  };
  const idA = resolveSafeId(queryA);
  const idB = resolveSafeId(queryB);
  if (!idA || !idB) return { found: false, ms: round2(performance.now() - t0), reason: 'endpoint_not_found' };
  if (idA === idB) {
    const n = net.neurons[idA];
    return { found: true, ms: round2(performance.now() - t0), confidence: 1, path: [{ id: idA, code: n.code, title: n.title, edgeWeight: null }] };
  }

  const adj = new Map();
  for (const [id, n] of Object.entries(net.neurons)) {
    if (n.private && !allowPrivate) continue;
    if (!adj.has(id)) adj.set(id, new Map());
    for (const [v, w] of Object.entries(n.edges || {})) {
      const nv = net.neurons[v];
      if (!nv || (nv.private && !allowPrivate)) continue;
      if (w >= minWeight) {
        adj.get(id).set(v, Math.max(adj.get(id).get(v) || 0, w));
        if (!adj.has(v)) adj.set(v, new Map());
        adj.get(v).set(id, Math.max(adj.get(v).get(id) || 0, w));
      }
    }
  }

  const dist3d = (uId, vId) => {
    const a = net.neurons[uId]?.coord3d || { x: 0, y: 0, z: 0 };
    const b = net.neurons[vId]?.coord3d || { x: 0, y: 0, z: 0 };
    return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0), (a.z || 0) - (b.z || 0));
  };

  const gScore = new Map([[idA, 0]]);
  const hops = new Map([[idA, 0]]);
  const cameFrom = new Map();
  const edgeUsed = new Map();
  const closed = new Set();
  const open = [{ id: idA, f: dist3d(idA, idB) * 0.05 }];

  while (open.length) {
    open.sort((a, b) => a.f - b.f);
    const cur = open.shift().id;
    if (cur === idB) {
      const pathOut = [];
      let p = idB;
      while (p) {
        const n = net.neurons[p];
        pathOut.unshift({ id: p, code: n.code, title: n.title, edgeWeight: edgeUsed.get(p) ?? null });
        p = cameFrom.get(p);
      }
      const totalCost = gScore.get(idB) || 0;
      const confidence = round2(Math.max(0.15, Math.min(1, 1 - totalCost / 3.5)));
      return { found: true, ms: round2(performance.now() - t0), confidence, path: pathOut };
    }
    closed.add(cur);
    const curHops = hops.get(cur) || 0;
    if (curHops >= maxHops) continue;

    for (const [v, w] of (adj.get(cur) || new Map()).entries()) {
      if (closed.has(v)) continue;
      const deg = (adj.get(v)?.size || 0);
      const toll = v !== idB && deg >= 7 ? (deg - 6) * hubPenalty : 0;
      const stepCost = (1.15 - w) + toll;
      const gCand = (gScore.get(cur) || 0) + stepCost;
      if (gCand < (gScore.get(v) ?? Infinity)) {
        gScore.set(v, gCand);
        hops.set(v, curHops + 1);
        cameFrom.set(v, cur);
        edgeUsed.set(v, round2(w));
        open.push({ id: v, f: gCand + dist3d(v, idB) * 0.05 });
      }
    }
  }

  return { found: false, ms: round2(performance.now() - t0), reason: 'no_strong_bridge' };
}

function computeLogicalVsAffectivePole(seeds, neurons) {
  let wSum = 0;
  let poleSum = 0;
  for (const [id, e] of seeds) {
    const d = neurons[id]?.dim4;
    if (!d) continue;
    const logical = Math.max(d.exploration || 0, d.execution || 0);
    const affective = Math.max(d.support || 0, (d.daily || 0) * 0.85);
    poleSum += (logical - affective) * e;
    wSum += e;
  }
  return wSum > 0 ? poleSum / wSum : 0;
}

function interLobeFirewall(seedPole, dstDim4) {
  if (!dstDim4) return 1.0;
  const logical = Math.max(dstDim4.exploration || 0, dstDim4.execution || 0);
  const affective = Math.max(dstDim4.support || 0, (dstDim4.daily || 0) * 0.85);
  const dstPole = logical - affective;
  if ((seedPole >= 0.30 && dstPole <= -0.25) || (seedPole <= -0.30 && dstPole >= 0.25)) {
    return 0.55;
  }
  return 1.0;
}

function computeCmiMultiplier(n, cmi) {
  const validMult = n.valid ? 1.0 : (cmi?.validFalseMult ?? 0.45);
  const weightMult = 0.72 + 0.38 * Number(n.weight || 0.65);
  let cmiMult = 1.0;
  if (cmi) {
    const isEphemeral = n.type === 'inbox' || n.type === 'episodic';
    const isVerifiedCore = n.valid && (n.type === 'lesson' || n.type === 'graph' || n.type === 'knowledge') && Number(n.weight || 0) >= 0.78;
    if (cmi.recencyWeight) {
      const yVal = clamp01(n.coord3d?.y ?? 0.90);
      cmiMult *= (1 - cmi.recencyWeight) + cmi.recencyWeight * yVal;
    }
    const stuffTarget = cmi.stuffInboxOnly ? (n.type === 'inbox') : isEphemeral;
    if (cmi.antiStuffing && stuffTarget) {
      const tagCount = (n.tags || []).length;
      if (tagCount > (cmi.stuffTagThreshold ?? 15)) {
        cmiMult *= Math.max(cmi.stuffFloor ?? 0.85, (cmi.stuffTagThreshold ?? 15) / tagCount);
      }
    }
    if (cmi.authority) {
      if (isEphemeral) cmiMult *= (cmi.ephemeralFactor ?? 0.80);
      else if (isVerifiedCore) cmiMult *= (cmi.verifiedFactor ?? 1.14);
    }
  }
  return validMult * weightMult * cmiMult;
}

/**
 * M5: Turrigiano Homeostatic Reinforcement & Synaptic Scaling.
 */
export function applyReinforcement(net, winners, options = {}) {
  const useHomeostasis = options.useHomeostasis !== false;
  const wMax = options.wMax ?? 5.2;
  const hebbianLinks = [];
  const edgeDeltas = [];
  const actDeltas = [];
  let homeostasisRescaled = 0;

  if (!winners || winners.length < 2) {
    return { hebbianLinks, homeostasisRescaled, edgeDeltas, actDeltas };
  }

  if (useHomeostasis) {
    const w0E = winners[0].energy;
    const passesConfidence = (w0E >= 1.5) || (winners.length < 4 ? w0E >= 1.0 : (w0E - winners[3].energy) >= 0.35);
    if (!passesConfidence) {
      return { hebbianLinks, homeostasisRescaled, edgeDeltas, actDeltas };
    }
  }

  const top = winners.slice(0, 3);
  for (const w of top) {
    const n = net.neurons[w.id];
    if (!n) continue;
    n.activations = Number(n.activations || 0) + 1;
    actDeltas.push({ id: n.id, activations: n.activations });
  }

  for (let i = 0; i < top.length; i++) {
    for (let j = i + 1; j < top.length; j++) {
      const a = net.neurons[top[i].id];
      const b = net.neurons[top[j].id];
      if (!a || !b) continue;
      const nextW = round2(Math.min(1.0, Number(a.edges[b.id] || 0.35) + 0.04));
      a.edges[b.id] = nextW;
      b.edges[a.id] = nextW;
      edgeDeltas.push({ source: a.id, target: b.id, weight: nextW });
      edgeDeltas.push({ source: b.id, target: a.id, weight: nextW });
      hebbianLinks.push(`${a.code}<->${b.code}:${nextW}`);
    }
  }

  if (useHomeostasis) {
    for (const w of top) {
      const u = net.neurons[w.id];
      if (!u || !u.edges) continue;
      const sumW = Object.values(u.edges).reduce((s, v) => s + Number(v), 0);
      if (sumW > wMax) {
        const scale = wMax / sumW;
        let didRescale = false;
        for (const [k, wVal] of Object.entries(u.edges)) {
          const rescaledW = round2(Math.max(0.10, Number(wVal) * scale));
          u.edges[k] = rescaledW;
          edgeDeltas.push({ source: u.id, target: k, weight: rescaledW });
          didRescale = true;
        }
        if (didRescale) homeostasisRescaled++;
      }
    }
  }

  return { hebbianLinks, homeostasisRescaled, edgeDeltas, actDeltas };
}

export function activateNetworkInMemory(net, query = '', options = {}) {
  const t0 = performance.now();
  const topK = options.topK ?? 6;
  const maxHops = options.disablePropagation ? 0 : (options.maxHops ?? 2);
  const allowPrivate = Boolean(options.allowPrivate) || Boolean(options.disablePrivacyFirewall);
  const useQuantumPhase = options.useQuantumPhase !== false;
  const useSynapticMass = options.useSynapticMass !== false;

  const extraWeight = options.extraWeight ?? 0.25;
  const useF4 = options.useF4 !== false;
  const u4 = options.u4 !== false;
  // Default false matches V17_Opt_U1234 exact parity; set { useWordBoundary: true } for strict S1_WB title word-boundary mode.
  const useWordBoundary = Boolean(options.useWordBoundary);
  const cmi = options.cmi === false
    ? null
    : { ...CMI_ZERO_REG_DEFAULTS, ...(typeof options.cmi === 'object' ? options.cmi : {}) };
  const pmiSafe = options.pmiSafe === false
    ? null
    : { ...PMI_SAFE_DEFAULTS, ...(typeof options.pmiSafe === 'object' ? options.pmiSafe : {}) };
  const u1 = options.u1 === false
    ? null
    : { ...U1_DEFAULTS, ...(typeof options.u1 === 'object' ? options.u1 : {}) };
  const u2 = options.u2 === false
    ? null
    : { ...U2_DEFAULTS, ...(typeof options.u2 === 'object' ? options.u2 : {}) };
  const pauli = options.pauli === false
    ? null
    : { ...PAULI_DEFAULTS, ...(typeof options.pauli === 'object' ? options.pauli : {}) };

  const neurons = net.neurons || {};
  const idx = ensureIndex(net);
  const N = idx.N;
  const df = allowPrivate ? idx.dfAll : idx.dfPublic;

  const rawQTokens = tokenize(query);
  const qTokens = u4
    ? (() => {
        const filtered = rawQTokens.filter((t) => !EXTRA_STD_STOP.has(t));
        return filtered.length > 0 ? filtered : rawQTokens;
      })()
    : rawQTokens;
  const qNorm = norm(query);

  const qTokIdf = new Map();
  const qInfoToks = [];
  let totalQInfoIdf = 0;
  const minInfoIdf = pauli?.minInfoIdf ?? 1.10;
  for (let qi = 0; qi < qTokens.length; qi++) {
    const qt = qTokens[qi];
    if (qTokIdf.has(qt)) continue;
    const freq = df.get(qt) || 2;
    const idf = Math.max(0.4, Math.log10((N + 1) / freq) + 0.5);
    qTokIdf.set(qt, idf);
    if (idf >= minInfoIdf && qt.length >= 3) {
      qInfoToks.push(qt);
      totalQInfoIdf += idf;
    }
  }

  const energy = new Map();
  const trace = new Map();
  const tokContribById = pauli ? new Map() : null;

  const inertiaMod = (id) => {
    const n = neurons[id];
    let m = idx.mass.get(id) ?? 1.384;
    if (cmi?.resetObsoleteMass && n && !n.valid) m = 1.0;
    return Math.max(0.82, Math.min(1.18, Math.pow(m, 0.18)));
  };

  // Candidate gathering
  const candIds = new Set();
  const hyphens = qNorm.match(/[a-z0-9]+(?:-[a-z0-9]+)+/g) || [];
  for (const h of hyphens) {
    const id = idx.codeMap.get(h);
    if (id) {
      const c = neurons[id]?.code;
      if (c && qNorm.includes(norm(c))) candIds.add(id);
    }
  }
  for (const w of qNorm.split(/\s+/)) {
    const id = idx.codeMap.get(w);
    if (id) {
      const c = neurons[id]?.code;
      if (c && qNorm.includes(norm(c))) candIds.add(id);
    }
  }

  const ctxActive = Boolean(idx.hasSessions && idx.maxMergedCtxW > 0 && u1);
  const u1Active = Boolean(u1 && idx.hasSessions);
  const u2Active = Boolean(u2 && idx.hasSessions && idx.knownSpeakerToks?.size > 0);
  const mergedCtxTokToIds = ctxActive ? idx.mergedCtxTokToIds : null;

  for (let qi = 0; qi < qTokens.length; qi++) {
    const qt = qTokens[qi];
    const exactIds = idx.exactTokToIds.get(qt);
    if (exactIds) {
      for (let i = 0; i < exactIds.length; i++) candIds.add(exactIds[i]);
    }
    if (ctxActive) {
      const mIds = mergedCtxTokToIds.get(qt);
      if (mIds) for (let i = 0; i < mIds.length; i++) candIds.add(mIds[i]);
    }
    if (qt.length >= 4) {
      const p4Toks = idx.prefix4ToToks.get(qt.slice(0, 4));
      if (p4Toks) {
        for (let i = 0; i < p4Toks.length; i++) {
          const tok = p4Toks[i];
          if (tokenMatchScore(qt, tok) > 0) {
            const tokIds = idx.exactTokToIds.get(tok);
            if (tokIds) for (let j = 0; j < tokIds.length; j++) candIds.add(tokIds[j]);
            if (ctxActive) {
              const mIds = mergedCtxTokToIds.get(tok);
              if (mIds) for (let j = 0; j < mIds.length; j++) candIds.add(mIds[j]);
            }
          }
        }
      }
    }
    const twIds = idx.titleWordToIds.get(qt);
    if (twIds) {
      for (let i = 0; i < twIds.length; i++) candIds.add(twIds[i]);
    }
    if (qt.length >= 4) {
      const p4Words = idx.titlePrefix4ToWords.get(qt.slice(0, 4));
      if (p4Words) {
        for (let i = 0; i < p4Words.length; i++) {
          const w = p4Words[i];
          if (w.startsWith(qt) || (useWordBoundary && qt.startsWith(w))) {
            const wIds = idx.titleWordToIds.get(w);
            if (wIds) for (let j = 0; j < wIds.length; j++) candIds.add(wIds[j]);
          }
        }
      }
    }
  }

  // PMI_Safe lazy lookup & expansion (separate map so pos=0 is shielded)
  let pmiQBoost = null;
  let pmiQBoostEntries = null;
  if (pmiSafe) {
    let pmiMap = allowPrivate ? idx.pmiTopAll : idx.pmiTopPublic;
    if (!pmiMap) {
      pmiMap = buildPmiTopTableSafe({
        groups: idx.sessionGroups,
        coreTagsById: idx.coreTagsById,
        extraTagsById: idx.extraTagsById,
        dfMap: allowPrivate ? idx.dfAll : idx.dfPublic,
        N,
        includePrivate: allowPrivate,
      });
      if (allowPrivate) idx.pmiTopAll = pmiMap;
      else idx.pmiTopPublic = pmiMap;
    }

    if (pmiMap && pmiMap.size > 0) {
      const qTokSet = new Set(qTokens);
      const pmiWeight = pmiSafe.weight ?? 0.22;
      const maxAssoc = pmiSafe.maxAssoc ?? 3;
      const minQIdf = pmiSafe.minQIdf ?? 1.5;
      const maxDfRatio = pmiSafe.maxDfRatio ?? 0.12;
      pmiQBoost = new Map();

      for (const qt of qTokSet) {
        if (qt.length < 4) continue;
        if (N > 0 && ((df.get(qt) || 0) / N) > maxDfRatio) continue;
        const qIdf = qTokIdf.get(qt) || 0;
        if (qIdf < minQIdf) continue;

        const assocs = pmiMap.get(qt);
        if (!assocs || !assocs.length) continue;

        let added = 0;
        for (let ai = 0; ai < assocs.length && added < maxAssoc; ai++) {
          const [assocTok, normW] = assocs[ai];
          if (qTokSet.has(assocTok)) continue;
          added++;
          const boostVal = pmiWeight * qIdf * normW;
          const prev = pmiQBoost.get(assocTok) || 0;
          if (boostVal > prev) pmiQBoost.set(assocTok, boostVal);

          const assocIds = idx.exactTokToIds.get(assocTok);
          if (assocIds) {
            for (let j = 0; j < assocIds.length; j++) candIds.add(assocIds[j]);
          }
        }
      }

      if (pmiQBoost.size === 0) pmiQBoost = null;
      else pmiQBoostEntries = [...pmiQBoost.entries()];
    }
  }

  // 1. Direct Sparking
  const pmiScoreById = pmiQBoostEntries ? new Map() : null;
  const capPmiNode = pmiSafe?.capNode ?? 1.6;
  const u2MentionMult = u2?.mentionMult ?? 0.60;
  const maxCtxW = ctxActive ? idx.maxMergedCtxW : 0;

  const qLen = qTokens.length;
  const qP4 = new Array(qLen);
  const qIdfs = new Array(qLen);
  const qIsKnownSpk = u2Active ? new Array(qLen) : null;
  for (let qi = 0; qi < qLen; qi++) {
    const qt = qTokens[qi];
    qP4[qi] = qt.length >= 4 ? qt.slice(0, 4) : '';
    qIdfs[qi] = qTokIdf.get(qt) ?? 0.4;
    if (u2Active) qIsKnownSpk[qi] = idx.knownSpeakerToks.has(qt);
  }

  for (const id of candIds) {
    const n = neurons[id];
    if (!n || (n.private && !allowPrivate)) continue;
    let score = 0;
    const matched = [];
    const contribMap = tokContribById ? new Map() : null;

    const normCode = idx.normCodes.get(id) || '';
    if (normCode && qNorm.includes(normCode)) {
      score += 3.2;
      matched.push(n.code);
    }

    const titleNorm = idx.titleNorms.get(id) || '';
    const titleToks = idx.titleWordsById.get(id) || [];
    const coreSet = idx.coreSetById.get(id);
    const coreByP4 = idx.coreByP4ById.get(id);
    const extraSet = idx.extraSetById.get(id);
    const extraByP4 = idx.extraByP4ById.get(id);
    const nDend = idx.dendEntries.get(id) || [];
    const mergedCtxMap = ctxActive ? idx.mergedCtxMapById.get(id) : null;
    const mergedCtxByP4 = ctxActive ? idx.mergedCtxByP4ById.get(id) : null;
    const authorTokSet = u2Active ? idx.authorTokSetById.get(id) : null;

    for (let qi = 0; qi < qLen; qi++) {
      const qt = qTokens[qi];
      const p4 = qP4[qi];
      let bestTok = 0;
      let isDend = false;

      if (coreSet.has(qt)) {
        bestTok = 1.0;
      } else if (p4) {
        const arr = coreByP4.get(p4);
        if (arr) {
          for (let ti = 0; ti < arr.length; ti++) {
            const m = tokenMatchScore(qt, arr[ti]);
            if (m > bestTok) bestTok = m;
          }
        }
      }

      if (extraSet && bestTok < extraWeight) {
        if (extraSet.has(qt)) {
          if (extraWeight > bestTok) bestTok = extraWeight;
        } else if (p4) {
          const arr = extraByP4.get(p4);
          if (arr) {
            for (let ti = 0; ti < arr.length; ti++) {
              const m = tokenMatchScore(qt, arr[ti]) * extraWeight;
              if (m > bestTok) bestTok = m;
            }
          }
        }
      }

      if (nDend.length > 0) {
        for (let di = 0; di < nDend.length; di++) {
          const m = tokenMatchScore(qt, nDend[di][0]) * nDend[di][1];
          if (m > bestTok) {
            bestTok = m;
            isDend = true;
          }
        }
      }

      if (mergedCtxMap && bestTok < maxCtxW) {
        const exactW = mergedCtxMap.get(qt);
        if (exactW !== undefined && exactW > bestTok) bestTok = exactW;
        if (p4 && bestTok < maxCtxW && mergedCtxByP4) {
          const arr = mergedCtxByP4.get(p4);
          if (arr) {
            for (let ci = 0; ci < arr.length; ci++) {
              const cw = arr[ci][1];
              if (cw <= bestTok) continue;
              const m = tokenMatchScore(qt, arr[ci][0]) * cw;
              if (m > bestTok) bestTok = m;
            }
          }
        }
      }

      const spkAuthMod = (u2Active && qIsKnownSpk[qi] && !authorTokSet?.has(qt))
        ? u2MentionMult
        : 1.0;

      if (bestTok > 0) {
        const delta = bestTok * qIdfs[qi] * (isDend ? 1.12 : 1.0) * spkAuthMod;
        score += delta;
        matched.push(isDend ? `~${qt}` : qt);
        if (contribMap) contribMap.set(qt, (contribMap.get(qt) || 0) + delta);
      } else {
        let titleHit = false;
        if (useWordBoundary) {
          for (let ti = 0; ti < titleToks.length; ti++) {
            const tt = titleToks[ti];
            if (tt === qt || (qt.length >= 4 && tt.length >= 4 && (tt.startsWith(qt) || qt.startsWith(tt)))) {
              titleHit = true;
              break;
            }
          }
        } else if (qt.length >= 3 && titleNorm.includes(qt)) {
          titleHit = true;
        }
        if (titleHit) {
          const titleDelta = 0.55 * spkAuthMod;
          score += titleDelta;
          matched.push(qt);
          if (contribMap) contribMap.set(qt, (contribMap.get(qt) || 0) + titleDelta);
        }
      }
    }

    let pmiDelta = 0;
    if (pmiQBoostEntries) {
      for (let pi = 0; pi < pmiQBoostEntries.length; pi++) {
        const [assocTok, b] = pmiQBoostEntries[pi];
        if (coreSet.has(assocTok)) {
          pmiDelta += b;
        } else if (extraSet && extraSet.has(assocTok)) {
          pmiDelta += b * extraWeight;
        }
      }
    }

    if (score > 0 || pmiDelta > 0) {
      const fullMult = computeCmiMultiplier(n, cmi);
      if (score > 0) {
        energy.set(id, score * fullMult);
        trace.set(id, `spark (${[...new Set(matched)].slice(0, 4).join(', ')})`);
        if (tokContribById && contribMap && contribMap.size > 0) {
          tokContribById.set(id, contribMap);
        }
      }
      if (pmiDelta > 0 && pmiScoreById) {
        pmiScoreById.set(id, Math.min(capPmiNode, pmiDelta) * fullMult);
      }
    }
  }

  // 2. Spreading Activation (M1 + M2 + M4 + M6a + F4 + U1)
  const direct = new Map(energy);
  const maxDirect = direct.size ? Math.max(...direct.values()) : 0;

  if (maxHops > 0 && maxDirect > 0) {
    const useM2Gating = options.useM2Gating !== false;
    const useM3KWta = options.useM3KWta !== false;
    const useM4Firewall = options.useM4Firewall !== false;
    const useGabaSoft = Boolean(options.useGabaSoft);

    const topEmit = useM3KWta ? (options.topEmit ?? 4) : (options.topEmit ?? 12);
    const relWta = useM3KWta ? (options.relWta ?? 0.45) : 0;
    const outFan = options.outFan ?? 3;
    const minEdgeW = options.minEdgeW ?? 0.58;
    const kRef = options.kRef ?? 1.8;
    const damping = options.damping ?? 0.55;

    if (!useM2Gating) {
      const fired = new Set();
      for (let hop = 0; hop < maxHops; hop++) {
        const curMax = energy.size ? Math.max(...energy.values()) : 0;
        const snapshot = [...energy.entries()]
          .filter(([id, e]) => !fired.has(id) && e >= relWta * curMax)
          .sort((a, b) => b[1] - a[1])
          .slice(0, topEmit);
        if (!snapshot.length) break;

        const hopPole = useM4Firewall ? computeLogicalVsAffectivePole(snapshot, neurons) : 0;
        for (const [srcId, srcE] of snapshot) {
          fired.add(srcId);
          const nSrc = neurons[srcId];
          if (!nSrc) continue;
          const obsoleteEmitMod = (cmi?.brakeObsoleteProp && !nSrc.valid) ? (cmi.obsoleteEmitFactor ?? 0.35) : 1.0;

          const topEdges = Object.entries(nSrc.edges || {})
            .map(([dstId, w]) => [dstId, Number(w)])
            .filter(([dstId, w]) => {
              if (w < minEdgeW || dstId === srcId || fired.has(dstId)) return false;
              const nDst = neurons[dstId];
              return nDst && (!nDst.private || allowPrivate);
            })
            .sort((a, b) => b[1] - a[1])
            .slice(0, outFan);

          if (!topEdges.length) continue;
          const sumTopM = topEdges.reduce((s, [, w]) => s + w, 0);
          const denom = Math.max(kRef, sumTopM);

          for (const [dstId, edgeW] of topEdges) {
            const nDst = neurons[dstId];
            const lobeMod = useM4Firewall ? interLobeFirewall(hopPole, nDst.dim4) : 1.0;
            const massFactor = useSynapticMass ? inertiaMod(srcId) : 1.0;
            const dstValidMod = nDst.valid ? 1.0 : (cmi?.brakeObsoleteProp ? (cmi.obsoleteDstFactor ?? 0.20) : 0.45);
            const imp = srcE * (edgeW / denom) * damping * lobeMod * dstValidMod * massFactor * obsoleteEmitMod;
            if (imp < 0.12) continue;
            energy.set(dstId, (energy.get(dstId) || 0) + imp);
            if (!trace.has(dstId)) {
              trace.set(dstId, `hop${hop + 1} <- ${nSrc.code} (${edgeW})`);
            }
          }
        }
      }
    } else {
      const seeds = [...direct.entries()]
        .filter(([, e]) => e >= relWta * maxDirect)
        .sort((a, b) => b[1] - a[1] || ((idx.order.get(a[0]) ?? 0) - (idx.order.get(b[0]) ?? 0)))
        .slice(0, topEmit);

      const seedPole = useM4Firewall ? computeLogicalVsAffectivePole(seeds, neurons) : 0;

      const propSum = new Map();
      const parentsOf = new Map();
      const maxParentE = new Map();
      const impulsesOf = new Map();
      const seedIds = new Set(seeds.map(([id]) => id));

      for (const [srcId, srcE] of seeds) {
        const nSrc = neurons[srcId];
        if (!nSrc) continue;
        const obsoleteEmitMod = (cmi?.brakeObsoleteProp && !nSrc.valid) ? (cmi.obsoleteEmitFactor ?? 0.35) : 1.0;

        const topEdges = Object.entries(nSrc.edges || {})
          .map(([dstId, w]) => [dstId, Number(w)])
          .filter(([dstId, w]) => {
            if (w < minEdgeW || dstId === srcId) return false;
            const nDst = neurons[dstId];
            return nDst && (!nDst.private || allowPrivate);
          })
          .sort((a, b) => b[1] - a[1])
          .slice(0, outFan);

        if (!topEdges.length) continue;
        const sumTopM = topEdges.reduce((s, [, w]) => s + w, 0);
        const denom = Math.max(kRef, sumTopM);

        for (const [dstId, edgeW] of topEdges) {
          const nDst = neurons[dstId];
          const lobeMod = useM4Firewall ? interLobeFirewall(seedPole, nDst.dim4) : 1.0;
          const massFactor = useSynapticMass ? inertiaMod(srcId) : 1.0;
          const dstValidMod = nDst.valid ? 1.0 : (cmi?.brakeObsoleteProp ? (cmi.obsoleteDstFactor ?? 0.20) : 0.45);
          const imp = srcE * (edgeW / denom) * damping * lobeMod * dstValidMod * massFactor * obsoleteEmitMod;
          if (imp < 0.12) continue;

          propSum.set(dstId, (propSum.get(dstId) || 0) + imp);
          if (!parentsOf.has(dstId)) parentsOf.set(dstId, new Set());
          parentsOf.get(dstId).add(srcId);
          maxParentE.set(dstId, Math.max(maxParentE.get(dstId) || 0, srcE));

          if (!impulsesOf.has(dstId)) impulsesOf.set(dstId, []);
          impulsesOf.get(dstId).push({ srcId, imp });

          if (!trace.has(dstId)) {
            trace.set(dstId, `hop1 <- ${nSrc.code} (${edgeW})`);
          }
        }
      }

      for (const [dstId, rawProp] of propSum.entries()) {
        const dE = direct.get(dstId) || 0;
        const kParents = parentsOf.get(dstId)?.size || 0;
        const pMax = maxParentE.get(dstId) || 0;
        const nDst = neurons[dstId];
        const dstDeg = Object.keys(nDst.edges || {}).length;
        const gabaBrake = useGabaSoft && dstDeg > 4 ? 1 / (1 + 0.08 * (dstDeg - 4)) : 1.0;

        if (useQuantumPhase) {
          const imps = impulsesOf.get(dstId) || [];
          const psi = [0, 0, 0, 0];
          for (let i = 0; i < imps.length; i++) {
            const { srcId, imp } = imps[i];
            const amp = Math.sqrt(imp);
            const phi = idx.phase4d.get(srcId) || [0.5, 0.5, 0.5, 0.5];
            psi[0] += amp * phi[0];
            psi[1] += amp * phi[1];
            psi[2] += amp * phi[2];
            psi[3] += amp * phi[3];
          }
          const eQuantum = psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2] + psi[3] * psi[3];
          const coherence = rawProp > 0 ? eQuantum / rawProp : 0;

          if (dE > 0) {
            const effProp = Math.min(rawProp, eQuantum) * gabaBrake;
            let boosted = dE + 0.45 * effProp;
            if (kParents === 1 && !seedIds.has(dstId)) {
              boosted = Math.min(boosted, pMax - 0.01);
            }
            energy.set(dstId, boosted);
          } else if (kParents >= 2 && coherence >= 0.70) {
            const cohScale = coherence < 0.85 ? 0.5 : 1.0;
            const effProp = eQuantum * cohScale * gabaBrake;
            const nmdaGain = (nDst.type === 'episodic' ? 1.15 : 0.95) * (1 + 0.35 * Math.log(kParents));
            const nmdaE = Math.min(effProp * nmdaGain, 0.92 * pMax);
            energy.set(dstId, nmdaE);
            trace.set(dstId, `q-nmda(k=${kParents}, coh=${round2(coherence)}) <- ${[...parentsOf.get(dstId)].map((id) => neurons[id].code).join('+')}`);
          }
        } else {
          const effProp = rawProp * gabaBrake;
          if (dE > 0) {
            let boosted = dE + 0.45 * effProp;
            if (kParents === 1 && !seedIds.has(dstId)) {
              boosted = Math.min(boosted, pMax - 0.01);
            }
            energy.set(dstId, boosted);
          } else if (kParents >= 2) {
            const nmdaGain = (nDst.type === 'episodic' ? 1.15 : 0.95) * (1 + 0.35 * Math.log(kParents));
            const nmdaE = Math.min(effProp * nmdaGain, 0.92 * pMax);
            energy.set(dstId, nmdaE);
            trace.set(dstId, `nmda(${kParents}x) <- ${[...parentsOf.get(dstId)].map((id) => neurons[id].code).join('+')}`);
          }
        }
      }

      // F4: Working Memory NEXT (1-hop) + U1 SameSpk T2
      if (useF4 && (idx.nextNeighbors.size > 0 || (u1Active && idx.sameSpkNeighbors.size > 0))) {
        const anchors = seeds
          .filter(([, e]) => e >= WM_NEXT_CFG.minAnchorRel * maxDirect)
          .slice(0, WM_NEXT_CFG.maxAnchorSeeds);

        for (let ai = 0; ai < anchors.length; ai++) {
          const [srcId, srcE] = anchors[ai];
          const nbs = idx.nextNeighbors.get(srcId);
          if (nbs && nbs.length) {
            const srcHasQ = Boolean(idx.hasQuestionById.get(srcId));
            const isTop1 = ai === 0;

            for (let ni = 0; ni < nbs.length; ni++) {
              const { dstId, dir } = nbs[ni];
              const nDst = neurons[dstId];
              if (!nDst || (nDst.private && !allowPrivate)) continue;

              const dstHasQ = Boolean(idx.hasQuestionById.get(dstId));
              let wmMult;
              if (dir === 'fwd') {
                wmMult = (srcHasQ || isTop1) ? WM_NEXT_CFG.fwdQuestionWeight : WM_NEXT_CFG.fwdWeight;
              } else {
                wmMult = dstHasQ ? WM_NEXT_CFG.bwdQuestionWeight : WM_NEXT_CFG.bwdWeight;
              }
              const wmE = srcE * wmMult * (nDst.valid ? 1.0 : (cmi?.validFalseMult ?? 0.45));
              if (wmE < 0.12) continue;

              const curE = energy.get(dstId) || 0;
              const dE = direct.get(dstId) || 0;
              if (dE === 0) {
                if (wmE > curE) {
                  energy.set(dstId, wmE);
                  trace.set(dstId, `wm-next(${dir}) <- ${neurons[srcId].code}`);
                }
              } else {
                const boosted = Math.min(curE + wmE * WM_NEXT_CFG.directBoostScale, Math.max(curE, srcE - 0.01));
                if (boosted > curE) energy.set(dstId, boosted);
              }
            }
          }

          if (u1Active) {
            const spkNbs = idx.sameSpkNeighbors.get(srcId);
            if (spkNbs && spkNbs.length) {
              const spkFwd = u1.sameSpkFwd ?? 0.42;
              const spkBwd = u1.sameSpkBwd ?? 0.35;
              for (let ni = 0; ni < spkNbs.length; ni++) {
                const { dstId, dir } = spkNbs[ni];
                const nDst = neurons[dstId];
                if (!nDst || (nDst.private && !allowPrivate)) continue;
                const wmMult = dir === 'fwd' ? spkFwd : spkBwd;
                const wmE = srcE * wmMult * (nDst.valid ? 1.0 : (cmi?.validFalseMult ?? 0.45));
                if (wmE < 0.12) continue;

                const curE = energy.get(dstId) || 0;
                const dE = direct.get(dstId) || 0;
                if (dE === 0) {
                  if (wmE > curE) {
                    energy.set(dstId, wmE);
                    trace.set(dstId, `wm-samespk(${dir}) <- ${neurons[srcId].code}`);
                  }
                } else {
                  const boosted = Math.min(curE + wmE * WM_NEXT_CFG.directBoostScale, Math.max(curE, srcE - 0.01));
                  if (boosted > curE) energy.set(dstId, boosted);
                }
              }
            }
          }
        }
      }
    }
  }

  // M6b Synaptic Mass Ranking Modulator
  if (useSynapticMass && energy.size >= 2) {
    for (const [id, score] of energy.entries()) {
      energy.set(id, score * inertiaMod(id));
    }
  }

  // 3. Ranking + PMI_Safe (pos=0 shielded) + P1b_PureCov Pauli Reranking
  let sorted = [...energy.entries()].sort(
    (a, b) => b[1] - a[1] || ((idx.order.get(a[0]) ?? 0) - (idx.order.get(b[0]) ?? 0))
  );

  if (pmiScoreById && pmiScoreById.size > 0) {
    if (sorted.length === 0) {
      sorted = [...pmiScoreById.entries()].sort(
        (a, b) => b[1] - a[1] || ((idx.order.get(a[0]) ?? 0) - (idx.order.get(b[0]) ?? 0))
      );
      for (const [id] of sorted) {
        if (!trace.has(id)) trace.set(id, 'pmi-safe');
      }
    } else {
      const firstPair = sorted[0];
      const firstId = firstPair[0];
      const restMap = new Map();
      for (let i = 1; i < sorted.length; i++) {
        restMap.set(sorted[i][0], sorted[i][1]);
      }
      for (const [id, boostVal] of pmiScoreById.entries()) {
        if (id === firstId || boostVal <= 0) continue;
        const prev = restMap.get(id) || 0;
        restMap.set(id, prev + boostVal);
        if (prev === 0 && !trace.has(id)) {
          trace.set(id, 'pmi-safe');
        }
      }
      const restSorted = [...restMap.entries()].sort(
        (a, b) => b[1] - a[1] || ((idx.order.get(a[0]) ?? 0) - (idx.order.get(b[0]) ?? 0))
      );
      sorted = [firstPair, ...restSorted];
    }
  }

  const maxInbox = Math.min(2, Math.max(1, Math.floor(topK / 2)));
  const maxEpisodic = options.maxEpisodic ?? 1;
  const selected = [];
  let pauliReorders = 0;

  const canSelectNow = (n, inboxCount, episodicCount) => {
    if (n.type === 'inbox' && inboxCount >= maxInbox) return false;
    if (n.type === 'episodic' && episodicCount >= maxEpisodic) return false;
    return true;
  };

  if (!pauli || sorted.length <= 2) {
    const deferred = [];
    let inboxCount = 0;
    let episodicCount = 0;
    for (const pair of sorted) {
      if (selected.length >= topK) break;
      const n = neurons[pair[0]];
      if (!n) continue;
      if (canSelectNow(n, inboxCount, episodicCount)) {
        selected.push(pair);
        if (n.type === 'inbox') inboxCount++;
        else if (n.type === 'episodic') episodicCount++;
      } else {
        deferred.push(pair);
      }
    }
    for (const pair of deferred) {
      if (selected.length >= topK) break;
      selected.push(pair);
    }
  } else {
    const poolSize = Math.min(sorted.length, pauli.poolSize ?? 40);
    const pool = sorted.slice(0, poolSize);
    const used = new Array(pool.length).fill(false);
    const deferred = [];
    let inboxCount = 0;
    let episodicCount = 0;

    let firstIdx = -1;
    for (let i = 0; i < pool.length; i++) {
      const n = neurons[pool[i][0]];
      if (!n) continue;
      if (canSelectNow(n, inboxCount, episodicCount)) {
        firstIdx = i;
        break;
      }
    }

    const covCount = new Map();
    const sessCount = new Map();
    const selectedIds = [];

    const recordSelected = (poolIdx) => {
      used[poolIdx] = true;
      const pair = pool[poolIdx];
      const [id] = pair;
      const n = neurons[id];
      selected.push(pair);
      selectedIds.push(id);
      if (n.type === 'inbox') inboxCount++;
      else if (n.type === 'episodic') episodicCount++;

      const cm = tokContribById?.get(id);
      if (cm) {
        for (const qt of cm.keys()) {
          covCount.set(qt, (covCount.get(qt) || 0) + 1);
        }
      }
      if (idx.hasSessions) {
        const sk = idx.sessionById.get(id);
        if (sk) sessCount.set(sk, (sessCount.get(sk) || 0) + 1);
      }
    };

    if (firstIdx !== -1) recordSelected(firstIdx);

    const gamma = pauli.gamma ?? 0.65;
    const alpha = pauli.alpha ?? 0.08;
    const beta = pauli.beta ?? 0.22;
    const sessDecay = pauli.sessDecay ?? 1.0;
    const exemptNext = pauli.exemptNext !== false;
    const minInfoGate = pauli.minInfoToks ?? 0;
    const gateUncoveredOnly = Boolean(pauli.gateUncoveredOnly);
    const onlyWithSessions = Boolean(pauli.onlyWithSessions);
    const startPos = pauli.startPos ?? 1;

    const gatePassedStatic = (!onlyWithSessions || idx.hasSessions) && (qInfoToks.length >= minInfoGate);

    while (selected.length < topK) {
      let hasUncoveredInfo = false;
      if (gateUncoveredOnly) {
        for (let i = 0; i < qInfoToks.length; i++) {
          if (!covCount.has(qInfoToks[i])) { hasUncoveredInfo = true; break; }
        }
      }
      const applyPauliNow = (selected.length >= startPos) && gatePassedStatic && (!gateUncoveredOnly || hasUncoveredInfo);

      let bestIdx = -1;
      let bestEffScore = -Infinity;
      let defaultFirstIdx = -1;

      for (let i = 0; i < pool.length; i++) {
        if (used[i]) continue;
        const [id, baseScore] = pool[i];
        const n = neurons[id];
        if (!n || !canSelectNow(n, inboxCount, episodicCount)) continue;
        if (defaultFirstIdx === -1) defaultFirstIdx = i;

        if (!applyPauliNow) {
          bestIdx = i;
          break;
        }

        const cm = tokContribById?.get(id);
        let effScore = baseScore;

        if (cm && cm.size > 0) {
          let rawQ = 0;
          let availQ = 0;
          let novelIdf = 0;
          for (const [qt, c] of cm.entries()) {
            rawQ += c;
            const prevCov = covCount.get(qt) || 0;
            if (prevCov === 0) {
              availQ += c;
              const idf = qTokIdf.get(qt) || 0;
              if (idf >= minInfoIdf) novelIdf += idf;
            } else {
              availQ += c * Math.pow(gamma, prevCov);
            }
          }
          const redRatio = rawQ > 0 ? (rawQ - availQ) / rawQ : 0;
          const novelRatio = totalQInfoIdf > 0 ? Math.min(1.0, novelIdf / totalQInfoIdf) : 0;
          effScore = baseScore * (1 - alpha * redRatio + beta * novelRatio);
        }

        if (sessDecay < 1.0 && idx.hasSessions) {
          const sk = idx.sessionById.get(id);
          const sc = sk ? (sessCount.get(sk) || 0) : 0;
          if (sc > 0) {
            let isNextOfSel = false;
            if (exemptNext) {
              for (let si = 0; si < selectedIds.length; si++) {
                if (idx.nextPairSet.has(`${selectedIds[si]}|${id}`)) {
                  isNextOfSel = true;
                  break;
                }
              }
            }
            if (!isNextOfSel) {
              effScore *= Math.pow(sessDecay, sc);
            }
          }
        }

        if (effScore > bestEffScore) {
          bestEffScore = effScore;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;
      if (bestIdx !== defaultFirstIdx) pauliReorders++;
      recordSelected(bestIdx);
    }

    if (selected.length < topK) {
      const selSet = new Set(selectedIds);
      for (const pair of sorted) {
        if (selected.length >= topK) break;
        if (selSet.has(pair[0])) continue;
        const n = neurons[pair[0]];
        if (!n) continue;
        if (canSelectNow(n, inboxCount, episodicCount)) {
          selected.push(pair);
          selSet.add(pair[0]);
          if (n.type === 'inbox') inboxCount++;
          else if (n.type === 'episodic') episodicCount++;
        } else {
          deferred.push(pair);
        }
      }
      for (const pair of deferred) {
        if (selected.length >= topK) break;
        selected.push(pair);
      }
    }
  }

  const winners = selected.map(([id, score]) => {
    const n = neurons[id];
    return {
      id,
      code: n.code,
      type: n.type,
      community: n.community,
      title: n.title,
      summary: n.summary,
      valid: n.valid,
      ...(n.echo ? { echo: n.echo } : {}),
      energy: round2(score),
      via: trace.get(id) || 'spark',
      dim4: n.dim4,
      coord3d: n.coord3d,
      ptr: n.ptr,
    };
  });

  // 4. Chemical & Limbic Synthesis
  let dopa = 0.50, oxi = 0.42, focus = 0.50;
  const lobeTotals = { daily: 0, exploration: 0, execution: 0, support: 0 };
  for (const w of winners) {
    for (const k of Object.keys(lobeTotals)) lobeTotals[k] += (w.dim4?.[k] || 0) * w.energy;
    if (w.echo) {
      const e = norm(w.echo);
      if (/(warm|relief|trust|proud|happy|calm|grateful|heart|tenderness)/.test(e)) oxi += 0.10;
      if (/(curious|excited|breakthrough|idea|discovery)/.test(e)) dopa += 0.08;
      if (/(alert|urgent|incident|bug|security|stress)/.test(e)) focus += 0.10;
    }
  }
  const maxLobeVal = Math.max(0.001, ...Object.values(lobeTotals));
  const balance4d = Object.fromEntries(
    Object.entries(lobeTotals).map(([k, v]) => [k, Math.round((v / maxLobeVal) * 100)])
  );
  const dominantLobe = Object.entries(balance4d).sort((a, b) => b[1] - a[1])[0]?.[0] || 'exploration';
  dopa = round2(clamp01(dopa + (balance4d.exploration / 100) * 0.25));
  focus = round2(clamp01(focus + (balance4d.execution / 100) * 0.28));
  oxi = round2(clamp01(oxi + (balance4d.support / 100) * 0.30));
  const sero = round2(clamp01(0.72 - Math.abs(dopa - focus) * 0.20 + oxi * 0.15));

  // 5. Real-Time Anti-Redundancy Map (`DOC-* <-> Lesson-*`)
  const evaluatedConcepts = [];
  const seenC = new Set();
  for (const w of winners) {
    const n = neurons[w.id];
    for (const c of n?.concepts || []) {
      const k = norm(c);
      if (seenC.has(k)) continue;
      seenC.add(k);
      evaluatedConcepts.push({ concept: c, source: n.code });
    }
  }

  // 6. Optional In-Memory Reinforcement
  let hebbianLinks = [];
  let homeostasisRescaled = 0;
  if (options.reinforce === true) {
    const re = applyReinforcement(net, winners, options);
    hebbianLinks = re.hebbianLinks;
    homeostasisRescaled = re.homeostasisRescaled;
  }

  const fullRanking = options.includeFullRanking
    ? sorted.map(([id, score]) => ({
        id,
        code: neurons[id]?.code,
        type: neurons[id]?.type,
        valid: neurons[id]?.valid,
        energy: round2(score),
      }))
    : undefined;

  return {
    query,
    ms: round2(performance.now() - t0),
    dominantLobe,
    balance4d,
    chemistry: { dopamine: dopa, oxytocin: oxi, focus, serotonin: sero },
    winners,
    evaluatedConcepts: evaluatedConcepts.slice(0, 10),
    ...(options.includeFullRanking ? { fullRanking } : {}),
    ...(options.includeDiagnostics ? { diagnostics: { pauliReorders } } : {}),
    ...(options.reinforce === true ? { hebbianLinks, homeostasisRescaled } : {}),
  };
}

export function activateNetwork(memoryDir, query = '', options = {}) {
  const net = loadOrCompile(memoryDir, options);
  const res = activateNetworkInMemory(net, query, options);

  let hebbianLinks = [];
  let homeostasisRescaled = 0;
  if (options.reinforce !== false && res.winners.length >= 2) {
    const re = applyReinforcement(net, res.winners, options);
    hebbianLinks = re.hebbianLinks;
    homeostasisRescaled = re.homeostasisRescaled;
    if (re.edgeDeltas.length || re.actDeltas.length) {
      applyHebbianDelta(memoryDir, { activations: re.actDeltas, edges: re.edgeDeltas });
    }
  }

  return {
    ...res,
    hebbianLinks,
    homeostasisRescaled,
    storagePath: getActiveStoragePath(memoryDir),
  };
}

function printCliHelp() {
  console.log([
    `HebbMesh v${VERSION}`,
    'Usage:',
    '  node src/engine_v1.7.mjs compile [memoryDir] [--json]',
    '  node src/engine_v1.7.mjs spark   [memoryDir] <query> [--json]',
    '  node src/engine_v1.7.mjs bridge  [memoryDir] <conceptA> <conceptB> [--json]',
    '  node src/engine_v1.7.mjs --help',
  ].join('\n'));
}

// CLI Entrypoint
const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const rawArgs = process.argv.slice(2);
  if (rawArgs.includes('--help') || rawArgs.includes('-h')) {
    printCliHelp();
    process.exit(0);
  }
  const asJson = rawArgs.includes('--json');
  const pos = rawArgs.filter((a) => a !== '--json' && a !== '--help' && a !== '-h');
  const [cmd = 'compile', dirArg = './demo/memory', ...rest] = pos;
  const memDir = path.resolve(process.cwd(), dirArg);

  if (cmd === 'compile') {
    const net = compileNetwork(memDir);
    if (asJson) console.log(JSON.stringify(net.metrics));
    else console.log(`[HebbMesh v1.7] Compiled ${net.metrics.totalNeurons} neurons | ${net.metrics.totalEdges} edges | ${net.metrics.compileMs} ms`);
  } else if (cmd === 'spark') {
    const q = rest.join(' ') || 'how did we solve slow database reads';
    const res = activateNetwork(memDir, q, { topK: 6 });
    if (asJson) {
      console.log(JSON.stringify(res));
    } else {
      console.log(`[HebbMesh v1.7] "${q}" (${res.ms} ms | Lobe: ${res.dominantLobe.toUpperCase()})`);
      for (const w of res.winners) {
        console.log(`  - [${w.code}] (${w.energy}) ${w.title}  [${w.via}]`);
      }
      if (res.hebbianLinks && res.hebbianLinks.length) console.log(`  Hebb: ${res.hebbianLinks.join(' | ')}`);
    }
  } else if (cmd === 'bridge') {
    const net = loadOrCompile(memDir);
    const [a = 'sqlite', b = 'burnout'] = rest;
    const br = findBridge(net, a, b);
    if (asJson) console.log(JSON.stringify(br));
    else console.log(`[HebbMesh GPS v1.7] "${a}" -> "${b}" (${br.ms} ms):`, br.found ? br.path.map((p) => p.code).join(' -> ') : br.reason);
  } else {
    printCliHelp();
    process.exit(1);
  }
}
