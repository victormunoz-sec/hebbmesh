import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  loadNetwork,
  saveNetwork,
  applyHebbianDelta,
  getStorageMtime,
  getActiveStoragePath,
} from './storage.mjs';

/**
 * HebbMesh Engine v1.3.0-HebbMesh-4D-Quantum
 * Zero-Bloat 4D Synaptic Memory over Markdown
 * Copyright (c) 2026 Victor Munoz Jara (Santiago, CL) · Dual License: AGPL-3.0 / Commercial
 *
 * v1.3 Quantum Turbo Enhancements:
 *  - M0/Turbo 10k: Capped dfMax <= 40 synaptogenesis and lazy non-enumerable net._index
 *  - M6a: Quantum 4D Phase Interference (destructive interference gating on polysemic noise)
 *  - M6b: Synaptic Mass & Inertia (log-sigmoidal resistance against ephemeral drift)
 *  - M5: Turrigiano Homeostatic Synaptic Scaling with confidence gating
 */

export const VERSION = '1.3.0-HebbMesh-4D-Quantum';

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

/**
 * Extracts inline Zero-Embedding Seeds (`| sem: a, b, c`) and Affective Echo (`| eco: ...`).
 */
export function parseMetadata(lineRaw = '') {
  let clean = String(lineRaw || '').replace(/<!--.*?-->/g, '').trim();
  let echo = '';
  let sem = [];
  let sessionId = '';

  // Peel trailing metadata suffixes (`| sem: ...`, `| eco: ...`, `| session: ...`) ignoring `|` inside inline backticks
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

  // Legacy italic parenthetical forms outside inline backticks
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

  // Semantic nudges across lobes
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
  if (type === 'episodic') z = 0.62; // Anti-super-hub backing weight
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
    const { clean, echo, sem } = parseMetadata(mG[3]);
    const id = `n_${code.replace('-', '_')}`;
    const { dim4, coord3d, weight } = compute4DCoords('graph', community, clean, valid);
    neurons.set(id, {
      id, code, type: 'graph', community,
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(clean),
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
    const { clean, echo, sem } = parseMetadata(mA[4]);
    const full = `${title}: ${clean}`;
    const id = `n_${code.replace('-', '_')}`;
    const { dim4, coord3d, weight } = compute4DCoords('lesson', 'EXPLORATION', full, valid);
    const concepts = extractKeyConcepts(full, 8);
    neurons.set(id, {
      id, code, type: 'lesson', community: 'EXPLORATION',
      title: title.slice(0, 90),
      summary: full.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(concepts.length ? { concepts } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(full),
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
    neurons.set(id, {
      id, code, type: 'milestone', community: 'EXECUTION',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(clean),
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
    const { clean, echo, sem } = parseMetadata(line.replace(/^\s*-\s+/, ''));
    if (clean.length < 8) continue;
    const code = `PERFIL-${idxP}`;
    const id = `n_PERFIL_${idxP}`;
    const { dim4, coord3d, weight } = compute4DCoords('profile', 'DAILY', clean, true);
    neurons.set(id, {
      id, code, type: 'profile', community: 'DAILY',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(clean),
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
    const { clean, echo, sem } = parseMetadata(mI[2]);
    const code = `INBOX-${idxI}`;
    const id = `n_INBOX_${idxI}`;
    const isPriv = sub === 'PRIVATE' || sub === 'INTIMO';
    const { dim4, coord3d, weight } = compute4DCoords('inbox', sub, clean, true);
    neurons.set(id, {
      id, code, type: 'inbox', community: sub,
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(clean),
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
    const { clean, echo, sem } = parseMetadata(line.replace(/^\s*-\s+/, ''));
    if (clean.length < 8) continue;
    const code = `PRIV-${idxV}`;
    const id = `n_PRIV_${idxV}`;
    const { dim4, coord3d, weight } = compute4DCoords('private', 'PRIVATE', clean, true);
    neurons.set(id, {
      id, code, type: 'private', community: 'PRIVATE',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
      ...(!disableSeeds && sem.length ? { sem } : {}),
      tags: extractTags(clean),
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
      const summary = mSum ? `${title}: ${mSum[1].trim()}` : `${title} — ${raw.replace(/^---[\s\S]*?---/, '').replace(/\s+/g, ' ').trim().slice(0, 140)}`;
      const sem = !disableSeeds && mSem ? mSem[1].split(/[,;]+/).map((s) => s.trim()).filter(Boolean) : [];
      const concepts = extractKeyConcepts(raw, 12);
      const code = `DOC-${idxD}`;
      const id = `n_DOC_${idxD}`;
      const { dim4, coord3d, weight } = compute4DCoords('knowledge', 'KNOWLEDGE', `${title} ${summary}`, true);
      neurons.set(id, {
        id, code, type: 'knowledge', community: 'KNOWLEDGE',
        title: title.slice(0, 90),
        summary: summary.slice(0, 180),
        ...(concepts.length ? { concepts } : {}),
        ...(sem.length ? { sem } : {}),
        tags: extractTags(`${title} ${summary} ${raw.slice(0, 900)}`),
        dim4, coord3d, weight, valid: true, private: false,
        ptr: `knowledge/${f}`,
        edges: {}, dendrites: {}, activations: 0,
      });
    }
  }

  // 8. Weekly Episodic Syntheses (`synthesis/AAAA-Wnn.md` -> S-Wnn-F1..F4 with triple anti-super-hub throttle)
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
          title: `Week ${semCode} · ${labels[fKey]}`,
          summary: body.slice(0, 180),
          tags, dim4, coord3d, weight, valid: true, private: false,
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
  const tagToIds = new Map();
  for (const n of list) {
    for (const tag of n.tags) {
      if (!tagToIds.has(tag)) tagToIds.set(tag, []);
      tagToIds.get(tag).push(n.id);
    }
  }

  const N = Math.max(1, list.length);
  const pairScores = new Map();

  if (m0Topology) {
    // M0: Semantic Synaptogenesis (tags + concepts + | sem:) & Clique Pruning (df <= min(40, max(6, 5% N)))
    const dfMax = Math.min(40, Math.max(6, Math.round(N * 0.05)));
    const tokToEntries = new Map();
    const registerTok = (tok, id, factor) => {
      let m = tokToEntries.get(tok);
      if (!m) { m = new Map(); tokToEntries.set(tok, m); }
      const prev = m.get(id) || 0;
      if (factor > prev) m.set(id, factor);
    };

    for (const n of list) {
      for (const tag of n.tags) registerTok(tag, n.id, 1.0);
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
    // Legacy v1.1: shared-tag IDF edge weighting (df <= 24)
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
    // Privacy Firewall (F-01): never link public neurons to private-vault neurons
    if (!disablePrivacyFirewall && Boolean(neurons.get(a)?.private) !== Boolean(neurons.get(b)?.private)) continue;
    const w = round2(Math.min(0.92, 0.34 + score * 0.62));
    addEdge(a, b, w);
    addEdge(b, a, w);
  }

  // Preserve previous Hebbian weights
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

  // Keep top 8 strongest edges per neuron (rich yet sparse graph)
  for (const n of list) {
    n.edges = Object.fromEntries(
      Object.entries(n.edges).sort((a, b) => b[1] - a[1]).slice(0, 8)
    );
  }

  // Diffuse dendrites: Zero-Embedding Seeds (`| sem:`, weight 0.80) + 1-hop neighbor IDF diffusion
  for (const n of list) {
    const ownTags = new Set(n.tags);
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
      // Defense-in-depth (F-01): never diffuse private-vault tags into public dendrites
      if (!nb || (!disablePrivacyFirewall && !n.private && nb.private)) continue;
      for (const tag of nb.tags) {
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

  // Native Temporal NEXT Edges (v1.3): bidirectional links between consecutive nodes sharing the same sessionId
  // Active by default whenever `sessionId` (or `session`) is present; 0.0 effect if no node has sessionId.
  const linkTemporalNext = options.linkTemporalNext !== false;
  const temporalNextWeight = round2(Number(options.temporalNextWeight ?? 0.72));
  if (linkTemporalNext && temporalNextWeight > 0) {
    const lastBySession = new Map();
    for (const n of list) {
      const sid = n.sessionId ?? n.session;
      if (sid === undefined || sid === null || String(sid).trim() === '') continue;
      const key = String(sid).trim();
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
}

/**
 * Lazy non-enumerable net._index builder and accessor (Turbo 10k).
 */
export function ensureIndex(net, forceRebuild = false) {
  const neurons = net.neurons || {};
  const currentCount = Object.keys(neurons).length;

  if (!forceRebuild && net._index && net._index.N === currentCount) {
    return net._index;
  }

  const neuronList = Object.values(neurons);
  const N = currentCount;

  const dfPublic = new Map();
  const dfAll = new Map();
  const exactTokToIds = new Map();
  const codeMap = new Map();
  const normCodes = new Map();
  const order = new Map();
  const dendEntries = new Map();
  const titleNorms = new Map();
  const titleWordToIds = new Map();
  const mass = new Map();

  let sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

  for (let idxPos = 0; idxPos < neuronList.length; idxPos++) {
    const n = neuronList[idxPos];
    order.set(n.id, idxPos);
    dendEntries.set(n.id, Object.entries(n.dendrites || {}));

    // 4D centroid accumulator
    const d4 = n.dim4 || { daily: 0.15, exploration: 0.15, execution: 0.15, support: 0.15 };
    sum0 += Number(d4.daily || 0);
    sum1 += Number(d4.exploration || 0);
    sum2 += Number(d4.execution || 0);
    sum3 += Number(d4.support || 0);

    // M6b: Synaptic Mass m = 1 / (sigma^2)
    // Consolidated long-term memory (lesson, graph, knowledge, milestone, profile) has low volatility (high mass),
    // while weekly episodic summaries and ephemeral inbox notes have higher volatility (lower mass).
    let sigma = 0.85;
    if (n.type === 'lesson' || n.type === 'graph' || n.type === 'knowledge') {
      sigma = 0.65;
    } else if (n.type === 'milestone' || n.type === 'profile') {
      sigma = 0.68;
    } else if (n.type === 'episodic') {
      sigma = 0.90;
    } else if (n.type === 'inbox') {
      sigma = 1.35;
    }
    mass.set(n.id, 1 / (sigma * sigma));

    // Code Map
    const nc = norm(n.code || '');
    normCodes.set(n.id, nc);
    if (nc) {
      codeMap.set(nc, n.id);
      const unhyphen = nc.replace(/[^a-z0-9]/g, '');
      if (unhyphen && !codeMap.has(unhyphen)) {
        codeMap.set(unhyphen, n.id);
      }
    }

    // Title indexation
    const tNorm = norm(n.title || '');
    titleNorms.set(n.id, tNorm);
    const words = tNorm.split(/[^a-z0-9_-]+/).filter((w) => w.length >= 3);
    const seenWords = new Set(words);
    for (const w of seenWords) {
      let wArr = titleWordToIds.get(w);
      if (!wArr) { wArr = []; titleWordToIds.set(w, wArr); }
      wArr.push(n.id);
    }

    // Tags and dendrites for lexical index & DF maps
    const seenToks = new Set([...(n.tags || []), ...Object.keys(n.dendrites || {})]);
    for (const tok of seenToks) {
      dfAll.set(tok, (dfAll.get(tok) || 0) + 1);
      if (!n.private) {
        dfPublic.set(tok, (dfPublic.get(tok) || 0) + 1);
      }
      let tArr = exactTokToIds.get(tok);
      if (!tArr) { tArr = []; exactTokToIds.set(tok, tArr); }
      tArr.push(n.id);
    }
  }

  // 4-character prefix maps for fast exact candidate pruning
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

  // 4D Functional-Pole Quantum Phase unit vectors (Hadamard-like basis, corpus-invariant):
  //   Axis 0 (Macro Pole):  Technical (exploration + execution) vs Affective/Life (daily + support)
  //   Axis 1 (Intra-Tech):  Exploration vs Execution (acute positive angle cos > 0 between tech siblings)
  //   Axis 2 (Intra-Life):  Daily vs Support (acute positive angle cos > 0 between life siblings)
  //   Axis 3 (Neutral):     Orthogonal fallback for balanced/neutral nodes
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
    if (mag < 1e-6) {
      phase4d.set(n.id, [0, 0, 0, 1]);
    } else {
      phase4d.set(n.id, [v0 / mag, v1 / mag, v2 / mag, 0]);
    }
  }

  const idx = {
    N,
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
    titlePrefix4ToWords,
    meanDim4,
    phase4d,
    mass,
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
    architecture: 'HebbMesh 4D/Quantum Synaptic Memory + Zero-Embedding Seeding + O(1) SQLite WAL',
    updatedAt: new Date().toISOString(),
    metrics: {
      totalNeurons: map.size,
      totalEdges,
      compileMs: round2(performance.now() - t0),
    },
    neurons: neuronsObj,
  };

  // Pre-seed non-enumerable index
  ensureIndex(net, true);

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

  // Symmetric adjacency filtered by Privacy Firewall
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
    // 1) Confidence gate: only reinforce if winners.length >= 2 AND (w[0] >= 1.5 || (len < 4 ? w[0] >= 1.0 : (w[0] - w[3]) >= 0.35))
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

  // 2) Symmetric edge reinforcement (+0.04 cap 1.0)
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

  // 3) Turrigiano Synaptic Scaling if sumW > wMax
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
  const neurons = net.neurons || {};

  const idx = ensureIndex(net);
  const N = idx.N;
  const df = allowPrivate ? idx.dfAll : idx.dfPublic;

  const qTokens = tokenize(query);
  const qNorm = norm(query);
  const energy = new Map();
  const trace = new Map();

  const inertiaMod = (id) => {
    const m = idx.mass.get(id) ?? 1.384;
    return Math.max(0.82, Math.min(1.18, Math.pow(m, 0.18)));
  };

  // Turbo 10k: Sub-millisecond candidate ID gathering
  const candIds = new Set();

  // a) codeMap: hyphenated patterns and words
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

  // b) For each qt in qTokens: exact, prefix-4 and title fallback
  for (const qt of qTokens) {
    const exactIds = idx.exactTokToIds.get(qt);
    if (exactIds) {
      for (let i = 0; i < exactIds.length; i++) candIds.add(exactIds[i]);
    }
    if (qt.length >= 4) {
      const p4Toks = idx.prefix4ToToks.get(qt.slice(0, 4));
      if (p4Toks) {
        for (let i = 0; i < p4Toks.length; i++) {
          const tok = p4Toks[i];
          if (tokenMatchScore(qt, tok) > 0) {
            const tokIds = idx.exactTokToIds.get(tok);
            if (tokIds) {
              for (let j = 0; j < tokIds.length; j++) candIds.add(tokIds[j]);
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
          if (w.startsWith(qt)) {
            const wIds = idx.titleWordToIds.get(w);
            if (wIds) {
              for (let j = 0; j < wIds.length; j++) candIds.add(wIds[j]);
            }
          }
        }
      }
    }
  }

  // 1. Direct Sparking (Score ONLY candidate IDs, skipping private when not allowed)
  for (const id of candIds) {
    const n = neurons[id];
    if (!n || (n.private && !allowPrivate)) continue;
    let score = 0;
    const matched = [];

    const normCode = idx.normCodes.get(id) || '';
    if (normCode && qNorm.includes(normCode)) {
      score += 3.2;
      matched.push(n.code);
    }

    const titleNorm = idx.titleNorms.get(id) || '';
    const nTags = n.tags || [];
    const nDend = idx.dendEntries.get(id) || [];
    for (let qi = 0; qi < qTokens.length; qi++) {
      const qt = qTokens[qi];
      let bestTok = 0;
      let isDend = false;

      for (let ti = 0; ti < nTags.length; ti++) {
        const m = tokenMatchScore(qt, nTags[ti]);
        if (m > bestTok) bestTok = m;
      }
      for (let di = 0; di < nDend.length; di++) {
        const m = tokenMatchScore(qt, nDend[di][0]) * nDend[di][1];
        if (m > bestTok) { bestTok = m; isDend = true; }
      }
      if (bestTok > 0) {
        const freq = df.get(qt) || 2;
        const idf = Math.max(0.4, Math.log10((N + 1) / freq) + 0.5);
        score += bestTok * idf * (isDend ? 1.12 : 1.0);
        matched.push(isDend ? `~${qt}` : qt);
      } else if (titleNorm.includes(qt)) {
        score += 0.55;
        matched.push(qt);
      }
    }

    if (score > 0) {
      const validMult = n.valid ? 1.0 : 0.45;
      const weightMult = 0.72 + 0.38 * Number(n.weight || 0.65);
      energy.set(id, score * validMult * weightMult);
      trace.set(id, `spark (${[...new Set(matched)].slice(0, 4).join(', ')})`);
    }
  }

  // 2. Spreading Activation
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
      // M1 only
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
            const imp = srcE * (edgeW / denom) * damping * lobeMod * (nDst.valid ? 1.0 : 0.45) * massFactor;
            if (imp < 0.12) continue;
            energy.set(dstId, (energy.get(dstId) || 0) + imp);
            if (!trace.has(dstId)) {
              trace.set(dstId, `hop${hop + 1} <- ${nSrc.code} (${edgeW})`);
            }
          }
        }
      }
    } else {
      // M1 + M2 (AMPA / NMDA) + M6a Quantum Phase Interference
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
          const imp = srcE * (edgeW / denom) * damping * lobeMod * (nDst.valid ? 1.0 : 0.45) * massFactor;
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

      // M2 Integration: Classic vs. M6a Quantum Phase Interference
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
          } else if (kParents >= 2) {
            if (coherence < 0.70) {
              // Destructive interference: block activation of noise hub
            } else {
              const cohScale = coherence < 0.85 ? 0.5 : 1.0;
              const effProp = eQuantum * cohScale * gabaBrake;
              const nmdaGain = (nDst.type === 'episodic' ? 1.15 : 0.95) * (1 + 0.35 * Math.log(kParents));
              const nmdaE = Math.min(effProp * nmdaGain, 0.92 * pMax);
              energy.set(dstId, nmdaE);
              trace.set(dstId, `q-nmda(k=${kParents}, coh=${round2(coherence)}) <- ${[...parentsOf.get(dstId)].map((id) => neurons[id].code).join('+')}`);
            }
          }
        } else {
          // Classic v1.2 linear propagation
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
    }
  }

  // M6b Synaptic Mass Ranking Modulator
  if (useSynapticMass && energy.size >= 2) {
    for (const [id, score] of energy.entries()) {
      energy.set(id, score * inertiaMod(id));
    }
  }

  // 3. Diversity & Anti-Super-Hub Throttling
  const sorted = [...energy.entries()].sort(
    (a, b) => b[1] - a[1] || ((idx.order.get(a[0]) ?? 0) - (idx.order.get(b[0]) ?? 0))
  );
  const maxInbox = Math.min(2, Math.max(1, Math.floor(topK / 2)));
  const maxEpisodic = options.maxEpisodic ?? 1;
  const selected = [];
  const deferred = [];
  let inboxCount = 0;
  let episodicCount = 0;

  for (const pair of sorted) {
    if (selected.length >= topK) break;
    const n = neurons[pair[0]];
    if (!n) continue;
    if (n.type === 'inbox') {
      if (inboxCount < maxInbox) { selected.push(pair); inboxCount++; }
      else deferred.push(pair);
    } else if (n.type === 'episodic') {
      if (episodicCount < maxEpisodic) { selected.push(pair); episodicCount++; }
      else deferred.push(pair);
    } else {
      selected.push(pair);
    }
  }
  for (const pair of deferred) {
    if (selected.length >= topK) break;
    selected.push(pair);
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

  return {
    query,
    ms: round2(performance.now() - t0),
    dominantLobe,
    balance4d,
    chemistry: { dopamine: dopa, oxytocin: oxi, focus, serotonin: sero },
    winners,
    evaluatedConcepts: evaluatedConcepts.slice(0, 10),
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

// CLI Entrypoint
const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const rawArgs = process.argv.slice(2);
  const asJson = rawArgs.includes('--json');
  const pos = rawArgs.filter((a) => a !== '--json');
  const [cmd = 'compile', dirArg = './demo/memory', ...rest] = pos;
  const memDir = path.resolve(process.cwd(), dirArg);
  if (cmd === 'compile') {
    const net = compileNetwork(memDir);
    if (asJson) console.log(JSON.stringify(net.metrics));
    else console.log(`[HebbMesh v1.3] Compiled ${net.metrics.totalNeurons} neurons | ${net.metrics.totalEdges} edges | ${net.metrics.compileMs} ms`);
  } else if (cmd === 'spark') {
    const q = rest.join(' ') || 'how did we solve slow database reads';
    const res = activateNetwork(memDir, q, { topK: 6 });
    if (asJson) {
      console.log(JSON.stringify(res));
    } else {
      console.log(`[HebbMesh v1.3] "${q}" (${res.ms} ms · Lobe: ${res.dominantLobe.toUpperCase()})`);
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
    else console.log(`[HebbMesh GPS v1.3] "${a}" -> "${b}" (${br.ms} ms):`, br.found ? br.path.map((p) => p.code).join(' -> ') : br.reason);
  }
}
