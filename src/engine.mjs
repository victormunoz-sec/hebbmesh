import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  loadNetwork,
  saveNetwork,
  applyHebbianDelta,
  getStorageMtime,
  getActiveStoragePath,
} from './storage.mjs';

/**
 * HebbMesh Engine v1.0.0 — Zero-Bloat 4D Synaptic Memory over Markdown
 * Copyright (c) 2026 Victor Muñoz Jara (Santiago, CL) · Dual License: AGPL-3.0 / Commercial
 *
 * Capabilities:
 *  - Markdown -> 4D/3D Neurons (Daily, Exploration, Execution, Support)
 *  - Zero-Embedding Seeding (`| sem: ...`) -> 0 MB runtime semantic bridging
 *  - O(1) Hebbian edge & dendrite reinforcement over native `node:sqlite` WAL
 *  - Weekly Episodic Nodes (`S-Wnn-F1..F4`) with triple anti-super-hub throttling
 *  - Real-time Anti-Redundancy Mapping (`DOC-* <-> Lesson-*`)
 *  - 3D A* Synaptic Bridging (`findBridge`) with anti-pareidolia + Privacy Firewall
 */

export const VERSION = '1.2.0-HebbMesh-4D';

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

  // Peel trailing metadata suffixes (`| sem: ...`, `| eco: ...`) ignoring `|` inside inline backticks
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

  return { clean: clean.replace(/\|\s*$/, '').trim(), echo, sem };
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
    const { clean, echo, sem } = parseMetadata(line.replace(/^\s*-\s+\[[^\]]+\]\s*/, ''));
    if (!clean) continue;
    const code = `HITO-${idxM}`;
    const id = `n_HITO_${idxM}`;
    const { dim4, coord3d, weight } = compute4DCoords('milestone', 'EXECUTION', clean, true);
    neurons.set(id, {
      id, code, type: 'milestone', community: 'EXECUTION',
      title: clean.slice(0, 90),
      summary: clean.slice(0, 180),
      ...(echo ? { echo } : {}),
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
    // M0: Semantic Synaptogenesis (tags + concepts + | sem:) & Clique Pruning (df <= max(6, 5% N))
    const dfMax = Math.max(6, Math.round(N * 0.05));
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
    architecture: 'HebbMesh 4D/3D Synaptic Memory + Zero-Embedding Seeding + O(1) SQLite WAL',
    updatedAt: new Date().toISOString(),
    metrics: {
      totalNeurons: map.size,
      totalEdges,
      compileMs: round2(performance.now() - t0),
    },
    neurons: neuronsObj,
  };

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

export function activateNetworkInMemory(net, query = '', options = {}) {
  const t0 = performance.now();
  const topK = options.topK ?? 6;
  const maxHops = options.disablePropagation ? 0 : (options.maxHops ?? 2);
  const allowPrivate = Boolean(options.allowPrivate) || Boolean(options.disablePrivacyFirewall);
  const neurons = net.neurons || {};

  const qTokens = tokenize(query);
  const qNorm = norm(query);
  const energy = new Map();
  const trace = new Map();

  // Compute IDF over tags + dendrites for sharp lexical weighting
  const N = Math.max(1, Object.keys(neurons).length);
  const df = new Map();
  for (const n of Object.values(neurons)) {
    if (n.private && !allowPrivate) continue;
    const seen = new Set([...(n.tags || []), ...Object.keys(n.dendrites || {})]);
    for (const tok of seen) df.set(tok, (df.get(tok) || 0) + 1);
  }

  // 1. Direct Sparking (exact code, tags, Zero-Embedding dendrites, and title/summary fallback)
  for (const [id, n] of Object.entries(neurons)) {
    if (n.private && !allowPrivate) continue;
    let score = 0;
    const matched = [];

    if (qNorm.includes(norm(n.code))) {
      score += 3.2;
      matched.push(n.code);
    }

    const titleNorm = norm(n.title);
    for (const qt of qTokens) {
      let bestTok = 0;
      let isDend = false;

      for (const tag of n.tags || []) {
        const m = tokenMatchScore(qt, tag);
        if (m > bestTok) bestTok = m;
      }
      for (const [dTok, dW] of Object.entries(n.dendrites || {})) {
        const m = tokenMatchScore(qt, dTok) * dW;
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

  // 2. Spreading Activation (v1.2: M1 Sparse Out-Fan + Regularized Sub-stochastic + Refractory,
  //    M2 AMPA/NMDA Coincidence Gating + Parental Dominance, M3 k-WTA Lateral Inhibition, M4 4D Inter-Lobe Firewall)
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
      // M1 only (or M1 + M3/M4 without AMPA/NMDA gating): multi-hop sparse out-fan with refractory period
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
            const imp = srcE * (edgeW / denom) * damping * lobeMod * (nDst.valid ? 1.0 : 0.45);
            if (imp < 0.12) continue;
            energy.set(dstId, (energy.get(dstId) || 0) + imp);
            if (!trace.has(dstId)) {
              trace.set(dstId, `hop${hop + 1} <- ${nSrc.code} (${edgeW})`);
            }
          }
        }
      }
    } else {
      // M1 + M2 (+ optional M3 k-WTA, M4 4D Firewall, GABA soft)
      const seeds = [...direct.entries()]
        .filter(([, e]) => e >= relWta * maxDirect)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topEmit);

      const seedPole = useM4Firewall ? computeLogicalVsAffectivePole(seeds, neurons) : 0;

      const propSum = new Map();
      const parentsOf = new Map();
      const maxParentE = new Map();
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
          const imp = srcE * (edgeW / denom) * damping * lobeMod * (nDst.valid ? 1.0 : 0.45);
          if (imp < 0.12) continue;

          propSum.set(dstId, (propSum.get(dstId) || 0) + imp);
          if (!parentsOf.has(dstId)) parentsOf.set(dstId, new Set());
          parentsOf.get(dstId).add(srcId);
          maxParentE.set(dstId, Math.max(maxParentE.get(dstId) || 0, srcE));

          if (!trace.has(dstId)) {
            trace.set(dstId, `hop1 <- ${nSrc.code} (${edgeW})`);
          }
        }
      }

      // M2: Post-synaptic AMPA vs. NMDA integration + Parental Dominance (+ optional GABA soft hub brake)
      for (const [dstId, rawProp] of propSum.entries()) {
        const dE = direct.get(dstId) || 0;
        const kParents = parentsOf.get(dstId)?.size || 0;
        const pMax = maxParentE.get(dstId) || 0;
        const nDst = neurons[dstId];
        const dstDeg = Object.keys(nDst.edges || {}).length;
        const gabaBrake = useGabaSoft && dstDeg > 4 ? 1 / (1 + 0.08 * (dstDeg - 4)) : 1.0;
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

  // 3. Diversity & Anti-Super-Hub Throttling (max 2 inbox, max 1 weekly episodic)
  const sorted = [...energy.entries()].sort((a, b) => b[1] - a[1]);
  const maxInbox = Math.min(2, Math.max(1, Math.floor(topK / 2)));
  const maxEpisodic = options.maxEpisodic ?? 1;
  const selected = [];
  const deferred = [];
  let inboxCount = 0;
  let episodicCount = 0;

  for (const pair of sorted) {
    if (selected.length >= topK) break;
    const n = neurons[pair[0]];
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

  return {
    query,
    ms: round2(performance.now() - t0),
    dominantLobe,
    balance4d,
    chemistry: { dopamine: dopa, oxytocin: oxi, focus, serotonin: sero },
    winners,
    evaluatedConcepts: evaluatedConcepts.slice(0, 10),
  };
}

export function activateNetwork(memoryDir, query = '', options = {}) {
  const net = loadOrCompile(memoryDir, options);
  const res = activateNetworkInMemory(net, query, options);

  // O(1) Hebbian reinforcement on top-3 winners
  const hebbianLinks = [];
  if (options.reinforce !== false && res.winners.length >= 2) {
    const top = res.winners.slice(0, 3);
    const actDeltas = [];
    const edgeDeltas = [];
    for (const w of top) {
      const n = net.neurons[w.id];
      n.activations = Number(n.activations || 0) + 1;
      actDeltas.push({ id: n.id, activations: n.activations });
    }
    for (let i = 0; i < top.length; i++) {
      for (let j = i + 1; j < top.length; j++) {
        const a = net.neurons[top[i].id];
        const b = net.neurons[top[j].id];
        const nextW = round2(Math.min(1.0, Number(a.edges[b.id] || 0.35) + 0.04));
        a.edges[b.id] = nextW;
        b.edges[a.id] = nextW;
        edgeDeltas.push({ source: a.id, target: b.id, weight: nextW });
        edgeDeltas.push({ source: b.id, target: a.id, weight: nextW });
        hebbianLinks.push(`${a.code}<->${b.code}:${nextW}`);
      }
    }
    applyHebbianDelta(memoryDir, { activations: actDeltas, edges: edgeDeltas });
  }

  return { ...res, hebbianLinks, storagePath: getActiveStoragePath(memoryDir) };
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
    else console.log(`[HebbMesh] Compiled ${net.metrics.totalNeurons} neurons | ${net.metrics.totalEdges} edges | ${net.metrics.compileMs} ms`);
  } else if (cmd === 'spark') {
    const q = rest.join(' ') || 'how did we solve slow database reads';
    const res = activateNetwork(memDir, q, { topK: 6 });
    if (asJson) {
      console.log(JSON.stringify(res));
    } else {
      console.log(`[HebbMesh] "${q}" (${res.ms} ms · Lobe: ${res.dominantLobe.toUpperCase()})`);
      for (const w of res.winners) {
        console.log(`  - [${w.code}] (${w.energy}) ${w.title}  [${w.via}]`);
      }
      if (res.hebbianLinks.length) console.log(`  Hebb O(1): ${res.hebbianLinks.join(' | ')}`);
    }
  } else if (cmd === 'bridge') {
    const net = loadOrCompile(memDir);
    const [a = 'sqlite', b = 'burnout'] = rest;
    const br = findBridge(net, a, b);
    if (asJson) console.log(JSON.stringify(br));
    else console.log(`[HebbMesh GPS] "${a}" -> "${b}" (${br.ms} ms):`, br.found ? br.path.map((p) => p.code).join(' -> ') : br.reason);
  }
}
