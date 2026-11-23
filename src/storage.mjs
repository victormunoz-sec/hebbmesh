import fs from 'fs';
import path from 'path';

/**
 * HebbMesh Storage — Native `node:sqlite` WAL persistence (0 external dependencies)
 * with transparent atomic JSON fallback for Node < 22.5.
 *
 * Design invariant:
 *   - Plain Markdown files are ALWAYS the single source of truth.
 *   - `hebbmesh.sqlite` is a rebuildable derived index supporting O(1) Hebbian updates.
 */

let DatabaseSync = null;
try {
  const mod = await import('node:sqlite');
  if (mod && typeof mod.DatabaseSync === 'function') {
    DatabaseSync = mod.DatabaseSync;
  }
} catch {
  DatabaseSync = null;
}

export const SUPPORTS_SQLITE = Boolean(DatabaseSync);

export function getSqlitePath(memoryDir) {
  return path.resolve(memoryDir, '.hebbmesh.sqlite');
}

export function getJsonPath(memoryDir) {
  return path.resolve(memoryDir, '.hebbmesh.json');
}

export function getActiveStoragePath(memoryDir) {
  return SUPPORTS_SQLITE ? getSqlitePath(memoryDir) : getJsonPath(memoryDir);
}

export function getStorageMtime(memoryDir) {
  const paths = SUPPORTS_SQLITE
    ? [getSqlitePath(memoryDir), `${getSqlitePath(memoryDir)}-wal`, getJsonPath(memoryDir)]
    : [getJsonPath(memoryDir)];
  let maxMtime = 0;
  for (const p of paths) {
    try {
      if (fs.existsSync(p)) {
        const mt = fs.statSync(p).mtimeMs || 0;
        if (mt > maxMtime) maxMtime = mt;
      }
    } catch { }
  }
  return maxMtime;
}

function openDb(memoryDir, initSchema = false) {
  const dbPath = getSqlitePath(memoryDir);
  if (initSchema) fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA busy_timeout = 3000; PRAGMA synchronous = NORMAL;');
  if (initSchema) {
    db.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS neurons (
        id TEXT PRIMARY KEY,
        code TEXT NOT NULL,
        type TEXT NOT NULL,
        community TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        echo TEXT,
        concepts_json TEXT,
        tags_json TEXT NOT NULL,
        dim4_json TEXT NOT NULL,
        coord3d_json TEXT NOT NULL,
        weight REAL NOT NULL,
        active_flag INTEGER NOT NULL,
        is_private INTEGER NOT NULL DEFAULT 0,
        ptr TEXT NOT NULL,
        activations INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS edges (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (source, target)
      );
      CREATE TABLE IF NOT EXISTS dendrites (
        neuron_id TEXT NOT NULL,
        token TEXT NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (neuron_id, token)
      );
      CREATE INDEX IF NOT EXISTS idx_dendrites_token ON dendrites(token);
    `);
  }
  return db;
}

function writeJsonAtomic(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(tmp, content, 'utf8');
  for (let i = 0; i < 5; i++) {
    try {
      fs.renameSync(tmp, filePath);
      return;
    } catch {
      if (i === 4) {
        fs.writeFileSync(filePath, content, 'utf8');
        try { fs.unlinkSync(tmp); } catch { }
        return;
      }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 30);
    }
  }
}

export function loadNetwork(memoryDir) {
  if (SUPPORTS_SQLITE) {
    const dbPath = getSqlitePath(memoryDir);
    if (fs.existsSync(dbPath)) {
      let db = null;
      try {
        db = openDb(memoryDir, false);
        const metaRows = db.prepare('SELECT key, value FROM meta').all();
        if (!metaRows.length) return loadJsonFallback(memoryDir);
        const meta = {};
        for (const r of metaRows) {
          try { meta[r.key] = JSON.parse(r.value); } catch { meta[r.key] = r.value; }
        }
        const rowsN = db.prepare('SELECT * FROM neurons').all();
        if (!rowsN.length) return loadJsonFallback(memoryDir);

        const neurons = {};
        for (const r of rowsN) {
          const obj = {
            id: r.id,
            code: r.code,
            type: r.type,
            community: r.community,
            title: r.title,
            summary: r.summary,
            tags: JSON.parse(r.tags_json || '[]'),
            dim4: JSON.parse(r.dim4_json || '{}'),
            coord3d: JSON.parse(r.coord3d_json || '{}'),
            weight: Number(r.weight),
            valid: Boolean(r.active_flag),
            private: Boolean(r.is_private),
            ptr: r.ptr,
            activations: Number(r.activations || 0),
            edges: {},
            dendrites: {},
          };
          if (r.echo) obj.echo = r.echo;
          if (r.concepts_json) {
            const c = JSON.parse(r.concepts_json);
            if (Array.isArray(c) && c.length) obj.concepts = c;
          }
          neurons[r.id] = obj;
        }

        for (const e of db.prepare('SELECT source, target, weight FROM edges').all()) {
          if (neurons[e.source]) neurons[e.source].edges[e.target] = Number(e.weight);
        }
        for (const d of db.prepare('SELECT neuron_id, token, weight FROM dendrites').all()) {
          if (neurons[d.neuron_id]) neurons[d.neuron_id].dendrites[d.token] = Number(d.weight);
        }

        return {
          version: meta.version || '',
          architecture: meta.architecture || '',
          updatedAt: meta.updatedAt || '',
          metrics: meta.metrics || {},
          storage: 'node:sqlite',
          neurons,
        };
      } catch {
        return loadJsonFallback(memoryDir);
      } finally {
        try { db?.close(); } catch { }
      }
    }
  }
  return loadJsonFallback(memoryDir);
}

function loadJsonFallback(memoryDir) {
  const p = getJsonPath(memoryDir);
  try {
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return null;
  }
}

export function saveNetwork(memoryDir, net) {
  if (!SUPPORTS_SQLITE) {
    writeJsonAtomic(getJsonPath(memoryDir), JSON.stringify(net, null, 2));
    return 'json';
  }
  let db = null;
  try {
    db = openDb(memoryDir, true);
    db.exec('BEGIN');
    db.exec('DELETE FROM meta; DELETE FROM edges; DELETE FROM dendrites; DELETE FROM neurons;');

    const insMeta = db.prepare('INSERT INTO meta (key, value) VALUES (?, ?)');
    for (const k of ['version', 'architecture', 'updatedAt', 'metrics']) {
      if (net[k] !== undefined) insMeta.run(k, JSON.stringify(net[k]));
    }

    const insN = db.prepare(`
      INSERT INTO neurons (id, code, type, community, title, summary, echo, concepts_json, tags_json, dim4_json, coord3d_json, weight, active_flag, is_private, ptr, activations)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insE = db.prepare('INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)');
    const insD = db.prepare('INSERT INTO dendrites (neuron_id, token, weight) VALUES (?, ?, ?)');

    for (const n of Object.values(net.neurons || {})) {
      insN.run(
        n.id, n.code, n.type, n.community, n.title, n.summary,
        n.echo || null,
        n.concepts?.length ? JSON.stringify(n.concepts) : null,
        JSON.stringify(n.tags || []),
        JSON.stringify(n.dim4 || {}),
        JSON.stringify(n.coord3d || {}),
        Number(n.weight || 0.65),
        n.valid ? 1 : 0,
        n.private ? 1 : 0,
        n.ptr || '',
        Number(n.activations || 0)
      );
      for (const [target, w] of Object.entries(n.edges || {})) {
        insE.run(n.id, target, Number(w));
      }
      for (const [tok, w] of Object.entries(n.dendrites || {})) {
        insD.run(n.id, tok, Number(w));
      }
    }
    db.exec('COMMIT');
    return 'node:sqlite';
  } catch (err) {
    try { db?.exec('ROLLBACK'); } catch { }
    writeJsonAtomic(getJsonPath(memoryDir), JSON.stringify(net, null, 2));
    return 'json';
  } finally {
    try { db?.close(); } catch { }
  }
}

/**
 * O(1) Incremental Hebbian Persistence — updates only fired neurons, edges, and dendrites.
 */
export function applyHebbianDelta(memoryDir, delta = {}) {
  if (!SUPPORTS_SQLITE) return false;
  const dbPath = getSqlitePath(memoryDir);
  if (!fs.existsSync(dbPath)) return false;

  const { activations = [], edges = [], dendrites = [], updatedAt = new Date().toISOString() } = delta;
  if (!activations.length && !edges.length && !dendrites.length) return true;

  let db = null;
  try {
    db = openDb(memoryDir, false);
    db.exec('BEGIN');
    const upAct = db.prepare('UPDATE neurons SET activations = ? WHERE id = ?');
    for (const a of activations) upAct.run(Number(a.activations), a.id);

    const upEdge = db.prepare('INSERT OR REPLACE INTO edges (source, target, weight) VALUES (?, ?, ?)');
    for (const e of edges) upEdge.run(e.source, e.target, Number(e.weight));

    const upDend = db.prepare('INSERT OR REPLACE INTO dendrites (neuron_id, token, weight) VALUES (?, ?, ?)');
    for (const d of dendrites) upDend.run(d.neuronId, d.token, Number(d.weight));

    db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)').run('updatedAt', JSON.stringify(updatedAt));
    db.exec('COMMIT');
    return true;
  } catch {
    try { db?.exec('ROLLBACK'); } catch { }
    return false;
  } finally {
    try { db?.close(); } catch { }
  }
}
