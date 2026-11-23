import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  loadOrCompile,
  activateNetwork,
  findBridge,
} from './engine.mjs';

/**
 * HebbMesh Local 3D Viewer Server (0 external dependencies)
 * Security invariant: strictly bound to 127.0.0.1 with zero wildcard CORS headers.
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const viewerHtmlPath = path.resolve(__dirname, '..', 'viewer', 'index.html');

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(body);
}

function sanitizeNetworkForViewer(net) {
  const safeNeurons = {};
  for (const [id, n] of Object.entries(net.neurons || {})) {
    if (n.private) continue; // Privacy Firewall enforced at HTTP boundary
    const safeEdges = {};
    for (const [dst, w] of Object.entries(n.edges || {})) {
      if (net.neurons[dst] && !net.neurons[dst].private) safeEdges[dst] = w;
    }
    safeNeurons[id] = { ...n, edges: safeEdges };
  }
  return {
    version: net.version,
    updatedAt: net.updatedAt,
    metrics: net.metrics,
    neurons: safeNeurons,
  };
}

export function startViewerServer(memoryDir, port = 4322) {
  const host = '127.0.0.1';
  const server = http.createServer((req, res) => {
    const url = new URL(req.url || '/', `http://${host}:${port}`);

    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
      try {
        const html = fs.readFileSync(viewerHtmlPath, 'utf8');
        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'X-Content-Type-Options': 'nosniff',
          'Content-Security-Policy': "default-src 'self' 'unsafe-inline'; connect-src 'self'",
        });
        res.end(html);
      } catch (err) {
        sendJson(res, 500, { error: 'viewer_not_found' });
      }
      return;
    }

    if (url.pathname.startsWith('/api/')) {
      const fetchSite = String(req.headers['sec-fetch-site'] || '').toLowerCase();
      if (fetchSite === 'cross-site') {
        return sendJson(res, 403, { error: 'cross_site_blocked' });
      }
    }

    if (req.method === 'GET' && url.pathname === '/api/graph') {
      const net = loadOrCompile(memoryDir);
      sendJson(res, 200, sanitizeNetworkForViewer(net));
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/spark') {
      const q = (url.searchParams.get('q') || '').trim().slice(0, 240);
      if (!q) return sendJson(res, 400, { error: 'missing_query' });
      const { storagePath: _omitPath, ...safeSpark } = activateNetwork(memoryDir, q, {
        topK: 6, maxHops: 2, allowPrivate: false, reinforce: true,
      });
      sendJson(res, 200, safeSpark);
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/bridge') {
      const a = (url.searchParams.get('a') || '').trim().slice(0, 120);
      const b = (url.searchParams.get('b') || '').trim().slice(0, 120);
      if (!a || !b) return sendJson(res, 400, { error: 'missing_endpoints' });
      const net = loadOrCompile(memoryDir);
      const resBridge = findBridge(net, a, b, { allowPrivate: false });
      sendJson(res, 200, resBridge);
      return;
    }

    sendJson(res, 404, { error: 'not_found' });
  });

  server.listen(port, host, () => {
    console.log(`\n[HebbMesh 3D Viewer] Live at http://${host}:${port} (100% local, 0 telemetry)`);
    console.log('   Press Ctrl+C to stop.\n');
  });
  return server;
}
