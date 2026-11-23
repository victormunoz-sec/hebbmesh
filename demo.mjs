import path from 'path';
import { fileURLToPath } from 'url';
import { compileNetwork, activateNetwork, findBridge } from './src/engine.mjs';
import { startViewerServer } from './src/viewer-server.mjs';

/**
 * HebbMesh 60-Second Cold-Start Demo
 * Usage:
 *   node demo.mjs                    (runs built-in synthetic demo on 127.0.0.1:4322)
 *   node demo.mjs ./my-memory        (opens 3D Viewer on your own Markdown folder)
 *   node demo.mjs --no-server        (headless CI / smoke check)
 */

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cliArgs = process.argv.slice(2);
const noServer = cliArgs.includes('--no-server');
const customDir = cliArgs.find((a) => !a.startsWith('--'));
const memoryDir = customDir ? path.resolve(process.cwd(), customDir) : path.resolve(__dirname, 'demo', 'memory');

const net = compileNetwork(memoryDir, { fresh: true });
console.log(`\n[HebbMesh] Compiled ${net.metrics.totalNeurons} neurons & ${net.metrics.totalEdges} edges in ${net.metrics.compileMs} ms (0 external deps)`);

const spark = activateNetwork(memoryDir, 'what is Alex furry pet companion', { topK: 4 });
console.log(`\n[Zero-Embedding Metaphor Spark] "${spark.query}" (${spark.ms} ms · Lobe: ${spark.dominantLobe.toUpperCase()})`);
for (const w of spark.winners) {
  console.log(`   - [${w.code}] (${w.energy}) ${w.title}  [${w.via}]`);
}

const bridge = findBridge(net, 'stripe webhook', 'cloudflare waf');
console.log(
  `\n[3D A* Synaptic Bridge] "stripe webhook" -> "cloudflare waf" (${bridge.ms} ms):`,
  bridge.found ? bridge.path.map((p) => p.code).join(' -> ') : bridge.reason
);

if (!noServer) {
  startViewerServer(memoryDir, Number(process.env.PORT || 4322));
}
