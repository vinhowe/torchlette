const fs = require('fs');
const path = require('path');
const rootDir = path.resolve(__dirname, '..');
const { TraceMap, originalPositionFor } = require(path.join(rootDir, 'node_modules/.pnpm/@jridgewell+trace-mapping@0.3.30/node_modules/@jridgewell/trace-mapping'));

const profilePath = process.argv[2];
const profile = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
const rawMap = JSON.parse(fs.readFileSync('/tmp/profile-bundle.mjs.map', 'utf8'));
const map = new TraceMap(rawMap);

const nodeMap = new Map();
for (const node of profile.nodes) nodeMap.set(node.id, node);

const samples = profile.samples;
const deltas = profile.timeDeltas;

const startIdx = Math.floor(samples.length * 0.70);
const endIdx = Math.floor(samples.length * 0.95);

const selfTime = new Map();
for (let i = startIdx; i < endIdx; i++) {
  const node = nodeMap.get(samples[i]);
  if (!node) continue;
  const cf = node.callFrame;

  let key;
  if (cf.url && cf.url.includes('profile-bundle.mjs')) {
    const orig = originalPositionFor(map, { line: cf.lineNumber + 1, column: cf.columnNumber });
    if (orig.source) {
      const file = orig.source.replace(/.*\//, '');
      const name = orig.name || cf.functionName || '(anon)';
      key = name + ' @ ' + file + ':' + orig.line;
    } else {
      key = (cf.functionName || '(anon)') + ' @ bundle:' + cf.lineNumber;
    }
  } else {
    const file = (cf.url || '(native)').replace(/.*\//,'');
    key = (cf.functionName || '(anon)') + ' @ ' + file + ':' + (cf.lineNumber + 1);
  }
  selfTime.set(key, (selfTime.get(key) || 0) + deltas[i]);
}

const sorted = [...selfTime.entries()].sort((a, b) => b[1] - a[1]);
let totalTime = 0;
for (const [_, t] of sorted) totalTime += t;

console.log('Top 50 functions by self-time (steady-state, source-mapped):');
console.log('─'.repeat(110));
let cum = 0;
for (let i = 0; i < 50 && i < sorted.length; i++) {
  const [key, time] = sorted[i];
  cum += time;
  console.log(
    (time/1000).toFixed(1).padStart(8) + 'ms (' +
    (time/totalTime*100).toFixed(1).padStart(5) + '%, cum ' +
    (cum/totalTime*100).toFixed(1).padStart(5) + '%)  ' + key
  );
}
console.log('\nTotal sampled: ' + (totalTime/1000).toFixed(0) + 'ms');
