// Print a slice of a captured upstream bundle by 1-based line range.
//
// Card port notes and the `source_ranges` in cards/ported.json both use these
// line numbers, so this is the fastest way to check an upstream contract.
//
// Usage:
//   node scripts/source-slice.mjs <ruleset-id> <start> [end]
//   node scripts/source-slice.mjs <ruleset-id> --grep <regex> [--limit N]
//
// The bundle is located through the ruleset's sources/source-manifest.json, so
// no bundle filename is hardcoded and a new snapshot needs no code change.

import { readFileSync } from 'node:fs';

const root = new URL('../', import.meta.url);
const args = process.argv.slice(2);
const usage = `usage:
  node scripts/source-slice.mjs <ruleset-id> <start> [end]
  node scripts/source-slice.mjs <ruleset-id> --grep <regex> [--limit N]`;

const rulesetId = args[0];
if (!rulesetId || rulesetId.startsWith('--')) {
  console.error(usage);
  process.exit(2);
}

// Resolve the captured bundle from the ruleset's own source manifest so the
// correct file is used even when a snapshot ships several bundles. Two manifest
// shapes exist: newer snapshots list a `files` array, older ones a single
// `bundle` object.
function bundlePath() {
  const base = new URL(`rulesets/${rulesetId}/`, root);
  const manifest = JSON.parse(
    readFileSync(new URL('sources/source-manifest.json', base), 'utf8'),
  );
  const candidates = [];
  if (Array.isArray(manifest.files)) {
    candidates.push(...manifest.files.map(file => file.path));
  }
  if (manifest.bundle?.path) candidates.push(manifest.bundle.path);
  const bundle = candidates.find(path => path.includes('main') && path.endsWith('.js'));
  if (bundle) return new URL(bundle, base);
  // The oldest snapshot records only the upstream URL. The bundle still lives in
  // the snapshot under sources/raw/, so derive the path from the URL basename.
  const url = manifest.bundle?.url;
  if (url) {
    const name = url.split('/').pop();
    const raw = new URL(`sources/raw/${name}`, base);
    try {
      readFileSync(raw);
      return raw;
    } catch {
      // Fall through to the error below with the resolved name.
    }
  }
  throw new Error(
    `${rulesetId}: no captured bundle on disk. The manifest records only its URL and SHA-256, ` +
      `so only rulesets with sources/raw/main*.js can be sliced (currently 0915 and later).`,
  );
}

let bundle;
try {
  bundle = bundlePath();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}

const lines = readFileSync(bundle, 'utf8').split(/\r?\n/);
const grepAt = args.indexOf('--grep');
if (grepAt >= 0) {
  const pattern = new RegExp(args[grepAt + 1]);
  const limitAt = args.indexOf('--limit');
  const limit = limitAt >= 0 ? Number(args[limitAt + 1]) : Infinity;
  let shown = 0;
  for (let index = 0; index < lines.length && shown < limit; index += 1) {
    if (pattern.test(lines[index])) {
      console.log(`${index + 1}: ${lines[index].trim().slice(0, 200)}`);
      shown += 1;
    }
  }
  process.exit(shown > 0 ? 0 : 1);
}

const start = Number(args[1]);
const end = args[2] === undefined ? start : Number(args[2]);
if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) {
  console.error(usage);
  process.exit(2);
}

for (let n = start; n <= Math.min(end, lines.length); n += 1) {
  console.log(`${n}: ${lines[n - 1]}`);
}
