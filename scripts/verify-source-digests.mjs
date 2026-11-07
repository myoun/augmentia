// Verify that every staged upstream bundle still matches its manifest sha256.
//
// Git may normalize line endings on add, which silently changes the bytes and
// breaks provenance. This reads the index copy through git and compares it to
// each ruleset's source-manifest.json.
//
// Usage: node scripts/verify-source-digests.mjs

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const root = new URL('../', import.meta.url);
const read = path => readFileSync(new URL(path, root), 'utf8');

// Discover rulesets from the index (not HEAD) so newly staged snapshots are
// covered before their first commit.
const rulesetDirs = execFileSync('git', ['ls-files', '--stage', 'rulesets/'], {
  encoding: 'utf8',
})
  .split(/\r?\n/)
  .filter(Boolean)
  .map(line => line.split('\t')[1])
  .filter(path => path.startsWith('rulesets/'))
  .map(path => path.split('/')[1])
  .filter(name => name && name !== '');

const uniqueRulesets = [...new Set(rulesetDirs)].sort();

let failures = 0;
let checked = 0;

for (const ruleset of uniqueRulesets) {
  const manifestPath = `rulesets/${ruleset}/sources/source-manifest.json`;
  let manifest;
  try {
    manifest = JSON.parse(read(manifestPath));
  } catch {
    console.log(`${ruleset}: no readable source-manifest.json, skipped`);
    continue;
  }

  const entries = [];
  if (Array.isArray(manifest.files)) {
    entries.push(...manifest.files);
  }
  if (manifest.bundle?.path) entries.push(manifest.bundle);

  for (const entry of entries) {
    if (!entry.path || !entry.sha256) continue;
    const repoPath = `rulesets/${ruleset}/${entry.path}`;
    // Read the staged (index) bytes so the check covers what would be committed.
    let bytes;
    try {
      bytes = execFileSync('git', ['cat-file', 'blob', `:${repoPath}`], {
        maxBuffer: 512 * 1024 * 1024,
      });
    } catch {
      console.log(`${ruleset}: ${entry.path} not staged, skipped`);
      continue;
    }
    checked += 1;
    const digest = createHash('sha256').update(bytes).digest('hex');
    const sizeOk = bytes.length === entry.byte_length;
    const hashOk = digest === entry.sha256;
    if (sizeOk && hashOk) {
      console.log(`ok   ${ruleset} ${entry.path} (${bytes.length} bytes)`);
    } else {
      failures += 1;
      console.log(`FAIL ${ruleset} ${entry.path}`);
      console.log(`       size ${bytes.length} vs manifest ${entry.byte_length}`);
      console.log(`       sha  ${digest}`);
      console.log(`       want ${entry.sha256}`);
    }
  }
}

console.log('');
console.log(`checked ${checked} captured files, ${failures} failure(s)`);
process.exit(failures === 0 ? 0 : 1);

