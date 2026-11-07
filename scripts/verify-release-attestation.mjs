// Verify that a snapshot's release attestation matches the repository.
//
// A manifest cannot record the hash of the commit that contains it, so the release is
// split across two commits: the engine commit holds the code and profile, and the next
// commit changes only this manifest's `engine_commit` field. This script proves that
// shape, which is what makes the recorded hash trustworthy.
//
// Checks, in order:
//   1. The manifest names a release tag for its own id and revision.
//   2. `engine_commit` is a full hash that exists in the repository.
//   3. The tag exists and points at an annotated tag object.
//   4. The tagged commit touches this manifest and nothing else.
//   5. Going from the engine commit to the tagged commit changes only `engine_commit`.
//
// Usage: node scripts/verify-release-attestation.mjs [ruleset-id]

import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL('../', import.meta.url);
const rulesetId = process.argv[2] ?? 'augment_chess_20261102';
const manifestPath = `rulesets/${rulesetId}/manifest.json`;

const git = (...args) =>
  execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();

const fail = message => {
  console.error(`FAIL ${message}`);
  process.exit(1);
};
const pass = message => console.log(`ok   ${message}`);

const manifest = JSON.parse(readFileSync(new URL(manifestPath, root), 'utf8'));
const release = manifest.release;
if (!release) fail(`${manifestPath} has no release block`);

// 1. The tag must name this snapshot and revision.
if (release.ruleset_tag !== `ruleset/${rulesetId}-${release.revision}`) {
  fail(`tag ${release.ruleset_tag} does not match ${rulesetId} ${release.revision}`);
}
pass(`tag name ${release.ruleset_tag}`);

// 2. The engine commit must be a real, full hash.
const engineCommit = release.engine_commit;
if (!/^[0-9a-f]{40}$/.test(engineCommit ?? '')) {
  fail(`engine_commit is not a full hash: ${JSON.stringify(engineCommit)}`);
}
try {
  git('cat-file', '-e', `${engineCommit}^{commit}`);
} catch {
  fail(`engine_commit ${engineCommit} is not a commit in this repository`);
}
pass(`engine commit ${engineCommit.slice(0, 12)} exists`);

// 3. The tag must exist and be annotated, not lightweight.
let tagType;
try {
  tagType = git('cat-file', '-t', `refs/tags/${release.ruleset_tag}`);
} catch {
  fail(`tag ${release.ruleset_tag} does not exist`);
}
if (tagType !== 'tag') fail(`tag ${release.ruleset_tag} is ${tagType}, expected an annotated tag`);
pass(`annotated tag exists`);
const taggedCommit = git('rev-list', '-n', '1', release.ruleset_tag);

// 4. The tagged commit must be a manifest-only change.
const changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', taggedCommit)
  .split('\n')
  .filter(Boolean);
if (changed.length !== 1 || changed[0] !== manifestPath) {
  fail(`tagged commit ${taggedCommit.slice(0, 12)} changed [${changed.join(', ')}], expected only ${manifestPath}`);
}
pass(`tagged commit changes only ${manifestPath}`);

// 5. The only difference from the engine commit must be the recorded hash.
const diff = git('diff', engineCommit, taggedCommit, '--', manifestPath);
if (!diff) fail('tagged manifest is identical to the engine commit manifest');
const before = JSON.parse(git('show', `${engineCommit}:${manifestPath}`));
const onlyCommitChanged =
  JSON.stringify({ ...before, release: { ...before.release, engine_commit: null } })
  === JSON.stringify({ ...manifest, release: { ...manifest.release, engine_commit: null } });
if (!onlyCommitChanged) {
  fail('the tagged manifest differs from the engine commit manifest by more than engine_commit');
}
pass('engine_commit is the only field that changed');

// The recorded hash must name the commit that actually holds the released state.
const parent = git('rev-parse', `${taggedCommit}^`);
if (parent !== engineCommit) {
  fail(`engine_commit ${engineCommit} is not the parent of the tagged commit ${taggedCommit}`);
}
pass('engine_commit is the tagged commit parent');

console.log(`\nRelease attestation verified: ${release.ruleset_tag}`);
