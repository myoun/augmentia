// Offline, no browser/DOM/server: structure checks and isolated UI helpers.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
const base = new URL('../crates/augment-app/src/web/static/', import.meta.url);
const html = readFileSync(new URL('index.html', base), 'utf8');
const css = readFileSync(new URL('styles.css', base), 'utf8');
const js = readFileSync(new URL('app.js', base), 'utf8');
const allIds = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(allIds.length, new Set(allIds).size, 'duplicate HTML IDs');
for (const id of ['firstNode','previousNode','nextNode','lastNode','flipBoardButton','blackCardRow','whiteCardRow','infoPopover','setupClose']) assert.ok(allIds.includes(id), id);
assert.ok(!/<header\b/.test(html), 'no global header');
assert.ok(!html.includes('ambient'), 'no legacy ornamentation');
assert.match(css, /html, body\s*\{[^}]*overflow:\s*hidden/);
assert.match(css, /\.workspace\s*\{[^}]*height:\s*100dvh/);
assert.match(css, /grid-template-rows:\s*44px minmax\(0, 1fr\) 44px 38px/);
assert.match(css, /\.card-list\s*\{[^}]*overflow-x:\s*auto;[^}]*overflow-y:\s*hidden/);
assert.match(css, /\.notation-tree\s*\{[^}]*min-height:\s*0;[^}]*overflow:\s*auto/);
assert.match(css, /\.board-column\.flipped #whiteCardRow\s*\{\s*grid-row:\s*1/);
assert.match(css, /\.board-column\.flipped #blackCardRow\s*\{\s*grid-row:\s*3/);
assert.match(js, /setTimeout\(\(\) => openInfoPopover\(anchor\), 300\)/);
assert.match(js, /registerInfo\(square,/);
assert.match(js, /registerInfo\(button, card.name/);
assert.match(js, /registerInfo\(tag,/);
assert.match(js, /const rank = boardFlipped \? displayRank : rows - 1 - displayRank/);
assert.match(js, /const file = boardFlipped \? cols - 1 - displayFile : displayFile/);
assert.match(js, /event.key === "F2"/);

const controls = new Map();
const popup = { style: {}, classList: { add(){}, remove(){}, contains(){return false;} }, getBoundingClientRect: () => ({width:340,height:400}) };
const context = vm.createContext({
  game:null, acgFrame:null, acgDocument:null, branchChoices:new Map(), busy:false, editor:null,
  popoverAnchor:null, popoverRect:null, popoverShowTimer:null, popoverHideTimer:null,
  elements:{actionDialog:{open:false}}, window:{innerWidth:1366,innerHeight:768},
  clearTimeout(){}, setTimeout(){},
  document:{getElementById(id){ if (id === 'infoPopover') return popup; if (!controls.has(id)) controls.set(id,{}); return controls.get(id); }},
}, {codeGeneration:{strings:false,wasm:false}});
const helpers = js.slice(js.indexOf('function registerInfo('), js.indexOf('\nasync function request('));
vm.runInContext(helpers, context, {timeout:250});
const nodes = [{id:0,parent:null},{id:5,parent:0},{id:10,parent:5},{id:11,parent:5},{id:12,parent:11},{id:20,parent:10}];
context.game = {notation:{nodes,cursor:12}};
context.rememberBranch(nodes,12);
const targets = () => JSON.parse(JSON.stringify(context.navigationTargets()));
assert.deepEqual(targets(), {first:0,previous:11,next:12,last:12,cursor:12});
context.game.notation.cursor = 5;
assert.equal(targets().next,11); assert.equal(targets().last,12);
context.rememberBranch(nodes,20);
assert.equal(targets().next,10); assert.equal(targets().last,20);
context.game.notation.cursor = 0;
assert.equal(targets().previous,0); assert.equal(targets().last,20);
context.renderNavigation();
assert.equal(controls.get('firstNode').disabled,true);
assert.equal(controls.get('nextNode').disabled,false);
context.busy = true; context.renderNavigation();
for (const id of ['firstNode','previousNode','nextNode','lastNode']) assert.equal(controls.get(id).disabled,true);
context.busy = false; context.editor = {};
context.renderNavigation(); assert.equal(controls.get('nextNode').disabled,true);
context.editor = null;
context.acgFrame = {}; context.acgDocument = {record:{replay:{events:Array(10)}}}; context.acgIndex = 4;
assert.deepEqual(targets(), {first:0,previous:3,next:5,last:10,cursor:4});
context.acgIndex = 10; context.renderNavigation(); assert.equal(controls.get('lastNode').disabled,true);
context.acgFrame = null; context.game = null; context.renderNavigation();
assert.equal(controls.get('nextNode').disabled,true);

for (const rect of [{left:1300,right:1330,top:700},{left:0,right:30,top:0},{left:600,right:650,top:400}]) {
  context.positionInfoPopover(rect);
  const x = parseFloat(popup.style.left), y = parseFloat(popup.style.top);
  assert.ok(x >= 12 && x + 340 <= 1354);
  assert.ok(y >= 12 && y + 400 <= 756);
}
const anchor = {dataset:{},attrs:{title:'old'},setAttribute(k,v){this.attrs[k]=v;},removeAttribute(k){delete this.attrs[k];}};
context.registerInfo(anchor,'카드','<not markup>');
assert.equal(anchor.dataset.infoDescription,'<not markup>');
assert.equal(anchor.attrs.title,undefined);
assert.equal(anchor.attrs['aria-haspopup'],'dialog');
// Regression: the next draft is rendered while busy, before the request's finally.
const actionButtons = [];
context.document.querySelectorAll = selector => { assert.equal(selector,'.decision-card, .card-button, .ability-button'); return actionButtons; };
context.document.body = {classList:{toggle(){}}};
context.elements.newGameButton = {};
context.renderSetupPicker = () => {};
context.renderSuggestions = () => {};
vm.runInContext(js.slice(js.indexOf('function setBusy('), js.indexOf('\nfunction setConnection(')), context, {timeout:250});
for (const phase of ['black opening draft','later draft','effect choice']) {
  context.setBusy(true);
  actionButtons.splice(0,actionButtons.length,...Array.from({length:3},()=>({disabled:true})));
  context.setBusy(false);
  assert.ok(actionButtons.every(button=>!button.disabled), `${phase} must unlock after request`);
  context.setBusy(true);
  assert.ok(actionButtons.every(button=>button.disabled), `${phase} must lock during request`);
}
context.editor = {};
context.setBusy(false);
assert.ok(actionButtons.every(button=>button.disabled),'editing must still block actions');
console.log('Workspace structure, branch navigation, flip wiring, popover bounds, and post-request action unlock passed (offline).');
