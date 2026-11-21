// Run with: node test.js
const assert = require("assert");
const { LogReg } = require("./model.js");
let s = 7; const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;

// two separable blobs
const blobs = [];
for (let i = 0; i < 40; i++) blobs.push({ p: [-0.5 + (rand() - .5) * .4, -0.4 + (rand() - .5) * .4], label: 0 },
                                        { p: [0.5 + (rand() - .5) * .4, 0.4 + (rand() - .5) * .4], label: 1 });
const m = new LogReg();
const before = m.loss(blobs);
for (let i = 0; i < 300; i++) m.step(blobs);
assert.ok(Math.abs(before - Math.log(2)) < 1e-9, "starts at 50/50 guess");
assert.ok(m.loss(blobs) < 0.1 && m.accuracy(blobs) === 1, "learns separable data");

// loss never goes up with a small learning rate
const m2 = new LogReg(); let prev = Infinity;
for (let i = 0; i < 100; i++) { const l = m2.loss(blobs); assert.ok(l <= prev + 1e-12); prev = l; m2.step(blobs, 0.3); }

// a ring (class 1 inside a circle) needs the curved model
const ring = [];
for (let i = 0; i < 120; i++) { const x = rand() * 2 - 1, y = rand() * 2 - 1; ring.push({ p: [x, y], label: x * x + y * y < 0.35 ? 1 : 0 }); }
const straight = new LogReg(false), curved = new LogReg(true);
for (let i = 0; i < 3000; i++) { straight.step(ring, 1); curved.step(ring, 1); }
console.log(`  ring: straight line ${(straight.accuracy(ring) * 100).toFixed(0)}% vs curved ${(curved.accuracy(ring) * 100).toFixed(0)}%`);
assert.ok(curved.accuracy(ring) > 0.9 && curved.accuracy(ring) > straight.accuracy(ring) + 0.15);
console.log("All 4 tests passed ✓");
