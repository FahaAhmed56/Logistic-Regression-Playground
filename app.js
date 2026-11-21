const $ = id => document.getElementById(id);
const plot = $("plot"), g = plot.getContext("2d"), S = plot.width;
let points = [], model = new LogReg(false), steps = 0, losses = [], running = null;
const toWorld = (px, py) => [px / S * 2 - 1, 1 - py / S * 2];          // canvas -> [-1, 1]
const toPx = ([x, y]) => [(x + 1) / 2 * S, (1 - y) / 2 * S];

function draw() {
  // background: the model's probability at each spot (blue = class 0, orange = class 1)
  const R = 46, img = g.createImageData(R, R);
  for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
    const p = model.prob(toWorld((j + .5) * S / R, (i + .5) * S / R)), k = (i * R + j) * 4;
    img.data[k] = 59 + p * (249 - 59); img.data[k + 1] = 130 + p * (115 - 130); img.data[k + 2] = 246 + p * (22 - 246); img.data[k + 3] = 70;
  }
  const tmp = document.createElement("canvas"); tmp.width = tmp.height = R; tmp.getContext("2d").putImageData(img, 0, 0);
  g.fillStyle = "#0b1020"; g.fillRect(0, 0, S, S);
  g.imageSmoothingEnabled = true; g.drawImage(tmp, 0, 0, S, S);
  // the decision boundary: where probability = 0.5
  g.fillStyle = "#fff";
  for (let px = 0; px < S; px += 3) for (let py = 0; py < S; py += 3) {
    const a = model.prob(toWorld(px, py)) > .5, b = model.prob(toWorld(px + 3, py)) > .5, c = model.prob(toWorld(px, py + 3)) > .5;
    if (a !== b || a !== c) g.fillRect(px, py, 2, 2);
  }
  for (const { p, label } of points) {
    const [x, y] = toPx(p);
    g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2);
    g.fillStyle = label ? "#f97316" : "#3b82f6"; g.fill(); g.lineWidth = 2; g.strokeStyle = "#fff"; g.stroke();
  }
  const n = points.length;
  $("step").textContent = steps;
  $("loss").textContent = n ? model.loss(points).toFixed(3) : "–";
  $("acc").textContent = n ? `${Math.round(model.accuracy(points) * 100)}%` : "–";
  // honest yardstick: accuracy of always guessing the more common colour
  const ones = points.filter(q => q.label).length, base = n ? Math.max(ones, n - ones) / n : 0;
  const acc = n ? model.accuracy(points) : 0;
  $("base").innerHTML = !n ? "" : `Guessing the more common colour every time scores <b>${Math.round(base * 100)}%</b>` +
    (steps > 0 && acc <= base + 1e-9 ? " – so this model has learned <b>nothing useful</b> yet." : steps > 0 ? ` – the model beats that by ${Math.round((acc - base) * 100)} points.` : ".");
  $("weights").textContent = `Weights: [${model.w.map(w => w.toFixed(2)).join(", ")}]`;
  drawCurve();
}

function drawCurve() {
  const c = $("curve"), k = c.getContext("2d"), W = c.width, H = c.height;
  k.clearRect(0, 0, W, H);
  k.strokeStyle = "#334155"; k.beginPath(); k.moveTo(30, 8); k.lineTo(30, H - 18); k.lineTo(W - 4, H - 18); k.stroke();
  k.fillStyle = "#94a3b8"; k.font = "10px system-ui"; k.fillText("loss", 2, 14); k.fillText("training steps →", W - 90, H - 4);
  if (losses.length < 2) return;
  const max = Math.max(...losses);
  k.strokeStyle = "#22c55e"; k.lineWidth = 2; k.beginPath();
  losses.forEach((l, i) => { const x = 30 + i / (losses.length - 1) * (W - 36), y = 8 + (1 - l / max) * (H - 28); i ? k.lineTo(x, y) : k.moveTo(x, y); });
  k.stroke(); k.lineWidth = 1;
}

function train() {
  if (running) { stop(); return; }
  if (!points.length || new Set(points.map(p => p.label)).size < 2) { alert("Add points of both colours first."); return; }
  $("train").textContent = "⏸ Pause";
  running = setInterval(() => {
    for (let i = 0; i < 5; i++) { model.step(points, +$("lr").value); steps++; }
    losses.push(model.loss(points));
    draw();
    if (steps >= 3000) stop();
  }, 30);
}
function stop() { clearInterval(running); running = null; $("train").textContent = "▶ Train"; }
function resetModel() { stop(); model = new LogReg($("curved").checked); steps = 0; losses = []; draw(); }

plot.addEventListener("contextmenu", e => e.preventDefault());
plot.addEventListener("mousedown", e => {
  const r = plot.getBoundingClientRect();
  points.push({ p: toWorld((e.clientX - r.left) * S / r.width, (e.clientY - r.top) * S / r.height), label: e.button === 2 || e.shiftKey ? 1 : 0 });
  draw();
});
$("train").addEventListener("click", train);
$("reset").addEventListener("click", resetModel);
$("clear").addEventListener("click", () => { points = []; resetModel(); });
$("curved").addEventListener("change", resetModel);
$("lr").addEventListener("input", () => ($("lrOut").textContent = $("lr").value));
document.querySelectorAll("[data-demo]").forEach(b => b.addEventListener("click", () => {
  points = [];
  for (let i = 0; i < 70; i++) {
    if (b.dataset.demo === "blobs") {
      const c = i % 2, r = () => (Math.random() - .5) * .7;
      points.push({ p: [(c ? .4 : -.4) + r(), (c ? .3 : -.35) + r()], label: c });
    } else {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
      points.push({ p: [x, y], label: x * x + y * y < .3 ? 1 : 0 });
    }
  }
  resetModel();
}));
draw();
