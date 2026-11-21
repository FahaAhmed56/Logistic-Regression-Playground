// Logistic regression trained with gradient descent. No DOM, so it can be tested in Node.
// Optional "curved" mode adds x², y², xy features so the boundary can bend.
const sigmoid = z => 1 / (1 + Math.exp(-z));

function features([x, y], curved) {
  return curved ? [1, x, y, x * x, y * y, x * y] : [1, x, y];
}

class LogReg {
  constructor(curved = false) { this.curved = curved; this.w = new Array(curved ? 6 : 3).fill(0); }
  prob(p) { return sigmoid(features(p, this.curved).reduce((s, f, i) => s + f * this.w[i], 0)); }

  // Average cross-entropy loss over the points (lower = better)
  loss(points) {
    return points.reduce((s, { p, label }) => {
      const q = Math.min(1 - 1e-12, Math.max(1e-12, this.prob(p)));
      return s - (label ? Math.log(q) : Math.log(1 - q));
    }, 0) / points.length;
  }
  accuracy(points) { return points.filter(({ p, label }) => (this.prob(p) > 0.5) === !!label).length / points.length; }

  // One step of gradient descent: w -= lr * average of (prediction - label) * features
  step(points, lr = 0.5) {
    const grad = new Array(this.w.length).fill(0);
    for (const { p, label } of points) {
      const err = this.prob(p) - label, f = features(p, this.curved);
      f.forEach((v, i) => (grad[i] += err * v / points.length));
    }
    this.w = this.w.map((w, i) => w - lr * grad[i]);
  }
}

if (typeof module !== "undefined") module.exports = { LogReg, sigmoid, features };
