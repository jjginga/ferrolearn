import init, { WasmAbalone, WasmLogisticRegression, wasm_grid_search_logistic, wasm_cv_weights_logistic } from "../../../pkg/rust_ai.js";
import { fetchCSV } from "../../utils/fetch.js";
import { ABALONE_URL } from "../../datasets/abalone.js";

// ── Constants ──────────────────────────────────────────────────────────────
const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };
const W = 480, H = 320;
const IW = W - MARGIN.left - MARGIN.right;
const IH = H - MARGIN.top - MARGIN.bottom;

// ── Grid ───────────────────────────────────────────────────────────────────
// Single source of truth for the λ search space.
function makeGrid() {
  const logMin = +document.getElementById("grid-log-min").value;
  const logMax = +document.getElementById("grid-log-max").value;
  const num    = +document.getElementById("grid-steps").value;
  const vals = Array.from({ length: num }, (_, i) =>
    Math.pow(10, logMin + (logMax - logMin) * i / (num - 1))
  );
  return new Float64Array([0, ...vals]);  // λ=0 prepended as the no-regularization baseline
}

// Wire up grid control displays
["grid-log-min", "grid-log-max", "grid-steps", "cv-epochs"].forEach(id => {
  const el = document.getElementById(id);
  const display = document.getElementById(`${id}-val`);
  el.addEventListener("input", () => { display.textContent = el.value; });
});

// ── State ──────────────────────────────────────────────────────────────────
let abalone = null;

// ── Boot ───────────────────────────────────────────────────────────────────
async function main() {
  await init();
  setStatus("loading dataset…");

  const csv = await fetchCSV(ABALONE_URL);
  abalone = new WasmAbalone(csv);

  setStatus(`ready — ${abalone.sample_count()} samples (infants excluded from classification)`);
  document.getElementById("train-btn").disabled = false;
  document.getElementById("search-btn").disabled = false;
}

// ── Controls ───────────────────────────────────────────────────────────────
function logSlider(id, displayId) {
  const el = document.getElementById(id);
  const display = document.getElementById(displayId);
  const update = () => { display.textContent = Math.pow(10, +el.value).toExponential(2); };
  el.addEventListener("input", update);
  update();
  return () => Math.pow(10, +el.value);
}

const getLambda = logSlider("lambda", "lambda-val");
const getLr     = logSlider("lr", "lr-val");

const epochsEl = document.getElementById("epochs");
const epochsDisplay = document.getElementById("epochs-val");
epochsEl.addEventListener("input", () => { epochsDisplay.textContent = epochsEl.value; });

// ── Train ──────────────────────────────────────────────────────────────────
document.getElementById("train-btn").addEventListener("click", () => {
  if (!abalone) return;

  const regType = document.getElementById("reg-type").value;
  const lambda  = getLambda();
  const lr      = getLr();
  const epochs  = +epochsEl.value;

  setStatus("training…");
  document.getElementById("train-btn").disabled = true;

  setTimeout(() => {
    const model  = new WasmLogisticRegression(lr, regType, lambda, epochs);
    const result = model.fit(abalone);

    drawLoss(result.loss_train, result.loss_val);
    drawAccuracy(result.acc_train, result.acc_val);
    drawProbabilityDist(model.predictions(abalone));
    drawWeights(model.weights());

    const finalAcc = result.acc_val[result.acc_val.length - 1];
    setStatus(`done — val accuracy ${(finalAcc * 100).toFixed(1)}%`);
    document.getElementById("train-btn").disabled = false;
  }, 10);
});

// ── Search & train ─────────────────────────────────────────────────────────
document.getElementById("search-btn").addEventListener("click", () => {
  if (!abalone) return;

  const regType = document.getElementById("reg-type").value;
  const lr      = getLr();
  const epochs  = +epochsEl.value;

  setStatus("running grid search…");
  document.getElementById("search-btn").disabled = true;
  document.getElementById("train-btn").disabled  = true;

  setTimeout(() => {
    const cvEpochs = +document.getElementById("cv-epochs").value;
    const results  = wasm_grid_search_logistic(abalone, regType, 5, lr, cvEpochs, makeGrid());
    drawGridSearch(results);

    // pick the λ with the highest cross-validation accuracy
    const best = results.reduce((a, b) => a.score > b.score ? a : b);
    setStatus(`grid search done — best λ=${best.lambda.toExponential(2)} · retraining…`);

    setTimeout(() => {
      const model  = new WasmLogisticRegression(lr, regType, best.lambda, epochs);
      const result = model.fit(abalone);

      drawLoss(result.loss_train, result.loss_val);
      drawAccuracy(result.acc_train, result.acc_val);
      drawProbabilityDist(model.predictions(abalone));
      drawWeights(model.weights());

      // Weight stability: run k-fold CV with the best lambda and plot weight spread across folds
      const foldWeights = wasm_cv_weights_logistic(abalone, regType, 5, lr, cvEpochs, best.lambda);
      drawWeightStability(foldWeights);

      const finalAcc = result.acc_val[result.acc_val.length - 1];
      setStatus(
        `done — best λ=${best.lambda.toExponential(2)} · val accuracy ${(finalAcc * 100).toFixed(1)}%`
      );
      document.getElementById("search-btn").disabled = false;
      document.getElementById("train-btn").disabled  = false;
    }, 10);
  }, 10);
});

// ── Charts ─────────────────────────────────────────────────────────────────

function svgBase(containerId, margin = MARGIN) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  return {
    g:  d3.select(container)
          .append("svg")
          .attr("width", W)
          .attr("height", H)
          .append("g")
          .attr("transform", `translate(${margin.left},${margin.top})`),
    iw: W - margin.left - margin.right,
    ih: H - margin.top  - margin.bottom,
  };
}

function drawLoss(lossTrain, lossVal) {
  const { g, iw, ih } = svgBase("loss-chart");

  const x = d3.scaleLinear().domain([0, lossTrain.length - 1]).range([0, iw]);
  const allLoss = [...lossTrain, ...lossVal].filter(v => isFinite(v));
  const y = d3.scaleLinear()
    .domain([d3.min(allLoss) * 0.95, d3.max(allLoss) * 1.05])
    .range([ih, 0]);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("epoch");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("binary cross-entropy");

  const line = d3.line().defined(d => isFinite(d));

  g.append("path").datum(lossTrain)
    .attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2)
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  g.append("path").datum(lossVal)
    .attr("fill", "none").attr("stroke", "var(--accent-highlight, #f5a623)")
    .attr("stroke-width", 2).attr("stroke-dasharray", "5,3")
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  const legend = g.append("g").attr("transform", `translate(${iw - 120}, ${ih - 40})`);
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 0).attr("y2", 0)
    .attr("stroke", "var(--accent)").attr("stroke-width", 2);
  legend.append("text").attr("x", 25).attr("y", 4).attr("class", "axis-label").text("train");
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 16).attr("y2", 16)
    .attr("stroke", "var(--accent-highlight, #f5a623)").attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,3");
  legend.append("text").attr("x", 25).attr("y", 20).attr("class", "axis-label").text("val");
}

function drawAccuracy(accTrain, accVal) {
  const { g, iw, ih } = svgBase("acc-chart");

  const x = d3.scaleLinear().domain([0, accTrain.length - 1]).range([0, iw]);
  const allAcc = [...accTrain, ...accVal].filter(v => isFinite(v));
  const yMin = Math.max(0, d3.min(allAcc) - 0.05);
  const y = d3.scaleLinear().domain([yMin, 1]).range([ih, 0]);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6));
  g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("epoch");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("accuracy");

  const line = d3.line().defined(d => isFinite(d));

  g.append("path").datum(accTrain)
    .attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2)
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  g.append("path").datum(accVal)
    .attr("fill", "none").attr("stroke", "var(--accent-highlight, #f5a623)")
    .attr("stroke-width", 2).attr("stroke-dasharray", "5,3")
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  const legend = g.append("g").attr("transform", `translate(${iw - 120}, 10)`);
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 0).attr("y2", 0)
    .attr("stroke", "var(--accent)").attr("stroke-width", 2);
  legend.append("text").attr("x", 25).attr("y", 4).attr("class", "axis-label").text("train");
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 16).attr("y2", 16)
    .attr("stroke", "var(--accent-highlight, #f5a623)").attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,3");
  legend.append("text").attr("x", 25).attr("y", 20).attr("class", "axis-label").text("val");
}

function drawProbabilityDist(predictions) {
  // Strip plot: x = P(male), two rows by actual sex (actual=1 → M, actual=0 → F).
  // Shows how well the model separates the two classes.
  const { g, iw, ih } = svgBase("prob-chart", { ...MARGIN, left: 50 });

  const classes = ["M", "F"];
  const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
  const y = d3.scaleBand().domain(classes).range([0, ih]).padding(0.3);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".1f")));
  g.append("g").call(d3.axisLeft(y));

  // Decision boundary at 0.5
  g.append("line")
    .attr("x1", x(0.5)).attr("y1", 0)
    .attr("x2", x(0.5)).attr("y2", ih)
    .attr("stroke", "#aaa").attr("stroke-dasharray", "4,3").attr("stroke-width", 1.5);

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("P(male)");

  // Downsample if large
  const sample = predictions.length > 600
    ? predictions.filter((_, i) => i % Math.ceil(predictions.length / 600) === 0)
    : predictions;

  // Seed-based jitter so dots are spread vertically within each band
  const bw = y.bandwidth();
  sample.forEach((d, i) => {
    const label = d.actual >= 0.5 ? "M" : "F";
    const jitter = (((i * 17 + 31) % 100) / 100 - 0.5) * bw * 0.8;
    g.append("circle")
      .attr("cx", x(d.predicted))
      .attr("cy", y(label) + bw / 2 + jitter)
      .attr("r", 2)
      .attr("fill", label === "M" ? "var(--accent)" : "var(--accent-neg, #e05252)")
      .attr("fill-opacity", 0.4);
  });

  // Legend
  const legend = g.append("g").attr("transform", `translate(${iw - 90}, 4)`);
  legend.append("circle").attr("cx", 6).attr("cy", 6).attr("r", 4)
    .attr("fill", "var(--accent)").attr("fill-opacity", 0.7);
  legend.append("text").attr("x", 14).attr("y", 10).attr("class", "axis-label").text("actual M");
  legend.append("circle").attr("cx", 6).attr("cy", 22).attr("r", 4)
    .attr("fill", "var(--accent-neg, #e05252)").attr("fill-opacity", 0.7);
  legend.append("text").attr("x", 14).attr("y", 26).attr("class", "axis-label").text("actual F");
}

function drawWeights(weights) {
  const sorted = [...weights].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const { g, iw, ih } = svgBase("weights-chart", { ...MARGIN, left: 110 });

  const x = d3.scaleLinear()
    .domain([d3.min(sorted, d => d.weight) * 1.1, d3.max(sorted, d => d.weight) * 1.1])
    .range([0, iw]);
  const y = d3.scaleBand()
    .domain(sorted.map(d => d.name))
    .range([0, ih])
    .padding(0.25);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").call(d3.axisLeft(y));

  g.append("line")
    .attr("x1", x(0)).attr("y1", 0)
    .attr("x2", x(0)).attr("y2", ih)
    .attr("stroke", "#aaa").attr("stroke-width", 1);

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("weight value");

  g.selectAll("rect")
    .data(sorted)
    .enter()
    .append("rect")
    .attr("x", d => d.weight >= 0 ? x(0) : x(d.weight))
    .attr("y", d => y(d.name))
    .attr("width", d => Math.abs(x(d.weight) - x(0)))
    .attr("height", y.bandwidth())
    .attr("fill", d => d.weight >= 0 ? "var(--accent)" : "var(--accent-neg, #e05252)");
}

function drawWeightStability(foldWeights) {
  const byFeature = d3.group(foldWeights, d => d.name);
  const stats = Array.from(byFeature, ([name, vals]) => {
    const ws = vals.map(d => d.weight).sort(d3.ascending);
    return {
      name,
      min:    ws[0],
      q1:     d3.quantile(ws, 0.25),
      median: d3.quantile(ws, 0.5),
      q3:     d3.quantile(ws, 0.75),
      max:    ws[ws.length - 1],
      values: ws,
    };
  }).sort((a, b) => Math.abs(b.median) - Math.abs(a.median));

  const { g, iw, ih } = svgBase("stability-chart", { ...MARGIN, left: 110 });

  const allVals = foldWeights.map(d => d.weight);
  const ext = d3.extent(allVals);
  const pad = (ext[1] - ext[0]) * 0.1;
  const x = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([0, iw]);
  const y = d3.scaleBand().domain(stats.map(d => d.name)).range([0, ih]).padding(0.3);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").call(d3.axisLeft(y));

  g.append("line")
    .attr("x1", x(0)).attr("y1", 0)
    .attr("x2", x(0)).attr("y2", ih)
    .attr("stroke", "#aaa").attr("stroke-width", 1);

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("weight value");

  const bw = y.bandwidth();
  stats.forEach(d => {
    const cy = y(d.name) + bw / 2;
    g.append("line")
      .attr("x1", x(d.min)).attr("x2", x(d.max))
      .attr("y1", cy).attr("y2", cy)
      .attr("stroke", "#888").attr("stroke-width", 1.5);
    g.append("rect")
      .attr("x", x(Math.min(d.q1, d.q3)))
      .attr("y", y(d.name) + bw * 0.15)
      .attr("width", Math.abs(x(d.q3) - x(d.q1)))
      .attr("height", bw * 0.7)
      .attr("fill", "var(--accent)").attr("fill-opacity", 0.3)
      .attr("stroke", "var(--accent)").attr("stroke-width", 1.5);
    g.append("line")
      .attr("x1", x(d.median)).attr("x2", x(d.median))
      .attr("y1", y(d.name) + bw * 0.15).attr("y2", y(d.name) + bw * 0.85)
      .attr("stroke", "var(--accent)").attr("stroke-width", 2.5);
    d.values.forEach(v => {
      g.append("circle")
        .attr("cx", x(v)).attr("cy", cy).attr("r", 3)
        .attr("fill", "var(--accent)").attr("fill-opacity", 0.7);
    });
  });
}

function drawGridSearch(results) {
  const { g, iw, ih } = svgBase("grid-chart");

  const sorted = [...results].sort((a, b) => a.lambda - b.lambda);
  // best λ = highest validation accuracy
  const best = sorted.reduce((a, b) => a.score > b.score ? a : b);

  // λ=0 can't live on a log scale — replace with a small proxy value for positioning only
  const lambdas = sorted.map(d => d.lambda === 0 ? 1e-8 : d.lambda);
  const x = d3.scaleLog().domain([d3.min(lambdas), d3.max(lambdas)]).range([0, iw]);
  const allScores = sorted.map(d => d.score);
  const y = d3.scaleLinear()
    .domain([d3.min(allScores) * 0.98, Math.min(1, d3.max(allScores) * 1.02)])
    .range([ih, 0]);

  g.append("g").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(8, ".0e"));
  g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("λ");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("validation accuracy");

  const line = d3.line().x((d, i) => x(lambdas[i])).y(d => y(d.score));
  g.append("path")
    .datum(sorted)
    .attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2)
    .attr("d", line);

  g.selectAll("circle")
    .data(sorted)
    .enter()
    .append("circle")
    .attr("cx", (d, i) => x(lambdas[i]))
    .attr("cy", d => y(d.score))
    .attr("r", 5)
    .attr("fill", d => d === best ? "var(--accent-highlight, #f5a623)" : "var(--accent)")
    .attr("stroke", "#fff").attr("stroke-width", 1.5);

  const bi = sorted.indexOf(best);
  g.append("text")
    .attr("x", x(lambdas[bi]) + 8)
    .attr("y", y(best.score) - 6)
    .attr("class", "axis-label")
    .text(`best: λ=${best.lambda === 0 ? "0" : best.lambda.toExponential(1)}`);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

// ── Go ─────────────────────────────────────────────────────────────────────
main().catch(err => {
  console.error(err);
  setStatus("error: " + err.message);
});
