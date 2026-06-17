import init, { WasmAbalone, WasmLinearRegression, wasm_grid_search } from "../../../pkg/rust_ai.js";
import { fetchCSV } from "../../utils/fetch.js";
import { ABALONE_URL } from "../../datasets/abalone.js";

// ── Constants ──────────────────────────────────────────────────────────────
const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };
const W = 480, H = 320;
const IW = W - MARGIN.left - MARGIN.right;
const IH = H - MARGIN.top - MARGIN.bottom;

// ── Grid ───────────────────────────────────────────────────────────────────
// Single source of truth for the λ search space.
// Reads logMin, logMax, and num from the grid search controls — the user owns the range.
// Generates `num` values evenly spaced on a log scale, plus λ=0 as a no-regularization baseline.
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

  setStatus(`ready — ${abalone.sample_count()} samples`);
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

  // Run in next tick so the UI updates before the heavy Rust call
  setTimeout(() => {
    const model   = new WasmLinearRegression(lr, regType, lambda, epochs);
    const result  = model.fit(abalone);
    const predictions = model.predictions(abalone);
    const weights     = model.weights();

    drawR2(result.r2_train, result.r2_val);
    drawScatter(predictions);
    drawWeights(weights);

    const finalR2 = result.r2_val[result.r2_val.length - 1];
    setStatus(`done — val R² ${finalR2.toFixed(3)}`);
    document.getElementById("train-btn").disabled = false;
  }, 10);
});

// ── Search & train ─────────────────────────────────────────────────────────
// Single source of truth for the combined workflow:
//   1. Grid search finds the best λ via 5-fold CV
//   2. That λ — and only that λ — is used to train the final model
// The manual train button above is for free exploration; this button is for
// the correct ML workflow where hyperparameters are chosen before final training.
document.getElementById("search-btn").addEventListener("click", () => {
  if (!abalone) return;

  // Read all hyperparameters from the controls — same source as the train button,
  // except λ comes from the grid search result, not the slider
  const regType = document.getElementById("reg-type").value;
  const lr      = getLr();
  const epochs  = +epochsEl.value;

  setStatus("running grid search…");
  document.getElementById("search-btn").disabled = true;
  document.getElementById("train-btn").disabled  = true;

  setTimeout(() => {
    // Step 1: grid search — evaluates λ values via 5-fold CV using fewer epochs than final training
    // cv epochs are intentionally lower: CV only needs to compare λ values, not fully converge
    const cvEpochs = +document.getElementById("cv-epochs").value;
    const results = wasm_grid_search(abalone, regType, 5, lr, cvEpochs, makeGrid());
    drawGridSearch(results);

    // Step 2: pick the λ with the lowest cross-validation MSE
    const best = results.reduce((a, b) => a.rmse < b.rmse ? a : b);
    setStatus(`grid search done — best λ=${best.lambda.toExponential(2)} · retraining…`);

    // Step 3: retrain the final model on the full dataset using the best λ
    // (CV was only for selecting λ — the final model uses all available data)
    setTimeout(() => {
      const model  = new WasmLinearRegression(lr, regType, best.lambda, epochs);
      const result = model.fit(abalone);

      drawR2(result.r2_train, result.r2_val);
      drawScatter(model.predictions(abalone));
      drawWeights(model.weights());

      const finalR2 = result.r2_val[result.r2_val.length - 1];
      setStatus(
        `done — best λ=${best.lambda.toExponential(2)} · val R² ${finalR2.toFixed(3)}`
      );
      document.getElementById("search-btn").disabled = false;
      document.getElementById("train-btn").disabled  = false;
    }, 10);
  }, 10);
});

// ── Charts ─────────────────────────────────────────────────────────────────

// Creates an SVG with a <g> translated to the inner plot area.
// Accepts an optional margin override — used by charts that need wider label space.
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

function drawR2(r2Train, r2Val) {
  const { g, iw, ih } = svgBase("r2-chart");

  const x = d3.scaleLinear().domain([0, r2Train.length - 1]).range([0, iw]);

  // Clip y-axis to [-0.5, 1] — early epochs start very negative but that region
  // is uninformative. clamp(true) pins out-of-range values to the axis edges
  // so the lines don't disappear; they just enter from the bottom.
  const allR2 = [...r2Train, ...r2Val].filter(v => isFinite(v));
  const yMin = Math.max(d3.min(allR2), -0.5);
  const y = d3.scaleLinear().domain([yMin, 1]).range([ih, 0]).clamp(true);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("epoch");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("R²");

  const line = d3.line().defined(d => isFinite(d));

  // Training R² — blue
  g.append("path")
    .datum(r2Train)
    .attr("fill", "none")
    .attr("stroke", "var(--accent)")
    .attr("stroke-width", 2)
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  // Validation R² — orange dashed
  g.append("path")
    .datum(r2Val)
    .attr("fill", "none")
    .attr("stroke", "var(--accent-highlight, #f5a623)")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,3")
    .attr("d", line.x((_, i) => x(i)).y(d => y(d)));

  // Legend
  const legend = g.append("g").attr("transform", `translate(${iw - 120}, 10)`);
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 0).attr("y2", 0)
    .attr("stroke", "var(--accent)").attr("stroke-width", 2);
  legend.append("text").attr("x", 25).attr("y", 4).attr("class", "axis-label").text("train");
  legend.append("line").attr("x1", 0).attr("x2", 20).attr("y1", 16).attr("y2", 16)
    .attr("stroke", "var(--accent-highlight, #f5a623)").attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,3");
  legend.append("text").attr("x", 25).attr("y", 20).attr("class", "axis-label").text("val");
}

function drawScatter(predictions) {
  const { g, iw, ih } = svgBase("scatter-chart");

  const allVals = predictions.flatMap(d => [d.actual, d.predicted]);
  const ext = d3.extent(allVals);
  const pad = (ext[1] - ext[0]) * 0.05;
  const domain = [ext[0] - pad, ext[1] + pad];

  const x = d3.scaleLinear().domain(domain).range([0, iw]);
  const y = d3.scaleLinear().domain(domain).range([ih, 0]);

  g.append("line")
    .attr("x1", x(domain[0])).attr("y1", y(domain[0]))
    .attr("x2", x(domain[1])).attr("y2", y(domain[1]))
    .attr("stroke", "#aaa").attr("stroke-dasharray", "4,3").attr("stroke-width", 1);

  g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("actual rings");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("predicted rings");

  const sample = predictions.length > 800
    ? predictions.filter((_, i) => i % Math.ceil(predictions.length / 800) === 0)
    : predictions;

  g.selectAll("circle")
    .data(sample)
    .enter()
    .append("circle")
    .attr("cx", d => x(d.actual))
    .attr("cy", d => y(d.predicted))
    .attr("r", 2.5)
    .attr("fill", "var(--accent)")
    .attr("fill-opacity", 0.45);
}

function drawWeights(weights) {
  // weights: [{name, weight}, …], sorted by |weight| descending
  const sorted = [...weights].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  // Wider left margin so full feature names like "shucked_weight" aren't clipped
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

  // Zero line
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

function drawGridSearch(results) {
  const { g, iw, ih } = svgBase("grid-chart");

  const sorted = [...results].sort((a, b) => a.lambda - b.lambda);
  const best   = sorted.reduce((a, b) => a.rmse < b.rmse ? a : b);

  // λ=0 can't live on a log scale — replace with a small proxy value for positioning only
  const lambdas = sorted.map(d => d.lambda === 0 ? 1e-5 : d.lambda);
  const x = d3.scaleLog().domain([d3.min(lambdas), d3.max(lambdas)]).range([0, iw]);
  const y = d3.scaleLinear()
    .domain([d3.min(sorted, d => d.rmse) * 0.98, d3.max(sorted, d => d.rmse) * 1.02])
    .range([ih, 0]);

  g.append("g").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(8, ".0e"));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", iw / 2).attr("y", ih + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("λ");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -ih / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("validation rmse");

  const line = d3.line().x((d, i) => x(lambdas[i])).y(d => y(d.rmse));
  g.append("path")
    .datum(sorted)
    .attr("fill", "none")
    .attr("stroke", "var(--accent)")
    .attr("stroke-width", 2)
    .attr("d", line);

  g.selectAll("circle")
    .data(sorted)
    .enter()
    .append("circle")
    .attr("cx", (d, i) => x(lambdas[i]))
    .attr("cy", d => y(d.rmse))
    .attr("r", 5)
    .attr("fill", d => d === best ? "var(--accent-highlight, #f5a623)" : "var(--accent)")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  const bi = sorted.indexOf(best);
  g.append("text")
    .attr("x", x(lambdas[bi]) + 8)
    .attr("y", y(best.mse) - 6)
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
