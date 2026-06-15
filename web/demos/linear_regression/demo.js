import init, { WasmAbalone, WasmLinearRegression, wasm_grid_search } from "../../../pkg/rust_ai.js";
import { fetchCSV } from "../../utils/fetch.js";
import { ABALONE_URL } from "../../datasets/abalone.js";

// ── Constants ──────────────────────────────────────────────────────────────
const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };
const W = 480, H = 320;
const IW = W - MARGIN.left - MARGIN.right;
const IH = H - MARGIN.top - MARGIN.bottom;

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
    const model = new WasmLinearRegression(lr, regType, lambda, epochs);
    const lossHistory = model.fit(abalone);   // Float64Array or JS Array
    const predictions = model.predictions(abalone);
    const weights     = model.weights();

    drawLoss(Array.from(lossHistory));
    drawScatter(predictions);
    drawWeights(weights);

    const finalMse = lossHistory[lossHistory.length - 1];
    setStatus(`done — final mse ${finalMse.toFixed(4)}`);
    document.getElementById("train-btn").disabled = false;
  }, 10);
});

// ── Grid search ────────────────────────────────────────────────────────────
document.getElementById("search-btn").addEventListener("click", () => {
  if (!abalone) return;

  const regType = document.getElementById("reg-type").value;
  const lr      = getLr();
  const epochs  = +epochsEl.value;

  setStatus("running grid search…");
  document.getElementById("search-btn").disabled = true;

  setTimeout(() => {
    const results = wasm_grid_search(abalone, regType, 5, lr, epochs);
    drawGridSearch(results);

    const best = results.reduce((a, b) => a.mse < b.mse ? a : b);
    setStatus(`grid search done — best λ=${best.lambda.toExponential(2)}, mse=${best.mse.toFixed(4)}`);
    document.getElementById("search-btn").disabled = false;
  }, 10);
});

// ── Charts ─────────────────────────────────────────────────────────────────

function svgBase(containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  return d3.select(container)
    .append("svg")
    .attr("width", W)
    .attr("height", H)
    .append("g")
    .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
}

function drawLoss(loss) {
  const g = svgBase("loss-chart");

  const x = d3.scaleLinear().domain([0, loss.length - 1]).range([0, IW]);
  const y = d3.scaleLinear().domain([0, d3.max(loss) * 1.05]).range([IH, 0]);

  // Axes
  g.append("g").attr("transform", `translate(0,${IH})`).call(d3.axisBottom(x).ticks(6));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  // Axis labels
  g.append("text").attr("x", IW / 2).attr("y", IH + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("epoch");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -IH / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("mse");

  // Line
  const line = d3.line().x((_, i) => x(i)).y(d => y(d));
  g.append("path")
    .datum(loss)
    .attr("fill", "none")
    .attr("stroke", "var(--accent)")
    .attr("stroke-width", 2)
    .attr("d", line);
}

function drawScatter(predictions) {
  // predictions: [{actual, predicted}, …]
  const g = svgBase("scatter-chart");

  const allVals = predictions.flatMap(d => [d.actual, d.predicted]);
  const ext = d3.extent(allVals);
  const pad = (ext[1] - ext[0]) * 0.05;
  const domain = [ext[0] - pad, ext[1] + pad];

  const x = d3.scaleLinear().domain(domain).range([0, IW]);
  const y = d3.scaleLinear().domain(domain).range([IH, 0]);

  // Diagonal reference line
  g.append("line")
    .attr("x1", x(domain[0])).attr("y1", y(domain[0]))
    .attr("x2", x(domain[1])).attr("y2", y(domain[1]))
    .attr("stroke", "#aaa").attr("stroke-dasharray", "4,3").attr("stroke-width", 1);

  // Axes
  g.append("g").attr("transform", `translate(0,${IH})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", IW / 2).attr("y", IH + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("actual rings");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -IH / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("predicted rings");

  // Dots — subsample to keep the DOM lean
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

  const g = svgBase("weights-chart");

  const x = d3.scaleLinear()
    .domain([d3.min(sorted, d => d.weight) * 1.1, d3.max(sorted, d => d.weight) * 1.1])
    .range([0, IW]);
  const y = d3.scaleBand()
    .domain(sorted.map(d => d.name))
    .range([0, IH])
    .padding(0.25);

  g.append("g").attr("transform", `translate(0,${IH})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").call(d3.axisLeft(y));

  // Zero line
  g.append("line")
    .attr("x1", x(0)).attr("y1", 0)
    .attr("x2", x(0)).attr("y2", IH)
    .attr("stroke", "#aaa").attr("stroke-width", 1);

  g.append("text").attr("x", IW / 2).attr("y", IH + 40)
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
  // results: [{lambda, mse}, …]
  const g = svgBase("grid-chart");

  const sorted = [...results].sort((a, b) => a.lambda - b.lambda);
  const best   = sorted.reduce((a, b) => a.mse < b.mse ? a : b);

  // Use log scale for lambda; skip lambda=0 (replace with tiny value)
  const lambdas = sorted.map(d => d.lambda === 0 ? 1e-5 : d.lambda);
  const xMin = d3.min(lambdas);
  const xMax = d3.max(lambdas);

  const x = d3.scaleLog().domain([xMin, xMax]).range([0, IW]);
  const y = d3.scaleLinear()
    .domain([d3.min(sorted, d => d.mse) * 0.98, d3.max(sorted, d => d.mse) * 1.02])
    .range([IH, 0]);

  g.append("g").attr("transform", `translate(0,${IH})`)
    .call(d3.axisBottom(x).ticks(sorted.length, ".0e"));
  g.append("g").call(d3.axisLeft(y).ticks(5));

  g.append("text").attr("x", IW / 2).attr("y", IH + 40)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("λ");
  g.append("text").attr("transform", "rotate(-90)")
    .attr("x", -IH / 2).attr("y", -45)
    .attr("text-anchor", "middle").attr("class", "axis-label").text("validation mse");

  // Line
  const line = d3.line()
    .x((d, i) => x(lambdas[i]))
    .y(d => y(d.mse));

  g.append("path")
    .datum(sorted)
    .attr("fill", "none")
    .attr("stroke", "var(--accent)")
    .attr("stroke-width", 2)
    .attr("d", line);

  // Dots
  g.selectAll("circle")
    .data(sorted)
    .enter()
    .append("circle")
    .attr("cx", (d, i) => x(lambdas[i]))
    .attr("cy", d => y(d.mse))
    .attr("r", 5)
    .attr("fill", d => d === best ? "var(--accent-highlight, #f5a623)" : "var(--accent)")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  // Best label
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
