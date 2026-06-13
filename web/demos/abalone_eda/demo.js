import init, { WasmAbalone } from "../../../pkg/rust_ai.js";
import { fetchCSV } from "../../utils/fetch.js";
import { ABALONE_URL } from "../../datasets/abalone.js";

function renderRingsHistogram(rings) {
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width  = 500 - margin.left - margin.right;
    const height = 300 - margin.top  - margin.bottom;

    const svg = d3.select("#histograms")
        .append("h2").text("Target Distribution — Rings (Age proxy)")
        .select(function() { return this.parentNode; })
        .append("svg")
        .attr("width",  width  + margin.left + margin.right)
        .attr("height", height + margin.top  + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // X scale — domain from min to max rings value
    const x = d3.scaleLinear()
        .domain(d3.extent(rings))
        .range([0, width]);

    // Bin the data into a histogram
    const bins = d3.bin()
        .domain(x.domain())
        .thresholds(x.ticks(20))(rings);

    // Y scale — domain from 0 to tallest bin
    const y = d3.scaleLinear()
        .domain([0, d3.max(bins, d => d.length)])
        .range([height, 0]);

    // Draw bars
    svg.selectAll("rect")
        .data(bins)
        .join("rect")
        .attr("x",      d => x(d.x0) + 1)
        .attr("width",  d => Math.max(0, x(d.x1) - x(d.x0) - 1))
        .attr("y",      d => y(d.length))
        .attr("height", d => height - y(d.length))
        .attr("fill", "steelblue");

    // Axes
    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(10));

    svg.append("g")
        .call(d3.axisLeft(y));

    // Axis labels
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height + 35)
        .attr("text-anchor", "middle")
        .text("Rings");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("text-anchor", "middle")
        .text("Count");
}

function renderFeatureHistograms(abalone) {
    const features = [
        "length", "diameter", "height",
        "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"
    ];

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width  = 300 - margin.left - margin.right;
    const height = 200 - margin.top  - margin.bottom;

    const container = d3.select("#histograms")
        .append("h2").text("Feature Distributions")
        .select(function() { return this.parentNode; })
        .append("div")
        .style("display", "flex")
        .style("flex-wrap", "wrap")
        .style("gap", "20px");

    for (const feature of features) {
        const data = abalone.get_column(feature);

        const svg = container.append("svg")
            .attr("width",  width  + margin.left + margin.right)
            .attr("height", height + margin.top  + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(data))
            .range([0, width]);

        const bins = d3.bin()
            .domain(x.domain())
            .thresholds(x.ticks(15))(data);

        const y = d3.scaleLinear()
            .domain([0, d3.max(bins, d => d.length)])
            .range([height, 0]);

        svg.selectAll("rect")
            .data(bins)
            .join("rect")
            .attr("x",      d => x(d.x0) + 1)
            .attr("width",  d => Math.max(0, x(d.x1) - x(d.x0) - 1))
            .attr("y",      d => y(d.length))
            .attr("height", d => height - y(d.length))
            .attr("fill", "steelblue");

        svg.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5));

        svg.append("g")
            .call(d3.axisLeft(y).ticks(5));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + 35)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(feature);
    }
}

function renderCorrelationMatrix(corr) {
    const { names, matrix } = corr;
    const n = names.length;

    const margin = { top: 80, right: 30, bottom: 30, left: 100 };
    const cellSize = 60;
    const width  = n * cellSize;
    const height = n * cellSize;

    const svg = d3.select("#correlation")
        .append("h2").text("Correlation Matrix")
        .select(function() { return this.parentNode; })
        .append("svg")
        .attr("width",  width  + margin.left + margin.right)
        .attr("height", height + margin.top  + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Color scale: -1 = blue, 0 = white, 1 = red
    const color = d3.scaleDiverging()
        .domain([-1, 0, 1])
        .interpolator(d3.interpolateRdBu)
        .clamp(true);

    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = matrix[i * n + j];
            svg.append("rect")
                .attr("x", j * cellSize)
                .attr("y", i * cellSize)
                .attr("width",  cellSize)
                .attr("height", cellSize)
                .attr("fill", color(val));

            svg.append("text")
                .attr("x", j * cellSize + cellSize / 2)
                .attr("y", i * cellSize + cellSize / 2 + 4)
                .attr("text-anchor", "middle")
                .style("font-size", "10px")
                .style("fill", Math.abs(val) > 0.5 ? "white" : "black")
                .text(val.toFixed(2));
        }
    }

    // Row labels
    svg.selectAll(".row-label")
        .data(names)
        .join("text")
        .attr("x", -5)
        .attr("y", (_, i) => i * cellSize + cellSize / 2 + 4)
        .attr("text-anchor", "end")
        .style("font-size", "11px")
        .text(d => d);

    // Column labels
    svg.selectAll(".col-label")
        .data(names)
        .join("text")
        .attr("x", (_, i) => i * cellSize + cellSize / 2)
        .attr("y", -10)
        .attr("text-anchor", "start")
        .attr("transform", (_, i) => `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`)
        .style("font-size", "11px")
        .text(d => d);
}

function renderBoxplotsBySex(groups) {
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width  = 400 - margin.left - margin.right;
    const height = 300 - margin.top  - margin.bottom;

    const svg = d3.select("#boxplots")
        .append("h2").text("Rings by Sex")
        .select(function() { return this.parentNode; })
        .append("svg")
        .attr("width",  width  + margin.left + margin.right)
        .attr("height", height + margin.top  + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
        .domain(groups.map(g => g.sex))
        .range([0, width])
        .padding(0.4);

    const y = d3.scaleLinear()
        .domain([0, d3.max(groups, g => g.mean_rings + g.std_rings * 2)])
        .range([height, 0]);

    const colorMap = { M: "steelblue", F: "tomato", I: "seagreen" };

    // Draw mean bar + std error bar per group
    for (const g of groups) {
        const cx = x(g.sex) + x.bandwidth() / 2;

        // Bar for mean
        svg.append("rect")
            .attr("x",      x(g.sex))
            .attr("width",  x.bandwidth())
            .attr("y",      y(g.mean_rings))
            .attr("height", height - y(g.mean_rings))
            .attr("fill",   colorMap[g.sex])
            .attr("opacity", 0.7);

        // Error bar (±1 std)
        svg.append("line")
            .attr("x1", cx).attr("x2", cx)
            .attr("y1", y(g.mean_rings + g.std_rings))
            .attr("y2", y(g.mean_rings - g.std_rings))
            .attr("stroke", "black")
            .attr("stroke-width", 2);

        // Count label
        svg.append("text")
            .attr("x", cx)
            .attr("y", y(g.mean_rings) - 5)
            .attr("text-anchor", "middle")
            .style("font-size", "11px")
            .text(`n=${g.count}`);
    }

    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height + 35)
        .attr("text-anchor", "middle")
        .text("Sex");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("text-anchor", "middle")
        .text("Mean Rings ± 1 std");
}

function renderScatterBySex(abalone) {
    const length = abalone.get_column("length");
    const rings  = abalone.get_rings();
    const sex    = abalone.get_sex();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width  = 500 - margin.left - margin.right;
    const height = 350 - margin.top  - margin.bottom;

    const svg = d3.select("#scatter-sex")
        .append("h2").text("Length vs Rings — colored by Sex")
        .select(function() { return this.parentNode; })
        .append("svg")
        .attr("width",  width  + margin.left + margin.right)
        .attr("height", height + margin.top  + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain(d3.extent(length)).range([0, width]);
    const y = d3.scaleLinear().domain(d3.extent(rings)).range([height, 0]);
    const colorMap = { M: "steelblue", F: "tomato", I: "seagreen" };

    // Combine into array of points
    const points = length.map((l, i) => ({ l, r: rings[i], s: sex[i] }));

    svg.selectAll("circle")
        .data(points)
        .join("circle")
        .attr("cx",    d => x(d.l))
        .attr("cy",    d => y(d.r))
        .attr("r",     2.5)
        .attr("fill",  d => colorMap[d.s] || "gray")
        .attr("opacity", 0.5);

    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(8));

    svg.append("g")
        .call(d3.axisLeft(y).ticks(8));

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height + 35)
        .attr("text-anchor", "middle")
        .text("Length (mm)");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("text-anchor", "middle")
        .text("Rings");

    // Legend
    const legend = svg.append("g").attr("transform", `translate(${width - 80}, 10)`);
    Object.entries(colorMap).forEach(([s, c], i) => {
        legend.append("circle").attr("cx", 0).attr("cy", i * 18).attr("r", 5).attr("fill", c);
        legend.append("text").attr("x", 10).attr("y", i * 18 + 4).style("font-size", "11px").text(s === "M" ? "Male" : s === "F" ? "Female" : "Infant");
    });
}

async function main() {
    const status = document.getElementById("status");

    // 1. Load WASM
    await init();

    // 2. Fetch CSV
    status.textContent = "Fetching dataset...";
    const csv = await fetchCSV(ABALONE_URL);

    // 3. Parse once
    const abalone = new WasmAbalone(csv);
    status.textContent = `Loaded ${abalone.sample_count()} samples. Rendering...`;

    // 4. Get data from Rust
    const stats  = abalone.summary_stats();
    const groups = abalone.group_stats();
    const corr   = abalone.correlation_matrix();
    const rings  = abalone.get_rings();

    // 5. Render plots (to be implemented)
    console.log("stats",  stats);
    console.log("groups", groups);
    console.log("corr",   corr);
    console.log("rings",  rings);

    status.textContent = "Done.";

    renderRingsHistogram(rings);
    renderFeatureHistograms(abalone);
    renderCorrelationMatrix(corr);
    renderBoxplotsBySex(groups);
    renderScatterBySex(abalone);
}

main().catch(err => {
    document.getElementById("status").textContent = "Error: " + err.message;
    console.error(err);
});