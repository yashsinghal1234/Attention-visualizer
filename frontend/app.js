/**
 * Attention Visualizer Client Application
 * Communicates with the FastAPI backend engine to visualize Multi-Head Attention.
 */

// Configuration and State
const API_BASE = window.location.origin.startsWith("http") ? "" : "http://localhost:8000";

let state = {
  text: "the cat sat on the mat",
  numHeads: 4,
  dModel: 64,
  causal: false,
  temperature: 1.0,
  activeHead: 0, // 0..numHeads-1, or -1 for Average
  viewMode: "heatmap", // 'heatmap' or 'arc'
  matrixType: "attention", // 'attention', 'scaled', 'raw'
  backendData: null,
  isBackendHealthy: false,
};

// Color scale for attention weights [0.0, 1.0]
function getAttentionColor(val, minVal = 0.0, maxVal = 1.0) {
  const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1e-6)));
  // Deep indigo -> Cyan -> Amber/Gold
  let r, g, b;
  if (norm < 0.5) {
    const u = norm * 2;
    r = Math.round(15 + u * 25);
    g = Math.round(20 + u * 190);
    b = Math.round(55 + u * 165);
  } else {
    const u = (norm - 0.5) * 2;
    r = Math.round(40 + u * 215);
    g = Math.round(210 - u * 60);
    b = Math.round(220 - u * 170);
  }
  return `rgb(${r}, ${g}, ${b})`;
}

// Check Backend Health
async function checkBackendHealth() {
  const badge = document.getElementById("status-badge");
  const statusText = document.getElementById("status-text");

  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      state.isBackendHealthy = true;
      badge.className = "status-badge";
      statusText.textContent = `Connected (${data.torch_version} • ${data.device})`;
      return true;
    }
  } catch (err) {
    // Backend offline
  }

  state.isBackendHealthy = false;
  badge.className = "status-badge error";
  statusText.textContent = "Backend Offline (Run: python run.py)";
  return false;
}

// Fetch Presets
async function loadPresets() {
  const container = document.getElementById("preset-container");
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    if (res.ok) {
      const presets = await res.json();
      container.innerHTML = "";
      presets.forEach(p => {
        const btn = document.createElement("button");
        btn.className = "preset-chip";
        btn.textContent = p.title;
        btn.title = p.description;
        btn.onclick = () => {
          document.getElementById("input-text").value = p.text;
          triggerCompute();
        };
        container.appendChild(btn);
      });
    }
  } catch (err) {
    console.warn("Using offline presets fallback");
  }
}

// Fetch Attention Computation from Backend
async function triggerCompute() {
  const textInput = document.getElementById("input-text").value.trim();
  if (!textInput) return;

  state.text = textInput;
  state.numHeads = parseInt(document.getElementById("select-heads").value, 10);
  state.dModel = parseInt(document.getElementById("select-dim").value, 10);
  state.causal = document.getElementById("toggle-causal").checked;
  state.temperature = parseFloat(document.getElementById("input-temp").value);

  // Reset activeHead if out of bounds
  if (state.activeHead >= state.numHeads) {
    state.activeHead = 0;
  }

  const payload = {
    text: state.text,
    num_heads: state.numHeads,
    d_model: state.dModel,
    causal: state.causal,
    temperature: state.temperature,
    seed: 42,
  };

  try {
    const res = await fetch(`${API_BASE}/api/attention`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`API Error: ${err.detail || "Failed to compute attention"}`);
      return;
    }

    state.backendData = await res.json();
    renderAll();
  } catch (err) {
    console.error("Backend compute failed:", err);
    // Render offline simulated calculation if backend not connected
    computeOfflineFallback();
  }
}

// Offline fallback computation if user directly opened HTML without python server running
function computeOfflineFallback() {
  const tokens = state.text.split(/\s+/).filter(Boolean);
  const n = tokens.length;
  const numHeads = state.numHeads;
  const heads = [];

  for (let h = 0; h < numHeads; h++) {
    const mat = [];
    for (let i = 0; i < n; i++) {
      let row = [];
      for (let j = 0; j < n; j++) {
        if (state.causal && j > i) {
          row.push(0);
        } else {
          // Simple deterministic hash
          const score = Math.sin(i * 3 + j * 7 + h * 13);
          row.push(Math.exp(score / state.temperature));
        }
      }
      const sum = row.reduce((a, b) => a + b, 0) || 1;
      mat.push(row.map(v => Math.round((v / sum) * 10000) / 10000));
    }

    heads.push({
      head_index: h,
      head_name: `Head ${h + 1}`,
      attention_matrix: mat,
      raw_scores: mat,
      scaled_scores: mat,
      entropy: mat.map(() => 1.5),
      top_edges: [],
    });
  }

  // Average
  const avg = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let h = 0; h < numHeads; h++) s += heads[h].attention_matrix[i][j];
      row.push(Math.round((s / numHeads) * 10000) / 10000);
    }
    avg.push(row);
  }

  state.backendData = {
    tokens,
    seq_len: n,
    num_heads: numHeads,
    d_model: state.dModel,
    d_k: state.dModel / numHeads,
    causal: state.causal,
    temperature: state.temperature,
    heads,
    average_attention: avg,
    token_importance: tokens.map((t, idx) => ({ token: t, index: idx, importance: 1 / n })),
    top_global_edges: [],
  };

  renderAll();
}

// Render Master Controller
function renderAll() {
  if (!state.backendData) return;
  renderHeadChips();
  renderCurrentView();
  renderSalience();
  renderSummaryBanner();
}

// Render Head Chips
function renderHeadChips() {
  const container = document.getElementById("head-chips-container");
  container.innerHTML = "";

  // Average chip
  const avgChip = document.createElement("div");
  avgChip.className = "chip" + (state.activeHead === -1 ? " active" : "");
  avgChip.textContent = "⚡ Head Average";
  avgChip.onclick = () => {
    state.activeHead = -1;
    renderHeadChips();
    renderCurrentView();
    renderSummaryBanner();
  };
  container.appendChild(avgChip);

  // Individual heads
  for (let h = 0; h < state.backendData.num_heads; h++) {
    const chip = document.createElement("div");
    chip.className = "chip" + (state.activeHead === h ? " active" : "");
    chip.textContent = `Head ${h + 1}`;
    chip.onclick = () => {
      state.activeHead = h;
      renderHeadChips();
      renderCurrentView();
      renderSummaryBanner();
    };
    container.appendChild(chip);
  }
}

// Render active view (Heatmap or Arc)
function renderCurrentView() {
  const heatmapView = document.getElementById("heatmap-view");
  const arcView = document.getElementById("arc-view");

  if (state.viewMode === "heatmap") {
    heatmapView.style.display = "block";
    arcView.style.display = "none";
    renderHeatmap();
  } else {
    heatmapView.style.display = "none";
    arcView.style.display = "flex";
    renderArcDiagram();
  }
}

// Render 2D Heatmap Grid
function renderHeatmap() {
  const root = document.getElementById("heatmap-view");
  root.innerHTML = "";

  const data = state.backendData;
  const tokens = data.tokens;
  const n = tokens.length;

  let matrix;
  if (state.activeHead === -1) {
    matrix = data.average_attention;
  } else {
    const h = data.heads[state.activeHead];
    if (state.matrixType === "scaled") matrix = h.scaled_scores;
    else if (state.matrixType === "raw") matrix = h.raw_scores;
    else matrix = h.attention_matrix;
  }

  // Determine min/max for color scaling
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const table = document.createElement("div");
  table.className = "hm-table";

  // Header row (Keys)
  const headerRow = document.createElement("div");
  headerRow.className = "hm-row";
  const spacer = document.createElement("div");
  spacer.className = "label-cell";
  spacer.textContent = "Query \\ Key";
  headerRow.appendChild(spacer);

  tokens.forEach(tok => {
    const th = document.createElement("div");
    th.className = "colheader";
    th.textContent = tok;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  // Rows (Queries)
  const tip = document.getElementById("tooltip");

  for (let i = 0; i < n; i++) {
    const row = document.createElement("div");
    row.className = "hm-row";

    const label = document.createElement("div");
    label.className = "label-cell";
    label.textContent = `${i}: ${tokens[i]}`;
    row.appendChild(label);

    for (let j = 0; j < n; j++) {
      const val = matrix[i][j];
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.style.background = getAttentionColor(val, minVal, maxVal);
      cell.textContent = val.toFixed(2);
      cell.dataset.i = i;
      cell.dataset.j = j;

      cell.addEventListener("mouseenter", () => {
        highlightMatrix(i, j);
        const headLabel = state.activeHead === -1 ? "Average" : `Head ${state.activeHead + 1}`;
        tip.innerHTML = `
          <strong>${headLabel}</strong><br/>
          Query [${i}]: <em>"${tokens[i]}"</em><br/>
          Key [${j}]: <em>"${tokens[j]}"</em><br/>
          Attention Weight: <strong>${val.toFixed(4)}</strong>
        `;
        tip.style.opacity = 1;
      });

      cell.addEventListener("mousemove", (e) => {
        tip.style.left = `${e.clientX + 14}px`;
        tip.style.top = `${e.clientY + 14}px`;
      });

      cell.addEventListener("mouseleave", () => {
        clearHighlights();
        tip.style.opacity = 0;
      });

      row.appendChild(cell);
    }
    table.appendChild(row);
  }

  root.appendChild(table);
}

// Highlight cell row and col
function highlightMatrix(rowIdx, colIdx) {
  document.querySelectorAll(".cell").forEach(c => {
    const i = +c.dataset.i, j = +c.dataset.j;
    c.classList.toggle("dim", !(i === rowIdx || j === colIdx));
    c.classList.toggle("hot", i === rowIdx && j === colIdx);
  });
}

function clearHighlights() {
  document.querySelectorAll(".cell").forEach(c => {
    c.classList.remove("dim", "hot");
  });
}

// Render SVG Arc Diagram View
function renderArcDiagram() {
  const svg = document.getElementById("arc-svg");
  const nodesContainer = document.getElementById("arc-nodes");
  svg.innerHTML = "";
  nodesContainer.innerHTML = "";

  const data = state.backendData;
  const tokens = data.tokens;
  const n = tokens.length;
  const matrix = state.activeHead === -1 ? data.average_attention : data.heads[state.activeHead].attention_matrix;

  const width = svg.clientWidth || 820;
  const height = 260;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const stepX = width / (n + 1);
  const baselineY = height - 20;

  // Render tokens
  tokens.forEach((tok, idx) => {
    const chip = document.createElement("div");
    chip.className = "node-chip";
    chip.textContent = tok;
    chip.dataset.idx = idx;
    nodesContainer.appendChild(chip);
  });

  // Render curved attention arcs
  const tip = document.getElementById("tooltip");

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const weight = matrix[i][j];
      if (weight < 0.08 || i === j) continue; // Skip negligible or self-loops for clarity

      const x1 = stepX * (i + 1);
      const x2 = stepX * (j + 1);
      const dist = Math.abs(x1 - x2);
      const arcHeight = Math.min(dist * 0.7, height - 40);

      // Quadratic bezier path
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = `M ${x1} ${baselineY} C ${x1} ${baselineY - arcHeight}, ${x2} ${baselineY - arcHeight}, ${x2} ${baselineY}`;
      path.setAttribute("d", d);
      path.setAttribute("class", "arc-path");
      path.setAttribute("stroke-width", Math.max(1, weight * 7));
      path.setAttribute("stroke-opacity", Math.min(1, weight + 0.15));

      path.addEventListener("mouseenter", () => {
        path.style.stroke = "var(--accent-cyan)";
        path.style.strokeWidth = `${weight * 10 + 2}px`;
        tip.innerHTML = `
          Flow: <strong>"${tokens[i]}"</strong> ➔ <strong>"${tokens[j]}"</strong><br/>
          Weight: <strong>${weight.toFixed(4)}</strong>
        `;
        tip.style.opacity = 1;
      });

      path.addEventListener("mousemove", (e) => {
        tip.style.left = `${e.clientX + 14}px`;
        tip.style.top = `${e.clientY + 14}px`;
      });

      path.addEventListener("mouseleave", () => {
        path.style.stroke = "";
        path.style.strokeWidth = "";
        tip.style.opacity = 0;
      });

      svg.appendChild(path);
    }
  }
}

// Render Token Importance Bars
function renderSalience() {
  const container = document.getElementById("salience-container");
  container.innerHTML = "";

  const items = state.backendData.token_importance || [];
  const maxImp = Math.max(...items.map(x => x.importance), 0.01);

  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "salience-item";

    const tokLabel = document.createElement("span");
    tokLabel.className = "salience-tok";
    tokLabel.textContent = item.token;

    const track = document.createElement("div");
    track.className = "salience-track";

    const fill = document.createElement("div");
    fill.className = "salience-fill";
    fill.style.width = `${(item.importance / maxImp) * 100}%`;

    const valLabel = document.createElement("span");
    valLabel.style.width = "40px";
    valLabel.style.fontSize = "11px";
    valLabel.style.color = "var(--text-muted)";
    valLabel.textContent = item.importance.toFixed(3);

    track.appendChild(fill);
    row.appendChild(tokLabel);
    row.appendChild(track);
    row.appendChild(valLabel);
    container.appendChild(row);
  });
}

// Render Summary Banner
function renderSummaryBanner() {
  const banner = document.getElementById("top-edges-banner");
  const data = state.backendData;
  if (!data) return;

  let edges;
  let title;
  if (state.activeHead === -1) {
    edges = data.top_global_edges || [];
    title = "Top Global Head-Averaged Attention";
  } else {
    edges = data.heads[state.activeHead].top_edges || [];
    title = `Top Attention Links in Head ${state.activeHead + 1}`;
  }

  const edgeStrings = edges.slice(0, 4).map(
    e => `<strong>${e.source_token}</strong> ➔ <strong>${e.target_token}</strong> (${e.weight.toFixed(3)})`
  );

  banner.innerHTML = `
    <span>${title}:</span> ${edgeStrings.join(" &nbsp;•&nbsp; ")}
  `;
}

// Setup Event Listeners
function setupEvents() {
  document.getElementById("btn-run").onclick = triggerCompute;

  // View switch buttons
  document.getElementById("tab-heatmap").onclick = (e) => {
    state.viewMode = "heatmap";
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    renderCurrentView();
  };

  document.getElementById("tab-arc").onclick = (e) => {
    state.viewMode = "arc";
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    renderCurrentView();
  };

  // Matrix type switch
  document.getElementById("select-matrix-type").onchange = (e) => {
    state.matrixType = e.target.value;
    if (state.viewMode === "heatmap") renderHeatmap();
  };

  // Temperature slider display
  const tempSlider = document.getElementById("input-temp");
  const tempVal = document.getElementById("temp-val");
  tempSlider.oninput = () => {
    tempVal.textContent = parseFloat(tempSlider.value).toFixed(1);
  };
  tempSlider.onchange = triggerCompute;

  // Head count & Dim select
  document.getElementById("select-heads").onchange = triggerCompute;
  document.getElementById("select-dim").onchange = triggerCompute;
  document.getElementById("toggle-causal").onchange = triggerCompute;

  // Enter in textarea to submit with Ctrl+Enter
  document.getElementById("input-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      triggerCompute();
    }
  });

  // Dynamic colorbar gradient
  const bar = document.getElementById("colorbar");
  let g = "linear-gradient(to right,";
  for (let i = 0; i <= 10; i++) {
    g += ` ${getAttentionColor(i / 10)} ${i * 10}%${i < 10 ? "," : ""}`;
  }
  g += ")";
  bar.style.background = g;
}

// Initialization
window.addEventListener("DOMContentLoaded", async () => {
  setupEvents();
  await checkBackendHealth();
  await loadPresets();
  await triggerCompute();
});

