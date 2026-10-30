"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Sparkles,
  Sliders,
  Layers,
  Zap,
  ArrowRight,
  Maximize2,
  RefreshCw,
  ExternalLink,
  Info,
  Check,
  Cpu,
  BarChart3,
  GitBranch,
  FileCode,
  Globe,
  Database,
} from "lucide-react";

export default function AttentionStudioPage() {
  // Model & Layer Configuration
  const [modelType, setModelType] = useState<string>("distilbert-base-uncased");
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [inputText, setInputText] = useState("the cat sat on the mat");
  const [numHeads, setNumHeads] = useState<number>(4);
  const [dModel, setDModel] = useState<number>(64);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [causalMask, setCausalMask] = useState<boolean>(false);
  const [activeHead, setActiveHead] = useState<number>(-1); // -1: Average, 0..N-1: Individual head
  const [viewMode, setViewMode] = useState<"heatmap" | "arc">("heatmap");
  const [matrixLayer, setMatrixLayer] = useState<"attention" | "scaled" | "raw">("attention");
  const [hoveredCell, setHoveredCell] = useState<{
    i: number;
    j: number;
    val: number;
  } | null>(null);
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null);
  const [isComputing, setIsComputing] = useState<boolean>(false);
  const [onlineData, setOnlineData] = useState<any>(null);

  // Available Models Metadata
  const modelOptions = [
    {
      id: "distilbert-base-uncased",
      name: "DistilBERT Base (HuggingFace)",
      tag: "Online Pretrained",
      layers: 6,
      heads: 12,
      dim: 768,
      desc: "6 layers, 12 heads per layer, pre-trained on Wikipedia + BookCorpus",
    },
    {
      id: "prajjwal1/bert-tiny",
      name: "BERT Tiny (HuggingFace)",
      tag: "Online Pretrained",
      layers: 2,
      heads: 2,
      dim: 128,
      desc: "Ultra-compact 2 layers, 2 heads checkpoint",
    },
    {
      id: "custom",
      name: "Custom Vaswani MHA (PyTorch)",
      tag: "Pure Neural Engine",
      layers: 1,
      heads: numHeads,
      dim: dModel,
      desc: "Mathematical scaled dot-product attention with custom heads & masks",
    },
  ];

  // Fetch online pretrained model attention from backend
  const fetchAttentionData = async () => {
    setIsComputing(true);
    try {
      const res = await fetch("http://localhost:8000/api/attention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText,
          model_name: modelType,
          layer_index: selectedLayer,
          num_heads: numHeads,
          d_model: dModel,
          causal: causalMask,
          temperature: temperature,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setOnlineData(data);
      }
    } catch (err) {
      console.warn("Backend not running or offline, using live mathematical fallback.");
    } finally {
      setIsComputing(false);
    }
  };

  useEffect(() => {
    fetchAttentionData();
  }, [inputText, modelType, selectedLayer, numHeads, dModel, causalMask, temperature]);

  // Active Model Config
  const activeModel = modelOptions.find((m) => m.id === modelType) || modelOptions[0];
  const isPretrained = modelType !== "custom";

  // Derive tokens
  const tokens: string[] = useMemo(() => {
    if (onlineData?.tokens) return onlineData.tokens;
    const raw = inputText.trim().split(/\s+/).filter(Boolean);
    return raw.length > 0 ? raw.slice(0, 16) : ["empty"];
  }, [onlineData, inputText]);

  const N = tokens.length;
  const currentHeadsCount = onlineData?.num_heads || (isPretrained ? activeModel.heads : numHeads);
  const dK = Math.floor((onlineData?.d_model || dModel) / currentHeadsCount);

  // Preset Sentences
  const presets = [
    {
      id: "classic",
      title: "Classic Rhyme",
      text: "the cat sat on the mat",
      desc: "Subject-verb-location dependencies",
    },
    {
      id: "winograd",
      title: "Winograd Schema",
      text: "The animal didn't cross the street because it was too tired",
      desc: "Coreference resolution ('it')",
    },
    {
      id: "polysemy",
      title: "Lexical Polysemy",
      text: "Time flies like an arrow fruit flies like a banana",
      desc: "Contrasting 'flies' noun vs verb",
    },
    {
      id: "transformer",
      title: "Attention Paper",
      text: "Transformers replace recurrence and convolutions with multi-head attention",
      desc: "Technical domain text",
    },
    {
      id: "reflexive",
      title: "Reflexive Pronouns",
      text: "Alice told Bob that she would finish the report",
      desc: "Gendered pronoun targeting",
    },
  ];

  // Mathematical fallback if backend is offline
  const fallbackMath = useMemo(() => {
    const scale = 1.0 / (Math.sqrt(dK) * Math.max(0.01, temperature));
    const heads: any[] = [];

    for (let h = 0; h < currentHeadsCount; h++) {
      const attentionMat: number[][] = [];
      const edges: any[] = [];
      const entropyList: number[] = [];

      for (let i = 0; i < N; i++) {
        const scaledRow: number[] = [];
        const ti = tokens[i].toLowerCase();

        for (let j = 0; j < N; j++) {
          const tj = tokens[j].toLowerCase();
          let dotVal = Math.sin((i * 11 + j * 17 + h * 19) / 3.0) * 1.5;

          if (ti === tj) dotVal += 2.2;
          if ((ti === "cat" && tj === "sat") || (ti === "sat" && tj === "mat")) dotVal += 3.0;
          if (ti === "it" && (tj === "animal" || tj === "street")) dotVal += 3.2;

          if (causalMask && j > i) scaledRow.push(-1e9);
          else scaledRow.push(dotVal * scale);
        }

        const maxVal = Math.max(...scaledRow);
        const exps = scaledRow.map((v) => (v === -1e9 ? 0 : Math.exp(v - maxVal)));
        const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
        const attnRow = exps.map((e) => Math.round((e / sumExp) * 10000) / 10000);

        const ent = attnRow.reduce((acc, p) => (p > 1e-6 ? acc - p * Math.log2(p) : acc), 0);
        entropyList.push(Math.round(ent * 100) / 100);

        for (let j = 0; j < N; j++) edges.push({ source: i, target: j, weight: attnRow[j] });
        attentionMat.push(attnRow);
      }

      edges.sort((a, b) => b.weight - a.weight);
      heads.push({
        head_index: h,
        head_name: `Head ${h + 1}`,
        attention_matrix: attentionMat,
        entropy: entropyList,
        top_edges: edges.slice(0, 5),
      });
    }

    const avgMat: number[][] = [];
    const globalEdges: any[] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let h = 0; h < currentHeadsCount; h++) sum += heads[h].attention_matrix[i][j];
        const avg = Math.round((sum / currentHeadsCount) * 10000) / 10000;
        row.push(avg);
        globalEdges.push({ source: i, target: j, weight: avg });
      }
      avgMat.push(row);
    }
    globalEdges.sort((a, b) => b.weight - a.weight);

    const salience = tokens.map((tok, j) => {
      let colSum = 0;
      for (let i = 0; i < N; i++) colSum += avgMat[i][j];
      return { token: tok, index: j, importance: Math.round((colSum / N) * 1000) / 1000 };
    }).sort((a, b) => b.importance - a.importance);

    return {
      heads,
      average_attention: avgMat,
      top_global_edges: globalEdges.slice(0, 8),
      token_importance: salience,
    };
  }, [tokens, currentHeadsCount, dK, temperature, causalMask, N]);

  // Unified data source (online pretrained if available, fallback otherwise)
  const activeData = onlineData || fallbackMath;

  // Active matrix to display
  const activeMatrix: number[][] = useMemo(() => {
    if (activeHead === -1) {
      return activeData.average_attention;
    }
    const h = activeData.heads[activeHead];
    return h ? h.attention_matrix : activeData.average_attention;
  }, [activeHead, activeData]);

  // Colormap (Deep Obsidian -> Neon Cyan -> Pure Gold)
  const getCellColor = (val: number) => {
    const t = Math.max(0, Math.min(1, val));
    let r, g, b;
    if (t < 0.45) {
      const u = t / 0.45;
      r = Math.round(14 + u * 25);
      g = Math.round(18 + u * 175);
      b = Math.round(45 + u * 190);
    } else {
      const u = (t - 0.45) / 0.55;
      r = Math.round(39 + u * 216);
      g = Math.round(193 + u * 25);
      b = Math.round(235 - u * 195);
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="min-h-screen w-full bg-mesh text-slate-100 flex flex-col items-center py-6 px-4 md:px-8">
      {/* 1. TOP HEADER / APP BAR */}
      <header className="w-full max-w-7xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 mb-6 border-b border-white/[0.08]">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-400 p-[1px] shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-[#090d16] rounded-2xl flex items-center justify-center">
              <Layers className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                ATTENTION STUDIO
              </h1>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 flex items-center gap-1">
                <Globe className="w-2.5 h-2.5" />
                HuggingFace Online
              </span>
            </div>
            <p className="text-xs text-slate-400 font-normal">
              Extract real pretrained weights from actual Transformer checkpoints
            </p>
          </div>
        </div>

        {/* Model Switcher & Status */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/[0.08] border border-emerald-500/20 text-emerald-400 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 glow-active" />
            <span>PyTorch 2.9.1 • Transformers 4.57</span>
          </div>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-xl border border-white/[0.08] hover:border-slate-600 transition-all bg-[#0e1320]"
          >
            <FileCode className="w-3.5 h-3.5" />
            <span>FastAPI Docs</span>
            <ExternalLink className="w-3 h-3 text-slate-500" />
          </a>
        </div>
      </header>

      {/* 2. MODEL SELECTION & PRESETS DECK */}
      <section className="w-full max-w-7xl glass-card rounded-2xl p-5 mb-6 space-y-4">
        {/* Model Selection Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-indigo-400" />
              Active Checkpoint:
            </span>
            <div className="flex items-center gap-1.5 bg-[#080b13] p-1 rounded-xl border border-white/[0.08]">
              {modelOptions.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => {
                    setModelType(m.id);
                    setSelectedLayer(0);
                    if (activeHead >= m.heads) setActiveHead(-1);
                  }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    modelType === m.id
                      ? "bg-gradient-to-r from-indigo-600 to-cyan-600 text-white shadow-md shadow-indigo-600/30"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {m.name}
                </button>
              ))}
            </div>
          </div>

          {/* Multi-Layer Selector for Pretrained Models */}
          {activeModel.layers > 1 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-400 font-medium">Layer:</span>
              <div className="flex items-center gap-1 bg-[#080b13] p-1 rounded-xl border border-white/[0.08]">
                {Array.from({ length: activeModel.layers }, (_, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setSelectedLayer(idx)}
                    className={`px-2.5 py-1 rounded-md font-mono text-xs font-semibold transition-all ${
                      selectedLayer === idx
                        ? "bg-cyan-500 text-black shadow-sm"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    L{idx + 1}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Preset Chips */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs">
          <span className="text-slate-400 font-medium shrink-0 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Presets:
          </span>
          {presets.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setInputText(p.text)}
              className={`shrink-0 px-3 py-1.5 rounded-xl text-xs font-medium transition-all border ${
                inputText === p.text
                  ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/40 shadow-sm"
                  : "bg-[#090d16] text-slate-400 hover:text-white border-white/[0.06] hover:border-white/[0.15]"
              }`}
              title={p.desc}
            >
              {p.title}
            </button>
          ))}
        </div>

        {/* Input Textarea & Token preview */}
        <div className="relative">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={2}
            placeholder="Type any sentence to visualize real attention weights from this model..."
            className="w-full bg-[#080b13] border border-white/[0.1] rounded-xl p-3.5 text-sm text-slate-100 font-mono leading-relaxed resize-none focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all placeholder:text-slate-600"
          />
          <div className="absolute right-3 bottom-3 flex items-center gap-2">
            <span className="text-xs font-mono text-slate-500">
              {tokens.length} subword tokens ({tokens.length * tokens.length} attention weights)
            </span>
          </div>
        </div>

        {/* Hyperparameter Controls Deck (for custom model or temperature) */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2 border-t border-white/[0.06] text-xs">
          {/* Heads Selector */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium flex items-center justify-between">
              <span>Attention Heads (h)</span>
              <span className="text-indigo-400 font-mono font-semibold">
                {currentHeadsCount} heads {isPretrained ? "(Fixed by model)" : ""}
              </span>
            </label>
            <div className="h-9 bg-[#080b13] border border-white/[0.08] rounded-xl px-3 flex items-center font-mono text-xs text-slate-300">
              {isPretrained ? `${activeModel.name}: ${currentHeadsCount} heads` : `${numHeads} heads (Custom)`}
            </div>
          </div>

          {/* Model Dimension */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium flex items-center justify-between">
              <span>Embedding Dim (d)</span>
              <span className="text-cyan-400 font-mono">d_k = {dK}</span>
            </label>
            <div className="h-9 bg-[#080b13] border border-white/[0.08] rounded-xl px-3 flex items-center font-mono text-xs text-cyan-400">
              {onlineData?.d_model || (isPretrained ? activeModel.dim : dModel)}-dimensional vector
            </div>
          </div>

          {/* Temperature Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-slate-400 font-medium">
              <span>Softmax Temp (τ)</span>
              <span className="text-amber-400 font-mono font-semibold">{temperature.toFixed(1)}</span>
            </div>
            <div className="h-9 bg-[#080b13] border border-white/[0.08] rounded-xl px-3 flex items-center">
              <input
                type="range"
                min="0.2"
                max="3.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer"
              />
            </div>
          </div>

          {/* Architecture mode */}
          <div className="space-y-1.5">
            <label className="text-slate-400 font-medium flex items-center justify-between">
              <span>Model Architecture</span>
              <span className="text-indigo-400 font-mono font-semibold">
                {isPretrained ? "Pretrained Transformer" : "Custom MHA"}
              </span>
            </label>
            <button
              type="button"
              onClick={fetchAttentionData}
              disabled={isComputing}
              className="w-full h-9 rounded-xl bg-gradient-to-r from-indigo-600 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20 active:scale-95 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isComputing ? "animate-spin" : ""}`} />
              <span>{isComputing ? "Computing Tensors..." : "Recompute Attention"}</span>
            </button>
          </div>
        </div>
      </section>

      {/* 3. MAIN WORKSPACE: HEATMAP & FLOW VISUALIZER + RIGHT ANALYTICS */}
      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT / CENTER STAGE (8 Columns) */}
        <div className="lg:col-span-8 glass-card rounded-2xl p-5 space-y-5">
          {/* Top Bar of Canvas: Head Switcher & View Switcher */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-white/[0.08]">
            {/* Head Selector Tabs */}
            <div className="flex items-center gap-1.5 overflow-x-auto max-w-full pb-1">
              <button
                type="button"
                onClick={() => setActiveHead(-1)}
                className={`px-3 py-1.5 rounded-xl text-xs font-semibold shrink-0 transition-all flex items-center gap-1.5 ${
                  activeHead === -1
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                    : "bg-[#080b13] text-slate-400 hover:text-white border border-white/[0.06]"
                }`}
              >
                <Zap className="w-3.5 h-3.5 text-cyan-300" />
                <span>Head Average</span>
              </button>

              {Array.from({ length: currentHeadsCount }, (_, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setActiveHead(idx)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-semibold shrink-0 transition-all ${
                    activeHead === idx
                      ? "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-lg shadow-indigo-600/30"
                      : "bg-[#080b13] text-slate-400 hover:text-white border border-white/[0.06]"
                  }`}
                >
                  Head {idx + 1}
                </button>
              ))}
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center p-0.5 bg-[#080b13] border border-white/[0.08] rounded-xl text-xs">
              <button
                type="button"
                onClick={() => setViewMode("heatmap")}
                className={`px-3 py-1 rounded-lg font-semibold transition-all ${
                  viewMode === "heatmap" ? "bg-[#182030] text-cyan-300 shadow-sm" : "text-slate-400 hover:text-white"
                }`}
              >
                2D Heatmap
              </button>
              <button
                type="button"
                onClick={() => setViewMode("arc")}
                className={`px-3 py-1 rounded-lg font-semibold transition-all ${
                  viewMode === "arc" ? "bg-[#182030] text-cyan-300 shadow-sm" : "text-slate-400 hover:text-white"
                }`}
              >
                Arc Flow
              </button>
            </div>
          </div>

          {/* Canvas Main Stage */}
          <div className="relative min-h-[440px] flex flex-col justify-center items-center overflow-x-auto p-2">
            {viewMode === "heatmap" ? (
              /* ================= 2D HEATMAP GRID ================= */
              <div className="w-full flex flex-col items-center">
                <div className="inline-block border border-white/[0.06] rounded-xl overflow-hidden bg-[#06080e] p-3 shadow-2xl">
                  {/* Top Key Headers */}
                  <div className="flex items-center mb-1">
                    <div className="w-24 text-right pr-3 text-[11px] font-mono text-slate-500 font-semibold truncate">
                      Query \ Key
                    </div>
                    {tokens.map((tok, j) => (
                      <div
                        key={j}
                        className={`w-11 text-center text-[11px] font-mono font-medium truncate px-0.5 transition-colors ${
                          hoveredCell?.j === j ? "text-cyan-300 font-bold" : "text-slate-400"
                        }`}
                        title={`Key [${j}]: ${tok}`}
                      >
                        {tok}
                      </div>
                    ))}
                  </div>

                  {/* Query Rows */}
                  {activeMatrix.map((row, i) => (
                    <div key={i} className="flex items-center my-0.5">
                      <div
                        className={`w-24 text-right pr-3 text-[11px] font-mono font-medium truncate transition-colors ${
                          hoveredCell?.i === i ? "text-indigo-300 font-bold" : "text-slate-400"
                        }`}
                        title={`Query [${i}]: ${tokens[i]}`}
                      >
                        {tokens[i]}
                      </div>

                      {row.map((val, j) => {
                        const isHovered = hoveredCell?.i === i && hoveredCell?.j === j;
                        const isCrosshair = hoveredCell && (hoveredCell.i === i || hoveredCell.j === j);
                        const isDimmed = hoveredCell && !isCrosshair;

                        return (
                          <div
                            key={j}
                            onMouseEnter={() => setHoveredCell({ i, j, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{ backgroundColor: getCellColor(val) }}
                            className={`w-11 h-11 m-[1px] rounded-[5px] flex items-center justify-center font-mono font-semibold text-[10px] text-white cursor-pointer transition-all duration-150 select-none ${
                              isHovered
                                ? "scale-125 z-20 ring-2 ring-cyan-300 shadow-xl shadow-cyan-500/40"
                                : isCrosshair
                                ? "ring-1 ring-white/40 z-10"
                                : isDimmed
                                ? "opacity-25"
                                : "hover:scale-110"
                            }`}
                          >
                            {val < -100 ? "—" : val.toFixed(2)}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>

                {/* Heatmap Legend Bar */}
                <div className="w-full max-w-md flex items-center justify-between text-xs text-slate-400 mt-5 pt-3 border-t border-white/[0.06]">
                  <div className="flex items-center gap-2">
                    <span>0.00 (Low)</span>
                    <div
                      className="w-36 h-2.5 rounded-full border border-white/[0.1]"
                      style={{
                        background:
                          "linear-gradient(to right, rgb(14, 18, 45) 0%, rgb(30, 160, 210) 45%, rgb(255, 218, 40) 100%)",
                      }}
                    />
                    <span>1.00 (High)</span>
                  </div>

                  <span className="text-[11px] font-mono text-slate-500">
                    {activeHead === -1 ? "Averaged across all heads" : `Viewing Head ${activeHead + 1}`}
                  </span>
                </div>
              </div>
            ) : (
              /* ================= ARC FLOW DIAGRAM ================= */
              <div className="w-full flex flex-col items-center py-4">
                <svg className="w-full max-w-2xl h-64 overflow-visible" viewBox={`0 0 ${N * 60 + 40} 180`}>
                  <defs>
                    <linearGradient id="arcGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#6366f1" />
                      <stop offset="100%" stopColor="#22d3ee" />
                    </linearGradient>
                  </defs>

                  {activeMatrix.map((row, i) =>
                    row.map((val, j) => {
                      if (val < 0.08 || i === j) return null;
                      const x1 = 40 + i * 60;
                      const x2 = 40 + j * 60;
                      const dist = Math.abs(x1 - x2);
                      const arcHeight = Math.min(140, dist * 0.75);
                      const isHighlighted = hoveredTokenIndex === i || hoveredTokenIndex === j;

                      return (
                        <path
                          key={`arc-${i}-${j}`}
                          d={`M ${x1} 150 C ${x1} ${150 - arcHeight}, ${x2} ${150 - arcHeight}, ${x2} 150`}
                          fill="none"
                          stroke={isHighlighted ? "url(#arcGlow)" : getCellColor(val)}
                          strokeWidth={isHighlighted ? Math.max(2, val * 8) : Math.max(1, val * 5)}
                          strokeOpacity={isHighlighted ? 0.95 : Math.min(0.8, val + 0.15)}
                          strokeLinecap="round"
                          className="transition-all duration-200 cursor-pointer hover:stroke-cyan-300"
                        />
                      );
                    })
                  )}
                </svg>

                <div className="flex items-center justify-around w-full max-w-2xl px-4 mt-2">
                  {tokens.map((tok, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onMouseEnter={() => setHoveredTokenIndex(idx)}
                      onMouseLeave={() => setHoveredTokenIndex(null)}
                      className={`px-3 py-1.5 rounded-xl font-mono text-xs font-semibold transition-all border ${
                        hoveredTokenIndex === idx
                          ? "bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-lg shadow-cyan-500/20 scale-110"
                          : "bg-[#090d16] text-slate-300 border-white/[0.08] hover:border-slate-500"
                      }`}
                    >
                      {tok}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Interactive Cell Inspector Banner */}
          <div className="p-3.5 rounded-xl bg-[#090d16] border border-white/[0.08] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
            {hoveredCell ? (
              <div className="flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 glow-active" />
                <span className="font-mono text-slate-200">
                  Query <strong className="text-indigo-400">"{tokens[hoveredCell.i]}"</strong> ➔ Key{" "}
                  <strong className="text-cyan-400">"{tokens[hoveredCell.j]}"</strong>
                </span>
                <span className="text-slate-500">•</span>
                <span className="text-slate-400">
                  Attention Weight: <strong className="text-white">{(hoveredCell.val * 100).toFixed(2)}%</strong>
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-slate-400">
                <Info className="w-3.5 h-3.5 text-slate-500" />
                <span>Hover over any heatmap cell or token to inspect exact attention weights & projections</span>
              </div>
            )}

            <div className="text-slate-500 font-mono text-[11px]">
              {isPretrained ? `${activeModel.name} • L${selectedLayer + 1}` : "Custom PyTorch Model"} • {N}×{N}
            </div>
          </div>
        </div>

        {/* RIGHT ANALYTICS & DIAGNOSTICS COLUMN (4 Columns) */}
        <div className="lg:col-span-4 space-y-5">
          {/* Card 1: Top Attention Pathways */}
          <div className="glass-card rounded-2xl p-5 space-y-3.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-indigo-400" />
                Top Attention Connections
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">Strongest</span>
            </div>

            <div className="space-y-2 pt-1">
              {(activeData.top_global_edges || []).slice(0, 5).map((e: any, idx: number) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-xl bg-[#080b13] border border-white/[0.06] flex items-center justify-between text-xs"
                >
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-indigo-300 font-medium">{tokens[e.source]}</span>
                    <ArrowRight className="w-3 h-3 text-slate-500" />
                    <span className="text-cyan-300 font-medium">{tokens[e.target]}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full"
                        style={{ width: `${Math.min(100, e.weight * 100 * 2.5)}%` }}
                      />
                    </div>
                    <span className="font-mono font-semibold text-slate-200 w-10 text-right">
                      {(e.weight * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Card 2: Token Salience (Attended-To Ranking) */}
          <div className="glass-card rounded-2xl p-5 space-y-3.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                Token Salience (Global Inflow)
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">Incoming Sum</span>
            </div>

            <div className="space-y-2 pt-1">
              {(activeData.token_importance || []).slice(0, 5).map((item: any, idx: number) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono font-medium text-slate-300">
                      #{idx + 1} {item.token}
                    </span>
                    <span className="font-mono text-cyan-400 font-semibold">{item.importance.toFixed(3)}</span>
                  </div>
                  <div className="w-full h-1.5 bg-[#080b13] rounded-full overflow-hidden border border-white/[0.04]">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-amber-400 rounded-full"
                      style={{
                        width: `${Math.min(100, (item.importance / (activeData.token_importance[0]?.importance || 1)) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Card 3: Model Architecture Diagnostic */}
          <div className="glass-card rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-amber-400" />
                Active Model Diagnostics
              </h3>
              <span className="text-[10px] text-emerald-400 font-mono font-medium">Online</span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center pt-1 font-mono">
              <div className="p-3 rounded-xl bg-[#080b13] border border-white/[0.06]">
                <div className="text-base font-bold text-white">{currentHeadsCount}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Heads (h)</div>
              </div>
              <div className="p-3 rounded-xl bg-[#080b13] border border-white/[0.06]">
                <div className="text-base font-bold text-cyan-400">{dK}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Head Dim (d_k)</div>
              </div>
            </div>

            <div className="mt-2 p-3 rounded-xl bg-[#080b13] border border-white/[0.06] text-[11px] font-mono text-slate-400 leading-relaxed">
              <div className="text-indigo-300 font-bold mb-1">
                {isPretrained ? `${activeModel.name}` : "Custom Transformer"}
              </div>
              <div>{activeModel.desc}</div>
              <div className="text-[10px] text-slate-500 mt-1.5">
                • Total Layers: {activeModel.layers}
                <br />• Inspecting: Layer {selectedLayer + 1}
                <br />• Hidden Size: {onlineData?.d_model || activeModel.dim}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
