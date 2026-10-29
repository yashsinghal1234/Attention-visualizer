# 🧠 Attention Studio: Multi-Head Self-Attention Workbench

A high-performance, modular full-stack deep learning educational platform and research workbench for visualizing **Multi-Head Scaled Dot-Product Attention** as introduced in *"Attention Is All You Need"* (Vaswani et al., 2017).

Built with **Next.js 14 (App Router)**, **TypeScript**, **Tailwind CSS**, and a dedicated **PyTorch 2.9 + FastAPI** backend engine.

---

## ⚡ Highlights

- **Interactive 2D Attention Heatmap Grid**: Smooth perceptual colormap (Obsidian $\rightarrow$ Cyan $\rightarrow$ Gold) with crosshair row/column highlighting and exact numerical weights.
- **Dynamic Arc Attention Flow Diagram**: Directional cubic bezier arcs connecting tokens, scaled by attention probability.
- **Multi-Head Exploration & Synthesis**: Inspect individual heads ($h \in \{1, 2, 4, 8\}$) or compute the unified head average.
- **Layer & Matrix Deep Dive**: Toggle between Softmax Weights ($A$), Scaled Logits ($QK^T / \sqrt{d_k}$), and Raw Dot Products ($QK^T$).
- **Linguistic Benchmark Presets**: Winograd Coreference Schemas, Polysemy/Puns, Transformer text, and Syntactic Dependencies.
- **Real-Time PyTorch Backend**: PyTorch linear projections, causal masking ($M$), and Shannon entropy diagnostics ($H$).
- **Terminal CLI**: Run full attention calculations and ASCII matrix heatmaps directly from the command line.
- **Research Ready**: Formulated for academic citations, paper methodology sections, and reproducible benchmarks.

---

## 📐 Mathematical Formulation

### 1. Scaled Dot-Product Attention
Given an input sequence $X \in \mathbb{R}^{N \times d_{\text{model}}}$ with sinusoidal positional encodings $PE$, Queries ($Q$), Keys ($K$), and Values ($V$) are linearly projected:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k} \cdot \tau} + M\right) V$$

Where:
- $d_k = d_{\text{model}} / h$ is the projection dimension per head.
- $\frac{1}{\sqrt{d_k}}$ counteracts the variance growth of dot products in high dimensions.
- $\tau \in [0.2, 3.0]$ is the Softmax Temperature scaling factor.
- $M$ is the attention mask operator:
  - **Bidirectional (BERT):** $M_{ij} = 0 \quad \forall i, j$
  - **Causal Autoregressive (GPT):** $M_{ij} = -\infty$ for $j > i$, and $0$ otherwise.

### 2. Multi-Head Synthesis
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

### 3. Diagnostic Metrics
- **Attention Entropy (Focus vs. Dispersion):**
  $$H_i = -\sum_{j=1}^N A_{ij} \log_2(A_{ij} + \epsilon)$$
- **Token Salience (Global Inflow Centrality):**
  $$S(t_j) = \frac{1}{N} \sum_{i=1}^N \bar{A}_{ij}, \quad \text{where } \bar{A} = \frac{1}{h}\sum_{k=1}^h A^{(k)}$$

---

## 📁 System Architecture

```
attention-visualizer/
├── app/                      # Next.js 14 App Router
│   ├── globals.css           # Ambient mesh background, glassmorphism & styles
│   ├── layout.tsx            # HTML root & metadata
│   └── page.tsx              # Attention Studio interactive UI
├── backend/                  # PyTorch & FastAPI Backend Engine
│   ├── __init__.py           # Package exports
│   ├── attention.py          # Multi-Head Attention module & tensor mathematics
│   ├── schemas.py            # Pydantic request/response data contracts
│   └── app.py                # FastAPI REST API & redirect service
├── tests/                    # Automated Test Suite
│   ├── __init__.py
│   └── test_attention.py     # Pytest invariant & endpoint validation (8/8 passing)
├── cli.py                    # Terminal CLI attention visualizer
├── run.py                    # Server launcher
├── package.json              # Next.js & React dependencies
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Node.js**: v18.0+ (Tested on v24)
- **Python**: 3.10+ (Tested on 3.13)

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Next.js frontend packages
npm install
```

### 3. Launch the Applications

#### Option A: Next.js Interactive Studio (Recommended)
```bash
npm run dev
```
Open **[http://localhost:3001](http://localhost:3001)** in your browser.

#### Option B: PyTorch FastAPI Backend Service
```bash
python run.py
```
- Web Application: **[http://localhost:3001](http://localhost:3001)**
- Interactive Swagger API: **[http://localhost:8000/docs](http://localhost:8000/docs)**
- ReDoc API: **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

---

## 💻 Terminal CLI Usage

Analyze attention directly from your console without opening a browser:

```bash
# Basic sentence analysis
python cli.py "the cat sat on the mat"

# With causal autoregressive mask (GPT-style) and 8 heads
python cli.py "the cat sat on the mat" --heads 8 --causal

# Inspect a specific head with custom temperature
python cli.py "time flies like an arrow" --head-index 0 --temperature 0.8
```

---

## 🧪 Verification & Automated Tests

Run the test suite to verify mathematical invariants (stochastic row-sum $\sum_j A_{ij} = 1.0$, causal triangular zeroing, entropy monotonicity, and API contracts):

```bash
python -m pytest tests/ -v
```

All 8 tests pass:
```
tests/test_attention.py::test_positional_encoding_shape PASSED
tests/test_attention.py::test_attention_shapes_and_row_sum PASSED
tests/test_attention.py::test_causal_masking PASSED
tests/test_attention.py::test_temperature_scaling PASSED
tests/test_attention.py::test_divisibility_assertion PASSED
tests/test_attention.py::test_api_health PASSED
tests/test_attention.py::test_api_presets PASSED
tests/test_attention.py::test_api_attention_endpoint PASSED
```

---

## 📚 Citation / Research Paper Reference

If you use this software or its methodology in your research, please cite:

```bibtex
@software{attention_studio_2026,
  author    = {DeepMind Advanced Agentic Coding},
  title     = {Attention Studio: Multi-Head Scaled Dot-Product Attention Workbench},
  url       = {https://github.com/your-username/attention-visualizer},
  year      = {2026}
}

@inproceedings{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention is All You Need},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017}
}
```

---

## 📄 License
MIT License. Open for educational and academic research use.
