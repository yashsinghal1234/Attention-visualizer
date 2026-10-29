"""
FastAPI Server for Attention Visualizer.

Serves the REST API and hosts the frontend interface.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import torch

from backend.schemas import AttentionRequest, AttentionResponse, PresetItem
from backend.attention import analyze_text_attention

app = FastAPI(
    title="Attention Visualizer API",
    description="Interactive Multi-Head Self-Attention Exploration Service",
    version="1.0.0",
)

# Enable CORS for flexible development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

PRESETS = [
    PresetItem(
        id="classic",
        title="Classic Nursery Rhyme",
        category="Syntactic Association",
        text="the cat sat on the mat",
        description="Simple sentence showing subject-verb-location dependencies."
    ),
    PresetItem(
        id="winograd",
        title="Winograd Schema (Coreference)",
        category="Semantic Ambiguity",
        text="The animal didn't cross the street because it was too tired",
        description="Coreference resolution: what does 'it' attend to? 'animal' or 'street'?"
    ),
    PresetItem(
        id="polysemy",
        title="Lexical Polysemy & Puns",
        category="Disambiguation",
        text="Time flies like an arrow fruit flies like a banana",
        description="Contrasting 'flies' as a verb vs 'flies' as a noun."
    ),
    PresetItem(
        id="transformer",
        title="Attention Is All You Need",
        category="Deep Learning",
        text="Transformers replace recurrence and convolutions with multi-head self-attention",
        description="Domain-specific sentence showcasing multi-head syntactic routing."
    ),
]


@app.get("/api/health")
def health_check():
    """Health status and environment diagnostic."""
    return {
        "status": "healthy",
        "service": "Attention Visualizer Backend",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/api/presets", response_model=list[PresetItem])
def get_presets():
    """Return list of pre-configured sample sentences for immediate exploration."""
    return PRESETS


@app.post("/api/attention", response_model=AttentionResponse)
def compute_attention(req: AttentionRequest):
    """
    Compute multi-head self-attention matrices, entropy metrics,
    and dependency graphs for the requested text and configuration.
    """
    if req.d_model % req.num_heads != 0:
        raise HTTPException(
            status_code=400,
            detail=f"d_model ({req.d_model}) must be divisible by num_heads ({req.num_heads})."
        )

    try:
        data = analyze_text_attention(
            text=req.text,
            num_heads=req.num_heads,
            d_model=req.d_model,
            causal=req.causal,
            temperature=req.temperature,
            seed=req.seed,
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Redirect root to Next.js app on port 3001
@app.get("/")
def redirect_to_nextjs():
    return RedirectResponse(url="http://localhost:3001", status_code=302)


# Mount frontend static directory for standalone / legacy preview
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/legacy")
    def serve_legacy_frontend():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend not found, check frontend/ directory."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)

