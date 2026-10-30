"""
FastAPI Server for Attention Visualizer.

Serves the REST API supporting both real online HuggingFace pretrained models
and custom PyTorch Multi-Head Scaled Dot-Product Attention.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import torch

from backend.schemas import (
    AttentionRequest,
    AttentionResponse,
    PresetItem,
    ModelInfo,
)
from backend.attention import (
    analyze_text_attention,
    AVAILABLE_ONLINE_MODELS,
)

app = FastAPI(
    title="Attention Visualizer API",
    description="Interactive Multi-Head Self-Attention Exploration Service",
    version="2.0.0",
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
        title="Classic Rhyme",
        category="Syntactic Association",
        text="the cat sat on the mat",
        description="Subject-verb-location dependencies."
    ),
    PresetItem(
        id="winograd",
        title="Winograd Schema (Coreference)",
        category="Semantic Ambiguity",
        text="The animal didn't cross the street because it was too tired",
        description="Coreference resolution: pronoun 'it' targets 'animal'."
    ),
    PresetItem(
        id="polysemy",
        title="Lexical Polysemy & Puns",
        category="Disambiguation",
        text="Time flies like an arrow fruit flies like a banana",
        description="Word sense disambiguation: 'flies' as verb vs. noun."
    ),
    PresetItem(
        id="transformer",
        title="Attention Is All You Need",
        category="Deep Learning",
        text="Transformers replace recurrence and convolutions with multi-head attention",
        description="Long-range semantic coupling across technical domain text."
    ),
]


@app.get("/api/health")
@app.get("/health")
def health_check():
    """Health status and environment diagnostic."""
    return {
        "status": "healthy",
        "service": "Attention Visualizer Backend",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "online_models_ready": list(AVAILABLE_ONLINE_MODELS.keys()),
    }


@app.get("/api/models", response_model=list[ModelInfo])
def get_available_models():
    """Return list of available pretrained online models and architectures."""
    custom_model = ModelInfo(
        id="custom",
        name="Custom Scaled Dot-Product Engine",
        layers=1,
        heads=4,
        dim=64,
        type="custom",
        description="Configurable PyTorch Multi-Head Attention layer with custom heads, dims, and masks."
    )
    online_list = [ModelInfo(**info) for info in AVAILABLE_ONLINE_MODELS.values()]
    return [custom_model] + online_list


@app.get("/api/presets", response_model=list[PresetItem])
def get_presets():
    """Return list of pre-configured sample sentences for immediate exploration."""
    return PRESETS


@app.post("/api/attention", response_model=AttentionResponse)
def compute_attention(req: AttentionRequest):
    """
    Compute multi-head self-attention matrices, entropy metrics,
    and dependency graphs using either online pretrained checkpoints or custom MHA.
    """
    if req.model_name == "custom":
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
            model_name=req.model_name,
            layer_index=req.layer_index,
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Redirect root to Next.js app on port 3001
@app.get("/")
def redirect_to_nextjs():
    return RedirectResponse(url="http://localhost:3001", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
