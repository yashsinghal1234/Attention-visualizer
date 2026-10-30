"""
Pydantic schemas for the Attention Visualizer API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AttentionRequest(BaseModel):
    text: str = Field(
        default="the cat sat on the mat",
        description="Input text sentence to tokenize and visualize."
    )
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="Pretrained model identifier ('distilbert-base-uncased', 'prajjwal1/bert-tiny', or 'custom')."
    )
    layer_index: int = Field(
        default=0,
        ge=0,
        le=12,
        description="Layer index to inspect in multi-layer pretrained models (0-based)."
    )
    num_heads: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of attention heads (for custom model)."
    )
    d_model: int = Field(
        default=64,
        ge=16,
        le=768,
        description="Total embedding dimensionality (for custom model)."
    )
    causal: bool = Field(
        default=False,
        description="Whether to apply causal (autoregressive) masking."
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Softmax temperature scaling factor."
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic initialization."
    )


class EdgeData(BaseModel):
    source: int
    target: int
    source_token: str
    target_token: str
    weight: float


class TokenImportance(BaseModel):
    token: str
    index: int
    importance: float


class HeadAttentionData(BaseModel):
    head_index: int
    head_name: str
    attention_matrix: List[List[float]]
    raw_scores: List[List[float]]
    scaled_scores: List[List[float]]
    entropy: List[float]
    top_edges: List[EdgeData]


class AttentionResponse(BaseModel):
    model_name: str
    is_pretrained: bool
    layer_index: int
    total_layers: int
    tokens: List[str]
    seq_len: int
    num_heads: int
    d_model: int
    d_k: int
    causal: bool
    temperature: float
    heads: List[HeadAttentionData]
    average_attention: List[List[float]]
    token_importance: List[TokenImportance]
    top_global_edges: List[EdgeData]


class PresetItem(BaseModel):
    id: str
    title: str
    category: str
    text: str
    description: str


class ModelInfo(BaseModel):
    id: str
    name: str
    layers: int
    heads: int
    dim: int
    type: str
    description: str
