"""
Multi-Head Self-Attention Engine.

Supports both:
1. Online / Pretrained Transformer Models (BERT, DistilBERT, GPT-2 via HuggingFace)
   extracting true multi-layer, multi-head attention weights.
2. The fundamental Scaled Dot-Product Attention mechanism (Vaswani et al., 2017)
   implemented in pure PyTorch with customizable heads, dimensions, temperature, and causal masks.
"""

import math
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global in-memory cache for loaded HuggingFace models & tokenizers
LOADED_MODELS: Dict[str, Any] = {}

AVAILABLE_ONLINE_MODELS = {
    "distilbert-base-uncased": {
        "id": "distilbert-base-uncased",
        "name": "DistilBERT Base (Uncased)",
        "layers": 6,
        "heads": 12,
        "dim": 768,
        "type": "encoder",
        "description": "6-layer transformer encoder pre-trained on Wikipedia + BookCorpus",
    },
    "prajjwal1/bert-tiny": {
        "id": "prajjwal1/bert-tiny",
        "name": "BERT Tiny (2 Layers, 2 Heads)",
        "layers": 2,
        "heads": 2,
        "dim": 128,
        "type": "encoder",
        "description": "Ultra-lightweight 2-layer compact BERT checkpoint",
    },
}


def get_pretrained_model(model_name: str):
    """Load and cache a pretrained HuggingFace model with output_attentions=True."""
    if model_name not in LOADED_MODELS:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()
        LOADED_MODELS[model_name] = (tokenizer, model)
    return LOADED_MODELS[model_name]


def analyze_pretrained_attention(
    text: str,
    model_name: str = "distilbert-base-uncased",
    layer_index: int = 0,
) -> Dict[str, Any]:
    """
    Pass text through an actual pretrained Transformer model and extract
    true attention weights from all heads at the specified layer.
    """
    tokenizer, model = get_pretrained_model(model_name)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    seq_len = len(tokens)

    # Run inference with attention capture
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of length num_layers
    # Each tensor has shape: (batch_size=1, num_heads, seq_len, seq_len)
    num_layers = len(outputs.attentions)
    target_layer = max(0, min(layer_index, num_layers - 1))
    layer_attn = outputs.attentions[target_layer][0]  # Shape: (num_heads, seq_len, seq_len)
    num_heads = layer_attn.shape[0]

    # Format per-head data
    heads_data = []
    avg_matrix = layer_attn.mean(dim=0).tolist()

    for h in range(num_heads):
        head_attn = layer_attn[h].tolist()
        
        # Entropy per token row: - sum(p * log(p))
        entropy_list = []
        for i in range(seq_len):
            row = layer_attn[h, i]
            ent = -torch.sum(row * torch.log2(row + 1e-9)).item()
            entropy_list.append(round(ent, 3))

        # Top attention connections
        edges = []
        for i in range(seq_len):
            for j in range(seq_len):
                edges.append({
                    "source": i,
                    "target": j,
                    "source_token": tokens[i],
                    "target_token": tokens[j],
                    "weight": round(head_attn[i][j], 4),
                })
        edges.sort(key=lambda e: e["weight"], reverse=True)

        heads_data.append({
            "head_index": h,
            "head_name": f"Head {h + 1}",
            "attention_matrix": [[round(val, 4) for val in row] for row in head_attn],
            "raw_scores": [[round(val, 4) for val in row] for row in head_attn],
            "scaled_scores": [[round(val, 4) for val in row] for row in head_attn],
            "entropy": entropy_list,
            "top_edges": edges[:5],
        })

    # Global token importance (column sum of average attention)
    avg_tensor = layer_attn.mean(dim=0)
    col_sums = avg_tensor.sum(dim=0).tolist()
    token_importance = [
        {"token": tok, "index": idx, "importance": round(col_sums[idx] / seq_len, 4)}
        for idx, tok in enumerate(tokens)
    ]

    # Global strongest edges across average
    global_edges = []
    for i in range(seq_len):
        for j in range(seq_len):
            global_edges.append({
                "source": i,
                "target": j,
                "source_token": tokens[i],
                "target_token": tokens[j],
                "weight": round(avg_matrix[i][j], 4),
            })
    global_edges.sort(key=lambda e: e["weight"], reverse=True)

    return {
        "model_name": model_name,
        "is_pretrained": True,
        "layer_index": target_layer,
        "total_layers": num_layers,
        "tokens": tokens,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "d_model": getattr(model.config, "hidden_size", 768),
        "d_k": getattr(model.config, "hidden_size", 768) // num_heads,
        "causal": False,
        "temperature": 1.0,
        "heads": heads_data,
        "average_attention": [[round(val, 4) for val in row] for row in avg_matrix],
        "token_importance": sorted(token_importance, key=lambda x: x["importance"], reverse=True),
        "top_global_edges": global_edges[:8],
    }


# =========================================================================
# Custom PyTorch Scaled Dot-Product Multi-Head Self-Attention Layer
# =========================================================================

def get_sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate standard sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def token_to_feature_vector(token: str, d_model: int, seed: int = 42) -> torch.Tensor:
    """
    Produce a deterministic pseudo-semantic embedding vector for a given token.
    Combines character n-gram hashing and linguistic heuristics.
    """
    token_clean = token.strip().lower()
    h = seed
    for char in token_clean:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    
    rng = torch.Generator()
    rng.manual_seed(h)
    vec = torch.randn(d_model, generator=rng)
    
    # Linguistic semantic clustering
    if token_clean in {"it", "they", "them", "he", "she", "this", "that", "these", "those"}:
        vec[0:4] += 1.8
    elif token_clean in {"cat", "dog", "animal", "street", "mat", "arrow", "banana", "robot", "transformer", "model"}:
        vec[4:8] += 1.8
    elif token_clean in {"sat", "ran", "jumped", "cross", "flies", "likes", "attention", "learns", "predicts"}:
        vec[8:12] += 1.8
    elif token_clean in {"the", "a", "an", "on", "in", "at", "by", "for", "with", "because"}:
        vec[12:16] += 1.5

    return F.normalize(vec, p=2, dim=0)


class MultiHeadSelfAttention(nn.Module):
    """
    Complete Multi-Head Self-Attention Layer with step-by-step tensor inspection.
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, seed: int = 42):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        torch.manual_seed(seed)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        nn.init.orthogonal_(self.w_q.weight)
        nn.init.orthogonal_(self.w_k.weight)
        nn.init.orthogonal_(self.w_v.weight)
        nn.init.orthogonal_(self.w_o.weight)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        seq_len, d_model = x.shape
        scale = 1.0 / (math.sqrt(self.d_k) * max(temperature, 1e-4))

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q_heads = Q.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)
        K_heads = K.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)
        V_heads = V.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)

        raw_scores = torch.bmm(Q_heads, K_heads.transpose(1, 2))  # (H, N, N)
        scaled_scores = raw_scores * scale

        if causal:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scaled_scores = scaled_scores + causal_mask.unsqueeze(0)

        attention_weights = F.softmax(scaled_scores, dim=-1)  # (H, N, N)
        O_heads = torch.bmm(attention_weights, V_heads)

        O_concat = O_heads.transpose(0, 1).contiguous().view(seq_len, d_model)
        output = self.w_o(O_concat)

        return {
            "raw_scores": raw_scores,
            "scaled_scores": scaled_scores,
            "attention_weights": attention_weights,
            "Q_heads": Q_heads,
            "K_heads": K_heads,
            "V_heads": V_heads,
            "output": output,
        }


def analyze_text_attention(
    text: str,
    num_heads: int = 4,
    d_model: int = 64,
    causal: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
    model_name: str = "custom",
    layer_index: int = 0,
) -> Dict[str, Any]:
    """
    Main entry point: routes to either an actual online pretrained model
    or the custom PyTorch Scaled Dot-Product Attention layer.
    """
    if model_name in AVAILABLE_ONLINE_MODELS:
        return analyze_pretrained_attention(text, model_name=model_name, layer_index=layer_index)

    # Otherwise compute via custom PyTorch MHA layer
    raw_tokens = text.strip().split()
    if not raw_tokens:
        raw_tokens = ["empty"]
    
    tokens = raw_tokens[:32]
    seq_len = len(tokens)

    emb_list = [token_to_feature_vector(tok, d_model, seed=seed) for tok in tokens]
    X_emb = torch.stack(emb_list, dim=0)
    pe = get_sinusoidal_positional_encoding(seq_len, d_model)
    X = X_emb + pe

    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, seed=seed)
    with torch.no_grad():
        result = model(X, causal=causal, temperature=temperature)

    attn_weights = result["attention_weights"]
    raw_scores = result["raw_scores"]
    scaled_scores = result["scaled_scores"]

    heads_data = []
    avg_matrix = attn_weights.mean(dim=0).tolist()

    for h in range(num_heads):
        head_attn = attn_weights[h].tolist()
        head_raw = raw_scores[h].tolist()
        head_scaled = scaled_scores[h].tolist()

        entropy_list = []
        for i in range(seq_len):
            row = attn_weights[h, i]
            ent = -torch.sum(row * torch.log2(row + 1e-9)).item()
            entropy_list.append(round(ent, 3))

        edges = []
        for i in range(seq_len):
            for j in range(seq_len):
                edges.append({
                    "source": i,
                    "target": j,
                    "source_token": tokens[i],
                    "target_token": tokens[j],
                    "weight": round(head_attn[i][j], 4),
                })
        edges.sort(key=lambda e: e["weight"], reverse=True)

        heads_data.append({
            "head_index": h,
            "head_name": f"Head {h + 1}",
            "attention_matrix": [[round(val, 4) for val in row] for row in head_attn],
            "raw_scores": [[round(val, 4) for val in row] for row in head_raw],
            "scaled_scores": [[round(val, 4) for val in row] for row in head_scaled],
            "entropy": entropy_list,
            "top_edges": edges[:5],
        })

    avg_tensor = attn_weights.mean(dim=0)
    col_sums = avg_tensor.sum(dim=0).tolist()
    token_importance = [
        {"token": tok, "index": idx, "importance": round(col_sums[idx] / seq_len, 4)}
        for idx, tok in enumerate(tokens)
    ]

    global_edges = []
    for i in range(seq_len):
        for j in range(seq_len):
            global_edges.append({
                "source": i,
                "target": j,
                "source_token": tokens[i],
                "target_token": tokens[j],
                "weight": round(avg_matrix[i][j], 4),
            })
    global_edges.sort(key=lambda e: e["weight"], reverse=True)

    return {
        "model_name": "custom",
        "is_pretrained": False,
        "layer_index": 0,
        "total_layers": 1,
        "tokens": tokens,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "d_model": d_model,
        "d_k": d_model // num_heads,
        "causal": causal,
        "temperature": temperature,
        "heads": heads_data,
        "average_attention": [[round(val, 4) for val in row] for row in avg_matrix],
        "token_importance": sorted(token_importance, key=lambda x: x["importance"], reverse=True),
        "top_global_edges": global_edges[:8],
    }
