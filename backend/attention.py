"""
Multi-Head Self-Attention Engine.

Implements the Scaled Dot-Product Attention mechanism as introduced in
"Attention Is All You Need" (Vaswani et al., 2017) using PyTorch.

Mathematical formulation:
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + M ) * V
"""

import math
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Produce a deterministic, rich pseudo-semantic embedding vector for a given token.
    Combines character n-gram hashing and linguistic heuristics (word length, vowels, syntax).
    """
    # Deterministic seed based on token
    token_clean = token.strip().lower()
    h = seed
    for char in token_clean:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    
    rng = torch.Generator()
    rng.manual_seed(h)
    vec = torch.randn(d_model, generator=rng)
    
    # Linguistic semantic biasing
    # 1. Pronouns and references
    if token_clean in {"it", "they", "them", "he", "she", "this", "that", "these", "those"}:
        vec[0:4] += 1.8
    # 2. Common nouns
    elif token_clean in {"cat", "dog", "animal", "street", "mat", "arrow", "banana", "robot", "transformer", "model"}:
        vec[4:8] += 1.8
    # 3. Action verbs
    elif token_clean in {"sat", "ran", "jumped", "cross", "flies", "likes", "attention", "learns", "predicts"}:
        vec[8:12] += 1.8
    # 4. Determiners & prepositions
    elif token_clean in {"the", "a", "an", "on", "in", "at", "by", "for", "with", "because"}:
        vec[12:16] += 1.5

    # Normalize vector
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
        # Learnable / initialized linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Initialize with orthogonal weights for distinct head representations
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
        """
        Compute multi-head attention over input tensor x.
        x: shape (seq_len, d_model)
        """
        seq_len, d_model = x.shape
        scale = 1.0 / (math.sqrt(self.d_k) * max(temperature, 1e-4))

        # 1. Project Q, K, V -> (seq_len, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. Reshape into (num_heads, seq_len, d_k)
        Q_heads = Q.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)
        K_heads = K.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)
        V_heads = V.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # (H, N, d_k)

        # 3. Scaled dot product: S = (Q * K^T) * scale -> (H, N, N)
        raw_scores = torch.bmm(Q_heads, K_heads.transpose(1, 2))  # (H, N, N)
        scaled_scores = raw_scores * scale

        # 4. Optional causal mask (mask future tokens with -inf)
        if causal:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scaled_scores = scaled_scores + causal_mask.unsqueeze(0)

        # 5. Attention Weights via Softmax over key dimension
        attention_weights = F.softmax(scaled_scores, dim=-1)  # (H, N, N)

        # 6. Weighted sum over Values: O_heads = A * V -> (H, N, d_k)
        O_heads = torch.bmm(attention_weights, V_heads)

        # 7. Concatenate heads and project -> (seq_len, d_model)
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
) -> Dict[str, Any]:
    """
    Tokenizes input text, applies multi-head self-attention, and extracts
    structured statistics, heatmaps, entropy, and top token dependencies.
    """
    # 1. Clean tokenization
    raw_tokens = text.strip().split()
    if not raw_tokens:
        raw_tokens = ["empty"]
    
    # Cap token count to maintain great interactivity (max 32 tokens)
    tokens = raw_tokens[:32]
    seq_len = len(tokens)

    # 2. Build input embeddings + positional encodings
    emb_list = [token_to_feature_vector(tok, d_model, seed=seed) for tok in tokens]
    X_emb = torch.stack(emb_list, dim=0)  # (seq_len, d_model)
    pe = get_sinusoidal_positional_encoding(seq_len, d_model)
    X = X_emb + pe

    # 3. Instantiate model & compute
    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, seed=seed)
    with torch.no_grad():
        result = model(X, causal=causal, temperature=temperature)

    attn_weights = result["attention_weights"]  # (H, N, N)
    raw_scores = result["raw_scores"]          # (H, N, N)
    scaled_scores = result["scaled_scores"]    # (H, N, N)

    # 4. Format per-head data
    heads_data = []
    avg_matrix = attn_weights.mean(dim=0).tolist()  # (N, N)

    for h in range(num_heads):
        head_attn = attn_weights[h].tolist()
        head_raw = raw_scores[h].tolist()
        head_scaled = scaled_scores[h].tolist()

        # Compute entropy per token query: - sum(p * log(p + eps))
        entropy_list = []
        for i in range(seq_len):
            row = attn_weights[h, i]
            ent = -torch.sum(row * torch.log(row + 1e-9)).item()
            entropy_list.append(round(ent, 4))

        # Top attention connections for this head
        edges = []
        for i in range(seq_len):
            for j in range(seq_len):
                weight = head_attn[i][j]
                edges.append({
                    "source": i,
                    "target": j,
                    "source_token": tokens[i],
                    "target_token": tokens[j],
                    "weight": round(weight, 4),
                })
        edges.sort(key=lambda e: e["weight"], reverse=True)
        top_edges = edges[:5]

        heads_data.append({
            "head_index": h,
            "head_name": f"Head {h + 1}",
            "attention_matrix": [[round(val, 4) for val in row] for row in head_attn],
            "raw_scores": [[round(val, 4) for val in row] for row in head_raw],
            "scaled_scores": [[round(val, 4) for val in row] for row in head_scaled],
            "entropy": entropy_list,
            "top_edges": top_edges,
        })

    # Global token importance (column sum of average attention: who receives the most attention?)
    avg_tensor = attn_weights.mean(dim=0)
    col_sums = avg_tensor.sum(dim=0).tolist()
    token_importance = [
        {"token": tok, "index": idx, "importance": round(col_sums[idx] / seq_len, 4)}
        for idx, tok in enumerate(tokens)
    ]

    # Global strongest attention edges across average
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
        "tokens": tokens,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "d_model": d_model,
        "d_k": d_model // num_heads,
        "causal": causal,
        "temperature": temperature,
        "heads": heads_data,
        "average_attention": [[round(val, 4) for val in row] for row in avg_matrix],
        "token_importance": token_importance,
        "top_global_edges": global_edges[:8],
    }

