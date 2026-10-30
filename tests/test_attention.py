"""
Unit tests for the Multi-Head Attention Engine and API endpoints.
"""

import pytest
import torch
from fastapi.testclient import TestClient

from backend.attention import (
    MultiHeadSelfAttention,
    analyze_text_attention,
    get_sinusoidal_positional_encoding,
)
from backend.app import app

client = TestClient(app)


def test_positional_encoding_shape():
    """Verify sinusoidal positional encoding shapes."""
    seq_len = 10
    d_model = 64
    pe = get_sinusoidal_positional_encoding(seq_len, d_model)
    assert pe.shape == (seq_len, d_model)


def test_attention_shapes_and_row_sum():
    """Test that attention weights sum to 1.0 along keys for every query."""
    seq_len = 7
    d_model = 32
    num_heads = 4
    x = torch.randn(seq_len, d_model)
    
    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    out = model(x, causal=False)
    
    attn = out["attention_weights"]
    assert attn.shape == (num_heads, seq_len, seq_len)
    
    # Check row sum equals 1.0
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_masking():
    """Test that future tokens receive zero attention when causal=True."""
    seq_len = 5
    d_model = 16
    num_heads = 2
    x = torch.randn(seq_len, d_model)
    
    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    out = model(x, causal=True)
    attn = out["attention_weights"]
    
    for h in range(num_heads):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn[h, i, j].item() == 0.0, f"Future token at ({i}, {j}) should be 0.0"


def test_temperature_scaling():
    """Lower temperature should lead to sharper attention (lower entropy)."""
    text = "transformers revolutionize artificial intelligence and language modeling"
    
    res_cold = analyze_text_attention(text, num_heads=2, d_model=32, temperature=0.2)
    res_hot = analyze_text_attention(text, num_heads=2, d_model=32, temperature=3.0)
    
    # Average entropy across all heads for first token
    ent_cold = res_cold["heads"][0]["entropy"][0]
    ent_hot = res_hot["heads"][0]["entropy"][0]
    assert ent_cold < ent_hot, f"Cold entropy ({ent_cold}) should be lower than hot ({ent_hot})"


def test_divisibility_assertion():
    """Ensure error is raised if d_model is not divisible by num_heads."""
    with pytest.raises(AssertionError):
        MultiHeadSelfAttention(d_model=50, num_heads=3)


def test_api_health():
    """Verify health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "torch_version" in data


def test_api_presets():
    """Verify presets endpoint."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    assert "text" in data[0]


def test_api_attention_endpoint():
    """Verify the compute attention endpoint with custom payload."""
    payload = {
        "text": "The cat sat on the mat",
        "model_name": "custom",
        "num_heads": 4,
        "d_model": 64,
        "causal": False,
        "temperature": 1.0,
    }
    response = client.post("/api/attention", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["seq_len"] == 6
    assert len(data["heads"]) == 4
    assert len(data["average_attention"]) == 6
    assert len(data["token_importance"]) == 6


def test_api_pretrained_model():
    """Verify online pretrained model extraction (bert-tiny)."""
    payload = {
        "text": "The cat sat on the mat",
        "model_name": "prajjwal1/bert-tiny",
        "layer_index": 0,
    }
    response = client.post("/api/attention", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_pretrained"] is True
    assert "[CLS]" in data["tokens"]
    assert len(data["heads"]) == 2

