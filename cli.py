"""
Command-Line Interface (CLI) for Attention Visualizer.

Run standalone in the terminal without a web browser:
    python cli.py "the cat sat on the mat"
    python cli.py "time flies like an arrow" --heads 8 --causal --temperature 0.8
"""

import argparse
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.attention import analyze_text_attention


def render_ascii_heatmap(tokens: list, matrix: list, title: str = "Attention Matrix"):
    """Render a text-based heatmap table in terminal."""
    n = len(tokens)
    print(f"\n=== {title} ===")
    
    # Calculate cell widths
    max_tok_len = max(len(t) for t in tokens)
    col_w = max(max_tok_len, 6)
    
    # Header row
    hdr = f"{'Query \\ Key':>{col_w + 4}} | " + " ".join(f"{t:>{col_w}}" for t in tokens)
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    # Density characters
    shades = [".", ":", "+", "*", "#"]

    for i in range(n):
        row_str = f"{tokens[i]:>{col_w + 4}} | "
        for j in range(n):
            val = matrix[i][j]
            shade_idx = min(int(val * len(shades)), len(shades) - 1)
            char = shades[shade_idx]
            row_str += f"{val:>{col_w - 2}.2f}{char} "
        print(row_str)
    print("-" * len(hdr))


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize Multi-Head Scaled Dot-Product Attention in your terminal."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="the cat sat on the mat",
        help="Input sentence to process."
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Model embedding dimension (default: 64)."
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Apply causal autoregressive mask (GPT-style)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature scaling factor (default: 1.0)."
    )
    parser.add_argument(
        "--head-index",
        type=int,
        default=None,
        help="Specific head index (0-based) to display (default: display all + average)."
    )

    args = parser.parse_args()

    print("\n[*] Running PyTorch Multi-Head Self-Attention Engine...")
    print(f"Text: '{args.text}'")
    print(f"Config: heads={args.heads}, d_model={args.dim}, causal={args.causal}, temperature={args.temperature}")

    data = analyze_text_attention(
        text=args.text,
        num_heads=args.heads,
        d_model=args.dim,
        causal=args.causal,
        temperature=args.temperature,
    )

    tokens = data["tokens"]

    if args.head_index is not None:
        if 0 <= args.head_index < args.heads:
            h_data = data["heads"][args.head_index]
            render_ascii_heatmap(tokens, h_data["attention_matrix"], title=f"Head {args.head_index + 1} Attention Matrix")
            print("\nTop 3 Attention Connections:")
            for e in h_data["top_edges"][:3]:
                print(f"  {e['source_token']} -> {e['target_token']}  (weight: {e['weight']:.4f})")
        else:
            print(f"Error: head-index must be between 0 and {args.heads - 1}")
            sys.exit(1)
    else:
        # Display each head's strongest connections
        for h in data["heads"]:
            print(f"\n--- Head {h['head_index'] + 1} Strongest Links ---")
            for e in h["top_edges"][:3]:
                print(f"  {e['source_token']} -> {e['target_token']}  (weight: {e['weight']:.4f})")
        
        # Display Average Attention Heatmap
        render_ascii_heatmap(tokens, data["average_attention"], title="Average Attention (All Heads)")

    print("\nToken Importance (Attended-to ranking):")
    for item in sorted(data["token_importance"], key=lambda x: x["importance"], reverse=True):
        bar = "#" * int(item["importance"] * 30)
        print(f"  {item['token']:<12} | {item['importance']:.3f} {bar}")

    print("\nDone! Start the interactive web UI with: python run.py\n")


if __name__ == "__main__":
    main()
