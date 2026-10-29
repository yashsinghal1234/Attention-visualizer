"""
Launcher script for the Attention Visualizer full-stack project.

Usage:
    python run.py
    python run.py --port 8000 --no-browser
"""

import argparse
import sys
import webbrowser
import threading
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def open_browser_delayed(url: str, delay: float = 1.2):
    """Wait briefly for server to bind, then open browser."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Launch Attention Visualizer Server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    url = f"http://localhost:3001"

    print("=" * 64)
    print("   🧠 UQURA NODES Visualizer - Full Next.js & PyTorch Project   ")
    print("=" * 64)
    print(f"[*] Next.js Web App:           http://localhost:3001")
    print(f"[*] FastAPI Backend API:       http://{args.host}:{args.port}")
    print(f"[*] API Documentation:         http://{args.host}:{args.port}/docs")
    print("-" * 64)

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    try:
        import uvicorn
        uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\n[*] Server shutdown cleanly.")


if __name__ == "__main__":
    main()

