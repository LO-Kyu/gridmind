"""
Hackathon entrypoint: run from repo root with:
  python inference.py
Delegates to python/inference.py (single source of truth).
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    impl = Path(__file__).resolve().parent / "python" / "inference.py"
    runpy.run_path(str(impl), run_name="__main__")
