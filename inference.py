"""
Hackathon entrypoint: run from repo root with:
  python inference.py

Reads environment variables:
  - API_BASE_URL (default: https://openrouter.ai/api/v1)
  - MODEL_NAME (default: meta-llama/llama-3.3-70b-instruct:free)
  - HF_TOKEN (mandatory, no default)

Emits hackathon-compliant stdout format:
  [START] task=<name> env=gridmind model=<model>
  [STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>

Delegates to python/inference.py (single source of truth).
"""
import os
import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Load .env file FIRST (if present)
    try:
        from dotenv import load_dotenv
        load_dotenv()  # reads .env from current directory or project root
    except ImportError:
        pass  # python-dotenv not installed — env vars must be set manually
    
    # Now validate HF_TOKEN after .env is loaded
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        # Allow OPENAI_API_KEY as fallback for development
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "[ERROR] HF_TOKEN environment variable is required "
                "(or OPENAI_API_KEY for development)",
                file=sys.stderr
            )
            sys.exit(1)
    
    impl = Path(__file__).resolve().parent / "python" / "inference.py"
    runpy.run_path(str(impl), run_name="__main__")
