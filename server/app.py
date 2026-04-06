"""
GridMind-RL server entry point.
The core simulation runs as a Go binary (main.go) on port 7860.
This module provides the Python entry point expected by OpenEnv spec
and pyproject.toml [project.scripts].

In Docker: the Go binary is started by supervisord.
For local dev: use `go run main.go` directly.
"""
import os
import subprocess
import sys


def main():
    """Start the GridMind-RL Go environment server."""
    port = os.getenv("PORT", "7860")
    
    # Look for compiled binary first
    binary_paths = [
        "/usr/local/bin/gridmind-server",  # Docker path
        "./gridmind-server",               # Local Linux/Mac
        "./gridmind-server.exe",           # Local Windows
    ]
    
    binary = None
    for path in binary_paths:
        if os.path.exists(path):
            binary = path
            break
    
    if binary:
        print(f"[GridMind-RL] Starting Go server on port {port}", flush=True)
        os.execv(binary, [binary])
    else:
        # Fallback: try go run for development
        print(f"[GridMind-RL] Binary not found, trying go run main.go", flush=True)
        try:
            proc = subprocess.run(
                ["go", "run", "main.go"],
                env={**os.environ, "PORT": port}
            )
            sys.exit(proc.returncode)
        except FileNotFoundError:
            print(
                "[GridMind-RL] ERROR: Neither compiled binary nor 'go' found.\n"
                "Build with: go build -o gridmind-server main.go\n"
                "Or run in Docker: docker run -p 7860:7860 gridmind-rl",
                file=sys.stderr
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
