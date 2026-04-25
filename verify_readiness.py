#!/usr/bin/env python3
"""Final project readiness verification."""

import json
import os
import subprocess
import sys
from pathlib import Path

GRIDMIND_ROOT = Path(".")

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description:<50} ({path})")
    return exists

def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description:<50} ({path})")
    return exists

def check_file_size(path: str, min_bytes: int, description: str) -> bool:
    """Check if a file exists and is above minimum size."""
    if not os.path.exists(path):
        print(f"  ✗ {description:<50} (not found)")
        return False
    size = os.path.getsize(path)
    ok = size >= min_bytes
    status = "✓" if ok else "✗"
    print(f"  {status} {description:<50} ({size} bytes, min {min_bytes})")
    return ok

print("=" * 70)
print("GridMind-RL PROJECT READINESS CHECK")
print("=" * 70)

all_ok = True

# 1. Essential Files
print("\n1. ESSENTIAL FILES")
all_ok &= check_file_exists("main.go", "Go server main file")
all_ok &= check_file_exists("inference.py", "Python inference script")
all_ok &= check_file_exists("go.mod", "Go module file")
all_ok &= check_file_exists("go.sum", "Go dependencies")

# 2. Environment Module
print("\n2. ENVIRONMENT PACKAGE")
all_ok &= check_directory_exists("env", "Environment package directory")
all_ok &= check_file_exists("env/environment.go", "Main environment logic")
all_ok &= check_file_exists("env/models.go", "Data models")
all_ok &= check_file_exists("env/rewards.go", "Reward computation")
all_ok &= check_file_exists("env/faults.go", "Fault system")
all_ok &= check_file_exists("env/tasks.go", "Task definitions")

# 3. Python Module
print("\n3. PYTHON PACKAGE")
all_ok &= check_directory_exists("python", "Python package directory")
all_ok &= check_file_exists("python/__init__.py", "Python package init")
all_ok &= check_file_exists("python/models.py", "Python models")
all_ok &= check_file_size("python/requirements.txt", 100, "Python requirements")

# 4. Notebooks
print("\n4. NOTEBOOKS")
all_ok &= check_file_size("scripts/gridmind_grpo_colab.ipynb", 20000, "Colab notebook (≥20KB)")

# 5. Dashboard
print("\n5. DASHBOARD")
all_ok &= check_directory_exists("dashboard", "Dashboard directory")
all_ok &= check_file_exists("dashboard/server.py", "Dashboard server")
all_ok &= check_file_exists("dashboard/static/index.html", "Dashboard HTML")
all_ok &= check_file_exists("dashboard/static/dashboard.js", "Dashboard JavaScript")

# 6. Test Files
print("\n6. TEST/DEMO FILES")
all_ok &= check_file_exists("scripts/demo_run.py", "Demo runner")
all_ok &= check_file_exists("scripts/full_demo.py", "Full demo")
all_ok &= check_file_exists("tests/environment_test.go", "Go tests")

# 7. README & Docs
print("\n7. DOCUMENTATION")
all_ok &= check_file_exists("README.md", "README")
all_ok &= check_file_exists("HF_BLOG_POST.md", "Blog post")

# 8. Key Features Check
print("\n8. KEY FEATURES (Code Inspection)")
try:
    with open("inference.py", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()
        has_coordinator = "--coordinator" in content and "coordinator_step" in content
        has_curriculum = "CurriculumManager" in content
        has_planning = "--use-planning" in content and "simulate" in content
        status = "✓" if has_coordinator else "✗"
        print(f"  {status} Multi-Agent Coordinator mode (Theme 1)")
        status = "✓" if has_curriculum else "✗"
        print(f"  {status} Curriculum Learning (Theme 4)")
        status = "✓" if has_planning else "✗"
        print(f"  {status} World Modeling (/simulate) (Theme 3)")
        all_ok &= has_coordinator and has_curriculum and has_planning
except Exception as e:
    print(f"  ✗ Could not read inference.py: {e}")
    all_ok = False

try:
    with open("main.go", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()
        has_coord_reset = "handleCoordinatorReset" in content
        has_coord_step = "handleCoordinatorStep" in content
        has_simulate = "handleSimulate" in content
        has_reset = "handleReset" in content
        status = "✓" if has_coord_reset else "✗"
        print(f"  {status} /coordinator/reset endpoint")
        status = "✓" if has_coord_step else "✗"
        print(f"  {status} /coordinator/step endpoint")
        status = "✓" if has_simulate else "✗"
        print(f"  {status} /simulate endpoint (world modeling)")
        status = "✓" if has_reset else "✗"
        print(f"  {status} /reset endpoint (task 1-4 support)")
        all_ok &= has_coord_reset and has_coord_step and has_simulate and has_reset
except Exception as e:
    print(f"  ✗ Could not read main.go: {e}")
    all_ok = False

# 9. Test Quick Functionality
print("\n9. QUICK FUNCTIONALITY TEST")
try:
    import requests
    health = requests.get("http://localhost:7860/health", timeout=5)
    if health.status_code == 200:
        print(f"  ✓ Server health check passed (port 7860)")
    else:
        print(f"  ✗ Server health check failed ({health.status_code})")
        all_ok = False
except Exception as e:
    print(f"  ✗ Could not reach server: {e}")
    all_ok = False

# Final Summary
print("\n" + "=" * 70)
if all_ok:
    print("✓ PROJECT READY FOR SUBMISSION")
    print("=" * 70)
    sys.exit(0)
else:
    print("✗ SOME CHECKS FAILED - REVIEW REQUIRED")
    print("=" * 70)
    sys.exit(1)
