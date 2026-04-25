#!/bin/bash
# GridMind-RL Baseline Scorer
# ----------------------------
# Runs two baseline policies (heuristic and zero-shot LLM) before training
# and saves scores to results/ for comparison with post-training results.

set -e
mkdir -p results

ENV_URL="${ENV_URL:-http://localhost:7860}"
EPISODES="${EPISODES:-3}"

echo "=== GridMind-RL Baseline Scorer ==="
echo "Environment: $ENV_URL"
echo "Episodes per task: $EPISODES"
echo ""

# --- Baseline 1: Heuristic Rule-Based Policy ---
echo "▶  Running Heuristic Baseline (no LLM)..."
python inference.py \
    --fast-mode \
    --episodes "$EPISODES" \
    --env-url "$ENV_URL" \
    --output results/baseline_heuristic.json

echo "✅ Heuristic baseline saved to results/baseline_heuristic.json"
echo ""

# --- Baseline 2: Zero-Shot LLM (pre-training) ---
echo "▶  Running Zero-Shot LLM Baseline (pre-training)..."
python inference.py \
    --episodes "$EPISODES" \
    --env-url "$ENV_URL" \
    --output results/baseline_zeroshot.json

echo "✅ Zero-shot LLM baseline saved to results/baseline_zeroshot.json"
echo ""

# --- Print Summary ---
echo "=== Baseline Summary ==="
python - <<'EOF'
import json, os

for label, path in [("Heuristic", "results/baseline_heuristic.json"),
                    ("Zero-Shot LLM", "results/baseline_zeroshot.json")]:
    if not os.path.exists(path):
        print(f"  {label}: file not found")
        continue
    with open(path) as f:
        data = json.load(f)
    avgs = data.get("task_averages", {})
    overall = data.get("overall_average", 0)
    print(f"\n  {label}:")
    for tid in ["1","2","3"]:
        print(f"    Task {tid}: {avgs.get(tid, 0):.4f}")
    print(f"    Overall: {overall:.4f}")
EOF

echo ""
echo "Run 'python scripts/train_unsloth.py' to start fine-tuning."
echo "After training, compare scores with results/post_training.json."
