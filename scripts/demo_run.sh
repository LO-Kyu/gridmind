#!/bin/bash
# GridMind-RL Judge Demo Script
# Runs a 3-minute before/after story showing heuristic vs LLM performance.

echo "═══════════════════════════════════════════════════════"
echo "  GridMind-RL — 3-Minute Judge Demo"
echo "  Theme: Multi-Agent Building Energy Management"
echo "═══════════════════════════════════════════════════════"
echo ""

ENV_URL="${ENV_URL:-http://localhost:7860}"

echo "[1/5] Checking environment health..."
curl -s "$ENV_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Server: {\"OK\" if d.get(\"status\")==\"ok\" else \"FAIL\"}')"
echo ""

echo "[2/5] Showing available tasks..."
curl -s "$ENV_URL/tasks" | python3 -c "
import sys,json
tasks = json.load(sys.stdin)
for t in tasks:
    print(f'  Task {t[\"id\"]}: {t[\"name\"]} ({t[\"difficulty\"]})')
"
echo ""

echo "[3/5] Running HEURISTIC baseline (Task 3 — demand response)..."
echo "  Policy: Fixed rules — charge at night, shed on grid stress, schedule batches"
python3 inference.py --fast-mode --task 3 --episodes 2 --output /tmp/heuristic_result.json 2>&1 | grep -E "score|grade" | head -5
HEURISTIC_SCORE=$(python3 -c "import json; print(json.load(open('/tmp/heuristic_result.json'))['overall_average'])" 2>/dev/null)
echo "  Heuristic avg score: ${HEURISTIC_SCORE:-TBD}"
echo ""

echo "[4/5] Running LLM AGENT (Task 3 — demand response)..."
echo "  Policy: Qwen2.5-7B-Instruct with world-model planning"
python3 inference.py --task 3 --episodes 2 --output /tmp/llm_result.json 2>&1 | grep -E "score|grade" | head -5
LLM_SCORE=$(python3 -c "import json; print(json.load(open('/tmp/llm_result.json'))['overall_average'])" 2>/dev/null)
echo "  LLM avg score: ${LLM_SCORE:-TBD}"
echo ""

echo "[5/5] Demonstrating FAULT INJECTION..."
curl -s -X POST "$ENV_URL/reset" -H "Content-Type: application/json" -d '{"task_id":3}' > /dev/null
curl -s -X POST "$ENV_URL/fault" -H "Content-Type: application/json" \
  -d '{"type":"chiller_failure","building_id":0,"severity":0.6,"duration_steps":10}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Injected: {d.get(\"fault_type\",\"N/A\")} (severity={d.get(\"severity\",0):.1f})')" 2>/dev/null || \
  echo "  (Fault endpoint not found — checking /step for active_faults)"
curl -s "$ENV_URL/state" | python3 -c "import sys,json; d=json.load(sys.stdin); faults=d.get('active_faults',[]); print(f'  Active faults: {len(faults)}')" 2>/dev/null
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════"
echo "  Heuristic baseline: ${HEURISTIC_SCORE:-TBD}"
echo "  LLM fine-tuned:     ${LLM_SCORE:-TBD}"
echo "  Improvement:       $(python3 -c "print(f'{(float(\"$LLM_SCORE\") - float(\"$HEURISTIC_SCORE\")) / float(\"$HEURISTIC_SCORE\") * 100:.1f}%')" 2>/dev/null || echo '  Run both agents to see delta')"
echo ""
echo "  Dashboard: $ENV_URL/dashboard"
echo "  HF Space:  https://lo-kyu-gridmind.hf.space"
echo "═══════════════════════════════════════════════════════"