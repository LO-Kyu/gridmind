#!/usr/bin/env python3
"""
GridMind-RL Baseline Comparison Script
Loads heuristic + LLM baseline scores and prints a delta table.
"""
import json
import os
import sys

def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

heuristic = load_json("results/baseline_scores_heuristic.json")
llm_baseline = load_json("baseline_scores.json")

if not heuristic:
    print("Run: python inference.py --fast-mode --episodes 3 --output results/baseline_scores_heuristic.json")
    sys.exit(1)

tasks = [1, 2, 3, 4]
task_names = {1: "Cost Minimization", 2: "Temperature Mgmt", 3: "Demand Response", 4: "Instruction Following"}

print("\n" + "=" * 72)
print(" GridMind-RL — Baseline Comparison")
print("=" * 72)
print(f" {'Policy':<25} {'Task 1':>9} {'Task 2':>9} {'Task 3':>9} {'Task 4':>9} {'Avg':>9}")
print("-" * 72)

h_averages = heuristic.get("task_averages", {})
h_row = ["Heuristic Baseline"]
h_total = 0
for t in tasks:
    s = h_averages.get(str(t), 0.0)
    h_row.append(f"{s:.3f}")
    h_total += s
h_row.append(f"{h_total/4:.3f}")
print(f" {''.join(h_row):<25} {' '.join(h_row[1:])}")

if llm_baseline and llm_baseline.get("model") not in ("<your-active-model>", None):
    llm_averages = llm_baseline.get("task_averages", {})
    llm_row = [llm_baseline.get("model", "LLM Baseline")]
    llm_total = 0
    for t in tasks:
        s = llm_averages.get(str(t), 0.0)
        llm_row.append(f"{s:.3f}")
        llm_total += s
    llm_row.append(f"{llm_total/4:.3f}")
    print(f" {llm_row[0]:<25} {' '.join(llm_row[1:])}")
else:
    print(f" {'LLM Baseline':<25} {'--':>9} {'--':>9} {'--':>9} {'--':>9} {'--':>9}")
    print("  (Run: python inference.py --task N --episodes 3 --output baseline_scores.json)")

print("-" * 72)
print("\n Delta vs Heuristic:")
if llm_baseline and llm_baseline.get("model") not in ("<your-active-model>", None):
    for t in tasks:
        h_s = h_averages.get(str(t), 0.0)
        l_s = llm_averages.get(str(t), 0.0)
        delta = l_s - h_s
        sign = "+" if delta >= 0 else ""
        print(f"   Task {t}: {sign}{delta:.3f} ({delta/h_s*100:+.1f}%)")
else:
    print("   Run LLM baseline to compute delta.")
print("=" * 72 + "\n")