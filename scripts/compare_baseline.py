#!/usr/bin/env python3
"""
GridMind-RL — Baseline Comparison
===================================
Loads heuristic and LLM baseline JSON files, prints a markdown table
showing scores per task and the improvement delta.

Usage:
    python scripts/compare_baseline.py
    python scripts/compare_baseline.py --heuristic results/heuristic.json --llm results/llm.json
    python scripts/compare_baseline.py --save       # also writes results/comparison.md
"""

import json
import argparse
from pathlib import Path

DEFAULT_HEURISTIC = "baseline_scores_heuristic.json"
DEFAULT_LLM       = "baseline_scores.json"
DEFAULT_TRAINED   = "results/training_log.csv"

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def extract_scores(data):
    """Return {task_id: score} from either format."""
    if data is None:
        return {}
    # Format 1: {"task_averages": {"1": 0.72, ...}}
    if "task_averages" in data:
        return {int(k): v for k, v in data["task_averages"].items()}
    # Format 2: {"all_results": [{"task_id": 1, "score": 0.72}, ...]}
    scores = {}
    for r in data.get("all_results", []):
        tid = r.get("task_id")
        sc  = r.get("score", 0)
        if tid is not None:
            scores.setdefault(tid, []).append(sc)
    return {tid: sum(v)/len(v) for tid, v in scores.items()}

def delta_str(a, b):
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"

def arrow(a, b):
    if a is None or b is None: return " "
    return "↑" if b > a else ("↓" if b < a else "=")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", default=DEFAULT_HEURISTIC)
    parser.add_argument("--llm",       default=DEFAULT_LLM)
    parser.add_argument("--trained",   default=None,
                        help="JSON from fine-tuned model (optional)")
    parser.add_argument("--save",      action="store_true",
                        help="Save output to results/comparison.md")
    args = parser.parse_args()

    h_data  = load(args.heuristic)
    llm_data = load(args.llm)
    tr_data  = load(args.trained) if args.trained else None

    h_scores  = extract_scores(h_data)
    llm_scores = extract_scores(llm_data)
    tr_scores  = extract_scores(tr_data)

    task_names = {
        1: "Cost Minimization",
        2: "Constrained Temperature",
        3: "Full Demand-Response",
        4: "Instruction Following",
    }
    all_tasks = sorted(set(list(h_scores) + list(llm_scores) + list(tr_scores)) or [1,2,3,4])

    lines = []
    lines.append("# GridMind-RL — Baseline Comparison\n")

    # ── Model metadata ────────────────────────────────────────────────────────
    if h_data:
        lines.append(f"- Heuristic file : `{args.heuristic}`")
    if llm_data:
        model = llm_data.get("model", "unknown")
        lines.append(f"- LLM file       : `{args.llm}` (model: `{model}`)")
    if tr_data:
        lines.append(f"- Trained file   : `{args.trained}`")
    lines.append("")

    # ── Score table ───────────────────────────────────────────────────────────
    has_trained = bool(tr_scores)
    if has_trained:
        header = "| Task | Task Name | Heuristic | Zero-Shot LLM | Fine-Tuned | Δ (LLM→FT) |"
        sep    = "|------|-----------|-----------|---------------|------------|------------|"
    else:
        header = "| Task | Task Name | Heuristic | Zero-Shot LLM | Δ (H→LLM) |"
        sep    = "|------|-----------|-----------|---------------|-----------|"

    lines.append(header)
    lines.append(sep)

    for tid in all_tasks:
        name = task_names.get(tid, f"Task {tid}")
        h   = h_scores.get(tid)
        llm = llm_scores.get(tid)
        tr  = tr_scores.get(tid)

        h_s   = f"{h:.4f}"   if h   is not None else "—"
        llm_s = f"{llm:.4f}" if llm is not None else "—"
        tr_s  = f"{tr:.4f}"  if tr  is not None else "—"

        if has_trained:
            d = delta_str(llm, tr)
            a = arrow(llm, tr)
            lines.append(f"| {tid} | {name} | {h_s} | {llm_s} | {tr_s} | {a} {d} |")
        else:
            d = delta_str(h, llm)
            a = arrow(h, llm)
            lines.append(f"| {tid} | {name} | {h_s} | {llm_s} | {a} {d} |")

    lines.append("")

    # ── Summary stats ─────────────────────────────────────────────────────────
    if h_scores and llm_scores:
        common = [t for t in all_tasks if t in h_scores and t in llm_scores]
        if common:
            avg_h   = sum(h_scores[t]   for t in common) / len(common)
            avg_llm = sum(llm_scores[t] for t in common) / len(common)
            gain    = (avg_llm - avg_h) / avg_h * 100 if avg_h else 0
            lines.append(f"**Overall averages** (Tasks {common})")
            lines.append(f"- Heuristic    : `{avg_h:.4f}`")
            lines.append(f"- Zero-Shot LLM: `{avg_llm:.4f}` ({gain:+.1f}% vs heuristic)")
            if tr_scores:
                common_tr = [t for t in common if t in tr_scores]
                if common_tr:
                    avg_tr = sum(tr_scores[t] for t in common_tr) / len(common_tr)
                    gain_tr = (avg_tr - avg_llm) / avg_llm * 100 if avg_llm else 0
                    lines.append(f"- Fine-Tuned   : `{avg_tr:.4f}` ({gain_tr:+.1f}% vs zero-shot)")
            lines.append("")

    # ── Missing files note ────────────────────────────────────────────────────
    missing = []
    if not h_data:
        missing.append(f"`{args.heuristic}` — run: python inference.py --fast-mode --episodes 3 --output {args.heuristic}")
    if not llm_data:
        missing.append(f"`{args.llm}` — run: python inference.py --episodes 3 --output {args.llm}")
    if missing:
        lines.append("## To generate missing files\n")
        for m in missing:
            lines.append(f"- {m}")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    if args.save:
        out_path = Path("results/comparison.md")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(output)
        print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
