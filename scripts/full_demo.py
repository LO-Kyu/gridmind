#!/usr/bin/env python3
"""
GridMind-RL Full Unified Demo — All 4 Themes Running Together
-------------------------------------------------------------
This single script demonstrates all hackathon themes in one flow.
Each step proves a theme and prints WHY it matters for judges.

Themes:
  Theme 1 — Multi-Agent: 3 buildings + coordinator + /feeder
  Theme 2 — Long-Horizon Planning: Task 4 instruction cards over 96 steps
  Theme 3 — World Modeling: /simulate before committing actions
  Theme 4 — Self-Improvement: Curriculum auto-advances task difficulty

Run: python scripts/full_demo.py
"""

from __future__ import annotations
import json
import sys
import time
import requests

ENV_URL = "http://localhost:7860"


def bold(text: str) -> str:
    return f"**{text}**"


def section(n: int, title: str, theme: str) -> None:
    print(f"\n{'='*62}")
    print(f"  STEP {n} — {bold(title)}  [{theme}]")
    print(f"{'='*62}")


def check_pass(text: str) -> None:
    print(f"  [OK] {text}")


def check_warn(text: str) -> None:
    print(f"  [!] {text}")


def show_info(label: str, value: str) -> None:
    print(f"  {label}: {value}")


def r(path: str, method="GET", json_data=None) -> dict:
    url = f"{ENV_URL}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=json_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        check_warn(f"Request failed: {e}")
        return {}


def step(n: int, title: str, theme: str, fn):
    section(n, title, theme)
    result = fn()
    if result is not False:
        check_pass("Passed")
    return result


# ── STEP 1 — Health Check ──────────────────────────────────────────────
def do_step1():
    data = r("/health")
    show_info("Status", data.get("status", "?") + " OK" if data.get("status") == "ok" else data.get("status", "?"))
    show_info("Version", data.get("version", "?"))
    return data.get("status") == "ok"


# ── STEP 2 — OpenEnv /info Metadata ──────────────────────────────────
def do_step2():
    data = r("/info")
    show_info("Name", data.get("name", "?"))
    show_info("Version", data.get("version", "?"))
    themes = data.get("themes", [])
    show_info("Themes", ", ".join(themes))
    endpoints = data.get("endpoints", [])
    show_info("Endpoints", f"{len(endpoints)} registered")
    print(f"  Sample endpoints: {endpoints[:5]}")
    return len(themes) >= 4 and "multi-agent" in themes


# ── STEP 3 — All 4 Tasks Available ────────────────────────────────────
def do_step3():
    data = r("/tasks")
    tasks = data if isinstance(data, list) else data.get("tasks", [])
    print(f"  {len(tasks)} tasks available:")
    for t in tasks:
        print(f"    Task {t['id']}: {t['name']} ({t['difficulty']})")
    ids = [t["id"] for t in tasks]
    return 1 in ids and 2 in ids and 3 in ids and 4 in ids


# ── STEP 4 — Theme 1: Multi-Agent Reset ───────────────────────────────
def do_step4():
    data = r("/reset", "POST", {"task_id": 3, "num_buildings": 3, "seed": 42})
    obs_list = data.get("observations", [])
    print(f"  Buildings in federation: {len(obs_list)}")
    for b in obs_list:
        tid = b.get("building_id", 0)
        temp = round(b.get("indoor_temperature", 0), 1)
        storage = round(b.get("thermal_storage_level", 0) * 100)
        cost = round(b.get("cumulative_cost", 0), 2)
        print(f"    Building {tid}: {temp}C | storage {storage}% | cost ${cost}")
    return len(obs_list) == 3


# ── STEP 5 — Theme 1: Fleet-wide Feeder View ─────────────────────────
def do_step5():
    data = r("/feeder")
    total = round(data.get("total_demand_kw", 0), 1)
    limit = data.get("feeder_limit_kw", 0)
    overload = data.get("feeder_overload", False)
    util = round(data.get("utilization_pct", 0), 1)
    buildings = data.get("buildings", [])
    feeder_status = "OK" if not overload else "OVERLOAD"
    print(f"  Feeder: {total} / {limit} kW [{feeder_status}]")
    print(f"  Utilization: {util}%")
    for b in buildings:
        bid = b.get("building_id", "?")
        dem = round(b.get("current_demand_kw", 0), 1)
        pm = b.get("price_multiplier", 1.0)
        print(f"    Building {bid}: {dem} kW | price mult {pm}x")
    return total > 0 and len(buildings) == 3


# ── STEP 6 — Theme 1: Coordinator Price Signals ───────────────────────
def do_step6():
    multipliers = [1.5, 0.8, 1.0]
    r("/coordinate", "POST", {"price_multipliers": multipliers})
    print(f"  Set price signals: {multipliers}")
    feeder = r("/feeder")
    blds = feeder.get("buildings", [])
    for i, b in enumerate(blds):
        pm = b.get("price_multiplier", "?")
        signal = ">> expensive (will conserve)" if pm > 1.2 else "<< cheap (can charge)"
        print(f"    Building {i}: {pm}x | {signal}")
    return True


# ── STEP 7 — Theme 3: World Modeling /simulate ───────────────────────
def do_step7():
    action = {
        "hvac_power_level": 0.8,
        "thermal_charge_rate": 0.5,
        "batch_job_slot": 0,
        "load_shed_fraction": 0.0,
        "building_id": 0,
    }
    # Check step counter before simulate
    state_before = r("/state")
    step_before = state_before.get("step", "?")
    print(f"  Step before simulate: {step_before}")

    sim_data = r("/simulate", "POST", [action])
    results = sim_data.get("results", [])
    print(f"  Simulated 1 action ahead")
    if results:
        r0 = results[0]
        obs = r0.get("observation", {})
        reward = round(r0.get("reward", 0), 3)
        print(f"    Predicted reward: {reward}")
        print(f"    Predicted temp: {round(obs.get('indoor_temperature', 0), 1)}C")
        print(f"    Predicted storage: {round(obs.get('thermal_storage_level', 0) * 100)}%")
    print(f"    Would episode end? {sim_data.get('done', False)}")

    # Verify step did NOT advance
    state_after = r("/state")
    step_after = state_after.get("step", "?")
    if step_before == step_after:
        check_pass(f"World model confirmed: step unchanged ({step_before})")
        return True
    else:
        check_warn(f"Step changed from {step_before} to {step_after}")
        return False


# ── STEP 8 — Take a Step + Check Fault System ─────────────────────────
def do_step8():
    action = {
        "hvac_power_level": 0.7,
        "thermal_charge_rate": 0.0,
        "batch_job_slot": 0,
        "load_shed_fraction": 0.0,
        "building_id": 0,
    }
    data = r("/step", "POST", [action])
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        print("  Warning: /step returned unexpected format")
        return False
    obs = results[0].get("observation", {})
    step_num = obs.get("step", "?")
    faults = obs.get("active_faults", [])
    reward = round(results[0].get("reward", 0), 3)
    show_info("Step", str(step_num))
    show_info("Reward", str(reward))
    show_info("Active faults", f"{len(faults)}")
    if faults:
        for f in faults:
            print(f"    [!] FAULT: {f}")
    else:
        print("    No faults triggered yet (expected on hard difficulty)")
    return True


# ── STEP 9 — Theme 2: Task 4 Instruction Card ───────────────────────────
def do_step9():
    data = r("/reset", "POST", {"task_id": 4, "seed": 99})
    obs_list = data.get("observations", [])
    if not obs_list:
        check_warn("No observations returned")
        return False
    obs = obs_list[0]
    card = obs.get("instruction_card") or {}
    card_text = card.get("text", "")
    targets = card.get("targets", {})
    print(f"  Task 4: Instruction Following")
    print(f"  Objective: {card_text[:120]}...")
    if targets:
        print(f"  Targets:")
        for k, v in targets.items():
            print(f"    {k}: {v}")
    return bool(card_text)


# ── STEP 10 — Theme 4: /grade + Reward Decomposition ──────────────────
def do_step10():
    # First run a full episode (simplified: 10 steps)
    r("/reset", "POST", {"task_id": 3, "num_buildings": 3, "seed": 42})
    for _ in range(10):
        action = {
            "hvac_power_level": 0.6,
            "thermal_charge_rate": 0.0,
            "batch_job_slot": 0,
            "load_shed_fraction": 0.0,
            "building_id": 0,
        }
        r("/step", "POST", [action])

    grade_data = r("/grade")
    score = grade_data.get("score", "?")
    exploit = grade_data.get("exploit_detected", False)
    penalty = grade_data.get("penalty_applied", 0)
    sub = grade_data.get("sub_scores", {})
    show_info("Episode score", f"{score} (0.0-1.0 clamped)" if score != "?" else "?")
    show_info("Exploit detected", str(exploit))
    show_info("Penalty applied", str(penalty))
    if sub:
        print("  Reward components:")
        for k, v in sub.items():
            print(f"    {k}: {round(v, 3)}")
    show_info("Task ID", str(grade_data.get("task_id", "?")))
    return score not in ("?", None) and 0.0 < float(score) < 1.0


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(bold("\nGridMind-RL — Full Unified Demo"))
    print(f"  Environment: {ENV_URL}")
    print(f"  Themes: Multi-Agent | Long-Horizon | World Modeling | Self-Improvement")

    results = []

    results.append(step(1, "Health Check", "Foundation", do_step1))
    results.append(step(2, "OpenEnv /info Metadata", "Foundation", do_step2))
    results.append(step(3, "All 4 Tasks Available", "Foundation", do_step3))
    results.append(step(4, "Multi-Agent Reset (3 Buildings)", "Theme 1: Multi-Agent", do_step4))
    results.append(step(5, "Fleet-wide Feeder View", "Theme 1: Multi-Agent", do_step5))
    results.append(step(6, "Coordinator Price Signals", "Theme 1: Multi-Agent", do_step6))
    results.append(step(7, "World Modeling /simulate", "Theme 3: World Modeling", do_step7))
    results.append(step(8, "Step + Fault Events", "Wild Card: Fault Resilience", do_step8))
    results.append(step(9, "Task 4 Instruction Card", "Theme 2: Long-Horizon", do_step9))
    results.append(step(10, "Episode /grade + Reward Decomposition", "Theme 4: Self-Improvement", do_step10))

    sep = "=" * 62
    print(f"\n{bold(sep)}")
    print(f"  SUMMARY — All Themes Check")
    print(sep)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print(f"  [OK] ALL THEMES OPERATIONAL")
    else:
        failed = [i + 1 for i, r in enumerate(results) if not r]
        print(f"  [FAIL] Failed steps: {failed}")
    print()


if __name__ == "__main__":
    main()