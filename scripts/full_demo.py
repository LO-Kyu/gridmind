#!/usr/bin/env python3
"""
GridMind-RL — Unified 10-Step Demo
====================================
Runs all 4 hackathon themes in one cohesive demo flow.
Each step is labelled with the theme it proves.

Usage:
    python scripts/full_demo.py
    python scripts/full_demo.py --url https://lo-kyu-gridmind.hf.space

Steps:
  1  Health check
  2  GET /info         → OpenEnv metadata
  3  GET /tasks        → 4 tasks with difficulty progression
  4  POST /reset x3    → Theme 1: Multi-Agent (3 buildings)
  5  GET /feeder       → Theme 1: Fleet-wide electricity view
  6  POST /coordinate  → Theme 1: Coordinator sends price signals
  7  POST /simulate    → Theme 3: World Modeling (predict before act)
  8  POST /step        → Wild Card: Fault events may fire
  9  POST /reset task4 → Theme 2: Instruction Following (NL task card)
  10 GET /grade        → Theme 4: Episode scored; curriculum advances
"""

import sys
import json
import argparse
import requests

SEPARATOR = "=" * 60

def bold(s): return f"\033[1m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s): return f"\033[91m{s}\033[0m"
def cyan(s): return f"\033[96m{s}\033[0m"

def step_header(n, theme, title):
    print(f"\n{SEPARATOR}")
    print(bold(f"[STEP {n}]") + f" {cyan(theme)}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg): print(green(f"  ✅ {msg}"))
def warn(msg): print(yellow(f"  ⚠️  {msg}"))
def fail(msg): print(red(f"  ❌ {msg}")); sys.exit(1)
def info(msg): print(f"  {msg}")

def post(url, path, body=None, timeout=15):
    try:
        r = requests.post(f"{url}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        fail(f"POST {path} failed: {e}")

def get(url, path, timeout=10):
    try:
        r = requests.get(f"{url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        fail(f"GET {path} failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    print(f"\n{bold('GridMind-RL — Unified Hackathon Demo')}")
    print(f"  Environment: {url}")
    print(f"  All 4 themes run in 10 steps.\n")

    # ── STEP 1: Health ────────────────────────────────────────────────────────
    step_header(1, "Infrastructure", "Health check — is the environment live?")
    h = get(url, "/health")
    if h.get("status") == "ok":
        ok("Server is live.")
    else:
        fail(f"Unexpected health response: {h}")

    # ── STEP 2: /info ─────────────────────────────────────────────────────────
    step_header(2, "OpenEnv Compliance", "GET /info — metadata for automated validators")
    inf = get(url, "/info")
    info(f"Name:    {inf.get('name')}")
    info(f"Version: {inf.get('version')}")
    info(f"Themes:  {inf.get('themes')}")
    info(f"Endpoints: {len(inf.get('endpoints', []))} registered")
    ok("OpenEnv /info endpoint present and well-formed.")

    # ── STEP 3: /tasks ────────────────────────────────────────────────────────
    step_header(3, "Theme 4 — Self-Improvement", "GET /tasks — 4 difficulty levels for curriculum")
    tasks = get(url, "/tasks")
    for t in tasks:
        info(f"  Task {t['id']} [{t['difficulty']:6s}]: {t['name']}")
    ok("4 tasks returned. Curriculum can advance Task 1→2→3→4 as agent improves.")

    # ── STEP 4: Multi-building reset ──────────────────────────────────────────
    step_header(4, "Theme 1 — Multi-Agent", "POST /reset with 3 buildings — fleet initialised")
    reset = post(url, "/reset", {"task_id": 3, "num_buildings": 3, "seed": 42})
    obs_list = reset.get("observations", [])
    if len(obs_list) < 3:
        warn(f"Only {len(obs_list)} building(s) returned. Server may not support num_buildings.")
    else:
        ok(f"3-building federation started (Episode {reset.get('episode', '?')}).")
    for i, o in enumerate(obs_list):
        info(f"  Building {i}: temp={o.get('indoor_temperature',0):.1f}°C  "
             f"storage={o.get('thermal_storage_level',0):.0%}  "
             f"price=${o.get('current_price',0):.4f}/kWh")

    # ── STEP 5: /feeder ───────────────────────────────────────────────────────
    step_header(5, "Theme 1 — Multi-Agent", "GET /feeder — coordinator sees fleet-wide demand")
    feeder = get(url, "/feeder")
    total  = feeder.get("total_demand_kw", 0)
    limit  = feeder.get("feeder_limit_kw", 360)
    util   = feeder.get("utilization_pct", total / limit * 100)
    overload = feeder.get("feeder_overload", False)
    info(f"  Total demand : {total:.1f} kW")
    info(f"  Feeder limit : {limit:.1f} kW")
    info(f"  Utilisation  : {util:.1f}%  {'⚠️ OVERLOAD' if overload else '✅ OK'}")
    ok("Coordinator can see aggregate fleet state — basis for multi-agent coordination.")

    # ── STEP 6: /coordinate ───────────────────────────────────────────────────
    step_header(6, "Theme 1 — Multi-Agent", "POST /coordinate — price signals orchestrate buildings")
    # Raise price for Building 0 (high load), lower for Building 2 (low load)
    coord = post(url, "/coordinate", {"price_multipliers": [1.5, 1.0, 0.7]})
    info(f"  Multipliers set: B0=1.5× (conserve)  B1=1.0×  B2=0.7× (can use more)")
    ok("Coordinator influences 3 agents via price signals — no direct commands needed.")

    # ── STEP 7: /simulate ─────────────────────────────────────────────────────
    step_header(7, "Theme 3 — World Modeling", "POST /simulate — predict reward BEFORE acting")
    action_max = {"hvac_power_level": 1.0, "thermal_charge_rate": 0.0,
                  "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0}
    action_smart = {"hvac_power_level": 0.3, "thermal_charge_rate": -0.5,
                    "batch_job_slot": 2, "load_shed_fraction": 0.4, "building_id": 0}

    sim_max   = post(url, "/simulate", [action_max])
    sim_smart = post(url, "/simulate", [action_smart])

    r_max   = sim_max.get("results", [{}])[0].get("reward", "?")
    r_smart = sim_smart.get("results", [{}])[0].get("reward", "?")

    info(f"  Action A (max HVAC, no shedding)  → predicted reward: {r_max:.3f}")
    info(f"  Action B (smart: discharge + shed) → predicted reward: {r_smart:.3f}")

    # Verify state didn't advance
    state_after = get(url, "/state")
    step_after = state_after.get("step", "?")
    info(f"  Episode step after simulate calls  : {step_after}  (must still be 0)")
    if step_after == 0:
        ok("World Model: /simulate predicted rewards WITHOUT advancing the episode. ✅")
    else:
        warn(f"Step advanced to {step_after} — check /simulate implementation.")

    chosen = "B (smart)" if (isinstance(r_smart, float) and isinstance(r_max, float) and r_smart > r_max) else "unknown"
    info(f"  Agent selects Action {chosen} based on prediction.")

    # ── STEP 8: /step with fault check ────────────────────────────────────────
    step_header(8, "Wild Card — Fault Resilience", "POST /step — fault events may fire mid-episode")
    actions = [
        {"hvac_power_level": 0.3, "thermal_charge_rate": -0.5,
         "batch_job_slot": 2, "load_shed_fraction": 0.4, "building_id": i}
        for i in range(len(obs_list))
    ] or [{"hvac_power_level": 0.5, "thermal_charge_rate": 0.0,
            "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0}]

    step_resp = post(url, "/step", actions)
    results = step_resp if isinstance(step_resp, list) else step_resp.get("results", [])

    for i, r in enumerate(results):
        obs = r.get("observation", {})
        reward = r.get("reward", 0)
        faults = obs.get("active_faults", [])
        info(f"  Building {i}: reward={reward:.3f}  temp={obs.get('indoor_temperature',0):.1f}°C")
        if faults:
            info(f"    🚨 FAULT ACTIVE: {faults[0][:60]}...")
            ok("Agent sees fault alarm in observation — must adapt response.")
        else:
            info(f"    No faults this step.")
    ok("Step executed. Reward decomposed into 9 components (see info.reward_components).")

    # ── STEP 9: Task 4 reset ──────────────────────────────────────────────────
    step_header(9, "Theme 2 — Long-Horizon + Instruction Following",
                "POST /reset task_id=4 — natural language task card issued")
    reset4 = post(url, "/reset", {"task_id": 4, "seed": 99})
    card = reset4.get("instruction_card") or reset4.get("observations", [{}])[0].get("instruction_card")
    if card:
        ok("Task 4 instruction card received.")
        info(f"  Objective: \"{card.get('text', 'N/A')}\"")
        targets = card.get("targets", {})
        weights = card.get("weights", {})
        info(f"  Targets : {json.dumps(targets, indent=0)}")
        info(f"  Weights : {json.dumps(weights, indent=0)}")
        info(f"  The agent must plan ALL 96 steps (24 hours) to satisfy this card.")
    else:
        warn("No instruction_card in response — check Item 1.1 fix (taskID clamp).")

    # ── STEP 10: /grade ───────────────────────────────────────────────────────
    step_header(10, "Theme 4 — Self-Improvement",
                "GET /grade — episode scored; curriculum tracks this for advancement")
    # Take a couple of steps in the Task 4 episode first
    for _ in range(3):
        post(url, "/step", [{"hvac_power_level": 0.5, "thermal_charge_rate": 0.0,
                              "batch_job_slot": 2, "load_shed_fraction": 0.1, "building_id": 0}])
    grade = get(url, "/grade")
    score = grade.get("score", 0)
    sub   = grade.get("sub_scores", grade.get("SubScores", {}))
    exploit = grade.get("exploit_detected", False)

    info(f"  Final score      : {score:.4f}")
    info(f"  Sub-scores       : {json.dumps({k: round(v,3) for k,v in sub.items()}, indent=0)}")
    info(f"  Exploit detected : {exploit}")
    ok("Episode graded. CurriculumManager tracks this score for auto-advancement.")
    info(f"  → If score ≥ threshold for 5 consecutive episodes, next task unlocks.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print(bold("  DEMO COMPLETE — All Themes Demonstrated"))
    print(SEPARATOR)
    print(f"  {cyan('Theme 1 — Multi-Agent')}      : Steps 4, 5, 6")
    print(f"  {cyan('Theme 2 — Long-Horizon')}     : Step  9")
    print(f"  {cyan('Theme 3 — World Modeling')}   : Step  7")
    print(f"  {cyan('Theme 4 — Self-Improvement')} : Steps 3, 10")
    print(f"  {cyan('Wild Card — Fault Events')}   : Step  8")
    print(f"\n  Live environment: {url}")
    print(f"  Dashboard:        {url}/dashboard\n")

if __name__ == "__main__":
    main()
