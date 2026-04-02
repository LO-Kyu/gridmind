"""
GridMind-RL Pre-Submission Validator
--------------------------------------
Validates the Go environment server against all OpenEnv spec requirements.
Run with: python python/validate.py [--env-url http://localhost:7860]
"""

import argparse
import json
import sys
import time
import traceback
from typing import Any

import requests

ENV_URL = "http://localhost:7860"

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    line = f"  {icon} {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    return condition


def get(url: str, timeout: int = 10) -> requests.Response:
    return requests.get(url, timeout=timeout)


def post(url: str, payload: Any = None, timeout: int = 10) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def validate(env_url: str) -> bool:
    base = env_url.rstrip("/")
    results = []

    print("\n" + "=" * 50)
    print("  GridMind-RL OpenEnv Validation Report")
    print("=" * 50 + "\n")

    # ── 1. Health & ping ─────────────────────────────────────────────────────
    print("1. Health & Ping")
    try:
        r = get(f"{base}/health")
        results.append(check("GET /health returns 200", r.status_code == 200, f"got {r.status_code}"))
        data = r.json()
        results.append(check("Response has 'status' field", "status" in data))
        rp = get(f"{base}/ping")
        results.append(check("GET /ping returns 200", rp.status_code == 200, f"got {rp.status_code}"))
    except Exception as e:
        results.append(check("GET /health reachable", False, str(e)))
        print(f"\n  [FAIL] Cannot reach server at {base}. Is it running?\n")
        return False

    # ── 2. Reset endpoint ───────────────────────────────────────────────────
    print("\n2. Reset Endpoint")
    reset_resp = None
    try:
        r = post(f"{base}/reset", {"task_id": 1, "seed": 42, "num_buildings": 1})
        results.append(check("POST /reset returns 200", r.status_code == 200, f"got {r.status_code}"))
        reset_resp = r.json()
        results.append(check("Response has 'observations'", "observations" in reset_resp))
        results.append(check("Response has 'episode'", "episode" in reset_resp))
        results.append(check("Response has 'seed'", "seed" in reset_resp))
        results.append(check("Response has 'task_id'", "task_id" in reset_resp))

        obs_list = reset_resp.get("observations", [])
        results.append(check("observations is a list", isinstance(obs_list, list)))
        results.append(check("At least 1 observation returned", len(obs_list) >= 1))

        if obs_list:
            obs = obs_list[0]
            obs_fields = ["indoor_temperature", "thermal_storage_level", "process_demand",
                          "current_price", "grid_stress_signal", "carbon_intensity",
                          "hour_of_day", "batch_queue", "cumulative_cost", "step"]
            for field in obs_fields:
                results.append(check(f"obs has '{field}'", field in obs))

        # Seed reproducibility
        r2 = post(f"{base}/reset", {"task_id": 1, "seed": 42})
        d2 = r2.json()
        obs1 = reset_resp.get("observations", [{}])[0]
        obs2 = d2.get("observations", [{}])[0]
        same = (abs(obs1.get("indoor_temperature", 0) - obs2.get("indoor_temperature", 0)) < 1e-6)
        results.append(check("Same seed produces same initial obs", same))
    except Exception as e:
        results.append(check("POST /reset succeeds", False, str(e)))
        traceback.print_exc()

    # ── 3. Step endpoint ────────────────────────────────────────────────────
    print("\n3. Step Endpoint")
    try:
        # Reset fresh
        post(f"{base}/reset", {"task_id": 1, "seed": 100})
        action = {
            "hvac_power_level": 0.5,
            "thermal_charge_rate": 0.1,
            "batch_job_slot": 1,
            "load_shed_fraction": 0.0,
            "building_id": 0,
        }
        r = post(f"{base}/step", action)
        results.append(check("POST /step returns 200", r.status_code == 200))
        step_resp = r.json()

        step_fields = ["observation", "reward", "done", "info"]
        for f in step_fields:
            results.append(check(f"step response has '{f}'", f in step_resp))

        results.append(check("reward is numeric", isinstance(step_resp.get("reward"), (int, float))))
        results.append(check("done is boolean", isinstance(step_resp.get("done"), bool)))

        info = step_resp.get("info", {})
        results.append(check("info has 'reward_components'", "reward_components" in info))
        results.append(check("info has 'energy_used_kwh'", "energy_used_kwh" in info))

        rc = info.get("reward_components", {})
        rc_fields = ["cost_savings", "temp_constraint", "grid_response",
                     "deadline_penalty", "efficiency_bonus", "stability_penalty",
                     "carbon_reward", "total"]
        for f in rc_fields:
            results.append(check(f"reward_components has '{f}'", f in rc))

        # Test array action format
        r2 = post(f"{base}/step", [action])
        results.append(check("POST /step accepts array of actions", r2.status_code == 200))
    except Exception as e:
        results.append(check("POST /step succeeds", False, str(e)))
        traceback.print_exc()

    # ── 4. State endpoint ───────────────────────────────────────────────────
    print("\n4. State Endpoint")
    try:
        r = get(f"{base}/state")
        results.append(check("GET /state returns 200", r.status_code == 200))
        state = r.json()
        state_fields = ["buildings", "price_curve_episode", "carbon_curve_episode",
                        "episode", "step", "task_id", "done", "seed"]
        for f in state_fields:
            results.append(check(f"state has '{f}'", f in state))
        curve_n = 24  # EpisodeSteps/4 (96/4) downsamples to hourly points
        results.append(check("price_curve_episode has 24 entries",
                             len(state.get("price_curve_episode", [])) == curve_n))
        results.append(check("carbon_curve_episode has 24 entries",
                             len(state.get("carbon_curve_episode", [])) == curve_n))
    except Exception as e:
        results.append(check("GET /state succeeds", False, str(e)))

    # ── 5. Replay endpoint ──────────────────────────────────────────────────
    print("\n5. Replay Endpoint")
    try:
        r = get(f"{base}/replay")
        results.append(check("GET /replay returns 200", r.status_code == 200))
        replay = r.json()
        results.append(check("response has 'replay' list", "replay" in replay))
        results.append(check("response has 'steps' count", "steps" in replay))
    except Exception as e:
        results.append(check("GET /replay succeeds", False, str(e)))

    # ── 6. Grade endpoint ───────────────────────────────────────────────────
    print("\n6. Grade Endpoint")
    try:
        # Run quick 10-step episode
        post(f"{base}/reset", {"task_id": 1, "seed": 777})
        action = {"hvac_power_level": 0.3, "thermal_charge_rate": 0.0,
                  "batch_job_slot": 0, "load_shed_fraction": 0.0}
        done = False
        while not done:
            r2 = post(f"{base}/step", action)
            if r2.json().get("done"):
                done = True
        r = get(f"{base}/grade")
        results.append(check("GET /grade returns 200", r.status_code == 200))
        grade = r.json()
        grade_fields = ["task_id", "score", "sub_scores", "exploit_detected"]
        for f in grade_fields:
            results.append(check(f"grade has '{f}'", f in grade))
        score = grade.get("score", -1)
        results.append(check("score in [0.0, 1.0]", 0.0 <= score <= 1.0, f"score={score:.4f}"))
    except Exception as e:
        results.append(check("GET /grade succeeds", False, str(e)))

    # ── 7. Tasks endpoint ───────────────────────────────────────────────────
    print("\n7. Tasks Endpoint")
    try:
        r = get(f"{base}/tasks")
        results.append(check("GET /tasks returns 200", r.status_code == 200))
        tasks = r.json()
        results.append(check("returns list of 3 tasks", len(tasks) == 3))
        task_fields = ["id", "name", "description", "difficulty", "weights"]
        for f in task_fields:
            results.append(check(f"task has '{f}'", f in tasks[0]))
    except Exception as e:
        results.append(check("GET /tasks succeeds", False, str(e)))

    # ── 8. Metrics endpoint ─────────────────────────────────────────────────
    print("\n8. Metrics Endpoint (Prometheus)")
    try:
        r = get(f"{base}/metrics")
        results.append(check("GET /metrics returns 200", r.status_code == 200))
        content = r.text
        results.append(check("metrics contain step counter",
                             "gridmind_steps_total" in content))
        results.append(check("metrics contain latency gauge",
                             "gridmind_step_latency_ms_avg" in content))
    except Exception as e:
        results.append(check("GET /metrics succeeds", False, str(e)))

    # ── 9. Grader score variation ───────────────────────────────────────────
    print("\n9. Grader Score Variation (non-trivial scores)")
    scores_nonzero = []
    scores_nonone = []
    for seed in [10, 20, 30]:
        try:
            post(f"{base}/reset", {"task_id": 1, "seed": seed})
            # Two different policies
            for a in [0.1, 0.9]:
                post(f"{base}/reset", {"task_id": 1, "seed": seed})
                done = False
                while not done:
                    r2 = post(f"{base}/step", {"hvac_power_level": a, "thermal_charge_rate": 0,
                                          "batch_job_slot": 0, "load_shed_fraction": 0})
                    if r2.json().get("done"):
                        done = True
                g = requests.get(f"{base}/grade", timeout=10).json()
                sc = g.get("score", 0)
                scores_nonzero.append(sc > 0.01)
                scores_nonone.append(sc < 0.999)
        except Exception:
            pass
    results.append(check("Scores are not always 0.0", any(scores_nonzero)))
    results.append(check("Scores are not always 1.0", any(scores_nonone)))

    # ── Summary ─────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    pct = 100 * passed // total if total > 0 else 0

    print(f"\n" + "=" * 50)
    print(f"  Result: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("  ALL CHECKS PASSED - Ready for submission!")
    else:
        print(f"  {total - passed} checks failed. Fix errors above.")
    print("=" * 50 + "\n")

    return passed == total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", type=str, default=ENV_URL)
    args = parser.parse_args()

    ok = validate(args.env_url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
