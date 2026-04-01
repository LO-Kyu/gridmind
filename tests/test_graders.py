"""
GridMind-RL Grader Validation Tests (pytest)
Run with: pytest tests/test_graders.py -v
"""

import json
import time
import pytest
import requests

ENV_URL = "http://localhost:7860"
BASE = ENV_URL


def wait_for_server(url: str, timeout: int = 15):
    for _ in range(timeout):
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="session", autouse=True)
def server_running():
    if not wait_for_server(ENV_URL):
        pytest.skip("GridMind-RL server not running at " + ENV_URL)


def reset(task_id=1, seed=42):
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id, "seed": seed, "num_buildings": 1})
    r.raise_for_status()
    return r.json()


def step(action: dict) -> dict:
    r = requests.post(f"{BASE}/step", json=action)
    r.raise_for_status()
    return r.json()


def grade() -> dict:
    r = requests.get(f"{BASE}/grade")
    r.raise_for_status()
    return r.json()


def run_full_episode(task_id: int, seed: int, hvac: float = 0.5) -> dict:
    reset(task_id=task_id, seed=seed)
    action = {"hvac_power_level": hvac, "thermal_charge_rate": 0, "batch_job_slot": 0, "load_shed_fraction": 0}
    for _ in range(96):
        resp = step(action)
        if resp.get("done"):
            break
    return grade()


# ── Task 1 ──────────────────────────────────────────────────────────────────

class TestTask1:
    def test_score_in_range(self):
        g = run_full_episode(task_id=1, seed=1)
        assert 0.0 <= g["score"] <= 1.0, f"Score {g['score']} out of [0,1]"

    def test_score_not_always_zero(self):
        g = run_full_episode(task_id=1, seed=2, hvac=0.2)
        assert g["score"] > 0.01, "Low HVAC policy should score > 0"

    def test_score_not_always_one(self):
        g = run_full_episode(task_id=1, seed=3, hvac=1.0)
        assert g["score"] < 0.999, "Always-on policy should not score 1.0"

    def test_deterministic(self):
        g1 = run_full_episode(task_id=1, seed=42)
        g2 = run_full_episode(task_id=1, seed=42)
        assert abs(g1["score"] - g2["score"]) < 1e-6, "Grader not deterministic with same seed"

    def test_sub_scores_present(self):
        g = run_full_episode(task_id=1, seed=5)
        assert "cost" in g["sub_scores"], "Task 1 grade missing 'cost' sub-score"

    def test_exploit_shedding_penalized(self):
        """Always shedding 50% should be detected and penalized."""
        reset(task_id=1, seed=10)
        action = {"hvac_power_level": 0.5, "thermal_charge_rate": 0, "batch_job_slot": 0, "load_shed_fraction": 0.5}
        for _ in range(96):
            step(action)
        g = grade()
        # Score should be reduced OR exploit flagged
        assert g["exploit_detected"] or g["score"] < 0.9


# ── Task 2 ──────────────────────────────────────────────────────────────────

class TestTask2:
    def test_score_in_range(self):
        g = run_full_episode(task_id=2, seed=20)
        assert 0.0 <= g["score"] <= 1.0

    def test_has_temp_sub_score(self):
        g = run_full_episode(task_id=2, seed=21)
        assert "temperature" in g["sub_scores"]

    def test_temp_score_range(self):
        g = run_full_episode(task_id=2, seed=22)
        ts = g["sub_scores"].get("temperature", -1)
        assert 0.0 <= ts <= 1.0, f"Temperature sub-score {ts} out of [0,1]"

    def test_weights_sum_correct(self):
        """Task 2 score = 0.6*cost + 0.4*temp."""
        g = run_full_episode(task_id=2, seed=23)
        expected = g["sub_scores"]["cost"] * 0.6 + g["sub_scores"]["temperature"] * 0.4
        assert abs(g["score"] - expected) < 0.01 or g["exploit_detected"]

    def test_score_varies_with_policy(self):
        g_low  = run_full_episode(task_id=2, seed=24, hvac=0.1)
        g_high = run_full_episode(task_id=2, seed=24, hvac=0.9)
        # Scores should differ (policy matters)
        assert abs(g_low["score"] - g_high["score"]) > 0.001


# ── Task 3 ──────────────────────────────────────────────────────────────────

class TestTask3:
    def test_score_in_range(self):
        g = run_full_episode(task_id=3, seed=30)
        assert 0.0 <= g["score"] <= 1.0

    def test_has_all_sub_scores(self):
        g = run_full_episode(task_id=3, seed=31)
        for key in ["cost", "temperature", "grid_response", "batch_deadline"]:
            assert key in g["sub_scores"], f"Missing sub-score: {key}"

    def test_all_sub_scores_in_range(self):
        g = run_full_episode(task_id=3, seed=32)
        for key, val in g["sub_scores"].items():
            assert 0.0 <= val <= 1.0, f"Sub-score '{key}' = {val} out of [0,1]"

    def test_weights_sum_correct(self):
        g = run_full_episode(task_id=3, seed=33)
        ss = g["sub_scores"]
        expected = ss["cost"]*0.35 + ss["temperature"]*0.25 + ss["grid_response"]*0.25 + ss["batch_deadline"]*0.15
        assert abs(g["score"] - expected) < 0.01 or g["exploit_detected"]

    def test_grid_response_sub_score(self):
        g = run_full_episode(task_id=3, seed=34)
        gs = g["sub_scores"].get("grid_response", -1)
        assert 0.0 <= gs <= 1.0, f"grid_response={gs} out of [0,1]"

    def test_batch_deadline_sub_score(self):
        g = run_full_episode(task_id=3, seed=35)
        bd = g["sub_scores"].get("batch_deadline", -1)
        assert 0.0 <= bd <= 1.0


# ── Multi-building grading ────────────────────────────────────────────────────

class TestMultiBuilding:
    def test_2_building_grade(self):
        requests.post(f"{BASE}/reset", json={"task_id": 1, "seed": 50, "num_buildings": 2}).raise_for_status()
        action = [
            {"hvac_power_level": 0.4, "thermal_charge_rate": 0, "batch_job_slot": 0, "load_shed_fraction": 0, "building_id": 0},
            {"hvac_power_level": 0.6, "thermal_charge_rate": 0, "batch_job_slot": 0, "load_shed_fraction": 0, "building_id": 1},
        ]
        for _ in range(96):
            r = requests.post(f"{BASE}/step", json=action)
            if r.json()[0].get("done"):
                break
        g = grade()
        assert 0.0 <= g["score"] <= 1.0
