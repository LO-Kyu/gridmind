#!/usr/bin/env python3
"""Quick test of coordinator endpoints."""

import requests
import json

ENV_URL = "http://localhost:7860"

print("=" * 60)
print("COORDINATOR ENDPOINT TEST")
print("=" * 60)

# Test coordinator reset
print("\n1. Testing /coordinator/reset...")
try:
    r = requests.post(f"{ENV_URL}/coordinator/reset", json={}, timeout=10)
    print(f"   Status: {r.status_code}")
    resp = r.json()
    obs_list = resp.get("observations", [])
    print(f"   Observations count: {len(obs_list)}")
    if obs_list:
        print(f"   First observation keys: {list(obs_list[0].keys())[:5]}")
        print(f"   First building temp: {obs_list[0].get('indoor_temperature', 'N/A')}°C")
except Exception as e:
    print(f"   ERROR: {e}")

# Test coordinator step
print("\n2. Testing /coordinator/step...")
actions = [
    {"hvac_power_level": 0.5, "thermal_charge_rate": 0.0, "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0},
    {"hvac_power_level": 0.6, "thermal_charge_rate": 0.1, "batch_job_slot": 1, "load_shed_fraction": 0.1, "building_id": 1},
    {"hvac_power_level": 0.4, "thermal_charge_rate": -0.2, "batch_job_slot": 2, "load_shed_fraction": 0.0, "building_id": 2},
]
try:
    r = requests.post(f"{ENV_URL}/coordinator/step", json=actions, timeout=10)
    print(f"   Status: {r.status_code}")
    resp = r.json()
    responses = resp.get("responses", [])
    print(f"   Responses count: {len(responses)}")
    done = resp.get("done", False)
    print(f"   Episode done: {done}")
    
    if responses:
        for i, sr in enumerate(responses):
            reward = sr.get("reward", 0.0)
            obs = sr.get("observation", {})
            temp = obs.get("indoor_temperature", "N/A")
            print(f"   Building {i}: reward={reward:.4f}, temp={temp}°C")
except Exception as e:
    print(f"   ERROR: {e}")

# Test several steps to verify stateful behavior
print("\n3. Testing multi-step coordinator episode...")
try:
    # Reset
    r = requests.post(f"{ENV_URL}/coordinator/reset", json={}, timeout=10)
    resp = r.json()
    obs_list = resp.get("observations", [])
    print(f"   Reset: {len(obs_list)} buildings")
    
    # Take 3 steps
    for step_num in range(3):
        actions = [
            {"hvac_power_level": 0.5, "thermal_charge_rate": 0.0, "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": i}
            for i in range(len(obs_list))
        ]
        r = requests.post(f"{ENV_URL}/coordinator/step", json=actions, timeout=10)
        resp = r.json()
        responses = resp.get("responses", [])
        rewards = [sr.get("reward", 0.0) for sr in responses]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        done = resp.get("done", False)
        print(f"   Step {step_num+1}: avg_reward={avg_reward:.4f}, done={done}")
        
        # Update obs for next iteration
        obs_list = [sr.get("observation", {}) for sr in responses]
        
        if done:
            print(f"   Episode completed at step {step_num+1}")
            break
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("✓ Coordinator endpoint test complete!")
print("=" * 60)
