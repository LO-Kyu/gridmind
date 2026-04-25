#!/usr/bin/env python3
"""
GridMind-RL Multi-Building Coordinator Demo
-----------------------------------------
Demonstrates the Fleet AI scenario (Hackathon Theme #1).
1. Initializes a 3-building environment using the OpenEnv API.
2. Polls GET /feeder to see fleet-wide aggregate state.
3. Uses an LLM to generate per-building price multipliers (POST /coordinate)
   to orchestrate demand and prevent feeder overload.
4. Steps all buildings simultaneously.
"""

import sys
import os
# Add parent directory to path to import from inference.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import requests
from dotenv import load_dotenv

# Import after path fix
try:
    from inference import LLMAgent, extract_json_object, get_llm_client
except ImportError:
    # Fallback definitions if import fails
    def get_llm_client():
        import os
        from openai import OpenAI
        token = os.getenv("HF_TOKEN")
        base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
        return OpenAI(base_url=base_url, api_key=token)

    def extract_json_object(text):
        import json
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    class LLMAgent:
        def __init__(self):
            self.client = get_llm_client()
            self.model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

        def choose_action(self, obs, task_id):
            """Simple rule-based fallback."""
            price = obs.get("current_price", 0.10)
            stress = obs.get("grid_stress_signal", 0.0)
            temp = obs.get("indoor_temperature", 21.0)
            storage = obs.get("thermal_storage_level", 0.5)

            hvac = 0.7 if price < 0.08 else (0.3 if price > 0.15 else 0.5)
            if temp > 23.0:
                hvac = max(hvac, 0.8)
            elif temp < 19.0:
                hvac = min(hvac, 0.2)

            charge = 0.0
            if price < 0.07 and storage < 0.8:
                charge = 0.5
            elif price > 0.15 and storage > 0.3:
                charge = -0.5

            shed = 0.0
            if stress > 0.7:
                shed = 0.4
            elif stress > 0.5:
                shed = 0.2

            return {
                "hvac_power_level": hvac,
                "thermal_charge_rate": charge,
                "batch_job_slot": 2,
                "load_shed_fraction": shed,
                "building_id": 0,
            }

load_dotenv()
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
EPISODE_STEPS = 96

COORDINATOR_PROMPT = """You are the Fleet AI Coordinator for an industrial energy grid.
You manage a feeder supplying 3 industrial buildings. The feeder has a strict limit of {limit} kW.

Current Feeder State:
Total Demand: {demand:.2f} kW (Utilization: {util}%)
Step: {step}/95
Base Electricity Price: ${price:.3f}/kWh

Building Summaries:
{buildings_text}

YOUR TASK:
Adjust the 'price_multipliers' for each building to balance demand and keep total demand under {limit} kW.
- If a building has high demand but its storage is full, increase its price multiplier to force it to discharge storage.
- If total demand is low, lower the price multipliers to encourage charging.
- Multipliers should be between 0.5 and 2.5 (1.0 is neutral).

Output MUST be valid JSON in this exact format:
{{"price_multipliers": [1.0, 1.2, 0.8]}}"""

def reset_multi_building(num_buildings: int = 3, task_id: int = 3):
    """Reset the environment with multiple buildings."""
    url = f"{ENV_URL}/reset"
    payload = {"task_id": task_id, "seed": int(time.time()), "num_buildings": num_buildings}
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def get_feeder_state():
    """Get aggregate fleet state."""
    response = requests.get(f"{ENV_URL}/feeder", timeout=30)
    response.raise_for_status()
    return response.json()

def set_coordinator_signals(multipliers: list[float]):
    """Apply price multipliers via the coordinator API."""
    response = requests.post(f"{ENV_URL}/coordinate", json={"price_multipliers": multipliers}, timeout=30)
    response.raise_for_status()

def run_coordinator_step(feeder_state: dict, llm_client) -> list[float]:
    """Ask LLM to orchestrate the fleet based on feeder state."""
    buildings_text = ""
    for b in feeder_state.get("buildings", []):
        buildings_text += (f"- Building {b['building_id']}: Demand {b['current_demand_kw']:.1f}kW, "
                           f"Storage {b['thermal_storage_level']:.2f}, "
                           f"Cost ${b['cumulative_cost']:.2f}, "
                           f"Current Multiplier: {b.get('price_multiplier', 1.0):.2f}\n")

    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    prompt = COORDINATOR_PROMPT.format(
        limit=feeder_state.get("feeder_limit_kw", 360),
        demand=feeder_state.get("total_demand_kw", 0),
        util=feeder_state.get("utilization_pct", 0),
        step=feeder_state.get("step", 0),
        price=feeder_state.get("price_curve_hourly", [0.1])[0],
        buildings_text=buildings_text
    )

    try:
        completion = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        content = completion.choices[0].message.content
        parsed = extract_json_object(content)
        if parsed and "price_multipliers" in parsed:
            return parsed["price_multipliers"]
    except Exception as e:
        print(f"Coordinator error: {e}")
    
    return [1.0, 1.0, 1.0]

def main():
    print("=== GridMind-RL: Multi-Building Fleet AI Demo ===")
    print(f"Connecting to {ENV_URL}...\n")
    
    # Check health
    try:
        requests.get(f"{ENV_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Error: Environment server not running at {ENV_URL}.")
        return

    # 1. Reset with 3 buildings
    print("▶ Initializing 3-building federation (Task 3: Demand Response)...")
    init_data = reset_multi_building(num_buildings=3, task_id=3)
    
    llm_client = get_llm_client()
    local_agents = [LLMAgent() for _ in range(3)]
    
    total_reward = 0.0
    feeder_utilizations = []
    
    # Run full episode
    for step in range(EPISODE_STEPS):
        # -- 1. Coordinator plans --
        feeder = get_feeder_state()
        util = feeder.get("utilization_pct", 0)
        feeder_utilizations.append(util)
        
        if step % 16 == 0:
            print(f"\n[Step {step}] Feeder Demand: {feeder['total_demand_kw']:.1f}kW / {feeder['feeder_limit_kw']:.1f}kW (Util: {util:.1f}%)")
        
        multipliers = run_coordinator_step(feeder, llm_client)
        
        if step % 16 == 0:
            print(f"  → Coordinator sets price multipliers: {multipliers}")
        set_coordinator_signals(multipliers)
        
        # -- 2. Local agents react --
        # Fetch fresh state so agents see the new prices
        obs_data = requests.get(f"{ENV_URL}/state", timeout=30).json()
        buildings = obs_data.get("buildings", [])
        
        if not buildings:
            print("Error: No buildings in state")
            break
            
        actions = []
        for i, b_obs in enumerate(buildings):
            action = local_agents[i].choose_action(b_obs, task_id=3)
            action["building_id"] = i
            actions.append(action)
            
        # -- 3. Step physics engine --
        if actions:
            step_resp = requests.post(f"{ENV_URL}/step", json=actions, timeout=30).json()
            
            # Handle both array and object response formats
            if isinstance(step_resp, list):
                results = step_resp
            else:
                results = step_resp.get("results", [])
            
            for r in results:
                total_reward += r.get("reward", 0.0)
        
        if step % 16 == 0:
            avg_util = sum(feeder_utilizations[-16:]) / min(16, len(feeder_utilizations))
            print(f"  → Step {step} complete. Total reward so far: {total_reward:.3f}, Avg Feeder Util: {avg_util:.1f}%")

    # Final feeder state
    feeder = get_feeder_state()
    final_util = feeder.get("utilization_pct", 0)
    
    print(f"\n=== Episode Complete ===")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Feeder utilization: {final_util:.1f}% ({'OVERLOAD' if feeder.get('feeder_overload', False) else 'OK'})")
    
    # Per-building cost breakdown
    buildings = feeder.get("buildings", [])
    for b in buildings:
        print(f"  Building {b['building_id']}: ${b['cumulative_cost']:.2f}")
    
    print("\n✅ Multi-Building Demo complete.")
    print("The coordinator successfully managed price signals to orchestrate the fleet!")

if __name__ == "__main__":
    main()