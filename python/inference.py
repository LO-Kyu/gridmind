"""
GridMind-RL Baseline Inference Script
--------------------------------------
Runs an LLM agent against all 3 tasks for N episodes each.
Uses OpenAI-compatible API via API_BASE_URL / MODEL_NAME / HF_TOKEN environment variables.

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    export HF_TOKEN=hf_xxxx
    python python/inference.py [--episodes 3] [--env-url http://localhost:7860]
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_EPISODES = 3
DEFAULT_SEED_BASE = 1000  # episodes use seed BASE+episode_idx for reproducibility
MAX_RETRIES = 3

SYSPROMPT = """You are GridMind, an expert industrial energy management controller.
You control a building's HVAC, thermal storage, batch job scheduling, and load shedding.
Your goal is to minimize electricity costs while maintaining comfort and meeting grid demand-response signals.
Always respond with a single valid JSON object matching the action schema. No explanation needed."""

TASK_DESCRIPTIONS = {
    1: "Task 1 (Easy - Cost Minimization): Minimize total energy cost over 24 hours. No temperature constraints. Use cheap off-peak periods and thermal storage arbitrage.",
    2: "Task 2 (Medium - Temperature Management): Minimize cost AND keep indoor temperature within 19-23°C at all times. Balance comfort vs cost.",
    3: "Task 3 (Hard - Full Demand Response): Minimize cost, maintain temperature, respond to grid stress events by shedding load when grid_stress_signal > 0.7, AND schedule all batch jobs before their deadlines.",
}

ACTION_SCHEMA_STR = """{
  "hvac_power_level": <float 0.0-1.0>,
  "thermal_charge_rate": <float -1.0 to 1.0>,
  "batch_job_slot": <int 0-4>,
  "load_shed_fraction": <float 0.0-0.5>,
  "building_id": 0
}"""


# ── Environment client ───────────────────────────────────────────────────────

class GridMindEnvClient:
    """Simple HTTP client for the GridMind-RL Go environment server."""

    def __init__(self, base_url: str = ENV_URL, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: int = 1, seed: int = 42, num_buildings: int = 1) -> dict:
        payload = {"task_id": task_id, "seed": seed, "num_buildings": num_buildings}
        r = requests.post(f"{self.base}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = requests.post(f"{self.base}/step", json=action, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        r = requests.get(f"{self.base}/grade", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base}/state", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ── LLM agent ───────────────────────────────────────────────────────────────

class LLMAgent:
    """OpenAI-compatible LLM agent that chooses actions given observations."""

    def __init__(self):
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN if HF_TOKEN else "none",
        )
        self.model = MODEL_NAME

    def choose_action(self, obs: dict, task_id: int) -> dict:
        """Prompt the LLM with current observation, return parsed action dict."""
        task_desc = TASK_DESCRIPTIONS.get(task_id, TASK_DESCRIPTIONS[1])

        prompt = f"""{task_desc}

Current observation:
- Indoor temperature: {obs.get('indoor_temperature', 21):.1f}°C (target: 21°C, bounds: 19-23°C)
- Thermal storage level: {obs.get('thermal_storage_level', 0.5):.2f} (0=empty, 1=full)
- Process demand: {obs.get('process_demand', 15):.1f} kW
- Current electricity price: ${obs.get('current_price', 0.10):.4f}/kWh
- Grid stress signal: {obs.get('grid_stress_signal', 0):.3f} (>0.7 = critical, shed load!)
- Carbon intensity: {obs.get('carbon_intensity', 300):.0f} gCO2/kWh
- Hour of day: {obs.get('hour_of_day', 12)} (0=midnight, peak prices 8-12 and 17-21)
- Pending batch job deadlines: {obs.get('batch_queue', [])}
- Cumulative cost so far: ${obs.get('cumulative_cost', 0):.4f}
- Episode step: {obs.get('step', 0)}/95

Strategy hints:
- Charge thermal storage when price < $0.08/kWh, discharge when price > $0.15/kWh
- Set HVAC low during peak prices (0.3-0.4) and use storage for temperature control
- Shed 30-50% load if grid_stress_signal > 0.7
- Schedule batch jobs early if deadline is close (slot 0 or 1)

Respond with ONLY a JSON action:
{ACTION_SCHEMA_STR}"""

        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSPROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=128,
                    temperature=0.1,
                )
                content = completion.choices[0].message.content.strip()
                return self._parse_action(content)
            except Exception as e:
                print(f"  [LLM attempt {attempt+1}/{MAX_RETRIES}] error: {e}")
                time.sleep(1)

        # Fallback: rule-based heuristic
        return self._heuristic_action(obs)

    def _parse_action(self, content: str) -> dict:
        """Extract and validate JSON action from LLM response."""
        # Try direct JSON parse
        try:
            action = json.loads(content)
            return self._clamp_action(action)
        except json.JSONDecodeError:
            pass
        # Try to extract JSON block from text
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group())
                return self._clamp_action(action)
            except json.JSONDecodeError:
                pass
        # Fallback
        print(f"  [WARN] could not parse LLM response: {content[:100]}")
        return self._default_action()

    def _clamp_action(self, action: dict) -> dict:
        return {
            "hvac_power_level": max(0.0, min(1.0, float(action.get("hvac_power_level", 0.5)))),
            "thermal_charge_rate": max(-1.0, min(1.0, float(action.get("thermal_charge_rate", 0.0)))),
            "batch_job_slot": max(0, min(4, int(action.get("batch_job_slot", 0)))),
            "load_shed_fraction": max(0.0, min(0.5, float(action.get("load_shed_fraction", 0.0)))),
            "building_id": int(action.get("building_id", 0)),
        }

    def _heuristic_action(self, obs: dict) -> dict:
        """Simple rule-based heuristic when LLM is unavailable."""
        price = obs.get("current_price", 0.10)
        stress = obs.get("grid_stress_signal", 0.0)
        temp = obs.get("indoor_temperature", 21.0)
        storage = obs.get("thermal_storage_level", 0.5)
        queue = obs.get("batch_queue", [])

        # HVAC: reduce during peak
        hvac = 0.7 if price < 0.08 else (0.3 if price > 0.15 else 0.5)
        # Adjust for temperature
        if temp > 23.0:
            hvac = max(hvac, 0.8)
        elif temp < 19.0:
            hvac = min(hvac, 0.2)

        # Storage arbitrage
        charge = 0.0
        if price < 0.07 and storage < 0.8:
            charge = 0.5
        elif price > 0.15 and storage > 0.3:
            charge = -0.5

        # Load shedding
        shed = 0.0
        if stress > 0.7:
            shed = 0.4
        elif stress > 0.5:
            shed = 0.2

        # Batch jobs: schedule soon if deadline approaching
        slot = 2
        if queue and min(queue) < 10:
            slot = 0

        return {
            "hvac_power_level": hvac,
            "thermal_charge_rate": charge,
            "batch_job_slot": slot,
            "load_shed_fraction": shed,
            "building_id": 0,
        }

    def _default_action(self) -> dict:
        return {"hvac_power_level": 0.5, "thermal_charge_rate": 0.0,
                "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0}


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(env_client: GridMindEnvClient, agent: LLMAgent,
                task_id: int, seed: int, verbose: bool = False) -> dict[str, Any]:
    """Run a single episode and return grade + metadata."""
    reset_resp = env_client.reset(task_id=task_id, seed=seed)
    obs = reset_resp["observations"][0]

    total_reward = 0.0
    total_steps = 0
    start_time = time.time()

    step_resp = {}
    _step = 0
    while not step_resp.get("done", False):
        action = agent.choose_action(obs, task_id)
        step_resp = env_client.step(action)

        if step_resp is None or "observation" not in step_resp:
            print(f"  [WARN] step {_step}: server returned invalid response, skipping step")
            _step += 1
            break

        obs = step_resp["observation"]
        total_reward += step_resp["reward"]
        total_steps += 1

        if verbose and _step % 16 == 0:
            print(f"    step={_step:02d} price=${obs['current_price']:.3f} "
                  f"temp={obs['indoor_temperature']:.1f}°C "
                  f"stress={obs['grid_stress_signal']:.2f} "
                  f"cost=${obs['cumulative_cost']:.2f} "
                  f"reward={step_resp['reward']:.3f}")
        _step += 1

    elapsed = time.time() - start_time
    grade = env_client.grade()

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "score": grade.get("score", 0.0),
        "sub_scores": grade.get("sub_scores", {}),
        "exploit_detected": grade.get("exploit_detected", False),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridMind-RL baseline inference")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--env-url", type=str, default=ENV_URL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="baseline_scores.json")
    args = parser.parse_args()

    print("=" * 60)
    print("GridMind-RL Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Env:   {args.env_url}")
    print(f"  Episodes per task: {args.episodes}")
    print("=" * 60)

    env_client = GridMindEnvClient(base_url=args.env_url)

    # Wait for env server to be healthy
    print("\nWaiting for environment server...")
    for attempt in range(30):
        if env_client.health():
            print("  ✓ Environment server is healthy")
            break
        time.sleep(2)
        if attempt == 29:
            print("  ✗ Environment server not reachable. Exiting.")
            sys.exit(1)

    agent = LLMAgent()
    all_results = []

    for task_id in [1, 2, 3]:
        print(f"\n── Task {task_id}: {TASK_DESCRIPTIONS[task_id][:60]}...")
        task_scores = []
        for ep in range(args.episodes):
            seed = DEFAULT_SEED_BASE + task_id * 100 + ep
            print(f"  Episode {ep+1}/{args.episodes} (seed={seed})")
            result = run_episode(env_client, agent, task_id=task_id, seed=seed, verbose=args.verbose)
            task_scores.append(result["score"])
            all_results.append(result)
            print(f"    → score={result['score']:.4f} | reward={result['total_reward']:.3f} | {result['elapsed_sec']:.1f}s")

        avg_score = sum(task_scores) / len(task_scores)
        print(f"  Task {task_id} average score: {avg_score:.4f}")

    # Score summary table
    print("\n" + "=" * 60)
    print("BASELINE SCORES SUMMARY")
    print("=" * 60)
    print(f"{'Task':<10} {'Model':<30} {'Score':<10} {'Episodes':<10}")
    print("-" * 60)

    task_avgs = {}
    for task_id in [1, 2, 3]:
        scores = [r["score"] for r in all_results if r["task_id"] == task_id]
        avg = sum(scores) / len(scores) if scores else 0.0
        task_avgs[task_id] = avg
        print(f"Task {task_id:<6} {MODEL_NAME:<30} {avg:<10.4f} {len(scores)}")

    print("-" * 60)
    overall = sum(task_avgs.values()) / len(task_avgs)
    print(f"{'Overall':<10} {'':<30} {overall:<10.4f}")

    # Save results
    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "episodes_per_task": args.episodes,
        "seed_base": DEFAULT_SEED_BASE,
        "task_averages": {str(k): v for k, v in task_avgs.items()},
        "overall_average": overall,
        "all_results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
