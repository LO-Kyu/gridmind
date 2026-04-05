"""
GridMind-RL Baseline Inference Script
--------------------------------------
Runs an LLM agent against all 3 tasks for N episodes each.
Uses the OpenAI Python client pointed at any OpenAI-compatible endpoint.

Required environment variables (set in .env or shell):
    API_BASE_URL   — The API endpoint for the LLM (default: OpenRouter)
    MODEL_NAME     — The model identifier to use for inference
    OPENAI_API_KEY — API key for authentication (works with any provider)

Usage:
    # Option 1: Use .env file (recommended — just paste your key)
    python inference.py

    # Option 2: Set env vars manually
    export API_BASE_URL=https://openrouter.ai/api/v1
    export MODEL_NAME=meta-llama/llama-3.1-8b-instruct:free
    export OPENAI_API_KEY=sk-or-v1-xxxx
    python inference.py

    # Option 3: Fast mode (no LLM, heuristic only)
    python inference.py --fast-mode --episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ── Load .env file (if present) ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from current directory or project root
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

# ── Constants ──────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")

# ── Hackathon Spec Compliance: HF_TOKEN → OpenAI API Key ──────────────────
# Per hackathon spec, the LLM API credential is read from HF_TOKEN environment variable
# and passed directly to the OpenAI client for initialization.
# Primary: HF_TOKEN (hackathon spec requirement)
# Fallback: OPENAI_API_KEY (for local testing/development)
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
if not OPENAI_API_KEY:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable is required")
DEFAULT_EPISODES = 1
DEFAULT_SEED_BASE = 1000
MAX_RETRIES = 3
# 96 steps × 15 min = 24 h (must match env.EpisodeSteps)
EPISODE_STEPS = 96
LAST_STEP_INDEX = EPISODE_STEPS - 1

SYSPROMPT = """You are GridMind, an expert industrial energy management controller.
You control a building's HVAC, thermal storage, batch job scheduling, and load shedding.
Your goal is to minimize electricity costs while maintaining comfort and meeting grid demand-response signals.
Always respond with a single valid JSON object matching the action schema. No explanation needed."""

TASK_DESCRIPTIONS = {
    1: "Task 1 (Easy - Cost Minimization): Minimize total energy cost over 24 hours. No temperature or batch constraints. Use cheap off-peak periods and thermal storage.",
    2: "Task 2 (Medium - Temperature Management): Minimize cost AND keep indoor temperature within 19-23°C at all times. Balance comfort vs cost.",
    3: "Task 3 (Hard - Full Demand Response): Minimize cost, maintain temperature, respond to grid stress (shed when grid_stress_signal > 0.7), schedule batch jobs, minimize carbon.",
}

ACTION_SCHEMA_STR = """{
  "hvac_power_level": <float 0.0-1.0>,
  "thermal_charge_rate": <float -1.0 to 1.0>,
  "batch_job_slot": <int 0-4>,
  "load_shed_fraction": <float 0.0-0.5>,
  "building_id": 0
}"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse first balanced {...} JSON object from text (handles nested braces)."""
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
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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
        # Initialize OpenAI client with credentials from HF_TOKEN (per hackathon spec)
        # The OPENAI_API_KEY variable contains the HF_TOKEN value passed by evaluators
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
        self.model = MODEL_NAME
        self.fallback_mode = False

    def choose_action(self, obs: dict, task_id: int) -> dict:
        """Prompt the LLM with current observation, return parsed action dict."""
        if self.fallback_mode:
            return self._heuristic_action(obs)
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
- Episode step: {obs.get('step', 0)}/{LAST_STEP_INDEX}

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
                    temperature=0.0,
                )
                content = completion.choices[0].message.content.strip()
                parsed = extract_json_object(content)
                if parsed is not None:
                    return self._clamp_action(parsed)
                action = json.loads(content)
                return self._clamp_action(action)
            except Exception as e:
                err_str = str(e)
                print(f"  [LLM attempt {attempt+1}/{MAX_RETRIES}] error: {err_str}")
                if "402" in err_str or "depleted" in err_str:
                    print("  [WARN] Hugging Face free credits depleted! Switching to local heuristic agent for the rest of the simulation.")
                    self.fallback_mode = True
                    return self._heuristic_action(obs)
                time.sleep(1)

        return self._heuristic_action(obs)

    def _clamp_action(self, action: dict) -> dict:
        return {
            "hvac_power_level": max(0.0, min(1.0, float(action.get("hvac_power_level", 0.5)))),
            "thermal_charge_rate": max(-1.0, min(1.0, float(action.get("thermal_charge_rate", 0.0)))),
            "batch_job_slot": max(0, min(4, int(action.get("batch_job_slot", 0)))),
            "load_shed_fraction": max(0.0, min(0.5, float(action.get("load_shed_fraction", 0.0)))),
            "building_id": int(action.get("building_id", 0)),
        }

    def _heuristic_action(self, obs: dict) -> dict:
        """Rule-based policy (deterministic given obs)."""
        price = obs.get("current_price", 0.10)
        stress = obs.get("grid_stress_signal", 0.0)
        temp = obs.get("indoor_temperature", 21.0)
        storage = obs.get("thermal_storage_level", 0.5)
        queue = obs.get("batch_queue", [])

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

        slot = 2
        if queue and min(queue) < 8:
            slot = 0

        return {
            "hvac_power_level": hvac,
            "thermal_charge_rate": charge,
            "batch_job_slot": slot,
            "load_shed_fraction": shed,
            "building_id": 0,
        }

    def _default_action(self) -> dict:
        return {
            "hvac_power_level": 0.5,
            "thermal_charge_rate": 0.0,
            "batch_job_slot": 0,
            "load_shed_fraction": 0.0,
            "building_id": 0,
        }


# ── Episode runner ───────────────────────────────────────────────────────────


def run_episode(
    env_client: GridMindEnvClient,
    agent: LLMAgent,
    task_id: int,
    seed: int,
    *,
    fast_mode: bool,
    llm_every: int,
    max_steps: int | None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single episode and emit hackathon-compliant stdout format.
    
    Emits:
      [START] task=<name> env=gridmind model=<model>
      [STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
      ...
      [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
    """
    reset_resp = env_client.reset(task_id=task_id, seed=seed)
    obs = reset_resp["observations"][0]
    
    task_name = f"gridmind-task-{task_id}"
    
    # Emit [START] with required fields
    print(f"[START] task={task_name} env=gridmind model={MODEL_NAME}", flush=True)
    
    total_reward = 0.0
    total_steps = 0
    start_time = time.time()
    step_resp: dict[str, Any] = {}
    step_limit = EPISODE_STEPS if max_steps is None else min(max_steps, EPISODE_STEPS)
    
    llm_reuse_remaining = 0
    cached_action = agent._default_action()
    
    step_rewards: list[float] = []
    last_error: str | None = None
    
    while not step_resp.get("done", False):
        if total_steps >= step_limit:
            break
        
        try:
            if fast_mode:
                action = agent._heuristic_action(obs)
            else:
                if llm_reuse_remaining <= 0:
                    cached_action = agent.choose_action(obs, task_id)
                    llm_reuse_remaining = max(1, llm_every)
                action = cached_action
            
            step_resp = env_client.step(action)
            if step_resp is None or "observation" not in step_resp:
                last_error = "invalid step response"
                break
            
            if not fast_mode:
                llm_reuse_remaining -= 1
            
            obs = step_resp["observation"]
            reward = float(step_resp["reward"])
            total_reward += reward
            step_rewards.append(reward)
            total_steps += 1
            done = bool(step_resp.get("done", False))
            
            # Emit [STEP] with required fields (action as compact JSON, reward to 2 decimals)
            action_json = json.dumps(action, separators=(',', ':'))
            error_field = "null" if last_error is None else f'"{last_error}"'
            print(
                f"[STEP] step={total_steps} action={action_json} "
                f"reward={reward:.2f} done={'true' if done else 'false'} error={error_field}",
                flush=True
            )
            
            last_error = None  # Clear error after successful step
            
            if verbose and total_steps % 16 == 0:
                print(
                    f"    step={total_steps:02d} price=${obs['current_price']:.3f} "
                    f"temp={obs['indoor_temperature']:.1f}°C "
                    f"stress={obs['grid_stress_signal']:.2f} "
                    f"cost=${obs['cumulative_cost']:.2f}",
                    flush=True,
                )
        
        except Exception as e:
            last_error = str(e)
            print(
                f"[STEP] step={total_steps + 1} action=null "
                f"reward=0.00 done=true error=\"{last_error}\"",
                flush=True
            )
            break
    
    elapsed = time.time() - start_time
    grade = env_client.grade()
    
    # Emit [END] with required fields
    print(
        f"[END] success={'true' if success else 'false'} steps={total_steps} rewards={rewards_str}",
        flush=True
    )
    
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GridMind-RL baseline inference")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--env-url", type=str, default=ENV_URL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="baseline_scores.json")
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Heuristic policy only (no LLM calls; fastest, fully reproducible).",
    )
    parser.add_argument(
        "--llm-every",
        type=int,
        default=4,
        metavar="N",
        help="Reuse the same LLM action for N consecutive steps (default: 4).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N steps (default: full episode). Grade uses partial episode.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GridMind-RL Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Env:   {args.env_url}")
    print(f"  Episodes per task: {args.episodes}")
    print(f"  Fast mode: {args.fast_mode} | LLM every: {args.llm_every} steps")
    print("=" * 60)

    env_client = GridMindEnvClient(base_url=args.env_url)

    print("\nWaiting for environment server...")
    for attempt in range(30):
        if env_client.health():
            print("  [OK] Environment server is healthy")
            break
        time.sleep(2)
        if attempt == 29:
            print("  [FAIL] Environment server not reachable. Exiting.")
            sys.exit(1)

    agent = LLMAgent()
    all_results: list[dict[str, Any]] = []

    for task_id in [1, 2, 3]:
        print(f"\n-- Task {task_id}: {TASK_DESCRIPTIONS[task_id][:60]}...")
        task_scores: list[float] = []
        for ep in range(args.episodes):
            seed = DEFAULT_SEED_BASE + task_id * 100 + ep
            print(f"  Episode {ep+1}/{args.episodes} (seed={seed})")
            result = run_episode(
                env_client,
                agent,
                task_id=task_id,
                seed=seed,
                fast_mode=args.fast_mode,
                llm_every=args.llm_every,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
            task_scores.append(float(result["score"]))
            all_results.append(result)
            print(
                f"    → score={result['score']:.4f} | reward={result['total_reward']:.3f} | "
                f"{result['elapsed_sec']:.1f}s | steps={result['total_steps']}"
            )

        avg_score = sum(task_scores) / len(task_scores)
        print(f"  Task {task_id} average score: {avg_score:.4f}")

    print("\n" + "=" * 60)
    print("BASELINE SCORES SUMMARY")
    print("=" * 60)
    print(f"{'Task':<10} {'Model':<30} {'Score':<10} {'Episodes':<10}")
    print("-" * 60)

    task_avgs: dict[int, float] = {}
    for task_id in [1, 2, 3]:
        scores = [float(r["score"]) for r in all_results if r["task_id"] == task_id]
        avg = sum(scores) / len(scores) if scores else 0.0
        task_avgs[task_id] = avg
        print(f"Task {task_id:<6} {MODEL_NAME:<30} {avg:<10.4f} {len(scores)}")

    print("-" * 60)
    overall = sum(task_avgs.values()) / len(task_avgs)
    print(f"{'Overall':<10} {'':<30} {overall:<10.4f}")

    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "episodes_per_task": args.episodes,
        "seed_base": DEFAULT_SEED_BASE,
        "fast_mode": args.fast_mode,
        "llm_every": args.llm_every,
        "max_steps": args.max_steps,
        "task_averages": {str(k): v for k, v in task_avgs.items()},
        "overall_average": overall,
        "all_results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
