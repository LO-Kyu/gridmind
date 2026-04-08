"""
GridMind-RL Inference Script
----------------------------
Runs an LLM agent against all 3 tasks for N episodes each.
Uses the OpenAI Python client pointed at any OpenAI-compatible endpoint.

Required environment variables:
    HF_TOKEN     — Hugging Face / API token (mandatory, no default)
    API_BASE_URL — API endpoint for the LLM (has default)
    MODEL_NAME   — Model identifier (has default)

STDOUT FORMAT (machine-parsed by judge):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Optional

import requests
from openai import OpenAI

# ── Load .env file ─────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Environment Variables ────────────────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
HF_TOKEN     = os.getenv("HF_TOKEN")  # Mandatory — no default
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK     = "gridmind"
EPISODE_STEPS = 96
LAST_STEP     = EPISODE_STEPS - 1
MAX_RETRIES   = 3
DEFAULT_EPISODES = 1
DEFAULT_SEED_BASE = 1000

# Reward range per step in this environment: (0.10, 0.90)
# Worst action -> 0.10, best action -> 0.90
REWARD_MIN = 0.10
REWARD_MAX = 0.90

# Score clamp buffer (never output exactly 0.0 or 1.0)
SCORE_EPSILON = 0.01

# ── System Prompt ────────────────────────────────────────────────────────────
SYSPROMPT = """You are GridMind, an expert industrial energy management controller.
You control a building's HVAC, thermal storage, batch job scheduling, and load shedding.
Your goal is to minimize electricity costs while maintaining comfort and meeting grid demand-response signals.
Always respond with a single valid JSON object matching the action schema. No explanation needed."""

TASK_DESCRIPTIONS = {
    1: "Task 1 (Easy - Cost Minimization): Minimize total energy cost over 24 hours. No temperature or batch constraints. Use cheap off-peak periods and thermal storage.",
    2: "Task 2 (Medium - Temperature Management): Minimize cost AND keep indoor temperature within 19-23°C at all times. Balance comfort vs cost.",
    3: "Task 3 (Hard - Full Demand Response): Minimize cost, maintain temperature, respond to grid stress (shed when grid_stress_signal > 0.7), schedule batch jobs, minimize carbon.",
}

ACTION_SCHEMA = """{
  "hvac_power_level": <float 0.0-1.0>,
  "thermal_charge_rate": <float -1.0 to 1.0>,
  "batch_job_slot": <int 0-4>,
  "load_shed_fraction": <float 0.0-0.5>,
  "building_id": 0
}"""


# ── Logging Helpers (judge-parsed format) ────────────────────────────────────
def log_start(task: str, env_name: str, model: str) -> None:
    """[START] line — emitted once at episode begin."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    """[STEP] line — emitted after each env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """[END] line — always emitted (even on exception)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Utility Functions ─────────────────────────────────────────────────────────
def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse first balanced {...} JSON object from text."""
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


def clamp_open_score(score: float) -> float:
    """Clamp score to strictly between 0 and 1 (never 0.0 or 1.0)."""
    if score <= 0.0:
        return SCORE_EPSILON
    if score >= 1.0:
        return 1.0 - SCORE_EPSILON
    return score


def normalize_reward(raw_reward: float, raw_min: float, raw_max: float) -> float:
    """Normalize raw reward to (REWARD_MIN, REWARD_MAX) range."""
    if raw_max == raw_min:
        return (REWARD_MIN + REWARD_MAX) / 2
    normalized = (raw_reward - raw_min) / (raw_max - raw_min)
    normalized = normalized * (REWARD_MAX - REWARD_MIN) + REWARD_MIN
    return clamp_open_score(normalized)


def compute_score(rewards: list[float]) -> float:
    """Return mean reward clamped strictly to (0.01, 0.99)."""
    if not rewards:
        return SCORE_EPSILON
    mean_reward = sum(rewards) / len(rewards)
    return clamp_open_score(round(mean_reward, 4))


# ── LLM Client ───────────────────────────────────────────────────────────────
def get_llm_client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── LLM Agent ────────────────────────────────────────────────────────────────
class LLMAgent:
    def __init__(self):
        self.client = get_llm_client()
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
- Grid stress signal: {obs.get('grid_stress_signal', 0):.3f} (>0.7 = critical, MUST shed 0.2-0.5 load!)
- Carbon intensity: {obs.get('carbon_intensity', 300):.0f} gCO2/kWh
- Hour of day: {obs.get('hour_of_day', 12)} (0=midnight, peak prices 8-12 and 17-21)
- Pending batch job deadlines: {obs.get('batch_queue', [])}
- Cumulative cost so far: ${obs.get('cumulative_cost', 0):.4f}
- Episode step: {obs.get('step', 0)}/{LAST_STEP}

IMPORTANT RULES:
- thermal_charge_rate: NEGATIVE = DISCHARGE storage, POSITIVE = CHARGE
- load_shed_fraction: MUST be 0.2-0.5 when grid_stress_signal > 0.7, otherwise 0.0
- shed load during grid stress to earn rewards

Strategy hints:
- Charge thermal storage (positive) when price < $0.08/kWh
- Discharge thermal storage (negative) when price > $0.15/kWh
- MUST shed load (0.2-0.5) when grid_stress_signal > 0.7
- Set HVAC low during peak prices (0.3-0.4) and use storage for temperature control
- Schedule batch jobs early if deadline is close (slot 0 or 1)

Respond with ONLY a JSON action:
{ACTION_SCHEMA}"""

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
                print(f"  [LLM attempt {attempt+1}/{MAX_RETRIES}] error: {err_str}", file=sys.stderr)
                if "402" in err_str or "depleted" in err_str:
                    print("  [WARN] API credits depleted! Switching to heuristic agent.", file=sys.stderr)
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
        """Rule-based fallback policy."""
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


# ── Environment Client ────────────────────────────────────────────────────────
class GridMindEnvClient:
    def __init__(self, base_url: str = ENV_URL, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: int = 1, seed: int = 42, num_buildings: int = 1) -> Optional[dict]:
        try:
            payload = {"task_id": task_id, "seed": seed, "num_buildings": num_buildings}
            r = requests.post(f"{self.base}/reset", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
            return None

    def step(self, action: dict) -> Optional[dict]:
        try:
            r = requests.post(f"{self.base}/step", json=action, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to step environment: {e}", file=sys.stderr)
            return None

    def grade(self) -> dict:
        try:
            r = requests.get(f"{self.base}/grade", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to grade: {e}", file=sys.stderr)
            return {"score": SCORE_EPSILON, "sub_scores": {}, "exploit_detected": False}

    def state(self) -> Optional[dict]:
        try:
            r = requests.get(f"{self.base}/state", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to get state: {e}", file=sys.stderr)
            return None

    def close(self) -> None:
        return None


# ── Episode Runner ────────────────────────────────────────────────────────────
def run_episode(
    env_client: GridMindEnvClient,
    agent: LLMAgent,
    task_id: int,
    seed: int,
    *,
    fast_mode: bool,
    llm_every: int,
    max_steps: Optional[int],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single episode and emit hackathon-compliant stdout format."""
    task_name = f"gridmind-task-{task_id}"
    log_start(task=task_name, env_name=BENCHMARK, model=MODEL_NAME)

    total_reward = 0.0
    total_steps = 0
    start_time = time.time()
    step_resp: dict[str, Any] = {"done": False}
    step_limit = EPISODE_STEPS if max_steps is None else min(max_steps, EPISODE_STEPS)

    llm_reuse_remaining = 0
    cached_action = agent._default_action()

    raw_rewards: list[float] = []
    reward_min = float('inf')
    reward_max = float('-inf')
    success = False
    obs: dict[str, Any] = {}

    try:
        reset_resp = env_client.reset(task_id=task_id, seed=seed)
        if reset_resp is None:
            raise RuntimeError("reset failed")
        obs_list = reset_resp.get("observations", [{}])
        obs = obs_list[0] if obs_list else {}

        while not step_resp.get("done", False):
            if total_steps >= step_limit:
                break

            if fast_mode:
                action = agent._heuristic_action(obs)
            else:
                if llm_reuse_remaining <= 0:
                    cached_action = agent.choose_action(obs, task_id)
                    llm_reuse_remaining = max(1, llm_every)
                action = cached_action

            step_resp = env_client.step(action)
            if step_resp is None or not isinstance(step_resp, dict) or "observation" not in step_resp:
                log_step(
                    step=total_steps + 1,
                    action="null",
                    reward=0.0,
                    done=True,
                    error="invalid step response from environment",
                )
                break

            if not fast_mode:
                llm_reuse_remaining -= 1

            obs = step_resp["observation"]
            raw_reward = float(step_resp["reward"])
            total_reward += raw_reward
            raw_rewards.append(raw_reward)

            if raw_reward < reward_min:
                reward_min = raw_reward
            if raw_reward > reward_max:
                reward_max = raw_reward

            total_steps += 1
            done = bool(step_resp.get("done", False))

            normalized_reward = normalize_reward(raw_reward, reward_min, reward_max)

            action_json = json.dumps(action, separators=(',', ':'))
            last_action_error = step_resp.get("last_action_error")
            log_step(
                step=total_steps,
                action=action_json,
                reward=normalized_reward,
                done=done,
                error=last_action_error,
            )

            if verbose and total_steps % 16 == 0:
                print(
                    f"    step={total_steps:02d} price=${obs['current_price']:.3f} "
                    f"temp={obs['indoor_temperature']:.1f}°C "
                    f"stress={obs['grid_stress_signal']:.2f} "
                    f"cost=${obs['cumulative_cost']:.2f}",
                    flush=True,
                    file=sys.stderr,
                )

        success = bool(step_resp.get("done", False))

    except Exception as e:
        err = str(e) or "unknown error"
        err = err.replace("\n", " ").replace("\r", " ")
        if total_steps < step_limit:
            log_step(
                step=total_steps + 1,
                action="null",
                reward=0.0,
                done=True,
                error=err,
            )

    finally:
        env_client.close()

    elapsed = time.time() - start_time
    normalized_rewards = [normalize_reward(r, reward_min, reward_max) for r in raw_rewards]
    episode_score = compute_score(normalized_rewards)

    log_end(
        success=success,
        steps=total_steps,
        score=episode_score,
        rewards=normalized_rewards,
    )

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "score": episode_score,
        "sub_scores": {},
        "exploit_detected": False,
    }


# ── Environment Server Starter ────────────────────────────────────────────────
def start_environment_server(port: int = 7860) -> Optional[subprocess.Popen]:
    """Start the GridMind-RL environment server as a background process."""
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=2)
        if r.status_code == 200:
            print(f"[INFO] Environment server already running on port {port}", file=sys.stderr)
            return None
    except Exception:
        pass

    print(f"[INFO] Starting environment server on port {port}...", file=sys.stderr)

    try:
        env = os.environ.copy()
        env["PORT"] = str(port)

        binary_paths = [
            "/usr/local/bin/gridmind-server",
            "./gridmind-server",
            "./gridmind-server.exe",
        ]

        for binary_path in binary_paths:
            if os.path.exists(binary_path):
                try:
                    proc = subprocess.Popen(
                        [binary_path],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(2)
                    if proc.poll() is None:
                        return proc
                except Exception as e:
                    print(f"[DEBUG] Failed with {binary_path}: {e}", file=sys.stderr)

        try:
            subprocess.run(
                ["go", "build", "-o", "gridmind-server", "main.go"],
                capture_output=True,
                timeout=60,
                cwd=".",
            )
            proc = subprocess.Popen(["./gridmind-server"], env=env)
            time.sleep(2)
            if proc.poll() is None:
                return proc
        except Exception:
            pass

        proc = subprocess.Popen(
            [sys.executable, "-m", "server.app"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        if proc.poll() is None:
            return proc

    except Exception as e:
        print(f"[WARNING] Could not start environment server: {e}", file=sys.stderr)

    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="GridMind-RL inference script")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--env-url", type=str, default=ENV_URL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="baseline_scores.json")
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Heuristic policy only (no LLM calls).",
    )
    parser.add_argument(
        "--llm-every",
        type=int,
        default=8,
        metavar="N",
        help="Reuse the same LLM action for N steps (default: 8).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N steps.",
    )
    args = parser.parse_args()

    server_proc = start_environment_server(port=7860)

    try:
        env_client = GridMindEnvClient(base_url=args.env_url)

        for attempt in range(30):
            if env_client.health():
                break
            time.sleep(2)
            if attempt == 29:
                print("Environment server not reachable.", file=sys.stderr)
                sys.exit(1)

        agent = LLMAgent()
        all_results: list[dict[str, Any]] = []

        for task_id in [1, 2, 3]:
            task_scores: list[float] = []
            for ep in range(args.episodes):
                seed = DEFAULT_SEED_BASE + task_id * 100 + ep
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

        task_avgs: dict[int, float] = {}
        for task_id in [1, 2, 3]:
            scores = [float(r["score"]) for r in all_results if r["task_id"] == task_id]
            avg = clamp_open_score(sum(scores) / len(scores)) if scores else SCORE_EPSILON
            task_avgs[task_id] = avg

        overall = clamp_open_score(sum(task_avgs.values()) / len(task_avgs))

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

    finally:
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
