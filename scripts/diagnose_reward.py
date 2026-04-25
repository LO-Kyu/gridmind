#!/usr/bin/env python3
"""
Diagnostic cell to verify reward function is working before training.
Run this BEFORE training to catch zero-loss issues early.
"""

import json
import numpy as np
import random
import re
import requests

ENV_URL = "https://prajwal782007-gridmind.hf.space"


def gridmind_reward_fn(completions, env_url=ENV_URL, **kwargs):
    """
    Fixed reward function for GRPO with environment reset per completion.
    Returns varied rewards to enable GRPO learning.
    """
    rewards = []
    batch_rewards = []
    call_count = 0
    
    for i, completion in enumerate(completions):
        call_count += 1
        
        text = completion[0]["content"] if isinstance(completion, list) else completion
        
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not match:
                rewards.append(-1.0)
                batch_rewards.append(-1.0)
                continue
            
            action = json.loads(match.group())
            
            step_action = {
                "hvac_power_level": float(max(0, min(1, action.get("hvac_power_level", 0.5)))),
                "thermal_charge_rate": float(max(-1, min(1, action.get("thermal_charge_rate", 0.0)))),
                "batch_job_slot": int(max(0, min(4, action.get("batch_job_slot", 0)))),
                "load_shed_fraction": float(max(0, min(0.5, action.get("load_shed_fraction", 0.0)))),
                "building_id": 0
            }
            
            # VARY SEED each call to ensure different episodes
            seed = 1000 + call_count
            task_id = (call_count % 3) + 1
            
            # CRITICAL: Reset environment for each completion
            reset_resp = requests.post(
                f"{env_url}/reset",
                json={"task_id": task_id, "seed": seed},
                timeout=30
            )
            if reset_resp.status_code != 200:
                rewards.append(-0.5)
                batch_rewards.append(-0.5)
                continue
            
            # Run 8 steps
            num_steps = 8
            total_reward = 0.0
            for _ in range(num_steps):
                step_resp = requests.post(
                    f"{env_url}/step",
                    json=[step_action],
                    timeout=30
                )
                if step_resp.status_code != 200:
                    break
                step_data = step_resp.json()
                if isinstance(step_data, list):
                    step_data = step_data[0]
                total_reward += float(step_data.get("reward", 0))
            
            avg_reward = total_reward / num_steps if num_steps > 0 else 0
            
            # Get episode score from /grade
            grade_resp = requests.get(f"{env_url}/grade", timeout=30)
            if grade_resp.status_code == 200:
                episode_score = float(grade_resp.json().get("score", 0.5))
                normalized = max(0.0, min(1.0, (episode_score - 0.4) / 0.32))
                final_reward = normalized
            else:
                final_reward = max(-1.0, min(1.0, avg_reward / 10.0))
            
            rewards.append(final_reward)
            batch_rewards.append(final_reward)
            
        except json.JSONDecodeError:
            rewards.append(-0.8)
            batch_rewards.append(-0.8)
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(-0.5)
            batch_rewards.append(-0.5)
    
    return rewards


def run_diagnostic():
    print("=== PRE-TRAINING REWARD FUNCTION DIAGNOSTIC ===")
    print("Testing reward variance with 8 random actions...\n")
    
    requests.post(f"{ENV_URL}/reset", json={"task_id": 1}, timeout=10)
    
    test_completions = [
        # Good action — efficient
        '{"hvac_power_level": 0.3, "thermal_charge_rate": 0.8, "batch_job_slot": 2, "load_shed_fraction": 0.0, "building_id": 0}',
        # Bad action — wasteful
        '{"hvac_power_level": 1.0, "thermal_charge_rate": -1.0, "batch_job_slot": 0, "load_shed_fraction": 0.5, "building_id": 0}',
        # Medium action
        '{"hvac_power_level": 0.5, "thermal_charge_rate": 0.0, "batch_job_slot": 1, "load_shed_fraction": 0.1, "building_id": 0}',
        # Invalid JSON — should get -1.0
        'I will set the HVAC to medium power level',
        # Another good action
        '{"hvac_power_level": 0.2, "thermal_charge_rate": 0.6, "batch_job_slot": 3, "load_shed_fraction": 0.0, "building_id": 0}',
        # Another bad action
        '{"hvac_power_level": 0.9, "thermal_charge_rate": -0.8, "batch_job_slot": 0, "load_shed_fraction": 0.4, "building_id": 0}',
        # Good charge during cheap hours
        '{"hvac_power_level": 0.4, "thermal_charge_rate": 0.9, "batch_job_slot": 2, "load_shed_fraction": 0.0, "building_id": 0}',
        # Bad during peak
        '{"hvac_power_level": 0.8, "thermal_charge_rate": -0.5, "batch_job_slot": 0, "load_shed_fraction": 0.3, "building_id": 0}',
    ]
    
    test_rewards = gridmind_reward_fn(test_completions)
    
    print("Completion type            → Reward")
    print("-" * 45)
    labels = [
        "Good (efficient)",
        "Bad (wasteful)",
        "Medium",
        "Invalid JSON",
        "Good (store)",
        "Bad (discharge peak)",
        "Good (charge cheap)",
        "Bad (peak demand)",
    ]
    for label, reward in zip(labels, test_rewards):
        bar = "█" * int(abs(reward) * 20)
        sign = "+" if reward >= 0 else "-"
        print(f"  {label:<25} → {reward:+.4f}  {bar}")
    
    if len(test_rewards) > 1:
        variance = np.var(test_rewards)
        reward_range = max(test_rewards) - min(test_rewards)
        print(f"\nReward variance:  {variance:.4f}")
        print(f"Reward range:     {reward_range:.4f}")
        
        if variance < 0.01:
            print("\n❌ CRITICAL: Reward variance is near zero!")
            print("   GRPO cannot learn from this. Fix the reward function before training.")
            print("   Check that the environment is being reset between calls.")
            return False
        elif variance < 0.05:
            print("\n⚠️  WARNING: Low reward variance. Training may be slow.")
            print("   Consider amplifying reward differences.")
            return True
        else:
            print("\n✓ Reward variance is sufficient for GRPO training.")
            print("  Proceed to training.")
            return True
    
    return False


if __name__ == "__main__":
    success = run_diagnostic()
    exit(0 if success else 1)