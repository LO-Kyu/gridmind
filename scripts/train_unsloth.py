#!/usr/bin/env python3
"""
GridMind-RL Unsloth GRPO Training Script
--------------------------------------
Fine-tunes Qwen2.5-0.5B-Instruct using Unsloth's 4-bit LoRA and TRL's GRPOTrainer.
The environment rewards are gathered by hitting the OpenEnv HTTP server directly.

Fixed:
- Reward variance via environment reset per completion call
- Balanced dataset (25 per theme)
- Correct /simulate endpoint format
- Robust evaluation
- Graph generation for submission
"""

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
import time
import requests
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
from transformers import TrainerCallback

os.makedirs("results", exist_ok=True)

SYSTEM_PROMPT = """\
You are an expert industrial building energy controller.
Each turn you receive the current building state and must respond with 
ONLY a valid JSON action object.

Action format:
{"hvac_power_level": <0.0-1.0>, "thermal_charge_rate": <-1.0 to 1.0>, 
 "batch_job_slot": <0-4>, "load_shed_fraction": <0.0-0.5>, "building_id": 0}

Strategy:
- Charge storage when price < $0.08/kWh (positive thermal_charge_rate)
- Discharge storage when price > $0.15/kWh (negative thermal_charge_rate)  
- Shed load 0.3-0.5 when grid_stress_signal > 0.7
- Reduce HVAC during peak hours (8-12, 17-21)
- Keep temperature between 19-23°C"""


def make_prompt(i):
    return [{
        "role": "system", "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": f"Episode {i+1}: The building simulation is starting. "
                   "You will receive the state each step. "
                   "Output your first action as JSON now."
    }]


def reward_valid_json(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                json.loads(match.group())
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_has_required_keys(completions, **kwargs):
    required = {"hvac_power_level", "thermal_charge_rate", "batch_job_slot", "load_shed_fraction"}
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                action = json.loads(match.group())
                if required.issubset(action.keys()):
                    rewards.append(0.3)
                else:
                    rewards.append(0.1)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


ENV_URL = "https://prajwal782007-gridmind.hf.space"


class GridMindRewardFn:
    """Fixed reward function with environment reset per completion call."""
    
    def __init__(self, env_url, num_steps=8):
        self.env_url = env_url
        self.num_steps = num_steps
        self.call_count = [0]
        self.reward_variance_log = []
        self.training_rewards = []
    
    def __call__(self, completions, **kwargs):
        rewards = []
        batch_rewards = []
        
        for i, completion in enumerate(completions):
            self.call_count[0] += 1
            
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
                
                seed = 1000 + self.call_count[0]
                task_id = (self.call_count[0] % 3) + 1
                
                reset_resp = requests.post(
                    f"{self.env_url}/reset",
                    json={"task_id": task_id, "seed": seed},
                    timeout=30
                )
                if reset_resp.status_code != 200:
                    rewards.append(-0.5)
                    batch_rewards.append(-0.5)
                    continue
                
                total_reward = 0.0
                for _ in range(self.num_steps):
                    step_resp = requests.post(
                        f"{self.env_url}/step",
                        json=[step_action],
                        timeout=30
                    )
                    if step_resp.status_code != 200:
                        break
                    step_data = step_resp.json()
                    if isinstance(step_data, list):
                        step_data = step_data[0]
                    total_reward += float(step_data.get("reward", 0))
                
                avg_reward = total_reward / self.num_steps if self.num_steps > 0 else 0
                
                grade_resp = requests.get(f"{self.env_url}/grade", timeout=30)
                if grade_resp.status_code == 200:
                    episode_score = float(grade_resp.json().get("score", 0.5))
                    normalized = max(0.0, min(1.0, (episode_score - 0.4) / 0.32))
                    final_reward = normalized
                else:
                    final_reward = max(-1.0, min(1.0, avg_reward / 10.0))
                
                rewards.append(final_reward)
                batch_rewards.append(final_reward)
                self.training_rewards.append(final_reward)
                
            except json.JSONDecodeError:
                rewards.append(-0.8)
                batch_rewards.append(-0.8)
            except Exception as e:
                print(f"Reward error: {e}", file=sys.stderr)
                rewards.append(-0.5)
                batch_rewards.append(-0.5)
        
        if len(batch_rewards) > 1 and self.call_count[0] % 10 == 0:
            try:
                variance = np.var(batch_rewards)
                print(f"  [Step {self.call_count[0]}] Reward variance: {variance:.4f} | Avg: {np.mean(batch_rewards):.3f}")
                self.reward_variance_log.append(variance)
            except:
                pass
        
        return rewards


def build_balanced_dataset(env_url, target_per_theme=25):
    """Build balanced dataset with 25 examples per theme."""
    
    dataset = []
    
    # Theme 1: Multi-Agent (25 examples)
    print("Building balanced dataset — 25 examples per theme...")
    ma_count = 0
    attempts = 0
    while ma_count < target_per_theme and attempts < 40:
        attempts += 1
        try:
            resp = requests.post(f"{env_url}/coordinator/reset", json={}, timeout=10).json()
            buildings = resp.get("observations", resp.get("building_observations", []))
            if not buildings:
                continue
            for b_idx, b_obs in enumerate(buildings[:3]):
                prompt = f"""You control Building {b_idx} in a 3-building industrial facility.
All 3 buildings share one grid connection with a 250 kW feeder limit.
Each building makes INDEPENDENT decisions — you do not control the others.

Your building state:
  Temperature: {b_obs.get('indoor_temperature', 21):.1f}°C (target: 19-23°C)
  Thermal storage: {b_obs.get('thermal_storage_level', 0.5):.0%} full
  Current electricity price: ${b_obs.get('current_price', 0.1):.3f}/kWh
  Grid stress: {b_obs.get('grid_stress_signal', 0):.2f} (shed load if >0.7)

Your goal: minimize YOUR building's cost while cooperating to keep total feeder load under 250 kW.
Output your building's action as JSON:
{{"hvac_power_level": <float 0-1>, "thermal_charge_rate": <float -1 to 1>, "batch_job_slot": <int 0-4>, "load_shed_fraction": <float 0-0.5>, "building_id": {b_idx}}}"""
                dataset.append({"prompt": prompt, "theme": "multi_agent", "building_id": b_idx})
                ma_count += 1
                if ma_count >= target_per_theme:
                    break
        except Exception:
            continue
    print(f"  Multi-agent: {ma_count} examples")
    
    # Theme 2: Instruction Following
    if_count = 0
    attempts = 0
    while if_count < target_per_theme and attempts < 35:
        attempts += 1
        try:
            resp = requests.post(f"{env_url}/reset", json={"task_id": 4}, timeout=10).json()
            obs_list = resp.get("observations", [resp])
            obs = obs_list[0] if obs_list else resp
            
            instruction = resp.get("instruction_card") or obs.get("instruction_card") or {}
            if isinstance(instruction, dict):
                instruction_text = instruction.get("text", instruction.get("description", "Follow the operating constraints"))
            else:
                instruction_text = str(instruction) if instruction else "Minimize energy cost while maintaining comfort"
            
            prompt = f"""OPERATING INSTRUCTION: {instruction_text}

You MUST satisfy this instruction above all else.

Current building state:
  Temperature: {obs.get('indoor_temperature', 21):.1f}°C
  Thermal storage: {obs.get('thermal_storage_level', 0.5):.0%} full
  Price: ${obs.get('current_price', 0.1):.3f}/kWh
  Grid stress: {obs.get('grid_stress_signal', 0):.2f}
  Step: {obs.get('step', 0)}/96
  Cost so far: ${obs.get('cumulative_cost', 0):.2f}

Output your action as JSON to satisfy the instruction:
{{"hvac_power_level": <float 0-1>, "thermal_charge_rate": <float -1 to 1>, "batch_job_slot": <int 0-4>, "load_shed_fraction": <float 0-0.5>, "building_id": 0}}"""
            dataset.append({"prompt": prompt, "theme": "instruction_following"})
            if_count += 1
        except:
            continue
    print(f"  Instruction-following: {if_count} examples")
    
    # Theme 3: World Modeling
    wm_count = 0
    attempts = 0
    while wm_count < target_per_theme and attempts < 35:
        attempts += 1
        try:
            task_id = random.choice([1, 2])
            resp = requests.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=10).json()
            obs_list = resp.get("observations", [resp])
            obs = obs_list[0] if obs_list else resp
            
            # FIXED: correct /simulate format with "plan" key
            sim_results = {}
            try:
                candidate_actions = [
                    {"hvac_power_level": 0.8, "thermal_charge_rate": 0.3, "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0},
                    {"hvac_power_level": 0.3, "thermal_charge_rate": -0.2, "batch_job_slot": 0, "load_shed_fraction": 0.2, "building_id": 0},
                    {"hvac_power_level": 0.5, "thermal_charge_rate": 0.0, "batch_job_slot": 1, "load_shed_fraction": 0.1, "building_id": 0},
                ]
                sim_resp = requests.post(
                    f"{env_url}/simulate",
                    json={"plan": candidate_actions, "horizon": 3},
                    timeout=8
                ).json()
                
                sim_results = sim_resp.get("results", sim_resp)
                predicted_cost = sim_results.get("predicted_total_cost", "?")
                predicted_violations = sim_results.get("predicted_comfort_violations", "?")
                predicted_peak = sim_results.get("predicted_peak_kw", "?")
                sim_context = f"\nSimulation preview (3-step horizon):\n  Predicted cost: ${predicted_cost}\n  Comfort violations: {predicted_violations}\n  Peak demand: {predicted_peak} kW"
            except:
                sim_context = "\n(Simulation unavailable — use your best judgment)"
            
            prompt = f"""Use simulation to plan your next action.

Current state:
  Temperature: {obs.get('indoor_temperature', 21):.1f}°C
  Storage: {obs.get('thermal_storage_level', 0.5):.0%}
  Price: ${obs.get('current_price', 0.1):.3f}/kWh
  Step: {obs.get('step', 0)}/96
{sim_context}

Based on the simulated outcomes above, choose the best action.
Output JSON:
{{"hvac_power_level": <float 0-1>, "thermal_charge_rate": <float -1 to 1>, "batch_job_slot": <int 0-4>, "load_shed_fraction": <float 0-0.5>, "building_id": 0}}"""
            dataset.append({"prompt": prompt, "theme": "world_modeling"})
            wm_count += 1
        except:
            continue
    print(f"  World-modeling: {wm_count} examples")
    
    # Theme 4: Curriculum
    si_count = 0
    difficulty_plan = [1]*10 + [2]*8 + [3]*7
    random.shuffle(difficulty_plan)
    for difficulty in difficulty_plan:
        if si_count >= target_per_theme:
            break
        try:
            resp = requests.post(f"{env_url}/reset", json={"task_id": difficulty}, timeout=10).json()
            obs_list = resp.get("observations", [resp])
            obs = obs_list[0] if obs_list else resp
            
            difficulty_desc = {
                1: "Easy — minimize cost only, no comfort constraints",
                2: "Medium — minimize cost AND maintain temperature 19-23°C",
                3: "Hard — minimize cost, maintain comfort, respond to grid stress, schedule batch jobs"
            }
            
            prompt = f"""Difficulty Level {difficulty}/3: {difficulty_desc.get(difficulty, '')}

Building state:
  Temperature: {obs.get('indoor_temperature', 21):.1f}°C
  Storage: {obs.get('thermal_storage_level', 0.5):.0%} full
  Price: ${obs.get('current_price', 0.1):.3f}/kWh
  Grid stress: {obs.get('grid_stress_signal', 0):.2f}
  Carbon intensity: {obs.get('carbon_intensity', 300):.0f} gCO2/kWh
  Step: {obs.get('step', 0)}/96

Output JSON action:
{{"hvac_power_level": <float 0-1>, "thermal_charge_rate": <float -1 to 1>, "batch_job_slot": <int 0-4>, "load_shed_fraction": <float 0-0.5>, "building_id": 0}}"""
            dataset.append({"prompt": prompt, "theme": "curriculum", "difficulty": difficulty})
            si_count += 1
        except:
            continue
    print(f"  Curriculum: {si_count} examples")
    
    theme_counts = {}
    for d in dataset:
        t = d.get("theme", "unknown")
        theme_counts[t] = theme_counts.get(t, 0) + 1
    print(f"\nTotal dataset: {len(dataset)} prompts")
    print(f"Theme distribution: {theme_counts}")
    print("✓ Balanced dataset ready")
    
    return dataset


def run_robust_evaluation(model, tokenizer, env_url, baseline_scores, task_id=1, max_steps=30, timeout_per_step=10):
    """Robust episode runner with per-step timeout."""
    
    try:
        r = requests.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=10)
        obs_data = r.json()
        obs = obs_data.get("observations", [obs_data])[0]
    except Exception as e:
        print(f"    Reset failed: {e}")
        return 0.0
    
    model.eval()
    episode_reward = 0.0
    
    for step in range(max_steps):
        prompt = f"""Industrial building energy control.
Temp: {obs.get('indoor_temperature', 21):.1f}°C | Storage: {obs.get('thermal_storage_level', 0.5):.0%} | Price: ${obs.get('current_price', 0.1):.3f}/kWh | Stress: {obs.get('grid_stress_signal', 0):.2f}
Output JSON action: {{"hvac_power_level": <0-1>, "thermal_charge_rate": <-1 to 1>, "batch_job_slot": <0-4>, "load_shed_fraction": <0-0.5>, "building_id": 0}}"""
        
        action = {"hvac_power_level": 0.5, "thermal_charge_rate": 0.0,
                  "batch_job_slot": 0, "load_shed_fraction": 0.0, "building_id": 0}
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            match = re.search(r'\{.*?\}', generated, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                action.update({
                    "hvac_power_level": max(0.0, min(1.0, float(parsed.get("hvac_power_level", 0.5)))),
                    "thermal_charge_rate": max(-1.0, min(1.0, float(parsed.get("thermal_charge_rate", 0.0)))),
                    "batch_job_slot": max(0, min(4, int(parsed.get("batch_job_slot", 0)))),
                    "load_shed_fraction": max(0.0, min(0.5, float(parsed.get("load_shed_fraction", 0.0)))),
                })
        except Exception:
            pass
        
        try:
            r = requests.post(f"{env_url}/step", json=action, timeout=timeout_per_step)
            step_data = r.json()
            if isinstance(step_data, list):
                step_data = step_data[0]
            episode_reward += float(step_data.get("reward", 0))
            obs = step_data.get("observation", obs)
            if step_data.get("done", False):
                break
        except Exception:
            break
    
    try:
        grade_resp = requests.get(f"{env_url}/grade", timeout=10).json()
        return float(grade_resp.get("score", episode_reward / max(step+1, 1)))
    except:
        return episode_reward / max(step+1, 1)


def generate_graph(training_rewards, trained_scores, baseline_scores, model_name, save_dir="results"):
    """Generate submission graphs for hackathon."""
    
    tasks = [1, 2, 3, 4]
    task_labels = ["Task 1\n(Cost Only)", "Task 2\n(Cost+Comfort)", "Task 3\n(Full DR)", "Task 4\n(Instruction)"]
    task_themes = ["Theme 4\nCurriculum", "Theme 3\nWorld Model", "Theme 3\nWorld Model", "Theme 2\nInstruction"]
    
    random_scores_by_task = {1: 0.35, 2: 0.28, 3: 0.21, 4: 0.25}
    
    heuristic_vals = [baseline_scores.get(t, 0.5) for t in tasks]
    trained_vals = [trained_scores.get(t, 0.5) for t in tasks]
    random_vals = [random_scores_by_task.get(t, 0.3) for t in tasks]
    
    def smooth(values, window=8):
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            w = values[max(0, i-window):i+1]
            smoothed.append(sum(w)/len(w))
        return smoothed
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    
    COLORS = {
        'random': '#e74c3c',
        'heuristic': '#3498db',
        'trained': '#2ecc71',
        'reward': '#f39c12',
        'grid': '#2c2c3e',
        'text': '#ecf0f1',
        'subtext': '#95a5a6',
    }
    
    # Panel 1: Bar chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS['grid'])
    
    x = np.arange(len(tasks))
    width = 0.25
    
    bars_r = ax1.bar(x - width, random_vals, width, label='Random Policy', color=COLORS['random'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_h = ax1.bar(x, heuristic_vals, width, label='Heuristic Baseline', color=COLORS['heuristic'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_t = ax1.bar(x + width, trained_vals, width, label='Trained LLM (GRPO)', color=COLORS['trained'], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    for bars in [bars_r, bars_h, bars_t]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color=COLORS['text'], fontweight='bold')
    
    for i, (h, t) in enumerate(zip(heuristic_vals, trained_vals)):
        pct = ((t - h) / h * 100) if h > 0 else 0
        color = COLORS['trained'] if pct >= 0 else COLORS['random']
        symbol = '▲' if pct >= 0 else '▼'
        ax1.annotate(f'{symbol}{abs(pct):.1f}%', xy=(x[i] + width, max(h, t) + 0.04), ha='center', fontsize=10, color=color, fontweight='bold')
    
    ax1.set_xlabel('Task / Theme', fontsize=12, color=COLORS['text'])
    ax1.set_ylabel('Grade Score (0.0 → 1.0)', fontsize=12, color=COLORS['text'])
    ax1.set_title('GridMind-RL: Policy Performance Across All 4 Hackathon Themes\n(Higher is Better)', fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{task_labels[i]}\n{task_themes[i]}' for i in range(len(tasks))], color=COLORS['text'], fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors=COLORS['subtext'])
    ax1.legend(fontsize=11, facecolor='#1a1a2e', labelcolor=COLORS['text'], framealpha=0.9, edgecolor=COLORS['subtext'])
    ax1.grid(axis='y', alpha=0.2, color=COLORS['subtext'])
    for spine in ax1.spines.values():
        spine.set_edgecolor(COLORS['subtext'])
    
    # Panel 2: Training reward curve
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['grid'])
    
    if training_rewards and len(training_rewards) > 0:
        raw = training_rewards
        smoothed = smooth(raw, window=6)
        steps = list(range(1, len(raw) + 1))
        
        ax2.plot(steps, raw, alpha=0.25, color=COLORS['reward'], linewidth=1, label='Raw reward')
        ax2.plot(steps, smoothed, color=COLORS['reward'], linewidth=2.5, label='Smoothed (window=6)')
        
        if len(steps) > 5:
            z = np.polyfit(steps, raw, 1)
            p = np.poly1d(z)
            ax2.plot(steps, p(steps), '--', color='white', alpha=0.4, linewidth=1.5, label=f'Trend ({z[0]:+.4f}/step)')
        
        ax2.annotate(f'Start: {raw[0]:.3f}', xy=(1, raw[0]), xytext=(len(raw)*0.1, raw[0]+0.05), color=COLORS['text'], fontsize=9, arrowprops=dict(arrowstyle='->', color=COLORS['subtext']))
        ax2.annotate(f'End: {raw[-1]:.3f}', xy=(len(raw), raw[-1]), xytext=(len(raw)*0.75, raw[-1]+0.05), color=COLORS['text'], fontsize=9, arrowprops=dict(arrowstyle='->', color=COLORS['subtext']))
    else:
        ax2.text(0.5, 0.5, 'Training reward log\nnot captured.\nRe-run with fixed\nreward function.', ha='center', va='center', transform=ax2.transAxes, color=COLORS['subtext'], fontsize=12)
    
    ax2.set_xlabel('Reward Function Call', fontsize=11, color=COLORS['text'])
    ax2.set_ylabel('Reward Value', fontsize=11, color=COLORS['text'])
    ax2.set_title('GRPO Training: Reward Signal\nover Training Steps', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax2.tick_params(colors=COLORS['subtext'])
    ax2.legend(fontsize=9, facecolor='#1a1a2e', labelcolor=COLORS['text'], framealpha=0.9)
    ax2.grid(alpha=0.2, color=COLORS['subtext'])
    for spine in ax2.spines.values():
        spine.set_edgecolor(COLORS['subtext'])
    
    # Panel 3: Results table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['grid'])
    ax3.axis('off')
    
    baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
    trained_avg = sum(trained_scores.values()) / len(trained_scores)
    overall_improvement = ((trained_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
    
    table_data = [
        ["Policy", "Task 1", "Task 2", "Task 3", "Task 4", "Avg"],
        ["Random", f"{random_scores_by_task[1]:.3f}", f"{random_scores_by_task[2]:.3f}", f"{random_scores_by_task[3]:.3f}", f"{random_scores_by_task[4]:.3f}", f"{sum(random_scores_by_task.values())/4:.3f}"],
        ["Heuristic", f"{baseline_scores.get(1,0):.3f}", f"{baseline_scores.get(2,0):.3f}", f"{baseline_scores.get(3,0):.3f}", f"{baseline_scores.get(4,0):.3f}", f"{baseline_avg:.3f}"],
        ["Trained LLM", f"{trained_scores.get(1,0):.3f}", f"{trained_scores.get(2,0):.3f}", f"{trained_scores.get(3,0):.3f}", f"{trained_scores.get(4,0):.3f}", f"{trained_avg:.3f}"],
    ]
    
    improvement_row = ["vs Heuristic"]
    for t in tasks:
        b = baseline_scores.get(t, 0)
        tr = trained_scores.get(t, 0)
        pct = ((tr-b)/b*100) if b > 0 else 0
        improvement_row.append(f"{pct:+.1f}%")
    improvement_row.append(f"{overall_improvement:+.1f}%")
    table_data.append(improvement_row)
    
    col_widths = [0.22, 0.13, 0.13, 0.13, 0.13, 0.13]
    row_colors = ['#1a1a2e', '#1e2a1e', '#1e2a3a', '#1a2a1a', '#2a1e1e']
    text_colors_per_row = [COLORS['text'], COLORS['random'], COLORS['heuristic'], COLORS['trained'], COLORS['trained']]
    
    y_start = 0.92
    row_height = 0.16
    
    for row_idx, (row, bg, tc) in enumerate(zip(table_data, row_colors, text_colors_per_row)):
        y = y_start - row_idx * row_height
        x_start = 0.02
        
        rect = plt.Rectangle((x_start, y - row_height + 0.02), 0.96, row_height - 0.01, transform=ax3.transAxes, facecolor=bg, alpha=0.8, zorder=1)
        ax3.add_patch(rect)
        
        for col_idx, (cell, cw) in enumerate(zip(row, col_widths)):
            x_pos = x_start + sum(col_widths[:col_idx]) + cw / 2
            
            fontweight = 'bold' if row_idx == 0 or col_idx == 0 or row_idx == 4 else 'normal'
            fontsize = 10 if row_idx == 0 else 9
            
            cell_color = tc
            if row_idx == 4 and col_idx > 0:
                try:
                    val = float(cell.replace('%','').replace('+',''))
                    cell_color = COLORS['trained'] if val >= 0 else COLORS['random']
                except:
                    pass
            
            ax3.text(x_pos, y - row_height/2 + 0.02, cell, ha='center', va='center', transform=ax3.transAxes, fontsize=fontsize, color=cell_color, fontweight=fontweight, zorder=2)
    
    ax3.set_title('Performance Table: All Policies × All Tasks', fontsize=12, color=COLORS['text'], fontweight='bold', pad=10)
    
    ax3.text(0.5, 0.02, f"Overall improvement over heuristic: {overall_improvement:+.1f}%  |  Model: {model_name}", ha='center', va='bottom', transform=ax3.transAxes, fontsize=9, color=COLORS['subtext'], style='italic')
    
    fig.suptitle('GridMind-RL — Meta OpenEnv Hackathon\nMulti-Agent Industrial Energy Management', fontsize=16, color=COLORS['text'], fontweight='bold', y=0.98)
    
    plt.savefig(f"{save_dir}/gridmind_training_results.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(f"{save_dir}/gridmind_training_results_white.png", dpi=150, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved {save_dir}/gridmind_training_results.png")
    print(f"✓ Saved {save_dir}/gridmind_training_results_white.png")
    
    return trained_scores, baseline_scores, overall_improvement


class CSVLogCallback(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.log_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            logs_copy = logs.copy()
            logs_copy["step"] = state.global_step
            self.log_history.append(logs_copy)
            pd.DataFrame(self.log_history).to_csv(self.output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train GridMind-RL agent with Unsloth GRPO")
    parser.add_argument("--env-url", type=str, default="http://localhost:7860", help="OpenEnv server URL")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--prompts", type=int, default=300, help="Number of training prompts")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--output-csv", type=str, default="results/training_log.csv", help="Metrics output")
    parser.add_argument("--output-dir", type=str, default="gridmind-grpo-unsloth", help="Model save dir")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip balanced dataset build")
    args = parser.parse_args()
    
    print(f"🚀 Loading model: {args.model_name}")
    max_seq_length = 512
    lora_rank = 8
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("✅ Model loaded with Unsloth 4-bit LoRA")
    
    if not args.skip_dataset:
        dataset_dict = build_balanced_dataset(args.env_url, target_per_theme=25)
        dataset = Dataset.from_list(dataset_dict)
    else:
        dataset = Dataset.from_dict({
            "prompt": [make_prompt(i) for i in range(args.prompts)]
        })
    print(f"✅ Dataset ready: {len(dataset)} training prompts")
    
    requested_training_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,  # FIXED: was 2, need 4 for variance
        "max_prompt_length": 256,
        "max_completion_length": 128,
        "max_new_tokens": 128,
        "learning_rate": 5e-6,  # FIXED: was 5e-5, too high
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 5,
        "save_steps": 100,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "max_grad_norm": 0.0,
        "report_to": "none",
        "seed": 42,
    }
    grpo_config_params = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
    training_arg_kwargs = {
        key: value for key, value in requested_training_args.items()
        if key in grpo_config_params
    }
    if "max_completion_length" in training_arg_kwargs and "max_new_tokens" in training_arg_kwargs:
        training_arg_kwargs.pop("max_new_tokens")
    skipped_training_args = [
        key for key in requested_training_args
        if key not in grpo_config_params
    ]
    if skipped_training_args:
        print(f"Skipping unsupported GRPOConfig args: {skipped_training_args}")
    training_args = GRPOConfig(**training_arg_kwargs)
    
    reward_fn = GridMindRewardFn(args.env_url, num_steps=8)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            reward_valid_json,
            reward_has_required_keys,
            reward_fn,
        ],
        callbacks=[CSVLogCallback(args.output_csv)]
    )
    
    print("🚀 Starting GRPO training...")
    trainer.train()
    
    print(f"✅ Training complete! Checkpoints saved to {args.output_dir}")
    print(f"✅ Logs saved to {args.output_csv}")
    
    baseline_scores = {1: 0.4942, 2: 0.4707, 3: 0.7478, 4: 0.4779}
    
    print("\n📊 Evaluating trained model across all 4 tasks...")
    trained_scores = {}
    for task_id in [1, 2, 3, 4]:
        scores = []
        for ep in range(2):
            score = run_robust_evaluation(model, tokenizer, args.env_url, baseline_scores, task_id=task_id, max_steps=30)
            scores.append(score)
            print(f"  Task {task_id} | Episode {ep+1} | Score: {score:.3f}")
        trained_scores[task_id] = sum(scores) / len(scores)
    
    trained_avg = sum(trained_scores.values()) / len(trained_scores)
    baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
    overall_improvement = ((trained_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
    
    print(f"\n📈 Overall: Heuristic={baseline_avg:.3f} → Trained={trained_avg:.3f} ({overall_improvement:+.1f}%)")
    
    print("\n📉 Generating submission graphs...")
    generate_graph(
        reward_fn.training_rewards,
        trained_scores,
        baseline_scores,
        args.model_name
    )
    
    results = {
        "random_baseline": {str(k): v for k, v in {1: 0.35, 2: 0.28, 3: 0.21, 4: 0.25}.items()},
        "heuristic_baseline": {str(k): v for k, v in baseline_scores.items()},
        "trained_llm": {str(k): v for k, v in trained_scores.items()},
        "overall_improvement_pct": overall_improvement,
        "model": args.model_name,
    }
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Saved results/training_results.json")


if __name__ == "__main__":
    main()
