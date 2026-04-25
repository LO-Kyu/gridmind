---
title: GridMind-RL: Training LLMs to Manage Industrial Buildings Under Faults and Grid Stress
description: An OpenEnv-compatible RL environment where LLMs learn to control HVAC, thermal storage, and batch scheduling across multi-building industrial facilities.
---

**Every industrial building wastes 20–30% of its energy because control systems can't handle real-time pricing, equipment faults, and grid stress simultaneously.** GridMind-RL is an OpenEnv-compatible RL environment that makes LLMs trainable on this problem.

## The Problem

Industrial buildings consume ~40% of global electricity. Most still use naive "always-on" HVAC policies. The capability gap is clear:

- LLMs can understand complex pricing curves, fault alerts, and natural language instructions
- But no environment exists to train them on real building energy management
- Existing RL environments are mostly grid-worlds or toy games — not genuine industrial problems

GridMind-RL closes this gap by simulating a complete building energy system where agents must:

- Navigate 24-hour price volatility (off-peak vs peak: 4¢ to 32¢/kWh)
- Maintain comfort (19–23°C) while minimizing cost
- Respond to grid stress emergencies
- Handle equipment faults (chiller failure, sensor malfunction, grid outages, tariff spikes)
- Parse and follow natural language objective cards

## The Environment

GridMind-RL is a 96-step episode (24 simulated hours at 15-minute resolution) with:

| Field | Value |
|-------|-------|
| **Observation** | 13 fields: temperature, storage, price, stress, carbon, faults, HVAC efficiency, instruction card |
| **Actions** | HVAC level (0–1), thermal charge (−1 to 1), batch slot (0–4), load shed (0–0.5) |
| **Reward** | 9-component weighted sum: cost, temperature, grid, deadline, efficiency, stability, carbon, instruction, fault_mitigation |
| **Tasks** | 4 types: cost minimization, temperature management, demand response, instruction following |

### Four Hackathon Themes in One Environment

**Track 1 — Multi-Agent Interactions:** A coordinator LLM reads `/feeder` to see fleet-wide demand across 3 buildings, then sets per-building price multipliers via `/coordinate` to orchestrate behavior.

**Track 2 — Long-Horizon Planning & Instruction Following:** Task 4 presents a natural language objective card like "Keep total energy cost under $2.50 while maintaining 19–23°C." Agents must plan across all 96 steps.

**Track 3 — World Modeling:** The `/simulate` endpoint lets agents ask "what if?" before acting. When HVAC efficiency is low or faults are active, the agent simulates the proposed action and revises if the predicted reward is poor.

**Track 4 — Fault Handling:** Four fault types inject unpredictability:
- **Chiller failure**: HVAC drops to 20% capacity
- **Grid outage**: Price ×3, stress = 1.0
- **Sensor fault**: Temperature readings jitter ±5°C
- **Tariff spike**: Emergency 4× price surge

**Track 5 — Self-Improvement:** Curriculum learning auto-advances the agent from task 1 to task 4 when performance thresholds are met.

## Results

Heuristic baseline scores (fixed policy, no learning) across all 4 tasks:

| Policy | Task 1 | Task 2 | Task 3 | Task 4 |
|--------|--------|--------|--------|--------|
| **Heuristic Baseline** | 0.506 | 0.459 | 0.600 | 0.492 |

The GRPO fine-tuned model shows improvement over the zero-shot LLM baseline. The training curve below shows the learning trajectory:

![Training Curve](https://raw.githubusercontent.com/LO-Kyu/gridmind/main/results/training_curve.png)

## Training

GridMind-RL uses GRPO (Group Relative Policy Optimization) via HuggingFace TRL with Unsloth 4-bit LoRA fine-tuning of Qwen2.5-0.5B-Instruct. The training script connects to the live environment via HTTP, running 8-step rollouts and using the `/grade` endpoint (episode-level score 0.0–1.0) as the primary reward signal.

```python
# Training runs against the live environment
python scripts/train_unsloth.py --steps 500 --output-csv results/training_log.csv
```

Or run the Colab notebook: [gridmind_grpo_colab.ipynb](https://colab.research.google.com/)

## How to Try It

```bash
# Quick health check
curl https://lo-kyu-gridmind.hf.space/health

# Run a heuristic baseline
python inference.py --fast-mode --task 3 --episodes 5

# Run the LLM agent
python inference.py --task 3 --episodes 5
```

Live environment: [https://lo-kyu-gridmind.hf.space](https://lo-kyu-gridmind.hf.space)  
Dashboard: [https://lo-kyu-gridmind.hf.space/dashboard](https://lo-kyu-gridmind.hf.space/dashboard)

Code: [github.com/LO-Kyu/gridmind](https://github.com/LO-Kyu/gridmind)

---

*GridMind-RL was built for the Meta PyTorch OpenEnv Hackathon Grand Finale, April 25–26, 2026, at Scaler School of Technology, Bangalore.*