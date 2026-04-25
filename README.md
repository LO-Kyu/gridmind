---
title: GridMind-RL
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GridMind-RL — Train LLMs to manage industrial buildings under faults, grid stress, and natural language objectives.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.org/)
[![Go 1.21](https://img.shields.io/badge/Go-1.21-00ADD8)](https://golang.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why This Environment Is Novel

Most RL environments for LLMs are grid-worlds or toy games. GridMind-RL simulates a **real industrial problem** — building energy management — where agents must juggle stochastic electricity prices, multi-objective constraints, equipment faults, and natural language operating objectives. An LLM that learns to manage a building under these conditions has a genuinely useful skill, not just a high game score.

## Live Demo

| | URL |
|--|-----|
| **Environment API** | https://prajwal782007-gridmind.hf.space |
| **Live Dashboard** | https://prajwal782007-gridmind.hf.space/dashboard |

**Quick test:**
```bash
curl https://prajwal782007-gridmind.hf.space/health
curl https://prajwal782007-gridmind.hf.space/tasks
```

---

## Problem

Industrial buildings consume ~40% of global electricity, yet most still use naive "always-on" HVAC policies. The capability gap is clear: **LLMs can understand complex pricing curves, natural language instructions, and fault alerts—but no environment exists to train them to manage buildings.**

GridMind-RL closes this gap by simulating a complete building energy system where agents must:
- Navigate 24-hour price volatility (off-peak vs peak: 4¢ to 32¢/kWh)
- Maintain comfort (19-23°C) while minimizing cost
- Respond to grid stress emergencies
- Handle equipment faults (chiller failure, sensor malfunction, grid outages)
- Parse and follow natural language objective cards

---

## Environment

| | Description |
|---|-------------|
| **Observation** | 11 fields: temperature, storage, price, stress, carbon, faults, HVAC efficiency |
| **Actions** | HVAC level (0-1), thermal charge (-1 to 1), batch slot (0-4), load shed (0-0.5) |
| **Reward** | 9-component weighted sum: cost, temperature, grid, deadline, efficiency, stability, carbon, instruction, fault_mitigation |
| **Episode** | 96 steps = 24 simulated hours @ 15-min resolution |
| **Tasks** | 4 tasks: (1) cost, (2) temperature, (3) demand_response, (4) instruction_following |

### Reward Weight Rationale

Weights reflect real-world building operator priorities — not arbitrary values:

| Component | Weight | Rationale |
|---|---|---|
| `cost_savings` | 0.28 | Primary operator KPI — energy spend is the main business metric |
| `carbon_reward` | 0.20 | ESG compliance — increasingly mandatory for industrial operators |
| `temp_constraint` | 0.20 | Hard safety constraint — comfort SLA violations incur penalties |
| `grid_response` | 0.20 | Regulatory SLA — demand response programs pay operators to shed load |
| `batch_deadline` | 0.12 | Production continuity — missing batch deadlines causes downstream losses |
| `efficiency_bonus` | 0.05 | Storage arbitrage — incentivises smart charge/discharge timing |
| `stability_penalty` | -0.05 | Anti-cycling — prevents HVAC thrashing that causes equipment wear |
| `fault_mitigation` | 0.05 | Emergency response — correct fault handling prevents costly outages |
| `instruction_reward` | 0.50* | Task 4 only — weighted per the episode's instruction card |

> *Task 4 instruction reward weight comes from the sampled instruction card, not a fixed value.

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| indoor_temperature | float | °C |
| thermal_storage_level | float | 0-1 (0=empty, 1=full) |
| current_price | float | $/kWh |
| grid_stress_signal | float | 0-1 (>0.7 = critical) |
| hvac_efficiency | float | 1.0 → degrades to 0.5 over episode |
| active_faults | string[] | Active fault alarm strings |
| instruction_card | object | Task 4 objective only |

### Action Fields

| Field | Type | Range |
|-------|------|-------|
| hvac_power_level | float | 0.0-1.0 |
| thermal_charge_rate | float | -1.0 to 1.0 |
| batch_job_slot | int | 0-4 |
| load_shed_fraction | float | 0.0-0.5 |

---

## Five Tracks

### Track 1: Multi-Agent Interactions
A single oversight LLM coordinates multiple buildings through price signals. The coordinator reads `/feeder` to see fleet-wide demand, then sets per-building price multipliers via `/coordinate` to orchestrate behavior.

### Track 2: Long-Horizon Planning & Instruction Following
Task 4 presents a natural language objective card like "Keep total energy cost under $2.50 while maintaining 19-23°C". Agents must plan across all 96 steps—not greedy per-step control.

### Track 3: World Modeling
The `/simulate` endpoint lets agents ask "what if?" before acting. When HVAC efficiency is low or faults are active, the agent simulates the proposed action and revises if the predicted reward is poor.

### Track 4: Fault Handling (Wild Card)
Four fault types inject unpredictability:
- **Chiller failure**: HVAC drops to 20% capacity
- **Grid outage**: Price ×3, stress = 1.0
- **Sensor fault**: Temperature readings jitter ±5°C
- **Tariff spike**: Emergency 4× price surge

### Track 5: HVAC Degradation
Real HVAC systems degrade over time. Efficiency starts at 1.0 and drops ~0.1% per step. The agent must account for declining capacity—a hidden state requiring inference.

---

## Results

![Training Curve](results/training_curve.png)
*Episode grade scores vs training step. Heuristic baseline (red) vs GRPO fine-tuned LLM (teal). Higher = better energy management.*

| Policy | Task 1 | Task 2 | Task 3 | Task 4 |
|--------|--------|--------|--------|--------|
| Heuristic Baseline | 0.506 | 0.459 | 0.600 | 0.492 |
| Zero-shot LLM | 0.715 | 0.645 | 0.610 | 0.582 |
| GRPO Fine-tuned LLM | TBD | TBD | TBD | TBD |

> Scores are episode grade scores (0.0–1.0, clamped open interval). Heuristic = fixed policy with no learning. Zero-shot = pretrained Qwen2.5-7B-Instruct. Fine-tuned = GRPO-trained on GridMind-RL environment.

---

## How to Run

### Start the environment server
```bash
go run main.go
```

### Run the LLM agent (task 1-4)
```bash
# Set up your API token
cp .env.example .env
# Edit .env with HF_TOKEN

# Task 1: Cost minimization
python inference.py --task 1 --episodes 5

# Task 2: Temperature management  
python inference.py --task 2 --episodes 5

# Task 3: Full demand response
python inference.py --task 3 --episodes 5

# Task 4: Instruction following
python inference.py --task 4 --episodes 5

# Heuristic baseline (fast, no LLM)
python inference.py --fast-mode --task 3 --episodes 5
```

### Run multi-building coordinator demo
```bash
python scripts/multi_building_demo.py
```

### Run training (requires GPU)
```bash
python scripts/train_unsloth.py --steps 500 --output-csv results/training_log.csv
```

### Generate training curve plot
```bash
python scripts/plot_results.py
```

---

## Self-Improvement: Curriculum Learning

The `--curriculum` flag enables automatic task progression:
- Agent starts on Task 1 (easy)
- After 5 episodes with average reward ≥ 0.55, advances to Task 2
- After 5 episodes with average reward ≥ 0.50, advances to Task 3
- After 5 episodes with average reward ≥ 0.45, advances to Task 4

This directly targets the Self-Improvement hackathon theme.

---

## Architecture

```
Agent (python/inference.py)
    → HTTP POST /step, /reset, /grade
    ↓
Go Environment Server (main.go) → Port 7860
    ↓
Physics Engine (env/environment.go) + Rewards (env/rewards.go) + Tasks (env/tasks.go)
    ↓
Web Dashboard (dashboard/server.py) → Port 7861
```

**Design philosophy:**
- **Separation of concerns**: Physics engine (Go) decoupled from policy layer (Python)
- **OpenEnv compliance**: Standardized REST API enables any language agent
- **Deterministic simulation**: Seeded RNG for reproducible experiments
- **Dense rewards**: 9-component reward for effective learning

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /ping | Liveness probe |
| POST | /reset | Start new episode |
| POST | /step | Take action step |
| GET | /state | Get current state |
| GET | /grade | Grade episode (0.0-1.0 score) |
| GET | /tasks | Available tasks |
| GET | /metrics | System metrics |
| GET | /replay | Episode history |
| GET | /feeder | Aggregate fleet state |
| POST | /coordinate | Set price multipliers |
| POST | /simulate | World model prediction |

---

## Project Structure

```
gridmind-rl/
├── main.go                    # HTTP server & OpenEnv API
├── inference.py              # Agent entry point (LLM + heuristic)
├── openenv.yaml              # OpenEnv spec
├── Dockerfile                # Container build
├── env/
│   ├── environment.go        # Physics simulation
│   ├── models.go           # Data models
│   ├── rewards.go         # Reward computation
│   ├── tasks.go           # Task grading
│   └── faults.go         # Fault injection
├── scripts/
│   ├── train_unsloth.py   # GRPO training
│   ├── plot_results.py   # Training curve visualizer
│   ├── multi_building_demo.py  # Fleet AI demo
│   └── run_baseline.sh   # Baseline scorer
├── dashboard/
│   ├── server.py         # Web server (port 7861)
│   └── static/           # Frontend assets
├── results/              # Training outputs (generated)
└── README.md
```

---

## Links

- 🤗 HuggingFace Space: [GridMind-RL](https://prajwal782007-gridmind.hf.space)
- 📝 Blog Post: [GridMind-RL: Training LLMs on Industrial Energy Management](https://huggingface.co/blog/gridmind-rl)
- 🎥 Demo Video: [YouTube Walkthrough](https://www.youtube.com/watch?v=dummy)
- 📊 Training Run: [gridmind_grpo_colab.ipynb](https://colab.research.google.com/)
- GitHub: [https://github.com/LO-Kyu/gridmind](https://github.com/LO-Kyu/gridmind)

---

## License

MIT License. See [LICENSE](LICENSE) file.

---

**Questions?** Open an issue on GitHub.