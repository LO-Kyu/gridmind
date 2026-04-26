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

Industrial buildings consume ~40% of global electricity yet rely on naive "always-on" HVAC policies. LLMs can reason about pricing curves, fault alerts, and natural language objectives—but no environment trains them for this. GridMind-RL simulates a full 24-hour building energy system with stochastic electricity prices, equipment faults, and instruction cards, creating a genuinely challenging domain where learned policies translate to real operational value.

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

## Environment

| | Description |
|---|-------------|
| **Observation** | 13 fields: temperature, storage, price, stress, carbon, faults, HVAC efficiency, process demand, batch queue, price forecast |
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
| `task_satisfaction` | 0.50* | Task 4 only — weighted per the episode's instruction card |
| `fault_mitigation` | dynamic | Emergency response — computed based on fault type and response |

> *Task 4 instruction reward weight comes from the sampled instruction card, not a fixed value.

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| indoor_temperature | float | °C |
| thermal_storage_level | float | 0-1 (0=empty, 1=full) |
| process_demand | float | kW current industrial power demand |
| current_price | float | $/kWh |
| grid_stress_signal | float | 0-1 (>0.7 = critical) |
| carbon_intensity | float | gCO2/kWh |
| hour_of_day | int | 0-23 |
| batch_queue | int[] | pending job deadline slots |
| cumulative_cost | float | $ total incurred this episode |
| hvac_efficiency | float | 1.0 → degrades to 0.5 over episode |
| active_faults | string[] | Active fault alarm strings |
| instruction_card | object | Task 4 objective only |
| price_forecast | float[] | 4-step upcoming price preview |

### Action Fields

| Field | Type | Range |
|-------|------|-------|
| hvac_power_level | float | 0.0-1.0 |
| thermal_charge_rate | float | -1.0 to 1.0 |
| batch_job_slot | int | 0-4 |
| load_shed_fraction | float | 0.0-0.5 |

---

## Core Capabilities

### Multi-Agent Coordination
A single oversight LLM coordinates multiple buildings through price signals. The coordinator reads `/feeder` to see fleet-wide demand, then sets per-building price multipliers via `/coordinate` to orchestrate behavior.

### Long-Horizon Instruction Following
Task 4 presents a natural language objective card like "Keep total energy cost under $2.50 while maintaining 19-23°C". Agents must plan across all 96 steps—not greedy per-step control.

These two capabilities map directly to Theme 1 and Theme 3 of the OpenEnv Hackathon.

---

## Results

### What the Agent Learns

A naive heuristic runs HVAC at fixed levels based on time-of-day. After GRPO training on GridMind-RL, the agent learns to charge thermal storage during off-peak hours (4¢/kWh) and discharge during peak (32¢/kWh), voluntarily shed load during grid stress signals above 0.7, and adjust HVAC intensity as efficiency degrades over the episode. None of these behaviors are hardcoded — the agent discovers them through the reward signal alone.

| Policy | Task 1 | Task 2 | Task 3 | Task 4 |
|--------|--------|--------|--------|--------|
| Heuristic Baseline | 0.494 | 0.471 | 0.748 | 0.478 |
| Zero-shot LLM | 0.715 | 0.645 | 0.610 | 0.582 |
| GRPO Fine-tuned LLM | — | — | — | — |

> *GRPO fine-tuned scores updating after full training run on T4 GPU.
> Training plots below show live progress from the actual run.*

![Reward Curve](curves/train%202/reward_curve.png)
*Reward vs training step. Blue = per-step reward, red dashed = smoothed average.*

![Loss Curve](curves/train%202/loss_curve.png)
*Training loss decreasing over steps — confirms the model is updating.*

![Baseline Comparison](curves/train%202/baseline_comparison.png)
*Grade scores per task: heuristic baseline vs GRPO-trained LLM.*

> Scores are episode grade scores (0.0–1.0, clamped open interval). Heuristic = fixed policy with no learning. Zero-shot = Qwen2.5-1.5B-Instruct prompted with task description, no fine-tuning, evaluated over 1 episode per task. Fine-tuned = GRPO-trained on GridMind-RL environment.

> 🔄 **Live update:** GRPO fine-tuned scores will be filled in here immediately
> after the final training run completes on the T4 GPU.

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
| GET | /metrics | Prometheus metrics |
| GET | /replay | Episode history |
| GET | /feeder | Aggregate fleet state |
| POST | /coordinate | Set price multipliers |
| POST | /simulate | World model prediction |
| POST | /coordinator/reset | Reset multi-building episode |
| POST | /coordinator/step | Step with per-building actions |
| GET | /info | OpenEnv metadata |
| GET | /ws | WebSocket endpoint |

---

## Project Structure

```
gridmind-rl/
├── main.go                    # HTTP server & OpenEnv API
├── inference.py              # Agent entry point (LLM + heuristic)
├── openenv.yaml              # OpenEnv spec
├── Dockerfile                # Container build
├── HF_BLOG_POST.md           # Blog write-up
├── baseline_scores.json      # Heuristic baseline scores
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
│   └── gridmind_grpo_colab.ipynb  # Colab training notebook
├── server/
│   └── app.py            # Python fallback server
├── dashboard/
│   ├── server.py         # Web server (port 7861)
│   └── static/           # Frontend assets
├── curves/               # Training curves (train N/)
│   └── train N/         # Per-run plots
├── results/              # Training outputs (generated)
└── README.md
```

---

## Links

- 🤗 HuggingFace Space: [GridMind-RL](https://prajwal782007-gridmind.hf.space)
- 📓 Training Notebook: [gridmind_grpo_colab.ipynb](https://colab.research.google.com/github/LO-Kyu/gridmind/blob/main/scripts/gridmind_grpo_colab.ipynb)
- 📝 Blog Post: [Read the write-up](./HF_BLOG_POST.md)
- 🐙 GitHub: [Code Repository](https://github.com/LO-Kyu/gridmind)

---

## License

MIT License. See [LICENSE](LICENSE) file.

---

**Questions?** Open an issue on GitHub.