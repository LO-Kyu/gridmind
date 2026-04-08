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

# GridMind-RL

**Industrial building energy management reinforcement learning environment**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.org/)
[![Go 1.21](https://img.shields.io/badge/Go-1.21-00ADD8)](https://golang.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Live Demo

| | URL |
|--|-----|
| **Environment API** | https://lo-kyu-gridmind.hf.space |
| **Live Dashboard** | https://lo-kyu-gridmind.hf.space/dashboard |

**Quick test:**
```bash
curl https://lo-kyu-gridmind.hf.space/health
curl https://lo-kyu-gridmind.hf.space/tasks
```

## Overview

GridMind-RL is a reinforcement learning environment for training and evaluating intelligent control policies in industrial building energy management. The environment simulates realistic HVAC control, thermal storage management, batch job scheduling, and demand response scenarios under stochastic electricity pricing and grid stress events.

**Key challenges solved by the environment:**
- **Cost minimization**: Navigate complex electricity pricing curves across 24-hour periods
- **Comfort maintenance**: Keep indoor temperature within comfort bounds while optimizing cost
- **Grid responsiveness**: Respond to grid stress signals with intelligent load shedding
- **Carbon reduction**: Minimize grid carbon intensity through demand response
- **Batch scheduling**: Schedule compute-intensive batch jobs optimally
- **Storage management**: Efficiently use thermal storage for load shifting

This environment is ideal for training deep reinforcement learning agents, testing heuristic policies, and benchmarking control algorithms. It provides dense reward signals enabling efficient policy learning.

---

## Architecture

GridMind-RL consists of three tightly integrated components:

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
- **Dense rewards**: 7-component reward for effective learning

---

## Environment Specification

### Observation Space (11 fields)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `indoor_temperature` | float | [15-27] °C | Building indoor temperature |
| `thermal_storage_level` | float | [0-1] | Thermal storage charge (0=empty, 1=full) |
| `process_demand` | float | [5-50] kW | Baseline demand |
| `current_price` | float | [0.03-0.25] $/kWh | Electricity price |
| `grid_stress_signal` | float | [0-1] | Grid stress (>0.7 = critical) |
| `carbon_intensity` | float | [50-800] gCO2/kWh | Grid carbon intensity |
| `hour_of_day` | int | [0-23] | Time of day |
| `batch_queue` | list | Up to 10 items | Batch job deadlines |
| `cumulative_cost` | float | [0-1000] $ | Total cost this episode |
| `step` | int | [0-95] | Current step (96 steps = 24 hours) |
| `building_id` | int | {0} | Building identifier |

### Action Space (5 fields)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hvac_power_level` | float | [0-1] | HVAC power (0=off, 1=max) |
| `thermal_charge_rate` | float | [-1 to 1] | Storage charge/discharge rate |
| `batch_job_slot` | int | [0 to 4] | Batch job scheduling slot |
| `load_shed_fraction` | float | [0 to 0.5] | Load shedding fraction |
| `building_id` | int | {0} | Building identifier |

### Reward System

#### Raw Reward Components (7 Components)

| Component | Description |
|-----------|-------------|
| **Cost Savings** | Negative cost per energy consumed |
| **Temperature Constraint** | Penalty if T outside [19-23]°C |
| **Grid Response** | Bonus for load shedding during stress |
| **Deadline Penalty** | Penalty for missed batch deadlines |
| **Efficiency Bonus** | Bonus for off-peak charging |
| **Stability Penalty** | Penalty for rapid control changes |
| **Carbon Reward** | Bonus for low-carbon periods |

#### Reward Normalization

The inference script normalizes rewards to a standardized range for consistent scoring:

| Metric | Range | Description |
|--------|-------|-------------|
| **Per-step reward** | [0.10, 0.90] | Worst action → 0.10, Best action → 0.90 |
| **Episode score** | (0.01, 0.99) | Clamped to avoid exact 0.0 or 1.0 |

**Normalization formula:**
```
normalized_reward = ((raw_reward - raw_min) / (raw_max - raw_min)) * 0.80 + 0.10
episode_score = clamp(mean(normalized_rewards), 0.01, 0.99)
```

This ensures:
- Scores are strictly between 0 and 1 (never exactly 0.0 or 1.0)
- Relative performance matters more than absolute values
- Fair comparison across different episodes and tasks

---

## Output Format

The inference script emits machine-parsed stdout for judge evaluation:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

**Rules:**
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after `env.close()`, always emitted (even on exception)
- `reward` and `rewards` are formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the raw `last_action_error` string, or `null` if none

**Example:**
```
[START] task=gridmind-task-1 env=gridmind model=Qwen2.5-7B-Instruct
[STEP] step=1 action={"hvac_power_level":0.7,"thermal_charge_rate":0.5,...} reward=0.50 done=false error=null
[STEP] step=2 action={"hvac_power_level":0.5,"thermal_charge_rate":-0.3,...} reward=0.83 done=false error=null
[STEP] step=96 action={"hvac_power_level":0.3,"thermal_charge_rate":0.0,...} reward=0.90 done=true error=null
[END] success=true steps=96 score=0.683 rewards=0.50,0.55,0.83,...,0.90
```

---

## Tasks

| Task | Difficulty | Objective | Baseline Score |
|------|-----------|-----------|----------------|
| Task 1 | Easy | Minimize cost only | **0.708** |
| Task 2 | Medium | Minimize cost + maintain comfort | **0.633** |
| Task 3 | Hard | Full demand response + scheduling | **0.598** |

**Task 1 (Easy)**: Cost minimization, no constraints  
**Task 2 (Medium)**: Cost + temperature comfort (19-23°C)  
**Task 3 (Hard)**: Cost + comfort + grid response + batch scheduling + carbon

---

## Quickstart

### Docker (Recommended)

```bash
docker build -t gridmind-rl .
docker run -p 7860:7860 -p 7861:7861 gridmind-rl
```

### Local Development

**Terminal 1: Start Go server**
```bash
go run main.go
```

**Terminal 2: Run agent**
```bash
# Copy and configure .env file
cp .env.example .env
# Edit .env with your API keys

# Heuristic policy (no LLM, fastest)
python inference.py --fast-mode --episodes 1

# LLM agent (default: reuses action for 8 steps)
python inference.py --episodes 1

# LLM agent (custom reuse interval)
python inference.py --llm-every 4 --episodes 1
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | **Yes** | — | Hugging Face / LLM API token |
| `API_BASE_URL` | No | `https://api-inference.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-7B-Instruct` | Model identifier |
| `ENV_URL` | No | `http://localhost:7860` | Environment server URL |

**Example `.env` file:**
```bash
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

---

## API Reference

All endpoints on port 7860 (OpenEnv standard).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/ping` | Liveness probe |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Take action step |
| `GET` | `/state` | Get current state |
| `GET` | `/grade` | Grade episode (0.0-1.0 score) |
| `GET` | `/tasks` | Available tasks |
| `GET` | `/metrics` | System metrics |
| `GET` | `/replay` | Episode history |

---

## Baseline Performance

Reference heuristic policy scores (rule-based, deterministic):

| Task | Score | Policy |
|------|-------|--------|
| Task 1 | 0.708 | Simple load-shifting heuristic |
| Task 2 | 0.633 | Temperature-aware heuristic |
| Task 3 | 0.598 | Full demand response heuristic |

LLM and RL agents are expected to exceed these scores.

---

## Project Structure

```
gridmind-rl/
+-- main.go                    # HTTP server & OpenEnv API
+-- inference.py               # Agent entry point
+-- openenv.yaml               # OpenEnv spec
+-- Dockerfile                 # Container build
+-- env/
    +-- environment.go         # Physics simulation
    +-- models.go              # Data models
    +-- rewards.go             # Reward computation
    +-- tasks.go               # Task grading
+-- python/
    +-- inference.py           # LLM agent
    +-- models.py              # Pydantic models
    +-- requirements.txt
+-- dashboard/
    +-- server.py              # Web server (port 7861)
    +-- static/                # Frontend assets
+-- data/
    +-- price_curves.json      # Price data
    +-- generate_prices.py     # Price generator
+-- tests/
    +-- test_graders.py        # Python tests
    +-- environment_test.go    # Go tests
+-- baseline_scores.json       # Reference scores
+-- .env.example               # Environment template
+-- LICENSE                    # MIT License
```

---

## Development

### Running Tests

```bash
# Go tests
go test ./tests/... -v

# Python tests (requires server running on 7860)
pytest tests/test_graders.py -v
```

### Rebuilding Price Data

```bash
python data/generate_prices.py
```

---

## License

MIT License. See [LICENSE](LICENSE) file.

---

**Questions?** Open an issue on GitHub.
