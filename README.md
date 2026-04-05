# GridMind-RL

**Industrial building energy management reinforcement learning environment**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.org/)
[![Go 1.21](https://img.shields.io/badge/Go-1.21-00ADD8)](https://golang.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

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
    ?? HTTP POST /step, /reset, /grade
    ?
Go Environment Server (main.go) — Port 7860
    ?
Physics Engine (env/environment.go) + Rewards (env/rewards.go) + Tasks (env/tasks.go)
    ?
Web Dashboard (dashboard/server.py) — Port 7861
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
| `thermal_charge_rate` | float | [-1-1] | Storage charge/discharge rate |
| `batch_job_slot` | int | [0-4] | Batch job scheduling slot |
| `load_shed_fraction` | float | [0-0.5] | Load shedding fraction |
| `building_id` | int | {0} | Building identifier |

### Reward Function (7 Components)

| Component | Description |
|-----------|-------------|
| **Cost Savings** | Negative cost per energy consumed |
| **Temperature Constraint** | Penalty if T outside [19-23]°C |
| **Grid Response** | Bonus for load shedding during stress |
| **Deadline Penalty** | Penalty for missed batch deadlines |
| **Efficiency Bonus** | Bonus for off-peak charging |
| **Stability Penalty** | Penalty for rapid control changes |
| **Carbon Reward** | Bonus for low-carbon periods |

---

## Tasks

| Task | Difficulty | Objective | Baseline Score |
|------|-----------|-----------|-----------------|
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
export HF_TOKEN="your_api_key"
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="meta-llama/llama-3.3-70b-instruct:free"

# Heuristic policy (no LLM)
python inference.py --fast-mode --episodes 1

# LLM agent
python inference.py --episodes 1
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | LLM API key |
| `API_BASE_URL` | No | `https://openrouter.ai/api/v1` | LLM endpoint |
| `MODEL_NAME` | No | `meta-llama/llama-3.3-70b-instruct:free` | Model ID |
| `ENV_URL` | No | `http://localhost:7860` | Environment server URL |
| `OPENAI_API_KEY` | No | — | Alternative to HF_TOKEN |

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
¦   +-- environment.go         # Physics simulation
¦   +-- models.go              # Data models
¦   +-- rewards.go             # Reward computation
¦   +-- tasks.go               # Task grading
+-- python/
¦   +-- inference.py           # LLM agent
¦   +-- models.py              # Pydantic models
¦   +-- requirements.txt
+-- dashboard/
¦   +-- server.py              # Web server (port 7861)
¦   +-- static/                # Frontend assets
+-- data/
¦   +-- price_curves.json      # Price data
¦   +-- generate_prices.py     # Price generator
+-- tests/
¦   +-- test_graders.py        # Python tests
¦   +-- environment_test.go    # Go tests
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
