# ⚡ GridMind-RL

**A real-world RL environment for building energy management** — control HVAC systems, thermal storage, batch job scheduling, and demand response under stochastic electricity prices and grid stress events.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification. Containerized. Ready for Hugging Face Spaces.

---

## 🎯 Why GridMind-RL?

Optimizing building energy use is a **real problem** that utilities, building managers, and industrial operators face every day. An agent must balance:

- **Cost** — buy electricity when it's cheap, avoid peak pricing
- **Comfort** — keep indoor temperature within comfortable bounds
- **Grid compliance** — shed load when the grid signals demand-response events
- **Scheduling** — complete batch processing jobs before their deadlines
- **Carbon** — minimize carbon emissions by timing consumption to clean-grid periods

This isn't a toy or a game. It's a simulation of decisions that **humans actually make** in industrial energy management, packaged as an RL environment where agents can learn to do it better.

---

## 📐 Observation Space

Each timestep (15 minutes of simulated time), the agent receives:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `indoor_temperature` | float | 10–40 °C | Current building temperature |
| `thermal_storage_level` | float | 0.0–1.0 | Thermal tank fill level (0=empty, 1=full) |
| `process_demand` | float | ≥ 0 kW | Current industrial power demand |
| `current_price` | float | > 0 $/kWh | Real-time electricity price |
| `grid_stress_signal` | float | 0.0–1.0 | Utility demand-response urgency (>0.7 = critical) |
| `carbon_intensity` | float | ≥ 0 gCO₂/kWh | Grid carbon intensity |
| `hour_of_day` | int | 0–23 | Current hour |
| `batch_queue` | int[] | — | Deadline slots of pending batch jobs |
| `cumulative_cost` | float | ≥ 0 $ | Total energy cost so far this episode |
| `step` | int | 0–95 | Current timestep (96 steps = 24 hours) |
| `building_id` | int | 0+ | Building index in multi-building mode |

## 🕹️ Action Space

Each timestep, the agent sends:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hvac_power_level` | float | 0.0–1.0 | Fraction of max HVAC power (0=off, 1=full) |
| `thermal_charge_rate` | float | -1.0–1.0 | Charge (+) or discharge (-) thermal storage |
| `batch_job_slot` | int | 0–4 | Schedule next batch job: 0=now, 1–4=defer |
| `load_shed_fraction` | float | 0.0–0.5 | Fraction of non-critical load to shed |
| `building_id` | int | 0+ | Which building this action targets |

## 💰 Reward Structure

The environment provides a **dense, multi-component reward** every step — not just a binary win/lose at the end. Each step returns a scalar `reward` (the sum) plus a detailed `reward_components` breakdown:

| Component | Key | Description |
|-----------|-----|-------------|
| Cost Savings | `cost_savings` | Rewards reducing energy spend vs baseline |
| Temperature | `temp_constraint` | Gaussian bonus near setpoint, penalty outside bounds |
| Grid Response | `grid_response` | Bonus for shedding load during grid stress |
| Efficiency | `efficiency_bonus` | Thermal storage arbitrage + balanced usage |
| Stability | `stability_penalty` | Rewards smooth control, penalizes oscillation |
| Deadlines | `deadline_penalty` | Penalty for missed batch jobs |
| Carbon | `carbon_reward` | Bonus for low-carbon operation |

---

## 📋 Tasks (3 difficulty levels)

Each task defines a concrete objective with a **deterministic programmatic grader** that scores performance from **0.0 to 1.0**.

| ID | Difficulty | Name | What the Agent Must Do | Grader Weights |
|----|:----------:|------|------------------------|----------------|
| 1 | 🟢 Easy | **Cost Minimization** | Minimize total energy cost over 24 hours. No temperature or scheduling constraints. | cost: 100% |
| 2 | 🟡 Medium | **Constrained Temperature** | Minimize cost **and** keep temperature within 19–23°C at all times. | cost: 60%, temperature: 40% |
| 3 | 🔴 Hard | **Full Demand Response** | Minimize cost, maintain temperature, respond to grid stress, complete batch jobs on time, minimize carbon. | cost: 28%, temperature: 20%, grid: 20%, batch: 12%, carbon: 20% |

**Graders are deterministic**: given the same seed, the same actions always produce the same score.

---

## 🚀 Getting Started (Step by Step)

### Prerequisites

- **Docker** — [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Python 3.9+** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)

### Step 1: Clone the Repository

```bash
git clone https://github.com/LO-Kyu/gridmind.git
cd gridmind
```

### Step 2: Build and Start the Environment Server

```bash
docker build -t gridmind-rl .
docker run --rm -d -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl
```

This starts the GridMind-RL environment server on port **7860**. Verify it's running:

```bash
# Linux/macOS
curl http://localhost:7860/health

# Windows (PowerShell)
Invoke-RestMethod -Uri http://localhost:7860/health
```

You should see: `{"status":"ok","version":"1.0.0"}`

### Step 3: Install Python Dependencies

Open a **new terminal** (keep Docker running) and install:

```bash
pip install -r python/requirements.txt
```

### Step 4: Get a Free API Key

The inference script uses an LLM to make decisions. You need a **free** API key:

1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign in with Google or GitHub (free)
3. Click **"Create Key"** and copy it

### Step 5: Configure Your API Key

Open the `.env` file in the project root and paste your key:

```env
API_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=meta-llama/llama-3.1-8b-instruct:free
OPENAI_API_KEY=sk-or-v1-paste-your-actual-key-here
ENV_URL=http://localhost:7860
```

> **Note:** The model `meta-llama/llama-3.1-8b-instruct:free` is **completely free** on OpenRouter. No credit card needed.

### Step 6: Run the Baseline Inference

```bash
# Run LLM agent on all 3 tasks
python inference.py --episodes 1

# Or run without LLM (fast heuristic mode — no API key needed)
python inference.py --fast-mode --episodes 1
```

The script will:
1. Connect to the environment server
2. Run the agent on Task 1 (easy), Task 2 (medium), Task 3 (hard)
3. Print `[START]`, `[STEP1]`...`[STEP96]`, `[END]` for each episode
4. Save results to `baseline_scores.json`

### Step 7: Stop the Server (When Done)

```bash
docker stop gridmind
```

---

## 📊 Baseline Scores

Produced by running `python inference.py --fast-mode --episodes 1` (heuristic policy):

| Task | Difficulty | Score | Details |
|------|:----------:|:-----:|---------|
| 1 — Cost Minimization | 🟢 Easy | **0.7063** | cost: 0.706 |
| 2 — Temperature Management | 🟡 Medium | **0.6333** | cost: 0.701, temperature: 0.531 |
| 3 — Full Demand Response | 🔴 Hard | **0.5966** | cost: 0.670, temp: 0.573, grid: 0.214, batch: 1.000, carbon: 0.657 |
| **Overall Average** | | **0.6454** | |

Scores are in the **0.0–1.0** range. Higher is better.

---

## 🔌 HTTP API Reference

Base URL: `http://localhost:7860`

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check → `{"status":"ok","version":"1.0.0"}` |
| `GET` | `/ping` | Lightweight liveness check |
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": 1, "seed": 42}` |
| `POST` | `/step` | Take one action. Body: action JSON (see Action Space above) |
| `GET` | `/state` | Full environment state snapshot |
| `GET` | `/grade` | Episode score (0.0–1.0) with sub-scores |
| `GET` | `/replay` | Full step-by-step replay of the episode |
| `GET` | `/tasks` | List all task definitions and grader weights |
| `GET` | `/metrics` | Prometheus-format operational metrics |

### Example API Calls

```bash
# Reset to Task 1 (easy) with seed 42
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'

# Take one step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"hvac_power_level": 0.5, "thermal_charge_rate": 0.1, "batch_job_slot": 1, "load_shed_fraction": 0.0}'

# Check score after episode
curl http://localhost:7860/grade
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  inference.py (LLM Agent or Heuristic)                          │
│       │                                                         │
│       │ HTTP: POST /reset, /step  ·  GET /grade, /state         │
│       ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Docker Container                                         │  │
│  │                                                           │  │
│  │  ┌─────────────────────┐   ┌───────────────────────────┐ │  │
│  │  │  Go Environment     │   │  Python Dashboard         │ │  │
│  │  │  Server (:7860)     │   │  FastAPI + UI (:7861)     │ │  │
│  │  │                     │   │                           │ │  │
│  │  │  • Physics engine   │   │  • Proxies /api → :7860  │ │  │
│  │  │  • Reward function  │◄──│  • Real-time charts      │ │  │
│  │  │  • Task graders     │   │  • State visualization   │ │  │
│  │  └─────────────────────┘   └───────────────────────────┘ │  │
│  │                                                           │  │
│  │  Isolated · Reproducible · Non-root user                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
gridmind/
├── inference.py              ← Hackathon entrypoint (root)
├── openenv.yaml              ← OpenEnv spec manifest
├── Dockerfile                ← Multi-stage build (Go + Python)
├── .env                      ← API credentials (git-ignored)
├── baseline_scores.json      ← Produced by inference.py
│
├── main.go                   ← HTTP server (routes, middleware, metrics)
├── env/                      ← Core environment logic (Go)
│   ├── environment.go        ← Simulation: physics, thermal dynamics
│   ├── models.go             ← All data types (Observation, Action, etc.)
│   ├── rewards.go            ← 7-component dense reward function
│   └── tasks.go              ← 3 task definitions + deterministic graders
│
├── python/                   ← Python support layer
│   ├── inference.py          ← Full LLM agent + heuristic fallback
│   ├── models.py             ← Typed Pydantic models (mirrors Go structs)
│   ├── validate.py           ← OpenEnv spec validation suite
│   └── requirements.txt      ← Python dependencies
│
├── tests/                    ← Automated tests
│   ├── environment_test.go   ← Go unit tests (determinism, bounds, etc.)
│   └── test_graders.py       ← Python grader tests (pytest)
│
└── dashboard/                ← Optional web dashboard
    ├── server.py             ← FastAPI server
    └── static/               ← Frontend assets
```

---

## 🐳 Docker

| Action | Command |
|--------|---------|
| **Build** | `docker build -t gridmind-rl .` |
| **Run (foreground)** | `docker run --rm -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl` |
| **Run (background)** | `docker run --rm -d -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl` |
| **Stop** | `docker stop gridmind` |
| **Run inference inside container** | `docker exec -it gridmind python /app/inference.py --fast-mode` |

The Dockerfile uses a **multi-stage build**:
1. **Stage 1** — Go 1.21 Alpine: compiles the environment server binary
2. **Stage 2** — Python 3.11 slim: runs the Go binary + Python dashboard via Supervisor

---

## ☁️ Hugging Face Space Deployment

### 1. Create a New Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):
- **SDK:** Docker
- **Hardware:** CPU Basic (2 vCPU, 16 GB — free tier)

### 2. Push to HF

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/gridmind-rl
git push hf main
```

### 3. Verify

```bash
curl https://YOUR_USERNAME-gridmind-rl.hf.space/health
# → {"status":"ok","version":"1.0.0"}

curl -X POST https://YOUR_USERNAME-gridmind-rl.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":1,"seed":42}'
```

> **Note:** HF Spaces exposes port **7860** publicly. The dashboard (7861) is for local development only.

---

## 🧪 Testing

### Run Go Unit Tests

```bash
cd gridmind
go test ./tests/ -v
```

### Run Python Grader Tests (requires server running)

```bash
pytest tests/test_graders.py -v
```

### Run Full OpenEnv Validation

```bash
python python/validate.py --env-url http://localhost:7860
```

---

## 📝 Inference Script Reference

The `inference.py` script at the project root is the **hackathon entrypoint**.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API endpoint |
| `MODEL_NAME` | `meta-llama/llama-3.1-8b-instruct:free` | Model to use |
| `OPENAI_API_KEY` | — | API key (any OpenAI-compatible provider) |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

### Command-Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1 | Episodes per task (tasks 1–3 run in sequence) |
| `--fast-mode` | off | Use heuristic policy only (no LLM, fully reproducible) |
| `--llm-every N` | 4 | Reuse each LLM action for N steps (reduces API calls) |
| `--max-steps N` | 96 | Stop early after N steps |
| `--env-url URL` | from env | Override environment URL |
| `--output FILE` | `baseline_scores.json` | Output results file |
| `--verbose` | off | Print detailed step logs |

### Stdout Log Format

Each episode emits structured markers for automated evaluation:

```
[START]
[STEP1]
[STEP2]
...
[STEP96]
[END]
```

---

## 📎 OpenEnv Spec Compliance

| Requirement | Status |
|-------------|--------|
| `openenv.yaml` with metadata | ✅ |
| Typed Pydantic models (Observation, Action, Reward) | ✅ |
| `step(action)` → observation, reward, done, info | ✅ |
| `reset()` → initial observation | ✅ |
| `state()` → current state | ✅ |
| 3 tasks with programmatic graders (0.0–1.0) | ✅ |
| Dense reward function (not binary) | ✅ |
| Baseline inference using OpenAI client | ✅ |
| Working Dockerfile | ✅ |
| Deterministic with seed | ✅ |
| Exploit detection | ✅ |

---

## 📄 License

See `LICENSE` in the repository.
