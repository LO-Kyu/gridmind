# 🏢 GridMind-RL — Energy Management Reinforcement Learning Environment

**A real-world RL environment for intelligent building energy optimization.** Control HVAC systems, thermal storage, batch job scheduling, and demand-response under stochastic electricity prices and grid stress events.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification. Containerized. Ready for Hugging Face Spaces deployment.

---

## 📖 Overview & Motivation

Building energy management is a **real-world optimization problem** facing utilities, facility operators, and industrial sites globally. Traditional rule-based controls waste billions in energy costs and miss opportunities for grid participation.

**GridMind-RL** simulates decisions that facility operators must make daily:

- **Cost Optimization** — Buy electricity when prices are low, avoid peak surcharges
- **Comfort & Safety** — Maintain indoor temperature within acceptable ranges while managing thermal inertia
- **Grid Participation** — Respond to demand-response signals and grid stress events  
- **Batch Scheduling** — Coordinate industrial process timings to meet deadlines and minimize energy cost
- **Carbon Minimization** — Shift consumption to periods when grid carbon intensity is low

**Why this matters:** An RL agent trained in this environment can learn strategies that would be difficult or impossible for humans to hand-craft. The combination of continuous control (HVAC power, thermal storage), discrete decisions (batch scheduling), and multiple simultaneous objectives (cost, comfort, grid, deadlines, carbon) creates a realistic, challenging benchmark.

**Episode Length:** 96 steps = 24 hours at 15-minute resolution. A complete episode requires strategic decision-making across a full day-night cycle.

---

## � Observation Space

At each timestep, the environment provides the following observations. **Episode length: 96 steps** (15-minute intervals = 24 hours).

| Field | Data Type | Range / Values | Description |
|-------|-----------|-----------------|-------------|
| `indoor_temperature` | float | 10–40 °C | Current building interior temperature |
| `thermal_storage_level` | float | 0.0–1.0 | Thermal tank charge state (0 = empty, 1 = full) |
| `process_demand` | float | ≥ 0 kW | Current industrial batch process power draw |
| `current_price` | float | > 0 $/kWh | Real-time spot electricity price |
| `grid_stress_signal` | float | 0.0–1.0 | Utility demand-response urgency (0.7+ = critical) |
| `carbon_intensity` | float | ≥ 0 gCO₂/kWh | Current grid carbon intensity |
| `hour_of_day` | int | 0–23 | Time-of-day context |
| `batch_queue` | int array | — | Pending batch jobs with deadline slots |
| `cumulative_cost` | float | ≥ 0 $ | Energy cost accumulated in current episode so far |
| `step` | int | 0–95 | Current timestep (96 total = 24 hours) |
| `building_id` | int | 0+ | Building identifier (for multi-building scenarios) |

**Observation Properties:**
- Observations are **deterministic** given the seed — same seed produces identical sequences
- All fields are **normalized or bounded** for stable learning
- Prices follow realistic time-of-use patterns; carbon intensity varies with grid mix
- Batch queue starts empty; jobs appear stochastically based on the task/seed

---

## 🎮 Action Space

At each step, the agent sends an action controlling four independent subsystems:

| Field | Data Type | Range | Description |
|-------|-----------|-------|-------------|
| `hvac_power_level` | float | 0.0–1.0 | HVAC system power (0 = off, 1 = full) |
| `thermal_charge_rate` | float | -1.0–1.0 | Thermal storage control (+charge, -discharge) |
| `batch_job_slot` | int | 0–4 | Schedule next batch job: 0=immediate, 1–4=defer |
| `load_shed_fraction` | float | 0.0–0.5 | Non-critical load reduction (0–50%) for demand-response |
| `building_id` | int | 0+ | Building identifier (routing) |

**Action Space Properties:**
- **Continuous** (HVAC, thermal charging, load shedding) + **discrete** (batch scheduling) → hybrid control
- Actions are applied every 15-minute step
- Load shedding is capped at 50% to ensure safety/habitability
- Batch scheduling decisions affect energy cost and deadline compliance

---

## 💡 Reward Function

The environment provides **dense rewards every step** (not sparse, not binary). Each step returns:
- A scalar reward (sum of components)
- A dictionary of 7 weighted sub-components for transparency

| Component | Purpose | Possible Values |
|-----------|---------|-----------------|
| **cost_savings** | Minimize energy bill | Negative (cost increases) to positive (savings vs baseline) |
| **temp_constraint** | Maintain comfort | Gaussian bonus near 21°C, penalty outside 19–23°C bounds |
| **grid_response** | Shift load during stress | Bonus proportional to shed fraction when grid signal > 0.7 |
| **efficiency_bonus** | Exploit thermal storage | Reward charge/discharge timing and thermal arbitrage |
| **stability_penalty** | Smooth control | Small penalty for rapid oscillations in HVAC/storage |
| **deadline_penalty** | Meet job deadlines | Large penalty if batch job finishes after deadline |
| **carbon_reward** | Low-carbon consumption | Bonus for consuming during low-carbon grid periods |

**Example Reward Calculation:**  
If an agent takes a well-timed action during high-price, high-stress period:
- Large positive `cost_savings` (avoided expensive hour)
- Positive `grid_response` (shed load successfully)
- Possible positive `carbon_reward` (if grid is clean)
- **Total step reward** = weighted sum of all components

This multi-objective reward structure encourages **learning tradeoffs** between cost, comfort, grid support, and carbon efficiency.

---

---

## 📋 Tasks & Difficulty Levels

Three independent tasks with **deterministic programmatic graders**. Scores range **0.0–1.0**; higher is better.

### Task 1 — Cost Minimization (🟢 Easy)

**Objective:** Minimize total energy cost in 24 hours with no other constraints.

**Difficulty Rationale:** Only one objective (cost) to optimize; temperature and grid constraints are relaxed.

**Grader Metrics:**
- **Cost score (100%)** — Compares total episode energy cost to a deterministic baseline. Higher savings → higher score.

**Baseline Score:** **0.7063**

---

### Task 2 — Constrained Temperature Control (🟡 Medium)

**Objective:** Minimize cost while maintaining indoor temperature between **19–23°C** throughout the episode.

**Difficulty Rationale:** Introduces a hard constraint (temperature bounds). Agent must use thermal storage strategically to meet both cost and comfort goals.

**Grader Metrics:**
- **Cost score (60%)** — Total energy cost vs baseline
- **Temperature score (40%)** — Fraction of steps within bounds (hard penalty for violations)

**Notes:** A naive agent might achieve low cost by disabling HVAC, but then temperatures drift out of bounds (0 score). Trade-off learning is required.

**Baseline Score:** **0.6333**

---

### Task 3 — Full Demand Response (🔴 Hard)

**Objective:** Minimize cost, maintain temperature, respond to grid events, complete batch jobs on time, and minimize carbon emissions. This is a **multi-objective constraint satisfaction** problem.

**Difficulty Rationale:** Most realistic. Agent must balance five competing objectives simultaneously; any single failure is costly.

**Grader Metrics:**
- **Cost score (28%)** — Energy cost
- **Temperature score (20%)** — Time within comfort bounds
- **Grid response score (20%)** — Load shed during demand-response events (signal > 0.7)
- **Batch deadline score (12%)** — Fraction of jobs completed before deadline
- **Carbon reward score (20%)** — Shift load to low-carbon periods

**Baseline Breakdown:**
- Cost: 0.670, Temperature: 0.573, Grid: 0.214, Batch: 1.000, Carbon: 0.657
- **Overall: 0.5966** 

**Challenge:** Grid response score (~0.21) shows that the baseline heuristic rarely sheds load opportunistically. Learning agents should discover that quick load shedding during high-price, high-stress periods yields significant cost savings.

**Grader Determinism:** Same seed always produces identical evaluations. Episodes are seeded internally; reproducible batches of evaluations can be generated for benchmark comparisons.

---

## 🚀 Setup & Usage

### Prerequisites

- **Docker** — [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)

### Quick Start (5 minutes)

#### 1. Clone the Repository

```bash
git clone https://github.com/LO-Kyu/gridmind-rl.git
cd gridmind-rl
```

#### 2. Build and Start the Environment Server

```bash
docker build -t gridmind-rl .
docker run --rm -d -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl
```

Verify the server is running:

```bash
# Check health endpoint
curl http://localhost:7860/health
# Expected: {"status":"ok","version":"1.0.0"}
```

#### 3. Install Python Dependencies

Open a **new terminal** and install:

```bash
pip install -r python/requirements.txt
```

#### 4. Run Inference (No LLM — Fast)

Run a fast, deterministic baseline using heuristic policy:

```bash
python inference.py --fast-mode --episodes 1
```

Expected output (sample):
```
[START] task=Cost_Minimization env=gridmind model=heuristic
[STEP1] step=1 action={...} reward=10.5 done=false
[STEP2] step=2 action={...} reward=12.3 done=false
...
[STEP96] step=96 action={...} reward=8.9 done=true
[END] success=true steps=96 rewards=[10.5, 12.3, ..., 8.9]
```

Results saved to: `baseline_scores.json`

#### 5. (Optional) Run with LLM

To use an LLM agent for decision-making:

1. Get a **free API key** from [openrouter.ai/keys](https://openrouter.ai/keys) (no credit card needed)
2. Create `.env` file (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```env
   HF_TOKEN=sk-or-v1-your-key-here
   # or
   OPENAI_API_KEY=sk-or-v1-your-key-here
   ```
4. Run with LLM:
   ```bash
   python inference.py --episodes 1
   ```

#### 6. Stop the Server (When Done)

```bash
docker stop gridmind
```

---

### Inference Script Reference

The `inference.py` script (project root) is the **hackathon submission entrypoint**.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required for submission) | API key for LLM provider or HF Spaces |
| `OPENAI_API_KEY` | (optional fallback) | Alternative OpenAI-compatible key |
| `API_BASE_URL` | `https://openrouter.ai/api/v1` | LLM endpoint URL |
| `MODEL_NAME` | `meta-llama/llama-3.3-70b-instruct:free` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Environment server address |

**Command-Line Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1 | Episodes per task (runs tasks 1, 2, 3 in sequence) |
| `--fast-mode` | off | Don't call LLM; use heuristic policy only (reproducible, no API calls) |
| `--llm-every N` | 4 | Reuse each LLM decision for N steps (reduces API calls) |
| `--max-steps N` | 96 | Stop episode early after N steps |
| `--env-url URL` | from env var | Override environment server URL |
| `--output FILE` | `baseline_scores.json` | Output results filename |
| `--verbose` | off | Print detailed logs for each step |

**Examples:**

```bash
# Run all 3 tasks with LLM (1 episode each)
python inference.py --episodes 1

# Reproduce baseline fast (no LLM)
python inference.py --fast-mode --episodes 1

# Only Task 2, heuristic, verbose output
python inference.py --fast-mode --episodes 1 --verbose

# Run 5 episodes per task with custom environment
python inference.py --episodes 5 --env-url http://my-server:7860
```

---

### HTTP API Reference

**Base URL:** `http://localhost:7860`

| Endpoint | Method | Purpose | Example Body |
|----------|--------|---------|---------------|
| `/health` | GET | Liveness check | — |
| `/ping` | GET | Lightweight ping | — |
| `/reset` | POST | Reset episode for a task | `{"task_id": 1, "seed": 42}` |
| `/step` | POST | Apply action, get next observation | `{"hvac_power_level": 0.5, "thermal_charge_rate": 0.1, ...}` |
| `/state` | GET | Current full state snapshot | — |
| `/grade` | GET | Episode score (0.0–1.0) with sub-scores | — |
| `/replay` | GET | Full step-by-step trajectory | — |
| `/tasks` | GET | Task definitions and grader weights | — |
| `/metrics` | GET | Prometheus-format metrics | — |

**Example Workflow:**

```bash
# 1. Reset to Task 1 with seed 42
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'

# 2. Get initial observation
curl http://localhost:7860/state

# 3. Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "hvac_power_level": 0.5,
    "thermal_charge_rate": 0.1,
    "batch_job_slot": 1,
    "load_shed_fraction": 0.0
  }'

# 4. Check final score after episode completes
curl http://localhost:7860/grade
```

---

## 📊 Baseline Performance Scores

The baseline is a **heuristic policy** (rule-based, no LLM) representing a reasonable but non-optimized control strategy. Your RL agent should aim to exceed these scores.

**Baseline Run:** `python inference.py --fast-mode --episodes 1`

### Summary Scores

| Task | Difficulty | Score | Model |
|------|:----------:|:-----:|-------|
| Task 1 — Cost Minimization | 🟢 Easy | **0.7063** | Heuristic |
| Task 2 — Temperature Control | 🟡 Medium | **0.6333** | Heuristic |
| Task 3 — Full Demand Response | 🔴 Hard | **0.5966** | Heuristic |
| **Overall Average** | — | **0.6454** | Heuristic |

### Detailed Breakdown

#### Task 1 Results
- **Task:** Cost minimization (96 hours × 15 min = 24 hours)
- **Score:** 0.7063  
- **Sub-score:** Cost = 0.706
- **Interpretation:** Heuristic achieves ~70% of optimal cost reduction vs baseline

#### Task 2 Results
- **Task:** Minimize cost while maintaining temperature 19–23°C
- **Score:** 0.6333
- **Sub-scores:**
  - Cost: 0.701
  - Temperature constraint: 0.531 (agent violated comfort bounds ~47% of the time)
- **Interpretation:** Temperature management is challenging for the heuristic. Tighter thermal control could improve this score significantly.

#### Task 3 Results (Most Interesting)
- **Task:** Multi-objective: cost, temperature, grid response, batch deadlines, carbon
- **Score:** 0.5966
- **Sub-scores:**
  - Cost: 0.670
  - Temperature: 0.573 (similar temperature control challenge as Task 2)
  - **Grid response: 0.214** ← Heuristic rarely participates in demand-response
  - Batch deadline: 1.000 (heuristic always completes jobs on time)
  - Carbon: 0.657

**Key Insight:** The heuristic's low grid response score (0.21) suggests that learned agents have significant room for improvement by:
1. Recognizing high-price + high-stress periods
2. Proactively shedding load to reduce cost
3. Using thermal storage to recover comfort afterward

This multi-objective setting is where RL agents typically exceed heuristic baselines.

### Reproducibility & Evaluation

- **Deterministic:** Baseline scores are **deterministic** — same seed always produces identical actions and rewards
- **Seeding:** Each task uses a fixed base seed (1100, 1200, 1300) for reproducible evaluation
- **Your Submissions:** Your agent will be evaluated on the same seed distribution; compare your scores directly to baseline

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

## ✅ OpenEnv Specification Compliance

GridMind-RL fully implements the OpenEnv specification for standardized RL environments. All components are present and tested:

| Requirement | Status | Notes |
|-------------|:------:|-------|
| Manifest (`openenv.yaml`) | ✅ | All metadata, schema definitions, and version info |
| Observation Schema | ✅ | 11-field object: temperature, storage, price, grid signal, carbon, hour, batch queue, cost, step, building_id |
| Action Schema | ✅ | 5-field object: HVAC, thermal rate, batch slot, load shed, building_id |
| HTTP Endpoints | ✅ | `/reset`, `/step`, `/state`, `/grade`, `/replay`, `/tasks`, `/health`, `/metrics` |
| Determinism | ✅ | Seeded episode generation; identical seeds produce identical trajectories |
| Typed Models | ✅ | Pydantic models (Python) mirror Go structs exactly |
| Dense Rewards | ✅ | 7-component reward breakdown every step |
| Graders | ✅ | 3 tasks with programmatic, deterministic graders (0.0–1.0 range) |
| Exploit Detection | ✅ | Built into grading pipeline to flag unrealistic scores |

---

## ❓ FAQ

**Q: Can I use a different model?**  
A: Yes. Set `MODEL_NAME` environment variable to any OpenAI-compatible model. The default (`meta-llama/llama-3.3-70b-instruct:free`) is free on OpenRouter with no credit card.

**Q: How do I avoid rate limiting?**  
A: (1) Use `--fast-mode` for local testing (no API calls), (2) Set `--llm-every 4` to reuse decisions, (3) Use a paid API tier for submission, or (4) Train & submit an offline policy.

**Q: Will my API key be exposed in submissions?**  
A: No. Store your API key in `.env` (git-ignored). On HF Spaces, set secrets via the Space settings UI; keys are never committed to the repo.

**Q: What's the difference between `HF_TOKEN` and `OPENAI_API_KEY`?**  
A: `HF_TOKEN` is used in HF Space deployments and external evaluations. `OPENAI_API_KEY` is a fallback for local development. The code tries `HF_TOKEN` first, then `OPENAI_API_KEY`. At least one must be set.

**Q: Can I submit an offline/trained policy?**  
A: Yes. Modify `python/inference.py` to use your trained agent instead of LLM calls. Ensure you still output the required `[START]`, `[STEP]`, `[END]` format.

**Q: What if my submission times out?**  
A: Each episode is 96 steps. The environment runs 3 episodes (one per task). Optimize for latency: reduce LLM calls (use `--llm-every`), use a faster model, or submit a heuristic/trained offline policy.

---

## 🎯 Submission Checklist

Before submitting, verify:

- [ ] Clone repo, build Docker, run `docker run -p 7860:7860 -p 7861:7861 gridmind-rl`
- [ ] Run `python inference.py --fast-mode --episodes 1` locally — should produce `baseline_scores.json`
- [ ] Check `[START]`, `[STEP]`, `[END]` markers in stdout
- [ ] Set `HF_TOKEN` or `OPENAI_API_KEY` in `.env` for LLM runs
- [ ] Test with LLM: `python inference.py --episodes 1`
- [ ] Verify Dockerfile builds without errors: `docker build -t gridmind-rl .`
- [ ] Create HF Space (Docker SDK, CPU Basic)
- [ ] Push repo to HF Space: `git push hf main`
- [ ] Set secrets in HF Space UI: `HF_TOKEN`, `API_BASE_URL` (optional), `MODEL_NAME` (optional)
- [ ] Verify Space is running: `curl https://YOUR_USERNAME-gridmind-rl.hf.space/health`
- [ ] Submit Space URL to hackathon organizers

---

## 📚 Additional Resources

- **OpenEnv Spec:** https://github.com/meta-pytorch/OpenEnv
- **OpenRouter Free Models:** https://openrouter.ai/keys
- **HF Spaces Docs:** https://huggingface.co/docs/hub/spaces
- **GridMind Repository:** https://github.com/LO-Kyu/gridmind-rl

---

## 📄 License

See `LICENSE` in the repository.
