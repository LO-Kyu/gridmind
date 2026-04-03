# GridMind-RL

**OpenEnv-style environment** for reinforcement learning and LLM agents on **building energy management**: HVAC, thermal storage, demand response, batch job scheduling, and load shedding under time-varying electricity prices and grid stress.

---

## Project overview

GridMind-RL simulates a **24-hour** control horizon at **15-minute resolution** (96 steps per episode). The agent observes prices, temperature, storage, process load, grid stress, carbon intensity, and batch job deadlines; it acts with continuous and discrete controls aligned with real **demand response** and **industrial/commercial** load-shaping problems.

**Why it matters:** Optimizing flexible loads against **time-of-use pricing** and **grid signals** reduces cost and emissions while respecting comfort and process constraints—an active area for RL and LLM-based control research.

**Strengths for judges**

| Area | Detail |
|------|--------|
| Spec | `openenv.yaml` documents server port, schemas, tasks, and endpoints |
| API | REST: reset, step, state, grade, health, ping, replay, tasks, metrics |
| Tasks | Three levels (easy / medium / hard) with deterministic episode grading |
| Baseline | Root `inference.py` + OpenAI-compatible LLM client and heuristic fallback |
| Ops | Multi-stage **Docker** image: Go environment + Python dashboard + deps |

---

## Quick start (copy-paste)

**Minimal flow** (API on **7860** only; keep Docker running, then run `python` in a **second** terminal from the repo root with `pip install -r python/requirements.txt` already done):

```bash
docker build -t gridmind-rl .
docker run -p 7860:7860 gridmind-rl

python inference.py --fast-mode --episodes 1
```

### 1. Build and run (Docker)

From the **repository root**:

```bash
docker build -t gridmind-rl .
docker run --rm -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl
```

- **7860** — Environment API (OpenEnv / agent traffic); http://localhost:7860  
- **7861** — Web dashboard (optional); http://localhost:7861  

**Windows (PowerShell)** — same commands in a terminal with Docker Desktop running.

### 2. Validate the API (optional)

With the container running, from the repo root (host Python with `requests`):

```bash
pip install requests
python python/validate.py --env-url http://localhost:7860
```

### 3. Run baseline inference

On the **host** (not inside the container unless you set `--env-url` to the env server):

```bash
pip install -r python/requirements.txt
```

**Windows — PowerShell:**

```powershell
$env:ENV_URL="http://localhost:7860"
python inference.py --fast-mode --episodes 1
```

**Windows — Command Prompt (cmd):**

```bat
set ENV_URL=http://localhost:7860
python inference.py --fast-mode --episodes 1
```

**Linux / macOS:**

```bash
export ENV_URL=http://localhost:7860
python inference.py --fast-mode --episodes 1
```

You can run the same entrypoint directly with `python python/inference.py` (e.g. `python python/inference.py --fast-mode`); flags match the root `inference.py` wrapper.

**LLM baseline** (requires Hugging Face or other OpenAI-compatible API credentials):

```bash
export ENV_URL=http://localhost:7860
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token_here
python inference.py --episodes 1 --llm-every 4
```

Results are written to `baseline_scores.json` by default (`--output` to change).

---

## Tasks

| ID | Difficulty | Name | Objective |
|----|------------|------|-----------|
| 1 | Easy | Cost minimization | Minimize total energy cost over the episode. No temperature or batch-job objectives in the grade. |
| 2 | Medium | Constrained temperature | Minimize cost while keeping indoor temperature within **±2 °C** of setpoint (19–23 °C) for graded temperature compliance. |
| 3 | Hard | Full demand response | Minimize cost, maintain temperature, respond to **grid stress** (e.g. shed load when stress is high), complete **batch jobs** on time, and reduce **carbon** vs a baseline policy in the composite score. |

Episode **grade** is returned by `GET /grade` after the episode completes (or after a partial run if you stopped stepping early). Sub-scores are task-dependent and documented in code (`env/tasks.go`).

---

## Reward Structure

The environment uses a **dense, multi-component reward** signal. Each step returns a scalar `reward` (the sum) and a detailed `reward_components` breakdown in `info`:

| Component | Key | Active | Description |
|-----------|-----|--------|-------------|
| **Cost Savings** | `cost_savings` | All tasks | Positive baseline (1.5) minus relative step cost. Smart agents that reduce energy use earn higher rewards. |
| **Temperature Constraint** | `temp_constraint` | Task 2, 3 | Gaussian bonus for staying near setpoint (21°C). Max +1.5 at setpoint, degrades toward bounds, penalty outside [19°C, 23°C]. |
| **Grid Demand Response** | `grid_response` | Task 3 | Bonus for shedding load during high grid stress (>0.7), readiness bonus during moderate stress, penalty for unnecessary shedding. |
| **Efficiency Bonus** | `efficiency_bonus` | All tasks | Rewards thermal storage arbitrage (charge during cheap prices, discharge during expensive) and maintaining useful storage levels. |
| **Stability Reward** | `stability_penalty` | All tasks | Positive reward for smooth, stable control; penalty for rapid HVAC/storage oscillation. |
| **Deadline Penalty** | `deadline_penalty` | Task 2, 3 | Penalty per missed batch job deadline (-1.5 each). Positive bonus for keeping jobs on track. |
| **Carbon Reward** | `carbon_reward` | Task 3 | Baseline bonus for low-carbon operation, reduced by carbon-heavy consumption. Extra bonus during clean grid periods. |

**Grading weights (Task 3):** cost 28%, temperature 20%, grid_response 20%, batch_deadline 12%, carbon 20%.

---

## HTTP API

Base URL: `http://<host>:7860` (default in container: port **7860**).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness; JSON `status`, `version` |
| GET | `/ping` | Lightweight liveness; JSON `status` |
| POST | `/reset` | Start episode: body e.g. `{"task_id": 1, "seed": 42, "num_buildings": 1}` |
| POST | `/step` | Advance one step: JSON action or array of actions (multi-building) |
| GET | `/state` | Full snapshot: buildings, downsampled price/carbon curves, step, task, etc. |
| GET | `/grade` | Episode score in `[0, 1]`, sub-scores, exploit flags |
| GET | `/replay` | Step replay list |
| GET | `/tasks` | Task metadata and grader weights |
| GET | `/metrics` | Prometheus-style text metrics |

**Action JSON fields** (single building): `hvac_power_level`, `thermal_charge_rate`, `batch_job_slot`, `load_shed_fraction`, optional `building_id`.

Schemas and primary endpoints: **`openenv.yaml`** at repo root (see Notes for additional endpoints like `/metrics`).

---

## Evaluation modes (`inference.py`)

There is **no** `--judge-mode` flag in this repository. Use the modes below.

| Mode | Command pattern | Behavior |
|------|-----------------|----------|
| **Fast (heuristic)** | `python inference.py --fast-mode` | No LLM calls; deterministic given env seed; fastest for CI or smoke tests. |
| **Default LLM** | `python inference.py` | Uses OpenAI-compatible API (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`); default `--llm-every 4` reuses each LLM action for 4 steps to limit API cost. |
| **Recommended for automated evaluation / judging** | `python inference.py --fast-mode --episodes 1` | Recommended when automated pipelines need **reproducibility** and **no external API** dependency. |

Other useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--episodes` | `1` | Episodes per task (tasks 1–3 run in sequence) |
| `--env-url` | `ENV_URL` or `http://localhost:7860` | Environment base URL |
| `--llm-every` | `4` | Steps per LLM call (ignored in `--fast-mode`) |
| `--max-steps` | full episode | Stop after N steps; grade reflects **partial** episode |
| `--output` | `baseline_scores.json` | Results path |
| `--verbose` | off | Extra step logs |

---

## Logging format (baseline)

For each episode the script prints, in order:

1. **`[START]`** — episode beginning (after `reset`)  
2. **`[STEP1]` … `[STEP96]`** (full episode) — one line per successful `POST /step`; a full episode has **96** steps (`[STEP1]` through `[STEP96]`) unless `--max-steps` or an early error stops the loop  
3. **`[END]`** — after `GET /grade` for that episode  

Additional lines (banners, task headers, `[OK]` / `[WARN]`) may appear; parsers should match the bracketed markers above.

Example shape:

```text
[START]
[STEP1]
[STEP2]
...
[STEP96]
[END]
```

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  Client: python inference.py (LLM or heuristic)             │
│       │ HTTP (reset / step / grade)                         │
│       ▼                                                     │
│  ┌──────────────────┐     ┌─────────────────────────────┐ │
│  │ gridmind-server  │     │  Dashboard (optional)        │ │
│  │  Go :7860        │◄────│  FastAPI + static UI :7861   │ │
│  │  env/* simulation│     │  proxies /api → :7860       │ │
│  └──────────────────┘     └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

- **Core:** `main.go` + `env/` (physics, rewards, tasks, grading)  
- **Baseline:** `inference.py` (root) → `python/inference.py`  
- **Dashboard:** `dashboard/server.py`, `dashboard/static/`  
- **Spec:** `openenv.yaml`

---

## Docker (detailed)

| Step | Command |
|------|---------|
| Build | `docker build -t gridmind-rl .` |
| Run (foreground) | `docker run --rm -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl` |
| Run (background) | `docker run -d --rm -p 7860:7860 -p 7861:7861 --name gridmind gridmind-rl` |
| Stop (background) | `docker stop gridmind` |
| Inference **inside** container | `docker exec -it gridmind python /app/inference.py --fast-mode --env-url http://127.0.0.1:7860` |

The image runs **supervisord** as a non-root user with two programs: Go server (`PORT=7860`) and uvicorn dashboard (`7861`).

---

## Notes for judges and operators

| Topic | Detail |
|-------|--------|
| **Ports** | **7860** = environment API; **7861** = dashboard. Some hosts only expose one public port—API is the required one for OpenEnv-style evaluation. |
| **Episode length** | **96 steps** = 24 h at 15 min/step. Observation `step` is **0–95** for a full episode. |
| **`openenv.yaml`** | Lists main endpoints; **`/metrics`** exists at runtime but may not appear in the YAML block—treat as an extra ops endpoint. |
| **Reproducibility** | Env is seed-controlled. LLM outputs may still vary by provider even at `temperature=0`. |
| **`--max-steps`** | Produces a **partial** episode; final `GET /grade` reflects that partial trajectory. |
| **Manual run (no Docker)** | Install Go 1.21+, `go run .` from repo root (default port 7860); install Python deps and run `python inference.py` as above. |
| **Runtime** | The baseline completes within typical hackathon limits (<20 minutes). |

---

## Example API Calls

With the container running (`docker run -p 7860:7860 gridmind-rl`):

```bash
# Health check
curl http://localhost:7860/health
# → {"status":"ok","version":"1.0.0"}

# Reset to Task 3 (hard) with seed 42
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 3, "seed": 42, "num_buildings": 1}'
# → {"observations":[{"indoor_temperature":21.3,...}],"episode":1,"task_id":3,"seed":42}

# Take one step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"hvac_power_level": 0.6, "thermal_charge_rate": 0.3, "batch_job_slot": 1, "load_shed_fraction": 0.1}'
# → {"observation":{...},"reward":2.15,"done":false,"info":{"reward_components":{...},...}}

# Get current state
curl http://localhost:7860/state
# → {"buildings":[...],"price_curve_episode":[...],"step":1,"task_id":3,...}

# Grade after episode ends (run 96 steps first)
curl http://localhost:7860/grade
# → {"task_id":3,"score":0.3115,"sub_scores":{"cost":0.25,"temperature":0.14,...},...}

# List all tasks
curl http://localhost:7860/tasks
# → [{"id":1,"name":"Cost Minimization","difficulty":"easy",...},...]
```

---

## Hugging Face Space Deployment

### 1. Create a new HF Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create a **Docker** space. Select:
- **SDK:** Docker
- **Hardware:** CPU Basic (2 vCPU, 16GB)

### 2. Push code to HF

```bash
# Clone and push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/gridmind-rl
git push hf main
```

### 3. Verify deployment

Once the Space builds, verify at your Space URL:

```bash
curl https://YOUR_USERNAME-gridmind-rl.hf.space/health
# → {"status":"ok","version":"1.0.0"}

curl -X POST https://YOUR_USERNAME-gridmind-rl.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":1,"seed":42}'
```

> **Note:** HF Spaces only exposes **one port** publicly. Port **7860** (the OpenEnv API) is the primary port and will be the public endpoint. The dashboard on port 7861 is for local development only.

---

## License

See `LICENSE` in the repository.
