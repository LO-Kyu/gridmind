# GridMind-RL

GridMind-RL is an OpenEnv-compliant reinforcement learning environment simulating a commercial/industrial building energy management system.

An RL agent acts as the energy controller, shaping electrical load profiles by adjusting HVAC setpoints, managing thermal storage, and scheduling batch processes. The goal is to optimize operations in response to real-time electricity prices, grid carbon intensity, and utility demand-response signals.

---

## 🙋 Beginner? Start Here

If you're new to this project, you probably have these questions:

### ❓ Why do I need an API?

In this project, the "brain" that makes energy decisions is an **AI language model (LLM)** — like Llama.

Instead of running the full AI model on your own computer (which requires a powerful GPU), you connect to an **API** (Application Programming Interface) — a remote server that already has the model running. You send it the current building state (temperature, price, etc.) and it sends back what action to take (e.g. "charge thermal storage").

Think of it like this:
```
Your Computer  ──(asks question)──►  API Server (has the AI)  ──(sends answer)──►  Your Computer
```

Without an API key, your script has no way to reach the AI model and the inference won't work.

---

### ❓ How do I get an API key?

This project uses **Hugging Face** — a free platform that hosts AI models.

#### Step-by-step:

1. **Create a free account** at [https://huggingface.co/join](https://huggingface.co/join)

2. **Go to your profile → Settings → Access Tokens**
   Direct link: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. Click **"New token"**, give it any name (e.g. `gridmind`), and select role **"Read"**

4. Copy the token — it looks like: `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ`

5. You'll paste this token in the terminal when running the project (shown below)

> **💡 It's free!** Hugging Face's inference API has a free tier that's enough to run this project.

---

### ❓ Why Llama? What even is Llama?

**Llama** (Large Language Model Meta AI) is an open-source AI model made by Meta (Facebook). Think of it like a smarter, programmable version of ChatGPT that you can use via an API.

**Why this project uses Llama specifically:**

| Reason | Explanation |
|--------|-------------|
| 🆓 Free to use | Available on Hugging Face at no cost |
| 📖 Open-source | The weights and code are public — no black box |
| 🧠 Smart enough | Llama 3.1 8B is capable of reading sensor data and outputting valid JSON actions |
| ⚡ Fast | The 8B (8 billion parameter) version is small enough to run quickly on Hugging Face's servers |
| 🔄 OpenAI-compatible | It uses the same API format as OpenAI, so the code works with many models |

The model reads the building state (temperature, electricity price, grid stress) and outputs a JSON action like:
```json
{
  "hvac_power_level": 0.4,
  "thermal_charge_rate": 0.5,
  "batch_job_slot": 2,
  "load_shed_fraction": 0.0,
  "building_id": 0
}
```

> **You can also swap Llama for any other OpenAI-compatible model** (GPT-4, Mistral, etc.) by changing the environment variables.

---

## 🏗️ Architecture

```text
 ┌──────────────────────┐        ┌─────────────────────────────┐
 │                      │        │                             │
 │    LLM RL Agent      │◄───────┤    GridMind-RL Server       │
 │   (Python Script)    │ POST   │    (Go OpenEnv Backend)     │
 │                      ├───────►│  Port 7860                  │
 └──────────────────────┘ Action │                             │
                                 └──────────────┬──────────────┘
                                          State │
                                        Polling │
                                 ┌──────────────▼──────────────┐
                                 │                             │
                                 │     Visualization UI        │
                                 │    (FastAPI + HTML/JS)      │
                                 │  Port 7861                  │
                                 └─────────────────────────────┘
```

---

## 🚀 How to Run the Project (Step by Step)

### 🧸 Super Simple Quick-Start (The 10-Year-Old Guide)

Think of this project like a video game. Here is the super-simple manual on how to turn it on whenever you sit down at your computer.

#### Step 1: Wake up the Engine 🐳
Before you can play the game, you need to turn on the engine that powers it.
1. Click on your Windows Start menu.
2. Search for the app called **Docker Desktop**.
3. Open it and wait a minute until the little blue whale icon at the bottom of your screen says "Engine running".

#### Step 2: Start the Game Server 🎮
Now that the engine is awake, you need to tell it to run your specific game.
1. Open your terminal in this project folder.
2. Copy, paste, and run this exact command (it runs it in the background so it won't crash!):
   ```bash
   docker run -d -p 7860:7860 -p 7861:7861 gridmind-rl
   ```

#### Step 3: Open the TV Screen 📺
The game is running invisibly in the background. To actually *see* it, you use your web browser as the TV screen!
1. Open Google Chrome or Edge.
2. In the very top bar, type exactly this:
   **http://localhost:7861**
3. Press Enter. The dashboard will load on your screen!

#### Step 4: Connect the Super Smart AI Brain 🧠
To watch the AI actually play and learn, you need to connect the brain.
1. Make sure you have your Hugging Face API password (from the Beginner section above).
2. Open a new terminal window in the project folder.
3. Run the Python brain script (PowerShell command):
   ```powershell
   pip install -r python/requirements.txt
   $env:API_BASE_URL="https://router.huggingface.co/v1"
   $env:MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
   $env:HF_TOKEN="your_password_here"
   python python/inference.py --episodes 3
   ```
4. Sit back, don't click anything on the dashboard, and watch the AI do all the work and get smarter!

---

### ⚙️ Advanced Setup Options

There are **two advanced ways** to set up this project from scratch:
- **Option A** — Using Docker (recommended, easiest)
- **Option B** — Running manually without Docker

---

### Option A: Docker (Recommended)

Docker packages everything into a container so you don't need to install Go, Python versions, etc. separately.

#### Prerequisites

- Install Docker Desktop: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- A Hugging Face API token (see above ☝️)

#### Step 1 — Build the Docker image

Open a terminal in the project folder and run:

```bash
docker build -t gridmind-rl .
```

This may take a few minutes the first time.

#### Step 2 — Start the environment server

```bash
docker run -p 7860:7860 -p 7861:7861 gridmind-rl
```

You should see the server start. Keep this terminal open.

- **Environment API:** http://localhost:7860
- **Visualization Dashboard:** http://localhost:7861

#### Step 3 — Install Python dependencies

Open a **new terminal** (keep the Docker one running) and run:

```bash
pip install -r python/requirements.txt
```

#### Step 4 — Set your API credentials

**On Windows (Command Prompt):**
```cmd
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
set HF_TOKEN=hf_your_token_here
```

**On Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN     = "hf_your_token_here"
```

**On Mac/Linux:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token_here
```

Replace `hf_your_token_here` with your actual Hugging Face token.

#### Step 5 — Run the AI agent

```bash
python python/inference.py --episodes 3
```

You'll see the agent play through 3 episodes across all 3 tasks and print scores.

---

### Option B: Manual (Without Docker)

Use this if you don't have Docker installed.

#### Prerequisites

- [Go 1.21+](https://go.dev/dl/) — for running the environment server
- [Python 3.9+](https://www.python.org/downloads/) — for the AI agent script
- A Hugging Face API token (see above ☝️)

#### Step 1 — Start the Go environment server

```bash
go run main.go
```

The server starts on port `7860`. Keep this terminal open.

#### Step 2 — Open a new terminal and install Python dependencies

```bash
pip install -r python/requirements.txt
```

#### Step 3 — Set your API credentials (same as Option A, Step 4 above)

#### Step 4 — Validate the environment is working

```bash
python python/validate.py --env-url http://localhost:7860
```

You should see a series of checks pass. If they do, you're good to go.

#### Step 5 — Run the AI agent

```bash
python python/inference.py --episodes 3
```

---

## 📊 What Happens When You Run It

The agent runs through **3 tasks** (Easy → Medium → Hard), each for the number of episodes you specify:

| Task | Difficulty | Goal |
|------|-----------|------|
| Task 1 | Easy | Minimize energy costs only |
| Task 2 | Medium | Minimize costs + keep temperature 19°C–23°C |
| Task 3 | Hard | Costs + temperature + batch job deadlines + grid stress response |

At the end, you'll see a score table like:
```
============================================================
BASELINE SCORES SUMMARY
============================================================
Task       Model                          Score      Episodes
------------------------------------------------------------
Task 1     meta-llama/Llama-3.1-8B-Instruct  0.7823     3
Task 2     meta-llama/Llama-3.1-8B-Instruct  0.6541     3
Task 3     meta-llama/Llama-3.1-8B-Instruct  0.5102     3
------------------------------------------------------------
Overall                                    0.6489
```

Results are also saved to `baseline_scores.json`.

---

## 📐 Observation Space

These are the sensor readings the agent sees at each step:

| Name | Type | Range | Description |
|------|------|-------|-------------|
| `indoor_temperature` | float | [15.0, 30.0] | Current indoor temperature (°C). Goal is usually 21°C. |
| `thermal_storage_level` | float | [0.0, 1.0] | Thermal storage capacity fill level. |
| `process_demand` | float | [0.0, 50.0] | Current uncontrolled process power demand (kW). |
| `current_price` | float | [0.02, 0.50] | Real-time electricity price ($/kWh). |
| `grid_stress_signal` | float | [0.0, 1.0] | Utility signal indicating grid stress. >0.7 requires shedding. |
| `carbon_intensity` | float | [100, 700] | Grid carbon emissions intensity (gCO2/kWh). |
| `hour_of_day` | int | [0, 23] | Current hour, useful for scheduling. |
| `batch_queue` | list[int] | N/A | List of deadline slots for pending batch jobs. |
| `cumulative_cost` | float | [0.0, inf) | Running energy cost in $. |
| `step` | int | [0, 95] | Current episode timestep (15-min intervals over 24h). |
| `building_id` | int | [0, 2] | ID of the building in multi-building federated mode. |

---

## 🕹️ Action Space

These are the controls the agent outputs at each step:

| Name | Type | Range | Description |
|------|------|-------|-------------|
| `hvac_power_level` | float | [0.0, 1.0] | Fraction of max HVAC cooling/heating power to apply. |
| `thermal_charge_rate` | float | [-1.0, 1.0] | Charge (positive) or discharge (negative) thermal storage. |
| `batch_job_slot` | int | [0, 4] | Delay scheduling the next batch job by 0-4 time slots. |
| `load_shed_fraction` | float | [0.0, 0.5] | Fraction of non-critical load to shed (max 50%). |
| `building_id` | int | [0, 2] | Select which building to apply this action to (federation). |

---

## 🏆 Reward Function

The dense reward includes several components:
* **Cost Savings:** Proportional to energy savings vs the baseline flat tariff policy.
* **Temp Constraint:** Gaussian bonus for being close to the setpoint, harsh penalty for exiting [19°C, 23°C].
* **Grid Response:** Large bonus if `load_shed_fraction` > 0 when `grid_stress_signal` > 0.7.
* **Deadline Penalty:** Heavy negative reward for jobs that execute past their deadline slot.
* **Efficiency Bonus:** Rewards charging thermal storage when the current price is *below* the future moving average.
* **Stability Penalty:** Penalizes rapid oscillation of the HVAC and storage controls.

*Exploit Detection:* The grader detects degenerate strategies (e.g. permanently shedding 40% load) and applies up to a 30% score penalty.

---

## 🔧 Extensions

* **Multi-building mode:** Switch the environment to 3 buildings via `POST /reset {"num_buildings": 3}` and output action arrays for coordinated dispatch.
* **Use a different model:** Just change `MODEL_NAME` to any OpenAI-compatible model (e.g. `mistralai/Mistral-7B-Instruct-v0.3`).
* **Add new tasks:** Edit `env/tasks.go` and implement a new `gradeTaskX` component.

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on port 7860 | Make sure the Docker container or `go run main.go` is still running |
| `401 Unauthorized` from Hugging Face | Your `HF_TOKEN` is wrong or expired — generate a new one |
| `Model not found` error | Some large models require you to accept terms on Hugging Face first. Go to the model page and click "Agree to terms" |
| Python package errors | Make sure you ran `pip install -r python/requirements.txt` |
| `docker: command not found` | Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop) |
