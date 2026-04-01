# GridMind-RL

GridMind-RL is an OpenEnv-compliant reinforcement learning environment simulating a commercial/industrial building energy management system.

An RL agent acts as the energy controller, shaping electrical load profiles by adjusting HVAC setpoints, managing thermal storage, and scheduling batch processes. The goal is to optimize operations in response to real-time electricity prices, grid carbon intensity, and utility demand-response signals.

## Architecture

```text
 ┌──────────────────────┐        ┌─────────────────────────────┐
 │                      │        │                             │
 │    LLM RL Agent      │◄───────┤    GridMind-RL Server       │
 │   (Inference Script) │ POST   │    (Go OpenEnv Backend)     │
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

## Observation Space

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

## Action Space

| Name | Type | Range | Description |
|------|------|-------|-------------|
| `hvac_power_level` | float | [0.0, 1.0] | Fraction of max HVAC cooling/heating power to apply. |
| `thermal_charge_rate` | float | [-1.0, 1.0] | Charge (positive) or discharge (negative) thermal storage. |
| `batch_job_slot` | int | [0, 4] | Delay scheduling the next batch job by 0-4 time slots. |
| `load_shed_fraction` | float | [0.0, 0.5] | Fraction of non-critical load to shed (max 50%). |
| `building_id` | int | [0, 2] | Select which building to apply this action to (federation). |

## Tasks

GridMind-RL features 3 progressively difficult tasks:

1. **Task 1: Cost Minimization (Easy)**
   Minimize total energy costs by moving load to off-peak periods using thermal storage. No temperature constraints.
2. **Task 2: Temperature Management (Medium)**
   Minimize costs while keeping indoor temperatures strictly within 19°C – 23°C.
3. **Task 3: Full Demand Response (Hard)**
   Minimize cost, maintain temperature, successfully schedule batch jobs before deadlines, and shed loads when the grid stress signal exceeds 0.7.

## Reward Function

The dense reward includes several components:
* **Cost Savings:** Proportional to energy savings vs the baseline flat tariff policy.
* **Temp Constraint:** Gaussian bonus for being close to the setpoint, harsh penalty for exiting [19°C, 23°C].
* **Grid Response:** Large bonus if `load_shed_fraction` > 0 when `grid_stress_signal` > 0.7.
* **Deadline Penalty:** Heavy negative reward for jobs that execute past their deadline slot.
* **Efficiency Bonus:** Rewards charging thermal storage when the current price is *below* the future moving average.
* **Stability Penalty:** Penalizes rapid oscillation of the HVAC and storage controls.

*Exploit Detection:* The grader detects degenerate strategies (e.g. permanently shedding 40% load) and applies up to a 30% score penalty.

## Usage

### Local Docker Build

```bash
docker build -t gridmind-rl .
docker run -p 7860:7860 -p 7861:7861 gridmind-rl
```

* Backend OpenEnv server: http://localhost:7860
* Visualization Dashboard: http://localhost:7861

### Validating the Environment

```bash
python python/validate.py --env-url http://localhost:7860
```

### Running Baseline Inference

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token

# Install dependencies
pip install -r python/requirements.txt

# Run inference
python python/inference.py --episodes 3
```

## Extensions
* **Multi-building mode:** Switch the environment to 3 buildings via `POST /reset {"num_buildings": 3}` and output action arrays for coordinated dispatch.
* **Add new tasks:** Edit `env/tasks.go` and implement a new `gradeTaskX` component.
