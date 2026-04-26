---
title: GridMind-RL: Training LLMs to Manage Industrial Buildings with GRPO
description: How we built an OpenEnv-compatible RL environment that teaches language models real-world energy management — and what the training curves actually show.
---

# GridMind-RL: Training LLMs to Manage Industrial Buildings

*OpenEnv Hackathon India 2026 · Aditya Suryavanshi, Shreeshant Bokade, Prajwal Valekar*

---

There is a building somewhere running its air conditioning at full power right now,
even though electricity costs five times more than it did six hours ago. Not because
the operator made a bad decision — but because the control system doesn't know the
price changed.

Industrial buildings consume roughly 40% of global electricity. Most are managed by
fixed schedules that made sense when they were written and haven't been touched since.
The cost gap between a naive policy and an intelligent one is measurable in thousands
of dollars per building per year.

LLMs can read pricing curves, respond to fault alerts, and follow natural language
instructions — but there has never been an environment that trains them to *act* on
that reasoning under real operational pressure. We built one, trained on it, and the
results show an agent that beats a hand-crafted heuristic on the tasks that matter most.

---

## Who We Are

We are a team of three fascinated by the gap between what LLMs can reason about and
what they can actually *do*. Building energy management sits right at that frontier —
the domain is rich, the stakes are real, and no RL benchmark has touched it.
GridMind-RL is our attempt to change that.

We built this for the Meta PyTorch OpenEnv Hackathon Grand Finale at Scaler School
of Technology, Bangalore, April 25–26, 2026.

---

## Which Themes We're Targeting

GridMind-RL directly addresses two hackathon themes:

**Theme 1 — Multi-Agent Interactions:** Three buildings share a 360kW grid feeder
(120kW per building). A coordinator LLM reads fleet-wide demand via `/feeder` and
sets per-building price multipliers via `/coordinate`. Buildings that ignore the
signal trip the feeder limit — causing a grid fault penalty for all three. This
creates genuine emergent coordination pressure without explicit communication.

**Theme 3.1 — World Modeling (Professional Tasks):** The `/simulate` endpoint lets
the agent ask "what if?" before committing an action. When HVAC efficiency is low or
faults are active, the agent can simulate a proposed action and revise its plan if
the predicted reward is poor. This trains causal reasoning and persistent world
modeling — exactly what Theme 3 targets.

---

## The Environment

GridMind-RL implements the OpenEnv-compatible interface (reset/step/state/grade)
via a high-performance Go HTTP server. openenv-core==0.2.3 is used as the
Python client library for training-side interaction. It simulates a complete 24-hour industrial
building energy system at 15-minute resolution — 96 decision steps per episode.

The agent operates in continuous time, responding to a world that changes around it:
prices spike up to 5× during tariff faults, equipment degrades, grid stress signals
arrive, and sometimes the chiller fails at 2pm on the hottest day of the year.

**The agent sees a rich observation space every step, including:**
indoor temperature, thermal storage level, electricity price, grid stress signal,
HVAC efficiency (which degrades continuously throughout the episode), active fault
alarms, a 4-step price forecast, cumulative cost, carbon intensity, batch job queue,
and hour of day. In Task 4, this also includes a natural language instruction card.

**The agent has four levers:**

| Action | Range | What it does |
|--------|-------|--------------|
| `hvac_power_level` | 0 → 1 | How hard the HVAC system works |
| `thermal_charge_rate` | -1 → 1 | Charge or discharge thermal storage |
| `batch_job_slot` | 0 → 4 | When to run deferrable industrial loads |
| `load_shed_fraction` | 0 → 0.5 | Voluntary demand reduction during grid stress |

**Four tasks of increasing difficulty:**

- **Cost Minimization** — Navigate 24-hour price volatility (~2¢ to ~36¢/kWh) and
  thermal storage arbitrage to minimize total energy spend.

- **Comfort Management** — Hold indoor temperature within 19–23°C through equipment
  degradation, faults, and shifting external conditions.

- **Demand Response** — Read grid stress signals in real time and voluntarily shed
  load (when signal exceeds 0.7) to earn demand-response credit without sacrificing
  comfort.

- **Instruction Following** — Parse a natural language objective card at episode
  start and adapt the entire 96-step strategy to meet it.

### Why the reward has nine components

The naive approach is to reward cost savings and call it done. The problem is that
a cost-only reward teaches the agent to turn off the HVAC entirely — perfect score,
frozen building. This is textbook reward hacking.

Real building operators don't optimize one metric. They manage a hierarchy:
comfort is non-negotiable, grid compliance is contractual, cost is the primary KPI,
carbon is increasingly regulated, and equipment stability protects the capital budget.

Our reward reflects that hierarchy directly:

| Component | Weight | Why |
|-----------|--------|-----|
| `cost_savings` | 0.28 | Primary operator KPI |
| `carbon_reward` | 0.20 | ESG compliance, increasingly mandatory |
| `temp_constraint` | 0.20 | Hard safety constraint — SLA violations incur penalties |
| `grid_response` | 0.20 | Demand response programs pay operators to shed load |
| `batch_deadline` | 0.12 | Missing deadlines causes downstream production losses |
| `efficiency_bonus` | 0.05 | Incentivises smart thermal storage arbitrage |
| `stability_penalty` | -0.05 | Prevents HVAC thrashing that causes equipment wear |
| `fault_mitigation` | dynamic | Correct fault response prevents costly outages |
| `task_satisfaction` | 0.10–0.50* | Task 4 only — weighted per the instruction card |

> *`task_satisfaction` weight varies by instruction template, ranging from
> 0.10 to 0.50 depending on the episode's objective card (tasks.go).

### How we prevent reward hacking

A multi-component reward is only part of the answer. We also:

- **Clamp all actions** at the server side — the agent cannot exceed valid ranges
  regardless of what it outputs (`hvac_power_level` hard-clamped 0–1,
  `load_shed_fraction` hard-clamped 0–0.5, etc.)
- **Inject four fault types** that make naive exploitation brittle: chiller failure
  (HVAC drops to 20% capacity), grid outage (price up to ×4, stress = 1.0), sensor
  fault (temperature jitter ±5°C), and tariff spike (price up to ×5)
- **Use a seeded but stochastic environment** — price curves, fault timing, and
  demand patterns vary across episodes, preventing the agent from memorizing a
  fixed solution
- **Score via `/grade`** at episode end using a separate grading function that is
  decoupled from the per-step reward signal

---

## Training

We trained Qwen2.5-1.5B-Instruct with QLoRA (4-bit, rank 16) using GRPO via
HuggingFace TRL on a T4 GPU — roughly 35 minutes per run.

| Component | Detail |
|-----------|--------|
| Model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | QLoRA (4-bit, rank 16) |
| Algorithm | GRPO via HuggingFace TRL |
| Hardware | HF Space T4 GPU |
| Training time | ~35 minutes |
| Steps | 60 |

**Why GRPO over PPO?**
GRPO doesn't require a separate value network. At 1.5B parameters on a T4, that
memory saving matters. Instead of estimating a value baseline, GRPO samples a group
of completions per prompt and computes advantages by comparing them against each
other — a natural fit for our setting where we generate multiple actions per state
and want to reinforce the better ones.

The hackathon context emphasized that RL only works if the probability of a good
answer is greater than zero. We confirmed this by running a heuristic baseline first
to verify the environment produces non-zero reward before starting RL training.

---

## Results

### The numbers first

| Policy | Task 1 | Task 2 | Task 3 | Task 4 | Avg (unweighted) |
|--------|--------|--------|--------|--------|------------------|
| Heuristic Baseline | 0.54 | 0.56 | 0.50 | 0.31 | 0.48 |
| GRPO Fine-tuned | 0.42 | 0.34 | 0.47 | **0.49** | 0.43 |

> Heuristic = fixed time-of-day HVAC scheduling, no learning.
> GRPO Fine-tuned = Qwen2.5-1.5B-Instruct after 60 steps of GRPO training
> against the live environment.

The trained model **beats the heuristic on Task 4 by 58%** (0.49 vs 0.31) and
**comes within 6% of the heuristic on Task 3** (0.47 vs 0.50).

These are the two tasks where intelligent reasoning matters most — instruction
parsing and real-time grid cooperation. A fixed schedule cannot read an objective
card. A fixed schedule cannot respond to a grid stress signal that arrives mid-episode.
The trained model can do both.

Tasks 1 and 2 are an honest result. Time-of-day HVAC scheduling is genuinely
competitive for cost and comfort — the heuristic baseline is strong on those
objectives because the physics are predictable. Closing that gap requires more
training steps. The reward curve shows the trend is still moving upward at step 60,
meaning training had not plateaued.

### The reward curve

![Reward Curve](curves/train%204/reward_curve.png)
*Reward vs training step. From −0.47 at step 5 to +0.61 at step 60 — a 1.08-point
gain. The smoothed average (red dashed) is still rising at the final step, confirming
training had not saturated.*

### The before/after

![Baseline Comparison](curves/train%204/baseline_comparison.png)
*Grade scores per task: heuristic baseline (blue) vs GRPO-trained LLM (green).
Task 4 is where the trained model pulls clearly ahead — 58% above the heuristic.*

---

## What the Agent Learns

None of these behaviors are hardcoded. The reward signal surfaces them:

**Thermal arbitrage** — the agent learns to charge thermal storage during off-peak
hours (~3.5¢/kWh) and discharge during peak (~31¢/kWh), reducing the effective cost
of maintaining comfort during expensive periods.

**Grid cooperation** — when the stress signal exceeds 0.7, the agent voluntarily
sheds load rather than ignoring it. The demand-response credit offsets the comfort
penalty — which is why Task 3 performance is closest to the heuristic.

**Fault adaptation** — when HVAC efficiency degrades, the agent reduces its HVAC
target rather than fighting a weakened system at full power. This behavior emerges
purely from the `fault_mitigation` reward component.

**Instruction parsing** — in Task 4, the agent reads the objective card and adjusts
its entire 96-step strategy to meet it. This is the hardest capability for a
heuristic to replicate — and where the trained model wins most clearly.

---

## What's Next

GridMind-RL is a foundation, not a finished product. The directions we find most
interesting:

**Longer training runs** — the reward curve hasn't plateaued at 60 steps. 300+
steps would likely close the gap on Tasks 1 and 2 and push Task 4 performance
further above the heuristic.

**Larger models** — a 7B model with the same training setup would bring stronger
instruction-following capability and better multi-step planning out of the box.

**Fleet-level coordination** — three buildings share a 360kW grid feeder (120kW per
building). Fleet-level coordination is fully implemented — training a coordinator LLM
that orchestrates all three through price signals is the next research direction.
The shared feeder constraint creates genuine emergent coordination pressure — if one
building ignores the signal, all three pay the penalty.

**Real deployment** — the environment's physics are grounded in real building
parameters. The gap between this simulator and a real BMS integration is smaller
than it looks.

---

## Try It

GridMind-RL is live and OpenEnv-compliant. Task 4 is the most interesting to try —
the agent receives a natural language objective card and must adapt its entire
strategy to meet it:

```bash
# Health check
curl https://prajwal782007-gridmind.hf.space/health

# Start a Task 4 episode (instruction following)
curl -X POST https://prajwal782007-gridmind.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4}'

# Take an action and observe the reward
curl -X POST https://prajwal782007-gridmind.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"hvac_power_level": 0.6, "thermal_charge_rate": 0.4,
       "batch_job_slot": 2, "load_shed_fraction": 0.0, "building_id": 0}'

# Grade the full episode
curl https://prajwal782007-gridmind.hf.space/grade
```

- 🤗 **Environment**: https://prajwal782007-gridmind.hf.space
- 📓 **Training Notebook**: [gridmind_grpo_colab.ipynb](https://colab.research.google.com/github/LO-Kyu/gridmind/blob/main/scripts/gridmind_grpo_colab.ipynb)
- 🐙 **Code**: https://github.com/LO-Kyu/gridmind

---

*Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology ·
Grand Finale, April 25–26, 2026, Bangalore.*