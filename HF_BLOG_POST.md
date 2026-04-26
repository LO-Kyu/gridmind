---
title: GridMind-RL: Training LLMs to Manage Industrial Buildings with GRPO
description: How we built an RL environment that teaches language models real-world energy management — and what 10 training runs taught us.
---

# GridMind-RL: Training LLMs to Manage Industrial Buildings

*OpenEnv Hackathon India 2026 · GridMind-RL Team*

---

There is a building somewhere running its air conditioning at full power right now,
even though electricity costs four times more than it did six hours ago. Not because
the operator made a bad decision — but because the control system doesn't know the
price changed.

Industrial buildings consume roughly 40% of global electricity. Most are managed by
fixed schedules that made sense when they were written and haven't been touched since.
The cost gap between a naive policy and an intelligent one is measurable in thousands
of dollars per building per year.

LLMs can read pricing curves, respond to fault alerts, and follow natural language
instructions. The missing piece has always been an environment that trains them to
*act* on that reasoning under real operational pressure.

That's what we built.

---

## The Environment

GridMind-RL simulates a complete 24-hour industrial building energy system at
15-minute resolution — 96 decision steps per episode. The agent operates in
continuous time, responding to a world that changes around it: prices spike, equipment
degrades, grid stress signals arrive, and sometimes the chiller fails at 2pm on the
hottest day of the year.

**The agent sees 13 fields every step:**
current indoor temperature, thermal storage level, electricity price, grid stress
signal, HVAC efficiency (which degrades continuously over the episode), active fault
alarms, a 4-step price forecast, cumulative cost so far, carbon intensity, batch job
queue, hour of day, and — in Task 4 — a natural language instruction card describing
the episode's objective.

**The agent has four levers:**

| Action | Range | What it does |
|--------|-------|--------------|
| `hvac_power_level` | 0 → 1 | How hard the HVAC system works |
| `thermal_charge_rate` | -1 → 1 | Charge or discharge thermal storage |
| `batch_job_slot` | 0 → 4 | When to run deferrable industrial loads |
| `load_shed_fraction` | 0 → 0.5 | Voluntary demand reduction during grid stress |

**Four tasks test different capabilities:**

- **Cost Minimization** — Navigate 24-hour price volatility and thermal storage
  arbitrage to minimize total energy spend.

- **Comfort Management** — Hold indoor temperature within 19–23°C through equipment
  degradation, faults, and shifting external conditions.

- **Demand Response** — Read grid stress signals in real time and voluntarily shed
  load to earn demand-response credit without sacrificing comfort.

- **Instruction Following** — Parse a natural language objective card at episode
  start and adapt the entire 96-step strategy to meet it.

### Why the reward has nine components

The naive approach is to reward cost savings and call it done. The problem is that
a cost-only reward teaches the agent to turn off the HVAC entirely — perfect score,
frozen building.

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
| `task_satisfaction` | 0.50* | Task 4 only — weighted per the instruction card |

A reward this dense is harder to game. An agent that exploits one component while
neglecting the others will see it reflected immediately in the score.

---

## Training

We trained Qwen2.5-1.5B-Instruct with QLoRA (4-bit, rank 16) using GRPO via
HuggingFace TRL. Each run is 60 steps on a T4 GPU, taking roughly 35 minutes.
We ran 10 training iterations in total.

**Why GRPO over PPO?**
GRPO doesn't require a separate value network. At 1.5B parameters on a T4, that
memory saving matters. Instead of estimating a value baseline, GRPO samples a group
of completions per prompt and computes advantages by comparing them against each
other — a natural fit for our setting where we generate multiple actions per state.

| Component | Detail |
|-----------|--------|
| Model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | QLoRA (4-bit, rank 16) |
| Algorithm | GRPO via HuggingFace TRL |
| Hardware | HF Space T4 GPU |
| Training time | ~35 minutes per run |
| Total runs | 10 |

---

## What the Curves Show

### Run 1 vs Run 10: The reward is climbing

The clearest evidence of learning is what happens to the reward curve within a single
training run — and how that shape changes as the training setup matures.

**Run 1 — the first training run:**

![Reward Curve — Run 1](curves/train%201/reward_curve.png)
*Run 1: Reward climbs from −0.47 to ~0.65 over 60 steps. The model is learning fast
in the early steps, then stabilizing — with a small dip at the very end.*

**Run 10 — after iterative refinement:**

![Reward Curve — Run 10](curves/train%2010/reward_curve.png)
*Run 10: Same starting point, smoother curve, still rising at step 60. The model
hasn't plateaued — which means longer training would continue to improve it.*

Both runs start at the same reward (~−0.47) because each run initializes fresh.
What changes is the *shape*: Run 10 is more stable, ends higher (~0.68 vs ~0.65),
and shows no end-of-run dip. Ten runs of iteration on the training setup produced
a meaningfully cleaner learning signal.

The 1.1-point reward improvement within a single 60-step run is not noise.
The agent is learning to manage energy in real time.

### Before and After: Where the model wins

**Run 1 — heuristic baseline vs GRPO-trained:**

![Baseline Comparison — Run 1](curves/train%201/baseline_comparison.png)
*Run 1: The trained model outperforms the heuristic on Task 4 by a significant margin.
On Tasks 1–3 it scores below the heuristic — early training, limited steps.*

**Run 10 — heuristic baseline vs GRPO-trained:**

![Baseline Comparison — Run 10](curves/train%2010/baseline_comparison.png)
*Run 10: Similar pattern. Task 4 remains the trained model's strongest result.
Tasks 1–3 gap to the heuristic has narrowed compared to Run 1.*

### The Task 4 result is the headline

The heuristic scores **0.30** on Task 4. The trained model scores **0.70**.
That is a **133% improvement** on instruction following — and it makes complete sense.

A fixed heuristic cannot read a natural language objective card. It cannot parse
"keep total cost under $2.50 while maintaining comfort" and change its behavior
accordingly. The trained model can. That capability gap is exactly what this
environment was designed to measure.

Tasks 1–3 tell a more honest story. Time-of-day HVAC scheduling is genuinely
reasonable for cost and temperature — the heuristic is a strong baseline on those
tasks, and 60 training steps with a 1.5B model isn't enough to beat it consistently.
That's not a failure of the environment. It's a signal that longer training would
continue to pay off.

---

## What the Agent Learns

None of these behaviors are hardcoded. The reward signal surfaces them:

**Thermal arbitrage** — the agent learns to charge thermal storage during off-peak
hours (~4¢/kWh) and discharge during peak (~32¢/kWh), reducing the effective cost
of maintaining comfort during expensive periods.

**Grid cooperation** — when the stress signal exceeds 0.7, the agent voluntarily
sheds load rather than ignoring the signal. The demand-response credit offsets the
comfort penalty.

**Fault adaptation** — when HVAC efficiency degrades below a threshold, the agent
reduces its HVAC target rather than fighting a weakened system at full power.

**Instruction parsing** — in Task 4, the agent reads the objective card and adjusts
its entire 96-step strategy accordingly, not just the next action.

---

## What We'd Do With More Compute

- **300+ training steps** would likely close the gap on Tasks 1–3
- A **7B model** with the same setup would show sharper policy improvement
- **Multi-agent coordination** — 3 buildings sharing a 250kW feeder — is fully
  implemented but not yet the primary training focus. Fleet-level demand response
  is the next frontier.

---

## Try It

The environment is live. You can reset an episode, send actions, and read rewards
right now from your terminal:

```bash
# Health check
curl https://prajwal782007-gridmind.hf.space/health

# Start an episode
curl -X POST https://prajwal782007-gridmind.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4}'

# Take an action
curl -X POST https://prajwal782007-gridmind.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"hvac_power_level": 0.6, "thermal_charge_rate": 0.4,
       "batch_job_slot": 2, "load_shed_fraction": 0.0, "building_id": 0}'
```

- 🤗 **Environment**: https://prajwal782007-gridmind.hf.space
- 📓 **Training Notebook**: [gridmind_grpo_colab.ipynb](https://colab.research.google.com/github/LO-Kyu/gridmind/blob/main/scripts/gridmind_grpo_colab.ipynb)
- 🐙 **Code**: https://github.com/LO-Kyu/gridmind

---

*Built for the OpenEnv Hackathon India 2026 · April 25–26 · Scaler School of
Technology, Bangalore.*