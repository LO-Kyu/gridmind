---
title: GridMind-RL: Training LLMs to Manage Industrial Buildings with GRPO
description: How we built an OpenEnv-compatible RL environment that teaches language models real-world energy management.
---

# GridMind-RL: Training LLMs to Manage Industrial Buildings

*OpenEnv Hackathon India 2026 · Aditya Suryavanshi, Shreeshant Bokade, Prajwal Valekar*

---

Industrial buildings consume 40% of global electricity, yet most are managed by static, inefficient schedules. GridMind-RL bridges the gap between LLM reasoning and real-world action, training agents to manage energy costs, comfort, and grid stability under pressure. Our results show a trained agent beating hand-crafted heuristics by **58%** on complex, instruction-driven tasks.

## Why it Matters
We built GridMind-RL for the **Meta PyTorch OpenEnv Hackathon Grand Finale**. It directly addresses:
- **Theme 1 (Multi-Agent):** Buildings share a grid feeder; actions in one affect stress for all.
- **Theme 3 (World Modeling):** Agents use a `/simulate` endpoint to "think ahead" before acting.

## The Environment
GridMind-RL is an OpenEnv-compatible Go server simulating a 24-hour industrial building cycle (96 steps). Agents must navigate volatile prices, equipment faults, and grid stress signals.

### Actions & Rewards
| Action | Purpose |
|--------|---------|
| `hvac_power_level` | Climate control |
| `thermal_charge_rate` | Energy storage arbitrage |
| `batch_job_slot` | Industrial load scheduling |
| `load_shed_fraction` | Demand response |

**The Reward System:** A 9-component verifiable reward (no LLM judge) balancing cost (28%), carbon (20%), comfort (20%), grid response (20%), and task satisfaction.

## Training & Results
We trained **Qwen2.5-1.5B-Instruct** using **GRPO via TRL** with QLoRA (4-bit) on a T4 GPU.

### The Results
| Policy | Task 1 (Cost) | Task 4 (Instruction) |
|--------|--------|------------------|
| Heuristic Baseline | 0.54 | 0.31 |
| **GRPO Fine-tuned** | 0.42 | **0.49 (+58%)** |

While heuristics are strong on simple scheduling, the **GRPO-trained model dominates Task 4**, proving LLMs can parse complex natural language objectives and adapt strategies mid-episode—a feat impossible for fixed schedules.

![Reward Curve](curves/train%204/reward_curve.png)
*Learning progress: Reward climbed from −0.47 to +0.61 in just 60 steps, with the curve still rising at termination.*

## Future Directions
- **Scaling:** Moving to 7B+ models for deeper planning.
- **Fleet Coordination:** Orchestrating building clusters to avoid feeder trips.
- **Real-world Bridge:** Deploying the physics-grounded simulator to live building management systems.

## Try It
```bash
# Start a Task 4 episode (instruction following)
curl -X POST https://prajwal782007-gridmind.hf.space/reset -d '{"task_id": 4}'

# Take an action and observe the reward
curl -X POST https://prajwal782007-gridmind.hf.space/step -d '{"hvac_power_level": 0.6, "building_id": 0}'

# Grade the full episode
curl https://prajwal782007-gridmind.hf.space/grade
```

- 🤗 **Environment**: [HF Space](https://prajwal782007-gridmind.hf.space)
- 📓 **Training**: [Colab Notebook](https://colab.research.google.com/github/LO-Kyu/gridmind/blob/main/scripts/gridmind_grpo_colab.ipynb)
- 🐙 **Code**: [GitHub Repository](https://github.com/LO-Kyu/gridmind)

---
*Built for the Meta PyTorch OpenEnv Hackathon · Grand Finale · Bangalore 2026*