#!/usr/bin/env python3
"""
GridMind-RL Unsloth GRPO Training Script
----------------------------------------
Fine-tunes Qwen2.5-0.5B-Instruct using Unsloth's 4-bit LoRA and TRL's GRPOTrainer.
The environment rewards are gathered by hitting the OpenEnv HTTP server directly.
"""

import argparse
import json
import os
import re
import sys
import requests
import pandas as pd
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
from transformers import TrainerCallback

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

SYSTEM_PROMPT = """\
You are an expert industrial building energy controller.
Each turn you receive the current building state and must respond with 
ONLY a valid JSON action object.

Action format:
{"hvac_power_level": <0.0-1.0>, "thermal_charge_rate": <-1.0 to 1.0>, 
 "batch_job_slot": <0-4>, "load_shed_fraction": <0.0-0.5>, "building_id": 0}

Strategy:
- Charge storage when price < $0.08/kWh (positive thermal_charge_rate)
- Discharge storage when price > $0.15/kWh (negative thermal_charge_rate)  
- Shed load 0.3-0.5 when grid_stress_signal > 0.7
- Reduce HVAC during peak hours (8-12, 17-21)
- Keep temperature between 19-23°C"""

def make_prompt(i):
    return [{
        "role": "system", "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": f"Episode {i+1}: The building simulation is starting. "
                   "You will receive the state each step. "
                   "Output your first action as JSON now."
    }]

def reward_valid_json(completions, **kwargs):
    """Reward 0.3 for any valid JSON output."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                json.loads(match.group())
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_has_required_keys(completions, **kwargs):
    """Reward 0.3 if JSON has all 4 required action keys."""
    required = {"hvac_power_level", "thermal_charge_rate", "batch_job_slot", "load_shed_fraction"}
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                action = json.loads(match.group())
                if required.issubset(action.keys()):
                    rewards.append(0.3)
                else:
                    rewards.append(0.1)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def get_reward_env_interaction(env_url):
    """Closure to capture the target environment URL for the reward function.

    Uses a SHORT (8-step) rollout to get a more genuine episode-level reward signal.
    The grade endpoint returns the true episode score (0.0-1.0 clamped open interval),
    which is what we use as the reward — not the step-level reward.
    """
    def reward_env_interaction(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                match = re.search(r'\{.*?\}', text, re.DOTALL)
                action = json.loads(match.group()) if match else {}
                step_action = {
                    "hvac_power_level": float(max(0, min(1, action.get("hvac_power_level", 0.5)))),
                    "thermal_charge_rate": float(max(-1, min(1, action.get("thermal_charge_rate", 0.0)))),
                    "batch_job_slot": int(max(0, min(4, action.get("batch_job_slot", 0)))),
                    "load_shed_fraction": float(max(0, min(0.5, action.get("load_shed_fraction", 0.0)))),
                    "building_id": 0
                }

                reset_resp = requests.post(
                    f"{env_url}/reset",
                    json={"task_id": 2, "seed": 42},
                    timeout=30
                )
                if reset_resp.status_code != 200:
                    rewards.append(0.0)
                    continue

                step_rewards = []
                for _ in range(8):
                    step_resp = requests.post(
                        f"{env_url}/step",
                        json=[step_action],
                        timeout=30
                    )
                    if step_resp.status_code != 200:
                        step_rewards.append(0.0)
                        continue
                    result = step_resp.json()
                    if isinstance(result, list) and len(result) > 0:
                        r = float(result[0].get("reward", 0.0))
                    elif isinstance(result, dict) and "results" in result:
                        r = float(result["results"][0].get("reward", 0.0))
                    else:
                        r = 0.0
                    step_rewards.append(r)

                grade_resp = requests.get(f"{env_url}/grade", timeout=30)
                if grade_resp.status_code == 200:
                    episode_score = float(grade_resp.json().get("score", 0.5))
                    val = episode_score * 0.4
                else:
                    mean_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
                    val = (mean_step_reward + 2.0) * 0.08
                rewards.append(min(0.4, max(0.0, val)))

            except Exception as e:
                print(f"Env error: {e}", file=sys.stderr)
                rewards.append(0.0)
        return rewards
    return reward_env_interaction

class CSVLogCallback(TrainerCallback):
    """Custom callback to continuously log training metrics to a CSV file."""
    def __init__(self, output_path):
        self.output_path = output_path
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            logs_copy = logs.copy()
            logs_copy["step"] = state.global_step
            self.log_history.append(logs_copy)
            pd.DataFrame(self.log_history).to_csv(self.output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Train GridMind-RL agent with Unsloth GRPO")
    parser.add_argument("--env-url", type=str, default="http://localhost:7860", help="OpenEnv server URL")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--prompts", type=int, default=300, help="Number of training prompts")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--output-csv", type=str, default="results/training_log.csv", help="Metrics output")
    parser.add_argument("--output-dir", type=str, default="gridmind-grpo-unsloth", help="Model save dir")
    args = parser.parse_args()

    print(f"🚀 Loading model: {args.model_name}")
    max_seq_length = 512
    lora_rank = 8

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("✅ Model loaded with Unsloth 4-bit LoRA")

    dataset = Dataset.from_dict({
        "prompt": [make_prompt(i) for i in range(args.prompts)]
    })
    print(f"✅ Dataset ready: {len(dataset)} training prompts")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,  # GRPO group size
        max_prompt_length=256,
        max_completion_length=128,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=100,
        fp16=True,
        report_to="none",  # We use our CSV callback instead
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            reward_valid_json,
            reward_has_required_keys,
            get_reward_env_interaction(args.env_url),
        ],
        callbacks=[CSVLogCallback(args.output_csv)]
    )

    print("🚀 Starting GRPO training...")
    trainer.train()

    print(f"✅ Training complete! Checkpoints saved to {args.output_dir}")
    print(f"✅ Logs saved to {args.output_csv}")

if __name__ == "__main__":
    main()