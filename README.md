# GridMind-RL 🏢⚡🤖

**An AI-powered energy management simulator** - Watch an AI agent learn to control building energy systems using real-time electricity prices, temperature control, and grid demands.

> **New to AI or coding?** No problem! This guide will get you running in 10 minutes.

---

## 🚀 Quick Start (3 Steps)

1. **Get a free AI API key** from [Hugging Face](https://huggingface.co/join) (takes 2 minutes)
2. **Run the simulator**: `docker run -p 7860:7860 -p 7861:7861 ghcr.io/your-repo/gridmind-rl:latest`
3. **Watch the AI learn**: `python inference.py --episodes 1` (or `--fast-mode` for a quick heuristic run, no API calls)

That's it! The AI will start making energy decisions and you'll see live results.

---

## 📖 What is GridMind-RL?

Imagine you're managing a commercial building's energy use. Electricity costs change every 15 minutes, the weather fluctuates, and the power grid sometimes needs help. Your job? Keep the building comfortable while saving money and helping the grid.

**GridMind-RL** is a computer simulation where an AI "brain" (like ChatGPT) learns to make these decisions. It controls:
- 🏭 HVAC cooling/heating
- 🔋 Thermal energy storage
- ⏰ Batch process scheduling
- ⚡ Load shedding during grid emergencies

The AI learns through trial and error, getting "rewards" for good decisions (saving money, staying comfortable) and "penalties" for bad ones (wasting energy, uncomfortable temperatures).

---

## 🛠️ Setup Guide

### Prerequisites (What You Need First)

- **🐳 Docker** - Download from [docker.com](https://www.docker.com/products/docker-desktop) (free)
- **🐍 Python 3.9+** - Download from [python.org](https://www.python.org/downloads/) (free)
- **🔑 Hugging Face API Key** - Free account at [huggingface.co](https://huggingface.co/join)

### Step 1: Get Your Free AI API Key

1. Go to [https://huggingface.co/join](https://huggingface.co/join) and create a free account
2. Click your profile → Settings → Access Tokens
3. Click "New token", name it `gridmind`, select "Read" role
4. Copy the token (starts with `hf_...`)

**This is free!** No credit card needed.

### Step 2: Download and Run the Simulator

#### Option A: Docker (Easiest - Recommended)

First, build the simulator:
```bash
docker build -t gridmind-rl .
```

Then run it:
```bash
docker run -p 7860:7860 -p 7861:7861 gridmind-rl
```

The simulator starts on:
- **API Server**: http://localhost:7860 (for the AI)
- **Live Dashboard**: http://localhost:7861 (watch in your browser!)

#### Option B: Manual Setup (If Docker Doesn't Work)

**Install Go** (for the simulator):
- Download from [go.dev/dl](https://go.dev/dl/)
- Install and restart your terminal

**Run the simulator**:
```bash
# Start the energy environment
go run main.go
```

**On Windows** (if you have the pre-built executable):
```powershell
# Run the compiled version (faster startup)
.\grid.exe
```

**Install Python tools**:
```bash
# Install required packages
pip install -r python/requirements.txt
```

**Start the Visualization Dashboard**:
Since you're running manually, the visualization dashboard needs to be started in a new terminal window:
```bash
python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 7861
```

### Step 3: Configure the AI

**On Windows (PowerShell - Recommended)**:
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "hf_your_token_here"  # Paste your token here
```

**On Windows (Command Prompt)**:
```cmd
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
set HF_TOKEN=hf_your_token_here
```

**On Mac/Linux**:
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token_here
```

### Step 4: Watch the AI Learn!

```bash
# Run 3 learning episodes (takes ~5 minutes)
python inference.py --episodes 3
```

You'll see output like:
```
Episode 1/3 - Task 1 (Easy): Learning to save energy...
AI Decision: Lowering HVAC to save $2.50
Score: 0.85

Episode 2/3 - Task 2 (Medium): Balancing cost + comfort...
AI Decision: Using thermal storage during cheap hours
Score: 0.72
```

---

## 📊 What the AI Learns

The AI progresses through **3 difficulty levels**:

| Level | Challenge | What It Learns |
|-------|-----------|----------------|
| **Easy** | Save money | Basic energy cost optimization |
| **Medium** | Stay comfortable | Keep building 68-74°F (19-23°C) |
| **Hard** | Handle emergencies | Respond to grid stress + meet production deadlines |

**Scoring**: 1.0 = Perfect, 0.0 = Random guessing. Good scores are 0.6+.

---

## 🎮 Interactive Dashboard

While the AI runs, open http://localhost:7861 in your browser to see:
- 📈 Live energy usage charts
- 🌡️ Temperature trends
- 💰 Cost savings over time
- ⚡ Grid stress responses

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker: command not found` | Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop) |
| `401 Unauthorized` | Your Hugging Face token is wrong - get a new one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `Connection refused` | Make sure the simulator is running (Docker or `go run main.go`) |
| Python errors | Run `pip install -r python/requirements.txt` |
| Model not found | Some models need you to accept terms on Hugging Face first |

---

## 🧠 Technical Details

### What the AI Sees (Sensors)
- Current temperature, electricity price, grid stress level
- Battery charge level, time of day, pending work deadlines
- Running energy costs and carbon emissions

### What the AI Controls (Actions)
- HVAC power level (0-100%)
- Battery charge/discharge rate
- When to run batch processes
- How much load to shed during emergencies

### Reward System
- ✅ **Bonus**: Saving money, staying comfortable, helping the grid
- ❌ **Penalty**: Wasting energy, temperature extremes, missing deadlines

---

## 🚀 Advanced Usage

**Try different AI models**:
```powershell
$env:MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Faster but less accurate
```

**Run longer training**:
```bash
python inference.py --episodes 10 --llm-every 4  # Scale LLM calls via --llm-every; use --fast-mode for tests
```

**Test the environment manually**:
```bash
python python/validate.py --env-url http://localhost:7860
```

---

## 📚 Learn More

- **Reinforcement Learning**: How AI learns through trial and error
- **Energy Management**: Real-world smart grid technologies
- **Hugging Face**: Free platform for AI models and datasets

**Happy learning!** 🎉 The AI will surprise you with how well it learns to manage energy.
