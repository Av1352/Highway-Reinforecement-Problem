# 🚗 Highway Reinforcement Learning Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Industry-ready reinforcement learning implementations for autonomous highway navigation**  
> MS in AI Portfolio Project | Northeastern University

---

## 🎯 **Live Interactive Demo**

### **👉 [Try the Streamlit App Now!](https://highway-reinforecement-problem.streamlit.app/) 👈**

Explore trained RL agents, view training curves, and watch demo videos in an interactive web interface!

[![Streamlit App](https://img.shields.io/badge/🚀_Launch-Streamlit_App-FF4B4B?style=for-the-badge)](https://highway-reinforecement-problem.streamlit.app/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Live Demo](#-live-interactive-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agents](#agents)
- [Results](#results)
- [Technologies](#technologies)
- [Contact](#contact)

---

## Overview

This project implements and compares three state-of-the-art reinforcement learning algorithms for autonomous driving in highway scenarios:

- **Rainbow DQN** - Advanced Deep Q-Network with prioritized experience replay
- **A3C** - Asynchronous Advantage Actor-Critic with parallel training
- **Decision Transformer** - Transformer-based offline RL approach

All agents are trained on the `highway-env` gymnasium environment with GPU acceleration and feature modular, production-ready code.

**🌐 [View Live Demo](https://highway-reinforecement-problem.streamlit.app/)**

---

## Features

✨ **Three Advanced RL Algorithms**
- Rainbow DQN with dueling architecture
- A3C with parallel workers
- Decision Transformer for offline learning

🚀 **Production-Ready Code**
- Modular agent implementations
- GPU-accelerated training (CUDA)
- Model checkpointing and evaluation
- Comprehensive logging and metrics

📊 **Interactive Visualization**
- **[Live Streamlit web app](https://highway-reinforecement-problem.streamlit.app/)** for demos
- Training curve plotting
- MP4 video generation
- Real-time performance metrics

🎓 **Educational & Portfolio-Focused**
- Clean, documented code
- Industry best practices
- Recruiter-friendly presentation

---

## Project Structure

```bash
Highway-Reinforcement-Problem/
│
├── README.md # This file
├── requirements.txt # Python dependencies
├── streamlit_app.py # 🎯 Interactive demo app
├── train_and_demo_agents.py # Training & demo generation
│
├── agents/
│ ├── rainbow_agent.py # Rainbow DQN implementation
│ ├── a3c_agent.py # A3C implementation
│ └── decision_transformer/
│ └── dt_agent.py # Decision Transformer
│
├── environments/
│ └── setup_env.py # Environment configuration
│
├── utils/
│ ├── visualizations.py # Plotting utilities
│ ├── training.py # Training helpers
│ └── unzip.py # Data utilities
│
├── demo/
│ └── sample_results/ # Training curves, videos, CSVs
│ ├── rainbow_demo.mp4
│ ├── a3c_demo.mp4
│ ├── dt_demo.gif
│ └── *.png, *.csv
│
├── notebooks/
│ └── Train.ipynb # Jupyter experiments
│
└── docs/
└── architecture.png # Architecture diagrams
```


---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional, for faster training)
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Av1352/Highway-Reinforecement-Problem.git
cd Highway-Reinforcement-Problem
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```


---

## Quick Start

### 1. View the Live Demo
**👉 [Launch Streamlit App](https://highway-reinforecement-problem.streamlit.app/) - No installation required!**

### 2. Run Locally

**Train Agents & Generate Demos:**

```bash
python train_and_demo_agents.py
```

This will:
- Train Rainbow DQN and A3C agents (200 episodes each)
- Save model weights
- Generate training curves
- Create demo videos (MP4 format)

**Launch Local Streamlit App:**

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

### 3. View Results
Check `demo/sample_results/` for:
- `rainbow_demo.mp4` - Trained Rainbow DQN agent demo
- `a3c_demo.mp4` - Trained A3C agent demo
- `*.png` - Training reward curves
- `*.csv` - Reward history data

---

## Agents

### 🌈 Rainbow DQN
Combines six improvements to DQN:
- Double Q-Learning
- Prioritized Experience Replay
- Dueling Networks
- Multi-step Learning
- Distributional RL
- Noisy Networks

**Key Features:**
- State dim: 25 (5×5 kinematics)
- Action space: 5 discrete actions
- Replay buffer: 100k transitions

### 🎭 A3C (Asynchronous Advantage Actor-Critic)
Policy gradient method with parallel workers:
- Actor-Critic architecture
- Advantage function estimation
- Asynchronous training

**Key Features:**
- Shared global model
- Local worker agents
- Policy and value optimization

### 🤖 Decision Transformer
Treats RL as sequence modeling:
- Transformer architecture (GPT-style)
- Offline learning from trajectories
- Return-conditioned generation

**Key Features:**
- Context length: 10
- Hidden dim: 192
- 4 transformer layers

---

## Results

### Training Performance

| Agent | Avg Reward (Final) | Max Reward | Episodes |
|-------|-------------------|------------|----------|
| Rainbow DQN | 32.4 | 48.2 | 200 |
| A3C | 28.7 | 42.1 | 200 |
| Decision Transformer | 25.3 | 38.9 | N/A (offline) |

**📊 [View Interactive Results on Streamlit](https://highway-reinforecement-problem.streamlit.app/)**

---

## Technologies

**Core Stack:**
- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8)
- Gymnasium (highway-env)
- NumPy, Matplotlib

**Visualization:**
- **Streamlit** (Interactive web app) - **[Live Demo](https://highway-reinforecement-problem.streamlit.app/)**
- imageio (Video generation)
- Matplotlib (Plotting)

**Training:**
- CUDA GPU acceleration
- Multiprocessing (A3C)
- Model checkpointing

---

## Contact

**Anju V**  
MS in AI @ Northeastern University

- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💼 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- 🌐 Portfolio: [vxanju.com](https://vxanju.com)
- 📄 Resume: [Download PDF](your-resume-link)
- 🚀 **Live Demo:** [Streamlit App](https://highway-reinforecement-problem.streamlit.app/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- highway-env by Edouard Leurent
- OpenAI Gymnasium
- PyTorch Team
- Streamlit Community

---

<div align="center">
  <h3>🚀 <a href="https://highway-reinforecement-problem.streamlit.app/">Try the Live Demo Now!</a></h3>
  <p><strong>⭐ If you found this project helpful, please star the repository!</strong></p>
  <p>Made with ❤️ for recruiters and the RL community</p>
</div>
