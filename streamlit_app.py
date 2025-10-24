# streamlit_app.py

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Highway RL Portfolio Demo", layout="wide")

# Title and intro
st.title("🚗 Highway Reinforcement Learning Portfolio")
st.markdown("""
Welcome to my **industry-ready RL project demo**!  
Select an agent from the sidebar to view training curves and demo videos.
""")

# Sidebar for agent selection
st.sidebar.header("🎯 Agent Selection")
agent_name = st.sidebar.selectbox(
    "Choose RL Agent:",
    ["Rainbow DQN", "A3C", "Decision Transformer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This Project")
st.sidebar.markdown("""
- **Environment:** highway-env (gymnasium)
- **Agents:** Rainbow DQN, A3C, Decision Transformer
- **Tech Stack:** PyTorch, Streamlit, Python
- **Training:** GPU-accelerated on CUDA
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Portfolio Links")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-profile)")
st.sidebar.markdown("[GitHub Repo](https://github.com/Av1352/Highway-Reinforecement-Problem)")
st.sidebar.markdown("[Resume (PDF)](your-resume-link)")

# Main content area
st.header(f"Agent: {agent_name}")

# Define file paths based on agent selection
if agent_name == "Rainbow DQN":
    reward_file = "demo/sample_results/rainbow/rainbow_rewards.csv"
    curve_img = "demo/sample_results/rainbow/rainbow_learning_curves.png"
    demo_video = "demo/sample_results/rainbow/rainbow_demo.gif"
    description = """
    **Rainbow DQN** combines multiple improvements to Deep Q-Networks:
    - Prioritized Experience Replay
    - Dueling Network Architecture
    - Double Q-Learning
    - Multi-step Learning
    
    This agent learns to navigate highway traffic by maximizing cumulative rewards.
    """
elif agent_name == "A3C":
    reward_file = "demo/sample_results/a3c/a3c_rewards.csv"
    curve_img = "demo/sample_results/a3c/a3c_learning_curves.png"
    demo_video = "demo/sample_results/a3c/a3c_demo.gif"
    description = """
    **A3C (Asynchronous Advantage Actor-Critic)** uses:
    - Actor-Critic architecture
    - Asynchronous parallel training
    - Policy gradient optimization
    
    This agent learns both value estimation and policy selection simultaneously.
    """
else:  # Decision Transformer
    reward_file = "demo/sample_results/dt/dt_rewards.csv"
    curve_img = "demo/sample_results/dt/dt_learning_curve.png"
    demo_video = "demo/sample_results/dt/dt_demo.gif"
    description = """
    **Decision Transformer** treats RL as a sequence modeling problem:
    - Transformer architecture
    - Offline learning from demonstrations
    - Conditioned on desired returns
    
    This agent learns to replicate expert behavior from pre-collected trajectories.
    """

st.markdown(description)
st.markdown("---")

# Display training curve
st.subheader("📈 Training Performance")
col1, col2 = st.columns([2, 1])

with col1:
    if os.path.exists(curve_img):
        st.image(curve_img, caption=f"{agent_name} Learning Curve", use_container_width=True)
    else:
        st.warning(f"Training curve not found at {curve_img}")

with col2:
    if os.path.exists(reward_file):
        rewards_df = pd.read_csv(reward_file, header=None, names=["Reward"])
        st.metric("Episodes Trained", len(rewards_df))
        st.metric("Final Avg Reward (last 10)", f"{rewards_df['Reward'].tail(10).mean():.2f}")
        st.metric("Max Reward", f"{rewards_df['Reward'].max():.2f}")
    else:
        st.info("Reward data not available")

st.markdown("---")

# Display demo video
st.subheader("🎥 Agent Demo Video")
if os.path.exists(demo_video):
    if demo_video.endswith('.mp4'):
        st.video(demo_video)
    elif demo_video.endswith('.gif'):
        st.image(demo_video, caption=f"{agent_name} Demo", use_container_width=True)
else:
    st.warning(f"Demo video not found at {demo_video}")
    st.info("Run `python train_and_demo_agents.py` to generate demo videos.")

st.markdown("---")

# Additional details
with st.expander("🔧 Implementation Details"):
    st.markdown(f"""
    ### {agent_name} Implementation
    
    **Key Features:**
    - Modular, production-ready code structure
    - GPU-accelerated training (CUDA support)
    - Configurable hyperparameters
    - Model checkpointing and evaluation
    
    **File Structure:**
    - `agents/{agent_name.lower().replace(' ', '_')}_agent.py` - Agent implementation
    - `utils/training.py` - Training utilities
    - `utils/visualizations.py` - Plotting and analysis
    - `environments/setup_env.py` - Environment configuration
    
    **Training Details:**
    - Episodes: 200 (configurable)
    - Learning rate: 1e-4
    - Gamma (discount): 0.99
    - Device: CUDA GPU (if available)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ by Anju V | MS in AI @ Northeastern University</p>
    <p>This project showcases modern reinforcement learning techniques for autonomous driving scenarios.</p>
</div>
""", unsafe_allow_html=True)
