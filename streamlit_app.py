# streamlit_app.py

import streamlit as st
import os
import numpy as np
from agents.rainbow_agent import RainbowAgent
from utils.visualizations import streamlit_plot_rewards, display_gif
from environments.setup_env import create_highway_env

# Optional imports for other agent demos
# from agents.a3c_agent import A3CAgent
# from agents.decision_transformer.dt_agent import DecisionTransformer

st.set_page_config(page_title="Highway RL Portfolio Demo", layout="centered")

st.title("🚗 Highway RL Portfolio Demo")
st.markdown(
    """
    Welcome to my industry-ready RL project demo!  
    - Choose an agent 👇  
    - Run a quick demo or load sample results  
    - Visualize agent performance & download plots  
    - See my explanations and portfolio links below
    """
)

st.sidebar.header("Agent Selection")
agent_name = st.sidebar.selectbox(
    "Choose RL Agent",
    ["Rainbow DQN", "A3C (demo only)", "Decision Transformer (demo only)"]
)

show_demo = st.sidebar.button("Run Demo Training (Fast)")

st.sidebar.markdown("---")
st.sidebar.header("Show Sample Results")
sample_result = st.sidebar.selectbox(
    "Type",
    ["Rainbow", "Decision Transformer"]
)

if st.sidebar.button("Load Sample Curve"):
    if sample_result == "Rainbow":
        curve_path = "demo/sample_results/rainbow_learning_curves.png"
    else:
        curve_path = "demo/sample_results/dt_learning_curve.png"
    st.image(curve_path, caption=f"{sample_result} learning curve")

if st.sidebar.button("Show Demo GIF"):
    demo_gif_path = "demo/sample_results/demo_gif.gif"
    display_gif(demo_gif_path, "Sample Highway Episode")

st.markdown("## Results")
if show_demo:
    # Only run a FAST demo (10 episodes) due to local compute; full train on notebook/colab
    env = create_highway_env()
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    agent = RainbowAgent(state_dim, action_dim)
    st.write("Running Rainbow DQN demo training (10 episodes)...")
    rewards = []
    for episode in range(10):
        s, _ = env.reset()
        s = s.flatten()
        total_reward = 0
        for step in range(100):
            action = agent.act(s)
            next_s, r, done, truncated, _ = env.step(action)
            next_s = next_s.flatten()
            agent.memory.push(s, action, r, next_s, done)
            agent.learn()
            s = next_s
            total_reward += r
            if done or truncated:
                break
        agent.update_target()
        rewards.append(total_reward)
    streamlit_plot_rewards(rewards, "Rainbow DQN Demo Reward Curve")
    st.success("Demo finished! Check the plot above. Full results and GIFs available in the sidebar.")

st.markdown("---")
st.markdown("### Project Highlights")
st.markdown(
    """
    - **Agents included:** Rainbow DQN, A3C, Decision Transformer
    - **Environment:** highway-env from gymnasium
    - Modular code (agents, utils, demos)
    - Fast demo and sample results for instant recruiter feedback
    """
)
st.markdown("---")
st.markdown("### Portfolio Links")
st.markdown("- [LinkedIn](your-link-here)")
st.markdown("- [Resume (PDF)](your-link-here)")

st.markdown("---")
st.markdown("**Want to reproduce or run full training?** See the README and notebooks for step-by-step instructions. Full results, GIFs, and detailed metrics are available in `demo/sample_results/`.")

st.markdown("Made with ❤️ by Anju Vilashni Nandhakumar")
