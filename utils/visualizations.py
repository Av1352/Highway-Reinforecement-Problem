# utils/visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_rewards(reward_list, title="Reward Curve", save_path=None, show=True):
    plt.figure(figsize=(8, 4))
    plt.plot(reward_list, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def streamlit_plot_rewards(reward_list, title="Reward Curve"):
    st.subheader(title)
    st.line_chart(reward_list)

def plot_metrics(metrics_dict, save_path=None, show=True):
    plt.figure(figsize=(8,5))
    for k, v in metrics_dict.items():
        plt.plot(v, label=k)
    plt.xlabel('Episode')
    plt.ylabel('Metrics')
    plt.title("Training Metrics")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def display_gif(gif_path, caption="Demo Episode"):
    st.image(gif_path, caption=caption)

def save_rewards_to_csv(reward_list, save_path):
    np.savetxt(save_path, reward_list, delimiter=",")
