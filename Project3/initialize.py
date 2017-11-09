import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math

import reinforcement_learning as rl

# Training

env_name = 'Breakout-v0'
#env_name = 'SpaceInvaders-v0'
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'
rl.update_paths(env_name=env_name)
rl.maybe_download_checkpoint(env_name=env_name)
agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)
model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=1)

# Testing

agent.epsilon_greedy.epsilon_testing
agent.training = False
agent.reset_episode_rewards()
agent.render = True
agent.run(num_episodes=1)
