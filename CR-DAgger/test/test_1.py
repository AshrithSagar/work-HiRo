"""
test_1.py
"""

import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.envs.franka_kitchen.franka_env import FrankaRobot

gym.register_envs(gymnasium_robotics)


# env = gym.make("Pusher-v5", render_mode="human")
env = gym.make(
    "FrankaKitchen-v1", tasks_to_complete=["microwave", "kettle"], render_mode="human"
)

env.reset()
env.render()

for _ in range(1000):
    action = env.action_space.sample()
    env.step(action)

env.close()
