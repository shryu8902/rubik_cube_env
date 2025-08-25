#%%
import gymnasium as gym
from envs.rubikscube333 import RubiksCubeEnv
import matplotlib.pyplot as plt
#%%
env = RubiksCubeEnv(render_mode='rgb_array')
obs1, info = env.reset(seed= 10, l_scramble=10)
for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info= env.reset(l_scramble=10)

env.close()
