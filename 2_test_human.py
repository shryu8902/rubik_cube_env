#%%
import gymnasium as gym
from rubik_envs.rubikscube333 import RubiksCubeEnv
import matplotlib.pyplot as plt

#%%
env = RubiksCubeEnv(render_mode='human')
observation = env.reset(l_scramble=0)

# Define valid actions explicitly
valid_actions = ["F", "F'", "B", "B'", "R", "R'", "L", "L'", "U", "U'", "D", "D'"]
action_to_int = {move: idx for idx, move in enumerate(valid_actions)}

print("Rubik's Cube Environment")
print("Enter actions such as 'F', 'F'', 'B', 'B'', 'R', 'R'', 'L', 'L'', 'U', 'U'', 'D', 'D''")
print("Type 'exit' to quit the simulation.")

while True:
    # Get user actions 
    user_action = input("Enter your action: ").strip()
    
    if user_action.lower() == 'exit':
        print("Exiting the simulation. Goodbye!")
        break  # quit program

    # valid action
    if user_action not in valid_actions:
        print("Invalid action. Please enter a valid action (e.g., 'F', 'F'').")
        continue  # ignoring invalid actions
    
    # interaction to the environment

    observation, reward, terminated, truncated, info = env.step(action_to_int[user_action])
    
    # print information
    print(f"Reward: {reward}, Step Count: {info['count']}")
    
    if terminated:
        print("Congratulations! You've solved the cube!")
        break  # Finish
    
    if truncated:
        print("Step limit reached. Resetting the cube.")
        observation = env.reset(l_scramble=10)

env.close()

