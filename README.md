# Rubik's Cube Gym Environment

A custom **Rubik's Cube environment** built for reinforcement learning research.  
This environment follows the **Gymnasium** interface.

---

## ✨ Features
- Rubik’s Cube simulator with configurable scramble length
- Fully compatible with Gymnasium API (`reset`, `step`, `render`)
- Support for both 2D/array-based observation
---

## 🚀 Example

Clone the repository:
```bash
git clone https://github.com/shryu8902/rubik_cube_env.git
cd rubik_cube_env
python 2_test_human.py
```
## Requirements:

- Python 
- gymnasium (or gym)
- numpy

## ⚙️ Environment Details

- Observation Space

  Flattened Box (original shape = (6, 3, 3) >>  flattened (54,)) representing cube face colors
  
- Action Space

  Discrete(N) where each action corresponds to a face rotation (e.g., U, U', R, R', etc.)

  ' represents counter clock-wise rotation

  There are total 12 possible actions in integer corresponds to... 
  
  actions = [0(F), 1(F'), 2(B), 3(B'), 4(R), 5(R'), 6(L), 7(L'), 8(U), 9(U'), 10(D), 11(D')] 


- Reward Function

  Example: +10 when cube is solved, small penalties (-0.5) otherwise

  (You can customize reward shaping for your own learning strategy)

- Episode Termination

  Cube is solved

  Maximum number of steps (300) reached

## Running Example

```python
from rubik_envs.rubikscube333 import RubiksCubeEnv
import gymnasium as gym

env = RubiksCubeEnv(render_mode='rgb_array')

obs1, info = env.reset(l_scramble=10)

for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info= env.reset(l_scramble=10)

env.close()
```
