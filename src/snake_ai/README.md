# Snake AI with Deep Q-Learning

This project implements a Snake game with an AI agent trained using Deep Q-Learning. The AI learns to play the game by interacting with the environment, receiving rewards for eating food, avoiding collisions, and maximizing survival.

## How It Works

### State Representation (Vision)

The AI agent uses a compact state representation to decide its next move. Each state includes:

1. **Immediate danger**: Binary indicators for whether moving forward, left, or right would result in a collision with the wall or the snake’s own body.
2. **Food proximity**: Binary flags for whether food is immediately in front, to the left, or to the right, and distance-based features that capture how far the food is in each direction (normalized to the board size).
3. **Reachable space**: Using a breadth-first search flood-fill, the agent estimates the number of free squares it can reach in each direction without colliding. This helps avoid getting trapped in dead-ends.
4. **Snake length**: Encoded as a normalized exponential function of its ratio to the map size, which allows the agent to adopt different strategies depending on the stage of the game.

This results in a state vector of length **14**, combining immediate danger, food information, distance-to-food, length, and directional BFS scores.

---

### Reward Function

The agent receives feedback at each step according to the following rules:

- **Eating food**: +10 reward for successfully reaching the food.
- **Collision**: -100 penalty for hitting the wall or itself (game over).
- **Survival bonus**: +0.1 reward per time step to encourage the snake to keep moving.
- **Distance shaping**: Small reward (+1) if the snake moves closer to the food, small penalty (-0.5) if it moves away.
- **Space exploration**: Additional reward proportional to the amount of reachable space around the snake to discourage moves that lead to dead-ends.

This reward design encourages the agent to **balance aggressive food-seeking with safe exploration**, improving long-term survival.

---

### Neural Network Architecture

The Q-network approximates the Q-values for the three possible actions: `[straight, right turn, left turn]`. The architecture is as follows:

- **Input layer**: 14 neurons corresponding to the state vector.
- **Hidden layers**:
  - Dense 512 neurons
  - Dense 256 neurons
  - Dense 128 neurons
- **Output layer**: 3 neurons, representing the Q-values for each action.

The network is trained using **Mean Squared Error** loss and **Adam optimizer** with a learning rate of 0.0001. Training uses both short-term updates per step and long-term replay from a memory buffer of up to 100,000 experiences.

This combination of a structured state, a shaped reward function, and a sufficiently deep network enables the agent to learn a robust policy for playing Snake efficiently.


## Installation


```bash
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install numpy pygame tensorflow tensorflowjs matplotlib IPython
python train.py
```

[Inspired by Patrick Loeber’s Deep Q-Learning tutorial series](https://www.youtube.com/watch?v=PJl4iabBEz0&list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV&index=1)

