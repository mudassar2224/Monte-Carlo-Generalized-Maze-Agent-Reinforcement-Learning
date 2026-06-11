# 🧩 Monte Carlo Generalized Maze Agent — Reinforcement Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black"/>
  <img src="https://img.shields.io/badge/Algorithm-Monte%20Carlo%20Control-8A2BE2?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1Tpu8BR4oQAo-R4FeG81bveQzXJ3WAiNW?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28"/>
  </a>
</p>

> **First-Visit Monte Carlo Control** agent that learns a generalised navigation policy across thousands of randomly generated mazes — solving obstacle layouts it has **never seen during training**.

---

## 📌 The Core Challenge

Most tabular RL agents memorise a single fixed environment. When obstacles change at runtime, they fail. This project solves that by training a position-aware, obstacle-sensing agent across **100,000 diverse, BFS-verified mazes** via domain randomisation — producing a policy that generalises to any solvable 8×8 maze.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Training Episodes | 100,000 (domain randomised) |
| Final Success Rate (last 1000 eps) | **~85–92%** |
| Evaluation on Unseen Mazes | **✅ Generalises** |
| Q-table Size | 4,096 entries |
| Algorithm | First-Visit MC Control |

---

## 🧠 Architecture & Design Decisions

### Why the naive approach fails
A standard `Q[row, col, action]` table trained on a **single** maze seed memorises that specific obstacle layout. When a new maze is loaded, the agent walks straight into walls.

A position-free local state `(n, s, w, e, dr, dc)` seems like a fix but introduces **state aliasing** — 49 different grid positions map to the identical tuple, causing infinite loops at demo time.

### The fix: Position-aware local observation state

```
State = (row, col, n_blocked, s_blocked, w_blocked, e_blocked)
```

| Component | What it provides |
|---|---|
| `(row, col)` | Unique identity per cell — **no looping** |
| `(n, s, w, e)` flags | Obstacle awareness — **generalises to new mazes** |

Q-table shape: `(8, 8, 2, 2, 2, 2, 4)` = **4,096 entries** — compact yet expressive.

### Domain Randomisation

Every training episode uses a freshly generated maze (new random seed), so the agent is exposed to ~100,000 different obstacle configurations. The Q-table converges to a policy that works across the full distribution of solvable mazes.

### BFS Validation

Every generated maze is verified solvable via BFS before training on it. This guarantees the agent always receives a positive terminal reward signal — a critical requirement for Monte Carlo methods which rely on complete episodes.

---

## 🗂️ Notebook Structure

| Cell | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Imports & configuration |
| 3 | `MazeEnvironment` — BFS-validated, reseedable |
| 4 | Visualisation helpers + policy grid renderer |
| 5 | `MonteCarloAgent` — First-Visit MC Control |
| 6 | Training loop with domain randomisation |
| 7 | Training dashboard (reward, SR, length curves) |
| 8 | Save Q-table |
| 9 | Load Q-table |
| 10 | Evaluation on 15 unseen maze seeds |
| 11 | Animated live demo (3 fresh mazes) |

---

## ⚙️ Hyperparameters

```python
MAZE_ROWS       = 8
MAZE_COLS       = 8
OBSTACLE_RATIO  = 0.22
MAX_STEPS       = 400
REWARD_EXIT     = 1000
REWARD_REVISIT  = -3.0     # strong anti-loop signal
REWARD_OBSTACLE = -8
EPISODES        = 100_000
GAMMA           = 0.995
EPSILON_START   = 1.0
EPSILON_END     = 0.02
EPSILON_DECAY   = 0.99997
```

---

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tpu8BR4oQAo-R4FeG81bveQzXJ3WAiNW?usp=sharing)

Click **Runtime → Run All** — no local setup required. Training takes ~5–10 minutes on a free Colab GPU/CPU.

---

## 🏃 Run Locally

```bash
git clone https://github.com/mudassar2224/Monte-Carlo-Generalized-Maze-Agent-Reinforcement-Learning
cd Monte-Carlo-Generalized-Maze-Agent-Reinforcement-Learning
pip install numpy matplotlib tqdm
jupyter notebook Monte_Carlo_Maze_Escape_FIXED.ipynb
```

---

## 📚 Concepts Demonstrated

- **Monte Carlo Control** (First-Visit, off-policy exploration via ε-greedy)
- **Domain Randomisation** for policy generalisation
- **Reward shaping** — distance-based + anti-revisit penalties
- **BFS reachability** to filter unsolvable training instances
- **State space design** — balancing position identity vs. obstacle generalisation
- **Anti-loop inference** — runtime loop-breaking without retraining

---

## 📁 Repository Structure

```
├── Monte_Carlo_Maze_Escape_FIXED.ipynb   # Main notebook (all 11 cells)
├── mc_fixed_qtable.pkl                   # Saved Q-table (after training)
├── mc_meta.pkl                           # Training metadata
├── dashboard.png                         # Training curves
├── eval_result.png                       # Evaluation snapshot
├── policy_map.png                        # Greedy policy visualisation
└── README.md
```

---

## 👤 Author

**Muhammad Mudassar**
BS Artificial Intelligence — University of Management and Technology (UMT), Lahore
Semester 6 | Expected Graduation: July 2027

[![GitHub](https://img.shields.io/badge/GitHub-mudassar2224-181717?style=flat&logo=github)](https://github.com/mudassar2224)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
