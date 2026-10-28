"""Tiny GridWorld for WM validation. No dependencies beyond numpy."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np

ACTIONS = ["up", "down", "left", "right"]
ACT2DELTA = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent = np.array([0, 0])
        self.goal = np.array([self.size-1, self.size-1])
        return self.obs()

    def obs(self):
        """Return 3-channel grid: 0=empty, 1=agent, 2=goal. Shape (3,H,W) flattened."""
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        grid[0, :, :] = 1.0  # empty bias
        grid[1, self.agent[0], self.agent[1]] = 1.0
        grid[2, self.goal[0], self.goal[1]] = 1.0
        # also flat vector for MLP
        return grid.reshape(-1)  # (3*H*W,)

    def step(self, act: int):
        delta = np.array(ACT2DELTA[act])
        nxt = self.agent + delta
        nxt = np.clip(nxt, 0, self.size-1)
        # simple wall: no movement if out of bounds (clipped)
        self.agent = nxt
        done = np.array_equal(self.agent, self.goal)
        reward = 1.0 if done else -0.01
        return self.obs(), reward, done, {}

    def render_ascii(self):
        g = np.full((self.size, self.size), ".", dtype=object)
        g[self.goal[0], self.goal[1]] = "G"
        g[self.agent[0], self.agent[1]] = "A"
        return "\n".join(" ".join(row) for row in g)
