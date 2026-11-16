"""Color Lines environment exposing the next three spawn colors.

The game is identical to engine.Env; `next3` additionally exposes the
colors of the pending spawn event (positions stay hidden).
"""
import numpy as np
from numba import njit

from engine import (SIZE, NCELLS, Env, _sm64, _event_state, _u64_to_f64)


@njit
def next3_colors(board, game_seed, event_idx):
    """Colors of the k = min(3, n_empty) balls spawned at event `event_idx`,
    zero-padded to length 3."""
    out = np.zeros(3, dtype=np.int64)
    n_empty = 0
    for p in range(NCELLS):
        if board[p // SIZE, p % SIZE] == 0:
            n_empty += 1
    k = 3 if n_empty >= 3 else n_empty

    state = _event_state(game_seed, np.uint64(event_idx))
    for t in range(k):
        state, z = _sm64(state)          # position draw (discarded)
    for t in range(k):
        state, z = _sm64(state)          # color draw
        color = int(_u64_to_f64(z) * 7) + 1
        if color > 7:
            color = 7
        out[t] = color
    return out


class EnvNext3(Env):
    def __init__(self):
        super().__init__()
        self.next3 = np.zeros(3, dtype=np.int64)

    def reset(self, seed=None):
        board, _ = super().reset(seed)
        self.next3 = next3_colors(self.board, self._seed,
                                  np.uint64(self._spawn_idx))
        return board, None

    def step(self, action):
        board, done, mask = super().step(action)
        self.next3 = next3_colors(self.board, self._seed,
                                  np.uint64(self._spawn_idx))
        return board, done, mask
