"""Numba-jitted Color Lines engine with a seedable, CRN-friendly RNG.

Spawn randomness is derived from (game_seed, spawn_event_idx), so two strategies
using the same seed share the same spawn draws. The spawn and clearing rules
match the original environment.
"""
import numpy as np
from numba import njit

SIZE = 9
NCELLS = SIZE * SIZE


# ------------------------- splitmix64 PRNG -------------------------

@njit
def _sm64(state):
    state = np.uint64(state)
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    z = z ^ (z >> np.uint64(31))
    return state, z


@njit
def _event_state(game_seed, event_idx):
    game_seed = np.uint64(game_seed)
    event_idx = np.uint64(event_idx)
    s = game_seed + np.uint64(0x9E3779B97F4A7C15) * (event_idx + np.uint64(1))
    _, z = _sm64(s)
    return z


@njit
def _u64_to_f64(z):
    lo = np.uint64(z) & np.uint64(0xFFFFFFFF)
    return np.float64(lo) / np.float64(4294967296.0)


# ------------------------- game core -------------------------

@njit
def _spawn(board, k, state):
    """Spawn k balls on `board` in place."""
    state = np.uint64(state)
    idx = np.zeros(NCELLS, dtype=np.int32)
    n_empty = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r, c] == 0:
                idx[n_empty] = r * SIZE + c
                n_empty += 1
    if k > n_empty:
        k = n_empty

    for t in range(k):
        state, z = _sm64(state)
        m = n_empty - t
        j = t + int(_u64_to_f64(z) * m)
        if j >= n_empty:
            j = n_empty - 1
        tmp = idx[t]
        idx[t] = idx[j]
        idx[j] = tmp

    for t in range(k):
        state, z = _sm64(state)
        color = int(_u64_to_f64(z) * 7) + 1
        if color > 7:
            color = 7
        board[idx[t] // SIZE, idx[t] % SIZE] = color


@njit
def _clear_lines(board):
    """Clear every >=5 same-color line in place; return the number of cleared cells."""
    to_clear = np.zeros((SIZE, SIZE), dtype=np.bool_)
    for r in range(SIZE):
        for c in range(SIZE):
            color = board[r, c]
            if color == 0:
                continue
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                length = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr, nc] == color:
                    length += 1
                    nr += dr
                    nc += dc
                if length >= 5:
                    for i in range(length):
                        to_clear[r + i * dr, c + i * dc] = True

    n = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if to_clear[r, c]:
                board[r, c] = 0
                n += 1
    return n


@njit
def _reachable_empty(board, sr, sc):
    """Mask of empty cells reachable from a ball at (sr, sc)."""
    mask = np.zeros(NCELLS, dtype=np.bool_)
    visited = np.zeros((SIZE, SIZE), dtype=np.bool_)
    stack = np.zeros(NCELLS, dtype=np.int32)
    top = 1
    stack[0] = sr * SIZE + sc
    visited[sr, sc] = True

    while top > 0:
        top -= 1
        p = stack[top]
        r = p // SIZE
        c = p % SIZE
        if r > 0 and not visited[r - 1, c] and board[r - 1, c] == 0:
            visited[r - 1, c] = True
            mask[(r - 1) * SIZE + c] = True
            stack[top] = (r - 1) * SIZE + c
            top += 1
        if r < SIZE - 1 and not visited[r + 1, c] and board[r + 1, c] == 0:
            visited[r + 1, c] = True
            mask[(r + 1) * SIZE + c] = True
            stack[top] = (r + 1) * SIZE + c
            top += 1
        if c > 0 and not visited[r, c - 1] and board[r, c - 1] == 0:
            visited[r, c - 1] = True
            mask[r * SIZE + c - 1] = True
            stack[top] = r * SIZE + c - 1
            top += 1
        if c < SIZE - 1 and not visited[r, c + 1] and board[r, c + 1] == 0:
            visited[r, c + 1] = True
            mask[r * SIZE + c + 1] = True
            stack[top] = r * SIZE + c + 1
            top += 1

    return mask


@njit
def _no_moves(board):
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r, c] != 0:
                if ((r > 0 and board[r - 1, c] == 0)
                        or (r < SIZE - 1 and board[r + 1, c] == 0)
                        or (c > 0 and board[r, c - 1] == 0)
                        or (c < SIZE - 1 and board[r, c + 1] == 0)):
                    return False
    return True


@njit
def reset_core(board, game_seed):
    board.fill(0)
    _spawn(board, 5, _event_state(game_seed, np.uint64(0)))


@njit
def step_core(board, src, dst, game_seed, spawn_idx):
    """Apply move src->dst; return (score_delta, done, spawned, ok)."""
    sr, sc = src // SIZE, src % SIZE
    dr, dc = dst // SIZE, dst % SIZE
    if board[sr, sc] == 0 or board[dr, dc] != 0:
        return 0, True, False, False

    reach = _reachable_empty(board, sr, sc)
    if not reach[dr * SIZE + dc]:
        return 0, True, False, False

    color = board[sr, sc]
    board[sr, sc] = 0
    board[dr, dc] = color
    cleared = _clear_lines(board)

    delta = 0
    spawned = False
    if cleared >= 5:
        delta += cleared * 2
    else:
        _spawn(board, 3, _event_state(game_seed, np.uint64(spawn_idx)))
        spawned = True
        extra = _clear_lines(board)
        if extra >= 5:
            delta += extra * 2

    return delta, _no_moves(board), spawned, True


# ------------------------- Python wrapper -------------------------

class Env:
    size = SIZE
    colors = 7

    def __init__(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.int64)
        self.score = 0
        self._seed = None
        self._spawn_idx = 1

    def reset(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(1, 2 ** 63 - 1))
        self._seed = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
        reset_core(self.board, self._seed)
        self.score = 0
        self._spawn_idx = 1
        return self.board, None

    def step(self, action):
        if self._seed is None:
            self._seed = np.uint64(int(np.random.randint(1, 2 ** 63 - 1))
                                   & 0xFFFFFFFFFFFFFFFF)
        src, dst = action // 81, action % 81
        delta, done, spawned, ok = step_core(
            self.board, np.int64(src), np.int64(dst), self._seed,
            np.uint64(self._spawn_idx))
        if not ok:
            raise ValueError(f"invalid move: src={src} dst={dst}")
        self.score += delta
        if spawned:
            self._spawn_idx += 1
        return self.board, done, None

    def get_valid_dests_for_source(self, src_idx):
        r, c = divmod(int(src_idx), SIZE)
        if self.board[r, c] == 0:
            return np.zeros(NCELLS, dtype=bool)
        m = _reachable_empty(self.board, r, c)
        m[r * SIZE + c] = False
        return m
