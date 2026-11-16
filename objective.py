from numba import njit

# State-function weights used by the policy helpers.
W_LOOKAHEAD = 3.4349883916159287   # window potential weight
W_RUN = 12.793657032881827         # all-runs score
W_ONEAWAY = 133.7480609952844      # spawn-completion value
W_EMPTY = 1886.3703620550432       # empty cells
W_CENTER = 1.2621526352367656      # Manhattan distance from center
W_COMP = 2790192.902035277         # completion bonus


@njit
def _score_line(board, r0, c0, dr, dc, L):
    score = 0
    i = 0
    while i < L:
        v = board[r0 + i * dr, c0 + i * dc]
        if v == 0:
            i += 1
            continue
        j = i
        while j + 1 < L and board[r0 + (j + 1) * dr, c0 + (j + 1) * dc] == v:
            j += 1
        run = j - i + 1
        if run >= 2:
            if run >= 5:
                score += 1000
            elif run == 4:
                score += 500
            elif run == 3:
                score += 200
            else:
                score += 50
        i = j + 1
    return score


@njit
def run_score_all(board, size):
    """Sum line_score over every maximal same-color run >= 2."""
    total = 0
    for r in range(size):
        total += _score_line(board, r, 0, 0, 1, size)
    for c in range(size):
        total += _score_line(board, 0, c, 1, 0, size)

    for r0 in range(size):
        for c0 in range(size):
            if not (r0 == 0 or c0 == 0):
                continue
            L = min(size - r0, size - c0)
            if L >= 2:
                total += _score_line(board, r0, c0, 1, 1, L)

    for r0 in range(size):
        for c0 in range(size):
            if not (r0 == 0 or c0 == size - 1):
                continue
            L = min(size - r0, c0 + 1)
            if L >= 2:
                total += _score_line(board, r0, c0, 1, -1, L)

    return total
