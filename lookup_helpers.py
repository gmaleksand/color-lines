import numpy as np
from numba import njit
import objective as obj

SIZE = 9
NCELLS = SIZE * SIZE
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
LAM = 12.0


# ------------------------- board helpers -------------------------

@njit
def check_and_clear(board, size):
    """Clear every >=5 same-color line in place."""
    to_clear = np.zeros((size, size), dtype=np.bool_)
    for r in range(size):
        for c in range(size):
            color = board[r, c]
            if color == 0:
                continue
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                length = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == color:
                    length += 1
                    nr += dr
                    nc += dc
                if length >= 5:
                    for i in range(length):
                        to_clear[r + i * dr, c + i * dc] = True

    for r in range(size):
        for c in range(size):
            if to_clear[r, c]:
                board[r, c] = 0


@njit
def total_window_potential(board, size):
    """Sum 10*b^5 over clean length-5 windows with b >= 2 balls."""
    total = 0

    for r in range(size):
        for start in range(size - 4):
            col = 0
            clean = True
            balls = 0
            for i in range(5):
                v = board[r, start + i]
                if v != 0:
                    if col == 0:
                        col = v
                    elif col != v:
                        clean = False
                    balls += 1
            if clean and balls >= 2:
                total += 10 * (balls ** 5)

    for c in range(size):
        for start in range(size - 4):
            col = 0
            clean = True
            balls = 0
            for i in range(5):
                v = board[start + i, c]
                if v != 0:
                    if col == 0:
                        col = v
                    elif col != v:
                        clean = False
                    balls += 1
            if clean and balls >= 2:
                total += 10 * (balls ** 5)

    for r0 in range(size):
        for c0 in range(size):
            if not (r0 == 0 or c0 == 0):
                continue
            L = min(size - r0, size - c0)
            if L < 5:
                continue
            for start in range(L - 4):
                col = 0
                clean = True
                balls = 0
                for i in range(5):
                    v = board[r0 + start + i, c0 + start + i]
                    if v != 0:
                        if col == 0:
                            col = v
                        elif col != v:
                            clean = False
                        balls += 1
                if clean and balls >= 2:
                    total += 10 * (balls ** 5)

    for r0 in range(size):
        for c0 in range(size):
            if not (r0 == 0 or c0 == size - 1):
                continue
            L = min(size - r0, c0 + 1)
            if L < 5:
                continue
            for start in range(L - 4):
                col = 0
                clean = True
                balls = 0
                for i in range(5):
                    v = board[r0 + start + i, c0 - start - i]
                    if v != 0:
                        if col == 0:
                            col = v
                        elif col != v:
                            clean = False
                        balls += 1
                if clean and balls >= 2:
                    total += 10 * (balls ** 5)

    return total


@njit
def spawn_completion_value_max(board, size):
    """Total value-weighted spawn completion over all empty cells."""
    dirs4 = ((1, 0), (0, 1), (1, 1), (1, -1))
    back_len = np.zeros((4, size, size), dtype=np.int32)
    fwd_len = np.zeros((4, size, size), dtype=np.int32)

    for d in range(4):
        dr = dirs4[d][0]
        dc = dirs4[d][1]
        for r in range(size):
            for c in range(size):
                v = board[r, c]
                if v == 0:
                    continue
                pr, pc = r - dr, c - dc
                if (0 <= pr < size and 0 <= pc < size
                        and board[pr, pc] == v
                        and back_len[d, pr, pc] > 0):
                    back_len[d, r, c] = back_len[d, pr, pc] + 1
                else:
                    back_len[d, r, c] = 1

        if dr == 0:
            r_range = range(size)
        else:
            r_range = range(size - 1, -1, -1)
        if dc == 0:
            c_range = range(size)
        elif dc > 0:
            c_range = range(size - 1, -1, -1)
        else:
            c_range = range(size)

        for r in r_range:
            for c in c_range:
                v = board[r, c]
                if v == 0:
                    continue
                sr2, sc2 = r + dr, c + dc
                if (0 <= sr2 < size and 0 <= sc2 < size
                        and board[sr2, sc2] == v
                        and fwd_len[d, sr2, sc2] > 0):
                    fwd_len[d, r, c] = fwd_len[d, sr2, sc2] + 1
                else:
                    fwd_len[d, r, c] = 1

    total = 0
    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                continue
            cand = np.zeros(8, dtype=np.int32)
            ncand = 0
            for d in range(4):
                dr = dirs4[d][0]
                dc = dirs4[d][1]
                lr, lc = r - dr, c - dc
                rr, rc = r + dr, c + dc
                lcol = board[lr, lc] if 0 <= lr < size and 0 <= lc < size else 0
                rcol = board[rr, rc] if 0 <= rr < size and 0 <= rc < size else 0
                for k in range(2):
                    cc = lcol if k == 0 else rcol
                    if cc == 0:
                        continue
                    seen = False
                    for j in range(ncand):
                        if cand[j] == cc:
                            seen = True
                            break
                    if not seen:
                        cand[ncand] = cc
                        ncand += 1

            for k in range(ncand):
                cc = cand[k]
                cells = 0
                line_count = 0
                for d in range(4):
                    dr = dirs4[d][0]
                    dc = dirs4[d][1]
                    lr, lc = r - dr, c - dc
                    rr, rc = r + dr, c + dc
                    lcol = board[lr, lc] if 0 <= lr < size and 0 <= lc < size else 0
                    llen = back_len[d, lr, lc] if (0 <= lr < size and 0 <= lc < size
                                             and lcol == cc) else 0
                    rcol = board[rr, rc] if 0 <= rr < size and 0 <= rc < size else 0
                    rlen = fwd_len[d, rr, rc] if (0 <= rr < size and 0 <= rc < size
                                             and rcol == cc) else 0
                    length = 1 + llen + rlen
                    if length >= 5:
                        cells += length
                        line_count += 1

                if line_count > 1:
                    cells -= line_count - 1
                if line_count >= 1:
                    total += cells + (line_count - 1) * 4

    return total


# ------------------------- threat / feature helpers -------------------------

@njit
def _bfs_mark(board, r0, c0, seen, stack):
    """Mark empty cells reachable from (r0, c0)."""
    for i in range(NCELLS):
        seen[i] = 0
    seen[r0 * SIZE + c0] = 1
    stack[0] = r0 * SIZE + c0
    n = 1
    head = 0

    while head < n:
        idx = stack[head]
        head += 1
        r = idx // SIZE
        c = idx % SIZE
        for d in range(4):
            dr, dc = ORTH[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                continue
            if board[nr, nc] != 0:
                continue
            k = nr * SIZE + nc
            if seen[k]:
                continue
            seen[k] = 1
            stack[n] = k
            n += 1


@njit
def ld_check(board, r, c, col, rr, cc, dr, dc):
    if not (0 <= rr < SIZE and 0 <= cc < SIZE) or board[rr, cc] != col:
        return False
    k = 1
    while True:
        ar, ac = r + k * dr, c + k * dc
        if not (0 <= ar < SIZE and 0 <= ac < SIZE) or board[ar, ac] != col:
            return False
        if ar == rr and ac == cc:
            return True
        k += 1


@njit
def ld_ok(board, r, c, col, rr, cc, dr, dc):
    return (ld_check(board, r, c, col, rr, cc, dr, dc)
            or ld_check(board, r, c, col, rr, cc, -dr, -dc))


@njit
def _on_line(board, r, c, col, rr, cc):
    """Is (rr, cc) on a line that (r, c) would complete for `col`?"""
    for dr, dc in DIRS:
        if ld_ok(board, r, c, col, rr, cc, dr, dc):
            return True
    return False


def make_scratch():
    return (
        np.zeros(NCELLS, dtype=np.int32),          # seen
        np.zeros(NCELLS, dtype=np.int32),          # stack
        np.zeros((4, SIZE, SIZE), dtype=np.int32), # rl
        np.zeros(8, dtype=np.int32),               # cnt
        np.zeros((8, 4), dtype=np.int32),          # ld
    )


@njit
def threat_stats(board, cw, seen, stack, rl, ld, cnt, size):
    """Return (threat_pending, n_threat_pending, win_now, nb)."""
    for i in range(8):
        cnt[i] = 0
    for d in range(4):
        for r in range(size):
            for c in range(size):
                rl[d, r, c] = 0

    nb = 0
    for r in range(size):
        for c in range(size):
            v = board[r, c]
            if v == 0:
                continue
            nb += 1
            cnt[v] += 1

    for d in range(4):
        dr, dc = DIRS[d]
        for r in range(size):
            for c in range(size):
                v = board[r, c]
                if v == 0 or rl[d, r, c] > 0:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < size and 0 <= pc < size and board[pr, pc] == v:
                    continue

                L = 1
                rr2, rc2 = r + dr, c + dc
                while 0 <= rr2 < size and 0 <= rc2 < size and board[rr2, rc2] == v:
                    L += 1
                    rr2 += dr
                    rc2 += dc

                for i in range(L):
                    rl[d, r + i * dr, c + i * dc] = L

    threat_pending = 0.0
    n_threat_pending = 0.0
    win_now = 0.0

    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                continue

            for i in range(8):
                for d in range(4):
                    ld[i, d] = 0

            for d in range(4):
                dr, dc = DIRS[d]
                for sgn in (-1, 1):
                    pr, pc = r + sgn * dr, c + sgn * dc
                    if not (0 <= pr < size and 0 <= pc < size):
                        continue
                    col = board[pr, pc]
                    if col == 0:
                        continue
                    if ld[col, d] < 1 + rl[d, pr, pc]:
                        ld[col, d] = 1 + rl[d, pr, pc]

            reachable = -1
            for col in range(1, 8):
                lines = 0
                cells = 0
                for d in range(4):
                    if ld[col, d] >= 5:
                        lines += 1
                        cells += ld[col, d]
                if lines == 0:
                    continue

                cells -= lines - 1
                fcells = float(cells)
                if cw[col] > 0:
                    threat_pending += cw[col] * fcells
                    n_threat_pending += cw[col]

                if cnt[col] < cells:
                    continue
                if win_now != 0.0:
                    continue

                if reachable < 0:
                    _bfs_mark(board, r, c, seen, stack)
                    reachable = 1

                ok = False
                for rr2 in range(size):
                    for cc2 in range(size):
                        if board[rr2, cc2] != col or _on_line(board, r, c, col, rr2, cc2):
                            continue
                        for d in range(4):
                            dr, dc = ORTH[d]
                            nr, nc = rr2 + dr, cc2 + dc
                            if 0 <= nr < size and 0 <= nc < size and seen[nr * size + nc]:
                                ok = True
                                break
                        if ok:
                            break
                    if ok:
                        win_now = 1.0
                        break

    return threat_pending, n_threat_pending, win_now, nb


def money_full(b1, cw, scratch):
    seen, stack, rl, cnt, ld = scratch
    threat_pending, n_threat_pending, win_now, nb = threat_stats(
        b1, cw, seen, stack, rl, ld, cnt, SIZE)

    n_empty = SIZE * SIZE - nb
    p_lot = (3.0 / 7.0) * threat_pending / (5.0 * max(float(n_empty), 1.0))
    lottery = n_threat_pending / max(float(n_empty), 1.0)
    p_money = win_now + p_lot
    p_hor = win_now + lottery

    denom = 3.0 - 8.0 * p_hor
    if denom < 0.15:
        denom = 0.15
    hor = float(SIZE * SIZE - nb) / denom
    return 10.0 * p_money * hor


@njit
def state_F(b1, cleared, w_pot, w_run, w_oa, w_empty, w_center, w_comp, size):
    pot = total_window_potential(b1, size)
    runs = obj.run_score_all(b1, size)
    one_away = spawn_completion_value_max(b1, size)

    n_balls = 0
    csum = 0.0
    for r in range(size):
        for c in range(size):
            col = b1[r, c]
            if col == 0:
                continue
            n_balls += 1
            csum += abs(2 * r - (size - 1)) + abs(2 * c - (size - 1))

    s = (w_pot * pot
         + w_run * runs
         + w_oa * one_away
         + w_empty * (size * size - n_balls)
         + w_center * csum)

    if cleared >= 5:
        s += w_comp + w_comp * (cleared - 5) / 2.0
    return s


def first_stage_score(b1, cleared, cw, scratch, lam=LAM):
    s = state_F(b1, cleared, obj.W_LOOKAHEAD, obj.W_RUN, obj.W_ONEAWAY,
                    obj.W_EMPTY, obj.W_CENTER, obj.W_COMP, SIZE)
    return s + lam * money_full(b1, cw, scratch)
