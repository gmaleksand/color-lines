import numpy as np
from numba import njit

_windows = []
for _dr, _dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
    for _r in range(9):
        for _c in range(9):
            if 0 <= _r + 4 * _dr < 9 and 0 <= _c + 4 * _dc < 9:
                _windows.append([(_r + _i * _dr) * 9 + _c + _i * _dc for _i in range(5)])
WINDOWS = np.asarray(_windows, dtype=np.int16)
CELL_WINDOWS = np.full((81, 20), -1, dtype=np.int16)
N_CELL_WINDOWS = np.zeros(81, dtype=np.int16)
for _w, _cells in enumerate(WINDOWS):
    for _p in _cells:
        CELL_WINDOWS[_p, N_CELL_WINDOWS[_p]] = _w
        N_CELL_WINDOWS[_p] += 1
NEIGHBORS = np.full((81, 4), -1, dtype=np.int16)
for _p in range(81):
    _r, _c = divmod(_p, 9)
    for _j, (_dr, _dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        if 0 <= _r + _dr < 9 and 0 <= _c + _dc < 9:
            NEIGHBORS[_p, _j] = (_r + _dr) * 9 + _c + _dc

# w2,w3,w4,empty,fragmentation,trapped,supply-discount,preview,overlap-mode,
# immediate-clear,edge,connectivity-fraction,unreachable-discount

TUNED_WEIGHTS = np.array([
    1.2430000621532056, 9.89853431273412, 32.0, 1.2,
    0.1749201519459942, 0.15104473694978854,
    0.9152404662602716, 0.4570147378532552,
    1.0, 90.0, 0.0, 1.0, 0.5245071261947328,
])


@njit(cache=True)
def components(b):
    labels = np.full(81, -1, np.int16)
    sizes = np.zeros(81, np.int16)
    queue = np.empty(81, np.int16)
    nc = 0
    for p in range(81):
        if b[p] != 0 or labels[p] >= 0:
            continue
        queue[0] = p
        labels[p] = nc
        head, tail = 0, 1
        while head < tail:
            q = queue[head]
            head += 1
            for j in range(4):
                v = NEIGHBORS[q, j]
                if v >= 0 and b[v] == 0 and labels[v] < 0:
                    labels[v] = nc
                    queue[tail] = v
                    tail += 1
        sizes[nc] = tail
        nc += 1
    access = np.zeros(81, np.uint64)
    for p in range(81):
        if b[p] != 0:
            for j in range(4):
                v = NEIGHBORS[p, j]
                if v >= 0 and labels[v] >= 0:
                    access[p] |= np.uint64(1) << np.uint64(labels[v])
    return labels, sizes, access, nc


@njit(cache=True)
def window_value(counts, inventory, preview, weights):
    color = 0
    n = 0
    for c in range(1, 8):
        if counts[c] > 0:
            if color > 0:
                return 0.0
            color = c
            n = counts[c]
    if n < 2:
        return 0.0
    if n == 5:
        return 1000.0
    missing = max(0.0, 5.0 - inventory[color] - weights[7] * preview[color])
    return weights[n - 2] * weights[6] ** missing


@njit(cache=True)
def clear_after_move(b, dst):
    color = b[dst]
    if color == 0:
        return 0
    marked = np.zeros(81, np.bool_)
    r, c = dst // 9, dst % 9
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        left, right = 0, 0
        for step in range(1, 9):
            rr, cc = r - step * dr, c - step * dc
            if not (0 <= rr < 9 and 0 <= cc < 9) or b[rr * 9 + cc] != color:
                break
            left += 1
        for step in range(1, 9):
            rr, cc = r + step * dr, c + step * dc
            if not (0 <= rr < 9 and 0 <= cc < 9) or b[rr * 9 + cc] != color:
                break
            right += 1
        if left + right >= 4:
            for step in range(-left, right + 1):
                marked[(r + step * dr) * 9 + c + step * dc] = True
    n = 0
    for p in range(81):
        if marked[p]:
            b[p] = 0
            n += 1
    return n


@njit(cache=True)
def evaluate(b, preview, weights):
    labels, sizes, access, nc = components(b)
    inventory = np.zeros(8, np.int16)
    donor_counts = np.zeros((8, 81), np.int16)
    empties = 0
    trapped = 0
    edge = 0.0
    for p in range(81):
        c = b[p]
        if c == 0:
            empties += 1
        else:
            inventory[c] += 1
            if access[p] == 0:
                trapped += 1
            edge += abs(p // 9 - 4) + abs(p % 9 - 4)
            for cid in range(nc):
                if access[p] & (np.uint64(1) << np.uint64(cid)):
                    donor_counts[c, cid] += 1
    per_ball = np.zeros(81, np.float64)
    # Deduplicate alternative windows on the same maximal geometric line.
    line_best = np.zeros((4, 17, 8), np.float64)
    opportunity_sum = 0.0
    for w in range(len(WINDOWS)):
        color = 0
        count = 0
        clean = True
        for j in range(5):
            v = b[WINDOWS[w, j]]
            if v:
                if color and color != v:
                    clean = False
                    break
                color = v
                count += 1
        if not clean or count < 2:
            continue
        if count >= 5:
            continue
        missing = max(0.0, 5.0 - inventory[color] - weights[7] * preview[color])
        value = weights[count - 2] * weights[6] ** missing
        holes_without_donor = 0
        if weights[11] > 0:
            for j in range(5):
                p = WINDOWS[w, j]
                if b[p] != 0:
                    continue
                cid = labels[p]
                available = donor_counts[color, cid]
                bit = np.uint64(1) << np.uint64(cid)
                for k in range(5):
                    q = WINDOWS[w, k]
                    if b[q] == color and access[q] & bit:
                        available -= 1
                if available == 0 and inventory[color] >= 5:
                    holes_without_donor += 1
            value *= weights[12] ** holes_without_donor
        if weights[8] >= 1.5:
            # One plan value per maximal line/colour.
            p = WINDOWS[w, 0]
            d = WINDOWS[w, 1] - p
            if d == 1:
                kind, line = 0, p // 9
            elif d == 9:
                kind, line = 1, p % 9
            elif d == 10:
                kind, line = 2, p // 9 - p % 9 + 8
            else:
                kind, line = 3, p // 9 + p % 9
            if value > line_best[kind, line, color]:
                line_best[kind, line, color] = value
        elif weights[8] >= 0.5:
            for j in range(5):
                p = WINDOWS[w, j]
                if b[p] == color and value / count > per_ball[p]:
                    per_ball[p] = value / count
        else:
            opportunity_sum += value
    for p in range(81):
        opportunity_sum += per_ball[p]
    for k in range(4):
        for line in range(17):
            for c in range(1, 8):
                opportunity_sum += line_best[k, line, c]
    fragmentation = float(empties)
    if empties:
        for cid in range(nc):
            fragmentation -= sizes[cid] * sizes[cid] / float(empties)
    return (opportunity_sum + weights[3] * empties
            - weights[4] * fragmentation - weights[5] * trapped + weights[10] * edge)


@njit(cache=True)
def candidates(b, preview, weights, k):
    labels, sizes, access, nc = components(b)
    inventory = np.zeros(8, np.int16)
    for p in range(81):
        inventory[b[p]] += 1
    counts = np.zeros((len(WINDOWS), 8), np.int16)
    old = np.zeros(len(WINDOWS), np.float64)
    for w in range(len(WINDOWS)):
        for j in range(5):
            counts[w, b[WINDOWS[w, j]]] += 1
        old[w] = window_value(counts[w], inventory, preview, weights)
    top_scores = np.full(k, -1e30)
    sources = np.full(k, -1, np.int16)
    dests = np.full(k, -1, np.int16)
    scratch = np.zeros(8, np.int16)
    affected = np.zeros(len(WINDOWS), np.bool_)
    for src in range(81):
        color = b[src]
        if color == 0 or access[src] == 0:
            continue
        # Remove source once for all its destinations.
        source_delta = 0.0
        for i in range(N_CELL_WINDOWS[src]):
            w = CELL_WINDOWS[src, i]
            affected[w] = True
            counts[w, color] -= 1
            source_delta += window_value(counts[w], inventory, preview, weights) - old[w]
        for dst in range(81):
            if b[dst] != 0 or not (access[src] & (np.uint64(1) << np.uint64(labels[dst]))):
                continue
            score = source_delta
            for j in range(N_CELL_WINDOWS[dst]):
                w = CELL_WINDOWS[dst, j]
                before = window_value(counts[w], inventory, preview, weights) if affected[w] else old[w]
                counts[w, color] += 1
                score += window_value(counts[w], inventory, preview, weights) - before
                counts[w, color] -= 1
            if score <= top_scores[k - 1]:
                continue
            pos = k - 1
            while pos > 0 and score > top_scores[pos - 1]:
                top_scores[pos] = top_scores[pos - 1]
                sources[pos] = sources[pos - 1]
                dests[pos] = dests[pos - 1]
                pos -= 1
            top_scores[pos] = score
            sources[pos] = src
            dests[pos] = dst
        for i in range(N_CELL_WINDOWS[src]):
            w = CELL_WINDOWS[src, i]
            counts[w, color] += 1
            affected[w] = False
    return sources, dests


@njit(cache=True)
def select_move(b, next3, weights, k=32):
    preview = np.zeros(8, np.float64)
    for c in next3:
        if c:
            preview[c] += 1.0
    sources, dests = candidates(b, preview, weights, k)
    best, best_src, best_dst = -1e30, -1, -1
    for i in range(k):
        src, dst = sources[i], dests[i]
        if src < 0:
            break
        after = b.copy()
        after[dst], after[src] = after[src], 0
        cleared = clear_after_move(after, dst)
        value = evaluate(after, preview, weights)
        if cleared:
            value += weights[9] * (cleared + 3.0) / 8.0
        if value > best:
            best, best_src, best_dst = value, src, dst
    return best_src, best_dst


class Policy:
    def __init__(self, k=64, weights=None):
        self.k = int(k)
        weights = TUNED_WEIGHTS if weights is None else weights
        self.weights = np.asarray(weights, dtype=np.float64)
        if not 1 <= self.k <= 6561:
            raise ValueError('k must be between 1 and 6561')
        if self.weights.shape != (13,) or not np.all(np.isfinite(self.weights)):
            raise ValueError('weights must contain 13 finite numbers')
        if self.weights[6] <= 0 or self.weights[12] < 0:
            raise ValueError('supply discount must be positive and accessibility discount nonnegative')

    def get_action(self, board, next3):
        b = np.asarray(board, dtype=np.int8).reshape(81)
        src, dst = select_move(b, np.asarray(next3, dtype=np.int64), self.weights, self.k)
        return None if src < 0 else (int(src), int(dst))
