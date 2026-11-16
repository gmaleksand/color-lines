"""Top-K move selection with component reachability and precomputed line scores."""
import numpy as np
from numba import njit

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@njit
def _line_score_masked(board, r0, c0, dr, dc, L, sr, sc, tr, tc, color):
    score = 0
    i = 0
    while i < L:
        r = r0 + i * dr
        c = c0 + i * dc
        if r == sr and c == sc:
            v = 0
        elif r == tr and c == tc:
            v = color
        else:
            v = board[r, c]
        if v == 0:
            i += 1
            continue
        j = i
        while j + 1 < L:
            r2 = r0 + (j + 1) * dr
            c2 = c0 + (j + 1) * dc
            if r2 == sr and c2 == sc:
                v2 = 0
            elif r2 == tr and c2 == tc:
                v2 = color
            else:
                v2 = board[r2, c2]
            if v2 != v:
                break
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
def _line_origin(r, c, dr, dc, size):
    t_back = size
    if dr > 0:
        t_back = r
    elif dr < 0:
        t_back = size - 1 - r
    if dc > 0:
        t2 = c
        if t2 < t_back:
            t_back = t2
    elif dc < 0:
        t2 = size - 1 - c
        if t2 < t_back:
            t_back = t2
    r0 = r - dr * t_back
    c0 = c - dc * t_back
    t_fwd = size
    if dr > 0:
        t_fwd = size - 1 - r
    elif dr < 0:
        t_fwd = r
    if dc > 0:
        t2 = size - 1 - c
        if t2 < t_fwd:
            t_fwd = t2
    elif dc < 0:
        t2 = c
        if t2 < t_fwd:
            t_fwd = t2
    L = t_back + t_fwd + 1
    return r0, c0, L


@njit
def _cheap_topk(board, size, K, w_run, w_comp):
    n_cells = size * size

    # ---- empty-cell connected components (reachability without per-source BFS)
    comp_id = np.full(n_cells, -1, dtype=np.int32)
    comp_cells = np.zeros(n_cells, dtype=np.int32)
    comp_start = np.zeros(n_cells, dtype=np.int32)
    comp_size = np.zeros(n_cells, dtype=np.int32)
    stack = np.zeros(n_cells, dtype=np.int32)

    ncells = 0
    ncomp = 0
    for p0 in range(n_cells):
        if board[p0 // size, p0 % size] != 0 or comp_id[p0] >= 0:
            continue
        comp_start[ncomp] = ncells
        top = 1
        stack[0] = p0
        comp_id[p0] = ncomp
        comp_cells[ncells] = p0
        ncells += 1
        while top > 0:
            top -= 1
            p = stack[top]
            r = p // size
            c = p % size
            if r > 0 and board[r - 1, c] == 0 and comp_id[p - size] < 0:
                q = p - size
                comp_id[q] = ncomp
                comp_cells[ncells] = q
                ncells += 1
                stack[top] = q
                top += 1
            if r < size - 1 and board[r + 1, c] == 0 and comp_id[p + size] < 0:
                q = p + size
                comp_id[q] = ncomp
                comp_cells[ncells] = q
                ncells += 1
                stack[top] = q
                top += 1
            if c > 0 and board[r, c - 1] == 0 and comp_id[p - 1] < 0:
                q = p - 1
                comp_id[q] = ncomp
                comp_cells[ncells] = q
                ncells += 1
                stack[top] = q
                top += 1
            if c < size - 1 and board[r, c + 1] == 0 and comp_id[p + 1] < 0:
                q = p + 1
                comp_id[q] = ncomp
                comp_cells[ncells] = q
                ncells += 1
                stack[top] = q
                top += 1
        comp_size[ncomp] = ncells - comp_start[ncomp]
        ncomp += 1

    # ---- valid destinations per source, using adjacent components
    reach = np.zeros((n_cells, n_cells), dtype=np.bool_)
    cids = np.zeros(4, dtype=np.int32)
    for sr in range(size):
        for sc in range(size):
            if board[sr, sc] == 0:
                continue
            nci = 0
            sidx = sr * size + sc
            if sr > 0 and board[sr - 1, sc] == 0:
                c = comp_id[sidx - size]
                if nci == 0 or cids[0] != c and cids[nci - 1] != c:
                    found = False
                    for j in range(nci):
                        if cids[j] == c:
                            found = True
                            break
                    if not found:
                        cids[nci] = c
                        nci += 1
            if sr < size - 1 and board[sr + 1, sc] == 0:
                c = comp_id[sidx + size]
                found = False
                for j in range(nci):
                    if cids[j] == c:
                        found = True
                        break
                if not found:
                    cids[nci] = c
                    nci += 1
            if sc > 0 and board[sr, sc - 1] == 0:
                c = comp_id[sidx - 1]
                found = False
                for j in range(nci):
                    if cids[j] == c:
                        found = True
                        break
                if not found:
                    cids[nci] = c
                    nci += 1
            if sc < size - 1 and board[sr, sc + 1] == 0:
                c = comp_id[sidx + 1]
                found = False
                for j in range(nci):
                    if cids[j] == c:
                        found = True
                        break
                if not found:
                    cids[nci] = c
                    nci += 1
            for j in range(nci):
                cc = cids[j]
                for t in range(comp_start[cc], comp_start[cc] + comp_size[cc]):
                    reach[sidx, comp_cells[t]] = True

    # ---- line geometry and pre-move full-line scores
    lr = np.zeros((4, n_cells), dtype=np.int32)
    lc = np.zeros((4, n_cells), dtype=np.int32)
    ll = np.zeros((4, n_cells), dtype=np.int32)
    pre = np.zeros((4, n_cells), dtype=np.int32)
    for d in range(4):
        dr = _DIRS[d][0]
        dc = _DIRS[d][1]
        for p in range(n_cells):
            r = p // size
            c = p % size
            r0, c0, L = _line_origin(r, c, dr, dc, size)
            lr[d, p] = r0
            lc[d, p] = c0
            ll[d, p] = L
            pre[d, p] = _line_score_masked(board, r0, c0, dr, dc, L,
                                   -1, -1, -1, -1, 0)

    top_s1 = np.full(K, -1e30)
    top_src = np.zeros(K, dtype=np.int32)
    top_dst = np.zeros(K, dtype=np.int32)
    top_clr = np.zeros(K, dtype=np.int32)

    for sr in range(size):
        for sc in range(size):
            color = board[sr, sc]
            if color == 0:
                continue
            sidx = sr * size + sc
            for dst in range(n_cells):
                if not reach[sidx, dst]:
                    continue
                dr = dst // size
                dc = dst % size

                # completion + cleared cells
                cleared = 0
                for k in range(4):
                    ddr = _DIRS[k][0]
                    ddc = _DIRS[k][1]
                    a = 0
                    while True:
                        r2 = dr - (a + 1) * ddr
                        c2 = dc - (a + 1) * ddc
                        if r2 < 0 or r2 >= size or c2 < 0 or c2 >= size:
                            break
                        if r2 == sr and c2 == sc:
                            v2 = 0
                        else:
                            v2 = board[r2, c2]
                        if v2 != color:
                            break
                        a += 1
                    b = 0
                    while True:
                        r2 = dr + (b + 1) * ddr
                        c2 = dc + (b + 1) * ddc
                        if r2 < 0 or r2 >= size or c2 < 0 or c2 >= size:
                            break
                        if r2 == sr and c2 == sc:
                            v2 = 0
                        else:
                            v2 = board[r2, c2]
                        if v2 != color:
                            break
                        b += 1
                    runlen = a + b + 1
                    if runlen >= 5:
                        cleared += runlen

                post = 0
                pre_score = 0
                for k in range(4):
                    ddr = _DIRS[k][0]
                    ddc = _DIRS[k][1]
                    post += _line_score_masked(board, lr[k, dst], lc[k, dst],
                                           ddr, ddc, ll[k, dst], sr, sc, dr, dc, color)
                    pre_score += pre[k, dst]
                    same = (sr - dr) * ddc == (sc - dc) * ddr
                    if not same:
                        post += _line_score_masked(board, lr[k, sidx], lc[k, sidx],
                                               ddr, ddc, ll[k, sidx], sr, sc, dr, dc, color)
                        pre_score += pre[k, sidx]

                s1 = w_run * (post - pre_score)
                if cleared >= 5:
                    s1 += w_comp + w_comp * (cleared - 5) // 2

                pos = K
                for t in range(K):
                    if s1 > top_s1[t]:
                        pos = t
                        break
                if pos < K:
                    for t in range(K - 1, pos, -1):
                        top_s1[t] = top_s1[t - 1]
                        top_src[t] = top_src[t - 1]
                        top_dst[t] = top_dst[t - 1]
                        top_clr[t] = top_clr[t - 1]
                    top_s1[pos] = s1
                    top_src[pos] = sidx
                    top_dst[pos] = dst
                    top_clr[pos] = cleared

    n_filled = 0
    for t in range(K):
        if top_s1[t] > -1e29:
            n_filled = t + 1
        else:
            break
    return top_src, top_dst, top_clr, n_filled
