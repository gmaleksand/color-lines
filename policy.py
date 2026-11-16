import numpy as np

import objective as obj
import topk as tk
import lookup_helpers as lh

SIZE = 9
DEFAULT_LAM = 12.0


class Policy:
    def __init__(self, k1=128, lam=DEFAULT_LAM):
        self.k1 = int(k1)
        self.lam = float(lam)
        self.scratch = lh.make_scratch()

    def get_action(self, board, next3):
        board = np.asarray(board, dtype=np.int8)
        src, dst, clr, n = tk._cheap_topk(board, SIZE, self.k1,
                                        obj.W_RUN, obj.W_COMP)
        if n == 0:
            return None

        cw = np.zeros(8, dtype=np.float64)
        for c in next3:
            if c > 0:
                cw[int(c)] += 1.0

        best_first = -1e30
        bt_first = 0

        for t in range(n):
            sidx, dcell = int(src[t]), int(dst[t])
            b1 = board.copy()
            b1[sidx // SIZE, sidx % SIZE] = 0
            b1[dcell // SIZE, dcell % SIZE] = board[sidx // SIZE, sidx % SIZE]
            c = int(clr[t])
            if c >= 5:
                lh.check_and_clear(b1, SIZE)
            sc = lh.first_stage_score(b1, c, cw, self.scratch, self.lam)

            if sc > best_first:
                best_first = sc
                bt_first = t

        return int(src[bt_first]), int(dst[bt_first])


if __name__ == '__main__':
    import env as E
    e = E.EnvNext3()
    p = Policy()
    for seed in range(1, 11):
        b, _ = e.reset(seed=seed)
        done = False
        while not done:
            a = p.get_action(b, e.next3)
            if a is None:
                break
            b, done, _ = e.step(a[0] * 81 + a[1])
        print(seed, e.score)
