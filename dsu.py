class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


import numpy as np

def select_k_time_uniform(roots: np.ndarray, ts: np.ndarray, k: int) -> np.ndarray:
    """
    Отбор k точек в каждом кластере так, чтобы они были максимально равномерны по времени:
    - всегда берём крайние (min ts, max ts)
    - затем делим самый большой временной промежуток, вставляя точку ближе к середине
    Возвращает mask selected (N,) bool.
    """
    n = len(roots)
    selected = np.zeros(n, dtype=bool)

    order = np.argsort(roots)
    rs = roots[order]
    i = 0

    while i < n:
        r = rs[i]
        j = i
        while j < n and rs[j] == r:
            j += 1

        idx = order[i:j]                 # глобальные индексы кластера
        m = len(idx)

        if m <= k:
            selected[idx] = True
            i = j
            continue

        # сортируем элементы кластера по времени
        idx = idx[np.argsort(ts[idx])]
        t = ts[idx]

        # выбранные позиции в idx (локальные)
        chosen = [0, m - 1]

        while len(chosen) < k:
            # находим самый большой gap между соседними выбранными
            chosen.sort()
            gaps = [(chosen[p], chosen[p+1], t[chosen[p+1]] - t[chosen[p]]) for p in range(len(chosen)-1)]
            lpos, rpos, _ = max(gaps, key=lambda x: x[2])

            if rpos - lpos <= 1:
                break  # больше некуда вставлять

            mid = 0.5 * (t[lpos] + t[rpos])

            # выбираем точку, ближайшую к mid, внутри (lpos, rpos)
            segment = t[lpos+1:rpos]
            rel = int(np.argmin(np.abs(segment - mid)))
            newpos = lpos + 1 + rel

            if newpos in chosen:
                break
            chosen.append(newpos)

        selected[idx[chosen]] = True
        i = j

    return selected