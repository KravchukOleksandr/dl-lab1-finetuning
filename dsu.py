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

def build_clusters_dsu(I: np.ndarray, D: np.ndarray | None = None, dist_thr: int | None = None):
    """
    I: (N, k) индексы соседей
    D: (N, k) hamming distances (0..64) — можно не передавать
    dist_thr: если хочешь мягкий порог, например 16 или 20. Если None — без порога.
    """
    n = I.shape[0]
    dsu = DSU(n)

    for i in range(n):
        for t in range(1, I.shape[1]):
            j = int(I[i, t])
            if j < 0:
                continue
            if dist_thr is not None and D is not None:
                if int(D[i, t]) > dist_thr:
                    continue
            dsu.union(i, j)

    # root id для каждого элемента
    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int32)
    return roots


import numpy as np

def pick_time_distributed_per_cluster(roots: np.ndarray, ts: np.ndarray, k: int):
    """
    roots: (N,) int cluster root per sample
    ts: (N,) sortable timestamps (int64, datetime64 ok)
    returns: mask (N,) bool selected
    """
    selected = np.zeros(len(roots), dtype=bool)

    # группируем индексы по root
    # быстрый способ: сортировка по roots
    order = np.argsort(roots)
    roots_sorted = roots[order]

    start = 0
    while start < len(order):
        r = roots_sorted[start]
        end = start
        while end < len(order) and roots_sorted[end] == r:
            end += 1

        idx = order[start:end]                 # индексы элементов кластера
        idx = idx[np.argsort(ts[idx])]         # отсортировали по времени
        n = len(idx)

        if n <= k:
            selected[idx] = True
        else:
            # квантильные позиции
            # i=0..k-1 => round(i*(n-1)/(k-1))
            pos = [round(i*(n-1)/(k-1)) for i in range(k)]
            selected[idx[pos]] = True

        start = end

    return selected