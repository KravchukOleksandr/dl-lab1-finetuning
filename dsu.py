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


def select_time_distributed(roots: np.ndarray, ts: np.ndarray, k_keep: int) -> np.ndarray:
    """
    roots: (N,) int cluster id
    ts: (N,) float unix time
    returns: selected mask (N,) bool
    """
    N = len(roots)
    selected = np.zeros(N, dtype=bool)

    order = np.argsort(roots)
    roots_sorted = roots[order]

    start = 0
    while start < N:
        r = roots_sorted[start]
        end = start
        while end < N and roots_sorted[end] == r:
            end += 1

        cluster_idx = order[start:end]  # индексы элементов этого кластера
        picked = pick_k_time_buckets(cluster_idx, ts, k_keep)
        selected[picked] = True

        start = end

    return selected