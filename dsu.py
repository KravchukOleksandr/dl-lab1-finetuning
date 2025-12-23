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
import math

def select_k_by_time_entropy(roots: np.ndarray, ts: np.ndarray, k: int) -> np.ndarray:
    """
    roots: (N,) int - id кластера (например DSU root)
    ts:    (N,) float - unix time
    k:     int - сколько отбирать из каждого кластера (если в кластере меньше, берём все)

    return: selected_mask (N,) bool
    """

    N = len(roots)
    selected = np.zeros(N, dtype=bool)

    # Чтобы группировать быстро
    order = np.argsort(roots)
    roots_sorted = roots[order]

    def delta_entropy(g: float, g1: float, g2: float, T: float) -> float:
        """
        Прирост энтропии при разбиении промежутка g на g1 и g2.
        Все величины в секундах (float), T = t_max - t_min фиксирован.
        Энтропия считается по p = gap/T.
        """
        # нормируем
        p  = g  / T
        p1 = g1 / T
        p2 = g2 / T

        # аккуратно с нулями (на практике g1,g2>0 если ts различаются)
        def h(x: float) -> float:
            return 0.0 if x <= 0.0 else -x * math.log(x)

        return (h(p1) + h(p2) - h(p))

    def pick_cluster(idx: np.ndarray) -> np.ndarray:
        """
        idx: глобальные индексы элементов кластера
        return: глобальные индексы выбранных элементов (len=min(k, m))
        """
        m = len(idx)
        if m == 0:
            return np.array([], dtype=int)

        # сортируем элементы кластера по времени
        idx_sorted = idx[np.argsort(ts[idx])]
        t = ts[idx_sorted].astype(np.float64)

        if m <= k:
            return idx_sorted

        # фиксируем границы
        # выбранные позиции (в координатах idx_sorted)
        chosen_pos = [0, m - 1]
        chosen_set = {0, m - 1}

        T = float(t[-1] - t[0])
        if T <= 0.0:
            # все времена одинаковые -> просто берём первые k по порядку
            return idx_sorted[:k]

        # список промежутков между выбранными точками: (lpos, rpos)
        gaps = [(0, m - 1)]

        # пока не набрали k, добавляем точки
        while len(chosen_pos) < k:
            best_gain = -1e300
            best_gap_i = -1
            best_newpos = None

            # перебираем текущие gaps
            for gi, (lpos, rpos) in enumerate(gaps):
                if rpos - lpos <= 1:
                    continue  # внутри нет точек

                # целевое время середины промежутка
                mid = 0.5 * (t[lpos] + t[rpos])

                # находим ближайшую позицию к mid среди внутренних [lpos+1 .. rpos-1]
                # через searchsorted в отсортированных t
                j = int(np.searchsorted(t, mid, side="left"))

                candidates = []
                # ограничиваем j в пределах внутреннего диапазона
                j = max(lpos + 1, min(rpos - 1, j))

                # попробуем пару соседей вокруг j, чтобы найти реально ближайшего
                for p in (j - 1, j, j + 1):
                    if lpos < p < rpos and p not in chosen_set:
                        candidates.append(p)

                if not candidates:
                    # может быть так, что j/соседи уже выбраны; тогда ищем ближайшего свободного вокруг
                    # простой линейный шаг наружу (обычно очень короткий)
                    left = j
                    right = j
                    found = None
                    while (left > lpos) or (right < rpos):
                        if left > lpos:
                            left -= 1
                            if left not in chosen_set and left > lpos:
                                found = left
                                break
                        if right < rpos:
                            right += 1
                            if right not in chosen_set and right < rpos:
                                found = right
                                break
                    if found is not None:
                        candidates = [found]
                    else:
                        continue

                # выбираем кандидата, который ближе всего к mid
                newpos = min(candidates, key=lambda p: abs(t[p] - mid))

                # считаем прирост энтропии для разбиения этого gap
                g  = float(t[rpos] - t[lpos])
                g1 = float(t[newpos] - t[lpos])
                g2 = float(t[rpos] - t[newpos])
                gain = delta_entropy(g, g1, g2, T)

                if gain > best_gain:
                    best_gain = gain
                    best_gap_i = gi
                    best_newpos = newpos

            if best_newpos is None:
                # не нашли куда вставить (теоретически может случиться, если что-то странное с временами)
                break

            # добавляем точку
            chosen_set.add(best_newpos)
            chosen_pos.append(best_newpos)
            chosen_pos.sort()

            # обновляем gaps: заменяем выбранный gap на два
            lpos, rpos = gaps.pop(best_gap_i)
            gaps.append((lpos, best_newpos))
            gaps.append((best_newpos, rpos))

        # возвращаем выбранные глобальные индексы в порядке времени
        chosen_pos = sorted(chosen_set)
        return idx_sorted[np.array(chosen_pos, dtype=int)]

    # группируем по roots
    start = 0
    while start < N:
        r = roots_sorted[start]
        end = start
        while end < N and roots_sorted[end] == r:
            end += 1

        cluster_global_idx = order[start:end]
        picked = pick_cluster(cluster_global_idx)
        selected[picked] = True

        start = end

    return selected