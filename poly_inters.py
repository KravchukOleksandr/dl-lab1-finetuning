import numpy as np

EPS = 1e-9

def polygon_self_intersects(poly: np.ndarray, eps: float = EPS) -> bool:
    """
    poly: (N,2) vertices in order (not repeated first vertex at the end).
    Returns True if polygon edges have a self-intersection.
    """
    poly = np.asarray(poly)
    n = poly.shape[0]
    if n < 4:
        return False  # triangle cannot self-intersect

    def orient(a, b, c) -> float:
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, p) -> bool:
        return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
                min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)

    def seg_intersect(a, b, c, d) -> bool:
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)

        # Proper intersection
        if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
           ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
            return True

        # Collinear / touching
        if abs(o1) <= eps and on_segment(a, b, c): return True
        if abs(o2) <= eps and on_segment(a, b, d): return True
        if abs(o3) <= eps and on_segment(c, d, a): return True
        if abs(o4) <= eps and on_segment(c, d, b): return True
        return False

    # edges: (i, i+1)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        for j in range(i + 1, n):
            # skip same/adjacent edges and the first-last pair (also adjacent in a cycle)
            if j == i or j == i + 1 or (i == 0 and j == n - 1):
                continue

            c = poly[j]
            d = poly[(j + 1) % n]

            # if edges share a vertex, skip (neighbors already skipped, this is extra safety)
            if (np.allclose(a, c, atol=eps) or np.allclose(a, d, atol=eps) or
                np.allclose(b, c, atol=eps) or np.allclose(b, d, atol=eps)):
                continue

            if seg_intersect(a, b, c, d):
                return True

    return False