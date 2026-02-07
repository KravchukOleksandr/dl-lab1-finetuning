def keys_to_int(d: dict) -> dict:
    """
    Convert dict keys to int if possible.
    {"0": v} -> {0: v}
    """
    out = {}
    for k, v in d.items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        out[ik] = v
    return out