import pandas as pd

def parse_bool_excel(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default

    # Числа
    if isinstance(value, (int, float)):
        return value == 1

    # Строки
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False

    # Если вообще непонятно, что это — дефолт (или raise)
    return default