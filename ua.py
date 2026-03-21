from plate_resolver import PlateFormat

UA_LETTERS = frozenset("ABCEHIKMOPTX")
DIGITS = frozenset("0123456789")

TO_LETTER = {
    "0": "O",
    "1": "I",
    "8": "B",
}

TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
}

CYR_TO_LAT = {
    "А": "A",
    "В": "B",
    "Е": "E",
    "І": "I",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
}

UA_PLATE_FORMAT = PlateFormat(
    allowed_by_pos=(
        UA_LETTERS,
        UA_LETTERS,
        DIGITS,
        DIGITS,
        DIGITS,
        DIGITS,
        UA_LETTERS,
        UA_LETTERS,
    ),
    replacements_by_pos=(
        TO_LETTER,
        TO_LETTER,
        TO_DIGIT,
        TO_DIGIT,
        TO_DIGIT,
        TO_DIGIT,
        TO_LETTER,
        TO_LETTER,
    ),
    global_replacements=CYR_TO_LAT,
)
