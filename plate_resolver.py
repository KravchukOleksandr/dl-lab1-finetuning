from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateFormat:
    """
    Describe a fixed-length plate format.

    Attributes:
        allowed_by_pos:
            Allowed characters for each position of the final normalized plate.
        replacements_by_pos:
            Position-specific replacement maps used to convert OCR characters
            into valid characters for that position.
        global_replacements:
            Replacements applied before positional normalization. This is useful
            for OCR-wide cleanup such as Cyrillic-to-Latin lookalike conversion.
    """

    allowed_by_pos: tuple[frozenset[str], ...]
    replacements_by_pos: tuple[dict[str, str], ...]
    global_replacements: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if len(self.allowed_by_pos) != len(self.replacements_by_pos):
            raise ValueError(
                "allowed_by_pos and replacements_by_pos must have the same length"
            )

    @property
    def length(self) -> int:
        """Return the expected normalized plate length."""
        return len(self.allowed_by_pos)


class PlateResolver:
    """
    Normalize OCR candidates into a fixed plate format and pick the final plate.

    The resolver ignores model confidence values and uses only deterministic
    rules:

    1. Apply global normalization.
    2. Drop candidates with the wrong length.
    3. Try to transform each remaining candidate into the configured format.
    4. Choose the candidate with the smallest number of transformations.
    5. If several different candidates tie for the best score, return None.
    """

    def __init__(self, plate_format: PlateFormat) -> None:
        """
        Initialize the resolver.

        Args:
            plate_format: Format definition for the target plate type.
        """
        self.plate_format = plate_format

    def _prepare(self, text: str) -> str:
        """
        Apply global normalization to raw OCR text.

        The method:
        - converts text to uppercase
        - applies global replacements
        - keeps only alphanumeric characters

        Args:
            text: Raw OCR string.

        Returns:
            Cleaned text ready for positional normalization.
        """
        text = text.upper()
        replacements = self.plate_format.global_replacements or {}
        text = "".join(replacements.get(ch, ch) for ch in text)
        return "".join(ch for ch in text if ch.isalnum())

    def normalize(self, text: str) -> tuple[str, int] | None:
        """
        Normalize a single OCR candidate into the configured plate format.

        Args:
            text: Raw OCR candidate.

        Returns:
            A tuple of:
                (normalized_plate, number_of_transformations)

            Returns None if the candidate cannot be transformed into a valid
            plate for this format.
        """
        text = self._prepare(text)
        if len(text) != self.plate_format.length:
            return None

        out: list[str] = []
        changes = 0

        for i, ch in enumerate(text):
            allowed = self.plate_format.allowed_by_pos[i]
            replacements = self.plate_format.replacements_by_pos[i]

            if ch in allowed:
                out.append(ch)
                continue

            replacement = replacements.get(ch)
            if replacement is None or replacement not in allowed:
                return None

            out.append(replacement)
            changes += 1

        return "".join(out), changes
