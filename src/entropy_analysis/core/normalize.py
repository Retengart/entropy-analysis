"""
Text normalization and tokenization for entropy analysis.

Implements the same normalization rules as the Rust version:
- rus29: 29-letter Russian alphabet (ё→е, й→и replacements; ъ, ь excluded)
- rus33: Full 33-letter Russian alphabet (all letters preserved)
- custom: User-defined alphabet
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum


class Alphabet(Enum):
    """Supported alphabet types."""

    RUS29 = "rus29"
    RUS33 = "rus33"
    CUSTOM = "custom"


# Standard alphabets
RUS29_LETTERS = "абвгдежзиклмнопрстуфхцчшщыэюя"
RUS33_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# Unicode regex for Cyrillic words
WORD_PATTERN = re.compile(r"(?ui)[а-яё]+")


@dataclass
class NormalizationConfig:
    """Configuration for text normalization."""

    alphabet: Alphabet = Alphabet.RUS29
    custom_letters: str | None = None
    keep_yo: bool = False  # Keep 'ё' instead of replacing with 'е'
    keep_j: bool = False  # Keep 'й' instead of replacing with 'и'
    min_token_len: int = 1

    def __post_init__(self) -> None:
        if self.alphabet == Alphabet.CUSTOM and not self.custom_letters:
            raise ValueError("Custom alphabet requires custom_letters to be specified")


@dataclass
class Normalizer:
    """Text normalizer with configurable alphabet and rules."""

    config: NormalizationConfig = field(default_factory=NormalizationConfig)
    _letters: list[str] = field(init=False, default_factory=list)
    _letter_to_rank: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the letter list based on configuration."""
        self._letters = self._build_alphabet()
        self._letter_to_rank = {letter: rank for rank, letter in enumerate(self._letters, 1)}

    def _build_alphabet(self) -> list[str]:
        """Build the alphabet based on configuration."""
        match self.config.alphabet:
            case Alphabet.RUS29:
                return list(RUS29_LETTERS)
            case Alphabet.RUS33:
                return list(RUS33_LETTERS)
            case Alphabet.CUSTOM:
                if not self.config.custom_letters:
                    raise ValueError("Custom alphabet requires letters")
                # Preserve order, remove duplicates
                seen: set[str] = set()
                letters: list[str] = []
                for ch in self.config.custom_letters:
                    if ch not in seen:
                        seen.add(ch)
                        letters.append(ch)
                return letters

    @property
    def letters(self) -> list[str]:
        """Get the alphabet letters."""
        return self._letters

    @property
    def alphabet_size(self) -> int:
        """Get the number of letters in the alphabet."""
        return len(self._letters)

    def get_rank(self, letter: str) -> int | None:
        """Get the rank (1-based position) of a letter in the alphabet."""
        return self._letter_to_rank.get(letter)

    def normalize_char(self, ch: str) -> str | None:
        """
        Normalize a single character according to the rules.

        Returns None if the character should be skipped.
        """
        ch = ch.lower()

        # Apply replacements for rus29 mode
        if self.config.alphabet == Alphabet.RUS29:
            if ch == "ё" and not self.config.keep_yo:
                ch = "е"
            if ch == "й" and not self.config.keep_j:
                ch = "и"
            if ch in ("ъ", "ь"):
                return None

        return ch if ch in self._letter_to_rank else None

    def extract_first_letters(self, text: str) -> Iterator[str]:
        """
        Extract and normalize first letters of words from text.

        Yields normalized first letters that pass validation.
        """
        for match in WORD_PATTERN.finditer(text):
            token = match.group()
            if len(token) < self.config.min_token_len:
                continue

            first_char = token[0].lower()
            normalized = self.normalize_char(first_char)
            if normalized is not None:
                yield normalized

    def extract_all_letters(self, text: str) -> Iterator[str]:
        """
        Extract and normalize all letters from text (not just first letters).

        This is useful for analyzing letter sequences, bigrams, and trigrams
        at the letter level, providing deeper insight into text structure.

        Yields normalized letters that pass validation.
        """
        for match in WORD_PATTERN.finditer(text):
            token = match.group()
            if len(token) < self.config.min_token_len:
                continue

            # Extract all letters from the token
            for char in token:
                normalized = self.normalize_char(char)
                if normalized is not None:
                    yield normalized

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.

        Returns list of lowercase tokens that start with valid alphabet letters.
        """
        tokens: list[str] = []
        for match in WORD_PATTERN.finditer(text):
            token = match.group().lower()
            if len(token) < self.config.min_token_len:
                continue

            # Check if first letter is valid
            first_char = self.normalize_char(token[0])
            if first_char is not None:
                tokens.append(token)

        return tokens

    def count_first_letters(self, text: str) -> dict[str, int]:
        """
        Count occurrences of first letters in text.

        Returns a dictionary mapping letters to counts (only non-zero counts).
        """
        counts: dict[str, int] = {}
        for letter in self.extract_first_letters(text):
            counts[letter] = counts.get(letter, 0) + 1
        return counts


def create_normalizer(
    alphabet: Alphabet | str = Alphabet.RUS29,
    custom_letters: str | None = None,
    keep_yo: bool = False,
    keep_j: bool = False,
    min_token_len: int = 1,
) -> Normalizer:
    """Factory function to create a Normalizer with common configurations."""
    if isinstance(alphabet, str):
        # Handle legacy "rus28" name for backward compatibility
        if alphabet == "rus28":
            alphabet = Alphabet.RUS29
        else:
            alphabet = Alphabet(alphabet)

    config = NormalizationConfig(
        alphabet=alphabet,
        custom_letters=custom_letters,
        keep_yo=keep_yo,
        keep_j=keep_j,
        min_token_len=min_token_len,
    )
    return Normalizer(config=config)
