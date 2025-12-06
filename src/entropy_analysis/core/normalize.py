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

# Patterns to detect brackets inside words
# Square brackets: "пти[чек]" -> "птичек"
BRACKETS_SQUARE_INSIDE_WORD = re.compile(r"([а-яё]+)\[([а-яё]+)\]([а-яё]*)", re.IGNORECASE)
# Angle brackets: "М<артынов>" -> "Мартынов"
BRACKETS_ANGLE_INSIDE_WORD = re.compile(r"([а-яё]+)<([а-яё]+)>([а-яё]*)", re.IGNORECASE)


def preprocess_text_for_brackets(text: str) -> str:
    """
    Preprocess text to handle brackets inside words.
    
    Removes brackets that split words (e.g., "пти[чек]" -> "птичек", "М<артынов>" -> "Мартынов"),
    but preserves brackets around complete words/phrases (e.g., "[Скоро странствию]" remains).
    
    This prevents words from being incorrectly split by OCR errors or editorial marks
    that appear inside words. The pattern requires letters BEFORE brackets to ensure
    we only process brackets that are inside words, not around complete words.
    
    Examples:
        "пти[чек]" -> "птичек" (brackets removed, word merged)
        "[Скоро странствию]" -> "[Скоро странствию]" (preserved, no letters before brackets)
        "М<артынов>" -> "Мартынов" (angle brackets removed)
        "т[е][к]ст" -> "текст" (multiple brackets processed sequentially)
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text with brackets inside words removed
    """
    if not text:
        return text
    
    def merge_bracketed_part(match):
        """Merge bracketed part into word."""
        before = match.group(1)
        inside = match.group(2)
        after = match.group(3) or ""
        # Merge: before + inside + after
        return before + inside + after
    
    # Remove square brackets inside words (between letters)
    # Pattern: letters[letters]letters -> letterslettersletters
    # Requires at least one letter before brackets to ensure it's inside a word
    text = BRACKETS_SQUARE_INSIDE_WORD.sub(merge_bracketed_part, text)
    
    # Remove angle brackets inside words (between letters)
    # Pattern: letters<letters>letters -> letterslettersletters
    text = BRACKETS_ANGLE_INSIDE_WORD.sub(merge_bracketed_part, text)
    
    # Process multiple brackets sequentially (e.g., "т[е][к]ст" -> "текст")
    # Keep applying until no more matches (handles nested/cascading brackets)
    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 0
    while iteration < max_iterations:
        original_text = text
        text = BRACKETS_SQUARE_INSIDE_WORD.sub(merge_bracketed_part, text)
        text = BRACKETS_ANGLE_INSIDE_WORD.sub(merge_bracketed_part, text)
        if text == original_text:
            break  # No more changes
        iteration += 1
    
    return text


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
        
        Args:
            ch: Single character to normalize
            
        Returns:
            Normalized character or None if should be skipped
        """
        if ch is None:
            return None
        
        if not isinstance(ch, str):
            return None
        
        if len(ch) == 0:
            return None
        
        # Take only first character if multiple provided
        ch = ch[0].lower()

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
        
        Args:
            text: Input text to extract letters from
        """
        if text is None or not isinstance(text, str):
            return
        
        # Preprocess to handle brackets inside words
        text = preprocess_text_for_brackets(text)
        
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
        
        Args:
            text: Input text to extract letters from
        """
        if text is None or not isinstance(text, str):
            return
        
        # Preprocess to handle brackets inside words
        text = preprocess_text_for_brackets(text)
        
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
        
        Args:
            text: Input text to tokenize
        """
        if text is None or not isinstance(text, str):
            return []
        
        # Preprocess to handle brackets inside words
        text = preprocess_text_for_brackets(text)
        
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

    def count_letters(self, text: str) -> dict[str, int]:
        """
        Count occurrences of all letters in text (not только первых).

        Используется как основной способ оценки распределения символов для метрик энтропии.
        """
        counts: dict[str, int] = {}
        for letter in self.extract_all_letters(text):
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
