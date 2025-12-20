"""
Additional discrete features for poetry recognition (authors, styles).

These features extend the baseline set in recognition_batch.py with
poetry-specific characteristics: meter, rhyme, enjambment, lexical markers, etc.

All features follow the same pattern: text -> discrete bin (string).
"""

import re
from collections import Counter
from typing import Dict, Set

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
VOWELS_RU = set("аеёиоуыэюя")


def _count_syllables(word: str) -> int:
    """Count syllables in a Russian word (vowel count heuristic)."""
    return sum(ch in VOWELS_RU for ch in word.lower())


# ========== 1. METER / PROSODY ==========

def bin_meter_pattern(text: str) -> str:
    """
    Estimate poetic meter based on syllables per line.
    
    Russian poetry:
    - 4-foot iamb: ~8-9 syllables/line (Pushkin's favorite)
    - 5-foot iamb: ~10-11 syllables/line
    - 6-foot iamb: ~12-13 syllables/line
    - Ternary meters (dactyl/amphibrach/anapest): variable
    
    Returns:
        Bin indicating typical syllable count per line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    
    syllables_per_line = []
    for ln in lines[:min(10, len(lines))]:  # analyze first 10 lines
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        total_sylls = sum(_count_syllables(w) for w in words)
        syllables_per_line.append(total_sylls)
    
    if not syllables_per_line:
        return "no_meter"
    
    avg_sylls = sum(syllables_per_line) / len(syllables_per_line)
    
    # Bins tuned for Russian 19th-century poetry
    if avg_sylls <= 6:
        return "very_short_meter"  # fragments, short lines
    elif avg_sylls <= 9:
        return "short_meter"  # 4-foot iamb (common for Pushkin)
    elif avg_sylls <= 12:
        return "mid_meter"  # 5-6 foot iamb
    elif avg_sylls <= 15:
        return "long_meter"  # 6+ foot or ternary
    else:
        return "very_long_meter"


def bin_meter_regularity(text: str) -> str:
    """
    Measure consistency of syllable count across lines (meter regularity).
    
    Pushkin: more regular meter
    Lermontov: more variation in meter
    
    Returns:
        Bin based on standard deviation of syllables per line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    
    syllables = []
    for ln in lines:
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        syllables.append(sum(_count_syllables(w) for w in words))
    
    if len(syllables) < 2:
        return "too_short"
    
    mean_sylls = sum(syllables) / len(syllables)
    variance = sum((s - mean_sylls) ** 2 for s in syllables) / len(syllables)
    std = variance ** 0.5
    
    if std < 1.0:
        return "very_regular"
    elif std < 1.5:
        return "regular"
    elif std < 2.5:
        return "mid_regular"
    elif std < 3.5:
        return "irregular"
    else:
        return "very_irregular"


# ========== 2. RHYME ==========

def bin_rhyme_scheme(text: str) -> str:
    """
    Detect rhyme scheme based on line-ending similarity.
    
    Common schemes:
    - ABAB: cross/alternate rhyme (Pushkin's favorite)
    - AABB: paired/couplet rhyme
    - ABBA: enclosed/ring rhyme
    - free: no clear pattern
    
    Returns:
        Dominant rhyme scheme in the text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return "too_short"
    
    # Extract line endings (last 2 letters of last word)
    endings = []
    for ln in lines[:min(16, len(lines))]:  # analyze up to 16 lines (4 quatrains)
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        last_word = words[-1]
        # Take last 2 chars for rhyme detection
        suffix = last_word[-2:] if len(last_word) >= 2 else last_word
        endings.append(suffix)
    
    if len(endings) < 4:
        return "too_short"
    
    def similar(s1: str, s2: str) -> bool:
        """Check if two endings rhyme (exact match for simplicity)."""
        return s1 == s2
    
    # Count evidence for each scheme
    matches_abab = 0
    matches_aabb = 0
    matches_abba = 0
    
    # Analyze in 4-line chunks (quatrains)
    for i in range(0, len(endings) - 3, 4):
        chunk = endings[i:i+4]
        if len(chunk) < 4:
            break
        
        # ABAB: lines 0,2 rhyme and 1,3 rhyme
        if similar(chunk[0], chunk[2]) and similar(chunk[1], chunk[3]):
            matches_abab += 1
        
        # AABB: lines 0,1 rhyme and 2,3 rhyme
        if similar(chunk[0], chunk[1]) and similar(chunk[2], chunk[3]):
            matches_aabb += 1
        
        # ABBA: lines 0,3 rhyme and 1,2 rhyme
        if similar(chunk[0], chunk[3]) and similar(chunk[1], chunk[2]):
            matches_abba += 1
    
    total_matches = matches_abab + matches_aabb + matches_abba
    
    if total_matches == 0:
        return "free_rhyme"
    
    # Return dominant scheme
    if matches_abab >= matches_aabb and matches_abab >= matches_abba:
        return "cross_rhyme"  # ABAB (Pushkin's favorite)
    elif matches_aabb >= matches_abba:
        return "paired_rhyme"  # AABB
    else:
        return "enclosed_rhyme"  # ABBA


def bin_rhyme_quality(text: str) -> str:
    """
    Estimate rhyme quality based on ending diversity.
    
    High diversity → many different rhymes → rich vocabulary
    Low diversity → repeated rhymes → simpler/folkloric style
    
    Returns:
        Bin based on unique rhyme ratio.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return "too_short"
    
    endings = []
    for ln in lines:
        words = WORD_RE.findall(ln.lower())
        if not words:
            continue
        last_word = words[-1]
        suffix = last_word[-3:] if len(last_word) >= 3 else last_word
        endings.append(suffix)
    
    if not endings:
        return "no_rhyme"
    
    unique_ratio = len(set(endings)) / len(endings)
    
    if unique_ratio < 0.3:
        return "very_repetitive_rhyme"
    elif unique_ratio < 0.5:
        return "repetitive_rhyme"
    elif unique_ratio < 0.7:
        return "mid_rhyme"
    elif unique_ratio < 0.85:
        return "diverse_rhyme"
    else:
        return "very_diverse_rhyme"


# ========== 3. ENJAMBMENT (LINE BREAKS) ==========

def bin_enjambment_rate(text: str) -> str:
    """
    Estimate enjambment frequency (syntactic continuation across line breaks).
    
    Heuristic: lines without terminal punctuation likely have enjambment.
    
    Lermontov: higher enjambment (more emotional flow)
    Pushkin: lower enjambment (more self-contained lines)
    
    Returns:
        Bin based on ratio of lines without terminal punctuation.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "empty"
    
    # Punctuation that typically ends syntactic unit
    terminal_punct = ".,;:!?—-–)]}\"'«»…"
    
    enjambments = 0
    for ln in lines[:-1]:  # exclude last line
        if not ln:
            continue
        last_char = ln[-1]
        # No terminal punctuation → probable enjambment
        if last_char not in terminal_punct:
            enjambments += 1
    
    rate = enjambments / max(len(lines) - 1, 1)
    
    if rate < 0.1:
        return "very_low_enjamb"
    elif rate < 0.25:
        return "low_enjamb"
    elif rate < 0.4:
        return "mid_enjamb"
    elif rate < 0.6:
        return "high_enjamb"
    else:
        return "very_high_enjamb"


# ========== 4. LEXICAL / SEMANTIC ==========

def bin_adjective_density(text: str) -> str:
    """
    Estimate density of adjectives (heuristic: common Russian adjective endings).
    
    Lermontov: more emotional → more adjectives
    Pushkin: more restrained → fewer adjectives
    
    Returns:
        Bin based on adjective ratio.
    """
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    
    # Common Russian adjective endings (simplified, covers ~70-80% of cases)
    adj_endings = {"ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ья", "ье", "го", "ей"}
    
    adj_count = 0
    for token in tokens:
        if len(token) > 3:  # ignore very short words
            ending_2 = token[-2:]
            if ending_2 in adj_endings:
                adj_count += 1
    
    density = adj_count / len(tokens)
    
    if density < 0.05:
        return "very_low_adj"
    elif density < 0.1:
        return "low_adj"
    elif density < 0.15:
        return "mid_adj"
    elif density < 0.2:
        return "high_adj"
    else:
        return "very_high_adj"


# Global storage for dynamically extracted author keywords (populated by build_author_lexicons)
_AUTHOR_LEXICONS: Dict[str, Set[str]] = {}


def build_author_lexicons(segments, top_n: int = 50, min_word_len: int = 3):
    """
    Автоматически извлекает характерные слова для каждого автора через TF-IDF.
    
    УЛУЧШЕНИЯ:
    - Нормализация по количеству текста (исправляет дисбаланс)
    - Фильтрация коротких слов
    - Стоп-слова
    - Диагностика качества
    
    Args:
        segments: список Segment с полями text и label
        top_n: сколько характерных слов извлечь для каждого автора
        min_word_len: минимальная длина слова (фильтр мусора)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("⚠️  sklearn не установлен, лексические признаки недоступны")
        return
    
    from collections import defaultdict
    
    # Группируем тексты по авторам
    texts_by_author = defaultdict(list)
    for seg in segments:
        if seg.label:
            texts_by_author[seg.label].append(seg.text)
    
    if len(texts_by_author) < 2:
        return
    
    # Статистика по авторам
    print("\n📊 Статистика корпуса:")
    authors = list(texts_by_author.keys())
    for author in authors:
        total_chars = sum(len(t) for t in texts_by_author[author])
        print(f"   {author}: {len(texts_by_author[author])} текстов, {total_chars:,} символов")
    
    # КРИТИЧЕСКИ ВАЖНО: создаём СБАЛАНСИРОВАННЫЕ документы
    # Вместо объединения всех текстов, берём СЛУЧАЙНУЮ ВЫБОРКУ одинакового размера
    import random
    random.seed(42)  # для воспроизводимости
    
    # Находим минимальное количество текстов среди авторов
    min_texts = min(len(texts_by_author[a]) for a in authors)
    balanced_docs = []
    
    for author in authors:
        # Берём случайную выборку для баланса
        sample = random.sample(texts_by_author[author], min(min_texts, len(texts_by_author[author])))
        balanced_docs.append(" ".join(sample))
    
    print(f"\n🔄 Сбалансировано: используем по {min_texts} текстов от каждого автора")
    
    # Расширенный список стоп-слов
    stop_words_ru = {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
        "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
        "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
        "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
        "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
        "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
        "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
        "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
        "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас"
    }
    
    # TF-IDF с улучшенными параметрами
    vectorizer = TfidfVectorizer(
        max_features=1000,  # больше для отбора лучших
        ngram_range=(1, 1),
        token_pattern=rf'\b\w{{{min_word_len},}}\b',  # слова >= min_word_len букв
        lowercase=True,
        min_df=1,
        max_df=0.85,  # более строгий порог для частых слов
        sublinear_tf=True,
        use_idf=True,
        stop_words=list(stop_words_ru),  # явные стоп-слова
    )
    
    tfidf_matrix = vectorizer.fit_transform(balanced_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Для каждого автора берём top_n слов
    global _AUTHOR_LEXICONS
    _AUTHOR_LEXICONS.clear()
    
    print(f"\n📚 Извлечённые лексиконы (топ-{top_n}):")
    
    for idx, author in enumerate(authors):
        scores = tfidf_matrix[idx].toarray()[0]
        
        # УЛУЧШЕНИЕ: дополнительная фильтрация
        # Берём только слова с TF-IDF > порога
        threshold = scores.mean() + 0.5 * scores.std()
        
        top_indices = scores.argsort()[-top_n*2:][::-1]  # берём с запасом
        
        # Фильтруем по порогу и проверяем на осмысленность
        top_words = set()
        for i in top_indices:
            if scores[i] > threshold and len(top_words) < top_n:
                word = feature_names[i]
                # Дополнительная проверка: только кириллица (фильтр мусора)
                if word.isalpha() and any(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word.lower()):
                    top_words.add(word)
        
        _AUTHOR_LEXICONS[author] = top_words
        
        # Диагностика
        sample_words = [feature_names[i] for i in top_indices[:15] if scores[i] > threshold]
        sample_words = [w for w in sample_words if w.isalpha()][:10]
        
        print(f"   {author}:")
        print(f"      Найдено: {len(top_words)} слов")
        if sample_words:
            print(f"      Топ-10: {', '.join(sample_words)}")
        else:
            print(f"      ⚠️  ПРОБЛЕМА: не найдено характерных слов!")
    
    print()


def bin_author_lexicon_density(text: str, author_name: str) -> str:
    """
    Универсальная функция для оценки плотности характерных слов ЛЮБОГО автора.
    
    Args:
        text: анализируемый текст
        author_name: имя автора (чьи слова искать)
    
    Returns:
        Bin с плотностью характерных слов этого автора.
    
    Note: Требует предварительного вызова build_author_lexicons()!
    """
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return "empty"
    
    # Если лексикон не построен, возвращаем "unknown"
    if author_name not in _AUTHOR_LEXICONS or not _AUTHOR_LEXICONS[author_name]:
        return "unknown_lexicon"
    
    author_words = _AUTHOR_LEXICONS[author_name]
    count = sum(1 for t in tokens if t in author_words)
    density = count / len(tokens)
    
    # Универсальные бины для любого автора
    if density < 0.005:
        return "very_low"
    elif density < 0.015:
        return "low"
    elif density < 0.03:
        return "mid"
    elif density < 0.05:
        return "high"
    else:
        return "very_high"


# ========== 5. PHONETICS ==========

def bin_consonant_repetition(text: str) -> str:
    """
    Measure alliteration (consonant repetition at word beginnings).
    
    Higher repetition → more deliberate sound patterning.
    
    Returns:
        Bin based on average max repetition per line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    
    consonants = "бвгджзйклмнпрстфхцчшщ"
    max_reps = []
    
    for ln in lines[:min(len(lines), 12)]:  # analyze up to 12 lines
        words = WORD_RE.findall(ln.lower())
        if len(words) < 2:
            continue
        
        # Collect first consonant of each word
        first_consonants = []
        for word in words:
            for ch in word:
                if ch in consonants:
                    first_consonants.append(ch)
                    break
        
        if not first_consonants:
            continue
        
        # Count most frequent consonant in this line
        counts = Counter(first_consonants)
        max_reps.append(max(counts.values()))
    
    if not max_reps:
        return "no_alliteration"
    
    avg_max = sum(max_reps) / len(max_reps)
    
    if avg_max < 1.5:
        return "low_alliter"
    elif avg_max < 2.0:
        return "mid_alliter"
    elif avg_max < 2.5:
        return "high_alliter"
    else:
        return "very_high_alliter"


def bin_voiced_consonant_ratio(text: str) -> str:
    """
    Ratio of voiced consonants (б, в, г, д, ж, з) vs total consonants.
    
    Can indicate phonetic style differences.
    """
    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return "empty"
    
    voiced = "бвгджз"
    consonants = "бвгджзклмнпрстфхцчшщ"
    
    consonant_letters = [ch for ch in letters if ch in consonants]
    if not consonant_letters:
        return "no_consonants"
    
    voiced_count = sum(1 for ch in consonant_letters if ch in voiced)
    ratio = voiced_count / len(consonant_letters)
    
    if ratio < 0.25:
        return "very_low_voiced"
    elif ratio < 0.35:
        return "low_voiced"
    elif ratio < 0.45:
        return "mid_voiced"
    elif ratio < 0.55:
        return "high_voiced"
    else:
        return "very_high_voiced"


# ========== 6. SYNTACTIC ==========

def bin_exclamation_ratio(text: str) -> str:
    """
    Ratio of exclamation marks (emotional intensity marker).
    
    Lermontov: more emotional/dramatic → more exclamations
    Pushkin: more balanced/classical → fewer exclamations
    """
    if not text:
        return "empty"
    
    exclam_count = text.count("!")
    total_punct = text.count(".") + text.count("!") + text.count("?") + text.count(";")
    
    if total_punct == 0:
        return "no_punct"
    
    ratio = exclam_count / total_punct
    
    if ratio < 0.05:
        return "very_low_exclam"
    elif ratio < 0.15:
        return "low_exclam"
    elif ratio < 0.3:
        return "mid_exclam"
    elif ratio < 0.5:
        return "high_exclam"
    else:
        return "very_high_exclam"


def bin_question_ratio(text: str) -> str:
    """
    Ratio of question marks (rhetorical/philosophical style).
    
    Lermontov: more questioning/existential
    """
    if not text:
        return "empty"
    
    quest_count = text.count("?")
    total_punct = text.count(".") + text.count("!") + text.count("?") + text.count(";")
    
    if total_punct == 0:
        return "no_punct"
    
    ratio = quest_count / total_punct
    
    if ratio < 0.05:
        return "very_low_quest"
    elif ratio < 0.15:
        return "low_quest"
    elif ratio < 0.3:
        return "mid_quest"
    elif ratio < 0.5:
        return "high_quest"
    else:
        return "very_high_quest"


# ========== EXPORT ==========

# Feature definitions for integration with recognition_batch.py
POETRY_FEATURE_DEFINITIONS = [
    # Meter / Prosody
    {
        "name": "meter_pattern",
        "values": ["empty", "very_short_meter", "short_meter", "mid_meter", "long_meter", "very_long_meter", "no_meter"],
        "extractor": bin_meter_pattern,
        "description": "Syllables per line (meter estimation)"
    },
    {
        "name": "meter_regularity",
        "values": ["empty", "too_short", "very_regular", "regular", "mid_regular", "irregular", "very_irregular"],
        "extractor": bin_meter_regularity,
        "description": "Consistency of meter across lines"
    },
    
    # Rhyme
    {
        "name": "rhyme_scheme",
        "values": ["too_short", "cross_rhyme", "paired_rhyme", "enclosed_rhyme", "free_rhyme"],
        "extractor": bin_rhyme_scheme,
        "description": "Dominant rhyme scheme (ABAB, AABB, etc.)"
    },
    {
        "name": "rhyme_quality",
        "values": ["too_short", "no_rhyme", "very_repetitive_rhyme", "repetitive_rhyme", "mid_rhyme", "diverse_rhyme", "very_diverse_rhyme"],
        "extractor": bin_rhyme_quality,
        "description": "Rhyme diversity"
    },
    
    # Enjambment
    {
        "name": "enjambment_rate",
        "values": ["empty", "very_low_enjamb", "low_enjamb", "mid_enjamb", "high_enjamb", "very_high_enjamb"],
        "extractor": bin_enjambment_rate,
        "description": "Frequency of line breaks mid-phrase"
    },
    
    # Lexical
    {
        "name": "adjective_density",
        "values": ["empty", "very_low_adj", "low_adj", "mid_adj", "high_adj", "very_high_adj"],
        "extractor": bin_adjective_density,
        "description": "Density of adjectives (emotional intensity)"
    },
    # NOTE: Author-specific lexicons are now built dynamically via build_author_lexicons()
    # and added programmatically in recognition_batch.py during training
    
    # Phonetics
    {
        "name": "consonant_repetition",
        "values": ["empty", "no_alliteration", "low_alliter", "mid_alliter", "high_alliter", "very_high_alliter"],
        "extractor": bin_consonant_repetition,
        "description": "Alliteration intensity"
    },
    {
        "name": "voiced_consonant_ratio",
        "values": ["empty", "no_consonants", "very_low_voiced", "low_voiced", "mid_voiced", "high_voiced", "very_high_voiced"],
        "extractor": bin_voiced_consonant_ratio,
        "description": "Ratio of voiced consonants"
    },
    
    # Syntactic / Emotional
    {
        "name": "exclamation_ratio",
        "values": ["empty", "no_punct", "very_low_exclam", "low_exclam", "mid_exclam", "high_exclam", "very_high_exclam"],
        "extractor": bin_exclamation_ratio,
        "description": "Ratio of exclamation marks (emotional intensity)"
    },
    {
        "name": "question_ratio",
        "values": ["empty", "no_punct", "very_low_quest", "low_quest", "mid_quest", "high_quest", "very_high_quest"],
        "extractor": bin_question_ratio,
        "description": "Ratio of question marks (philosophical/rhetorical)"
    },
]


# Recommended feature profiles for poetry analysis

POETRY_ESSENTIAL_NAMES = {
    # Core poetry features (universal for any authors)
    "meter_pattern",
    "rhyme_scheme",
    "enjambment_rate",
    "adjective_density",
    "exclamation_ratio",
    "meter_regularity",
    "consonant_repetition",
}

POETRY_FULL_NAMES = {
    # All poetry features (universal)
    "meter_pattern",
    "meter_regularity",
    "rhyme_scheme",
    "rhyme_quality",
    "enjambment_rate",
    "adjective_density",
    "consonant_repetition",
    "voiced_consonant_ratio",
    "exclamation_ratio",
    "question_ratio",
    # Author-specific lexicons added dynamically during training
}

