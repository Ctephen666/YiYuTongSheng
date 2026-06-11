from __future__ import annotations

import re
from functools import lru_cache

from src.common.io_utils import path_from_config
from src.common.json_utils import read_json, write_json


ARPABET_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


CONTRACTION_WORDS = {
    "i'm",
    "you're",
    "we're",
    "they're",
    "he's",
    "she's",
    "it's",
    "that's",
    "there's",
    "what's",
    "who's",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "i've",
    "you've",
    "we've",
    "they've",
    "i'll",
    "you'll",
    "we'll",
    "they'll",
    "i'd",
    "you'd",
    "we'd",
    "they'd",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text or ""))


def normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-z']", "", str(word or "")).lower()


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", str(phone or "").strip())


def is_vowel_phone(phone: str) -> bool:
    return strip_stress(phone) in ARPABET_VOWELS


def clean_arpabet_phones(phones: list[str]) -> list[str]:
    cleaned = []

    for phone in phones:
        phone = strip_stress(phone)

        if not phone:
            continue

        # Keep ARPABET-like symbols only.
        if re.fullmatch(r"[A-Z]+", phone):
            cleaned.append(phone)

    return cleaned


def fallback_letter_phonemes(word: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z']", "", str(word or ""))

    return [
        char.upper()
        for char in cleaned
        if char.strip()
    ]


def count_vowel_phones(phones: list[str]) -> int:
    count = 0

    for phone in phones:
        if is_vowel_phone(phone):
            count += 1

    return count


def has_silent_e(word: str) -> bool:
    word = normalize_word(word)

    if len(word) <= 3:
        return False

    if not word.endswith("e"):
        return False

    # Do not remove words where final e is usually pronounced or part of vowel pair.
    if word.endswith(("ee", "ye", "oe")):
        return False

    return True


def merge_trailing_silent_e(parts: list[str], original_word: str) -> list[str]:
    if not parts:
        return parts

    word = normalize_word(original_word)

    if not has_silent_e(word):
        return parts

    if len(parts) >= 2 and parts[-1] == "e":
        parts[-2] = parts[-2] + parts[-1]
        parts.pop()

    return parts


def adjust_parts_to_count(parts: list[str], target_count: int) -> list[str]:
    parts = [
        str(part or "").strip().lower()
        for part in parts
        if str(part or "").strip()
    ]

    if target_count <= 0:
        return parts

    if not parts:
        return []

    while len(parts) > target_count:
        # Merge the shortest part into a neighbor.
        idx = min(
            range(len(parts)),
            key=lambda i: len(parts[i]),
        )

        if idx < len(parts) - 1:
            parts[idx + 1] = parts[idx] + parts[idx + 1]
            parts.pop(idx)
        else:
            parts[idx - 1] = parts[idx - 1] + parts[idx]
            parts.pop(idx)

    while len(parts) < target_count:
        # Split the longest part.
        idx = max(
            range(len(parts)),
            key=lambda i: len(parts[i]),
        )

        part = parts[idx]

        if len(part) <= 2:
            break

        mid = max(1, len(part) // 2)
        left = part[:mid]
        right = part[mid:]

        if not left or not right:
            break

        parts[idx:idx + 1] = [left, right]

    return parts


def fallback_orthographic_syllables(word: str, target_count: int) -> list[str]:
    """Fallback display syllables.

    This is only used when pyphen/CMU information is insufficient.
    It tries to avoid bad silent-e splits like time -> tim/e.
    """

    word = normalize_word(word)

    if not word:
        return []

    if target_count <= 1:
        return [word]

    split_base = word

    if has_silent_e(split_base):
        split_base = split_base[:-1]

    vowel_re = re.compile(r"[aeiouy]+")
    matches = list(vowel_re.finditer(split_base))

    if not matches:
        return [word]

    parts = []
    start = 0

    for i, match in enumerate(matches):
        vowel_end = match.end()

        if i < len(matches) - 1:
            next_start = matches[i + 1].start()
            consonant_gap = split_base[vowel_end:next_start]

            if len(consonant_gap) <= 1:
                end = vowel_end
            else:
                end = next_start - 1
        else:
            end = len(split_base)

        end = max(end, start + 1)
        parts.append(split_base[start:end])
        start = end

    if start < len(split_base):
        if parts:
            parts[-1] += split_base[start:]
        else:
            parts.append(split_base[start:])

    if has_silent_e(word) and parts:
        parts[-1] += "e"

    parts = adjust_parts_to_count(parts, target_count)

    if len(parts) != target_count:
        parts = [word]

    return parts


def split_phones_to_syllable_groups(
    phones: list[str],
    syllable_count: int,
) -> list[list[str]]:
    """Split ARPABET phonemes into syllable-ish groups.

    This is approximate, but it is much better than letter-level phonemes.
    """

    phones = [
        strip_stress(phone)
        for phone in phones
        if strip_stress(phone)
    ]

    if not phones:
        return []

    if syllable_count <= 1:
        return [phones]

    vowel_indices = [
        i
        for i, phone in enumerate(phones)
        if is_vowel_phone(phone)
    ]

    if len(vowel_indices) != syllable_count:
        # Fallback: proportional split.
        groups = []

        for i in range(syllable_count):
            start = round(i * len(phones) / syllable_count)
            end = round((i + 1) * len(phones) / syllable_count)

            if i == syllable_count - 1:
                end = len(phones)

            group = phones[start:end]

            if group:
                groups.append(group)

        if len(groups) == syllable_count:
            return groups

        return [phones]

    groups = []
    start = 0

    for i in range(syllable_count):
        if i == syllable_count - 1:
            end = len(phones)
        else:
            current_vowel = vowel_indices[i]
            next_vowel = vowel_indices[i + 1]

            between = next_vowel - current_vowel - 1

            if between <= 0:
                end = next_vowel
            elif between == 1:
                # Consonant tends to attach to next syllable onset.
                end = current_vowel + 1
            else:
                # Split consonant cluster before next vowel.
                end = next_vowel - 1

        end = max(end, start + 1)
        groups.append(phones[start:end])
        start = end

    if len(groups) != syllable_count:
        return [phones]

    return groups


class EnglishPhonemizer:
    """Convert selected English lyrics into syllable-level phoneme units.

    Priority:
    1. CMUdict via pronouncing
    2. g2p-en
    3. fallback letter phonemes

    For syllable display units:
    1. Use CMU/ARPABET vowel count to decide real syllable count.
    2. Use pyphen only as an orthographic display splitter.
    3. Avoid silent-e errors such as:
       time -> tim/e
       freeze -> freez/e
       love -> lov/e
       fate -> fat/e
    """

    def __init__(self, config: dict):
        self.config = config
        self._pyphen_dic = None
        self._g2p = None

    # ------------------------------------------------------------------
    # Lazy dependency loading
    # ------------------------------------------------------------------

    def _get_pyphen(self):
        if self._pyphen_dic is not None:
            return self._pyphen_dic

        try:
            import pyphen

            self._pyphen_dic = pyphen.Pyphen(lang="en_US")
        except Exception:
            self._pyphen_dic = False

        return self._pyphen_dic

    def _get_g2p(self):
        if self._g2p is not None:
            return self._g2p

        try:
            from g2p_en import G2p

            self._g2p = G2p()
        except Exception:
            self._g2p = False

        return self._g2p

    # ------------------------------------------------------------------
    # CMUdict / G2P
    # ------------------------------------------------------------------

    @lru_cache(maxsize=4096)
    def _cmu_phones(self, normalized_word: str) -> list[str]:
        try:
            import pronouncing
        except Exception:
            return []

        if not normalized_word:
            return []

        candidates = pronouncing.phones_for_word(normalized_word)

        if not candidates:
            return []

        phones = candidates[0].split()
        return clean_arpabet_phones(phones)

    def _g2p_phones(self, word: str) -> list[str]:
        g2p = self._get_g2p()

        if not g2p:
            return []

        try:
            result = g2p(word)
        except Exception:
            return []

        phones = []

        for item in result:
            item = strip_stress(str(item or ""))

            if not item:
                continue

            if re.fullmatch(r"[A-Z]+", item):
                phones.append(item)

        return clean_arpabet_phones(phones)

    def _word_phones(self, word: str) -> tuple[list[str], str]:
        normalized = normalize_word(word)

        if not normalized:
            return [], "empty"

        phones = self._cmu_phones(normalized)

        if phones:
            return phones, "cmudict"

        phones = self._g2p_phones(word)

        if phones:
            return phones, "g2p_en"

        return fallback_letter_phonemes(word), "letter_fallback"

    # ------------------------------------------------------------------
    # Syllable display splitting
    # ------------------------------------------------------------------

    def _syllable_count_from_phones(self, phones: list[str], word: str) -> int:
        count = count_vowel_phones(phones)

        if count > 0:
            return count

        normalized = normalize_word(word)

        if not normalized:
            return 0

        return 1

    def _pyphen_syllables(
        self,
        word: str,
        target_count: int,
    ) -> list[str]:
        normalized = normalize_word(word)

        if not normalized:
            return []

        if target_count <= 1:
            return [normalized]

        pyphen_dic = self._get_pyphen()

        if not pyphen_dic:
            return []

        try:
            inserted = pyphen_dic.inserted(normalized)
        except Exception:
            return []

        if not inserted:
            return []

        parts = [
            part.strip().lower()
            for part in inserted.split("-")
            if part.strip()
        ]

        parts = merge_trailing_silent_e(parts, normalized)
        parts = adjust_parts_to_count(parts, target_count)

        if len(parts) == target_count:
            return parts

        return []

    def _split_word_to_syllable_texts(
        self,
        word: str,
        phones: list[str],
    ) -> list[str]:
        normalized = normalize_word(word)

        if not normalized:
            return []

        if normalized in CONTRACTION_WORDS:
            return [normalized]

        syllable_count = self._syllable_count_from_phones(phones, normalized)

        if syllable_count <= 1:
            return [normalized]

        pyphen_parts = self._pyphen_syllables(
            normalized,
            syllable_count,
        )

        if pyphen_parts:
            return pyphen_parts

        return fallback_orthographic_syllables(
            normalized,
            syllable_count,
        )

    # ------------------------------------------------------------------
    # Main conversion
    # ------------------------------------------------------------------

    def _build_word_item(
        self,
        word: str,
        word_index: int,
        unit_index_start: int,
    ) -> tuple[dict, list[dict], int]:
        phones, phone_source = self._word_phones(word)
        syllable_texts = self._split_word_to_syllable_texts(word, phones)

        if not syllable_texts:
            syllable_texts = [normalize_word(word) or word]

        phone_groups = split_phones_to_syllable_groups(
            phones,
            len(syllable_texts),
        )

        if len(phone_groups) != len(syllable_texts):
            phone_groups = [phones]
            syllable_texts = [normalize_word(word) or word]

        word_item = {
            "word": word,
            "word_index": word_index,
            "phonemes": phones,
            "phoneme_source": phone_source,
            "syllable_count": len(syllable_texts),
            "syllables": [],
        }

        units = []
        unit_index = unit_index_start

        for syllable_index, syllable_text in enumerate(syllable_texts):
            syllable_phones = phone_groups[syllable_index]

            unit = {
                "unit_id": unit_index,
                "unit": syllable_text,
                "word": word,
                "word_index": word_index,
                "syllable_index": syllable_index,
                "phonemes": syllable_phones,
                "phoneme_source": phone_source,
            }

            units.append(unit)
            word_item["syllables"].append(unit)
            unit_index += 1

        return word_item, units, unit_index

    def _build_units(self, text: str) -> tuple[list[dict], list[dict]]:
        syllable_units = []
        tokens = []
        unit_index = 0

        for word_index, word in enumerate(words(text)):
            word_item, word_units, unit_index = self._build_word_item(
                word=word,
                word_index=word_index,
                unit_index_start=unit_index,
            )

            tokens.append(word_item)
            syllable_units.extend(word_units)

        return syllable_units, tokens

    def run(self) -> dict:
        lyrics_singable = path_from_config(self.config, "lyrics_singable")
        output = path_from_config(self.config, "phonemes_target")

        lyric_data = read_json(lyrics_singable, {"phrases": []})
        phrases = lyric_data.get("phrases", [])

        if not phrases:
            raise RuntimeError("lyrics_singable.json contains no phrases.")

        output_phrases = []

        for phrase in phrases:
            text = (
                phrase.get("selected")
                or phrase.get("text")
                or phrase.get("literal_en")
                or ""
            ).strip()

            if not text:
                raise RuntimeError(
                    f"Missing selected English lyric for phrase id={phrase.get('id')}."
                )

            syllable_units, tokens = self._build_units(text)

            if not syllable_units:
                raise RuntimeError(
                    f"No syllable units generated for phrase id={phrase.get('id')}."
                )

            output_phrases.append(
                {
                    "id": phrase.get("id"),
                    "zh": phrase.get("zh", ""),
                    "literal_en": phrase.get("literal_en", ""),
                    "text": text,
                    "note_count": phrase.get("note_count"),
                    "phrase_duration": phrase.get("phrase_duration"),
                    "unit_type": "syllable",
                    "unit_count": len(syllable_units),
                    "units": syllable_units,
                    "tokens": tokens,
                }
            )

        write_json(
            output,
            {
                "language": "en",
                "unit_type": "syllable",
                "description": (
                    "English syllable-level phonemization using CMUdict/g2p-en "
                    "with fallback handling for unknown words and silent-e cases."
                ),
                "phrases": output_phrases,
            },
            self.config,
        )

        return {
            "status": "success",
            "outputs": {
                "phonemes_target": str(output),
            },
            "message": (
                "Generated syllable-level English phoneme units with CMUdict/g2p-en."
            ),
        }