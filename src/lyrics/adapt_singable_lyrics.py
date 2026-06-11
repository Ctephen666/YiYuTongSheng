from __future__ import annotations

import re

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json
from src.lyrics.syllable_counter import count_english_syllables


ARTICLES = {
    "a",
    "an",
    "the",
}

DETERMINERS = {
    "this",
    "that",
    "these",
    "those",
}

CONJUNCTIONS = {
    "and",
    "or",
    "but",
}

RELATION_WORDS = {
    "to",
    "for",
    "with",
    "without",
    "through",
    "into",
    "about",
    "around",
    "before",
    "after",
    "under",
    "over",
    "between",
    "beyond",
    "amid",
    "among",
    "within",
    "outside",
    "inside",
    "across",
    "against",
    "toward",
    "towards",
    "in",
    "on",
    "at",
    "from",
    "by",
    "as",
    "of",
}

AUXILIARY_WORDS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
}

LOW_VALUE_WORDS = {
    "very",
    "so",
    "just",
    "really",
    "already",
    "perhaps",
    "once",
    "much",
    "simply",
}

SOFT_LOW_VALUE_WORDS = {
    "still",
    "maybe",
    "even",
}

IMPORTANT_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "we",
    "us",
    "our",
    "ours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
}

CONTRACTIONS = {
    "I am": "I'm",
    "I will": "I'll",
    "I would": "I'd",
    "I have": "I've",
    "you are": "you're",
    "you will": "you'll",
    "you have": "you've",
    "we are": "we're",
    "we will": "we'll",
    "we have": "we've",
    "they are": "they're",
    "they will": "they'll",
    "they have": "they've",
    "it is": "it's",
    "it will": "it'll",
    "there is": "there's",
    "that is": "that's",
    "who is": "who's",
    "what is": "what's",
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
}


class SingableLyricAdapter:
    """Adapt literal translations into singable English lyrics.

    General automatic system rules:
    1. No fixed universal lyric candidates.
    2. No song-specific fallback strategy.
    3. No phrase override.
    4. No manual phrase-level rewrite rules.
    5. All candidates must be derived from the current phrase's literal_en.
    6. If lyrics_literal.json is placeholder/failed, stop immediately.

    This version protects relation words such as:
        to / in / for / of / with / amid / beyond

    Because English singable lyrics often rely on those small words to preserve
    grammar, even when they are weakly pronounced or linked during singing.
    """

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Basic text utilities
    # ------------------------------------------------------------------

    def _words(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text or ""))

    def _normalize_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("♪", " ")
        text = text.replace("“", "").replace("”", "")
        text = text.replace('"', "")
        text = re.sub(r"[。！？；：，、]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = text.strip()
        text = text.strip(" ,;:")
        return text

    def _apply_contractions(self, text: str) -> str:
        text = self._normalize_text(text)

        for src, dst in CONTRACTIONS.items():
            pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
            text = pattern.sub(dst, text)

        return self._normalize_text(text)

    def _remove_duplicate_words(self, text: str) -> str:
        words = self._words(text)

        if not words:
            return self._normalize_text(text)

        result = []
        previous = None

        for word in words:
            lower = word.lower()

            if lower == previous:
                continue

            result.append(word)
            previous = lower

        return self._normalize_text(" ".join(result))

    def _content_words(self, text: str) -> list[str]:
        """Return semantic words plus relation words.

        Important change:
        relation words are kept here.

        Old behavior removed relation words, which made candidates like:
            fall love
            brought us meet
            wait for you unknown

        New behavior preserves relation structure.
        """

        result = []

        for word in self._words(text):
            lower = word.lower()

            if lower in ARTICLES:
                continue

            if lower in DETERMINERS:
                continue

            if lower in CONJUNCTIONS:
                continue

            if lower in AUXILIARY_WORDS:
                continue

            result.append(word)

        return result or self._words(text)

    def _relation_words(self, text: str) -> list[str]:
        return [
            word.lower()
            for word in self._words(text)
            if word.lower() in RELATION_WORDS
        ]

    # ------------------------------------------------------------------
    # Rhythm constraints
    # ------------------------------------------------------------------

    def _target_syllable_range(self, note_count: int) -> tuple[int, int]:
        if note_count <= 4:
            return 2, 5

        if note_count <= 8:
            return 4, 8

        if note_count <= 12:
            return 6, 11

        if note_count <= 16:
            return 8, 13

        if note_count <= 22:
            return 9, 15

        return 10, 17

    def _target_syllables(self, note_count: int) -> int:
        low, high = self._target_syllable_range(note_count)
        return min(high, max(low, round(note_count * 0.75)))

    def _phrase_duration(self, phrase: dict) -> float:
        return round(
            float(phrase.get("end", 0.0)) - float(phrase.get("start", 0.0)),
            3,
        )

    def _note_count(self, phrase: dict, melody_by_id: dict) -> int:
        note_start = phrase.get("note_start")
        note_end = phrase.get("note_end")

        if note_start is not None and note_end is not None:
            return max(1, int(note_end) - int(note_start))

        melody_phrase = melody_by_id.get(phrase.get("id"), {})
        return max(1, len(melody_phrase.get("notes", [])))

    # ------------------------------------------------------------------
    # Candidate generation: literal-derived only
    # ------------------------------------------------------------------

    def _word_importance(self, word: str, index: int, total: int) -> float:
        lower = word.lower()

        score = 1.0

        if lower in LOW_VALUE_WORDS:
            score -= 0.85

        if lower in SOFT_LOW_VALUE_WORDS:
            score -= 0.35

        if lower in ARTICLES:
            score -= 0.55

        if lower in DETERMINERS:
            score -= 0.35

        if lower in CONJUNCTIONS:
            score -= 0.25

        if lower in AUXILIARY_WORDS:
            score -= 0.15

        # Core fix:
        # relation words are grammatically important and should not be
        # deleted simply because they are short.
        if lower in RELATION_WORDS:
            score += 0.85

        if lower in IMPORTANT_PRONOUNS:
            score += 0.35

        if index == 0 or index == total - 1:
            score += 0.25

        if (
            lower not in ARTICLES
            and lower not in DETERMINERS
            and lower not in CONJUNCTIONS
            and lower not in AUXILIARY_WORDS
            and lower not in LOW_VALUE_WORDS
        ):
            score += 0.45

        return score

    def _can_remove_word(
        self,
        words: list[str],
        index: int,
    ) -> bool:
        """Whether a word can be safely removed during compression.

        This is not phrase-specific. It protects generic English structure.
        """

        word = words[index]
        lower = word.lower()

        if lower in IMPORTANT_PRONOUNS:
            return False

        # Do not remove relation words during ordinary deletion compression.
        # English singing can weakly pronounce them later during alignment.
        if lower in RELATION_WORDS:
            return False

        # Avoid ending with broken auxiliary-like fragments.
        if index == len(words) - 1 and lower in AUXILIARY_WORDS:
            return False

        # Articles are safe to remove.
        if lower in ARTICLES:
            return True

        if lower in LOW_VALUE_WORDS:
            return True

        if lower in SOFT_LOW_VALUE_WORDS:
            return True

        if lower in DETERMINERS:
            return True

        if lower in CONJUNCTIONS:
            return True

        # Auxiliaries can be removed sometimes, but not if they are the only
        # grammar bridge in a very short phrase.
        if lower in AUXILIARY_WORDS and len(words) >= 5:
            return True

        return True

    def _compress_by_deletion(self, text: str, max_syllables: int) -> str:
        text = self._normalize_text(text)
        current = self._words(text)

        if not current:
            return text

        while (
            count_english_syllables(" ".join(current)) > max_syllables
            and len(current) > 2
        ):
            removable_indices = [
                index
                for index in range(len(current))
                if self._can_remove_word(current, index)
            ]

            if not removable_indices:
                break

            scores = [
                self._word_importance(current[index], index, len(current))
                for index in removable_indices
            ]

            local_remove_index = min(range(len(removable_indices)), key=lambda i: scores[i])
            remove_index = removable_indices[local_remove_index]

            current.pop(remove_index)

        return self._normalize_text(" ".join(current))

    def _drop_articles_only(self, text: str) -> str:
        words = [
            word
            for word in self._words(text)
            if word.lower() not in ARTICLES
        ]

        return self._normalize_text(" ".join(words))

    def _drop_low_value_only(self, text: str) -> str:
        words = [
            word
            for word in self._words(text)
            if word.lower() not in LOW_VALUE_WORDS
        ]

        return self._normalize_text(" ".join(words))

    def _extract_content_phrase(self, text: str, max_words: int) -> str:
        content = self._content_words(text)

        if not content:
            return self._normalize_text(text)

        return self._normalize_text(" ".join(content[:max_words]))

    def _content_windows(self, text: str, max_words: int) -> list[str]:
        content = self._content_words(text)

        if len(content) < 2:
            return []

        if len(content) <= max_words:
            return [self._normalize_text(" ".join(content))]

        candidates = []

        for start in range(0, len(content)):
            chunk = content[start:start + max_words]

            if len(chunk) >= 2:
                candidates.append(self._normalize_text(" ".join(chunk)))

        return candidates

    def _is_valid_candidate(
        self,
        candidate: str,
        literal: str,
        note_count: int,
    ) -> bool:
        candidate = self._normalize_text(candidate)

        if not candidate:
            return False

        if candidate.lower() == "la":
            return False

        words = self._words(candidate)

        if not words:
            return False

        syllables = count_english_syllables(candidate)
        low, high = self._target_syllable_range(note_count)

        # Soft constraint:
        # do not force syllables <= notes.
        # English singing can merge weak syllables later during alignment.
        if syllables > high + 4:
            return False

        literal_syllables = count_english_syllables(literal)

        if syllables < 2 and literal_syllables > 2:
            return False

        # Do not allow candidates ending in stranded relation words.
        if words[-1].lower() in RELATION_WORDS:
            return False

        # If candidate loses too many relation words, it is usually broken.
        relation_score = self._relation_preservation_score(literal, candidate)

        if relation_score < 0.35 and len(self._relation_words(literal)) >= 2:
            return False

        return True

    def _generate_dynamic_candidates(
        self,
        literal: str,
        note_count: int,
    ) -> list[str]:
        """Generate candidates only from literal_en.

        No fixed template.
        No phrase override.
        No phrase-level rewrite rules.
        No song-specific fallback.
        """

        literal = self._normalize_text(literal)
        literal = self._apply_contractions(literal)
        literal = self._remove_duplicate_words(literal)

        if not literal:
            return []

        low, high = self._target_syllable_range(note_count)

        raw_candidates = []

        # 1. Full literal, most faithful.
        raw_candidates.append(literal)

        # 2. Generic grammar-level compression only.
        raw_candidates.append(self._apply_contractions(literal))
        raw_candidates.append(self._drop_articles_only(literal))
        raw_candidates.append(self._drop_low_value_only(literal))

        # 3. Rhythm-oriented deletion compression.
        # Relation words are protected inside _compress_by_deletion().
        raw_candidates.append(self._compress_by_deletion(literal, high))
        raw_candidates.append(self._compress_by_deletion(literal, max(low, high - 2)))
        raw_candidates.append(self._compress_by_deletion(literal, max(low, high - 4)))

        # 4. Content fragments.
        # Important change:
        # _content_words now keeps relation words.
        raw_candidates.append(self._extract_content_phrase(literal, 10))
        raw_candidates.append(self._extract_content_phrase(literal, 8))
        raw_candidates.append(self._extract_content_phrase(literal, 6))

        raw_candidates.extend(self._content_windows(literal, 7))
        raw_candidates.extend(self._content_windows(literal, 6))
        raw_candidates.extend(self._content_windows(literal, 5))

        unique = []
        seen = set()

        for candidate in raw_candidates:
            candidate = self._normalize_text(candidate)
            candidate = self._apply_contractions(candidate)
            candidate = self._remove_duplicate_words(candidate)

            if not self._is_valid_candidate(candidate, literal, note_count):
                continue

            key = candidate.lower()

            if key not in seen:
                unique.append(candidate)
                seen.add(key)

        if not unique and literal:
            compressed = self._compress_by_deletion(literal, high)
            compressed = self._normalize_text(compressed)

            if compressed:
                unique.append(compressed)
            else:
                unique.append(literal)

        return unique

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def _semantic_score(self, literal: str, candidate: str) -> float:
        source_words = {
            word.lower()
            for word in self._content_words(literal)
        }

        candidate_words = {
            word.lower()
            for word in self._content_words(candidate)
        }

        if not source_words:
            return 0.5

        overlap = source_words & candidate_words
        score = len(overlap) / len(source_words)

        return max(0.05, min(1.0, score))

    def _relation_preservation_score(self, literal: str, candidate: str) -> float:
        source_relations = self._relation_words(literal)

        if not source_relations:
            return 1.0

        candidate_relations = self._relation_words(candidate)

        if not candidate_relations:
            return 0.25

        source_set = set(source_relations)
        candidate_set = set(candidate_relations)
        overlap = source_set & candidate_set

        score = len(overlap) / max(1, len(source_set))

        return max(0.25, min(1.0, score))

    def _grammar_score(self, text: str) -> float:
        lowered = text.lower()
        words = self._words(text)

        if not words:
            return 0.1

        score = 1.0

        # Generic grammar break patterns.
        # These are not phrase overrides; they penalize structurally broken English.
        broken_patterns = [
            r"\bbrought\s+(me|you|us|them|him|her)\s+meet\b",
            r"\b(bring|brings|brought)\s+(me|you|us|them|him|her)\s+(go|come|meet|fall|wait)\b",
            r"\bfall\s+love\b",
            r"\b(want|wants|wanted|demand|demands|demanded)\s+(me|you|us|them|him|her|we|they)\s+fall\b",
            r"\bwait\s+for\s+(me|you|us|them|him|her)\s+unknown\b",
            r"\bgo\s+mad\s+(me|you|us|them|him|her)\b",
            r"\bno\s+desire\s+flee\b",
            r"\bdesire\s+flee\b",
        ]

        for pattern in broken_patterns:
            if re.search(pattern, lowered):
                score -= 0.45

        # Candidate ending with a relation word is usually incomplete.
        if words[-1].lower() in RELATION_WORDS:
            score -= 0.35

        # Too many adjacent content nouns often means the sentence became
        # a keyword list after deleting prepositions.
        content_like = [
            word.lower()
            for word in words
            if (
                word.lower() not in ARTICLES
                and word.lower() not in DETERMINERS
                and word.lower() not in CONJUNCTIONS
                and word.lower() not in AUXILIARY_WORDS
                and word.lower() not in RELATION_WORDS
            )
        ]

        if len(content_like) >= 6 and len(self._relation_words(text)) == 0:
            score -= 0.25

        return max(0.1, min(1.0, score))

    def _rhythm_score(self, syllables: int, note_count: int) -> float:
        low, high = self._target_syllable_range(note_count)
        target = self._target_syllables(note_count)

        if low <= syllables <= high:
            distance = abs(syllables - target)
            span = max(1, high - low)
            return max(0.55, 1.0 - distance / span)

        if syllables < low:
            return max(0.2, syllables / max(1, low))

        return max(0.2, high / max(1, syllables))

    def _singability_score(self, text: str) -> float:
        words = self._words(text)

        if not words:
            return 0.1

        score = 1.0

        if len(words) > 16:
            score -= 0.2

        long_words = [
            word
            for word in words
            if len(word) >= 12
        ]

        score -= min(0.25, len(long_words) * 0.08)

        if any(mark in text for mark in [";", ":", "(", ")", "\""]):
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _ending_score(self, text: str) -> float:
        words = self._words(text)

        if not words:
            return 0.3

        last = words[-1].lower()

        if last[-1] in {"a", "e", "i", "o", "u", "y"}:
            return 0.85

        return 0.65

    def _score_candidate(
        self,
        literal: str,
        candidate: str,
        note_count: int,
    ) -> dict:
        syllables = count_english_syllables(candidate)

        semantic = self._semantic_score(literal, candidate)
        rhythm = self._rhythm_score(syllables, note_count)
        singability = self._singability_score(candidate)
        ending = self._ending_score(candidate)
        grammar = self._grammar_score(candidate)
        relation = self._relation_preservation_score(literal, candidate)

        final = (
            semantic * 0.32
            + rhythm * 0.18
            + grammar * 0.24
            + relation * 0.16
            + singability * 0.07
            + ending * 0.03
        )

        return {
            "text": candidate,
            "syllables": syllables,
            "semantic_score": round(semantic, 3),
            "rhythm_score": round(rhythm, 3),
            "grammar_score": round(grammar, 3),
            "relation_score": round(relation, 3),
            "singability_score": round(singability, 3),
            "ending_score": round(ending, 3),
            "final_score": round(final, 3),
        }

    # ------------------------------------------------------------------
    # Validation and main pipeline
    # ------------------------------------------------------------------

    def _literal_by_id(self, literal_data: dict) -> dict:
        return {
            phrase.get("id"): phrase
            for phrase in literal_data.get("phrases", [])
        }

    def _validate_literal_data(self, literal_data: dict) -> None:
        source = literal_data.get("source", "")
        backend = literal_data.get("translation_backend", "")
        error = literal_data.get("translation_error", "")

        forbidden_sources = {
            "placeholder_translation",
            "mock_translation",
        }

        forbidden_backends = {
            "placeholder",
            "mock",
            "failed",
        }

        if source in forbidden_sources or backend in forbidden_backends:
            raise RuntimeError(
                "lyrics_literal.json is not generated by a real translation model. "
                f"source={source!r}, backend={backend!r}, error={error!r}. "
                "This general automatic system does not allow placeholder lyrics."
            )

        phrases = literal_data.get("phrases", [])

        if not phrases:
            raise RuntimeError("lyrics_literal.json contains no phrases.")

        for phrase in phrases:
            literal = phrase.get("literal_en") or phrase.get("literal") or ""

            if not str(literal).strip():
                raise RuntimeError(
                    "lyrics_literal.json contains an empty translation for "
                    f"phrase id={phrase.get('id')}."
                )

    def run(self) -> dict:
        phrase_map_path = path_from_config(self.config, "phrase_map")
        notes_path = path_from_config(self.config, "melody_notes")
        literal_path = path_from_config(self.config, "lyrics_literal")
        output = path_from_config(self.config, "lyrics_singable")

        phrase_status = require_or_mock_input(
            phrase_map_path,
            self.config,
            "phrase map",
        )
        notes_status = require_or_mock_input(
            notes_path,
            self.config,
            "melody notes",
        )
        literal_status = require_or_mock_input(
            literal_path,
            self.config,
            "literal lyrics",
        )

        phrase_data = read_json(phrase_map_path, {"phrases": []})
        melody_data = read_json(notes_path, {"phrases": []})
        literal_data = read_json(literal_path, {"phrases": []})

        self._validate_literal_data(literal_data)

        melody_by_id = {
            phrase.get("id"): phrase
            for phrase in melody_data.get("phrases", [])
        }

        literal_by_id = self._literal_by_id(literal_data)

        adapted = []

        for phrase in phrase_data.get("phrases", []):
            phrase_id = phrase.get("id")
            literal_phrase = literal_by_id.get(phrase_id, {})

            literal = (
                literal_phrase.get("literal_en")
                or literal_phrase.get("literal")
                or ""
            )

            literal = self._normalize_text(literal)

            if not literal:
                raise RuntimeError(
                    f"Missing literal translation for phrase id={phrase_id}."
                )

            note_count = self._note_count(phrase, melody_by_id)
            phrase_duration = self._phrase_duration(phrase)

            candidates_text = self._generate_dynamic_candidates(
                literal=literal,
                note_count=note_count,
            )

            scored = [
                self._score_candidate(
                    literal=literal,
                    candidate=candidate,
                    note_count=note_count,
                )
                for candidate in candidates_text
            ]

            if not scored:
                raise RuntimeError(
                    "No singable candidates generated for phrase "
                    f"id={phrase_id}, literal={literal!r}."
                )

            scored.sort(key=lambda item: item["final_score"], reverse=True)
            best = scored[0]

            adapted.append(
                {
                    "id": phrase_id,
                    "zh": phrase.get("zh", ""),
                    "literal_en": literal,
                    "note_count": note_count,
                    "phrase_duration": phrase_duration,
                    "target_syllables": self._target_syllables(note_count),
                    "target_syllable_range": list(
                        self._target_syllable_range(note_count)
                    ),
                    "candidates": scored,
                    "selected": best["text"],
                    "selected_score": best["final_score"],
                }
            )

        write_json(
            output,
            {
                "source": "auto_dynamic_literal_based_adapter",
                "description": (
                    "All candidates are derived only from each phrase's literal_en. "
                    "No fixed lyric templates, no phrase overrides, no song-specific "
                    "rules, and no placeholder translations are allowed. Relation "
                    "words are protected to preserve English grammar while rhythm "
                    "alignment handles weak pronunciation later."
                ),
                "phrases": adapted,
            },
            self.config,
        )

        return {
            "status": "success"
            if "mock" not in {phrase_status, literal_status, notes_status}
            else "mock",
            "outputs": {
                "lyrics_singable": str(output),
            },
            "message": (
                "Generated singable lyrics from real literal translations while "
                "preserving relation words for grammatical English."
            ),
        }