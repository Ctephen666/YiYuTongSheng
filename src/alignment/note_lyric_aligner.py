from __future__ import annotations

from src.common.io_utils import path_from_config, write_text
from src.common.json_utils import read_json, write_json


VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
LONG_VOWELS = {"IY", "UW", "ER", "AO"}
DIPHTHONGS = {"AY", "AW", "OY", "OW", "EY"}

WEAK_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "with", "and", "or", "but", "as",
    "by", "that", "this", "these", "those", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "my", "your", "his", "her", "its", "our", "their", "me", "you", "him", "them", "us",
}

IMPORTANT_SHORT_WORDS = {"i", "me", "my", "mine", "you", "your", "we", "us", "no", "not"}


class NoteLyricAligner:
    """Align melody notes to English syllable groups with singing-aware duration allocation."""

    def __init__(self, config: dict):
        self.config = config

    def _alignment_config(self) -> dict:
        return self.config.get("alignment", {})

    def _cfg(self, key: str, default):
        return self._alignment_config().get(key, default)

    def _phrase_by_id(self, data: dict) -> dict:
        result = {}
        for phrase in data.get("phrases", []):
            phrase_id = phrase.get("id")
            result[phrase_id] = phrase
            result[str(phrase_id)] = phrase
        return result

    def _get_notes(self, melody_phrase: dict) -> list[dict]:
        notes = melody_phrase.get("notes", [])
        return notes if isinstance(notes, list) else []

    def _get_units(self, phoneme_phrase: dict) -> list[dict]:
        units = phoneme_phrase.get("units", [])
        return units if isinstance(units, list) else []

    def _note_start(self, note: dict) -> float:
        try:
            return float(note.get("start", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _note_end(self, note: dict) -> float:
        start = self._note_start(note)
        try:
            end = float(note.get("end", start))
        except (TypeError, ValueError):
            end = start
        if end <= start:
            try:
                end = start + float(note.get("duration", 0.0))
            except (TypeError, ValueError):
                end = start
        return end

    def _note_duration(self, note: dict) -> float:
        duration = self._note_end(note) - self._note_start(note)
        if duration <= 0:
            try:
                duration = float(note.get("duration", 0.0))
            except (TypeError, ValueError):
                duration = 0.0
        return max(0.0, duration)

    def _normalize_phone(self, phone: str) -> str:
        return str(phone or "").strip().upper().rstrip("0123456789")

    def _contains_alpha(self, text: str) -> bool:
        return any(ch.isalpha() for ch in str(text or ""))

    def _unit_text(self, unit: dict) -> str:
        return str(unit.get("unit", "") or "").strip()

    def _word_text(self, unit: dict) -> str:
        return str(unit.get("word", "") or "").strip()

    def _word_tokens(self, word: str) -> list[str]:
        cleaned = str(word or "").replace("+", " ").replace("-", " ")
        return [part.lower() for part in cleaned.split() if part.strip()]

    def _phonemes(self, unit: dict) -> list[str]:
        phonemes = unit.get("phonemes", [])
        return [str(phone) for phone in phonemes] if isinstance(phonemes, list) else []

    def _vowel_phones(self, phonemes: list[str]) -> list[str]:
        return [self._normalize_phone(phone) for phone in phonemes if self._normalize_phone(phone) in VOWELS]

    def _has_stress_mark(self, phonemes: list[str]) -> bool:
        return any("1" in str(phone) or "2" in str(phone) for phone in phonemes)

    def _is_weak_text(self, unit_text: str, word_text: str) -> bool:
        tokens = self._word_tokens(word_text) or self._word_tokens(unit_text)
        if not tokens:
            return True
        if all(token in WEAK_WORDS for token in tokens):
            return True
        compact_unit = str(unit_text or "").lower().strip()
        compact_word = str(word_text or "").lower().strip()
        if compact_unit in WEAK_WORDS or compact_word in WEAK_WORDS:
            return True
        if len(compact_unit) <= 2 and compact_unit not in IMPORTANT_SHORT_WORDS and compact_word not in IMPORTANT_SHORT_WORDS:
            return True
        return False

    def _single_unit_features(self, unit: dict) -> dict:
        unit_text = self._unit_text(unit)
        word_text = self._word_text(unit)
        phonemes = self._phonemes(unit)
        vowel_phones = self._vowel_phones(phonemes)
        is_weak = self._is_weak_text(unit_text, word_text)
        is_content_word = (not is_weak) and self._contains_alpha(word_text or unit_text)
        has_stress = self._has_stress_mark(phonemes)
        is_stressed = has_stress or (is_content_word and not any(ch.isdigit() for phone in phonemes for ch in str(phone)))
        return {
            "unit": unit_text,
            "word": word_text,
            "phonemes": phonemes,
            "vowel_phones": vowel_phones,
            "main_vowel": vowel_phones[0] if vowel_phones else "",
            "has_vowel": bool(vowel_phones),
            "is_weak": is_weak,
            "is_content_word": is_content_word,
            "is_stressed": is_stressed,
        }

    def _compute_weight_from_features(self, features: dict) -> float:
        weight = 1.0
        if features.get("is_weak"):
            weight *= float(self._cfg("weak_word_weight", 0.45))
        if features.get("is_content_word"):
            weight *= float(self._cfg("content_word_weight", 1.25))
        if features.get("is_stressed"):
            weight *= float(self._cfg("stressed_syllable_weight", 1.30))
        vowels = set(features.get("vowel_phones", []))
        if vowels & LONG_VOWELS:
            weight *= float(self._cfg("long_vowel_weight", 1.15))
        if vowels & DIPHTHONGS:
            weight *= float(self._cfg("diphthong_weight", 1.20))
        if not features.get("has_vowel"):
            weight *= 0.6
        return min(2.5, max(0.25, float(weight)))

    def _join_units(self, units: list[dict]) -> str:
        return "-".join(part for part in [self._unit_text(unit) for unit in units] if part)

    def _join_words(self, units: list[dict]) -> str:
        words = []
        previous = None
        for unit in units:
            word = self._word_text(unit)
            if word and word != previous:
                words.append(word)
                previous = word
        return " ".join(words)

    def _collect_phonemes(self, units: list[dict]) -> list[str]:
        phonemes = []
        for unit in units:
            phonemes.extend(self._phonemes(unit))
        return phonemes

    def _refresh_group(self, group: dict) -> dict:
        units = group.get("units", [])
        group["unit"] = self._join_units(units)
        group["word"] = self._join_words(units)
        group["phonemes"] = self._collect_phonemes(units)
        features = self._single_unit_features(group)
        single_weights = [self._compute_weight_from_features(self._single_unit_features(unit)) for unit in units]
        weight = sum(single_weights) * 0.85 if len(units) > 1 else self._compute_weight_from_features(features)
        group.update(features)
        group["singing_weight"] = min(2.5, max(0.25, float(weight)))
        return group

    def _make_alignment_groups(self, units: list[dict]) -> list[dict]:
        return [self._refresh_group({"units": [unit]}) for unit in units]

    def _merge_group_into(self, groups: list[dict], src_index: int, dst_index: int) -> None:
        if src_index == dst_index or src_index < 0 or dst_index < 0 or src_index >= len(groups) or dst_index >= len(groups):
            return
        src_units = groups[src_index].get("units", [])
        dst_units = groups[dst_index].get("units", [])
        groups[dst_index]["units"] = src_units + dst_units if src_index < dst_index else dst_units + src_units
        self._refresh_group(groups[dst_index])
        groups.pop(src_index)

    def _merge_pair(self, groups: list[dict], left_index: int) -> None:
        left_units = groups[left_index].get("units", [])
        right_units = groups[left_index + 1].get("units", [])
        groups[left_index]["units"] = left_units + right_units
        self._refresh_group(groups[left_index])
        groups.pop(left_index + 1)

    def _merge_weak_units_once(self, groups: list[dict]) -> bool:
        for index, group in enumerate(groups):
            if not group.get("is_weak"):
                continue
            left = groups[index - 1] if index > 0 else None
            right = groups[index + 1] if index < len(groups) - 1 else None
            if right and right.get("is_content_word"):
                self._merge_group_into(groups, index, index + 1)
            elif left and left.get("is_content_word") and (right is None or right.get("is_weak")):
                self._merge_group_into(groups, index, index - 1)
            elif right:
                self._merge_group_into(groups, index, index + 1)
            elif left:
                self._merge_group_into(groups, index, index - 1)
            else:
                return False
            return True
        return False

    def _merge_cost(self, left: dict, right: dict) -> float:
        cost = 1.0
        if left.get("is_weak") or right.get("is_weak"):
            cost -= 0.4
        left_words = set(self._word_tokens(left.get("word", "")))
        right_words = set(self._word_tokens(right.get("word", "")))
        if left_words and left_words == right_words:
            cost -= 0.3
        if left.get("is_content_word") and right.get("is_content_word"):
            cost += 0.8
        if left.get("is_stressed") and right.get("is_stressed"):
            cost += 0.6
        if not left.get("has_vowel") or not right.get("has_vowel"):
            cost -= 0.2
        if len(left.get("phonemes", [])) + len(right.get("phonemes", [])) > 10:
            cost += 0.2
        return cost

    def _merge_lowest_cost_pair_once(self, groups: list[dict]) -> bool:
        if len(groups) <= 1:
            return False
        best_index = min(range(len(groups) - 1), key=lambda index: self._merge_cost(groups[index], groups[index + 1]))
        self._merge_pair(groups, best_index)
        return True

    def _merge_overflow_units(self, units: list[dict], note_count: int) -> tuple[list[dict], dict]:
        groups = self._make_alignment_groups(units)
        original_unit_count = len(groups)
        weak_merge_count = 0
        forced_merge_count = 0
        if note_count <= 0:
            raise RuntimeError("Cannot merge units because note_count is 0.")
        while len(groups) > note_count:
            if not self._merge_weak_units_once(groups):
                break
            weak_merge_count += 1
        while len(groups) > note_count:
            if not self._merge_lowest_cost_pair_once(groups):
                break
            forced_merge_count += 1
        merge_info = {
            "original_unit_count": original_unit_count,
            "aligned_unit_count": len(groups),
            "note_count": note_count,
            "weak_merge_count": weak_merge_count,
            "forced_merge_count": forced_merge_count,
        }
        return groups, merge_info

    def _split_notes_evenly(self, notes: list[dict], alignment_units: list[dict]) -> list[list[dict]]:
        note_count = len(notes)
        unit_count = len(alignment_units)
        groups = []
        for index in range(unit_count):
            start = round(index * note_count / max(unit_count, 1))
            end = round((index + 1) * note_count / max(unit_count, 1))
            if index == unit_count - 1:
                end = note_count
            groups.append(notes[start:end])
        return groups

    def _split_notes_duration_aware(self, notes: list[dict], alignment_units: list[dict]) -> list[list[dict]]:
        note_count = len(notes)
        unit_count = len(alignment_units)
        if unit_count == 0 or note_count == 0:
            raise RuntimeError("Cannot align because units or notes are empty.")
        if unit_count == note_count:
            return [[note] for note in notes]
        if unit_count > note_count:
            return self._split_notes_evenly(notes, alignment_units)
        total_weight = sum(float(unit.get("singing_weight", 1.0)) for unit in alignment_units) or float(unit_count)
        total_duration = sum(self._note_duration(note) for note in notes) or float(note_count)
        groups = []
        note_index = 0
        for unit_index, unit in enumerate(alignment_units):
            remaining_units = unit_count - unit_index
            remaining_notes = note_count - note_index
            if remaining_units <= 1:
                groups.append(notes[note_index:])
                break
            max_take = max(1, remaining_notes - (remaining_units - 1))
            expected_duration = total_duration * float(unit.get("singing_weight", 1.0)) / total_weight
            take = 1
            current_duration = self._note_duration(notes[note_index])
            while take < max_take and current_duration < expected_duration:
                current_duration += self._note_duration(notes[note_index + take])
                take += 1
            groups.append(notes[note_index: note_index + take])
            note_index += take
        while len(groups) < unit_count:
            groups.append([])
        return groups

    def _estimate_vowel_region(self, item: dict, slot_start: float, slot_end: float) -> dict:
        slot_duration = max(0.0, slot_end - slot_start)
        phonemes = item.get("phonemes", [])
        normalized = [self._normalize_phone(phone) for phone in phonemes]
        vowel_indices = [index for index, phone in enumerate(normalized) if phone in VOWELS]
        if slot_duration <= 0:
            return {
                "estimated_onset_ratio": 0.0,
                "estimated_nucleus_ratio": 0.0,
                "estimated_coda_ratio": 0.0,
                "recommended_vowel_start": slot_start,
                "recommended_vowel_end": slot_end,
            }
        if vowel_indices:
            nucleus_ratio = float(self._cfg("vowel_center_ratio_default", 0.65))
            center = (vowel_indices[0] + 0.5) / max(len(normalized), 1)
        else:
            nucleus_ratio = 0.4
            center = 0.5
        nucleus_duration = slot_duration * min(max(nucleus_ratio, 0.1), 0.9)
        vowel_start = slot_start + center * slot_duration - nucleus_duration / 2.0
        vowel_start = min(max(vowel_start, slot_start), slot_end - nucleus_duration)
        vowel_end = vowel_start + nucleus_duration
        onset_ratio = (vowel_start - slot_start) / slot_duration
        actual_nucleus_ratio = (vowel_end - vowel_start) / slot_duration
        coda_ratio = max(0.0, 1.0 - onset_ratio - actual_nucleus_ratio)
        return {
            "estimated_onset_ratio": onset_ratio,
            "estimated_nucleus_ratio": actual_nucleus_ratio,
            "estimated_coda_ratio": coda_ratio,
            "recommended_vowel_start": vowel_start,
            "recommended_vowel_end": vowel_end,
        }

    def _item_from_group(self, group: dict, notes: list[dict]) -> dict:
        if notes:
            slot_start = min(self._note_start(note) for note in notes)
            slot_end = max(self._note_end(note) for note in notes)
        else:
            slot_start = 0.0
            slot_end = 0.0
        slot_duration = max(0.0, slot_end - slot_start)
        item = {
            "unit": group.get("unit", ""),
            "word": group.get("word", ""),
            "phonemes": group.get("phonemes", []),
            "merged_units": group.get("units", []),
            "vowel_phones": group.get("vowel_phones", []),
            "main_vowel": group.get("main_vowel", ""),
            "has_vowel": bool(group.get("has_vowel")),
            "is_weak": bool(group.get("is_weak")),
            "is_content_word": bool(group.get("is_content_word")),
            "is_stressed": bool(group.get("is_stressed")),
            "singing_weight": float(group.get("singing_weight", 1.0)),
            "slot_start": slot_start,
            "slot_end": slot_end,
            "slot_duration": slot_duration,
            "note_count": len(notes),
            "note_duration_sum": sum(self._note_duration(note) for note in notes),
            "notes": notes,
        }
        item.update(self._estimate_vowel_region(item, slot_start, slot_end))
        return item

    def _alignment_quality(self, raw_units: list[dict], items: list[dict], notes: list[dict], merge_info: dict) -> dict:
        raw_count = len(raw_units)
        forced_merge_count = int(merge_info.get("forced_merge_count", 0) or 0)
        forced_ratio = forced_merge_count / max(raw_count, 1)
        max_forced_ratio = float(self._cfg("max_forced_merge_ratio", 0.35))
        if forced_merge_count == 0 and len(items) <= len(notes):
            risk = "low"
        elif forced_ratio <= max_forced_ratio:
            risk = "medium"
        else:
            risk = "high"
        total_note_duration = sum(self._note_duration(note) for note in notes)
        return {
            "raw_unit_count": raw_count,
            "aligned_unit_count": len(items),
            "note_count": len(notes),
            "weak_merge_count": int(merge_info.get("weak_merge_count", 0) or 0),
            "forced_merge_count": forced_merge_count,
            "total_note_duration": total_note_duration,
            "average_note_duration": total_note_duration / max(len(notes), 1),
            "content_word_count": sum(1 for item in items if item.get("is_content_word")),
            "weak_word_count": sum(1 for item in items if item.get("is_weak")),
            "risk_level": risk,
        }

    def _build_preview(self, aligned_phrases: list[dict]) -> str:
        lines = []
        for phrase in aligned_phrases:
            quality = phrase.get("alignment_quality", {})
            lines.append(f"Phrase {phrase.get('id')}: {phrase.get('text', '')}")
            lines.append(
                f"Notes: {quality.get('note_count')}, Raw units: {quality.get('raw_unit_count')}, "
                f"Aligned units: {quality.get('aligned_unit_count')}, Risk: {quality.get('risk_level')}"
            )
            for index, item in enumerate(phrase.get("items", [])):
                midi_values = [note.get("midi") for note in item.get("notes", [])]
                lines.append(
                    f"[{index}] unit={item.get('unit')} | word={item.get('word')} | weak={item.get('is_weak')} | "
                    f"stressed={item.get('is_stressed')} | weight={float(item.get('singing_weight', 0.0)):.2f}"
                )
                lines.append(
                    f"    notes={len(item.get('notes', []))} | slot={float(item.get('slot_start', 0.0)):.2f}-"
                    f"{float(item.get('slot_end', 0.0)):.2f} | vowel={float(item.get('recommended_vowel_start', 0.0)):.2f}-"
                    f"{float(item.get('recommended_vowel_end', 0.0)):.2f} | midi={midi_values}"
                )
            lines.append("")
        return "\n".join(lines)

    def run(self) -> dict:
        melody_path = path_from_config(self.config, "melody_notes")
        phoneme_path = path_from_config(self.config, "phonemes_target")
        output = path_from_config(self.config, "note_lyric_alignment")
        melody_data = read_json(melody_path, {"phrases": []})
        phoneme_data = read_json(phoneme_path, {"phrases": []})
        melody_by_id = self._phrase_by_id(melody_data)
        aligned_phrases = []

        for phoneme_phrase in phoneme_data.get("phrases", []):
            phrase_id = phoneme_phrase.get("id")
            melody_phrase = melody_by_id.get(phrase_id) or melody_by_id.get(str(phrase_id))
            if not melody_phrase:
                raise RuntimeError(f"Missing melody phrase for id={phrase_id}.")
            notes = self._get_notes(melody_phrase)
            raw_units = self._get_units(phoneme_phrase)
            if not notes:
                raise RuntimeError(f"Missing notes for phrase id={phrase_id}.")
            if not raw_units:
                raise RuntimeError(f"Missing syllable units for phrase id={phrase_id}.")
            try:
                alignment_units, merge_info = self._merge_overflow_units(raw_units, note_count=len(notes))
                if bool(self._cfg("enable_duration_aware_assignment", True)):
                    note_groups = self._split_notes_duration_aware(notes, alignment_units)
                else:
                    note_groups = self._split_notes_evenly(notes, alignment_units)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Alignment failed at phrase id={phrase_id}, zh={phoneme_phrase.get('zh', '')!r}, "
                    f"text={phoneme_phrase.get('text', '')!r}. {exc}"
                ) from exc
            items = [self._item_from_group(group, group_notes) for group, group_notes in zip(alignment_units, note_groups)]
            quality = self._alignment_quality(raw_units, items, notes, merge_info)
            aligned_phrases.append({
                "id": phrase_id,
                "zh": phoneme_phrase.get("zh", ""),
                "text": phoneme_phrase.get("text", ""),
                "unit_type": "syllable_group",
                "alignment_strategy": "singing_duration_aware",
                "status": "success",
                "unit_count": len(alignment_units),
                "raw_unit_count": len(raw_units),
                "note_count": len(notes),
                "merge_info": merge_info,
                "alignment_quality": quality,
                "items": items,
            })

        result = {
            "unit_type": "syllable_group",
            "alignment_strategy": "singing_duration_aware",
            "description": "English syllable-level alignment with singing weights, duration-aware note assignment, and estimated vowel regions.",
            "phrases": aligned_phrases,
        }
        write_json(output, result, self.config)
        preview_text = self._build_preview(aligned_phrases)
        legacy_preview = output.parent / "alignment_preview.txt"
        singing_preview = output.parent / "alignment_singing_preview.txt"
        write_text(legacy_preview, preview_text, self.config)
        write_text(singing_preview, preview_text, self.config)
        return {
            "status": "success",
            "outputs": {
                "note_lyric_alignment": str(output),
                "alignment_preview": str(legacy_preview),
                "alignment_singing_preview": str(singing_preview),
            },
            "message": "Aligned melody notes to English syllable groups with singing-aware duration allocation.",
        }