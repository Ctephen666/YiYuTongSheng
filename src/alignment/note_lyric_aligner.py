from __future__ import annotations

from src.common.io_utils import path_from_config, write_text
from src.common.json_utils import read_json, write_json


WEAK_WORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "as",
    "by",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
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


IMPORTANT_SHORT_WORDS = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "we",
    "us",
    "no",
    "not",
}


class NoteLyricAligner:
    """Align melody notes to English syllable-level lyric units.

    This version uses soft syllable alignment:

    1. If syllable unit count <= note count:
       align directly.

    2. If syllable unit count > note count:
       merge weak syllables first, such as:
           to / the / a / in / of / for / and

    3. If still too many:
       merge shortest neighboring units.

    This better matches English singing, where weak function words can be
    reduced, linked, or sung quickly within the same note group.
    """

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Basic loading helpers
    # ------------------------------------------------------------------

    def _melody_by_id(self, melody_data: dict) -> dict:
        return {
            phrase.get("id"): phrase
            for phrase in melody_data.get("phrases", [])
        }

    def _phoneme_by_id(self, phoneme_data: dict) -> dict:
        return {
            phrase.get("id"): phrase
            for phrase in phoneme_data.get("phrases", [])
        }

    def _get_notes(self, melody_phrase: dict) -> list[dict]:
        notes = melody_phrase.get("notes", [])

        if not isinstance(notes, list):
            return []

        return notes

    def _get_units(self, phoneme_phrase: dict) -> list[dict]:
        units = phoneme_phrase.get("units", [])

        if not isinstance(units, list):
            return []

        return units

    # ------------------------------------------------------------------
    # Weak-unit merging
    # ------------------------------------------------------------------

    def _unit_text(self, unit: dict) -> str:
        return str(unit.get("unit", "") or "").lower().strip()

    def _word_text(self, unit: dict) -> str:
        return str(unit.get("word", "") or "").lower().strip()

    def _is_weak_unit(self, unit: dict) -> bool:
        unit_text = self._unit_text(unit)
        word_text = self._word_text(unit)

        if not unit_text and not word_text:
            return True

        if word_text in WEAK_WORDS:
            return True

        if unit_text in WEAK_WORDS:
            return True

        # Very short syllables are often reducible in English singing,
        # but do not merge important pronouns / negation blindly.
        if (
            len(unit_text) <= 2
            and unit_text not in IMPORTANT_SHORT_WORDS
            and word_text not in IMPORTANT_SHORT_WORDS
        ):
            return True

        return False

    def _join_units(self, units: list[dict]) -> str:
        parts = []

        for unit in units:
            text = str(unit.get("unit", "") or "").strip()

            if text:
                parts.append(text)

        return "-".join(parts)

    def _join_words(self, units: list[dict]) -> str:
        words = []

        previous = None

        for unit in units:
            word = str(unit.get("word", "") or "").strip()

            if not word:
                continue

            # Avoid output like hairline+hairline for syllables of same word.
            if word == previous:
                continue

            words.append(word)
            previous = word

        return "+".join(words)

    def _collect_phonemes(self, units: list[dict]) -> list[str]:
        phonemes = []

        for unit in units:
            unit_phonemes = unit.get("phonemes", [])

            if isinstance(unit_phonemes, list):
                phonemes.extend(unit_phonemes)

        return phonemes

    def _refresh_group(self, group: dict) -> dict:
        units = group.get("units", [])

        group["unit"] = self._join_units(units)
        group["word"] = self._join_words(units)
        group["phonemes"] = self._collect_phonemes(units)

        return group

    def _make_alignment_groups(self, units: list[dict]) -> list[dict]:
        groups = []

        for unit in units:
            group = {
                "units": [unit],
                "unit": str(unit.get("unit", "") or ""),
                "word": str(unit.get("word", "") or ""),
                "phonemes": unit.get("phonemes", []),
            }
            groups.append(self._refresh_group(group))

        return groups

    def _merge_group_into(
        self,
        groups: list[dict],
        src_index: int,
        dst_index: int,
    ) -> None:
        """Merge src group into dst group while preserving original order."""

        if src_index == dst_index:
            return

        if src_index < 0 or src_index >= len(groups):
            return

        if dst_index < 0 or dst_index >= len(groups):
            return

        src_units = groups[src_index].get("units", [])
        dst_units = groups[dst_index].get("units", [])

        if src_index < dst_index:
            merged_units = src_units + dst_units
        else:
            merged_units = dst_units + src_units

        groups[dst_index]["units"] = merged_units
        self._refresh_group(groups[dst_index])
        groups.pop(src_index)

    def _merge_weak_units_once(self, groups: list[dict]) -> bool:
        """Merge one weak group into its neighbor.

        Returns True if a merge happened.
        """

        for index, group in enumerate(groups):
            group_units = group.get("units", [])

            if not group_units:
                continue

            if not any(self._is_weak_unit(unit) for unit in group_units):
                continue

            # Prefer attaching weak unit to the next group.
            # If it is at the end, attach it to the previous group.
            if index < len(groups) - 1:
                self._merge_group_into(groups, index, index + 1)
            elif index > 0:
                self._merge_group_into(groups, index, index - 1)
            else:
                return False

            return True

        return False

    def _merge_shortest_group_once(self, groups: list[dict]) -> bool:
        """Merge the shortest group into a neighbor when weak merging is not enough."""

        if len(groups) <= 1:
            return False

        shortest_index = min(
            range(len(groups)),
            key=lambda i: len(str(groups[i].get("unit", "") or "")),
        )

        if shortest_index < len(groups) - 1:
            self._merge_group_into(groups, shortest_index, shortest_index + 1)
        else:
            self._merge_group_into(groups, shortest_index, shortest_index - 1)

        return True

    def _merge_overflow_units(
        self,
        units: list[dict],
        note_count: int,
    ) -> tuple[list[dict], dict]:
        """Merge syllable units until group count fits available notes."""

        groups = self._make_alignment_groups(units)

        original_unit_count = len(groups)
        weak_merge_count = 0
        forced_merge_count = 0

        if note_count <= 0:
            raise RuntimeError("Cannot merge units because note_count is 0.")

        # If there are more syllable units than notes, merge weak units first.
        while len(groups) > note_count:
            merged = self._merge_weak_units_once(groups)

            if not merged:
                break

            weak_merge_count += 1

        # If still too many, merge shortest groups.
        while len(groups) > note_count:
            merged = self._merge_shortest_group_once(groups)

            if not merged:
                break

            forced_merge_count += 1

        if len(groups) > note_count:
            raise RuntimeError(
                "Unable to merge lyric units enough for alignment: "
                f"original_unit_count={original_unit_count}, "
                f"merged_unit_count={len(groups)}, "
                f"note_count={note_count}."
            )

        merge_info = {
            "original_unit_count": original_unit_count,
            "aligned_unit_count": len(groups),
            "note_count": note_count,
            "weak_merge_count": weak_merge_count,
            "forced_merge_count": forced_merge_count,
        }

        return groups, merge_info

    # ------------------------------------------------------------------
    # Note distribution
    # ------------------------------------------------------------------

    def _split_notes_evenly(
        self,
        notes: list[dict],
        alignment_units: list[dict],
    ) -> list[list[dict]]:
        """Distribute notes to merged syllable groups.

        Uses proportional boundaries:
            unit i receives notes[round(i*N/U):round((i+1)*N/U)]

        The input alignment_units should already be merged so that:
            len(alignment_units) <= len(notes)
        """

        note_count = len(notes)
        unit_count = len(alignment_units)

        if unit_count == 0:
            raise RuntimeError("Cannot align because unit_count is 0.")

        if note_count == 0:
            raise RuntimeError("Cannot align because note_count is 0.")

        if unit_count > note_count:
            raise RuntimeError(
                "Internal alignment error: merged unit count is still larger "
                f"than note count. unit_count={unit_count}, note_count={note_count}"
            )

        groups = []
        previous_end = 0

        for index in range(unit_count):
            start = round(index * note_count / unit_count)
            end = round((index + 1) * note_count / unit_count)

            start = max(start, previous_end)
            end = max(end, start + 1)

            if index == unit_count - 1:
                end = note_count

            groups.append(notes[start:end])
            previous_end = end

        return groups

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _build_preview(self, aligned_phrases: list[dict]) -> str:
        lines = []

        for phrase in aligned_phrases:
            lines.append(f"[{phrase['id']}] {phrase.get('zh', '')}")
            lines.append(f"EN: {phrase.get('text', '')}")

            merge_info = phrase.get("merge_info", {})
            original_count = merge_info.get("original_unit_count")
            aligned_count = merge_info.get("aligned_unit_count")
            note_count = merge_info.get("note_count")
            weak_merge_count = merge_info.get("weak_merge_count", 0)
            forced_merge_count = merge_info.get("forced_merge_count", 0)

            lines.append(
                "ALIGN: "
                f"raw_units={original_count}, "
                f"aligned_units={aligned_count}, "
                f"notes={note_count}, "
                f"weak_merges={weak_merge_count}, "
                f"forced_merges={forced_merge_count}"
            )

            for item in phrase.get("items", []):
                unit = item.get("unit", "")
                word = item.get("word", "")
                notes = item.get("notes", [])
                merged_units = item.get("merged_units", [])

                merged_text = ""

                if len(merged_units) > 1:
                    merged_text = " [merged]"

                if not notes:
                    lines.append(f"{word}/{unit}{merged_text} -> <no notes>")
                    continue

                note_parts = []

                for note in notes:
                    pitch = note.get("pitch", "")
                    start = float(note.get("start", 0.0))
                    end = float(note.get("end", 0.0))
                    note_parts.append(f"{pitch} {start:.3f}-{end:.3f}")

                note_text = " + ".join(note_parts)
                lines.append(f"{word}/{unit}{merged_text} -> {note_text}")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> dict:
        melody_path = path_from_config(self.config, "melody_notes")
        phoneme_path = path_from_config(self.config, "phonemes_target")
        output = path_from_config(self.config, "note_lyric_alignment")

        melody_data = read_json(melody_path, {"phrases": []})
        phoneme_data = read_json(phoneme_path, {"phrases": []})

        melody_by_id = self._melody_by_id(melody_data)
        phoneme_by_id = self._phoneme_by_id(phoneme_data)

        aligned_phrases = []

        for phrase_id, phoneme_phrase in phoneme_by_id.items():
            melody_phrase = melody_by_id.get(phrase_id)

            if not melody_phrase:
                raise RuntimeError(
                    f"Missing melody phrase for id={phrase_id}."
                )

            notes = self._get_notes(melody_phrase)
            raw_units = self._get_units(phoneme_phrase)

            if not notes:
                raise RuntimeError(
                    f"Missing notes for phrase id={phrase_id}."
                )

            if not raw_units:
                raise RuntimeError(
                    f"Missing syllable units for phrase id={phrase_id}."
                )

            try:
                alignment_units, merge_info = self._merge_overflow_units(
                    raw_units,
                    note_count=len(notes),
                )

                note_groups = self._split_notes_evenly(
                    notes,
                    alignment_units,
                )

            except RuntimeError as exc:
                raise RuntimeError(
                    f"Alignment failed at phrase id={phrase_id}, "
                    f"zh={phoneme_phrase.get('zh', '')!r}, "
                    f"text={phoneme_phrase.get('text', '')!r}. "
                    f"{exc}"
                ) from exc

            items = []

            for alignment_unit, group in zip(alignment_units, note_groups):
                items.append(
                    {
                        "unit": alignment_unit.get("unit", ""),
                        "word": alignment_unit.get("word", ""),
                        "phonemes": alignment_unit.get("phonemes", []),
                        "merged_units": alignment_unit.get("units", []),
                        "notes": group,
                    }
                )

            aligned_phrases.append(
                {
                    "id": phrase_id,
                    "zh": phoneme_phrase.get("zh", ""),
                    "text": phoneme_phrase.get("text", ""),
                    "unit_type": "syllable_group",
                    "status": "success",
                    "unit_count": len(alignment_units),
                    "raw_unit_count": len(raw_units),
                    "note_count": len(notes),
                    "merge_info": merge_info,
                    "items": items,
                }
            )

        write_json(
            output,
            {
                "unit_type": "syllable_group",
                "description": (
                    "English syllable-level alignment with weak-unit merging. "
                    "Weak function words may be merged into neighboring syllable "
                    "groups when raw syllable count exceeds note count."
                ),
                "phrases": aligned_phrases,
            },
            self.config,
        )

        preview_path = output.parent / "alignment_preview.txt"

        write_text(
            preview_path,
            self._build_preview(aligned_phrases),
            self.config,
        )

        return {
            "status": "success",
            "outputs": {
                "note_lyric_alignment": str(output),
                "alignment_preview": str(preview_path),
            },
            "message": (
                "Aligned melody notes to English syllable groups with weak-unit merging."
            ),
        }