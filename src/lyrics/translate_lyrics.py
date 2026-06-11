from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


class LyricTranslator:
    """Translate Chinese lyric phrases into English with Qwen API.

    General automatic system rules:
    1. No local model.
    2. No fixed English placeholder bank.
    3. No song-specific phrase override.
    4. No manual phrase-level rewrite rules.
    5. No fake fallback translation.
    6. If Qwen API fails, stop the pipeline.
    7. lyrics_literal.json must only contain real API translations.
    """

    def __init__(self, config: dict):
        self.config = config
        self.translation_backend = "qwen_api"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _target_language(self) -> str:
        return self.config.get("project", {}).get("target_language", "en")

    def _model_name(self) -> str:
        env_model = os.getenv("QWEN_TRANSLATION_MODEL")
        if env_model:
            return env_model

        return (
            self.config.get("models", {}).get("translation_model")
            or "qwen-plus"
        )

    def _base_url(self) -> str:
        env_base_url = os.getenv("DASHSCOPE_BASE_URL")
        if env_base_url:
            return env_base_url

        return (
            self.config.get("models", {}).get("qwen_base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _batch_size(self) -> int:
        return int(
            self.config.get("models", {}).get("translation_batch_size", 12)
        )

    # ------------------------------------------------------------------
    # Text cleanup
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Only character-level cleanup.

        No semantic rewriting.
        No phrase-level replacement.
        No song-specific rule.
        """
        text = str(text or "")
        text = text.replace("♪", " ")
        text = text.replace("“", "").replace("”", "")
        text = text.replace('"', "")
        text = text.strip("。！？；：，、")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()

    def _strip_json_fence(self, text: str) -> str:
        text = str(text or "").strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        return text

    def _parse_json_object(self, text: str) -> dict:
        text = self._strip_json_fence(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Qwen translation API did not return valid JSON. "
                f"Raw output: {text[:1200]}"
            ) from exc

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _client(self) -> OpenAI:
        api_key = "sk-3b50237553164953a48de50695c440c8"

        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. "
                "Set it first, then reopen your terminal."
            )

        return OpenAI(
            api_key=api_key,
            base_url=self._base_url(),
        )

    def _build_translation_payload(self, phrases: list[dict]) -> str:
        payload = {
            "task": "translate_zh_lyrics_to_english",
            "target_language": self._target_language(),
            "rules": [
                "Translate each Chinese lyric line into faithful natural English.",
                "Keep the same number of items as input.",
                "Preserve each input id exactly.",
                "Do not merge lines.",
                "Do not split lines.",
                "Do not add imagery that is not present in the Chinese.",
                "Do not use fixed lyric templates.",
                "Do not use song-specific rules.",
                "Do not explain.",
                "Return valid JSON only."
            ],
            "required_output_schema": {
                "translations": [
                    {
                        "id": "same id from input",
                        "literal_en": "faithful English translation"
                    }
                ]
            },
            "items": [
                {
                    "id": phrase.get("id"),
                    "zh": phrase.get("zh", "")
                }
                for phrase in phrases
            ],
        }

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _call_qwen_api(self, phrases: list[dict]) -> list[dict]:
        client = self._client()

        system_prompt = (
            "You are a translation engine in a cross-language singing voice "
            "conversion system. Translate Chinese lyrics into faithful English "
            "literal lyrics. Do not polish into fixed templates. Do not apply "
            "song-specific rules. Preserve line count and ids. Return only valid JSON."
        )

        response = client.chat.completions.create(
            model=self._model_name(),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": self._build_translation_payload(phrases),
                },
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        result = self._parse_json_object(content)

        translations = result.get("translations", [])

        if not isinstance(translations, list):
            raise RuntimeError(
                "Qwen translation API JSON missing list field: translations"
            )

        return translations

    def _translate_batch(self, phrases: list[dict]) -> list[str]:
        translations = self._call_qwen_api(phrases)

        by_id: dict[int, str] = {}

        for item in translations:
            if not isinstance(item, dict):
                continue

            phrase_id = item.get("id")
            literal = self._clean_text(item.get("literal_en", ""))

            if phrase_id is None:
                continue

            if not literal:
                continue

            by_id[int(phrase_id)] = literal

        output_texts = []

        for phrase in phrases:
            phrase_id = int(phrase.get("id"))

            if phrase_id not in by_id:
                raise RuntimeError(
                    "Qwen translation result is missing phrase "
                    f"id={phrase_id}, zh={phrase.get('zh')!r}."
                )

            output_texts.append(by_id[phrase_id])

        return output_texts

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> dict:
        phrase_map = path_from_config(self.config, "phrase_map")
        input_status = require_or_mock_input(phrase_map, self.config, "phrase map")

        phrase_data = read_json(phrase_map, {"phrases": []})
        phrases = phrase_data.get("phrases", [])

        if not phrases:
            raise RuntimeError("phrase_map.json contains no phrases.")

        for phrase in phrases:
            zh = str(phrase.get("zh", "")).strip()

            if not zh:
                raise RuntimeError(
                    "phrase_map.json contains empty Chinese lyric text for "
                    f"phrase id={phrase.get('id')}."
                )

        translated_texts: list[str] = []
        batch_size = self._batch_size()

        try:
            for start in range(0, len(phrases), batch_size):
                batch = phrases[start:start + batch_size]
                translated_texts.extend(self._translate_batch(batch))

        except Exception as exc:
            raise RuntimeError(
                "Qwen API translation failed. The pipeline is stopped intentionally. "
                "This general automatic system does not allow placeholder lyrics. "
                f"Original error: {exc}"
            ) from exc

        if len(translated_texts) != len(phrases):
            raise RuntimeError(
                "Translation output count does not match phrase count: "
                f"{len(translated_texts)} translations for {len(phrases)} phrases."
            )

        target_language = self._target_language()
        translated_phrases = []

        for phrase, literal in zip(phrases, translated_texts):
            literal = self._clean_text(literal)

            if not literal:
                raise RuntimeError(
                    "Translation produced an empty result for phrase "
                    f"id={phrase.get('id')}, zh={phrase.get('zh')!r}."
                )

            translated_phrases.append(
                {
                    **phrase,
                    "target_language": target_language,
                    "literal": literal,
                    "literal_en": literal if target_language == "en" else ""
                }
            )

        output = path_from_config(self.config, "lyrics_literal")

        write_json(
            output,
            {
                "source": "api_translation",
                "translation_backend": self.translation_backend,
                "translation_model": self._model_name(),
                "translation_base_url": self._base_url(),
                "translation_error": "",
                "phrases": translated_phrases,
            },
            self.config,
        )

        return {
            "status": "success" if input_status != "mock" else "mock",
            "outputs": {
                "lyrics_literal": str(output),
            },
            "message": "Translated Chinese lyrics into English with Qwen API and no placeholder fallback.",
        }