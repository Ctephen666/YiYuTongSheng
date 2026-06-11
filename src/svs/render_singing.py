from __future__ import annotations

from pathlib import Path

from src.common.io_utils import path_from_config, project_root
from src.common.json_utils import read_json, write_json


class SingingRenderer:
    """Prepare real OpenUtau rendering request.

    This class no longer writes mock placeholder audio.

    It checks that target_language.ustx exists, then writes render_request.json.
    Actual wav rendering should be done by OpenUtau GUI or a configured external
    OpenUtau/DiffSinger command in a later implementation.
    """

    def __init__(self, config: dict):
        self.config = config

    def _path_or_default(self, key: str, default: str) -> Path:
        try:
            return Path(path_from_config(self.config, key))
        except Exception:
            return project_root(self.config) / default

    def _plan_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "openutau_export_plan.json"

    def _openutau_project_path(self) -> Path:
        return self._path_or_default(
            "openutau_project",
            "data/svs/target_language.ustx",
        )

    def _render_request_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "render_request.json"

    def _renderer_backend(self) -> str:
        return (
            self.config.get("models", {}).get("svs_backend")
            or self.config.get("svs", {}).get("backend")
            or "openutau_manual"
        )

    def _build_render_request(
        self,
        plan: dict,
        openutau_project: Path,
        svs_vocal: Path,
    ) -> dict:
        summary = plan.get("summary", {})
        note_events = plan.get("note_events", [])

        return {
            "source": "openutau_project",
            "format": "svs_render_request_v1",
            "backend": self._renderer_backend(),
            "language": plan.get("language", "en"),
            "unit_type": plan.get("unit_type", "syllable_group"),
            "inputs": {
                "openutau_export_plan": str(self._plan_path()),
                "openutau_project": str(openutau_project),
            },
            "outputs": {
                "svs_vocal": str(svs_vocal),
            },
            "summary": {
                "phrase_count": summary.get("phrase_count", 0),
                "note_event_count": summary.get("note_event_count", len(note_events)),
                "start": summary.get("start", 0.0),
                "end": summary.get("end", 0.0),
                "duration": summary.get("duration", 0.0),
            },
            "renderer_status": "ustx_ready_external_render_required",
            "next_step": (
                "Open target_language.ustx in OpenUtau, select an English-compatible "
                "voicebank/phonemizer, then render/export wav to target_language_vocal.wav."
            ),
        }

    def run(self) -> dict:
        plan_path = self._plan_path()

        if not plan_path.exists():
            raise RuntimeError(
                f"OpenUtau export plan does not exist: {plan_path}. "
                "Run OpenUtauExporter first."
            )

        plan = read_json(plan_path, {})
        note_events = plan.get("note_events", [])

        if not note_events:
            raise RuntimeError(
                f"openutau_export_plan.json contains no note_events: {plan_path}"
            )

        openutau_project = self._openutau_project_path()

        if not openutau_project.exists():
            raise RuntimeError(
                f"OpenUtau project file does not exist: {openutau_project}. "
                "OpenUtauExporter must generate target_language.ustx first."
            )

        svs_vocal = path_from_config(self.config, "svs_vocal")
        render_request = self._build_render_request(
            plan=plan,
            openutau_project=openutau_project,
            svs_vocal=svs_vocal,
        )

        render_request_path = self._render_request_path()
        write_json(render_request_path, render_request, self.config)

        return {
            "status": "success",
            "outputs": {
                "render_request": str(render_request_path),
                "openutau_project": str(openutau_project),
                "expected_svs_vocal": str(svs_vocal),
            },
            "message": (
                "Generated real OpenUtau render request. No mock audio was created."
            ),
        }
