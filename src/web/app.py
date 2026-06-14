from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.web.schemas import ApiResponse, RunRequest
from src.web.services import (
    PROJECT_ROOT,
    audio_list,
    build_evaluate_command,
    build_full_pipeline_commands,
    build_score_command,
    build_svc_command,
    build_svs_command,
    current_score,
    job_manager,
    latest_log,
    latest_evaluate_report,
    latest_reports,
    load_web_config,
    open_output_directory,
    phrase_segments,
    scan_songs,
    scan_voice_models,
)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="YiYuTongSheng Web Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def api_ok(data: Any = None, warnings: list[str] | None = None) -> JSONResponse:
    payload = ApiResponse(ok=True, data=data or {}, warnings=warnings or [], errors=[])
    return JSONResponse(payload.dict())


def api_error(message: str, status_code: int = 400) -> JSONResponse:
    payload = ApiResponse(ok=False, data={}, warnings=[], errors=[message])
    return JSONResponse(payload.dict(), status_code=status_code)


@app.get("/")
def index(request: Request):
    config = load_web_config()
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults", {}), dict) else {}
    return templates.TemplateResponse(request=request, name="index.html", context={"defaults": defaults})


@app.get("/api/status")
def api_status():
    reports = latest_reports()
    jobs = job_manager.list()
    return api_ok(
        {
            "project_root": str(PROJECT_ROOT),
            "songs": len(scan_songs()),
            "voice_models": len(scan_voice_models()),
            "active_jobs": [job for job in jobs if job["status"] in {"pending", "running"}],
            "reports": reports.get("reports", {}),
        },
        reports.get("warnings", []),
    )


@app.get("/api/songs")
def api_songs():
    return api_ok({"songs": scan_songs()})


@app.get("/api/voice-models")
def api_voice_models():
    return api_ok({"models": scan_voice_models()})


@app.get("/api/reports/latest")
def api_reports_latest():
    reports = latest_reports()
    data = dict(reports)
    data["segments"] = phrase_segments()
    return api_ok(data, reports.get("warnings", []))


@app.get("/api/score/current")
def api_score_current():
    return api_ok(current_score())


@app.get("/api/audio/list")
def api_audio_list(song_id: str | None = None):
    return api_ok({"audio": audio_list(song_id)})


@app.get("/api/logs/latest")
def api_logs_latest():
    return api_ok(latest_log())


@app.get("/api/evaluate/report")
def api_evaluate_report():
    report = latest_evaluate_report()
    warnings = [report["warning"]] if report.get("warning") else []
    return api_ok(report, warnings)


@app.get("/api/jobs")
def api_jobs():
    return api_ok({"jobs": job_manager.list()})


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        return api_error(f"Job not found: {job_id}", 404)
    return api_ok({"job": job})


def _start_job(name: str, commands: list[list[str]]):
    job, warning = job_manager.start(name, commands)
    if warning:
        return api_ok({"job": None}, [warning])
    return api_ok({"job": job})


@app.post("/api/run/score")
def api_run_score(request: RunRequest):
    return _start_job("score", [build_score_command(request)])


@app.post("/api/run/svs")
def api_run_svs(request: RunRequest):
    return _start_job("svs", [build_svs_command(request)])


@app.post("/api/run/svc")
def api_run_svc(request: RunRequest):
    return _start_job("svc", [build_svc_command(request)])


@app.post("/api/run/full")
def api_run_full(request: RunRequest):
    return _start_job("full", build_full_pipeline_commands(request))


@app.post("/api/run/evaluate")
def api_run_evaluate(request: RunRequest):
    return _start_job("evaluate", [build_evaluate_command(request)])


@app.post("/api/open-output-directory")
def api_open_output_directory():
    return api_ok(open_output_directory())


def _safe_audio_response(base: Path, filename: str):
    allowed_ext = {".wav", ".mp3", ".flac"}
    safe_name = Path(filename).name
    path = (base / safe_name).resolve()
    base_resolved = base.resolve()
    if path.suffix.lower() not in allowed_ext:
        return api_error("Unsupported audio extension.", 403)
    try:
        path.relative_to(base_resolved)
    except ValueError:
        return api_error("Audio path is outside allowed directory.", 403)
    if not path.exists():
        return api_error(f"Audio file does not exist: {safe_name}", 404)
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/audio/opencpop/{filename}")
def audio_opencpop(filename: str):
    return _safe_audio_response(PROJECT_ROOT / "data/dataset/opencpop/wavs", filename)


@app.get("/audio/svs/{filename}")
def audio_svs(filename: str):
    return _safe_audio_response(PROJECT_ROOT / "data/svs", filename)


@app.get("/audio/svc/{filename}")
def audio_svc(filename: str):
    return _safe_audio_response(PROJECT_ROOT / "data/svc", filename)


@app.get("/audio/segments/{filename}")
def audio_segments(filename: str):
    return _safe_audio_response(PROJECT_ROOT / "data/svs/segments", filename)
