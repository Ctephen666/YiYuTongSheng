from __future__ import annotations

from pathlib import Path

import numpy as np


def cosine_similarity(a, b) -> float | None:
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return None
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return None
    value = float(np.dot(va, vb) / denom)
    if not np.isfinite(value):
        return None
    return value


def _load_embedding(path: Path):
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        return None


def speaker_similarity_metrics(target_reference_dir: str | Path, source_reference_path: str | Path, converted_path: str | Path) -> dict:
    target_dir = Path(target_reference_dir)
    source_path = Path(source_reference_path)
    converted_audio_path = Path(converted_path)
    target_embedding_path = target_dir / "target_embedding.npy"
    source_embedding_path = source_path.with_suffix(".embedding.npy")
    converted_embedding_path = converted_audio_path.with_suffix(".embedding.npy")

    warnings: list[str] = []
    target_embedding = _load_embedding(target_embedding_path)
    source_embedding = _load_embedding(source_embedding_path)
    converted_embedding = _load_embedding(converted_embedding_path)
    target_count = len(list(target_dir.glob("*.wav"))) if target_dir.exists() else 0

    target_similarity = cosine_similarity(target_embedding, converted_embedding) if target_embedding is not None and converted_embedding is not None else None
    source_similarity = cosine_similarity(source_embedding, converted_embedding) if source_embedding is not None and converted_embedding is not None else None
    available = target_similarity is not None or source_similarity is not None
    if not available:
        warnings.append("Speaker embedding model not available; speaker similarity skipped.")

    return {
        "speaker_similarity_to_target": target_similarity,
        "speaker_similarity_to_source": source_similarity,
        "speaker_embedding_available": available,
        "target_reference_count": int(target_count),
        "source_reference_path": str(source_path),
        "converted_path": str(converted_audio_path),
        "target_embedding_path": str(target_embedding_path),
        "source_embedding_path": str(source_embedding_path),
        "converted_embedding_path": str(converted_embedding_path),
        "warnings": warnings,
    }
