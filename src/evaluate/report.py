
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audio_quality_score(metrics: dict | None) -> float | None:
    if not metrics or not metrics.get('is_valid_audio'):
        return None
    clipping = _finite(metrics.get('clipping_ratio')) or 0.0
    silence = _finite(metrics.get('silence_ratio')) or 0.0
    high = _finite(metrics.get('high_freq_ratio')) or 0.0
    peak = _finite(metrics.get('peak')) or 0.0
    rms = _finite(metrics.get('rms')) or 0.0
    score = 100.0
    score -= min(40.0, clipping * 400.0)
    score -= min(20.0, abs(silence - 0.10) * 80.0)
    score -= min(15.0, max(0.0, high - 0.35) * 60.0)
    score -= min(15.0, max(0.0, peak - 0.99) * 300.0)
    score -= min(20.0, abs(rms - 0.08) * 120.0)
    return _clamp(score)


def pitch_score(metrics: dict | None) -> float | None:
    if not metrics or metrics.get('f0_rmse_cents') is None:
        return None
    rmse = _finite(metrics.get('f0_rmse_cents'))
    corr = _finite(metrics.get('f0_correlation'))
    vuv = _finite(metrics.get('voiced_unvoiced_accuracy'))
    if rmse is None:
        return None
    rmse_score = _clamp(100.0 - rmse * 1.2)
    corr_score = _clamp(((corr if corr is not None else 0.0) + 1.0) * 50.0)
    vuv_score = _clamp((vuv if vuv is not None else 0.0) * 100.0)
    return _clamp(rmse_score * 0.65 + corr_score * 0.20 + vuv_score * 0.15)


def conversion_integrity_score(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    if not metrics.get('source_exists') or not metrics.get('converted_exists'):
        return None
    if metrics.get('likely_same_audio'):
        return 0.0
    mean_abs = _finite(metrics.get('mean_abs_diff'))
    corr = _finite(metrics.get('waveform_correlation'))
    if mean_abs is None:
        return None
    # Tiny differences are suspicious. A real conversion should change the waveform noticeably.
    diff_score = _clamp((mean_abs / 0.01) * 100.0)
    corr_penalty = _clamp(((corr if corr is not None else 0.0) - 0.999) * 10000.0, 0.0, 70.0)
    return _clamp(diff_score - corr_penalty)


def intelligibility_score(metrics: dict | None) -> float | None:
    if not metrics or metrics.get('cer') is None:
        return None
    cer = _finite(metrics.get('cer'))
    if cer is None:
        return None
    return _clamp(100.0 * (1.0 - cer))


def speaker_score(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    target = _finite(metrics.get('speaker_similarity_to_target'))
    source = _finite(metrics.get('speaker_similarity_to_source'))
    if target is None and source is None:
        return None
    target_score = ((target if target is not None else 0.0) + 1.0) * 50.0
    source_penalty = ((source if source is not None else 0.0) + 1.0) * 20.0 if source is not None else 0.0
    return _clamp(target_score - source_penalty)


def compute_overall(report: dict) -> dict:
    category_scores = {
        'audio_quality': audio_quality_score((report.get('audio_quality') or {}).get('svc') or (report.get('audio_quality') or {}).get('final')),
        'conversion_integrity': conversion_integrity_score(report.get('conversion_integrity')),
        'pitch_preservation': pitch_score(report.get('pitch_preservation')),
        'intelligibility': intelligibility_score(report.get('intelligibility')),
        'speaker_similarity': speaker_score(report.get('speaker_similarity')),
    }
    weights = {
        'audio_quality': 25.0,
        'conversion_integrity': 20.0,
        'pitch_preservation': 30.0,
        'intelligibility': 5.0,
        'speaker_similarity': 20.0,
    }
    available = [key for key, value in category_scores.items() if value is not None]
    skipped = [key for key, value in category_scores.items() if value is None]
    warnings: list[str] = []
    conversion = report.get('conversion_integrity') or {}
    if conversion.get('likely_same_audio'):
        warnings.append('SVC output is identical or nearly identical to SVS input; overall score is capped because conversion is not validated.')
    if not available:
        return {
            'score': None,
            'level': 'unknown',
            'score_confidence': 'none',
            'evaluation_completeness': 0.0,
            'category_scores': category_scores,
            'available_categories': [],
            'skipped_categories': skipped,
            'warnings': ['No score categories are available.'],
        }
    total_weight = sum(weights[key] for key in available)
    score = sum((category_scores[key] or 0.0) * weights[key] for key in available) / total_weight
    if conversion.get('likely_same_audio'):
        score = min(score, 50.0)
    if score >= 90:
        level = 'excellent'
    elif score >= 80:
        level = 'good'
    elif score >= 65:
        level = 'fair'
    elif conversion.get('likely_same_audio'):
        level = 'invalid_conversion'
    else:
        level = 'poor'
    max_weight = sum(weights.values())
    completeness = total_weight / max_weight if max_weight else 0.0
    if conversion.get('likely_same_audio'):
        confidence = 'invalid_conversion'
    elif completeness >= 0.85:
        confidence = 'full'
    elif completeness >= 0.55:
        confidence = 'partial'
    else:
        confidence = 'low'
    return {
        'score': round(float(score), 3),
        'level': level,
        'score_confidence': confidence,
        'evaluation_completeness': round(float(completeness), 3),
        'category_scores': {key: (round(value, 3) if value is not None else None) for key, value in category_scores.items()},
        'available_categories': available,
        'skipped_categories': skipped,
        'warnings': warnings,
    }


def write_json_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def markdown_report(report: dict) -> str:
    overall = report.get('overall', {})
    lines = [
        '# YiYuTongSheng Evaluate Report',
        '',
        '## Summary',
        f"- Status: {report.get('status')}",
        f"- Overall score: {overall.get('score')}",
        f"- Level: {overall.get('level')}",
        f"- Score confidence: {overall.get('score_confidence')}",
        f"- Evaluation completeness: {overall.get('evaluation_completeness')}",
        f"- Available categories: {', '.join(overall.get('available_categories', []))}",
        f"- Skipped categories: {', '.join(overall.get('skipped_categories', []))}",
        '',
        '## Inputs',
    ]
    for key, value in (report.get('inputs') or {}).items():
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Audio Quality'])
    for key, value in (report.get('audio_quality') or {}).items():
        if isinstance(value, dict):
            lines.append(f"- {key}: valid={value.get('is_valid_audio')} duration={value.get('duration_sec')} rms={value.get('rms')} clipping={value.get('clipping_ratio')}")
    lines.extend(['', '## Conversion Integrity'])
    conversion = report.get('conversion_integrity') or {}
    for key in ['likely_same_audio', 'mean_abs_diff', 'max_abs_diff', 'rms_diff', 'waveform_correlation', 'sha256_equal']:
        lines.append(f'- {key}: {conversion.get(key)}')
    lines.extend(['', '## Pitch Preservation'])
    pitch = report.get('pitch_preservation') or {}
    for key in ['f0_rmse_hz', 'f0_rmse_cents', 'f0_mae_hz', 'f0_mae_cents', 'f0_correlation', 'voiced_unvoiced_accuracy']:
        lines.append(f'- {key}: {pitch.get(key)}')
    lines.extend(['', '## Intelligibility'])
    text = report.get('intelligibility') or {}
    lines.append(f"- CER: {text.get('cer')}")
    lines.append(f"- WER: {text.get('wer')}")
    lines.append(f"- ASR available: {text.get('asr_available')}")
    lines.extend(['', '## Speaker Similarity'])
    speaker = report.get('speaker_similarity') or {}
    lines.append(f"- Similarity to target: {speaker.get('speaker_similarity_to_target')}")
    lines.append(f"- Similarity to source: {speaker.get('speaker_similarity_to_source')}")
    lines.append(f"- Embedding available: {speaker.get('speaker_embedding_available')}")
    lines.extend(['', '## Overall Score'])
    for key, value in (overall.get('category_scores') or {}).items():
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Warnings'])
    warnings = (report.get('warnings') or []) + (overall.get('warnings') or [])
    if warnings:
        lines.extend(f'- {item}' for item in dict.fromkeys(warnings))
    else:
        lines.append('- None')
    return '\n'.join(lines) + '\n'


def write_markdown_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding='utf-8')
