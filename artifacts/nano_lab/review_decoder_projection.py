"""Build a controlled review page for the decoder-projection sweep.

This builder reads completed artifacts only.  It does not load Torch, audio
models, or evaluation models.  The comparison must contain three text IDs and
four variants per text: projection strength 0/1, each with raw and mel output.
The report remains a diagnostic artifact.  It never promotes a checkpoint or
edits delivery reports.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import statistics
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = ROOT / "artifacts" / "nano_lab" / "decoder_projection_comparison"
DEFAULT_EVALUATION = DEFAULT_COMPARISON / "evaluation_matched.json"
DEFAULT_AUDIT = DEFAULT_COMPARISON / "small_audit.json"
DEFAULT_LEVEL = DEFAULT_COMPARISON / "level_matched" / "manifest.json"

TEXT_ORDER = ("morning", "narrative", "question")
GROUP_ORDER = ("projection0_raw", "projection0_mel", "projection1_raw", "projection1_mel")


def _read_json(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required review input is missing: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON review input: {path}") from exc


def _rows(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("inputs", "rows", "results", "manifest"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ValueError(f"review input has no row list: {path}")
    else:
        raise ValueError(f"review input must be a list or object with rows: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"review input contains a non-object row: {path}")
    return [dict(row) for row in rows]


def _index(rows: Iterable[Mapping[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("id", "label"):
            value = row.get(key)
            if value not in (None, ""):
                result[str(value)] = dict(row)
        for key in ("path", "audio_path"):
            value = row.get(key)
            if value not in (None, ""):
                result[Path(str(value)).stem] = dict(row)
    if not result:
        raise ValueError(f"review input has no row identifiers: {path}")
    return result


def _required(index: Mapping[str, Mapping[str, Any]], key: str, *, source: Path) -> dict[str, Any]:
    row = index.get(key)
    if row is None:
        raise ValueError(f"review input {source} is missing row {key!r}")
    return dict(row)


def _finite_number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_payload_signature(path: Path) -> str:
    """Hash a Torch ZIP payload while ignoring archive prefix and serialization ID."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"token artifact is missing: {path}")
    entries: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            # Torch's ZIP writer chooses a prefix from the output filename
            # (for example ``morning_projection0_raw.tokens/``) rather than a
            # fixed ``archive/`` directory.  Drop that first component for
            # payload comparison so filenames cannot create a false delta.
            if len(parts) > 1:
                parts = parts[1:]
            normalised = "/".join(parts)
            if not normalised or normalised.endswith("serialization_id"):
                continue
            entries.append((normalised, archive.read(info.filename)))
    digest = hashlib.sha256()
    for name, value in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def _metric_row(manifest: Mapping[str, Any], evaluation: Mapping[str, Any], audit: Mapping[str, Any], level: Mapping[str, Any]) -> dict[str, Any]:
    identity = evaluation.get("speaker_similarity")
    if not isinstance(identity, Mapping) or identity.get("mean_cosine") is None:
        raise ValueError(f"evaluation row {manifest.get('id')} has no speaker_similarity.mean_cosine")
    wer_payload = audit.get("wer_contraction_normalized") or audit.get("wer")
    if not isinstance(wer_payload, Mapping) or wer_payload.get("wer") is None:
        raise ValueError(f"audit row {manifest.get('id')} has no WER")
    runtime = manifest.get("decoder_projection_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"manifest row {manifest.get('id')} has no decoder_projection_runtime metadata")
    strength = _finite_number(manifest.get("decoder_projection_strength"), name="decoder_projection_strength")
    expected_enabled = strength > 0.0
    actual_enabled = bool(runtime.get("enabled"))
    if actual_enabled != expected_enabled:
        raise ValueError(f"projection metadata enabled mismatch for {manifest.get('id')}")
    runtime_strength = _finite_number(runtime.get("strength", 0.0), name="runtime projection strength")
    if abs(runtime_strength - strength) > 1e-9:
        raise ValueError(f"projection metadata strength mismatch for {manifest.get('id')}")
    if expected_enabled:
        for key in ("adapter_sha256", "base_target_weight_sha256", "model_checkpoint_sha256", "conditionals_sha256"):
            if not runtime.get(key):
                raise ValueError(f"enabled projection row {manifest.get('id')} is missing runtime {key}")
    level_path = level.get("path") or manifest.get("path")
    if not level_path:
        raise ValueError(f"level-matched row {manifest.get('id')} has no audio path")
    return {
        "id": str(manifest["id"]),
        "text": str(manifest.get("text") or evaluation.get("expected_text") or ""),
        "strength": strength,
        "enabled": expected_enabled,
        "raw_path": str(manifest.get("path") or evaluation.get("audio_path")),
        "master_path": str(manifest.get("master_path") or ""),
        "level_matched_path": str(level_path),
        "token_path": str(Path(str(manifest.get("path"))).with_suffix(".tokens.pt")),
        "identity_cosine": _finite_number(identity["mean_cosine"], name="speaker similarity"),
        "dnsmos_overall": _finite_number(evaluation.get("dnsmos_overall"), name="DNSMOS overall"),
        "wer": _finite_number(wer_payload["wer"], name="WER"),
        "peak_rss_mib": _finite_number(manifest.get("peak_rss_mib"), name="peak process RSS"),
        "generation_seconds": _finite_number(manifest.get("generation_seconds"), name="generation seconds"),
        "audio_seconds": _finite_number(manifest.get("seconds"), name="audio seconds"),
        "audio_sha256": evaluation.get("audio_sha256"),
    }


def build_review(
    *,
    comparison_dir: Path = DEFAULT_COMPARISON,
    evaluation_path: Path = DEFAULT_EVALUATION,
    audit_path: Path = DEFAULT_AUDIT,
    level_manifest_path: Path = DEFAULT_LEVEL,
    fit_report_path: Path | None = None,
) -> dict[str, Any]:
    comparison_dir = comparison_dir.expanduser().resolve()
    manifest_path = comparison_dir / "manifest.json"
    manifest_index = _index(_rows(_read_json(manifest_path), path=manifest_path), path=manifest_path)
    evaluation_index = _index(_rows(_read_json(evaluation_path), path=evaluation_path), path=evaluation_path)
    audit_index = _index(_rows(_read_json(audit_path), path=audit_path), path=audit_path)
    level_index = _index(_rows(_read_json(level_manifest_path), path=level_manifest_path), path=level_manifest_path)

    grouped: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, dict[str, Any]] = {}
    for group in GROUP_ORDER:
        group_rows: list[dict[str, Any]] = []
        for text_name in TEXT_ORDER:
            identifier = f"{text_name}_{group}"
            manifest = _required(manifest_index, identifier, source=manifest_path)
            evaluation = _required(evaluation_index, identifier, source=evaluation_path)
            audit = _required(audit_index, identifier, source=audit_path)
            level = _required(level_index, identifier, source=level_manifest_path)
            row = _metric_row(manifest, evaluation, audit, level)
            row["text_name"] = text_name
            group_rows.append(row)
            all_rows[identifier] = row
        grouped[group] = {
            "rows": group_rows,
            "mean": {
                key: statistics.mean(row[key] for row in group_rows)
                for key in ("identity_cosine", "dnsmos_overall", "wer", "peak_rss_mib", "generation_seconds", "audio_seconds")
            },
            "all_wer_zero": all(row["wer"] == 0.0 for row in group_rows),
        }

    token_checks: dict[str, Any] = {}
    for text_name in TEXT_ORDER:
        variants = [all_rows[f"{text_name}_{group}"] for group in GROUP_ORDER]
        signatures = {}
        for row in variants:
            path = Path(row["token_path"])
            if not path.is_absolute():
                path = comparison_dir / path
            signatures[row["id"]] = _token_payload_signature(path)
        token_checks[text_name] = {
            "variants": signatures,
            "payload_equal_across_four_variants": len(set(signatures.values())) == 1,
            "normalisation": "ZIP archive/ prefix removed; serialization_id entry excluded",
        }

    # Projection strength 0 is the baseline control.  Its six raw/mel WAV
    # files must be byte-identical to the already reviewed fitted-decoder
    # controls.  This catches accidental changes in the new sweep before any
    # metric comparison is interpreted.
    baseline_dir = comparison_dir.parent / "decoder_fitted_comparison"
    baseline_checks: dict[str, Any] = {}
    for text_name in TEXT_ORDER:
        for variant in ("raw", "mel"):
            identifier = f"{text_name}_projection0_{variant}"
            current = Path(all_rows[identifier]["raw_path"])
            if not current.is_absolute():
                current = comparison_dir / current
            expected = baseline_dir / f"{text_name}_fit_{variant}.wav"
            if not current.exists() or not expected.exists():
                raise FileNotFoundError(f"baseline byte-identity check needs {current} and {expected}")
            current_sha = _sha256(current)
            expected_sha = _sha256(expected)
            baseline_checks[identifier] = {
                "current": str(current),
                "expected": str(expected),
                "current_sha256": current_sha,
                "expected_sha256": expected_sha,
                "equal": current_sha == expected_sha,
            }
    baseline_equal = all(item["equal"] for item in baseline_checks.values())

    fit_path = fit_report_path or comparison_dir.parent / "decoder_projection_fit" / "fit_report.json"
    fit_payload = _read_json(fit_path, required=False)
    verification_path = fit_path.parent / "artifact_verification.json"
    verification_payload = _read_json(verification_path, required=False)
    if isinstance(fit_payload, dict):
        history = fit_payload.get("history")
        stopped = fit_payload.get("stopped_after_steps")
        if stopped is None and isinstance(history, list):
            stopped = len(history)
        best_step = fit_payload.get("best_step")
        fit_status = fit_payload.get("status")
    else:
        stopped = None
        best_step = None
        fit_status = "missing_fit_report"
    if best_step is None and isinstance(verification_payload, dict):
        best_step = verification_payload.get("best_step")
    fit_disclosure = {
        "status": fit_status,
        "stopped_after_steps": stopped,
        "best_step": best_step,
        "requested_max_steps": 40,
        "complete": bool(stopped == 40),
        "claim": "partial fitted experiment; no human acceptance or general zero-shot promotion",
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "decoder_projection_controlled_review",
        "promoted": False,
        "human_listening_accepted": False,
        "general_zero_shot_improvement": False,
        "comparison_dir": str(comparison_dir),
        "inputs": {
            "manifest": str(manifest_path),
            "evaluation": str(evaluation_path.resolve()),
            "small_audit": str(audit_path.resolve()),
            "level_matched": str(level_manifest_path.resolve()),
            "fit_report": str(fit_path.resolve()),
            "artifact_verification": str(verification_path.resolve()),
        },
        "texts": list(TEXT_ORDER),
        "groups": grouped,
        "token_payload_checks": token_checks,
        "baseline_projection0_identity": {
            "checks": baseline_checks,
            "all_six_equal": baseline_equal,
        },
        "fit_disclosure": fit_disclosure,
        "limitations": [
            "The projection fit is speaker-specific and uses a partial optimization run.",
            "Identity and DNSMOS are automatic proxies. They do not establish realism.",
            "WER is a content check only. It does not measure voice identity or naturalness.",
            "Token payload equality is required so raw/mel rows isolate decoder/vocoder controls rather than T3 sampling.",
        ],
    }
    (comparison_dir / "review.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (comparison_dir / "index.html").write_text(_html(report), encoding="utf-8")
    return report


def _relative_audio(path: str, *, base: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return os.path.relpath(candidate.resolve(), base.resolve())
    except ValueError:
        return str(candidate)


def _html(report: Mapping[str, Any]) -> str:
    base = Path(str(report["comparison_dir"]))
    style = "body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#12151b;color:#eef1f7}p{line-height:1.5}a{color:#a8cbff}article{background:#202633;border-radius:12px;padding:14px;margin:10px 0}audio{display:block;width:100%;margin-top:8px}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{padding:7px;border-bottom:1px solid #3b4352;text-align:left}summary{cursor:pointer}.warn{color:#ffd27d}"
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Decoder projection review</title><style>{style}</style>",
        "<h1>Decoder projection controlled review</h1>",
        "<p class='warn'>Diagnostic artifact only. The fitted projection is partial and speaker-specific. Automatic metrics do not establish realistic zero-shot cloning, and no human acceptance or general promotion is recorded.</p>",
        "<p><a href='review.json'>Machine-readable measurements</a></p>",
        f"<p>Projection-0 baseline byte identity against decoder-fitted controls: <strong>{report['baseline_projection0_identity']['all_six_equal']}</strong>.</p>",
    ]
    for text_name in TEXT_ORDER:
        parts.append(f"<h2>{html.escape(text_name.title())}</h2>")
        parts.append("<table><tr><th>Variant</th><th>Identity</th><th>DNSMOS</th><th>WER</th><th>Peak RSS MiB</th><th>Audio</th></tr>")
        for group in GROUP_ORDER:
            row = next(r for r in report["groups"][group]["rows"] if r["text_name"] == text_name)
            audio = _relative_audio(row["level_matched_path"], base=base)
            parts.append(
                f"<tr><td>{html.escape(group)}</td><td>{row['identity_cosine']:.4f}</td>"
                f"<td>{row['dnsmos_overall']:.3f}</td><td>{row['wer']:.3f}</td>"
                f"<td>{row['peak_rss_mib']:.1f}</td><td><a href='{html.escape(audio)}'>level-matched WAV</a></td></tr>"
            )
        parts.append("</table>")
        for group in GROUP_ORDER:
            row = next(r for r in report["groups"][group]["rows"] if r["text_name"] == text_name)
            raw = _relative_audio(row["raw_path"], base=base)
            master = _relative_audio(row["master_path"], base=base) if row["master_path"] else ""
            parts.append("<article>")
            parts.append(f"<strong>{html.escape(group)}</strong><audio controls preload='none' src='{html.escape(raw)}'></audio>")
            if master:
                parts.append(f"<small><a href='{html.escape(master)}'>mastered file</a></small>")
            parts.append("</article>")
    parts.append("<h2>Token isolation and fit disclosure</h2><ul>")
    for text_name, check in report["token_payload_checks"].items():
        parts.append(f"<li>{html.escape(text_name)}: four-way ZIP payload equality = {check['payload_equal_across_four_variants']}</li>")
    fit = report["fit_disclosure"]
    parts.append(f"</ul><p>Fit status: {html.escape(str(fit['status']))}; stopped after {html.escape(str(fit['stopped_after_steps']))} steps; best step {html.escape(str(fit['best_step']))}; requested maximum 40. Complete run: {fit['complete']}.</p></html>")
    return "".join(parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--small-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--level-manifest", type=Path, default=DEFAULT_LEVEL)
    parser.add_argument("--fit-report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_review(
        comparison_dir=args.comparison_dir,
        evaluation_path=args.evaluation,
        audit_path=args.small_audit,
        level_manifest_path=args.level_manifest,
        fit_report_path=args.fit_report,
    )
    print(json.dumps({"review": str(Path(report["comparison_dir"]) / "review.json"), "index": str(Path(report["comparison_dir"]) / "index.html")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
