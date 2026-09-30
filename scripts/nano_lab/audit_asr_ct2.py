"""Independent Whisper-small content audit using CPU int8 CTranslate2.

Run under bounded_job.py --small-job. This is an automated transcript check,
not a human listening assessment. No audio leaves the computer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _model_identity(model_path: Path) -> dict[str, object]:
    """Return a stable, read-only identity for a local CTranslate2 model."""

    model_path = model_path.resolve()
    download_manifest = model_path / "download_manifest.json"
    identity: dict[str, object] = {
        "path": str(model_path),
        "name": model_path.name,
        "download_manifest": str(download_manifest),
        "download_manifest_sha256": _sha256(download_manifest) if download_manifest.is_file() else None,
    }
    lower = model_path.name.lower()
    identity["size_class"] = "tiny" if "tiny" in lower else "small" if "small" in lower else "unknown"
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2]/"models/faster-whisper-small")
    parser.add_argument("--word-timestamps", action="store_true", help="include per-word timestamps and probabilities")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    from reference_evaluator import word_error_rate, _expand_contractions

    rows = json.loads(args.manifest.read_text())
    if isinstance(rows, dict):
        rows = rows.get("runs", rows.get("inputs", []))
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    model_identity = _model_identity(args.model)
    report = {"model": str(args.model.resolve()), "backend": "CTranslate2 CPU int8",
              "role": "independent_content_audit", "human_listening": False,
              "model_download_manifest": str(args.model.resolve()/"download_manifest.json"),
              "model_identity": model_identity, "word_timestamps": bool(args.word_timestamps), "inputs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("error"):
            continue
        path = Path(row.get("path", row.get("audio_path")))
        expected = row.get("text", row.get("expected_text"))
        started = time.perf_counter()
        segments, info = model.transcribe(str(path), language="en", beam_size=3,
                                         temperature=0.0, condition_on_previous_text=False,
                                         vad_filter=False, word_timestamps=args.word_timestamps)
        saved = []
        for segment in segments:
            saved_segment = {"text": segment.text.strip(), "start_s": segment.start, "end_s": segment.end,
                             "avg_logprob": segment.avg_logprob, "no_speech_prob": segment.no_speech_prob,
                             "compression_ratio": segment.compression_ratio}
            if args.word_timestamps:
                saved_segment["words"] = [
                    {
                        "word": getattr(word, "word", "").strip(),
                        "start_s": getattr(word, "start", None),
                        "end_s": getattr(word, "end", None),
                        "probability": getattr(word, "probability", None),
                    }
                    for word in (getattr(segment, "words", None) or [])
                ]
            saved.append(saved_segment)
        transcript = " ".join(s["text"] for s in saved)
        digest = _sha256(path)
        result = {"id": row.get("id", row.get("label", path.stem)), "path": str(path.resolve()),
                  "audio_sha256": digest, "expected_text": expected, "transcript": transcript,
                  "segments": saved, "wer": word_error_rate(expected, transcript),
                  "wer_contraction_normalized": word_error_rate(_expand_contractions(expected), _expand_contractions(transcript)),
                  "model_identity": model_identity,
                  "seconds": time.perf_counter()-started}
        report["inputs"].append(result)
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, args.out)
        print(json.dumps({"id": result["id"], "wer": result["wer_contraction_normalized"]["wer"], "transcript": transcript}), flush=True)


if __name__ == "__main__":
    main()
