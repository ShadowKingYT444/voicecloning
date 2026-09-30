"""Compare cached training conditioning with the actual inference tensors.

Use bounded_job.py with the CPU environment. No model weights are loaded.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("conditionals", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.cache.read_text())["rows"]
    reference = torch.load(args.conditionals, map_location="cpu", weights_only=True)["t3"]
    speaker = reference["speaker_emb"].flatten().float()
    prompt = reference["cond_prompt_speech_tokens"].flatten().long()
    reports = []
    for row in rows:
        if row["split"] == "reference":
            continue
        cached_speaker = torch.tensor(row["speaker_emb"], dtype=torch.float32)
        cached_prompt = torch.tensor(row["cond_prompt_speech_tokens"], dtype=torch.long)
        same_length = cached_prompt.shape == prompt.shape
        reports.append({
            "id": row["id"],
            "speaker_max_abs_difference": float((cached_speaker - speaker).abs().max()),
            "speaker_cosine": float(torch.nn.functional.cosine_similarity(cached_speaker[None], speaker[None])),
            "speaker_exact": torch.equal(cached_speaker, speaker),
            "prompt_exact": torch.equal(cached_prompt, prompt),
            "prompt_different_tokens": int((cached_prompt != prompt).sum()) if same_length else None,
            "cached_prompt_length": len(cached_prompt),
            "inference_prompt_length": len(prompt),
        })
    report = {"cache": str(args.cache), "inference_conditionals": str(args.conditionals),
              "all_exact": all(r["speaker_exact"] and r["prompt_exact"] for r in reports), "rows": reports}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_exact": report["all_exact"], "first_row": reports[0]}))


if __name__ == "__main__":
    main()
