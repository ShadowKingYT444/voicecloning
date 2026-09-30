"""Fit a per-voice T3 conditioning-vector residual.

This experiment changes only the 256-dimensional T3 speaker embedding used in
the cached conditioning rows.  The GPT-2 transformer, speech head, speech
tokenizer, decoder, and voice encoder remain frozen.  The adapted vector is
unit-normalized on every forward pass, and the residual can be constrained by
an L2 projection after each optimizer step.

The causal target is the same shifted Turbo objective as ``adaptation.py``:

``conditioning + text + BOS + speech[:-1] -> speech + EOS``

The best checkpoint contains a JSON-readable ``speaker_emb`` vector.  A
quality sweep can inject that vector into ``model.conds.t3.speaker_emb`` after
preparing its reference conditionals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import adaptation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "chatterbox-nano"
DEFAULT_CACHE = ROOT / "artifacts" / "nano_lab" / "adaptation_asmr_cache.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "nano_lab" / "conditioning_fit_best.pt"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    speaker_id: str | None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    """Select one speaker and require train plus disjoint validation rows."""

    return adaptation.select_speaker_rows(rows, speaker_id)


def base_embedding(train_rows: Sequence[Mapping[str, Any]]) -> Tensor:
    """Return a unit-norm base vector and reject malformed cache embeddings."""

    values = []
    for row in train_rows:
        value = np.asarray(row["speaker_emb"], dtype=np.float32).reshape(-1)
        if value.shape != (256,) or not np.isfinite(value).all():
            raise ValueError(f"row {row.get('id')} has invalid speaker_emb shape/value: {value.shape}")
        values.append(value)
    stacked = np.stack(values, axis=0)
    mean = torch.from_numpy(stacked.mean(axis=0)).float()
    if not torch.isfinite(mean).all() or float(mean.norm()) <= 1e-8:
        raise ValueError("train speaker embeddings have zero or non-finite mean norm")
    return F.normalize(mean, dim=0)


def adapted_embedding(base: Tensor, delta: Tensor, *, unit_normalize: bool) -> Tensor:
    value = base + delta
    if unit_normalize:
        value = F.normalize(value, dim=0)
    return value


def _teacher_forced_logits_with_embedding(
    model: Any,
    row: Mapping[str, Any],
    speaker_embedding: Tensor,
    *,
    device: torch.device,
    max_target_tokens: int | None,
) -> tuple[Tensor, Tensor]:
    """Build the exact shifted Turbo logits with a differentiable speaker vector."""

    from chatterbox.models.t3.modules.cond_enc import T3Cond

    text_tokens = torch.tensor(row["text_tokens"], dtype=torch.long, device=device).view(1, -1)
    speech = torch.tensor(row["speech_tokens"], dtype=torch.long, device=device)
    if max_target_tokens is not None and speech.numel() > max_target_tokens:
        raise ValueError(
            f"cache row {row.get('id')} has {speech.numel()} target tokens, above "
            f"--max-target-tokens {max_target_tokens}; refusing silent target truncation"
        )
    if speech.numel() < 2:
        raise ValueError(f"row {row.get('id')} has fewer than two target tokens")

    hp = model.t3.hp
    target = torch.cat((speech, torch.tensor([hp.stop_speech_token], dtype=torch.long, device=device)))
    decoder_input = torch.cat(
        (torch.tensor([hp.start_speech_token], dtype=torch.long, device=device), target[:-1])
    ).view(1, -1)
    prompt = torch.tensor(row["cond_prompt_speech_tokens"], dtype=torch.long, device=device).view(1, -1)
    cond = T3Cond(
        speaker_emb=speaker_embedding.view(1, 256),
        cond_prompt_speech_tokens=prompt,
        emotion_adv=None,
    )
    embeds, _ = model.t3.prepare_input_embeds(
        t3_cond=cond,
        text_tokens=text_tokens,
        speech_tokens=decoder_input,
        cfg_weight=0.0,
    )
    outputs = model.t3.tfmr(
        inputs_embeds=embeds,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state[:, -decoder_input.shape[1] :, :]
    return model.t3.speech_head(hidden), target


def conditioned_loss(
    model: Any,
    row: Mapping[str, Any],
    speaker_embedding: Tensor,
    *,
    device: torch.device,
    max_target_tokens: int | None,
) -> tuple[Tensor, int, int]:
    logits, target = _teacher_forced_logits_with_embedding(
        model,
        row,
        speaker_embedding,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.view(-1))
    correct = int((logits.argmax(dim=-1).view(-1) == target).sum().item())
    return loss, int(target.numel()), correct


def evaluate(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    speaker_embedding: Tensor,
    *,
    device: torch.device,
    max_target_tokens: int | None,
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for row in rows:
            loss, tokens, correct = conditioned_loss(
                model,
                row,
                speaker_embedding,
                device=device,
                max_target_tokens=max_target_tokens,
            )
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            total_correct += correct
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "tokens": total_tokens,
        "correct": total_correct,
        "token_accuracy": total_correct / max(total_tokens, 1),
        "perplexity": math.exp(min(20.0, mean_loss)),
    }


def project_delta(delta: Tensor, max_delta_norm: float) -> None:
    if max_delta_norm <= 0.0:
        return
    with torch.no_grad():
        norm = float(delta.norm().item())
        if norm > max_delta_norm:
            delta.mul_(max_delta_norm / max(norm, 1e-12))


def vector_payload(
    *,
    base: Tensor,
    delta: Tensor,
    adapted: Tensor,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    adapted_list = [float(x) for x in adapted.detach().cpu().tolist()]
    return {
        "format": "nano_t3_conditioning_vector_v1",
        "speaker_emb": adapted_list,
        "base_embedding": [float(x) for x in base.detach().cpu().tolist()],
        "delta": [float(x) for x in delta.detach().cpu().tolist()],
        "embedding_dim": 256,
        "unit_normalized": True,
        "adapted_norm": float(adapted.detach().norm().cpu()),
        "delta_norm": float(delta.detach().norm().cpu()),
        "quality_sweep_injection": {
            "target": "model.conds.t3.speaker_emb",
            "shape": [1, 256],
            "dtype": "float32",
            "operation": "torch.tensor(payload['speaker_emb']).view(1, 256).to(model.device)",
        },
        "metadata": dict(metadata),
    }


def save_checkpoint(
    checkpoint: Path,
    *,
    base: Tensor,
    delta: Tensor,
    adapted: Tensor,
    metadata: Mapping[str, Any],
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nano_t3_conditioning_vector_checkpoint_v1",
        "base_embedding": base.detach().cpu(),
        "delta": delta.detach().cpu(),
        "adapted_embedding": adapted.detach().cpu(),
        "metadata": dict(metadata),
    }
    torch.save(payload, checkpoint)
    write_json(
        checkpoint.with_suffix(".json"),
        vector_payload(base=base, delta=delta, adapted=adapted, metadata=metadata),
    )


def fit(args: argparse.Namespace) -> dict[str, Any]:
    adaptation.configure_runtime(args.threads)
    cache = adaptation.load_cache(Path(args.cache))
    device = torch.device(args.device)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    train_rows, valid_rows, selected_speaker = select_rows(cache["rows"], args.speaker_id)
    max_target_tokens = args.max_target_tokens or cache.get("max_target_tokens")
    if args.l2_lambda < 0.0 or not math.isfinite(args.l2_lambda):
        raise ValueError("--l2-lambda must be finite and >= 0")
    if args.max_delta_norm < 0.0 or not math.isfinite(args.max_delta_norm):
        raise ValueError("--max-delta-norm must be finite and >= 0")

    base = base_embedding(train_rows).to(device)
    delta = torch.nn.Parameter(torch.zeros_like(base))
    # Cached training rows already contain all reference features and tokens.
    # Load only T3, directly onto the target device, to keep host RSS bounded.
    from t3_training import load_cached_t3
    model=load_cached_t3(args.model_dir,device)
    for parameter in model.t3.parameters():
        parameter.requires_grad_(False)
    model.t3.eval()
    optimizer = torch.optim.AdamW([delta], lr=float(args.lr), weight_decay=float(args.weight_decay))

    def current_embedding() -> Tensor:
        return adapted_embedding(base, delta, unit_normalize=not args.no_unit_normalize)

    base_eval = evaluate(
        model,
        valid_rows,
        current_embedding(),
        device=device,
        max_target_tokens=max_target_tokens,
    )
    print(json.dumps({"event": "baseline", "epoch": 0, "valid": base_eval}), flush=True)
    best_valid = float(base_eval["loss"])
    best_epoch = 0
    best_saved = False
    no_improvement = 0
    started = time.perf_counter()
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train": None,
            "valid": base_eval,
            "objective": "initial validation loss at original speaker embedding",
        }
    ]
    checkpoint = Path(args.checkpoint)
    best_metadata: dict[str, Any] = {}

    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(train_rows)))
        random.Random((args.seed or 0) + epoch).shuffle(order)
        train_loss_sum = 0.0
        train_l2_sum = 0.0
        train_tokens = 0
        train_correct = 0
        for step_index, row_index in enumerate(order, start=1):
            row = train_rows[row_index]
            optimizer.zero_grad(set_to_none=True)
            embedding = current_embedding()
            loss, tokens, correct = conditioned_loss(
                model,
                row,
                embedding,
                device=device,
                max_target_tokens=max_target_tokens,
            )
            l2 = (embedding - base).pow(2).mean()
            objective = loss + float(args.l2_lambda) * l2
            objective.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([delta], float(args.grad_clip))
            optimizer.step()
            project_delta(delta, float(args.max_delta_norm))
            train_loss_sum += float(loss.detach().item()) * tokens
            train_l2_sum += float(l2.detach().item()) * tokens
            train_tokens += tokens
            train_correct += correct
            if args.max_steps and (epoch - 1) * len(order) + step_index >= args.max_steps:
                break

        train_stats = {
            "loss": train_loss_sum / max(train_tokens, 1),
            "tokens": train_tokens,
            "correct": train_correct,
            "token_accuracy": train_correct / max(train_tokens, 1),
            "perplexity": math.exp(min(20.0, train_loss_sum / max(train_tokens, 1))),
            "embedding_l2_to_base": train_l2_sum / max(train_tokens, 1),
        }
        valid_stats = evaluate(
            model,
            valid_rows,
            current_embedding(),
            device=device,
            max_target_tokens=max_target_tokens,
        )
        record = {
            "epoch": epoch,
            "train": train_stats,
            "valid": valid_stats,
            "objective": "shifted speech-token cross-entropy + embedding L2",
            "l2_lambda": float(args.l2_lambda),
            "max_delta_norm": float(args.max_delta_norm),
            "delta_norm": float(delta.detach().norm().item()),
            "adapted_embedding_norm": float(current_embedding().detach().norm().item()),
        }
        history.append(record)
        print(json.dumps({"event": "epoch", **record}), flush=True)
        improved = float(valid_stats["loss"]) < best_valid - float(args.min_delta)
        if improved:
            best_valid = float(valid_stats["loss"])
            best_epoch = epoch
            no_improvement = 0
            adapted = current_embedding().detach()
            best_metadata = {
                "experiment": "nano_t3_conditioning_vector",
                "mode": "conditioning_only",
                "objective": "exact Turbo causal shifted speech-token cross-entropy",
                "model_dir": str(Path(args.model_dir).resolve()),
                "device": str(device),
                "cpu_threads": int(args.threads),
                "speaker_id": selected_speaker,
                "cache": str(Path(args.cache).resolve()),
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "max_target_tokens": max_target_tokens,
                "trainable_parameters": 256,
                "unit_normalize": not args.no_unit_normalize,
                "l2_lambda": float(args.l2_lambda),
                "max_delta_norm": float(args.max_delta_norm),
                "weight_decay": float(args.weight_decay),
                "grad_clip": float(args.grad_clip),
                "training": {
                    "epoch": epoch,
                    "initial_valid": base_eval,
                    "best_valid_loss": best_valid,
                    "history": history,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                "adoption_gate": {
                    "baseline_valid_loss": float(base_eval["loss"]),
                    "required_improvement": float(args.min_delta),
                    "accepted_for_audio_evaluation": True,
                    "release_status": "unreviewed",
                },
                "quality_claim": "not established until matched free-running text and speaker evaluation",
            }
            save_checkpoint(
                checkpoint,
                base=base,
                delta=delta,
                adapted=adapted,
                metadata=best_metadata,
            )
            best_saved = True
        else:
            no_improvement += 1
        if no_improvement >= int(args.patience):
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
        if args.max_steps and (epoch - 1) * len(order) + len(order) >= args.max_steps:
            break

    if not best_saved:
        # Keep a usable identity-vector artifact even when the validation gate
        # rejects all updates.  The report still marks it as rejected.
        best_metadata = {
            "experiment": "nano_t3_conditioning_vector",
            "mode": "conditioning_only",
            "speaker_id": selected_speaker,
            "quality_claim": "rejected by validation gate; identity vector only",
        }
        save_checkpoint(
            checkpoint,
            base=base,
            delta=torch.zeros_like(delta),
            adapted=base,
            metadata=best_metadata,
        )

    report = {
        "format": "nano_t3_conditioning_fit_report_v1",
        "checkpoint": str(checkpoint.resolve()),
        "vector_json": str(checkpoint.with_suffix(".json").resolve()),
        "checkpoint_written": best_saved,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "initial_valid": base_eval,
        "adoption": {
            "accepted_by_teacher_forced_gate": best_saved,
            "reason": "heldout loss improved beyond min_delta" if best_saved else "no heldout improvement beyond min_delta",
            "speaker_id": selected_speaker,
        },
        "history": history,
        "trainable_parameters": 256,
        "delta_norm": float(delta.detach().norm().item()),
        "adapted_embedding_norm": float(current_embedding().detach().norm().item()),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "lr": float(args.lr),
            "l2_lambda": float(args.l2_lambda),
            "max_delta_norm": float(args.max_delta_norm),
            "unit_normalize": not args.no_unit_normalize,
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    }
    write_json(checkpoint.with_name(checkpoint.stem + "_report.json"), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--speaker-id")
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--l2-lambda", type=float, default=1e-2)
    parser.add_argument("--max-delta-norm", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--no-unit-normalize",
        action="store_true",
        help="disable unit normalization of base + delta (diagnostic only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
