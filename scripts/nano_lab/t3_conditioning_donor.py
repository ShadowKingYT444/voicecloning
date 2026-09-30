"""Copy selected T3 conditioning fields between loaded Conditionals objects.

The helper is intentionally narrow.  It can copy the donor prompt-token
sequence, the donor 256-D speaker embedding, or both.  It never reads or
changes ``destination.gen``.  All donor and destination checks happen before
the first assignment, so a malformed combined request cannot leave a partial
mutation behind.

Torch is imported inside the public function.  Importing this module is safe
in cache and manifest tools that do not load a model.
"""

from __future__ import annotations

from typing import Any


VALID_MODES = ("prompt", "speaker", "prompt_and_speaker")
PROMPT_FIELD = "t3.cond_prompt_speech_tokens"
SPEAKER_FIELD = "t3.speaker_emb"
PROMPT_EMB_FIELD = "t3.cond_prompt_speech_emb"
PROMPT_VOCAB = 6561
# ChatterboxTurboTTS.from_local overrides the YAML's 250 with 375 for Nano.
PROMPT_MAX_LENGTH = 375
SPEAKER_DIM = 256


def _require_t3(conditionals: Any, label: str) -> Any:
    t3 = getattr(conditionals, "t3", None)
    if t3 is None:
        raise ValueError(f"{label} conditionals has no t3 container")
    return t3


def _validate_prompt(torch: Any, t3: Any, label: str) -> Any:
    prompt = getattr(t3, "cond_prompt_speech_tokens", None)
    if not torch.is_tensor(prompt):
        raise ValueError(f"{label} {PROMPT_FIELD} must be a Torch tensor")
    if prompt.ndim != 2 or tuple(prompt.shape)[0] != 1 or not 1 <= int(prompt.shape[1]) <= PROMPT_MAX_LENGTH:
        raise ValueError(f"{label} {PROMPT_FIELD} must have shape (1,N), 1<=N<={PROMPT_MAX_LENGTH}")
    if prompt.dtype != torch.int64:
        raise ValueError(f"{label} {PROMPT_FIELD} must use int64 IDs")
    if torch.any(prompt < 0) or torch.any(prompt >= PROMPT_VOCAB):
        raise ValueError(f"{label} {PROMPT_FIELD} contains an out-of-range ID")
    return prompt


def _validate_speaker(torch: Any, t3: Any, label: str) -> Any:
    speaker = getattr(t3, "speaker_emb", None)
    if not torch.is_tensor(speaker):
        raise ValueError(f"{label} {SPEAKER_FIELD} must be a Torch tensor")
    if speaker.ndim != 2 or tuple(speaker.shape) != (1, SPEAKER_DIM):
        raise ValueError(f"{label} {SPEAKER_FIELD} must have shape (1,{SPEAKER_DIM})")
    if not speaker.is_floating_point() or not torch.isfinite(speaker).all():
        raise ValueError(f"{label} {SPEAKER_FIELD} must be finite floating point")
    return speaker


def _clone_for_destination(torch: Any, source: Any, destination: Any, *, field: str) -> Any:
    """Clone a donor tensor on the destination tensor's device and dtype."""

    if torch.is_tensor(destination):
        return source.detach().to(device=destination.device, dtype=destination.dtype).clone()
    # A real Conditionals object always has both fields.  The fallback keeps
    # the helper usable with a minimal fake in contract tests while preserving
    # the donor's device and dtype when no destination tensor exists.
    return source.detach().clone()


def copy_t3_conditioning(donor: Any, destination: Any, mode: str) -> dict[str, Any]:
    """Copy selected T3 fields from ``donor`` into ``destination``.

    Parameters
    ----------
    donor, destination:
        Loaded vendor ``Conditionals``-like objects with a ``.t3`` object.
    mode:
        ``"prompt"``, ``"speaker"``, or ``"prompt_and_speaker"``.

    Returns
    -------
    dict
        Provenance for the selected fields.  It contains no unselected field
        values and leaves file-hash recording to the caller.
    """

    import torch

    if mode not in VALID_MODES:
        raise ValueError(f"invalid T3 donor mode {mode!r}; expected one of {VALID_MODES}")
    donor_t3 = _require_t3(donor, "donor")
    destination_t3 = _require_t3(destination, "destination")
    use_prompt = mode in {"prompt", "prompt_and_speaker"}
    use_speaker = mode in {"speaker", "prompt_and_speaker"}

    # Validate every selected donor field and every destination copy target
    # before preparing or assigning either value.
    donor_prompt = _validate_prompt(torch, donor_t3, "donor") if use_prompt else None
    donor_speaker = _validate_speaker(torch, donor_t3, "donor") if use_speaker else None
    destination_prompt = getattr(destination_t3, "cond_prompt_speech_tokens", None)
    destination_speaker = getattr(destination_t3, "speaker_emb", None)
    if use_prompt and destination_prompt is not None and not torch.is_tensor(destination_prompt):
        raise ValueError(f"destination {PROMPT_FIELD} must be a tensor or None")
    if use_prompt and torch.is_tensor(destination_prompt):
        if destination_prompt.ndim != 2 or tuple(destination_prompt.shape)[0] != 1:
            raise ValueError(f"destination {PROMPT_FIELD} has an invalid shape")
        if destination_prompt.dtype != torch.int64:
            raise ValueError(f"destination {PROMPT_FIELD} must use int64 IDs")
    if use_speaker and destination_speaker is not None and not torch.is_tensor(destination_speaker):
        raise ValueError(f"destination {SPEAKER_FIELD} must be a tensor or None")
    if use_speaker and torch.is_tensor(destination_speaker):
        if destination_speaker.ndim != 2 or tuple(destination_speaker.shape) != (1, SPEAKER_DIM):
            raise ValueError(f"destination {SPEAKER_FIELD} has an invalid shape")
        if not destination_speaker.is_floating_point():
            raise ValueError(f"destination {SPEAKER_FIELD} must use a floating dtype")

    prompt_copy = _clone_for_destination(torch, donor_prompt, destination_prompt, field=PROMPT_FIELD) if use_prompt else None
    speaker_copy = _clone_for_destination(torch, donor_speaker, destination_speaker, field=SPEAKER_FIELD) if use_speaker else None
    if use_speaker and not torch.isfinite(speaker_copy).all():
        raise ValueError('speaker embedding conversion overflowed the destination dtype')

    # No validation or conversion remains after this point.  Mutations are
    # therefore atomic with respect to all input-contract failures.
    if use_prompt:
        destination_t3.cond_prompt_speech_tokens = prompt_copy
        # T3CondEnc lazily derives this embedding from the new IDs.  Keeping
        # the old derived sequence would silently pair the donor IDs with the
        # destination prompt features.
        destination_t3.cond_prompt_speech_emb = None
    if use_speaker:
        destination_t3.speaker_emb = speaker_copy

    selected = []
    if use_prompt:
        selected.append(PROMPT_FIELD)
    if use_speaker:
        selected.append(SPEAKER_FIELD)
    return {
        "mode": mode,
        "selected_fields": selected,
        "prompt_embedding_invalidated": bool(use_prompt),
        "prompt_shape": list(donor_prompt.shape) if donor_prompt is not None else None,
        "speaker_shape": list(donor_speaker.shape) if donor_speaker is not None else None,
        "gen_untouched": True,
    }


# Explicit alias for callers that prefer a verb matching the operation.
copy_t3_donor = copy_t3_conditioning


__all__ = [
    "PROMPT_FIELD",
    "PROMPT_EMB_FIELD",
    "PROMPT_MAX_LENGTH",
    "PROMPT_VOCAB",
    "SPEAKER_DIM",
    "SPEAKER_FIELD",
    "VALID_MODES",
    "copy_t3_conditioning",
    "copy_t3_donor",
]
