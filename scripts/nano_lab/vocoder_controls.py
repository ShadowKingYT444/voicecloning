"""Experimental control of HiFT's voiced excitation noise.

This changes a model input distribution, not a post-processing noise gate.
Unvoiced excitation remains unchanged. The default reproduces the checkpoint's
constructor setting. Reduced noise requires voice/content/listening validation.
"""
import math


def configure_vocoder(model, *, voiced_noise_scale=1.0):
    scale=float(voiced_noise_scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("voiced_noise_scale must be finite and in [0,1]")
    generator=model.s3gen.mel2wav.m_source.l_sin_gen
    if not hasattr(generator,"_lab_original_noise_std"):
        generator._lab_original_noise_std=generator.noise_std
    generator.noise_std=generator._lab_original_noise_std*scale
    return {"voiced_noise_scale":scale,"voiced_noise_std":generator.noise_std,
            "unvoiced_noise_std":generator.sine_amp/3,"experimental":scale!=1.0}
