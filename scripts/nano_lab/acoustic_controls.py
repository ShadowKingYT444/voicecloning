"""Explicit acoustic noise control for research without changing base weights.

Upstream CausalConditionalCFM accepts temperature but does not use it. Its
distilled meanflow branch also does not use inference_cfg_rate. Preserve the
default forward exactly; only replace it when noise amplitude is changed.
"""
import torch


def configure_acoustics(model, *, noise_scale=1.0, guidance=None):
    decoder=model.s3gen.flow.decoder
    if not hasattr(decoder,"_lab_original_forward"):
        decoder._lab_original_forward=decoder.forward
        decoder._lab_original_cfg=decoder.inference_cfg_rate
    decoder.inference_cfg_rate=decoder._lab_original_cfg if guidance is None else guidance
    if model.s3gen.meanflow and guidance is not None:
        raise ValueError("Meanflow is already distilled with guidance; acoustic_cfg has no effect")
    if not 0 < noise_scale <= 2:
        raise ValueError("noise_scale must be in (0,2]")
    if noise_scale == 1:
        decoder.forward=decoder._lab_original_forward
        return

    @torch.inference_mode()
    def forward(mu,mask,n_timesteps,temperature=1.,spks=None,cond=None,noised_mels=None,meanflow=False):
        z=torch.randn_like(mu)
        if noised_mels is not None:
            prompt_len=mu.size(2)-noised_mels.size(2)
            z[...,prompt_len:]=noised_mels
        z=z*noise_scale
        times=torch.linspace(0,1,n_timesteps+1,device=mu.device,dtype=mu.dtype)
        if not meanflow and decoder.t_scheduler=='cosine':
            times=1-torch.cos(times*.5*torch.pi)
        if meanflow:
            return decoder.basic_euler(z,t_span=times,mu=mu,mask=mask,spks=spks,cond=cond),None
        return decoder.solve_euler(z,t_span=times,mu=mu,mask=mask,spks=spks,cond=cond,meanflow=False),None
    decoder.forward=forward
