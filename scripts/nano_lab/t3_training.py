"""Load only the frozen speech-token model for cached-feature training."""
from pathlib import Path
from types import SimpleNamespace
import torch
from runtime import T3,_nano_hp,_delete_unused_t3_weights,_stream_load_safetensors,_repair_t3_runtime_buffers

def load_cached_t3(model_dir,device):
    device=torch.device(device)
    with torch.device('meta'):t3=T3(_nano_hp())
    _delete_unused_t3_weights(t3)
    t3.to_empty(device=device)
    report=_stream_load_safetensors(t3,Path(model_dir)/'t3_nano_v1.safetensors',device=device,strict=True,skip_prefixes=('tfmr.wte','text_head'))
    _repair_t3_runtime_buffers(t3,device)
    return SimpleNamespace(t3=t3.eval(),s3gen=None,ve=None,training_load_report=report)
