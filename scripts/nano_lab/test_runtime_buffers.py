"""Regression test for deterministic buffers omitted by old S3Gen weights."""
import tempfile
from pathlib import Path
import librosa
import numpy as np
import torch
from safetensors.torch import save_file
from runtime import _stream_load_safetensors,_repair_runtime_buffers,_repair_t3_runtime_buffers

class Tokenizer(torch.nn.Module):
    ignore_state_dict_missing=('_mel_filters','window')
    def __init__(self):
        super().__init__()
        self.register_buffer('window',torch.full((400,),float('nan')))
        self.register_buffer('_mel_filters',torch.full((80,201),float('nan')))
        self.weight=torch.nn.Parameter(torch.zeros(1))

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.tokenizer=Tokenizer()

def main():
    model=Decoder()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'old.safetensors';save_file({'tokenizer.weight':torch.ones(1)},path)
        report=_stream_load_safetensors(model,path,device=torch.device('cpu'))
    assert report['missing_tensors']==[]
    assert set(report['ignored_missing_tensors'])=={'tokenizer.window','tokenizer._mel_filters'}
    _repair_runtime_buffers(model,torch.device('cpu'),torch.float32,report['ignored_missing_tensors'])
    assert torch.equal(model.tokenizer.window,torch.hann_window(400))
    np.testing.assert_array_equal(model.tokenizer._mel_filters.numpy(),librosa.filters.mel(sr=16000,n_fft=400,n_mels=80))
    assert model.tokenizer.weight.item()==1
    assert torch.isfinite(model.trim_fade).all()
    layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    for layer in layers:
        layer.register_buffer('bias',torch.empty(1,1,32,32,dtype=torch.bool),persistent=False)
        layer.register_buffer('masked_bias',torch.empty(()),persistent=False)
    _repair_t3_runtime_buffers(layers,torch.device('cpu'))
    expected=torch.ones(32,32,dtype=torch.bool).tril().view(1,1,32,32)
    assert all(torch.equal(layer.bias,expected) for layer in layers)
    assert len({layer.bias.data_ptr() for layer in layers})==1
    assert all(layer.masked_bias.item()==-1e4 for layer in layers)
    print('PASS: omitted buffers restored after to_empty-style allocation; checkpoint parameter preserved')
    print('PASS: identical read-only causal masks share storage across layers')

if __name__=='__main__':main()
