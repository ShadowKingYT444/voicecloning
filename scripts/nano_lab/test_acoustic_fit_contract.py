"""Fault-injection checks for interrupted or changed acoustic preparation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import fit_decoder_embedding as fit


def prepared(tmp_path, monkeypatch):
    cache_rows=[]; prepared_rows=[]
    for name,split in [('a','train'),('b','valid')]:
        audio=tmp_path/(name+'.wav'); audio.write_bytes(b'recorded-source-'+name.encode())
        original=tmp_path/(name+'.source.npz');np.savez(original,source_mel=np.zeros((80,4),np.float32))
        mel=tmp_path/(name+'.mel.npz');np.savez(mel,source_mel=np.zeros((80,4),np.float32),reconstructed_mel=np.zeros((80,10),np.float32))
        cache_rows.append(dict(id=name,split=split,audio_path=str(audio),speech_tokens=[1,2],source_mel_path=str(original)))
        prepared_rows.append(dict(id=name,split=split,audio_path=str(audio),source_sha256=fit._sha256(audio),speech_token_count=2,seed=31,mel_path=str(mel),mel_sha256=fit._sha256(mel)))
    cache=tmp_path/'cache.json';cache.write_text(json.dumps(dict(format='nano_acoustic_cache_v1',rows=cache_rows)))
    monkeypatch.setattr(fit,'_load_cache',lambda path:json.loads(path.read_text()))
    cond=tmp_path/'conditionals.pt';cond.write_bytes(b'fixed-conditionals')
    manifest=dict(format='nano_mel_calibration_prepare_v1',acoustic_only=True,status='ready',completed_rows=2,
        reference_native_parity={'status':'passed'},reference_source_overlap={'status':'passed'},
        cache=str(cache),cache_sha256=fit._sha256(cache),conditionals_path=str(cond),conditioning_sha256=fit._sha256(cond),rows=prepared_rows)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return path,manifest


def test_complete_acoustic_prepare_and_partial_rejection(tmp_path,monkeypatch):
    path,manifest=prepared(tmp_path,monkeypatch)
    assert len(fit._validate_prepare_inputs(tmp_path)[1])==2
    manifest['status']='partial';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):fit._validate_prepare_inputs(tmp_path)


def test_changed_prepared_target_rejected_even_if_artifact_hash_updated(tmp_path,monkeypatch):
    path,manifest=prepared(tmp_path,monkeypatch);row=manifest['rows'][0];mel=Path(row['mel_path'])
    np.savez(mel,source_mel=np.ones((80,4),np.float32),reconstructed_mel=np.zeros((80,10),np.float32))
    with pytest.raises(ValueError,match='SHA-256'):fit._validate_prepare_inputs(tmp_path)
    row['mel_sha256']=fit._sha256(mel);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='source mel differs'):fit._validate_prepare_inputs(tmp_path)


def test_acoustic_cache_cannot_bypass_complete_preparation_flag(tmp_path,monkeypatch):
    path,manifest=prepared(tmp_path,monkeypatch);manifest.pop('acoustic_only');path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='explicitly acoustic-only'):fit._validate_prepare_inputs(tmp_path)
