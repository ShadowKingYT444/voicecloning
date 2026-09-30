import hashlib

import pytest

from profile_t3_donor import copy_t3_payload, resolve_t3_donor


def test_donor_cache_is_pinned_before_loading(tmp_path):
    path = tmp_path / 'donor.pt'
    path.write_bytes(b'cache')
    profile = dict(t3_donor_cache='donor.pt', t3_donor_cache_sha256=hashlib.sha256(b'cache').hexdigest())
    assert resolve_t3_donor(profile, tmp_path)['mode'] == 'prompt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        resolve_t3_donor(profile, tmp_path)


def test_missing_pin_and_orphan_mode_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='pin'):
        resolve_t3_donor(dict(t3_donor_cache='donor.pt'), tmp_path)
    with pytest.raises(ValueError, match='requires'):
        resolve_t3_donor(dict(t3_donor_mode='prompt'), tmp_path)
    assert resolve_t3_donor({}, tmp_path) is None


def test_weights_only_copy_changes_prompt_and_preserves_gen():
    import torch
    original_gen = dict(embedding=torch.ones(1, 192), prompt_feat=torch.ones(1, 10, 80))
    original_t3 = dict(speaker_emb=torch.ones(1, 256), cond_prompt_speech_tokens=torch.tensor([[1, 2]]),
                       cond_prompt_speech_emb=torch.ones(1, 2, 768))
    destination = dict(t3=original_t3, gen=original_gen)
    donor = dict(t3=dict(speaker_emb=torch.zeros(1, 256), cond_prompt_speech_tokens=torch.tensor([[71, 72, 73]])),
                 gen=dict(embedding=torch.zeros(1, 192)))
    metadata = copy_t3_payload(donor, destination, 'prompt')
    assert metadata['gen_untouched']
    assert destination['gen'] is original_gen
    assert torch.equal(destination['gen']['embedding'], torch.ones(1, 192))
    assert destination['t3']['cond_prompt_speech_tokens'].tolist() == [[71, 72, 73]]
    assert destination['t3']['speaker_emb'] is original_t3['speaker_emb']
    assert destination['t3']['cond_prompt_speech_emb'] is None
    assert original_t3['cond_prompt_speech_tokens'].tolist() == [[1, 2]]


def test_invalid_donor_does_not_replace_destination_dictionary():
    import torch
    target = dict(t3=dict(speaker_emb=torch.ones(1, 256), cond_prompt_speech_tokens=torch.tensor([[1, 2]])))
    original = target['t3']
    donor = dict(t3=dict(speaker_emb=torch.full((1, 256), float('nan')),
                        cond_prompt_speech_tokens=torch.tensor([[51, 52]])))
    with pytest.raises(ValueError):
        copy_t3_payload(donor, target, 'prompt_and_speaker')
    assert target['t3'] is original
