"""Model-free integrity checks for the acoustic ONNX stage."""
import hashlib
import json

import pytest

from onnx_staged_runtime import _read_manifest


def fixture(tmp_path):
    stage = tmp_path / 'meanflow_estimator'
    stage.mkdir()
    for name, data in [('graph.onnx', b'graph'), ('weights.bin', b'weights'), ('external.json', b'{}')]:
        (stage / name).write_bytes(data)
    sha = lambda name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
    patch = dict(format='nano_decoder_attention_merged_v1', strength=1.0,
                 weights_file='weights.bin', external_report_file='external.json',
                 graph_sha256=sha('graph.onnx'), patched_weights_sha256=sha('weights.bin'),
                 external_report_sha256=sha('external.json'))
    patch.update({key: 'a' * 64 for key in ('adapter_sha256', 'inventory_sha256', 'model_checkpoint_sha256',
                                          'initial_conditionals_sha256', 'base_weights_sha256')})
    verification = dict(status='verified', graph_sha256=patch['graph_sha256'],
                        weights_sha256=patch['patched_weights_sha256'], adapter_sha256=patch['adapter_sha256'], strength=1.0)
    manifest = dict(stage='meanflow_estimator', status='exported', graph=dict(path='graph.onnx'),
                    verification=dict(status='verified'), adapter_patch=patch,
                    decoder_attention_verification=verification)
    return stage, manifest


def save(stage, manifest):
    (stage / 'manifest.json').write_text(json.dumps(manifest))


def test_valid_and_unverified_scopes(tmp_path):
    stage, manifest = fixture(tmp_path)
    save(stage, manifest)
    assert _read_manifest(tmp_path, 'meanflow_estimator')['_graph_path'].endswith('graph.onnx')
    manifest['decoder_attention_verification']['status'] = 'pending'
    save(stage, manifest)
    with pytest.raises(RuntimeError, match='numerical verification'):
        _read_manifest(tmp_path, 'meanflow_estimator')
    _read_manifest(tmp_path, 'meanflow_estimator', require_verified=False)


@pytest.mark.parametrize('file', ['graph.onnx', 'weights.bin', 'external.json'])
def test_changed_bytes_fail_even_verification_only(tmp_path, file):
    stage, manifest = fixture(tmp_path)
    save(stage, manifest)
    _read_manifest(tmp_path, 'meanflow_estimator')
    (stage / file).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        _read_manifest(tmp_path, 'meanflow_estimator', require_verified=False)


@pytest.mark.parametrize('key,value', [('weights_sha256', 'b' * 64), ('adapter_sha256', 'b' * 64), ('strength', .5)])
def test_stale_verification_rejected(tmp_path, key, value):
    stage, manifest = fixture(tmp_path)
    manifest['decoder_attention_verification'][key] = value
    save(stage, manifest)
    with pytest.raises(RuntimeError, match='Stale'):
        _read_manifest(tmp_path, 'meanflow_estimator')


def test_wrong_path_and_legacy_verification_rejected(tmp_path):
    stage, manifest = fixture(tmp_path)
    manifest.pop('decoder_attention_verification')
    save(stage, manifest)
    with pytest.raises(RuntimeError, match='numerical verification'):
        _read_manifest(tmp_path, 'meanflow_estimator')
    manifest['adapter_patch']['weights_file'] = '../weights.bin'
    save(stage, manifest)
    with pytest.raises(RuntimeError, match='file path'):
        _read_manifest(tmp_path, 'meanflow_estimator', require_verified=False)


def test_profile_binds_adapter_cache_and_rejects_silent_base(tmp_path):
    from onnx_decoder_attention_profile import validate_decoder_attention_profile
    stage, manifest = fixture(tmp_path)
    (tmp_path / 'adapter.pt').write_bytes(b'adapter')
    (tmp_path / 'conditionals.pt').write_bytes(b'cache')
    patch = manifest['adapter_patch']
    patch['adapter_sha256'] = hashlib.sha256(b'adapter').hexdigest()
    patch['initial_conditionals_sha256'] = hashlib.sha256(b'cache').hexdigest()
    manifest['decoder_attention_verification']['adapter_sha256'] = patch['adapter_sha256']
    save(stage, manifest)
    profile = dict(decoder_attention='adapter.pt', decoder_attention_strength=1.0, conditioning_cache='conditionals.pt')
    result = validate_decoder_attention_profile(profile, tmp_path, tmp_path)
    assert result['enabled']
    with pytest.raises(ValueError, match='strengths differ'):
        validate_decoder_attention_profile({}, tmp_path, tmp_path)
    (tmp_path / 'conditionals.pt').write_bytes(b'other-cache')
    with pytest.raises(ValueError, match='conditioning cache differs'):
        validate_decoder_attention_profile(profile, tmp_path, tmp_path)
    manifest.pop('adapter_patch')
    save(stage, manifest)
    with pytest.raises(ValueError, match='folded acoustic'):
        validate_decoder_attention_profile(profile, tmp_path, tmp_path)


def test_cpu_proof_does_not_authorize_unverified_cuda(tmp_path):
    from onnx_staged_runtime import _require_decoder_attention_provider
    _, manifest = fixture(tmp_path)
    manifest['decoder_attention_verification']['provider_reports'] = {'cpu': 'verify_cpu.json'}
    _require_decoder_attention_provider(manifest, 'cpu')
    with pytest.raises(RuntimeError, match='requested ORT provider'):
        _require_decoder_attention_provider(manifest, 'cuda')
    manifest['decoder_attention_verification']['provider_reports']['cuda'] = 'verify_cuda.json'
    _require_decoder_attention_provider(manifest, 'cuda')
