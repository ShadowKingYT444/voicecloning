#!/usr/bin/env python3
"""Run a bounded CPU reference trace for the pinned Nano browser pipeline.

Run from the repository root through scripts/nano_lab/bounded_job.py. This
loads only the published embedding and language-model graphs by default. Pass
--decode to release those sessions, then load the published decoder graph and
write one browser-shaped PCM16 WAV.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time
import wave

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REVISION = '4a66d7dab72a9e98f24b515d49a1d7a81632df2e'
REPOSITORY = 'owensong/chatterbox-nano-ONNX'
SHA256SUMS_SHA256 = 'ecd0b0ee1fde0f9342564fda72fe1ebb4907baa6c9d013c09052bcad5220dcf4'
REFERENCE_SHA256 = '67f94a868976b22a47bce8fd00a873d4c5b7085ba6eedd698f8a898a26ce76c0'
STATE_SHA256 = '6f56c0dd844ecc038e5b60debff22d1e16b44c9d13386d0f7a91156bfca1b38e'
STATE_BYTES = 331200
DONOR_STATE_SHA256 = '77a2955a6b50122c581cf211f86d7c624381b7dfa8bb8ce61af3fb6359e16496'
DONOR_STATE_BYTES = 1083840
DONOR_CACHE_SHA256 = '3c78b8bedb9b5ac94d5aaf900d509162cb4c5274f59cf7fb525117498786c40c'
DONOR_CHECKPOINT_SHA256 = '72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9'
DONOR_PREFIX_SHA256 = '6d27b0087b442abb8264d0fea19425be649f663519830e9ca28bdb57c80fe311'
DONOR_PREFIX_FRAMES = 334
DONOR_PROMPT_TOKENS = 333
DONOR_ENCODER_SCOPE = ('three retained decoder-conditioning tensors copied bitwise from source_public_state; '
                       'does not describe native T3 prefix')
DONOR_PREFIX_PROCESSING = ('cond_enc.spkr_enc(speaker_emb) concatenated with '
                           'speech_emb(cond_prompt_speech_tokens)')
GRAPH_NAMES = ('embed_tokens_fp16', 'language_model_q4f16', 'conditional_decoder_q4')
DEFAULT_TEXT = 'Take a slow breath in, and let your shoulders relax.'
START_SPEECH = 6561
STOP_SPEECH = 6562
SILENCE = 4299
VOCAB_SIZE = STOP_SPEECH + 1
TOKEN_LIMIT = 256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_model_assets(source_dir):
    sums_path = source_dir / 'SHA256SUMS'
    if file_sha256(sums_path) != SHA256SUMS_SHA256:
        raise ValueError('Source SHA256SUMS does not match the pinned conversion.')
    expected = {}
    for line_number, line in enumerate(sums_path.read_text().splitlines(), 1):
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*').strip()
        if name in expected:
            raise ValueError(f'Duplicate SHA256SUMS entry on line {line_number}.')
        expected[name] = digest
    paths = ['tokenizer.json']
    for name in GRAPH_NAMES:
        paths.extend((f'onnx/{name}.onnx', f'onnx/{name}.onnx_data'))
    for name in paths:
        path = source_dir / name
        if name not in expected or file_sha256(path) != expected[name]:
            raise ValueError(f'Pinned public model asset failed SHA-256 verification: {name}.')
    return expected


def read_voice_state(manifest_path, data_path, model_hashes, expected_source):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema_version') != 1
            or manifest.get('format') != 'chatterbox-nano-reference-state-v1'):
        raise ValueError('Fixed voice-state schema is not supported.')
    model = manifest['model']
    if (model.get('repo_id') != REPOSITORY or model.get('revision') != REVISION
            or model.get('sha256sums_sha256') != SHA256SUMS_SHA256):
        raise ValueError('Fixed voice state is not from the pinned public conversion.')
    conditioning = manifest['conditioning']
    if (manifest['reference'].get('path') != 'browser_tts/public/voice/asmr_t3_seed47_fit.wav'
            or manifest['reference'].get('sha256') != REFERENCE_SHA256
            or conditioning.get('source') != expected_source
            or conditioning.get('speaker_specific') is not True
            or conditioning.get('zero_shot') is not False
            or conditioning.get('fitted_adapter_applied') is not False):
        raise ValueError('Fixed voice state does not match the requested pinned speaker-specific source.')
    encoder_graph = model['encoder_graph']
    encoder_data = model['encoder_external_weights']
    if (model_hashes.get(encoder_graph['path']) != encoder_graph['sha256']
            or model_hashes.get(encoder_data['path']) != encoder_data['sha256']):
        raise ValueError('Fixed voice-state encoder provenance differs from pinned SHA256SUMS.')
    encoder_execution = manifest.get('encoder_execution', {})
    if (encoder_execution.get('provider') != 'CPUExecutionProvider'
            or encoder_execution.get('onnxruntime_version') != '1.29.0'
            or encoder_execution.get('graph_optimization_level') != 'ORT_ENABLE_ALL'):
        raise ValueError('Fixed voice state does not record the pinned public CPU encoder execution.')

    if expected_source == 'native_t3_donor':
        donor = conditioning.get('native_t3_donor', {})
        if (donor.get('profile') != 'clean_asmr_original'
                or donor.get('cache') != {
                    'path': 'artifacts/nano_lab/decoder_embedding_fit/conditionals.pt',
                    'sha256': DONOR_CACHE_SHA256,
                }
                or donor.get('checkpoint') != {
                    'path': 'models/chatterbox-nano/t3_nano_v1.safetensors',
                    'repo_id': 'ResembleAI/chatterbox-nano',
                    'revision': '71ccd1d0081b430592cea481f4307e764e07bc64',
                    'sha256': DONOR_CHECKPOINT_SHA256,
                }
                or donor.get('prefix') != {
                    'frames': DONOR_PREFIX_FRAMES,
                    'path': 'artifacts/nano_lab/browser_measurements/local_asmr_cache_prefix_20261002/prefix.npy',
                    'processing': DONOR_PREFIX_PROCESSING,
                    'prompt_tokens': DONOR_PROMPT_TOKENS,
                    'sha256': DONOR_PREFIX_SHA256,
                }):
            raise ValueError('Native T3 donor source provenance does not match the pinned candidate.')
        execution = donor.get('execution', {})
        if (execution.get('provider') != 'CPU' or execution.get('framework') != 'PyTorch'
                or re.fullmatch(r'2\.\d+\.\d+\+cpu', execution.get('version', '')) is None
                or execution.get('threads') != 2):
            raise ValueError('Native T3 donor execution provenance is not pinned CPU PyTorch.')
        if (manifest.get('source_public_state') != {
                'path': 'browser_tts/public/voice/asmr-state.bin', 'sha256': STATE_SHA256,
            }
                or manifest.get('encoder_execution', {}).get('scope') != DONOR_ENCODER_SCOPE):
            raise ValueError('Native T3 donor does not identify the retained public decoder state.')
        expected_state_hash = DONOR_STATE_SHA256
        expected_state_bytes = DONOR_STATE_BYTES
        feature_frames = DONOR_PREFIX_FRAMES
    else:
        expected_state_hash = STATE_SHA256
        expected_state_bytes = STATE_BYTES
        feature_frames = 89
        if conditioning.get('source') != 'generated_reference' or conditioning.get('native_t3_donor'):
            raise ValueError('Legacy voice-state provenance changed.')

    data = data_path.read_bytes()
    state_meta = manifest['data']
    if (state_meta.get('file') != 'asmr-state.bin'
            or len(data) != expected_state_bytes or state_meta.get('size_bytes') != expected_state_bytes
            or state_meta.get('sha256') != expected_state_hash
            or hashlib.sha256(data).hexdigest() != expected_state_hash):
        raise ValueError('Fixed voice-state binary failed its pinned size/SHA-256 check.')

    expected = (
        ('audio_features', 'audio_features', 'float32', (1, feature_frames, 768)),
        ('audio_tokens', 'audio_tokens', 'int64', (1, 88)),
        ('speaker_embeddings', 'speaker_embeddings', 'float32', (1, 192)),
        ('speaker_features', 'speaker_features', 'float32', (1, 176, 80)),
    )
    tensors = {}
    offset = 0
    entries = manifest.get('tensors', [])
    if len(entries) != len(expected):
        raise ValueError('Fixed voice state must contain four pinned tensors.')
    for entry, (name, role, dtype_name, dims) in zip(entries, expected):
        dtype = np.dtype('<f4' if dtype_name == 'float32' else '<i8')
        byte_length = math.prod(dims) * dtype.itemsize
        if (entry.get('name') != name or entry.get('role') != role or entry.get('type') != dtype_name
                or tuple(entry.get('dims', ())) != dims
                or entry.get('offset') != offset or entry.get('byte_length') != byte_length
                or entry.get('serialization_round_trip_exact') is not True):
            raise ValueError(f'Fixed voice-state tensor contract differs for {name}.')
        values = np.frombuffer(data, dtype=dtype, count=math.prod(dims), offset=offset).reshape(dims).copy()
        if dtype_name == 'float32' and not np.isfinite(values).all():
            raise ValueError(f'Fixed voice-state tensor {name} contains non-finite values.')
        if name == 'audio_tokens' and (np.any(values < 0) or np.any(values >= VOCAB_SIZE)):
            raise ValueError('Fixed voice-state audio_tokens contain an invalid speech token.')
        tensors[name] = values
        offset += byte_length
    if offset != len(data):
        raise ValueError('Fixed voice-state tensor ranges do not cover the binary.')
    return manifest, tensors


def rss_snapshot():
    current_mib = None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                current_mib = int(line.split()[1]) / 1024
                break
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024
    return {'current_rss_mib': current_mib, 'peak_rss_mib': peak_mib}


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def create_session(source_dir, name):
    return ort.InferenceSession(
        str(source_dir / 'onnx' / f'{name}.onnx'),
        sess_options=session_options(), providers=['CPUExecutionProvider'])


def metadata(session):
    def describe(item):
        return {'name': item.name, 'type': item.type, 'shape': list(item.shape)}
    return {'inputs': [describe(item) for item in session.get_inputs()],
            'outputs': [describe(item) for item in session.get_outputs()]}


def numpy_type(ort_type):
    if ort_type == 'tensor(float16)':
        return np.float16
    if ort_type == 'tensor(float)':
        return np.float32
    if ort_type == 'tensor(int64)':
        return np.int64
    raise TypeError(f'Unsupported ONNX input type: {ort_type}.')


def ranked_ids(scores, count):
    return sorted(range(len(scores)), key=lambda token_id: (-float(scores[token_id]), token_id))[:count]


class WorkerSampler:
    def __init__(self, seed):
        self.state = seed & 0xffffffff

    def random(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xffffffff
        return self.state / 4294967296

    def sample(self, scores, history):
        if (len(scores) != VOCAB_SIZE or np.isnan(scores).any()
                or np.isposinf(scores).any() or not np.isfinite(scores).any()):
            raise ValueError('The language model returned invalid speech logits.')
        repeated = set(history)
        adjusted = []
        for token_id, score in enumerate(scores):
            value = float(score) / 0.8
            if token_id in repeated:
                value = value * 1.2 if value < 0 else value / 1.2
            adjusted.append(value)
        top = sorted(range(VOCAB_SIZE), key=lambda token_id: (-adjusted[token_id], token_id))[:1000]
        maximum = adjusted[top[0]] if top else 0.0
        weights = [math.exp(adjusted[token_id] - maximum) for token_id in top]
        total = 0.0
        for weight in weights:
            total += weight
        nucleus = []
        cumulative = 0.0
        for token_id, weight in zip(top, weights):
            nucleus.append((token_id, weight))
            cumulative += weight / total
            if cumulative >= 0.95:
                break
        mass = 0.0
        for _, weight in nucleus:
            mass += weight
        choice = self.random() * mass
        for token_id, weight in nucleus:
            choice -= weight
            if choice <= 0:
                return token_id
        return nucleus[-1][0] if nucleus else STOP_SPEECH


def run_decoder(source_dir, tensors, speech_tokens, output_dir):
    started = time.perf_counter()
    session = create_session(source_dir, 'conditional_decoder_q4')
    load_seconds = time.perf_counter() - started
    input_types = {item.name: numpy_type(item.type) for item in session.get_inputs()}
    decoder_ids = np.concatenate((
        tensors['audio_tokens'],
        np.asarray([speech_tokens], dtype=np.int64),
        np.full((1, 3), SILENCE, dtype=np.int64)), axis=1)
    feeds = {
        'speech_tokens': decoder_ids.astype(input_types['speech_tokens'], copy=False),
        'speaker_embeddings': tensors['speaker_embeddings'].astype(input_types['speaker_embeddings'], copy=False),
        'speaker_features': tensors['speaker_features'].astype(input_types['speaker_features'], copy=False),
    }
    output_name = session.get_outputs()[0].name
    inference_started = time.perf_counter()
    audio = np.asarray(session.run([output_name], feeds)[0], dtype=np.float32).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('The CPU decoder returned empty or non-finite audio.')
    np.save(output_dir / 'audio-raw.npy', audio)

    samples = audio.copy()
    fade_frames = round(24000 * 9 / 1000)
    for index in range(samples.size):
        edge = min(index, samples.size - index - 1)
        samples[index] = np.float32(float(samples[index]) * min(1.0, edge / fade_frames))
    pcm = np.empty(samples.size, dtype='<i2')
    for index, sample in enumerate(samples):
        value = max(-1.0, min(1.0, float(sample)))
        pcm[index] = int(value * (32768 if value < 0 else 32767))
    wav_path = output_dir / 'cpu-reference.wav'
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes(pcm.tobytes())
    result = {
        'sample_rate': 24000, 'sample_count': int(samples.size),
        'duration_seconds': samples.size / 24000,
        'session_load_seconds': load_seconds, 'inference_seconds': inference_seconds,
        'decoder_seconds': time.perf_counter() - started,
        'wav_path': str(wav_path), 'wav_bytes': wav_path.stat().st_size,
        'wav_sha256': file_sha256(wav_path), 'audio_raw_npy': str(output_dir / 'audio-raw.npy'),
        'model_metadata': metadata(session),
    }
    del session
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--source-dir', type=Path, default=root / 'models/chatterbox-nano-browser')
    parser.add_argument('--voice-state-dir', type=Path, help='Use the pinned original public voice state.')
    parser.add_argument('--native-donor-state-dir', type=Path,
                        help='Use the pinned 334-frame native T3 donor candidate.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--text', default=DEFAULT_TEXT)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--logit-probes', type=int, default=4)
    parser.add_argument('--decode', action='store_true', help='Run the CPU decoder after releasing embedding and language-model sessions.')
    args = parser.parse_args()
    if args.voice_state_dir is not None and args.native_donor_state_dir is not None:
        parser.error('Select only one of --voice-state-dir and --native-donor-state-dir.')
    native_donor = args.native_donor_state_dir is not None
    expected_source = 'native_t3_donor' if native_donor else 'generated_reference'
    voice_state_dir = args.native_donor_state_dir or args.voice_state_dir or root / 'browser_tts/public/voice'
    default_output = (
        root / 'artifacts/nano_lab/browser_measurements/cpu_reference_native_donor_20261002'
        if native_donor else root / 'artifacts/nano_lab/browser_measurements/cpu_reference_lossless_20261002'
    )
    output_dir = args.output_dir or default_output
    if ort.__version__ != '1.29.0' or np.__version__ != '1.26.4':
        raise ValueError(f'Expected ONNX Runtime 1.29.0 and NumPy 1.26.4; found {ort.__version__} and {np.__version__}.')
    if not args.text.strip() or not 1 <= args.logit_probes <= 32:
        raise ValueError('Text must be non-empty and --logit-probes must be between 1 and 32.')
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        'schema': 'voice-study.cpu-pipeline-probe/v1', 'status': 'running',
        'provider': 'CPUExecutionProvider', 'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__, 'tokenizers_version': __import__('tokenizers').__version__,
        'revision': REVISION, 'repo_id': REPOSITORY, 'text': args.text, 'seed': args.seed,
        'sampling': {'temperature': 0.8, 'repetition_penalty': 1.2, 'top_k': 1000,
                     'top_p': 0.95, 'token_limit': TOKEN_LIMIT, 'start_token': START_SPEECH,
                     'stop_token': STOP_SPEECH, 'sampler': 'JavaScript LCG32 / stable descending score then token ID'},
        'decode_requested': args.decode, 'stage_seconds': {}, 'rss': {}, 'logit_probes': [],
        'limitations': ['CPU reference only; it does not validate WebGPU behavior.',
                        'Fixed speaker state is used as exported; the reference encoder is not loaded or recomputed.',
                        'Voice-state source, prefix length, and speech-start count are recorded in voice_state.'],
        'output_dir': str(output_dir.resolve()),
    }
    try:
        model_hashes = verify_model_assets(args.source_dir)
        report['model_sha256'] = {
            name: model_hashes[f'onnx/{name}.onnx'] for name in GRAPH_NAMES
        }
        report['external_data_sha256'] = {
            name: model_hashes[f'onnx/{name}.onnx_data'] for name in GRAPH_NAMES
        }
        tokenizer_path = args.source_dir / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        ids = tokenizer.encode(args.text, add_special_tokens=True).ids
        report['token_ids'] = ids
        if len(ids) < 3 or ids[-2:] != [50256, 50256]:
            raise ValueError('The pinned tokenizer did not append the two 50256 speech-start markers.')
        report['tokenizer_sha256'] = model_hashes['tokenizer.json']
        voice_manifest, tensors = read_voice_state(
            voice_state_dir / 'asmr-state.json',
            voice_state_dir / 'asmr-state.bin', model_hashes, expected_source)
        speech_start_count = 1 if native_donor else 2
        report['voice_state'] = {
            'source': voice_manifest['conditioning']['source'],
            'manifest_path': str((voice_state_dir / 'asmr-state.json').resolve()),
            'speech_start_count': speech_start_count,
            'audio_feature_frames': int(tensors['audio_features'].shape[1]),
            'sha256': voice_manifest['data']['sha256'], 'bytes': voice_manifest['data']['size_bytes'],
            'reference_sha256': voice_manifest['reference']['sha256'],
            'conditioning': voice_manifest['conditioning'],
            'tensor_shapes': {name: list(value.shape) for name, value in tensors.items()},
        }
        if native_donor:
            _legacy_manifest, legacy_tensors = read_voice_state(
                root / 'browser_tts/public/voice/asmr-state.json',
                root / 'browser_tts/public/voice/asmr-state.bin', model_hashes, 'generated_reference')
            for name in ('audio_tokens', 'speaker_embeddings', 'speaker_features'):
                if tensors[name].tobytes(order='C') != legacy_tensors[name].tobytes(order='C'):
                    raise ValueError(f'Native donor changed the retained public decoder tensor {name}.')
            report['voice_state']['retained_decoder_tensors_match_public_state'] = True
        report['speech_start_count'] = speech_start_count
        report['rss']['before_sessions'] = rss_snapshot()

        t0 = time.perf_counter(); embed = create_session(args.source_dir, 'embed_tokens_fp16')
        report['stage_seconds']['embedding_session_load'] = time.perf_counter() - t0
        report['input_metadata'] = {'embedding': metadata(embed)}
        report['rss']['after_embedding_session'] = rss_snapshot()
        t0 = time.perf_counter(); lm = create_session(args.source_dir, 'language_model_q4f16')
        report['stage_seconds']['language_model_session_load'] = time.perf_counter() - t0
        report['input_metadata']['language_model'] = metadata(lm)
        report['rss']['after_language_model_session'] = rss_snapshot()
        input_meta = {item.name: item for item in lm.get_inputs()}
        output_names = [item.name for item in lm.get_outputs()]
        output_set = set(output_names)
        cache_inputs = [item.name for item in lm.get_inputs() if item.name.startswith('past_key_values.')]
        cache_outputs = [name.replace('past_key_values.', 'present.', 1) for name in cache_inputs]
        if (len(cache_inputs) != 24 or any(name not in output_set for name in cache_outputs)
                or 'inputs_embeds' not in input_meta or 'attention_mask' not in input_meta
                or 'position_ids' not in input_meta or 'logits' not in output_set):
            raise ValueError('The language model input or 24-pair KV cache contract differs from the browser worker.')

        t0 = time.perf_counter()
        text_embeddings = np.asarray(embed.run(None, {'input_ids': np.asarray([ids], dtype=np.int64)})[0], dtype=np.float32)
        if native_donor:
            if (text_embeddings.shape != (1, len(ids), 768)
                    or not np.array_equal(text_embeddings[:, -1, :], text_embeddings[:, -2, :])):
                raise ValueError('Native donor requires identical embedding rows for the final two speech-start markers.')
            text_embeddings = text_embeddings[:, :-1, :]
        combined = np.concatenate((tensors['audio_features'], text_embeddings), axis=1)
        sequence = combined.shape[1]
        feeds = {
            'inputs_embeds': combined.astype(numpy_type(input_meta['inputs_embeds'].type), copy=False),
            'attention_mask': np.ones((1, sequence), dtype=np.int64),
            'position_ids': np.arange(sequence, dtype=np.int64).reshape(1, sequence),
        }
        for name in cache_inputs:
            feeds[name] = np.empty((1, 12, 0, 64), dtype=numpy_type(input_meta[name].type))
        output_values = lm.run(output_names, feeds)
        result = dict(zip(output_names, output_values))
        report['stage_seconds']['initial_embedding_and_lm'] = time.perf_counter() - t0
        report['sequence'] = {'text_tokens': len(ids), 'text_embedding_frames': int(text_embeddings.shape[1]),
                              'speech_start_count': speech_start_count,
                              'audio_feature_frames': int(tensors['audio_features'].shape[1]),
                              'combined_frames': sequence, 'embedding_dim': int(combined.shape[-1])}
        report['rss']['after_initial_lm'] = rss_snapshot()

        sampler = WorkerSampler(args.seed)
        history = [START_SPEECH]
        speech_tokens = []
        step_seconds = []
        reached_eos = False
        generation_started = time.perf_counter()
        for step in range(TOKEN_LIMIT):
            logits = np.asarray(result['logits'], dtype=np.float32).reshape(-1, VOCAB_SIZE)[-1].copy()
            probe = None
            if step < args.logit_probes:
                probe_path = output_dir / f'logits-step-{step:03d}.npy'
                np.save(probe_path, logits)
                top_ids = ranked_ids(logits, 10)
                probe = {'step': step, 'path': str(probe_path), 'shape': list(logits.shape),
                         'top10_raw': [{'token_id': token_id, 'logit': float(logits[token_id])}
                                       for token_id in top_ids]}
                report['logit_probes'].append(probe)
            step_started = time.perf_counter()
            token_id = sampler.sample(logits, history)
            if probe is not None:
                probe['sampled_token'] = token_id
            if token_id == STOP_SPEECH:
                reached_eos = True
                step_seconds.append(time.perf_counter() - step_started)
                break
            speech_tokens.append(token_id)
            history.append(token_id)
            if step + 1 == TOKEN_LIMIT:
                step_seconds.append(time.perf_counter() - step_started)
                break
            token_vector = np.asarray(embed.run(
                None, {'input_ids': np.asarray([[token_id]], dtype=np.int64)})[0], dtype=np.float32)
            past = sequence + step
            feeds = {
                'inputs_embeds': token_vector.astype(numpy_type(input_meta['inputs_embeds'].type), copy=False),
                'attention_mask': np.ones((1, past + 1), dtype=np.int64),
                'position_ids': np.asarray([[past]], dtype=np.int64),
            }
            for input_name, output_name in zip(cache_inputs, cache_outputs):
                feeds[input_name] = result[output_name]
            output_values = lm.run(output_names, feeds)
            result = dict(zip(output_names, output_values))
            step_seconds.append(time.perf_counter() - step_started)
        report['speech_tokens'] = speech_tokens
        report['token_steps'] = len(speech_tokens) + (1 if reached_eos else 0)
        report['reached_eos'] = reached_eos
        report['truncated'] = not reached_eos
        report['stage_seconds']['autoregressive_inference'] = time.perf_counter() - generation_started
        report['step_seconds'] = step_seconds
        report['rss']['after_generation'] = rss_snapshot()
        if not speech_tokens:
            report['worker_equivalent_error'] = 'No speech tokens were generated.'
        elif not reached_eos:
            report['worker_equivalent_error'] = 'Passage reached the 256-token limit.'

        del result, feeds, output_values, lm, embed, text_embeddings, combined
        gc.collect()
        if args.decode and speech_tokens and reached_eos:
            report['rss']['before_decoder'] = rss_snapshot()
            report['decoder'] = run_decoder(args.source_dir, tensors, speech_tokens, output_dir)
            report['rss']['after_decoder'] = rss_snapshot()
        elif args.decode:
            report['decoder_skipped'] = 'Worker-equivalent tokenization did not complete speech generation.'
        report['rss']['peak'] = rss_snapshot()
        report['status'] = 'complete'
        report['total_seconds'] = time.perf_counter() - started
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['total_seconds'] = time.perf_counter() - started
        raise
    finally:
        report_path = output_dir / 'cpu-reference.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(report_path),
                      'speech_tokens': report.get('speech_tokens'),
                      'reached_eos': report.get('reached_eos'),
                      'stage_seconds': report.get('stage_seconds'),
                      'rss': report.get('rss')}, indent=2))


if __name__ == '__main__':
    main()
