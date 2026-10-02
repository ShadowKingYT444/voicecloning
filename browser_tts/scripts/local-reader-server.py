#!/usr/bin/env python3
"""Serve the built reader and its pinned CPU inference API on localhost.

Run this process through scripts/nano_lab/bounded_job.py. Model sessions are
created for one passage, then released before the decoder session is created.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import importlib.util
import ipaddress
import json
import mimetypes
from pathlib import Path
import signal
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


SCHEMA = 'voice-study.local-reader/v1'
STATUS_PATH = '/api/reader/status'
PREPARE_PATH = '/api/reader/prepare'
SYNTHESIZE_PATH = '/api/reader/synthesize'
STOP_PATH = '/api/reader/stop'
MAX_BODY_BYTES = 16 * 1024
MAX_TEXT_CHARS = 2048
MAX_TEXT_WORDS = 24
MAX_REQUEST_ID_CHARS = 128
SAMPLE_RATE = 24000
FORBIDDEN_CPU_OPTIONS = {
    'powerPreference', 'provider', 'gpu', 'modelBaseUrl', 'voiceStateManifestUrl',
    'embeddingManifestUrl', 'losslessEmbeddingManifestUrl', 'traceInference',
}


def load_probe_module(root: Path):
    """Load the existing verified sampler and pinned asset helpers."""
    probe_path = root / 'browser_tts/scripts/probe-cpu-pipeline.py'
    spec = importlib.util.spec_from_file_location('probe_cpu_pipeline', probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load CPU pipeline helpers from {probe_path}.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_loopback_bind_host(host: str) -> str:
    """Return an IPv4 loopback bind address; never expose this service publicly."""
    if host == 'localhost':
        return '127.0.0.1'
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError('--host must be 127.0.0.1, another IPv4 loopback, or localhost.') from error
    if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
        raise ValueError('--host must be an IPv4 loopback address.')
    return str(address)


def _authority(value: str, origin: bool = False):
    """Parse a Host or Origin authority and normalize its effective port."""
    try:
        parsed = urlsplit(value if origin else f'//{value}')
        if origin:
            if parsed.scheme.lower() != 'http' or parsed.path or parsed.query or parsed.fragment:
                return None
        elif parsed.path or parsed.query or parsed.fragment:
            return None
        if parsed.username is not None or parsed.password is not None or not parsed.hostname:
            return None
        hostname = parsed.hostname.lower()
        if hostname != 'localhost':
            address = ipaddress.ip_address(hostname)
            if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
                return None
        port = parsed.port or (80 if origin else None)
        return hostname, port
    except (ValueError, UnicodeError):
        return None


def _check_cpu_options(payload):
    for name in FORBIDDEN_CPU_OPTIONS:
        if payload.get(name) is not None:
            raise ValueError(f'{name} cannot be used by the local CPU reader.')
    backend = payload.get('backend')
    if backend is not None and backend not in ('CPU', 'Local CPU'):
        raise ValueError('This service supports the CPU backend only.')


class ReaderBackend:
    """Own verified lightweight voice data and one transient synthesis at a time."""

    def __init__(self, root: Path, source_dir: Path, voice_dir: Path, probe):
        self.root = root.resolve()
        self.source_dir = source_dir.resolve()
        self.voice_dir = voice_dir.resolve()
        self.probe = probe
        self.prepare_lock = threading.Lock()
        self.synthesis_lock = threading.Lock()
        self.active_lock = threading.Lock()
        self.active = None
        self.cancelled_requests = OrderedDict()
        self.prepared = None

    def _check_runtime_versions(self):
        if self.probe.ort.__version__ != '1.29.0' or self.probe.np.__version__ != '1.26.4':
            raise RuntimeError(
                'The local CPU reader requires ONNX Runtime 1.29.0 and NumPy 1.26.4; '
                f'found {self.probe.ort.__version__} and {self.probe.np.__version__}.'
            )

    def prepare(self):
        started = time.perf_counter()
        with self.prepare_lock:
            if self.prepared is not None:
                return {
                    'backend': 'Local CPU', 'ready': True, 'loadSeconds': 0.0,
                    'modelIdentity': self.prepared['model_identity'],
                }
            self._check_runtime_versions()
            model_hashes = self.probe.verify_model_assets(self.source_dir)
            tokenizer_path = self.source_dir / 'tokenizer.json'
            tokenizer = self.probe.Tokenizer.from_file(str(tokenizer_path))
            voice_manifest, tensors = self.probe.read_voice_state(
                self.voice_dir / 'asmr-state.json', self.voice_dir / 'asmr-state.bin',
                model_hashes, 'native_t3_donor')

            # The donor keeps the three public decoder tensors. Require exact
            # byte equality with the independently pinned public state.
            public_manifest, public_tensors = self.probe.read_voice_state(
                self.root / 'browser_tts/public/voice/asmr-state.json',
                self.root / 'browser_tts/public/voice/asmr-state.bin',
                model_hashes, 'generated_reference')
            for name in ('audio_tokens', 'speaker_embeddings', 'speaker_features'):
                if tensors[name].tobytes(order='C') != public_tensors[name].tobytes(order='C'):
                    raise ValueError(f'Native donor changed retained public decoder tensor {name}.')
            del public_manifest, public_tensors

            model_identity = {
                'repoId': self.probe.REPOSITORY,
                'revision': self.probe.REVISION,
                'sha256sumsSha256': self.probe.SHA256SUMS_SHA256,
                'provider': 'CPUExecutionProvider',
                'onnxruntimeVersion': self.probe.ort.__version__,
                'numpyVersion': self.probe.np.__version__,
                'tokenizerSha256': model_hashes['tokenizer.json'],
                'graphs': {
                    name: {
                        'sha256': model_hashes[f'onnx/{name}.onnx'],
                        'externalDataSha256': model_hashes[f'onnx/{name}.onnx_data'],
                    }
                    for name in self.probe.GRAPH_NAMES
                },
                'voiceState': {
                    'path': str(self.voice_dir.relative_to(self.root))
                    if self.voice_dir.is_relative_to(self.root) else str(self.voice_dir),
                    'sha256': voice_manifest['data']['sha256'],
                    'source': voice_manifest['conditioning']['source'],
                    'speakerSpecific': True,
                    'zeroShot': False,
                    'speechStartCount': 1,
                    'retainedDecoderTensorsMatchPublicState': True,
                    'fittedAdapterApplied': False,
                },
            }
            self.prepared = {
                'tokenizer': tokenizer,
                'tensors': tensors,
                'model_hashes': model_hashes,
                'model_identity': model_identity,
            }
        return {
            'backend': 'Local CPU', 'ready': True,
            'loadSeconds': time.perf_counter() - started,
            'modelIdentity': self.prepared['model_identity'],
        }

    def status(self):
        return {
            'schema': SCHEMA,
            'backend': 'CPU',
            'provider': 'CPUExecutionProvider',
            'ready': self.prepared is not None,
            'prepared': self.prepared is not None,
            'busy': self.synthesis_lock.locked(),
            'modelIdentity': self.prepared['model_identity'] if self.prepared else None,
            'rss': self.probe.rss_snapshot(),
            'rssScope': 'This server process, including loaded model sessions and native runtime memory.',
        }

    def begin_synthesis(self, request_id: str):
        if self.prepared is None:
            raise RuntimeError('Prepare the local CPU voice before reading.')
        if not self.synthesis_lock.acquire(blocking=False):
            raise SynthesisBusy('A passage is already being synthesized.')
        cancel = threading.Event()
        with self.active_lock:
            if request_id in self.cancelled_requests:
                del self.cancelled_requests[request_id]
                cancel.set()
            self.active = (request_id, cancel)
        return cancel

    def end_synthesis(self, request_id: str, cancel):
        with self.active_lock:
            if self.active == (request_id, cancel):
                self.active = None
        self.synthesis_lock.release()
        gc.collect()

    def cancel_active(self, request_id: str | None = None) -> bool:
        with self.active_lock:
            if self.active is None:
                if request_id is None:
                    return False
                self._remember_cancelled(request_id)
                return True
            active_id, cancel = self.active
            if request_id is not None and request_id != active_id:
                self._remember_cancelled(request_id)
                return True
            cancel.set()
            return True

    def _remember_cancelled(self, request_id: str):
        self.cancelled_requests[request_id] = None
        self.cancelled_requests.move_to_end(request_id)
        while len(self.cancelled_requests) > 64:
            self.cancelled_requests.popitem(last=False)

    def synthesize(self, text: str, seed: int, request_id: str, cancel):
        prepared = self.prepared
        if prepared is None:
            raise RuntimeError('Prepare the local CPU voice before reading.')
        if cancel.is_set():
            raise SynthesisCancelled('The passage was stopped before synthesis started.')
        started = time.perf_counter()
        speech_tokens, next_seed, stage_seconds = self._generate_speech_tokens(
            prepared, text, seed, cancel)
        gc.collect()
        if cancel.is_set():
            raise SynthesisCancelled('The passage was stopped before decoding.')
        audio, decoder_load_seconds, decoder_seconds = self._decode(prepared, speech_tokens, cancel)
        stage_seconds['decoderSessionLoad'] = decoder_load_seconds
        stage_seconds['decoder'] = decoder_seconds
        gc.collect()
        if cancel.is_set():
            raise SynthesisCancelled('The passage was stopped during decoding.')
        audio = self.probe.np.asarray(audio, dtype=self.probe.np.dtype('<f4')).reshape(-1)
        if not audio.size or not self.probe.np.isfinite(audio).all():
            raise ValueError('The CPU decoder returned empty or non-finite audio.')
        return audio.tobytes(order='C'), {
            'requestId': request_id,
            'sampleRate': SAMPLE_RATE,
            'synthesisSeconds': time.perf_counter() - started,
            'stageSeconds': stage_seconds,
            'speechTokens': speech_tokens,
            'truncated': False,
            'nextSeed': next_seed,
            'modelIdentity': prepared['model_identity'],
        }

    def _generate_speech_tokens(self, prepared, text: str, seed: int, cancel):
        p = self.probe
        stage_seconds = {}
        embed = lm = result = feeds = output_values = None
        embed_result = text_embeddings = combined = logits = None
        token_vector_result = token_vector = next_result = None
        try:
            embed_started = time.perf_counter()
            embed = p.create_session(self.source_dir, 'embed_tokens_fp16')
            stage_seconds['embeddingSessionLoad'] = time.perf_counter() - embed_started
            lm_started = time.perf_counter()
            lm = p.create_session(self.source_dir, 'language_model_q4f16')
            stage_seconds['languageModelSessionLoad'] = time.perf_counter() - lm_started

            conditioning_started = time.perf_counter()
            ids = prepared['tokenizer'].encode(text, add_special_tokens=True).ids
            if len(ids) < 3 or ids[-2:] != [50256, 50256]:
                raise ValueError('The pinned tokenizer did not append the two speech-start markers.')
            tensors = prepared['tensors']
            embed_result = embed.run(None, {'input_ids': p.np.asarray([ids], dtype=p.np.int64)})
            text_embeddings = p.np.asarray(embed_result[0], dtype=p.np.float32)
            embed_result = None
            if (text_embeddings.shape != (1, len(ids), 768)
                    or not p.np.array_equal(text_embeddings[:, -1, :], text_embeddings[:, -2, :])):
                raise ValueError('The native donor requires identical embeddings for the final speech-start markers.')
            text_embeddings = text_embeddings[:, :-1, :]
            combined = p.np.concatenate((tensors['audio_features'], text_embeddings), axis=1)
            sequence = combined.shape[1]
            input_meta = {item.name: item for item in lm.get_inputs()}
            output_names = [item.name for item in lm.get_outputs()]
            output_set = set(output_names)
            cache_inputs = [item.name for item in lm.get_inputs() if item.name.startswith('past_key_values.')]
            cache_outputs = [name.replace('past_key_values.', 'present.', 1) for name in cache_inputs]
            if (len(cache_inputs) != 24 or any(name not in output_set for name in cache_outputs)
                    or 'inputs_embeds' not in input_meta or 'attention_mask' not in input_meta
                    or 'position_ids' not in input_meta or 'logits' not in output_set):
                raise ValueError('The language model KV cache contract differs from the pinned browser worker.')
            feeds = {
                'inputs_embeds': combined.astype(p.numpy_type(input_meta['inputs_embeds'].type), copy=False),
                'attention_mask': p.np.ones((1, sequence), dtype=p.np.int64),
                'position_ids': p.np.arange(sequence, dtype=p.np.int64).reshape(1, sequence),
            }
            for name in cache_inputs:
                feeds[name] = p.np.empty((1, 12, 0, 64), dtype=p.numpy_type(input_meta[name].type))
            stage_seconds['conditioning'] = time.perf_counter() - conditioning_started

            if cancel.is_set():
                raise SynthesisCancelled('The passage was stopped before language-model prefill.')
            prefill_started = time.perf_counter()
            output_values = lm.run(output_names, feeds)
            result = dict(zip(output_names, output_values))
            output_values = feeds = combined = text_embeddings = None
            stage_seconds['prefill'] = time.perf_counter() - prefill_started
            if cancel.is_set():
                raise SynthesisCancelled('The passage was stopped during language-model prefill.')

            sampler = p.WorkerSampler(seed)
            history = [p.START_SPEECH]
            speech_tokens = []
            reached_eos = False
            autoregressive_started = time.perf_counter()
            for step in range(p.TOKEN_LIMIT):
                if cancel.is_set():
                    raise SynthesisCancelled('The passage was stopped during language-model inference.')
                logits = p.np.asarray(result['logits'], dtype=p.np.float32).reshape(-1, p.VOCAB_SIZE)[-1]
                token_id = sampler.sample(logits, history)
                if token_id == p.STOP_SPEECH:
                    reached_eos = True
                    break
                speech_tokens.append(token_id)
                history.append(token_id)
                if step + 1 == p.TOKEN_LIMIT:
                    break
                if cancel.is_set():
                    raise SynthesisCancelled('The passage was stopped between language-model calls.')
                token_vector_result = embed.run(
                    None, {'input_ids': p.np.asarray([[token_id]], dtype=p.np.int64)})
                token_vector = p.np.asarray(token_vector_result[0], dtype=p.np.float32)
                token_vector_result = None
                past = sequence + step
                feeds = {
                    'inputs_embeds': token_vector.astype(
                        p.numpy_type(input_meta['inputs_embeds'].type), copy=False),
                    'attention_mask': p.np.ones((1, past + 1), dtype=p.np.int64),
                    'position_ids': p.np.asarray([[past]], dtype=p.np.int64),
                }
                for input_name, output_name in zip(cache_inputs, cache_outputs):
                    feeds[input_name] = result[output_name]
                if cancel.is_set():
                    raise SynthesisCancelled('The passage was stopped before the next language-model call.')
                output_values = lm.run(output_names, feeds)
                next_result = dict(zip(output_names, output_values))
                output_values = feeds = token_vector = None
                result = next_result
                next_result = None
                if cancel.is_set():
                    raise SynthesisCancelled('The passage was stopped after a language-model call.')
            stage_seconds['autoregressive'] = time.perf_counter() - autoregressive_started
            if cancel.is_set():
                raise SynthesisCancelled('The passage was stopped before decoder loading.')
            if not reached_eos:
                raise ValueError('Passage reached the 256-token limit. Use shorter passages.')
            if not speech_tokens:
                raise ValueError('The language model generated no speech tokens.')
            next_seed = sampler.state
            result = None
            return speech_tokens, next_seed, stage_seconds
        finally:
            embed = lm = result = feeds = output_values = None
            embed_result = text_embeddings = combined = logits = None
            token_vector_result = token_vector = next_result = None
            gc.collect()

    def _decode(self, prepared, speech_tokens, cancel):
        if cancel.is_set():
            raise SynthesisCancelled('The passage was stopped before decoder loading.')
        p = self.probe
        started = time.perf_counter()
        decoder = output_values = feeds = decoder_ids = audio = None
        try:
            decoder = p.create_session(self.source_dir, 'conditional_decoder_q4')
            load_seconds = time.perf_counter() - started
            if cancel.is_set():
                raise SynthesisCancelled('The passage was stopped before decoder inference.')
            tensors = prepared['tensors']
            input_types = {item.name: p.numpy_type(item.type) for item in decoder.get_inputs()}
            decoder_ids = p.np.concatenate((
                tensors['audio_tokens'],
                p.np.asarray([speech_tokens], dtype=p.np.int64),
                p.np.full((1, 3), p.SILENCE, dtype=p.np.int64),
            ), axis=1)
            feeds = {
                'speech_tokens': decoder_ids.astype(input_types['speech_tokens'], copy=False),
                'speaker_embeddings': tensors['speaker_embeddings'].astype(
                    input_types['speaker_embeddings'], copy=False),
                'speaker_features': tensors['speaker_features'].astype(
                    input_types['speaker_features'], copy=False),
            }
            output_name = decoder.get_outputs()[0].name
            started = time.perf_counter()
            output_values = decoder.run([output_name], feeds)
            audio = p.np.asarray(output_values[0], dtype=p.np.float32).reshape(-1)
            inference_seconds = time.perf_counter() - started
            if cancel.is_set():
                raise SynthesisCancelled('The passage was stopped during decoder inference.')
            return audio, load_seconds, inference_seconds
        finally:
            decoder = output_values = feeds = decoder_ids = audio = None
            gc.collect()


class SynthesisBusy(Exception):
    pass


class SynthesisCancelled(Exception):
    pass


class LocalReaderHandler(SimpleHTTPRequestHandler):
    """Serve only dist files and the same-origin reader API."""

    protocol_version = 'HTTP/1.1'
    server_version = 'LocalReader/1'
    sys_version = ''

    def __init__(self, *args, backend: ReaderBackend, dist_dir: Path, **kwargs):
        self.backend = backend
        self.dist_dir = dist_dir
        super().__init__(*args, directory=str(dist_dir), **kwargs)

    def log_message(self, fmt, *args):
        # Do not log request bodies or user text.
        sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))

    def handle_expect_100(self):
        return self._send_error_json(417, 'expectation_failed', '100-continue is not supported.')

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def _send_bytes(self, status: int, body: bytes, content_type: str, headers=None):
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.close_connection = True
            if self.command != 'HEAD':
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            self.close_connection = True

    def _send_json(self, status: int, value):
        body = json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        self._send_bytes(status, body, 'application/json; charset=utf-8')

    def _send_error_json(self, status: int, code: str, message: str):
        self._send_json(status, {'error': message, 'code': code})
        return False

    def send_error(self, code, message=None, explain=None):
        status_message = message or 'Request failed.'
        self._send_error_json(code, 'http_error', status_message)

    def _valid_local_origin(self):
        host = _authority(self.headers.get('Host', ''))
        if host is None or host[1] != self.server.server_address[1]:
            return self._send_error_json(403, 'invalid_host', 'Use this reader through its loopback host and port.')
        origin_value = self.headers.get('Origin')
        if origin_value:
            origin = _authority(origin_value, origin=True)
            if origin is None or origin != host:
                return self._send_error_json(403, 'invalid_origin', 'The reader API accepts same-origin requests only.')
        fetch_site = self.headers.get('Sec-Fetch-Site')
        if fetch_site and fetch_site.lower() not in ('same-origin', 'none'):
            return self._send_error_json(403, 'invalid_origin', 'The reader API accepts same-origin requests only.')
        return True

    def _read_json(self):
        if self.headers.get('Transfer-Encoding'):
            self._send_error_json(400, 'invalid_body', 'Chunked request bodies are not supported.')
            return None
        if self.headers.get_content_type() != 'application/json':
            self._send_error_json(415, 'content_type', 'Send application/json.')
            return None
        try:
            length = int(self.headers.get('Content-Length', ''))
        except (TypeError, ValueError):
            self._send_error_json(411, 'content_length', 'A Content-Length header is required.')
            return None
        if length < 0:
            self._send_error_json(400, 'invalid_body', 'The request body length is invalid.')
            return None
        if length > MAX_BODY_BYTES:
            self._send_error_json(413, 'body_too_large', 'The request body is too large.')
            return None
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                self._send_error_json(400, 'invalid_body', 'The request body ended early.')
                return None
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error_json(400, 'invalid_json', 'The request body must contain valid JSON.')
            return None
        if not isinstance(value, dict):
            self._send_error_json(400, 'invalid_body', 'The JSON body must be an object.')
            return None
        return value

    def _api_not_found(self):
        self._send_error_json(404, 'not_found', 'Reader API route not found.')

    def do_GET(self):
        if not self._valid_local_origin():
            return
        path = urlsplit(self.path).path
        if path == STATUS_PATH:
            self._send_json(200, self.backend.status())
            return
        if path.startswith('/api/'):
            self._api_not_found()
            return
        self._serve_dist_file()

    def do_HEAD(self):
        if not self._valid_local_origin():
            return
        path = urlsplit(self.path).path
        if path.startswith('/api/'):
            self._send_error_json(405, 'method_not_allowed', 'Use GET or POST for this reader API route.')
            return
        self._serve_dist_file()

    def do_POST(self):
        if not self._valid_local_origin():
            return
        path = urlsplit(self.path).path
        if path not in (PREPARE_PATH, SYNTHESIZE_PATH, STOP_PATH):
            self._api_not_found()
            return
        payload = self._read_json()
        if payload is None:
            return
        try:
            _check_cpu_options(payload)
            if path == PREPARE_PATH:
                result = self.backend.prepare()
                self._send_json(200, result)
            elif path == STOP_PATH:
                request_id = self._request_id(payload.get('requestId'))
                stopped = self.backend.cancel_active(request_id)
                self._send_json(200, {'requestId': request_id, 'stopped': stopped})
            else:
                text, seed, request_id = self._synthesis_input(payload)
                cancel = self.backend.begin_synthesis(request_id)
                try:
                    audio, metadata = self.backend.synthesize(text, seed, request_id, cancel)
                finally:
                    self.backend.end_synthesis(request_id, cancel)
                header = json.dumps(metadata, separators=(',', ':'), ensure_ascii=True)
                self._send_bytes(200, audio, 'application/octet-stream', {
                    'X-Reader-Metadata': header,
                    'X-Reader-Request-Id': request_id,
                })
        except SynthesisBusy as error:
            self._send_error_json(409, 'busy', str(error))
        except SynthesisCancelled as error:
            self._send_error_json(409, 'cancelled', str(error))
        except ValueError as error:
            self._send_error_json(400, 'invalid_request', str(error))
        except RuntimeError as error:
            message = str(error)
            status = 409 if message.startswith('Prepare the local CPU voice') else 500
            self._send_error_json(status, 'not_ready' if status == 409 else 'runtime_error', message)
        except Exception as error:
            self._send_error_json(500, 'inference_failed', f'{type(error).__name__}: {error}')

    def do_PUT(self):
        self._send_error_json(405, 'method_not_allowed', 'This server does not accept PUT requests.')

    def do_DELETE(self):
        self._send_error_json(405, 'method_not_allowed', 'This server does not accept DELETE requests.')

    def do_OPTIONS(self):
        self._send_error_json(405, 'method_not_allowed', 'Cross-origin preflight requests are not supported.')

    def _request_id(self, value):
        if (not isinstance(value, str) or not value or len(value) > MAX_REQUEST_ID_CHARS
                or any(ord(character) < 32 for character in value)):
            raise ValueError(f'requestId must be a non-empty string of at most {MAX_REQUEST_ID_CHARS} characters.')
        return value

    def _synthesis_input(self, payload):
        text = payload.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a non-empty string.')
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f'text cannot exceed {MAX_TEXT_CHARS} characters.')
        if len(text.split()) > MAX_TEXT_WORDS:
            raise ValueError(f'text cannot exceed {MAX_TEXT_WORDS} words per passage.')
        seed = payload.get('seed')
        if type(seed) is not int or not 0 <= seed <= 0xffffffff:
            raise ValueError('seed must be an unsigned 32-bit integer.')
        return text, seed, self._request_id(payload.get('requestId'))

    def _serve_dist_file(self):
        raw_path = urlsplit(self.path).path
        try:
            path = unquote(raw_path, errors='strict')
            if '\x00' in path or '\\' in path:
                raise ValueError('invalid path')
            parts = [part for part in path.split('/') if part]
            if any(part in ('.', '..') for part in parts):
                raise ValueError('invalid path')
            candidate = (self.dist_dir / ('/'.join(parts) or 'index.html')).resolve(strict=True)
            candidate.relative_to(self.dist_dir)
            if candidate.is_dir():
                index = (candidate / 'index.html').resolve(strict=True)
                index.relative_to(self.dist_dir)
                candidate = index
            if not candidate.is_file():
                raise FileNotFoundError
            content_length = candidate.stat().st_size
        except (ValueError, OSError, RuntimeError):
            self._send_error_json(404, 'not_found', 'File not found.')
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(content_length))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            if self.command != 'HEAD':
                with candidate.open('rb') as source:
                    while block := source.read(1024 * 1024):
                        self.wfile.write(block)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            self.close_connection = True


class LocalReaderServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = False

    def __init__(self, address, backend, dist_dir):
        self.backend = backend
        self.dist_dir = dist_dir
        super().__init__(address, LocalReaderHandler)

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(
            request, client_address, self, backend=self.backend, dist_dir=self.dist_dir)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1', help='IPv4 loopback only (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=4187, help='Listen on this exact port (default: 4187).')
    parser.add_argument('--source-dir', type=Path, default=root / 'models/chatterbox-nano-browser')
    parser.add_argument('--dist-dir', type=Path, default=root / 'browser_tts/dist')
    parser.add_argument('--voice-state-dir', type=Path,
                        default=root / 'browser_tts/public/voice/native-asmr-donor')
    args = parser.parse_args(argv)
    try:
        host = validate_loopback_bind_host(args.host)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535.')
    dist_dir = args.dist_dir.resolve()
    if not dist_dir.is_dir():
        parser.error(f'Built distribution directory does not exist: {dist_dir}')
    probe = load_probe_module(root)
    backend = ReaderBackend(root, args.source_dir, args.voice_state_dir, probe)
    server = LocalReaderServer((host, args.port), backend, dist_dir)
    shutdown_started = threading.Event()

    def request_shutdown(_signum=None, _frame=None):
        backend.cancel_active()
        if not shutdown_started.is_set():
            shutdown_started.set()
            threading.Thread(target=server.shutdown, name='local-reader-shutdown', daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(f'Local CPU reader serving {dist_dir} at http://{host}:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        backend.cancel_active()
        server.server_close()
        print('Local CPU reader stopped.', flush=True)


if __name__ == '__main__':
    main()
