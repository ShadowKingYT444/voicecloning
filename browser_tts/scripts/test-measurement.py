"""CPU-only checks for bounded transfer, reader invariants, and CLI modes."""
import base64
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('browser_measurement', Path(__file__).with_name('measure-browser.py'))
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def wav(frames=24000):
    pcm = bytes((index % 251 for index in range(frames * 2)))
    header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36+len(pcm), b'WAVE', b'fmt ', 16, 1, 1, 24000, 48000, 2, 16, b'data', len(pcm))
    return header + pcm


class ExportTests(unittest.TestCase):
    def test_hardware_assertions_do_not_force_backend_flags(self):
        with mock.patch('sys.argv', ['measure-browser.py', '--require-hardware-webgpu']):
            args = runner.parse_args()
        self.assertTrue(args.require_hardware_webgpu)
        self.assertFalse(args.hardware_webgpu)
        flags = runner.chrome_arguments(Path('/opt/chrome'), Path('/tmp/isolated-profile'), True,
                                        hardware_webgpu=args.hardware_webgpu)
        self.assertNotIn('--enable-unsafe-webgpu', flags)
        self.assertNotIn('--use-angle=vulkan', flags)

    def test_heap_scopes_and_missing_targets_stay_explicit(self):
        fake = object.__new__(runner.BrowserMeasurement)
        fake.page_session = 'page'; fake.worker_sessions = {'worker': {'type': 'worker', 'targetId': 'w'}}
        def protocol(method, params, session, timeout):
            self.assertEqual(method, 'Runtime.getHeapUsage')
            return dict(usedSize=100 if session == 'page' else 200, totalSize=1000, backingStorageSize=5000)
        fake.protocol = protocol
        report = fake.js_heap_snapshot()
        self.assertEqual(report['observedTargetUsedBytes'], 300)
        self.assertTrue(report['allObservedTargetsAvailable'])
        self.assertEqual(report['targets'][0]['backingStorageSize'], 5000)
        fake.protocol = lambda *args, **kwargs: {'totalSize': 1000}
        missing = fake.js_heap_snapshot()
        self.assertIsNone(missing['observedTargetUsedBytes'])
        self.assertFalse(missing['allObservedTargetsAvailable'])

    def fake(self, contents, bad=None):
        # Reuse the production method while replacing browser I/O with bounded chunks.
        cls = next(value for value in vars(runner).values() if isinstance(value, type) and hasattr(value, 'export_wav'))
        fake = object.__new__(cls); fake.report = {'artifacts': {}}; fake.deadline = time.monotonic()+10
        fake.stage_name = None; fake.rows = []
        for offset in range(0, len(contents), 65536):
            block = contents[offset:offset+65536]
            fake.rows.append(dict(done=False, offsetBytes=offset, byteLength=len(block), base64=base64.b64encode(block).decode()))
        fake.rows.append(dict(done=True, offsetBytes=len(contents)))
        if bad: bad(fake.rows)
        fake.start_stage = lambda name: setattr(fake, 'stage_name', name)
        fake.finish_stage = lambda name, **kwargs: setattr(fake, 'stage_name', None)
        fake.check_timeout = lambda *args: None
        fake.sample = lambda *args: None
        def evaluate(expression, **kwargs):
            if 'openWavExport' in expression: return dict(bytes=len(contents), passages=3)
            if 'cancelWavExport' in expression: return None
            return fake.rows.pop(0)
        fake.evaluate = evaluate
        return fake

    def test_long_transfer_is_bounded_and_exact(self):
        contents = wav(24000*65)  # Beyond the old 60-second whole-base64 API.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'long.wav'
            result = self.fake(contents).export_wav('export', 'wav', path)
            self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(result['maximumTransferChunkBytes'], 65536)
            self.assertEqual(result['sha256'], hashlib.sha256(contents).hexdigest())
            self.assertEqual(result['audioSeconds'], 65)

    def test_out_of_order_and_truncated_transfer_fail(self):
        for mutate in (lambda rows: rows[0].update(offsetBytes=1), lambda rows: rows.pop(-2)):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(runner.RunnerError):
                    self.fake(wav(), mutate).export_wav('export', 'wav', Path(directory)/'bad.wav')

    def test_wav_length_and_format_are_checked(self):
        contents = wav()
        for bad in (contents[:-1], contents+b'x', contents[:20]+b'\x03\x00'+contents[22:]):
            with self.assertRaises(runner.RunnerError): runner.inspect_wav(bad)

    def test_complete_input_and_queue_checks_are_not_word_audits(self):
        snap = dict(chunks=[dict(index=0, text='Some new text.', truncated=False, speechTokens=[42], audioSeconds=1)],
                    metrics=dict(maximumScheduledQueueSize=2))
        runner.assert_complete_reading(snap, 'Some new text.')
        for text in ('Some text.', 'Some new text. Some new text.'):
            with self.assertRaises(runner.RunnerError): runner.assert_complete_reading(snap, text)
        snap['metrics']['maximumScheduledQueueSize'] = 3
        with self.assertRaises(runner.RunnerError): runner.assert_complete_reading(snap, 'Some new text.')


class StopRestartTests(unittest.TestCase):
    def test_voice_state_candidate_requires_full_mode(self):
        url = '/voice/native-asmr-donor/asmr-state.json'
        with mock.patch('sys.argv', ['measure-browser.py', '--voice-state-manifest', url]):
            self.assertEqual(runner.parse_args().voice_state_manifest, url)
        for mode in ('--ui-only', '--adapter-only'):
            with mock.patch('sys.argv', ['measure-browser.py', '--voice-state-manifest', url, mode]):
                with self.subTest(mode=mode), self.assertRaises(SystemExit):
                    runner.parse_args()

    def test_probe_text_contains_at_least_four_passages_of_words(self):
        text = runner.make_stop_probe_text('Take a quiet breath.', 18)
        self.assertGreaterEqual(len(text.split()), 4 * 18)
        self.assertLessEqual(len(text.split()), 5 * 18)

    def test_probe_text_rejects_empty_input(self):
        with self.assertRaises(runner.RunnerError):
            runner.make_stop_probe_text('  ', 18)

    def test_stopped_snapshot_requires_ready_idle_queue_and_audio(self):
        valid = dict(
            readingOutcome='stopped', busy=False, modelReady=True, status='ready',
            chunks=[dict(index=0, truncated=False, speechTokens=[42], audioSeconds=1)],
            metrics=dict(scheduledQueueSize=0),
        )
        runner.assert_stopped_reading(valid)
        for field, value in (
            ('readingOutcome', 'complete'), ('busy', True), ('modelReady', False), ('status', 'stopping'),
        ):
            bad = {**valid, field: value}
            with self.subTest(field=field), self.assertRaises(runner.RunnerError):
                runner.assert_stopped_reading(bad)
        bad_metrics = {**valid, 'metrics': {'scheduledQueueSize': 1}}
        with self.assertRaises(runner.RunnerError):
            runner.assert_stopped_reading(bad_metrics)
        bad_chunks = {**valid, 'chunks': []}
        with self.assertRaises(runner.RunnerError):
            runner.assert_stopped_reading(bad_chunks)

    def test_stop_restart_flag_is_full_mode_only(self):
        with mock.patch('sys.argv', ['measure-browser.py', '--check-stop-restart']):
            self.assertTrue(runner.parse_args().check_stop_restart)
        for mode in ('--ui-only', '--adapter-only'):
            with mock.patch('sys.argv', ['measure-browser.py', '--check-stop-restart', mode]):
                with self.subTest(mode=mode), self.assertRaises(SystemExit):
                    runner.parse_args()

    def test_inference_trace_flag_is_full_mode_only(self):
        with mock.patch('sys.argv', ['measure-browser.py', '--trace-inference']):
            self.assertTrue(runner.parse_args().trace_inference)
        for mode in ('--ui-only', '--adapter-only'):
            with mock.patch('sys.argv', ['measure-browser.py', '--trace-inference', mode]):
                with self.subTest(mode=mode), self.assertRaises(SystemExit):
                    runner.parse_args()
        with mock.patch('sys.argv', ['measure-browser.py', '--trace-inference', '--full-paper']):
            with self.assertRaises(SystemExit):
                runner.parse_args()


if __name__ == '__main__': unittest.main()
