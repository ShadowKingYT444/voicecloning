"""CPU-only checks for bounded transfer and complete-reading validation."""
import base64
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location('browser_measurement', Path(__file__).with_name('measure-browser.py'))
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def wav(frames=24000):
    pcm = bytes((index % 251 for index in range(frames * 2)))
    header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36+len(pcm), b'WAVE', b'fmt ', 16, 1, 1, 24000, 48000, 2, 16, b'data', len(pcm))
    return header + pcm


class ExportTests(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
