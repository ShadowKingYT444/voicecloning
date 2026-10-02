"""Model-free controls for the local reader's request and cancellation contract."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('local_reader', Path(__file__).with_name('local-reader-server.py'))
server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)


class ReaderControls(unittest.TestCase):
    def backend(self):
        root = Path(tempfile.gettempdir())
        backend = server.ReaderBackend(root, root, root, SimpleNamespace())
        backend.prepared = {}
        return backend

    def test_stop_before_request_is_latched_and_does_not_start_model(self):
        backend = self.backend()
        self.assertTrue(backend.cancel_active('early'))
        event = backend.begin_synthesis('early')
        with self.assertRaises(server.SynthesisCancelled): backend.synthesize('Hi.', 1, 'early', event)
        backend.end_synthesis('early', event)
        self.assertFalse(backend.synthesis_lock.locked())

    def test_only_one_request_runs_and_stop_does_not_cancel_another_reader(self):
        backend = self.backend()
        event = backend.begin_synthesis('one')
        with self.assertRaises(server.SynthesisBusy): backend.begin_synthesis('two')
        backend.cancel_active('two')
        self.assertFalse(event.is_set())
        backend.end_synthesis('one', event)
        pending = backend.begin_synthesis('two')
        self.assertTrue(pending.is_set())
        backend.end_synthesis('two', pending)

    def test_cancellation_history_is_bounded(self):
        backend = self.backend()
        for index in range(100): backend.cancel_active(str(index))
        self.assertEqual(len(backend.cancelled_requests), 64)

    def test_requests_reject_invalid_text_seed_and_browser_model_overrides(self):
        handler = object.__new__(server.LocalReaderHandler)
        for text, seed in [('', 1), ('a ' * 25, 1), ('a', True), ('a', -1), ('a', 2**32)]:
            with self.subTest(text=text, seed=seed), self.assertRaises(ValueError):
                handler._synthesis_input({'text': text, 'seed': seed, 'requestId': 'x'})
        for option in server.FORBIDDEN_CPU_OPTIONS:
            with self.subTest(option=option), self.assertRaises(ValueError): server._check_cpu_options({option: 'override'})
        self.assertEqual(handler._synthesis_input({'text': 'Hi.', 'seed': 1337, 'requestId': 'x'}), ('Hi.', 1337, 'x'))

    def test_service_binds_only_to_loopback(self):
        self.assertEqual(server.validate_loopback_bind_host('localhost'), '127.0.0.1')
        for host in ('0.0.0.0', 'example.com', '192.168.1.2'):
            with self.subTest(host=host), self.assertRaises(ValueError): server.validate_loopback_bind_host(host)


if __name__ == '__main__': unittest.main()
