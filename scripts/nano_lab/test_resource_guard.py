"""No-model regression checks for the desktop memory guard."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import bounded_job


class ResourceGuardTests(unittest.TestCase):
    def test_tiny_budgets_reserve_desktop_headroom(self):
        for cap in (640,768,1024):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as folder, \
                 patch.object(bounded_job,"ROOT",Path(folder)), \
                 patch("sys.argv",["bounded_job.py","--max-memory-mib",str(cap),"unused"]), \
                 patch.object(bounded_job,"available_mib",return_value=4096+cap-1), \
                 patch.object(bounded_job.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(SystemExit,"Not starting"):
                    bounded_job.main()
                launch.assert_not_called()
            process=Mock()
            process.poll.return_value=0
            process.wait.return_value=0
            with tempfile.TemporaryDirectory() as folder, \
                 patch.object(bounded_job,"ROOT",Path(folder)), \
                 patch("sys.argv",["bounded_job.py","--max-memory-mib",str(cap),"unused"]), \
                 patch.object(bounded_job,"available_mib",return_value=4096+cap), \
                 patch.object(bounded_job.subprocess,"Popen",return_value=process) as launch, \
                 patch.object(bounded_job.signal,"signal"):
                with self.assertRaises(SystemExit) as result:
                    bounded_job.main()
                self.assertEqual(result.exception.code,0)
                command=launch.call_args.args[0]
                self.assertIn(f"MemoryMax={cap}M",command)
                self.assertIn(f"MemoryHigh={int(cap*.8)}M",command)
                self.assertIn("MemorySwapMax=0",command)

    def test_small_job_reserves_full_cap_plus_desktop_headroom(self):
        process=Mock()
        process.poll.return_value=0
        process.wait.return_value=0
        with tempfile.TemporaryDirectory() as folder, patch.object(bounded_job,"ROOT",Path(folder)), \
             patch("sys.argv",["bounded_job.py","--small-job","unused"]), \
             patch.object(bounded_job,"available_mib",return_value=5376), \
             patch.object(bounded_job.subprocess,"Popen",return_value=process) as launch, \
             patch.object(bounded_job.signal,"signal"):
            with self.assertRaises(SystemExit) as result:
                bounded_job.main()
            self.assertEqual(result.exception.code,0)
            self.assertIn("MemoryMax=1280M",launch.call_args.args[0])
            self.assertIn("MemoryHigh=1024M",launch.call_args.args[0])

    def test_low_headroom_does_not_launch(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bounded_job, "ROOT", Path(folder)), \
             patch("sys.argv", ["bounded_job.py", "unused"]), \
             patch.object(bounded_job, "available_mib", return_value=6000), \
             patch.object(bounded_job.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(SystemExit, "less than 6 GiB"):
                bounded_job.main()
            launch.assert_not_called()

    def test_guard_stop_is_failure_even_if_systemd_returns_success(self):
        process = Mock()
        process.poll.side_effect = [None, 0]
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as folder, patch.object(bounded_job, "ROOT", Path(folder)), \
             patch("sys.argv", ["bounded_job.py", "unused"]), \
             patch.object(bounded_job, "available_mib", side_effect=[6500, 4000]), \
             patch.object(bounded_job.subprocess, "Popen", return_value=process), \
             patch.object(bounded_job.subprocess, "run") as stop, \
             patch.object(bounded_job.signal, "signal"):
            with self.assertRaises(SystemExit) as result:
                bounded_job.main()
            self.assertEqual(result.exception.code, 125)
            stop.assert_called_once_with(["systemctl", "--user", "stop", "nano-lab-model"], check=False)


if __name__ == "__main__":
    unittest.main()
