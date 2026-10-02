"""Live kernel checks; run inside bounded_job's 640 MiB single-process backend."""
import errno
import os
import resource
import subprocess
import threading
import unittest


class LiveGuardTests(unittest.TestCase):
    def test_kernel_address_space_cap_and_cpu_affinity(self):
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), (640*2**20,640*2**20))
        self.assertLessEqual(len(os.sched_getaffinity(0)),2)
        self.assertGreaterEqual(os.getpriority(os.PRIO_PROCESS,0),10)
        with self.assertRaises(MemoryError):
            bytearray(800*2**20)

    def test_child_processes_are_denied(self):
        with self.assertRaises(OSError) as failure:
            os.fork()
        self.assertEqual(failure.exception.errno,errno.EPERM)
        with self.assertRaises(OSError):
            subprocess.run(['/bin/true'],check=True)

    def test_threads_remain_allowed(self):
        values=[]
        threads=[threading.Thread(target=lambda:values.append(1)) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(values),4)

    def test_workload_cannot_raise_limits_or_expand_affinity(self):
        with self.assertRaises((OSError,ValueError)):
            resource.setrlimit(resource.RLIMIT_AS,(900*2**20,900*2**20))
        with self.assertRaises(OSError) as failure:
            os.sched_setaffinity(0,{0,1,2})
        self.assertEqual(failure.exception.errno,errno.EPERM)


if __name__=='__main__':unittest.main()
