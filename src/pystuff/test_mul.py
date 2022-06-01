#!/usr/bin/env python3

import sys
import numpy as np
import time
from pathlib import Path
import os
import signal
if True:
    build_dir = Path(__file__).parents[2].joinpath('build')
    sys.path.append(str(build_dir.joinpath('src/cudalib')))
    import gpu_library


size = 10000000
arr1 = np.linspace(1.0, 100.0, size)
arr2 = np.linspace(1.0, 100.0, size)
# PID = os.getpid()
# os.kill(PID, signal.SIGUSR1)
runs = 1000
factor = 1.0001
a = 1


t0 = time.time()
gpu_library.multiply_with_scalar(arr1, factor, runs)
print("gpu time: {}".format(time.time()-t0))
t0 = time.time()
for _ in range(runs):
    arr2 = arr2 * factor
print("cpu time: {}".format(time.time()-t0))

print("results match: {}".format(np.allclose(arr1, arr2)))
# a = 10
