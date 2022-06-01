import numpy as np
import sys
from pathlib import Path
if True:
    sys.path.append(
        str(Path(__file__).parents[1].joinpath('build/src/crclib')))
    import gpu_library
