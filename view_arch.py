#!/cm/shared/openmind/anaconda/2.5.0/bin/python
import numpy as np
import sys
print(np.load(sys.stdin.read().rstrip()))
