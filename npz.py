import sys
import numpy as np
import sounddevice as sd
import sys

arr = np.load(f"{sys.argv[1]}.npy", allow_pickle = True)
import pdb; pdb.set_trace()
sd.play(arr, 44100)
sd.wait()
