from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import fft, ifft
from tempfile import TemporaryFile
import sys
from scipy.io.wavfile import write

outfile = TemporaryFile()
fs = 44100
print("starting")
duration = 1.5  # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("stopping")
myrecording = np.array(myrecording)
myrecording = (myrecording*32767).astype('int16')
#np.save(sys.argv[1], myrecording)
write(sys.argv[1], 44100, myrecording)