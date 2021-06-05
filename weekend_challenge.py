import pdb
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

N = 1024
FILE_NAME = "file1.csv"

file_output_dict = {
    "file1.csv": "files/3829010287.tiff\x00121206\x00",
    "file2.csv": "files/2784058210.wav\x0046124\x00",
    "file3.csv": "files/1984570931.tiff\x00193994\x00",
    "file4.csv": "files/4738210983.tiff\x00130932\x00",
    "file5.csv": "files/5091376048.tiff\x0069482\x00",
    "file6.csv": "files/6884643201.wav\x0035202\x00",
    "file7.csv": "files/7256301952.tiff\x00122790\x00",
    "file8.csv": "files/8931746329.tiff\x00174068\x00",
    "file9.csv": "files/9103751287.wav\x0058584\x00",
}


def classify_fft_symbol(fft_symbol):
    digit_one = 0 if fft_symbol.imag >= 0 else 1
    digit_two = 0 if fft_symbol.real >= 0 else 1
    return digit_one, digit_two


def classify_fft_sequence(fft_sequence):
    output = []
    for f in fft_sequence:
        new_pair = classify_fft_symbol(f)
        output.append(new_pair[0])
        output.append(new_pair[1])

    return output


# Get into blocks of 32,1024,32... and discard the 32
input_signal = pd.read_csv(FILE_NAME, header=None)
with_cyclic_chunks = np.array_split(input_signal, len(input_signal) / 1056)
message_chunks = [list(w[32:][0]) for w in with_cyclic_chunks]

# Apply fft
fft_chunks = [fft(m, N) for m in message_chunks]


# Divide by channel gain
channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
channel_spectrum = fft(channel_impulse, N)

divided_fft_chunks = [np.divide(r, channel_spectrum) for r in fft_chunks]
divided_fft_chunks = [d[1:512] for d in divided_fft_chunks]


fig = plt.figure()
plt.scatter(divided_fft_chunks[0].real, divided_fft_chunks[0].imag)
fig.savefig("output_argand_ne.png", dpi=fig.dpi)

symbol_bits_sequence = []
for d in divided_fft_chunks:
    symbol_bits_sequence.extend(classify_fft_sequence(d))

symbol_bits_string = "".join((str(s) for s in symbol_bits_sequence))

output_bytes = [
    symbol_bits_string[i : i + 8] for i in range(0, len(symbol_bits_string), 8)
]
output_bytes = bytearray([int(i, 2) for i in output_bytes])
del output_bytes[: len(file_output_dict[FILE_NAME])]

if "tiff" in file_output_dict[FILE_NAME] or True:
    with open(f"output_{FILE_NAME[:-4]}.tiff", "w+b") as f:
        f.write(output_bytes)
