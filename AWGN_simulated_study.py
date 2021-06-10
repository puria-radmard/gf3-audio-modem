from scipy.signal import resample
import bitarray
from os import error
from numpy.testing._private.utils import print_assert_equal
from scipy import signal
from util_classes import (
    Modulation,
    Transmitter,
    Receiver,
    Protocol,
    NoCoding,
    NoDecoding,
    LDPCCoding,
    LDPCDecoding,
    Channel
)
import sys
from util_objects import *
import sounddevice as sd
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from visualisation_scripts import *
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import warnings

warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", MatplotlibDeprecationWarning)

#### 
# python experiment_interface.py action reader_idx protocol mode <optional numerical parameter 1> <optional numerical parameter 2> <optional numerical parameter 3>
# actions:
    # send --> write audio to .npy
    # receive --> process recording or simulated channel output
# reader_idx:
    # filename suffix of audio, e.g. mario
# protocol:
    # protocols tested in the report
    # current list: basic, rotation, shifting, shifting_only, bandwidth_standard, LDPC, pilot_reestimation, OFDM_reestimation
# mode:
    # custome name for differentiation
####

N = 2048
L = 256
T = 1
fs = 44100
c_func = lambda t: exponential_chirp(t, f0=100, f1=10000, t1=T)
total_num_metadata_bits = 200
num_metadata_reps = 5
uncoded_text_bits = translate_tiff("mario", "tif", num_metadata_reps)

protocol_parameters = {
    "pilot_reestimate": 0,
    "OFDM_reestimate": 0,
    "LDPC_noise_scale": 1,
    "pilot_tone_rotation": True,
    "pilot_tone_shifting": True,
    "discard_pilot": True,
    "repeat_estimation_blocks": False
}

parameters = dict(
    name = f"AWGN_test",
    N = N,
    L = L,
    chirp_length = 1,
    pilot_symbol=1 + 1j,
    total_num_metadata_bits = total_num_metadata_bits,
    num_metadata_reps = num_metadata_reps,
    chirp_func = np.vectorize(c_func),
    constellation_name = "gray",
    num_estimation_symbols = 10,
    num_message_payload = 10,
    parameters = protocol_parameters,
    pilot_idx = np.arange(0, N / 2 - 1, 8),
    unused_bins_idx = np.array([]),
    num_gap_symbols = 0,
    num_known_payload = 0,
)

encoder = NoCoding()
decoder = NoDecoding()
prot = Protocol(**parameters)

audio_file = "AWGN_simulated_experiment"
uncoded_text_bits += '0' * (encoder.mycode.K - len(uncoded_text_bits) % encoder.mycode.K)
encoded_text_bits = encoder(np.array([int(b) for b in uncoded_text_bits]))
transmitter = Transmitter("gray", N=N, L=L, protocol=prot)
transmitter.full_pipeline(encoded_text_bits, audio_file)

receiver = Receiver(N=N, L=L, constellation_name="gray", protocol = prot)
ground_truth_estimation_OFDM_frames = transmitter.data2OFDM(bitstring=prot.estimation_bits, return_frames=True, ignore_aug=True)
ground_truth_reestimation_OFDM_frames = transmitter.data2OFDM(bitstring=prot.mid_message_bits, return_frames=True, ignore_aug=True)

artificial_channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
channel = Channel(artificial_channel_impulse)
channel_input = np.load(f"{audio_file}.npy", allow_pickle=True).reshape(-1)
channel_output = channel.transmit(channel_input, noise = 0.001)

percentages = {0: None, 0.003: None, -0.003: None}
fig, axs = plt.subplots(1)

for percentage_off in percentages.keys():
    chirp_audio = channel_output[:T*fs].copy()
    signal_audio = channel_output[T*fs:].copy()
    len_signal = len(signal_audio)

    percent_relative = percentage_off/100 + 1
    new_length = int(percent_relative*len_signal)
    wrong_channel_output = resample(signal_audio, new_length)
    wrong_channel_output = np.concatenate([chirp_audio, wrong_channel_output])

    slope_history = receiver.full_pipeline(
        wrong_channel_output,
        ground_truth_estimation_OFDM_frames,
        ground_truth_reestimation_OFDM_frames,
        sample_shift=0,
        decoder=decoder,
        return_graph = True
    )

    axs.plot(np.array(slope_history), label = percentage_off)

plt.gcf().subplots_adjust(left=0.15)
axs.legend()
axs.set_xlim(-5, 300)
axs.set_title("$\hat\\tau$ history over AWGN")
axs.set_xlabel("OFDM symbol number")
axs.set_ylabel("$\hat\\tau$")
for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
        axs.get_xticklabels() + axs.get_yticklabels()):
    item.set_fontsize(15)
fig.savefig("test_AWGN.png")