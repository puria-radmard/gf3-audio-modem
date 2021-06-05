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
import bitarray

warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", MatplotlibDeprecationWarning)

recording_duration = 50
action = sys.argv[1]
file_name = sys.argv[2]

N = 2048
L = 256
T = 1
fs = 44100
c_func = lambda t: exponential_chirp(t, f0=100, f1=10000, t1=T)
total_num_metadata_bits = 200
num_metadata_reps = 5
pilot_idx = np.arange(0, N / 2 - 1, 8)
unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))

parameters = dict(
    name = f"demonstration_{file_name}",
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
    pilot_idx = pilot_idx,
    unused_bins_idx = unused_bins_idx,
    num_gap_symbols = 1,
    num_known_payload = 1,
)

# These are updated in the protocol if/else branches
protocol_parameters = {
    "pilot_tone_rotation": True,
    "pilot_tone_shifting": True,
    "discard_pilot": True,
    "repeat_estimation_blocks": True,
    "LDPC_noise_scale": 10,
    "pilot_reestimate": 0,
    "OFDM_reestimate": 0
}
parameters.update({"parameters": protocol_parameters})

prot = Protocol(**parameters)
encoder = LDPCCoding(standard="802.11n", rate="1/2", z=81, ptype="A")
decoder = LDPCDecoding(standard="802.11n", rate="1/2", z=81, ptype="A")

if action == "send":
    file_type = sys.argv[3]
    uncoded_text_bits = translate_file(file_name, file_type, num_metadata_reps)
    uncoded_text_bits += '0' * (encoder.mycode.K - len(uncoded_text_bits) % encoder.mycode.K)
    encoded_text_bits = encoder(np.array([int(b) for b in uncoded_text_bits]))
    transmitter = Transmitter("gray", N=N, L=L, protocol=prot)
    transmitter.full_pipeline(encoded_text_bits, f"group7_demonstration_audio_{file_name}")

elif sys.argv[1] == "receive":
    receiver = Receiver(N=N, L=L, constellation_name="gray", protocol = prot)
    recording_file = f"demonstation_recording_{file_name}"

    # outfile = TemporaryFile()
    # print("recording started!")
    # myrecording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=1)
    # sd.wait()
    # myrecording = np.array(myrecording)
    # # myrecording = np.mean(myrecording, 1)
    # np.save(recording_file, myrecording)  
    # print("recording done!")
    # channel_output = np.load(f"{recording_file}.npy", allow_pickle=True).reshape(-1)

    aud_file_name = sys.argv[3]
    artificial_channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
    channel = Channel(artificial_channel_impulse)
    try:
        channel_input = np.load(aud_file_name, allow_pickle=True).reshape(-1)
        channel_output = channel.transmit(channel_input, noise = 0)
    except:
        channel_input = np.load(aud_file_name, allow_pickle=True)[1].reshape(-1)
        channel_output = channel.transmit(channel_input, noise = 0)

    modu = Modulation("gray", N, L, protocol=prot)
    ground_truth_estimation_OFDM_frames = modu.data2OFDM(bitstring=prot.estimation_bits, return_frames=True, ignore_aug=True)
    ground_truth_reestimation_OFDM_frames = modu.data2OFDM(bitstring=prot.mid_message_bits, return_frames=True, ignore_aug=True)

    received_bitstring, derived_channel, phase_mismatch_trials, signal_metadata = receiver.full_pipeline(
        channel_output,
        ground_truth_estimation_OFDM_frames,
        ground_truth_reestimation_OFDM_frames,
        sample_shift=0,
        decoder=decoder,
    )
    received_bitstring = "".join(str(r) for r in received_bitstring)
    ftype = signal_metadata["filetype"]

    image_bits = bitarray.bitarray(received_bitstring)
    image_bytes = image_bits.tobytes()

    with open(f"group7_demonstration_output_{file_name}.{ftype}", 'w+b') as f:
        f.write(image_bytes)