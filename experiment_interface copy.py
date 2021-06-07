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
    # current list: basic, rotation, shifting, shifting_only, bandwidth_standard, LDPC, pilot_reestimation, OFDM_reestimation, standard
# mode:
    # custome name for differentiation
####

recording_duration = 30
action = sys.argv[1]
reader_idx = sys.argv[2]
protocol = sys.argv[3]
mode = sys.argv[4]

N = 2048
L = 256
T = 1
fs = 44100
c_func = lambda t: exponential_chirp(t, f0=100, f1=10000, t1=T)
total_num_metadata_bits = 200
num_metadata_reps = 5
uncoded_text_bits = translate_tiff("mario", num_metadata_reps)

parameters = dict(
    name = f"{protocol}_{mode}",
    N = N,
    L = L,
    chirp_length = 1,
    pilot_symbol=1 + 1j,
    total_num_metadata_bits = total_num_metadata_bits,
    num_metadata_reps = num_metadata_reps,
    chirp_func = np.vectorize(c_func),
    constellation_name = "gray",
    num_estimation_symbols = 10,
    num_message_payload = 10
)

# These are updated in the protocol if/else branches
protocol_parameters = {
    "pilot_reestimate": 0,
    "OFDM_reestimate": 0,
    "LDPC_noise_scale": 1,
    "pilot_tone_rotation": False,
    "pilot_tone_shifting": False,
    "discard_pilot": False,
    "repeat_estimation_blocks": False
}

encoder = NoCoding()
decoder = NoDecoding()

if protocol == "basic":
    # COMPLETELY BASIC
    pilot_idx = np.array([])
    unused_bins_idx = np.array([])
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)

elif protocol == "rotation":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array([])
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)

elif protocol == "shifting":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array([])
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)

elif protocol == "shifting_only":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array([])
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = False
    protocol_parameters["pilot_tone_shifting"] = True
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)

elif protocol == "bandwidth_standard":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
    unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    protocol_parameters["discard_pilot"] = True
    protocol_parameters["LDPC_noise_scale"] = float(sys.argv[5])
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)

elif protocol == "LDPC":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
    unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 0,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    protocol_parameters["discard_pilot"] = True
    protocol_parameters["LDPC_noise_scale"] = float(sys.argv[5])
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)
    encoder = LDPCCoding(standard="802.11n", rate="1/2", z=81, ptype="A")
    decoder = LDPCDecoding(standard="802.11n", rate="1/2", z=81, ptype="A")

elif protocol == "pilot_reestimation":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
    unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 1,
        num_known_payload = 0,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    protocol_parameters["discard_pilot"] = True
    protocol_parameters["LDPC_noise_scale"] = float(sys.argv[5])
    protocol_parameters["pilot_reestimate"] = float(sys.argv[6])
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)
    encoder = LDPCCoding(standard="802.11n", rate="1/2", z=81, ptype="A")
    decoder = LDPCDecoding(standard="802.11n", rate="1/2", z=81, ptype="A")

elif protocol == "OFDM_reestimation":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
    unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 1,
        num_known_payload = 1,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    protocol_parameters["discard_pilot"] = True
    protocol_parameters["LDPC_noise_scale"] = float(sys.argv[5])
    protocol_parameters["pilot_reestimate"] = float(sys.argv[6])
    protocol_parameters["OFDM_reestimate"] = float(sys.argv[7])
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)
    encoder = LDPCCoding(standard="802.11n", rate="1/2", z=81, ptype="A")
    decoder = LDPCDecoding(standard="802.11n", rate="1/2", z=81, ptype="A")

elif protocol == "standard":
    pilot_idx = np.arange(0, N / 2 - 1, 8)
    unused_bins_idx = np.array(list(range(0, 49)) + list(range(699, int(N / 2 - 1))))
    unused_bins_idx = sorted(np.array(list(set(unused_bins_idx) - set(pilot_idx))))
    custom_parameters = dict(
        pilot_idx = pilot_idx,
        unused_bins_idx = unused_bins_idx,
        num_gap_symbols = 1,
        num_known_payload = 1,
    )
    protocol_parameters["pilot_tone_rotation"] = True
    protocol_parameters["pilot_tone_shifting"] = True
    protocol_parameters["discard_pilot"] = True
    protocol_parameters["LDPC_noise_scale"] = float(sys.argv[5])
    protocol_parameters["pilot_reestimate"] = float(sys.argv[6])
    protocol_parameters["OFDM_reestimate"] = float(sys.argv[7])
    parameters.update({"parameters": protocol_parameters})
    parameters.update(custom_parameters)
    prot = Protocol(**parameters)
    encoder = LDPCCoding(standard="802.11n", rate="1/2", z=81, ptype="A")
    decoder = LDPCDecoding(standard="802.11n", rate="1/2", z=81, ptype="A")

if action == "send":
    # raise NotImplementedError("NEED TO REPEAT 5 BLOCKS")
    uncoded_text_bits += '0' * (encoder.mycode.K - len(uncoded_text_bits) % encoder.mycode.K)
    encoded_text_bits = encoder(np.array([int(b) for b in uncoded_text_bits]))
    transmitter = Transmitter("gray", N=N, L=L, protocol=prot)
    transmitter.full_pipeline(encoded_text_bits, f"full_pipeline_transmission_audio_{reader_idx}_{protocol}")

elif sys.argv[1] == "receive":
    receiver = Receiver(N=N, L=L, constellation_name="gray", protocol = prot)
    recording_file = f"final_pipeline_message_with_estimation_{reader_idx}_{protocol}_{mode}"
    video_folder_name = "videos_final"

    outfile = TemporaryFile()
    print("recording started!")
    myrecording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=1)
    sd.wait()
    myrecording = np.array(myrecording)
    # myrecording = np.mean(myrecording, 1)
    np.save(recording_file, myrecording)  
    print("recording done!")
    channel_output = np.load(f"{recording_file}.npy", allow_pickle=True).reshape(-1)

    # artificial_channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
    # channel = Channel(artificial_channel_impulse)
    # channel_input = np.load(f"full_pipeline_transmission_audio_{reader_idx}_{protocol}.npy", allow_pickle=True).reshape(-1)
    # channel_output = channel.transmit(channel_input, noise = 0)

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

    uncoded_data_bits = uncoded_text_bits[total_num_metadata_bits:]
    error_rate = 1 - sum(uncoded_data_bits[i] == received_bitstring[i] for i in range(len(received_bitstring)))/len(received_bitstring)

    print(error_rate)
    import pdb; pdb.set_trace()
    # generate_constellation_video(video_folder_name, receiver.constellation_figs, receiver.pre_rot_constallation_figs, f"constalletion_withpilotsync")
    # generate_phaseshifting_video(video_folder_name, receiver.pilot_sync_figs, f"pilotphaseshift_withpilotsync", prot.pilot_symbol)
    # generate_channel_estim_video(video_folder_name, derived_channel, f"channelupdates_withpilotsync")
    # OFDM_generation_plot(video_folder_name, derived_channel, phase_mismatch_trials, N)
    # BER_plot(video_folder_name, uncoded_text_bits, received_bitstring)

    decoded_image_bits = np.array([int(x) for x in received_bitstring])
    a, b, c = 150, 150, 4
    image_bits = np.packbits(decoded_image_bits).reshape(a, b, c)
    if c == 4:
        image_bits = image_bits[:,:,:3]
    pil_image = Image.fromarray(image_bits, 'RGB')
    pil_image.save(f"OUTPUT_IMAGE_{protocol}_{mode}.png")