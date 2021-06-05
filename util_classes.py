from math import ceil, floor
import numpy as np
from numpy.core.numeric import full
import pandas as pd
from scipy import interpolate
from scipy.interpolate.interpolate import RegularGridInterpolator
from scipy.interpolate.polyint import KroghInterpolator
from scipy.signal import fftconvolve as convolve
from typing import FrozenSet, List, Protocol, Tuple
# from scipy.fft import fft, ifft
from numpy.fft import fft, ifft
from scipy.signal.ltisys import impulse
from scipy.signal import chirp
from util_objects import *
import binascii
import sys
import sounddevice as sd
from visualisation_scripts import *
import random
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from fractions import Fraction

# import commpy.channelcoding as cc
from sklearn.linear_model import LinearRegression
import ldpc
from tqdm import tqdm

class Channel:
    def __init__(self, impulse_response: np.ndarray):
        self.impulse_response = impulse_response
        self.past_spectra = [self.transfer_function(len(impulse_response))]
        self.past_impulses = [impulse_response]

    def transmit(self, signal: np.ndarray, noise=0.01) -> np.ndarray:
        # Transmit sequence by convolving with impulse reponse
        echo = convolve(signal, self.impulse_response)#[: len(signal)]
        echo += np.random.randn(len(echo)) * noise
        return echo

    def update_channel_impulse(self, new_impulse, new_weight):
        self.impulse_response *= 1 - new_weight
        self.impulse_response += new_weight * new_impulse

    def update_channel_spectrum(self, new_spectrum, new_weight):
        current_spectrum = self.transfer_function(len(new_spectrum))
        updated_spectrum = (
            1 - new_weight
        ) * current_spectrum + new_weight * new_spectrum
        new_impulse_response = ifft(updated_spectrum, len(self.impulse_response))
        self.impulse_response = new_impulse_response

        self.past_impulses.append(new_impulse_response)
        self.past_spectra.append(updated_spectrum)

    def transfer_function(self, length: int) -> np.ndarray:
        return fft(self.impulse_response, length)


class Protocol:
    def __init__(
        self,
        name,
        pilot_idx,
        pilot_symbol,
        chirp_length,
        chirp_func,
        N,
        L,
        parameters,
        total_num_metadata_bits,
        num_metadata_reps,
        num_gap_symbols,
        num_estimation_symbols,
        num_known_payload,
        num_message_payload,
        unused_bins_idx,
        constellation_name
    ):
        self.name = name
        self.N = N
        self.L = L
        self.unused_bins_idx = np.array(unused_bins_idx)

        self.parameters=parameters

        self.chirp_func = chirp_func
        self.total_num_metadata_bits = total_num_metadata_bits
        self.num_metadata_reps = num_metadata_reps
        self.configure_chirp(chirp_length)

        self.pilot_idx = np.array(pilot_idx)
        self.num_pilots = len(pilot_idx)
        self.pilot_symbol = pilot_symbol

        self.num_gap_symbols = num_gap_symbols
        self.num_estimation_symbols = num_estimation_symbols
        self.num_known_payload = num_known_payload
        self.num_message_payload = num_message_payload

        self.gap_bit_rng = np.random.default_rng(2020) 
        self.initial_estimation_bit_rng = np.random.default_rng(2021) 
        self.mid_message_bit_rng = np.random.default_rng(2022) 

        assert set(pilot_idx).intersection(set(unused_bins_idx)) == set()
        self.constellation = CONSTELLATIONS_DICT[constellation_name]
        self.constellation_length = len(list(self.constellation.keys())[0])
        self.generate_all_random_bits()

    def configure_chirp(self, chirp_length):
        self.chirp_samples = int(chirp_length * fs)

        t = np.linspace(0, chirp_length, self.chirp_samples)
        chirp_arr = self.chirp_func(t)
        self.chirp = chirp_arr.reshape(-1)
        self.reversed_chirp = self.chirp[::-1]

        padded_chirp = np.concatenate(
            [np.zeros(self.chirp_samples), self.chirp, np.zeros(self.chirp_samples)]
        )
        auto_conv = convolve(padded_chirp, self.reversed_chirp)[: len(padded_chirp)]
        peak_sample_idx = np.argmax(auto_conv)
        self.delay_after_peak = 2 * self.chirp_samples - peak_sample_idx

        # self.total_post_peak_samples = (
        #     self.delay_after_peak + self.num_OFDM_symbols_chirp * (self.N + self.L)
        # )

    def matched_filter(self, data):
        return convolve(data, self.reversed_chirp)[: len(data)]

    def locate_chirp_slices(self, data, sample_shift=0):
        earliest_chirp_idx = np.argmax(data[:int(len(data)/2)])
        earliest_frame_start = earliest_chirp_idx + self.delay_after_peak

        start = earliest_frame_start
        new_slice = slice(max(0, start + sample_shift), len(data))

        return [new_slice]

    def get_bits_per_chunk(self, ignore_aug=False):
        information_frequency_bins = (self.N / 2) - 1 
        constellation_per_chunk = (
            information_frequency_bins if ignore_aug
            else information_frequency_bins - self.num_pilots - len(self.unused_bins_idx)
        )  # account for pilot
        bits_per_chunk = int(
            constellation_per_chunk * self.constellation_length
        )  # number of constallation values
        return bits_per_chunk

    def locate_first_chirp(self, data, threshold):
        # Not used
        if all(data < threshold):
            return None
        first_chirp_region_start_idx = np.argmax(data > threshold)
        first_chirp_region = slice(
            first_chirp_region_start_idx,
            first_chirp_region_start_idx + self.total_post_peak_samples,
        )
        first_peak_idx = (
            np.argmax(data[first_chirp_region]) + first_chirp_region_start_idx
        )
        return first_peak_idx

    def OLD_locate_chirp_slices(self, data, sample_shift=0):
        frames_removed = 0
        _data = data.copy()
        slices_located = []
        highest_sample = np.max(data)
        threshold = 0.8 * highest_sample

        while len(_data) > 0:  # (self.total_post_peak_samples):
            earliest_chirp_idx = self.locate_first_chirp(_data, threshold)
            if not earliest_chirp_idx:
                break
            earliest_frame_start = earliest_chirp_idx + self.delay_after_peak
            earliest_frame_end = earliest_chirp_idx + self.total_post_peak_samples

            start = earliest_frame_start + frames_removed
            stop = earliest_frame_end + frames_removed
            new_slice = slice(max(0, start + sample_shift), stop + sample_shift)
            slices_located.append(new_slice)

            cutout_samples = earliest_frame_end  # - self.chirp_samples
            frames_removed += cutout_samples
            _data = data[frames_removed:]

        return slices_located

    def insert_pilot_tones_and_remove_unused_frequencies(self, constallation_values):

        post_pilot_constallation_values = [None for _ in range(int(self.N/2-1))]
        
        for p_idx in self.pilot_idx:
            post_pilot_constallation_values[int(p_idx)] = self.pilot_symbol
        for u_idx in self.unused_bins_idx:
            post_pilot_constallation_values[int(u_idx)] = random.choice(list(self.constellation.values()))
        
        for j, value in enumerate(post_pilot_constallation_values):
            if value == None:
                post_pilot_constallation_values[j] = constallation_values[0]
                constallation_values = constallation_values[1:]

        # i = 0
        # for j in range(len(post_pilot_constallation_values)):
        #     if j in self.pilot_idx:
        #         post_pilot_constallation_values[j] = self.pilot_symbol
        #     elif j + 1 in self.unused_bins_idx:
        #         post_pilot_constallation_values[j] = random.choice(list(self.constellation.values()))
        #     else:
        #         post_pilot_constallation_values[j] = constallation_values[i]
        #         i += 1
        return post_pilot_constallation_values

    def get_pilot_idx(self):
        left_pilot_idx = (self.pilot_idx + 1).astype(int)
        if self.parameters["discard_pilot"]:
            u_b_i = self.unused_bins_idx
            left_pilot_idx = np.sort(np.array(list(set(left_pilot_idx) - set(u_b_i))))
        right_pilot_idx = (self.N - self.pilot_idx - 1).astype(int)[::-1]
        return left_pilot_idx, right_pilot_idx

    def generate_all_random_bits(self):
        bits_per_chunk = self.get_bits_per_chunk(ignore_aug=True)

        num_gap_bits = self.num_gap_symbols * bits_per_chunk
        self.gap_bits = self.gap_bit_rng.integers(low=0, high=2, size=num_gap_bits)

        if self.parameters["repeat_estimation_blocks"]:
            assert self.num_estimation_symbols % 2 == 0
            num_estimation_bits = int(self.num_estimation_symbols * bits_per_chunk / 2)
            first_block = self.initial_estimation_bit_rng.integers(low=0, high=2, size=num_estimation_bits)
            self.estimation_bits = np.concatenate([first_block, first_block])
        else:
            num_estimation_bits = self.num_estimation_symbols * bits_per_chunk
            self.estimation_bits = self.initial_estimation_bit_rng.integers(low=0, high=2, size=num_estimation_bits)

        num_mid_message_bits = self.num_known_payload * bits_per_chunk
        self.mid_message_bits = self.mid_message_bit_rng.integers(low=0, high=2, size=num_mid_message_bits)

        num_end_padding_bits = bits_per_chunk
        self.end_padding_bits = self.gap_bit_rng.integers(low=0, high=2, size=num_end_padding_bits)

class Modulation:
    def __init__(self, constellation_name: str, N: int, L: int, protocol: Protocol):
        self.constellation = CONSTELLATIONS_DICT[constellation_name]
        self.constellation_length = len(list(self.constellation.keys())[0])
        self.N = N
        self.L = L
        self.protocol = protocol

    def bits2constellation(self, bitstring: str) -> float:
        # Convert a single string of self.constellation length to the correct constellation value
        return self.constellation[tuple(int(b) for b in bitstring)]

    def sequence2constellation(self, bitstring: str) -> List[float]:
        # Convert a string of bits to a list of constellation value
        n = self.constellation_length
        frames = [bitstring[i : i + n] for i in range(0, len(bitstring), n)]
        if len(frames[-1]) < n:
            frames[-1] = frames[-1] + "0" * (n - len(frames[-1]))
        return [self.bits2constellation(frame) for frame in frames]

    def bits2OFDM(self, bitstring: str, ignore_aug=False) -> List[float]:

        constallation_values = self.sequence2constellation(bitstring)
        if not ignore_aug:
            constallation_values = self.protocol.insert_pilot_tones_and_remove_unused_frequencies(
                constallation_values
            )
        symmetric_conj = np.conj(constallation_values)[::-1]
        symmetric_conj_mirrored = np.concatenate(
            [[0], constallation_values, [0], symmetric_conj]
        )
        time_domain_symmetric_conj = ifft(symmetric_conj_mirrored)
        message_with_cyclical_prefix = np.concatenate(
            [time_domain_symmetric_conj[-self.L :], time_domain_symmetric_conj]
        )

        return message_with_cyclical_prefix.real

    @staticmethod
    def split_bitstring_into_chunks(data, N_mod):
        frames = [data[i : i + N_mod] for i in range(0, len(data), N_mod)]
        if len(frames):
            if len(frames[-1]) < N_mod:
                try:
                    frames[-1] = frames[-1] + "0" * (N_mod - len(frames[-1]))
                except:
                    frames[-1] = np.concatenate(
                        [frames[-1], [0] * (N_mod - len(frames[-1]))]
                    )
            return frames
        else:
            return []

    def data2OFDM(
        self, bitstring: str, return_frames=False, ignore_aug = False
    ) -> List[float]:

        bits_per_chunk = self.protocol.get_bits_per_chunk(ignore_aug)
        
        OFDM_data = self.split_bitstring_into_chunks(bitstring, bits_per_chunk)
        OFDM_data = [
            self.bits2OFDM(OFDM_symbol, ignore_aug) for OFDM_symbol in OFDM_data
        ]

        if not return_frames:
            return 
        else:
            return OFDM_data

    @staticmethod
    def publish_data(data, name):
        np.save(name, data, allow_pickle=True)


class Demodulation:
    def __init__(self, N: int, L: int, constellation_name: str, protocol: Protocol):
        self.N = N
        self.L = L
        self.constellation = CONSTELLATIONS_DICT[constellation_name]
        self.constellation_figs = []
        self.pre_rot_constallation_figs = []
        self.protocol = protocol

    def OFDM2constellation(self, channel_output: np.ndarray, channel: Channel):
        num_frames = len(channel_output) / (self.N + self.L)
        with_cyclic_frames = np.array_split(channel_output, num_frames)
        message_frames = [list(w[self.L :]) for w in with_cyclic_frames]
        fft_frames = [fft(m, self.N) for m in message_frames]
        channel_TF = channel.transfer_function(self.N)
        deconvolved_frames = [np.divide(r, channel_TF) for r in fft_frames]
        return deconvolved_frames

    def constellation2bits_single(self, constellation_value: float) -> Tuple[int]:
        return min(
            self.constellation.keys(),
            key=lambda x: abs(constellation_value - self.constellation[x]),
        )

    def constellation2bits_sequence(
        self, constellation_values: List[float], constellation_values_pre_rot, show=True
    ) -> str:

        symbol_bits_sequence = []
        const_hist = []
        const_hist_pre_rot = []
        for j, d in enumerate(constellation_values):
            if np.isnan(d):
                continue
            symbol_bits_sequence.extend(self.constellation2bits_single(d))
            if show:
                const_hist.extend([d])
                const_hist_pre_rot.extend([constellation_values_pre_rot[j]])
        output_bitstring = "".join([str(a) for a in symbol_bits_sequence])

        if show:
            new_fig = np.array(const_hist)  # new_fig = constellation_plot(const_hist)
            new_fig_pre_rot = np.array(
                const_hist_pre_rot
            )  # new_fig_pre_rot = constellation_plot(const_hist_pre_rot)
            self.constellation_figs.append(new_fig)
            self.pre_rot_constallation_figs.append(new_fig_pre_rot)

        return output_bitstring

    def bitstring2text(self, bitstring: str) -> str:
        output_bytes = [bitstring[i : i + 8] for i in range(0, len(bitstring), 8)]
        output_bytes = bytearray([int(i, 2) for i in output_bytes])

        return output_bytes

    def receive_channel_output(
        self,
        channel_output: np.ndarray,
        sample_shift=0,
        return_as_slices=False,
    ):

        OFDM_frames = []
        OFDM_slices = []

        chirp_filtered = self.protocol.matched_filter(channel_output)
        chirp_slices = self.protocol.locate_chirp_slices(
            chirp_filtered, sample_shift
        )

        slice_lengths = [int(sl.stop - sl.start) for sl in chirp_slices]
        for j, sl in enumerate(slice_lengths):
            sl_length = slice_lengths[j]
            num_symbols_in_slice = int(sl_length / (self.N + self.L))
            symbols_in_slices = [
                slice(
                    int(i * (self.N + self.L)) + chirp_slices[j].start,
                    int((i + 1) * (self.N + self.L)) + chirp_slices[j].start,
                )
                for i in range(num_symbols_in_slice)
            ]

            OFDM_slices.extend(symbols_in_slices)

        OFDM_frames = [channel_output[sl] for sl in OFDM_slices]

        if return_as_slices:
            return OFDM_slices, chirp_filtered, chirp_slices
        else:
            return OFDM_frames, chirp_filtered, chirp_slices

    def OFDM2bits(
        self,
        channel_output: np.ndarray,
        channel: Channel,
        sample_shift=0,
    ) -> str:

        OFDM_frames = self.receive_channel_output(
            channel_output, channel=channel, sample_shift=sample_shift
        )

        bitstring = ""
        for frame in OFDM_frames:
            deconvolved_frames = self.OFDM2constellation(frame, channel)
            deconvolved_frames = [
                dcf[1 : int(self.N / 2)] for dcf in deconvolved_frames
            ]
            bitstring += self.constellation2bits_sequence(deconvolved_frames)

        return bitstring


class Estimation(Demodulation):
    def __init__(self, N: int, L: int, constellation_name: str, protocol: Protocol):
        super().__init__(N, L, constellation_name, protocol)

    def transfer_function_trials(self, ground_truth_OFDM_frames, received_OFDM_frames):
        
        transfer_function_trials = []
        phase_mismatch_trials = []
        ground_truth_OFDM_frames = ground_truth_OFDM_frames

        for j, ground_truth_OFDM_frame in enumerate(ground_truth_OFDM_frames):
            input_spectrum = fft(
                ground_truth_OFDM_frame[self.L :]
            ).round()
            output_spectrum = fft(received_OFDM_frames[j][self.L :])
            transfer_function = np.divide(output_spectrum, input_spectrum)

            transfer_function[0] = 0
            transfer_function[int(self.N / 2)] = 0

            phase_mismatch = [
                np.angle(in_c) - np.angle(output_spectrum[j]) for j, in_c in enumerate(input_spectrum)
            ]

            phase_mismatch_trials.append(phase_mismatch)
            transfer_function_trials.append(transfer_function)

        return transfer_function_trials, phase_mismatch_trials

    def extract_average_impulse(self, transfer_function_trials, estimation_rotation):

        if type(estimation_rotation) != type(None):
            transfer_function_trials = np.array(transfer_function_trials)
            complex_array = np.array([np.exp(1j * er * np.array(range(self.N)).astype(complex)) for er in estimation_rotation])
            transfer_function_trials = transfer_function_trials / complex_array

        average_transfer_function = np.mean(transfer_function_trials, 0)
        average_impulse = ifft(average_transfer_function)

        return average_impulse

    def OFDM_channel_estimation(
        self, channel_output, ground_truth_OFDM_frames, sample_shift
    ):
        received_OFDM_frames = self.receive_channel_output(
            channel_output, sample_shift
        )

        transfer_function_trials, phase_mismatch_trials = self.transfer_function_trials(
            ground_truth_OFDM_frames, received_OFDM_frames
        )
        return self.extract_average_impulse(transfer_function_trials)


class Transmitter(Modulation):
    def __init__(
        self,
        constellation_name: str,
        N: int,
        L: int,
        protocol,
    ):
        super().__init__(constellation_name, N, L, protocol)

    def full_pipeline(self, message_bits, OFDM_file_name):
        message_bits = "".join(str(mb) for mb in message_bits)
        
        if self.protocol.num_gap_symbols:
            gap_frames = self.data2OFDM(bitstring=self.protocol.gap_bits, return_frames=True, ignore_aug=True)
        else:
            gap_frames = []
        if self.protocol.num_estimation_symbols:
            estimation_frames = self.data2OFDM(bitstring=self.protocol.estimation_bits, return_frames=True, ignore_aug=True)
        else:
            estimation_frames = []
        if self.protocol.num_known_payload:
            mid_message_frames = self.data2OFDM(bitstring=self.protocol.mid_message_bits, return_frames=True, ignore_aug=True)
        else:
            mid_message_frames = []
        message_frames = self.data2OFDM(bitstring=message_bits, return_frames=True, ignore_aug=False)
        padding_frames = self.data2OFDM(bitstring=self.protocol.end_padding_bits, return_frames=True)
        total_num_frames = len(gap_frames) + len(estimation_frames) + len(mid_message_frames) + len(message_frames)

        remaining_frames_num = self.protocol.num_message_payload - len(message_frames)%self.protocol.num_message_payload
        remaining_frames = [padding_frames[0] for _ in range(remaining_frames_num)]

        final_transmission = []
        final_transmission.extend(self.protocol.chirp)
        for gap_frame in gap_frames:
            final_transmission.extend(gap_frame)
        for estimation_frame in estimation_frames:
            final_transmission.extend(estimation_frame)

        for j, OFDM_symbol in enumerate(message_frames):
            if j % self.protocol.num_message_payload == 0:
                for mid_message_frame in mid_message_frames:
                    final_transmission.extend(mid_message_frame.copy())
            final_transmission.extend(OFDM_symbol)

        for rem_frame in remaining_frames:
            final_transmission.extend(rem_frame)

        final_transmission.extend(self.protocol.chirp[::-1])

        print(
            f"Total {total_num_frames} OFDM symbols generated"
        )
        self.publish_data(final_transmission, OFDM_file_name)
        print(f"OFDM data eith chirp and estimation symbols save to {OFDM_file_name}")


class Receiver(Estimation):
    def __init__(
        self,
        N: int,
        L: int,
        constellation_name: str,
        protocol: Protocol
    ):
        super().__init__(N, L, constellation_name, protocol)
        self.pilot_sync_figs = []

    def interpolate_channel(
        self,
        left_pilot_idx,
        right_pilot_idx,
        recovered_pilot_tones_left,
        recovered_pilot_tones_right,
        deconvolved_frames,
    ):

        pilot_spectrum_left = recovered_pilot_tones_left / self.protocol.pilot_symbol
        pilot_spectrum_right = recovered_pilot_tones_right / np.conj(
            self.protocol.pilot_symbol
        )

        x = (
            np.concatenate([[0], left_pilot_idx, [int(self.N / 2)], right_pilot_idx])
            / len(deconvolved_frames[0])
            - 0.5
        )
        y = np.concatenate(
            [[0 + 0j], pilot_spectrum_left, [0 + 0j], pilot_spectrum_right]
        )

        full_x = (
            np.array(range(len(deconvolved_frames[0]))) / len(deconvolved_frames[0])
            - 0.5
        )

        interpolator = interpolate.interp1d(x, y)
        full_pilot_spectrum = interpolator(full_x)

        return full_pilot_spectrum

    def get_recovered_pilot_tones(
        self, deconvolved_frames, left_pilot_idx, right_pilot_idx
    ):
        recovered_pilot_tones_left = deconvolved_frames[0][left_pilot_idx]
        recovered_pilot_tones_right = deconvolved_frames[0][right_pilot_idx]
        return recovered_pilot_tones_left, recovered_pilot_tones_right

    def get_left_phase_shifts(self, recovered_pilot_tones_left):
        get_phase = lambda x: np.angle(x)
        phase_shifts = [
            get_phase(r) + np.angle(self.protocol.pilot_symbol)
            for r in recovered_pilot_tones_left
        ]
        return phase_shifts

    def linear_regression_offset(self, left_pilot_idx, phase_shifts):
        # TODO: REFIND DODGY BIN WITH ONE EVERY 8
        # asf = np.delete(phase_shifts, 25)
        # asf2 = np.delete(left_pilot_idx, 25)
        asf = phase_shifts
        asf2 = left_pilot_idx
        # model = LinearRegression().fit(left_pilot_idx[1:, np.newaxis], phase_shifts[1:])
        model = LinearRegression().fit(asf2[1:, np.newaxis], asf[1:])
        slope = model.coef_[0]
    
        return slope

    def fix_constellation_frame(self, deconvolved_frame, lin_reg_slope, pilot_idx):
        complex_array = np.exp(
            1j * lin_reg_slope * np.array(range(len(deconvolved_frame))).astype(complex)
        )
        return deconvolved_frame / complex_array

    def get_useful_constellations(self, frames):
        frames[0][(self.protocol.pilot_idx + 1).astype(int)] = None
        frames[0][(self.protocol.unused_bins_idx + 1).astype(int)] = None
        cut_frames = [dcf[1 : int(self.N / 2)] for dcf in frames]
        return [d for d in cut_frames[0] if ~np.isnan(d)]

    def extract_metadata(self, first_constellations, derived_channel, decoder):
        channel_spectrum = derived_channel.transfer_function(self.N)
        kw = {
            "demodulator": self, 
            "received_constellations": first_constellations,
            "pre_rot_received_constellations": None,
            "show": False,
            "channel_estimation": channel_spectrum,
            "scaling": self.protocol.parameters["LDPC_noise_scale"]
        }
        decoded_bits = decoder.decode(**kw)
        metadata_bits = decoded_bits[:self.protocol.total_num_metadata_bits]
        return process_metadata(metadata_bits)

    def full_pipeline(
        self,
        channel_output,
        ground_truth_estimation_OFDM_frames,
        ground_truth_reestimation_OFDM_frames,
        sample_shift,
        decoder,
    ):
        print("receiving signal")
        received_OFDM_slices, chirp_filtered, chirp_slices = self.receive_channel_output(channel_output, sample_shift, return_as_slices=True)
        estimation_ofdm_slices = received_OFDM_slices[self.protocol.num_gap_symbols: self.protocol.num_estimation_symbols+self.protocol.num_gap_symbols]
        message_ofdm_slices = received_OFDM_slices[self.protocol.num_gap_symbols + self.protocol.num_estimation_symbols:]
        estimation_ofdm_frames = [channel_output[sl] for sl in estimation_ofdm_slices]
        transfer_function_trials, phase_mismatch_trials = self.transfer_function_trials(ground_truth_estimation_OFDM_frames, estimation_ofdm_frames)

        impulse_response = self.extract_average_impulse(transfer_function_trials, None)
        derived_channel = Channel(impulse_response.real)
        
        received_constellations, pre_rot_received_constellations, slope_history = [], [], []

        OFDM_generation_plot("./", derived_channel, phase_mismatch_trials, self.N, self.protocol.name)
        chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices, self.protocol.name)
        print("channel estimated - beginning demodulation...")

        total_cycle_num_symbols = self.protocol.num_known_payload + self.protocol.num_message_payload

        total_offset = 0
        num_required_frames_total = 0
        depth_set = False
        for o_idx, ofdm_slice in tqdm(enumerate(message_ofdm_slices)):

            offset_slice = slice(ofdm_slice.start - total_offset, ofdm_slice.stop - total_offset)
            frame = channel_output[offset_slice]

            if o_idx % total_cycle_num_symbols < self.protocol.num_known_payload:
                if self.protocol.parameters["OFDM_reestimate"]:
                    new_transfer_function_trials, _ = self.transfer_function_trials(
                        ground_truth_reestimation_OFDM_frames, [frame]
                    )
                    new_transfer_function = np.mean(new_transfer_function_trials, axis = 0)
                    derived_channel.update_channel_spectrum(new_transfer_function, self.protocol.parameters["OFDM_reestimate"])
                continue

            if len(frame) < (self.N + self.L):
                break
            deconvolved_frames = self.OFDM2constellation(frame, derived_channel)
            old_deconvolved_frames = deconvolved_frames.copy()

            left_pilot_idx, right_pilot_idx = self.protocol.get_pilot_idx()        

            if len(left_pilot_idx):

                recovered_pilot_tones_left, recovered_pilot_tones_right = self.get_recovered_pilot_tones(deconvolved_frames, left_pilot_idx, right_pilot_idx)
                # if not np.isclose(np.unique(recovered_pilot_tones_left[0])[0], 1+1j):
                #     import pdb; pdb.set_trace()
                phase_shifts = self.get_left_phase_shifts(recovered_pilot_tones_left)
                lin_reg_slope = self.linear_regression_offset(left_pilot_idx, phase_shifts)

                if self.protocol.parameters["pilot_tone_rotation"]:
                    deconvolved_frames = [self.fix_constellation_frame(d, lin_reg_slope, left_pilot_idx) for d in deconvolved_frames]

                # TODO: how does this cope with skipped o_idx
                current_delay = (self.N * lin_reg_slope) / (2 * np.pi)
                # slope_history.append(lin_reg_slope)
                slope_history.append(current_delay)
                if abs(current_delay) > 0.5 and self.protocol.parameters["pilot_tone_shifting"]:
                    print("SHIFTING!!")
                    if not depth_set:
                        depth = o_idx
                        depth_set = True
                    print(current_delay)
                    total_offset += int(current_delay / abs(current_delay))
                self.pilot_sync_figs.append((left_pilot_idx, recovered_pilot_tones_left, phase_shifts, self.N))

                if self.protocol.parameters["pilot_reestimate"]:
                    full_pilot_spectrum = self.interpolate_channel(
                        left_pilot_idx,
                        right_pilot_idx,
                        recovered_pilot_tones_left,
                        recovered_pilot_tones_right,
                        deconvolved_frames,
                    )
                    derived_channel.update_channel_spectrum(full_pilot_spectrum, self.protocol.parameters["pilot_reestimate"])

            received_constellations.extend(self.get_useful_constellations(deconvolved_frames))
            pre_rot_received_constellations.extend(self.get_useful_constellations(old_deconvolved_frames))

            req1 = not num_required_frames_total
            req2 = len(received_constellations) >= (self.protocol.total_num_metadata_bits / self.protocol.constellation_length + 1)/decoder.rate
            req3 = (len(received_constellations)/self.protocol.constellation_length) // decoder.mycode.N

            if req1 and req2 and req3:
                first_constellations = np.array(received_constellations).reshape(1, -1)
                signal_metadata = self.extract_metadata(first_constellations, derived_channel, decoder)
                unencoded_bits_required = self.protocol.total_num_metadata_bits + signal_metadata["total_data_bits"]
                encoded_bits_required = ceil(unencoded_bits_required / decoder.rate)

                num_required_message_frames = ceil(encoded_bits_required/self.protocol.get_bits_per_chunk())
                num_cycles = ceil(num_required_message_frames/self.protocol.num_message_payload)
                cycle_size = self.protocol.num_known_payload + self.protocol.num_message_payload

                num_required_frames_total = int(num_cycles * cycle_size)

            if o_idx == num_required_frames_total - 1:
                break
        
        if len(left_pilot_idx):
            try:
                graph(slope_history, depth, self.protocol.name)
            except UnboundLocalError:
                depth = len(slope_history)
                graph(slope_history, depth, self.protocol.name)

        print("demodulation completed - beginning decoding...")

        received_constellations = np.array(received_constellations).reshape(1, -1)
        pre_rot_received_constellations = np.array(pre_rot_received_constellations).reshape(1, -1)
        channel_spectrum = derived_channel.transfer_function(self.N)
        kw = {
            "demodulator": self, 
            "received_constellations": received_constellations,
            "pre_rot_received_constellations": pre_rot_received_constellations,
            "show": True,
            "channel_estimation": channel_spectrum,
            "scaling": self.protocol.parameters["LDPC_noise_scale"]
        }
        decoded_bits = decoder.decode(**kw)
        databits = decoded_bits[self.protocol.total_num_metadata_bits: unencoded_bits_required]
        return (
            databits,
            derived_channel,
            phase_mismatch_trials,
            signal_metadata,
        )


class Encoding:
    def __init__(self):
        pass

    def encode(self, inputs):
        raise NotImplementedError("Need to specify coding type")

    def enc_func(self):
        print("we did not define this for ConvCoding, but it can still be used by it")


# This is a rate 1/2 convolutional code
class ConvCoding(Encoding):
    def __init__(self, g_matrix=np.array([[0o5, 0o7]])):
        super().__init__()
        self.g_matrix = g_matrix

    def encode(self, inputs: np.ndarray, m: int = 2):
        memory = np.array([m])
        trellis = cc.convcode.Trellis(memory, self.g_matrix)
        outputs = cc.conv_encode(inputs, trellis)
        return outputs

class NoCoding(Encoding):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, inputs):
        return inputs

class LDPCCoding(Encoding):
    def __init__(self, standard, rate, z, ptype):
        super().__init__()
        self.mycode = ldpc.code(standard, rate, z, ptype)

    def __call__(self, inputs: np.ndarray):
        s = len(inputs)
        ceiling = np.ceil(s / self.mycode.K)
        pad = np.random.randint(2, size=int(ceiling * self.mycode.K - s))
        padded_inputs = np.concatenate((inputs, pad))
        padded_inputs_split = np.split(padded_inputs, ceiling)
        ldpc_coded = []
        for i in range(int(ceiling)):
            coded = self.mycode.encode(padded_inputs_split[i])
            ldpc_coded.append(coded)
        encoded_message = np.ravel(ldpc_coded)
        return encoded_message

class DummyMyCode:
    def __init__(self):
        self.N = 200 # PARAMETERISE THIS

class Decoding:
    def __init__(self):
        pass

    def decode(self, outputs):
        raise NotImplementedError("Need to specify decoding type")

    def enc_func(self):
        print("we did not define this for ConvDecoding, but it can still be used by it")


# This is for decoding a rate 1/2 convolutional code
class ConvDecoding(Decoding):
    def __init__(self, g_matrix=np.array([[0o5, 0o7]])):
        super().__init__()
        self.g_matrix = g_matrix

    def decode(self, outputs: np.ndarray, m: int = 2):
        memory = np.array([m])
        trellis = cc.convcode.Trellis(memory, self.g_matrix)
        decoded = cc.viterbi_decode(outputs, trellis)[:-m]
        return decoded

class NoDecoding(Decoding):
    def __init__(self):
        super().__init__()
        self.rate = 1
        self.mycode = DummyMyCode()

    def decode(self, **kwargs):
        demodulator = kwargs.get("demodulator")
        received_constellations = kwargs.get("received_constellations").reshape(-1)
        pre_rot_received_constellations = kwargs.get("pre_rot_received_constellations")
        if type(pre_rot_received_constellations) != type(None):
            pre_rot_received_constellations = pre_rot_received_constellations.reshape(-1)
        show = kwargs.get("show")
        return demodulator.constellation2bits_sequence(received_constellations, pre_rot_received_constellations, show)


class LDPCDecoding(Decoding):
    def __init__(self, standard, rate, z, ptype):
        super().__init__()
        self.mycode = ldpc.code(standard, rate, z, ptype)
        self.rate = float(Fraction(rate))

    def decode(self, **kwargs):

        received_constellation = kwargs.get("received_constellations")
        channel_estimation = kwargs.get("channel_estimation")
        scaling = kwargs.get("scaling")

        ckarraylen = int(len(channel_estimation) / 2 - 1)
        ckarray = channel_estimation[1 : 1 + ckarraylen]
        llr = []

        for i in range(len(received_constellation[0])):
            # take sigma squared to be 1 as they do not affect the results

            # POSSIBLY ADD NEW CONSTELLATIONS
            yir = received_constellation[0][i].real
            yii = received_constellation[0][i].imag
            ckindex = i % ckarraylen
            ck = ckarray[ckindex]
            ck_squared = ck * np.conjugate(ck)
            ck2 = ck_squared.real
            li2 = np.sqrt(2) * ck2 * yir / scaling
            li1 = np.sqrt(2) * ck2 * yii / scaling
            # Gray coding
            llr.append(li1)
            llr.append(li2)

        # each segmented component is of length mycode.N
        ceiling = floor(len(llr) // self.mycode.N)  # pr450
        llr = llr[: int(ceiling * self.mycode.N)]  # pr450
        llr_split = np.split(np.array(llr), ceiling)

        llr_ldpc_decoded = []
        for i in tqdm(range(len(llr_split))):
            app, it = self.mycode.decode(llr_split[i])
            app_half = app[: self.mycode.K]
            llr_ldpc_decoded.append(app_half)

        llr_ravel = np.ravel(llr_ldpc_decoded)

        decoded_message = []
        for i in llr_ravel:
            if i > 0:
                decoded_bit = 0
            else:
                decoded_bit = 1
            decoded_message.append(decoded_bit)

        n = 2
        precoded_llr = np.ravel(llr)
        predecoded_bits = (precoded_llr < 0).astype(int)
        bitchunks = [
            tuple(predecoded_bits[i : i + n]) for i in range(0, len(predecoded_bits), n)
        ]
        decoded_constellations = [CONSTELLATIONS_DICT["gray"][bc] for bc in bitchunks]

        return decoded_message#, decoded_constellations