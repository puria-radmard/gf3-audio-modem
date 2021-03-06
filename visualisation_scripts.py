from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import glob
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
import seaborn as sns

sns.set_style("darkgrid")

def BER_plot(video_folder, uncoded_text_bits, received_bitstring):

    correct_bits = 0
    BER_history = []
    for i, unc_bit in enumerate(uncoded_text_bits):
        if unc_bit == received_bitstring[i]:
            correct_bits += 1
        BER_history.append(correct_bits/(i+1))
    
    fig, axs = plt.subplots(1)
    axs.set_title("Average BER against received bit")
    axs.plot(BER_history)
    fig.savefig(f"{video_folder}/BER_history")


def OFDM_generation_plot(video_folder, derived_channel, phase_mismatch_trials, N, protocol):
        fig, axs = plt.subplots(2, figsize = (16, 20))
        axs[0].set_title("Channel response via OFDM")
        axs[0].set_xlabel("Sample number")
        axs[0].set_ylabel("Impulse coeff")
        axs[0].plot(derived_channel.impulse_response.real, label=f"no shift")
        axs[1].set_title("Channel spectrum via OFDM")
        axs[1].set_xlabel("Frequency bin")
        axs[1].set_ylabel("Magnitude")
        axs[1].set_yscale("log")
        channel_spectrum = np.abs(derived_channel.transfer_function(N))
        axs[1].plot(channel_spectrum, label=f"no shift")
        axs[1].set_ylim(
            sorted(abs(channel_spectrum))[2],
            sorted(abs(channel_spectrum))[-1],
        )
        
        for ax in axs:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
        # axs[2].set_title("Phase offset of OFDM constellations (with phase)")
        # axs[2].set_xlabel("Frequency bin")
        # axs[2].set_ylabel("Phase offset") 
        # for i, pmt in enumerate(phase_mismatch_trials):
        #     # axs[2].scatter(range(len(pmt)), pmt, label = i)
        #     axs[2].scatter(range(len(pmt)), pmt, label = i)
        # axs[2].legend()
        fig.savefig(f"{video_folder}/impulse_response_pipeline_{protocol}.png")
        # np.save(f"{video_folder}/impulse_response_pipline", derived_channel.impulse_response.real)


def generate_constellation_video(folder, receiver, animation_name):

    num_vals_per_block = int(receiver.protocol.get_bits_per_chunk()/receiver.protocol.constellation_length)
    img, img_pre_rot = receiver.constellation_figs, receiver.pre_rot_constallation_figs

    frames = []

    for i in tqdm(range(0, len(img), num_vals_per_block)):

        sl = slice(i, i + num_vals_per_block)

        arr = img[sl]
        arr_pre_rot = img_pre_rot[sl]

        fig, axs = plt.subplots(1, figsize=(10, 10))
        axs = [axs, None]
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(-2, 2)
        # axs[1].set_xlim(-2, 2)
        # axs[1].set_ylim(-2, 2)

        axs[0].scatter(
            arr.real, arr.imag, c=np.array(range(len(arr)))
        )
        axs[0].set_title("Value rotation and sample shifting")

        # last_scatter = axs[1].scatter(arr.real, arr.imag, c=np.array(range(len(arr))))
        # axs[1].set_title("Sample shifting and value rotation")
        # cbar= fig.colorbar(last_scatter,extend="max",location="bottom")

        fig.savefig(folder + "/file%02d" % i)
        frames.append(folder + "/file%02d.png" % i)

    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        f"{folder}/{animation_name}.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        1.5,
        (width, height),
    )

    for image in frames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for file_name in glob.glob(f"{folder}/file*.png"):
        os.remove(file_name)


def estimation_blocks_shift(shift, depth):
    model = LinearRegression().fit(np.arange(depth)[:, np.newaxis], shift[:depth])
    shift_estimate = model.predict(np.arange(-10,0)[:, np.newaxis])
    return(shift_estimate)

def graph(shift,depth,protocol):
    model2 = LinearRegression().fit(np.arange(depth)[:, np.newaxis], shift[:depth])
    ytrue = model2.predict(np.arange(-10,0)[:, np.newaxis])
    score=np.zeros(depth)
    hi=np.zeros(depth)
    for i in range(1,depth+1):
            model2 = LinearRegression().fit(np.arange(i)[:, np.newaxis], shift[:i])
            y_fit2 = model2.predict(np.arange(-10,i)[:, np.newaxis])
            hi[i-1]=model2.coef_[0]
            score[i-1]=model2.score(np.arange(-10,0)[:, np.newaxis],ytrue)
    # plt.figure()
    # plt.plot(hi)
    # plt.savefig(f'slope_{protocol}')
    # plt.figure() 
    # plt.plot(score[3:])
    # plt.savefig(f'score_{protocol}') 

    fig, axs = plt.subplots(1, figsize = (15, 10))
    # axs.plot(np.arange(-10,depth),y_fit2)
    axs.plot(shift)
    axs.set_xlabel("OFDM symbol number")
    axs.set_ylabel("$\hat\\tau$")
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
             axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(30)
    fig.savefig(f'actualregression_{protocol}')

def generate_phaseshifting_video(folder, img, animation_name, pilot_symbol):

    frames = []

    for j, (left_pilot_idx, recovered_pilot_tones_left, phase_shifts, N) in enumerate(
        img
    ):
        if j%10:
            continue
        #fig, axs = plt.subplots(2, figsize=(15, 30))
        fig, axs = plt.subplots(1, figsize=(15, 15))
        axs = [None, axs]

        #axs[0].set_ylim(0, 2)
        #axs[0].set_xlim(0, 2)
        # sc = axs[0].scatter(
        #     x=recovered_pilot_tones_left.real,
        #     y=recovered_pilot_tones_left.imag,
        #     c=left_pilot_idx,
        # )
        # cbar = plt.colorbar(sc)
        # cbar.set_label(f"Index in OFDM symbol (N = {N})")

        axs[1].set_title(f"Phase shift of pilot tones in OFDM symbol {j+1}")
        axs[1].plot(left_pilot_idx, phase_shifts)
        axs[1].set_ylabel("Phase shift")
        axs[1].set_xlabel(f"Index in OFDM symbol (N = {N})")

        for ax in [axs[1]]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text()] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(40)

        fig.savefig(f"{folder}/pilotestimationrotation{j}.png")
        frames.append(f"{folder}/pilotestimationrotation{j}.png")

    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        f"{folder}/{animation_name}.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        1.5,
        (width, height),
    )

    for image in frames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for file_name in glob.glob(f"{folder}/pilotestimationrotation*.png"):
        os.remove(file_name)


def viz_LDPC_constellation_correation(
    video_folder_name, received_constellations, decoded_constellations, filename
):

    fig, axs = plt.subplots(1)
    axs.set_title("LDPC correction ")

    all_consts = np.unique(decoded_constellations).tolist()
    colour_key = [all_consts.index(dc) for dc in decoded_constellations]

    axs.scatter(
        received_constellations.real, received_constellations.imag, c=colour_key
    )
    axs.set_xlim(-2, 2)
    axs.set_ylim(-2, 2)
    fig.savefig(f"{video_folder_name}/{filename}.png")


def generate_channel_estim_video(folder, channel, animation_name):

    frames = []

    for j, impulse in enumerate(channel.past_impulses):
        fig, axs = plt.subplots(2)
        axs[0].plot(impulse)
        axs[0].set_ylabel("Updated Channel Pulse Response")
        axs[0].set_xlabel("Sample Number")
        axs[0].xaxis.tick_top()
        axs[1].plot(abs(channel.past_spectra[j]))
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Updated Channel Spectrum")
        axs[1].set_xlabel("DFT Frequency Bin")
        axs[1].set_ylim(
            sorted(abs(channel.past_spectra[j]))[2],
            sorted(abs(channel.past_spectra[j]))[-1],
        )

        fig.savefig(f"{folder}/updated_channel{j}.png")
        frames.append(f"{folder}/updated_channel{j}.png")

    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        f"{folder}/{animation_name}.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        1.5,
        (width, height),
    )

    for image in frames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for file_name in glob.glob(f"{folder}/updated_channel*.png"):
        os.remove(file_name)


def constellation_plot(const_hist):
    fig, axs = plt.subplots(1)
    arr = np.array(const_hist)
    axs.scatter(arr.real, arr.imag)
    axs.set_xlim(-2, 2)
    axs.set_ylim(-2, 2)
    fig.savefig("received constallation bits", dpi=fig.dpi)
    return arr


def chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices, protocol):

    fig, axs = plt.subplots(2, figsize = (16, 10))
    axs[0].plot(channel_output)
    real_stop = min([chirp_slices[0].stop, len(channel_output)])
    axs[0].plot(
        range(chirp_slices[0].start, real_stop), channel_output[chirp_slices[0]]
    )

    axs[0].set_ylim(-0.4, 0.4)

    axs[1].plot(chirp_filtered)

    for csl in chirp_slices:
        axs[0].plot([csl.start, csl.stop], [0, 0])

    # m = max(chirp_filtered)
    # a = np.argmax(chirp_filtered)
    # axs[2].scatter(range(len(chirp_filtered)), chirp_filtered)
    # axs[2].set_xlim(0.8 * a, 1.2 * a)
    # axs[2].set_ylim(0.95 * m, 1.05 * m)
    axs[0].set_ylabel("Transmitted waveform")

    axs[1].set_xlabel("Sample number")
    axs[1].set_ylabel("Matched filter output")

    for ax in axs:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text()] +
                ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)

    fig.savefig(f"segmented_chirp_frames_{protocol}")
