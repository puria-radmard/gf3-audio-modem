import math
from sys import meta_path
from PIL import Image
import numpy as np
import io
import random

fs = 44100

CONSTELLATIONS_DICT = {
    "gray": {(0, 0): +1 + 1j, (0, 1): -1 + 1j, (1, 1): -1 - 1j, (1, 0): +1 - 1j}
}


def shift_slice(sl, idx):
    return slice(sl.start + idx, sl.stop + idx)


def s_to_bitlist(s):
    ords = (ord(c) for c in s)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    return [(o >> shift) & 1 for o in ords for shift in shifts]


def bitlist_to_chars(bl):
    bi = iter(bl)
    bytes = zip(*(bi,) * 8)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    for byte in bytes:
        yield chr(sum(bit << s for bit, s in zip(byte, shifts)))


def bitlist_to_s(bl):
    return "".join(bitlist_to_chars(bl))

def OLD_metadata(filetype, a, b, c):
    sep = [0] * 8
    filetype_ = s_to_bitlist(filetype)
    a_ = s_to_bitlist(str(a))
    b_ = s_to_bitlist(str(b))
    c_ = s_to_bitlist(str(c))
    meta = filetype_ + sep + a_ + sep + b_ + sep + c_ + sep
    return meta

def create_metadata(inputfile_name, input_bits):
    # here input_bits is the uncoded bits, if length of it is used, delete number_of_bits line
    file_type = inputfile_name.split('.')[-1]
    if file_type == 'wav':
        typedata = [1,0,1,0,0,1,0,1]
    elif file_type == 'tif':
        typedata =  [0,0,1,1,0,0,1,0]
    else: 
        # txt file
        typedata =  [0,1,0,0,1,0,0,0]
    number_of_bits = len(input_bits)
    file_size = bin(number_of_bits)[2:]
    pad = int(32-len(file_size))
    pad_zeros = np.zeros(pad)
    size_array = np.array(list(file_size), dtype = int)
    length_data = np.concatenate((pad_zeros, size_array)).tolist()
    type_data = np.array(typedata).tolist()
    meta_data = np.concatenate((type_data*5, length_data*5))
    return meta_data

def translate_tiff(tiff_name, num_reps):
    filetype = 'tif'
    image = Image.open(f'{tiff_name}.tif')
    data = np.asarray(image)
    image_bits = np.unpackbits(data)
    meta_data = create_metadata(filetype, image_bits)
    transmitted_bits = np.ravel(np.concatenate([meta_data, image_bits]))
    #transmitted_bits = np.ravel(np.concatenate((meta_data, meta_data, meta_data, image_bits)))
    transmitted_bits = "".join(str(int(t)) for t in transmitted_bits)
    return transmitted_bits


def translate_file(fname, ftype, num_md_raps):
    with open(f"{fname}.{ftype}", 'rb') as f:
        info_bytes = f.read()
    bit_string = ''
    for i in info_bytes:
        binary_string = '{0:08b}'.format(i)
        bit_string += binary_string
    file_bits = np.array([int(b) for b in bit_string])
    meta_data = create_metadata(ftype, file_bits)
    transmitted_bits = np.ravel(np.concatenate([meta_data, file_bits]))
    transmitted_bits = "".join(str(int(t)) for t in transmitted_bits)
    return transmitted_bits


def exponential_chirp(t, f0, f1, t1):
    r = f1 / f0
    window_strength = 50
    factor = 1/3
    return (
        factor * math.sin(2 * math.pi * t1 * f0 * ((r ** (t / t1) - 1) / (math.log(r, math.e))))
        * (1 - math.e ** (-window_strength * t))
        * (1 - math.e ** (window_strength * (t - t1)))
    )


def from_the_box_chirp(t, f0, f1, t1):
    return chirp(t, f0=f0, f1=f1, t1=t1, method="logarithmic")


def publish_random_bits(file_name, binary_length):
    text_bits = "{0:b}".format(random.getrandbits(binary_length))
    with open(f"{file_name}.txt", "w") as f:
        f.write(text_bits)
    print(f"random bits written to {file_name}")


def get_OFDM_data_from_bits(modulator, source_bits, ignore_aug):
    with open(f"{source_bits}.txt") as f:
        text_bits = f.read()
    OFDM_data = modulator.data2OFDM(
        bitstring=text_bits, return_frames=True, ignore_aug = ignore_aug
    )
    return OFDM_data


def publish_random_OFDM_sound(modulator, sound_file_name, source_bits):
    with open(f"{source_bits}.txt") as f:
        text_bits = f.read()
    asdf = modulator.data2OFDM(
        bitstring=text_bits, return_frames=True
    )
    print(len(asdf))
    modulator.publish_data(OFDM_transmission, sound_file_name)
    print(f"random bits audio written to {sound_file_name}")


def process_metadata(decoded_metadata):
    # Here decoded_metadata means the first 168 (with some errors), assuming it is nparray
    type_data_ = decoded_metadata[:40]
    typedatamajority = np.zeros(8)
    for i in range(len(type_data_)):
        m = i % 8
        if int(type_data_[i]) == 1:
            typedatamajority[m] += 1
    type_data = []
    for i in typedatamajority:
        if i > 2:
            type_data.append(1)
        else:
            type_data.append(0)
    dis_wav = 8 - np.sum(np.array(type_data) == np.array([1,0,1,0,0,1,0,1]))
    dis_tif = 8 - np.sum(np.array(type_data) == np.array([0,0,1,1,0,0,1,0]))
    dis_txt = 8 - np.sum(np.array(type_data) == np.array([0,1,0,0,1,0,0,0]))
    if dis_wav < dis_tif:
        if dis_wav < dis_txt:
            file_type = 'wav'
        elif dis_wav > dis_txt:
            file_type = 'txt'
        else:
            print('Error: Do not know what file (wav/ txt) this is.')
    elif dis_wav > dis_tif:
        if dis_tif < dis_txt:
            file_type = 'tif'
        elif dis_tif > dis_txt:
            file_type = 'txt'
        else:
            print('Error: Do not know what file (tif/ txt) this is.')
    else:
        if dis_wav > dis_txt:
            file_type = 'txt'
        elif dis_wav < dis_txt:
            print('Error: Do not know what file (wav/ tif) this is.')
        else:
            print('Error: Do not know what file (wav/ tif/ txt) this is.')
    length_data = decoded_metadata[40:]
    # single_length_data = []
    number_of_ones = np.zeros(32)
    for i in range(len(length_data)):
        m = i % 32
        if int(length_data[i]) == 1:
            number_of_ones[m] += 1
    decoded_length_data = []
    for i in number_of_ones:
        if i > 2:
            decoded_length_data.append(1)
        else:
            decoded_length_data.append(0)
    bin_str = '0b'+''.join(str(e) for e in decoded_length_data)
    length = int(bin_str,2)

    return {
        "filetype": file_type,
        "total_data_bits": length
    }



def OLD_process_metadata(metadata_bits, num_reps):

    # Inputs:
    # metadata_bits is all the bits for the metadata at the start of the signal.
            # This is all the reps (i.e. 360 in our case)
    # num_reps is how many repitions we know that metadata_bits contains, (i.e. 3 in our case)
    
    # Split all the repitions into the known number of repititions e.g. 360bits --> [120bits, 120bits, 120bits]
    metadata_bits = np.array([int(m) for m in metadata_bits])
    metadata_bits_reps = np.split(np.array(metadata_bits), num_reps)
    # Average over all reptitions and round - this is like majority vote in the repition code
    voted_metadata_bits = np.mean(metadata_bits_reps, axis = 0).round().astype(int)
    # Split the average voted bits into bytes of 8
    metadata_bytes = np.split(voted_metadata_bits, len(voted_metadata_bits)/8)
    
    # ADD STUFF HERE - inputting known values from mario.tiff

    a = 150
    b = 150
    c = 4
    return {
        "filetype": "tif",
        "dim1": a,
        "dim2": b,
        "dim3": c,
        "total_data_bits": a*b*c*8
    }
