import numpy as np
import bitarray

image = np.load("image.npy")
image = np.unpackbits(image)
received_bitstring = "".join(str(i) for i in image)

image_bits = bitarray.bitarray(received_bitstring)
image_bytes = image_bits.tobytes()

with open(f"asdf.tif", 'w+b') as f:
    f.write(image_bytes)