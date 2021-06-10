import matplotlib.pyplot as plt
from os import read
import sys
from util_objects import read_to_binary

real_file = sys.argv[1]
transmitted_file = sys.argv[2]

_, _, real_file_bits = read_to_binary(real_file)
_, _, transmitted_file_bits = read_to_binary(transmitted_file)

assert len(transmitted_file_bits) == len(real_file_bits)

wrong = 0
cumulative_BER = []

for i, t_bit in enumerate(transmitted_file_bits):
    c = 1 - int(t_bit == real_file_bits[i])
    wrong += c
    cumulative_BER.append(wrong / (i + 1))
plt.plot(cumulative_BER)
plt.savefig("BER_cumulative.png")

error_rate = 1 - sum([transmitted_file_bits[i] == real_file_bits[i] for i in range(len(transmitted_file_bits))])/len(transmitted_file_bits)

print(error_rate)