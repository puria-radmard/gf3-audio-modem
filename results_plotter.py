import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

results = {0: 0.0485,
0.02: 0.0272,
0.05: 0.0228,
0.1	: 0.0215,
0.2	: 0.0241,
0.5	: 0.0347,
0.7	: 0.038,
0.9	: 0.03207}

fig, axs = plt.subplots(1, figsize = (10, 7))
axs.plot(results.keys(), results.values(), linewidth = 3, marker = 'o', markersize = 10)
axs.set_xlabel("$\\alpha$")
axs.set_ylabel("Average BER")

for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
    axs.get_xticklabels() + axs.get_yticklabels()):
    item.set_fontsize(20)


fig.savefig("alpha")

