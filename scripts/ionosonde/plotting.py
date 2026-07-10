import matplotlib.pyplot as plt
import numpy as np

import sys, os

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde import params

save_to = "/scratch/mayas/ionosphere/plots/"

def plot1(freqs, corr):
    max_corr = np.max(np.abs(corr))
    x = np.arange(corr.shape[2]) / params.code_baudrate

    fig, axs = plt.subplots(1, 2, layout="constrained", figsize = (40, 10))

    for i in range(70, 40, -1):
        to_plot0 = np.abs(corr[i, 0, :])
        to_plot1 = np.abs(corr[i, 1, :])

        axs[0].plot(x - (2 * params.code_repeat_num + 1) * params.ipp * i, 3 * to_plot0/max_corr + i, label = f"{freqs[i]/1e6:.2f} MHz")
        axs[1].plot(x - (2 * params.code_repeat_num + 1) * params.ipp * i, 3 * to_plot1/max_corr + i, label = f"{freqs[i]/1e6:.2f} MHz")

    fig.savefig(save_to + "plot1.png", dpi = 600)

if __name__ == "__main__":
    file_name = "/scratch/mayas/ionosphere/output/iono_corr_2pols_1746818097_to_1746818112.npz"

    print("Plotting")

    data = np.load(file_name)

    plot1(data["freqs"], data["corr"])