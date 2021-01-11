# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 04:23:50 2021

@author: Tolga
"""


import os

#os.environ["PATH"] = '/home/sepehr/texlive/2020/bin/x86_64-linux:' + os.environ["PATH"]  # DELETE IF NOT SEPEHR

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
use_latex = True
tex_fonts = {
#     # Use LaTeX to write all text  
#     "text.usetex": True,           
     "font.family": "serif"
 }
if use_latex:
    plt.rcParams.update(tex_fonts)


def plot_val_bleu(file_paths, names, file_name):
    data_list = []
    min_epoch = 2000
    for file_path in file_paths:
        data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        data_list.append(data[:, 2])
        min_epoch = min(len(data), min_epoch)

    epoch_arr = np.arange(1, min_epoch + 1)
    for data, label in zip(data_list, names):
        plt.plot(epoch_arr, data[:min_epoch], '-o', label=label)
    plt.ylabel("Validation BLEU-4 score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    file_paths = ["plotting_csv/val_bleu.csv"]
    names = ["Attention model", "Attention model with GloVe"]
    file_name = "bleu4"
    plot_val_bleu(file_paths, names, file_name)
    plt.show()