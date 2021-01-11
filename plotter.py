import os

os.environ["PATH"] = '/home/sepehr/texlive/2020/bin/x86_64-linux:' + os.environ["PATH"]  # DELETE IF NOT SEPEHR

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
use_latex = True
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
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


def plot_acc(train_file_paths, val_file_paths, names, file_name):
    train_data_list = []
    val_data_list = []
    min_epoch = 2000
    for file_path in train_file_paths:
        data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        train_data_list.append(data[:, 2])
        min_epoch = min(len(data), min_epoch)

    for file_path in val_file_paths:
        data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        val_data_list.append(data[:, 2])
        min_epoch = min(len(data), min_epoch)

    epoch_arr = np.arange(1, min_epoch + 1)
    for train_data, val_data, label in zip(train_data_list, val_data_list, names):
        plt.plot(epoch_arr, train_data[:min_epoch], '-o', label=f"{label} training")
        plt.plot(epoch_arr, val_data[:min_epoch], '-v', label=f"{label} validation")
    plt.ylabel(r"Top-5 accuracy (\%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", bbox_inches='tight', pad_inches=0.01)

def plot_loss(train_file_paths, val_file_paths, names, file_name):
    train_data_list = []
    val_data_list = []
    min_epoch = 2000
    for file_path in train_file_paths:
        data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        train_data_list.append(data[:, 2])
        min_epoch = min(len(data), min_epoch)

    for file_path in val_file_paths:
        data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        val_data_list.append(data[:, 2])
        min_epoch = min(len(data), min_epoch)

    epoch_arr = np.arange(1, min_epoch + 1)
    for train_data, val_data, label in zip(train_data_list, val_data_list, names):
        plt.plot(epoch_arr, train_data[:min_epoch], '-o', label=f"{label} training")
        plt.plot(epoch_arr, val_data[:min_epoch], '-v', label=f"{label} validation")
    plt.ylabel("Cross entropy loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", bbox_inches='tight', pad_inches=0.01)



if __name__ == '__main__':
    # train_paths = ["plotting_csv/attend_train_acc.csv", "plotting_csv/attend_glove_train_acc.csv"]
    # val_paths = ["plotting_csv/attend_val_acc.csv", "plotting_csv/attend_glove_val_acc.csv"]
    # names = ["GloVeless", "GloVe"]
    # file_name = "attend_acc"
    # plot_acc(train_paths, val_paths, names, file_name)
    # plt.show()

    train_paths = ["plotting_csv/attend_train_loss.csv", "plotting_csv/attend_glove_train_loss.csv"]
    val_paths = ["plotting_csv/attend_val_loss.csv", "plotting_csv/attend_glove_val_loss.csv"]
    names = ["GloVeless", "GloVe"]
    file_name = "attend_loss"
    plot_loss(train_paths, val_paths, names, file_name)
    plt.show()

