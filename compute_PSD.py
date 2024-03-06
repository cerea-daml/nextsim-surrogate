#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf

import scipy.stats as stats

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


import xarray as xr
from tqdm import trange
import tensorflow_addons as tfa
import matplotlib.pyplot as plt



def time_analysis(file, T, N_pix, N_cycle, N, N_start=5, type_="pred", surr_time=1):

    score_tot = np.zeros(T)
    coef_tot = np.zeros(T)
    y_tot = np.zeros((T, N_pix // 2))
    pred_tot = np.zeros((T, N_pix // 2 - N_start, 1))
    for t in range(T):
        image = np.load(
            file + "cycle_" + str(N_cycle) + "_" + type_ + "_" + str(N) + ".npy"
        )

        x, y = create_power_spectra(image[t], N_pix)

        x_start, pred, coef, score = compute_regression(
            np.log(x), np.log(y), N_start=N_start
        )

        score_tot[t] = score
        coef_tot[t] = coef
        y_tot[t] = y
        pred_tot[t] = pred
    return score_tot, coef_tot, y_tot, pred_tot


def create_power_spectra(image, N_pix):
    npix = N_pix

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(npix) * npix

    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    return kvals, Abins


def compute_regression(x, y, N_start):
    x = x[N_start:].reshape(-1, 1)
    y = y[N_start:].reshape(-1, 1)
    reg = linear_model.LinearRegression()
    model = reg.fit(x, y)

    pred = reg.predict(x)
    score = r2_score(y, pred)
    return x, pred, reg.coef_, score


def compute_PSD(file_, T, N_cycle, N_pix=256):
    T = T
    N_it = np.arange(N_cycle)
    PSD_score_tot_surr = np.zeros((len(N_it), T))
    PSD_coef_tot_surr = np.zeros((len(N_it), T))
    PSD_fit_surr = np.zeros(((len(N_it), T, N_pix // 2 - 5)))
    PSD_tot_surr = np.zeros(((len(N_it), T, N_pix // 2)))

    PSD_score_tot_truth = np.zeros((len(N_it), T))
    PSD_coef_tot_truth = np.zeros((len(N_it), T))
    PSD_fit_truth = np.zeros(((len(N_it), T, N_pix // 2 - 5)))
    PSD_tot_truth = np.zeros(((len(N_it), T, N_pix // 2)))

    for n in range(N_cycle):
        print(n)
        score_tot_surr, coef_tot_surr, y_tot_surr, fit_PSD_tot_surr = time_analysis(
            file_,
            N_cycle=N_cycle,
            N=n,
            T=T,
            N_pix=N_pix,
            N_start=5,
            type_="pred",
            surr_time=61,
        )
        score_tot_truth, coef_tot_truth, y_tot_truth, fit_PSD_tot_truth = time_analysis(
            file_,
            N_cycle=N_cycle,
            N=n,
            T=T,
            N_pix=N_pix,
            N_start=5,
            type_="truth",
            surr_time=61,
        )

        PSD_score_tot_surr[n] = score_tot_surr
        PSD_coef_tot_surr[n] = coef_tot_surr
        PSD_fit_surr[n] = fit_PSD_tot_surr.squeeze()
        PSD_tot_surr[n] = y_tot_surr

        PSD_score_tot_truth[n] = score_tot_truth
        PSD_coef_tot_truth[n] = coef_tot_truth
        PSD_fit_truth[n] = fit_PSD_tot_truth.squeeze()
        PSD_tot_truth[n] = y_tot_truth

    return (
        PSD_score_tot_surr,
        PSD_coef_tot_surr,
        PSD_fit_surr,
        PSD_tot_surr,
        PSD_score_tot_truth,
        PSD_coef_tot_truth,
        PSD_fit_truth,
        PSD_tot_truth,
    )


path = "Results/17Fev23/lambda100/UNet2_1input_1output/"
N_cycle = 1
T = 720



(
    PSD_score_tot_surr,
    PSD_coef_tot_surr,
    PSD_fit_surr,
    PSD_tot_surr,
    PSD_score_tot_truth,
    PSD_coef_tot_truth,
    PSD_fit_truth,
    PSD_tot_truth,
) = compute_PSD(path, T=T, N_cycle=N_cycle, N_pix=256)

np.save(path + "cycle_" + str(N_cycle) + "_PSD_score_tot_surr.npy", PSD_score_tot_surr)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_coef_tot_surr.npy", PSD_coef_tot_surr)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_fit_surr.npy", PSD_fit_surr)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_tot_surr.npy", PSD_tot_surr)


np.save(
    path + "cycle_" + str(N_cycle) + "_PSD_score_tot_truth.npy", PSD_score_tot_truth
)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_coef_tot_truth.npy", PSD_coef_tot_truth)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_fit_truth.npy", PSD_fit_truth)
np.save(path + "cycle_" + str(N_cycle) + "_PSD_tot_truth.npy", PSD_tot_truth)
