import numpy as np


def gaussian(x, params):
    mu, sig = params
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def f_v(x, params):
    a, mu, sig = params
    return a * np.exp((np.cos(x - mu) - 1.) / np.power(sig, 2.))


def mse_fv(params, xdata, ydata):
    return np.sum(np.power(ydata - f_v(xdata, params), 2.))
