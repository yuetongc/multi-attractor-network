import numpy as np


def f_v(x, params):
    a, mu, sig = params
    return a * np.exp((np.cos(x - mu) - 1.) / np.power(sig, 2.))


def mse_fv(params, xdata, ydata):
    return np.sum(np.power(ydata - f_v(xdata, params), 2.))


def integral_app(ydata):
    return np.sum(ydata)


def firing_rate(V_in, gain=0.1, max_rate=100):
    return max_rate * np.tanh(gain * np.maximum(V_in, 0))


def firing_rate_app(ydata):
    return integral_app(firing_rate(ydata))
