import math
import numpy as np


def f_v(x, params):
    a, mu, log_var = params
    return a * np.exp((np.cos(x - mu) - 1.) / math.exp(log_var))


def mse(y, ypred):
    return np.sum(np.power(y - ypred, 2.))


def mse_fv(params, xdata, ydata):
    return mse(ydata, f_v(xdata, params))


def integral_app(ydata):
    return np.sum(ydata)


def firing_rate(V_in, gain=0.1, max_rate=100):
    return max_rate * np.tanh(gain * np.maximum(V_in, 0))


def firing_rate_app(ydata):
    return integral_app(firing_rate(ydata))


def r_squared(ydata, ypred):
    return 1 - mse(ydata, ypred) / np.sum(np.power(ydata - np.mean(ydata), 2.))
