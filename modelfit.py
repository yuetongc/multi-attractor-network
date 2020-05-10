import math
import numpy as np


def f_v(x, params):
    a, mu, log_var = params
    return a * np.exp((np.cos((x - mu)) - 1.) / math.exp(log_var))


def f_v_baseline(x, params):
    a, mu, log_var, b = params
    return a * np.exp((np.cos((x - mu)) - 1.) / math.exp(log_var)) + b


def f_v_a(x, params, a):
    mu, log_var = params
    return a * np.exp((np.cos((x - mu)) - 1.) / math.exp(log_var))


def grad_a_mse(x, y, mu, log_var, b):
    exp_term = np.exp((np.cos((x - mu)) - 1.) / math.exp(log_var))
    return np.sum((y - exp_term - b) * exp_term)


def opt_a(x, y, mu, log_var, b):
    exp_term = np.exp((np.cos((x - mu)) - 1.) / math.exp(log_var))
    den = np.sum((y - b) * exp_term)
    nom = np.sum(exp_term * exp_term)
    return den / nom


def mse(ydata, ypred):
    return np.mean(np.power(ydata - ypred, 2.))


def mse_fv(params, xdata, ydata):
    return mse(ydata, f_v(xdata, params))


def mse_fv_baseline_a(params, xdata, ydata, a, b):
    return mse(ydata, f_v_a(xdata, params, a) + b)


def r_squared(ydata, ypred):
    return 1 - (mse(ydata, ypred) / np.var(ydata))


def integral_app(ydata):
    return np.sum(ydata)


def firing_rate(V_in, gain=0.1, max_rate=100):
    return max_rate * np.tanh(gain * np.maximum(V_in, 0))


def firing_rate_app(ydata):
    return integral_app(firing_rate(ydata))


def init_p(p, params, t1, t2):
    for n in np.arange(t1, t2, 1):
        p[:, n] = params
    return p


def update_p(m, p, x, t1, t2):
    t = np.arange(t1, t2, 1)
    for n in t:
        m[:, n] = f_v_baseline(x, p[:, n])
    return m


def circular_mean(data):
    return np.angle(np.sum(1 * np.exp(1j*data)))


def circular_variance(data):
    return 1 - (np.absolute(np.sum(1 * np.exp(1j*data))) / data.size)


def circular_precision(data):
    return np.absolute(np.mean(1 * np.exp(1j*data)))


def grad_a(x, a, mu, var, b):
    return np.exp((np.cos((x - mu)) - 1.) / var)


def grad_mu(x, a, mu, var, b):
    return (a / var) * np.exp((np.cos((x - mu)) - 1.) / var) * np.sin(x - mu)


def grad_var(x, a, mu, var, b):
    return -a * (np.cos(x - mu) - 1) * np.exp((np.cos((x - mu)) - 1.) / var) / np.square(var)


def grad_b(x, a, mu, var, b):
    return np.ones(a.shape)


def uniform_mu_integrand(x, a, mu, var, p):
    return (a**p) * np.exp(p * (np.cos((x - mu)) - 1.) / var) / (2 * math.pi)


def exp_response(x, params):
    a, tau = params
    return a * (1 - np.exp(-x / tau))


def exp_response_mse(params, xdata, ydata):
    return np.mean(np.power(ydata - exp_response(xdata, params), 2.))
