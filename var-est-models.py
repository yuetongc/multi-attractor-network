import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import modelfit

import os
"path for simulation data"
os.chdir('/Users/yuetongyc/Desktop/Cambridge/IIB/Project/data')


rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False,
      "lines.linewidth": 2,
      "xtick.labelsize": 20,
      "ytick.labelsize": 20,
      'legend.fontsize': 18,
      "axes.labelsize": 24,
      }
plt.rcParams.update(rc)


a_df = pd.read_csv('a_tau_10_1.csv', index_col=False)
mu_df = pd.read_csv('mu_tau_10_1.csv', index_col=False)
var_df = pd.read_csv('var_tau_10_1.csv', index_col=False)
b_df = pd.read_csv('b_tau_10_1.csv', index_col=False)

a = np.transpose(a_df.values)
mu = np.transpose(mu_df.values)
var = np.transpose(var_df.values)
b = np.transpose(b_df.values)

v_var_df = pd.read_csv('v_var_tau_10_1.csv', index_col=False)
v_var = v_var_df.values


a_mean = np.mean(a, axis=0)
mu_mean = np.mean(mu, axis=0)
var_mean = np.mean(var, axis=0)
b_mean = np.mean(b, axis=0)

a_var = np.var(a, axis=0)
shifted_mu = mu - modelfit.circular_mean(mu)
mu_var = np.var(shifted_mu, axis=0)
var_var = np.var(var, axis=0)
b_var = np.var(b, axis=0)

v_var_mean = np.mean(v_var, axis=0)


N_trial = 24
N_neuron = 100
N_point = 23000
x = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
t_tick = np.arange(0, t3, 0.1)


x_mtx = np.reshape(np.repeat(x, N_point), (N_neuron, N_point))


a_mean_mtx = np.reshape(np.repeat(a_mean, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx = np.reshape(np.repeat(mu_mean, N_neuron), (N_point, N_neuron)).T
var_mean_mtx = np.reshape(np.repeat(var_mean, N_neuron), (N_point, N_neuron)).T
b_mean_mtx = np.reshape(np.repeat(b_mean, N_neuron), (N_point, N_neuron)).T

a_var_mtx = np.reshape(np.repeat(a_var, N_neuron), (N_point, N_neuron)).T
mu_var_mtx = np.reshape(np.repeat(mu_var, N_neuron), (N_point, N_neuron)).T
var_var_mtx = np.reshape(np.repeat(var_var, N_neuron), (N_point, N_neuron)).T
b_var_mtx = np.reshape(np.repeat(b_var, N_neuron), (N_point, N_neuron)).T


"Assumption 1"
a_mean_trial_mtx = np.repeat(a_mean_mtx.T[:, :, np.newaxis], N_trial, axis=2).T
mu_trial_mtx = np.repeat(mu[:, :, np.newaxis], N_neuron, axis=2).swapaxes(1, 2)
var_mean_trial_mtx = np.repeat(var_mean_mtx.T[:, :, np.newaxis], N_trial, axis=2).T
b_mean_trial_mtx = np.repeat(b_mean_mtx.T[:, :, np.newaxis], N_trial, axis=2).T

mu_v_est = a_mean_trial_mtx * np.exp((np.cos(x_mtx - mu_trial_mtx) - 1) / var_mean_trial_mtx) + b_mean_trial_mtx
mu_var_est = np.var(mu_v_est, axis=0)
mu_variability_est = np.mean(mu_var_est, axis=0)


"Assumption 2a"
a_grad = modelfit.grad_a(x_mtx, a_mean_mtx, mu_mean_mtx, var_mean_mtx, b_mean_mtx)
mu_grad = modelfit.grad_mu(x_mtx, a_mean_mtx, mu_mean_mtx, var_mean_mtx, b_mean_mtx)
var_grad = modelfit.grad_var(x_mtx, a_mean_mtx, mu_mean_mtx, var_mean_mtx, b_mean_mtx)
b_grad = modelfit.grad_b(x_mtx, a_mean_mtx, mu_mean_mtx, var_mean_mtx, b_mean_mtx)

a_mu_cov = np.zeros(N_point)
a_var_cov = np.zeros(N_point)
a_b_cov = np.zeros(N_point)
mu_var_cov = np.zeros(N_point)
mu_b_cov = np.zeros(N_point)
var_b_cov = np.zeros(N_point)

for i in range(N_point):
    a_mu_cov[i] = np.cov(np.stack((a[:, i], mu[:, i]), axis=0))[0][1]
    a_var_cov[i] = np.cov(np.stack((a[:, i], var[:, i]), axis=0))[0][1]
    a_b_cov[i] = np.cov(np.stack((a[:, i], b[:, i]), axis=0))[0][1]
    mu_var_cov[i] = np.cov(np.stack((mu[:, i], var[:, i]), axis=0))[0][1]
    mu_b_cov[i] = np.cov(np.stack((mu[:, i], b[:, i]), axis=0))[0][1]
    var_b_cov[i] = np.cov(np.stack((var[:, i], b[:, i]), axis=0))[0][1]

a_mu_cov_mtx = np.reshape(np.repeat(a_mu_cov, N_neuron), (N_point, N_neuron)).T
a_var_cov_mtx = np.reshape(np.repeat(a_var_cov, N_neuron), (N_point, N_neuron)).T
a_b_cov_mtx = np.reshape(np.repeat(a_b_cov, N_neuron), (N_point, N_neuron)).T
mu_var_cov_mtx = np.reshape(np.repeat(mu_var_cov, N_neuron), (N_point, N_neuron)).T
mu_b_cov_mtx = np.reshape(np.repeat(mu_b_cov, N_neuron), (N_point, N_neuron)).T
var_b_cov_mtx = np.reshape(np.repeat(var_b_cov, N_neuron), (N_point, N_neuron)).T

a_term = np.square(a_grad) * a_var_mtx
mu_term = np.square(mu_grad) * mu_var_mtx
var_term = np.square(var_grad) * var_var_mtx
b_term = np.square(b_grad) * b_var_mtx

a_mu_term = 2 * a_grad * mu_grad * a_mu_cov_mtx
a_var_term = 2 * a_grad * var_grad * a_var_cov_mtx
a_b_term = 2 * a_grad * b_grad * a_b_cov_mtx
mu_var_term = 2 * mu_grad * var_grad * mu_var_cov_mtx
mu_b_term = 2 * mu_grad * b_grad * mu_b_cov_mtx
var_b_term = 2 * var_grad * b_grad * var_b_cov_mtx

first_order_var_est_mtx = a_term + mu_term + var_term + b_term + \
                        a_mu_term + a_var_term + a_b_term + mu_var_term + mu_b_term + var_b_term
first_order_var_est = np.mean(first_order_var_est_mtx, axis=0)


"Assumption 1 + 2a"
mu_term_var_est = np.mean(mu_term, axis=0)


"Assumption 1 + Assumption 2b"
uniform_mu_var_est_df = pd.read_csv('uniform_mu_var_est.csv', index_col=False)
uniform_mu_var_est = np.mean(uniform_mu_var_est_df.values, axis=0)


"Assumption 1 + Assumption 2a + Assumption 3"
var_all_mean = np.mean(var)
b_all_mean = np.mean(b)
var_mtx_all_mean = np.reshape(var_all_mean * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean = np.reshape(b_all_mean * np.ones(N_point*N_neuron), (N_neuron, N_point))

mu_grad_params_const = modelfit.grad_mu(x_mtx, a_mean_mtx, mu_mean_mtx, var_mtx_all_mean, b_mtx_all_mean)
mu_term_params_const = np.square(mu_grad_params_const) * mu_var_mtx
mu_term_var_params_const_est = np.mean(mu_term_params_const, axis=0)


"Assumption 1 + Assumption 2b + Assumption 3"
uniform_mu_var_params_const_est_df = pd.read_csv('uniform_mu_var_params_const_est_df.csv', index_col=False)
uniform_mu_var_params_const_est = np.mean(uniform_mu_var_params_const_est_df.values, axis=0)


fig1, ax = plt.subplots(figsize=(16, 5.5))
ax.plot(t_tick, v_var_mean, color='dimgrey', label='true')
ax.plot(t_tick, mu_variability_est, color='firebrick', label='assumption 1')
ax.plot(t_tick, first_order_var_est, color='orange', label='assumption 2a')
ax.plot(t_tick, mu_term_var_params_const_est, color='limegreen', label='assumption 1+2a+3')
ax.plot(t_tick, uniform_mu_var_params_const_est, color='royalblue', label='assumption 1+2b+3')
ax.plot(t_tick, mu_term_var_est, color='seagreen', linewidth=2, linestyle='dashed', label='assumption 1+2a')
ax.plot(t_tick, uniform_mu_var_est, color='mediumblue', linewidth=2, linestyle='dashed', label='assumption 1+2b')
ax.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax.set_ylim(0, 3.2)
ax.set_ylabel(r'${\hat{\sigma}^{2}}$')
ax.set_xlabel('t [ms]')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1], handles[2], handles[5], handles[3], handles[6], handles[4]]
labels = [labels[0], labels[1], labels[2], labels[5], labels[3], labels[6], labels[4]]
ax.legend(handles, labels, frameon=False, loc='upper right')
plt.tight_layout()
plt.show()
fig1.savefig('var_reconstruction')
