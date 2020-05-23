import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelfit

import os
"path for simulation data"
os.chdir('/Users/yuetongyc/Desktop/Cambridge/IIB/Project/data')


rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False,
      "lines.linewidth": 2,
      "xtick.labelsize": 24,
      "ytick.labelsize": 24,
      'legend.fontsize': 24,
      "axes.labelsize": 28,
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

circular_mu_mean = np.apply_along_axis(modelfit.circular_mean, 0, mu)
circular_mu_var = np.apply_along_axis(modelfit.circular_variance, 0, mu)

a_var = np.var(a, axis=0)
mu_var = np.var(mu, axis=0)
var_var = np.var(var, axis=0)
b_var = np.var(b, axis=0)

v_variance = np.mean(v_var, axis=0)
v_variability = np.sqrt(v_variance)


N_trial = a.shape[0]
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

v_variance_estimate_mtx = a_term + mu_term + var_term + b_term + \
                             a_mu_term + a_var_term + a_b_term + mu_var_term + mu_b_term + var_b_term

v_variance_estimate = np.mean(v_variance_estimate_mtx, axis=0)
v_variability_estimate = np.sqrt(v_variance_estimate)


a_term_avg = np.mean(a_term, axis=0)
mu_term_avg = np.mean(mu_term, axis=0)
var_term_avg = np.mean(var_term, axis=0)
b_term_avg = np.mean(b_term, axis=0)
a_mu_term_avg = np.mean(a_mu_term, axis=0)
a_var_term_avg = np.mean(a_var_term, axis=0)
a_b_term_avg = np.mean(a_b_term, axis=0)
mu_var_term_avg = np.mean(mu_var_term, axis=0)
mu_b_term_avg = np.mean(mu_b_term, axis=0)
var_b_term_avg = np.mean(var_b_term, axis=0)

exp_term = np.exp((np.cos(x_mtx - np.mean(mu_mean_mtx)) - 1) / var_mean_mtx)
sin_term = np.sin(x_mtx - mu_mean_mtx)
mu_grad_estimate = a_mean_mtx * exp_term * sin_term / var_mean_mtx

parameter_pairs = [a_mu_term_avg, a_var_term_avg, a_b_term_avg, mu_var_term_avg, mu_b_term_avg, var_b_term_avg]
pair_labels = [r'$Cov[\hat{a}, \hat{\mu}]}$', r'$Cov[\hat{a}, \hat{\sigma^{2}}]$', r'$Cov[\hat{a}, \hat{b}]$',
               r'$Cov[\hat{\mu}, \hat{\sigma^{2}}]$', r'$Cov[\hat{\mu}, \hat{b}]$', r'$Cov[\hat{\sigma^{2}}, \hat{b}]$']


fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
for n in range(N_trial):
    ax1.plot(t_tick, a[n], color='black')
    ax2.plot(t_tick, np.rad2deg(mu[n]), color='black')
    ax3.plot(t_tick, var[n], color='black')
    ax4.plot(t_tick, b[n], color='black')
ax1.set_ylabel(r'$\hat{a}$ [mV]')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_ylabel(r'$\hat{\mu}\ [\degree]$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_yticks([-180, 0, 180])
ax3.set_ylabel(r'$\hat{w}^{2}$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_ylabel(r'$\hat{b}$ [mV]')
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_xlabel('t [ms]')
plt.tight_layout()
fig1.savefig('parameter_stats')

fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
ax1.plot(t_tick, a_mean, color='black')
ax1.set_ylabel(r'$\bar{a}$ [mV]')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, np.rad2deg(circular_mu_mean), color='black')
ax2.set_ylabel(r'$\bar{\mu}\ [\degree]$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_yticks([-180, 0, 180])
ax3.plot(t_tick, var_mean, color='black')
ax3.set_ylabel(r'$\bar{w}^{2}$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.set_yticks([1.8, 1.9, 2])
ax4.plot(t_tick, b_mean, color='black')
ax4.set_ylabel(r'$\bar{b}$ [mV]')
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_xlabel('t [ms]')
plt.tight_layout()
fig2.savefig('parameter_mean')

fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
ax1.plot(t_tick, a_var, color='black')
ax1.set_ylabel(r'$\sigma^{2}_{a}$')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, circular_mu_var, color='black')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.plot(t_tick, var_var, color='black')
ax3.set_ylabel(r'$\sigma^{2}_{w^{2}}$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.plot(t_tick, b_var, color='black')
ax4.set_ylabel(r'$\sigma^{2}_{b}$')
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_xlabel('t [ms]')
plt.tight_layout()
fig3.savefig('parameter_var')

fig5, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
bottom, top = 0.1, 0.9
left, right = 0.1, 0.85
fig5.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
ax1.plot(t_tick, np.mean(np.square(mu_grad_estimate), axis=0), color='black')
ax1.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \rangle_{i}$')
ax1.set_ylim(0, 3.5)
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, np.mean(np.square(a_mean_mtx), axis=0), color='black')
ax2.set_ylabel(r'$\bar{a}(t)^{2}$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.plot(t_tick, np.mean(np.square(exp_term), axis=0), color='black')
ax3.set_ylabel(r'$\langle (\exp(\frac{\sin(\theta - \bar{\mu}(t)) - 1}{\bar{w}^{2}}))^{2} \rangle_{i}$')
ax3.set_ylim(0.4, 0.5)
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.plot(t_tick, np.mean(np.square(sin_term), axis=0), color='black')
ax4.set_ylabel(r'$\langle \sin^{2}(\theta - \bar{\mu}(t)) \rangle_{i}$')
ax4.set_ylim(0.3, 0.7)
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_xlabel('t [ms]')
plt.tight_layout()
fig5.savefig('Mu Parameter Coefficient')
