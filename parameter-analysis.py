import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelfit


import os
os.chdir('/Users/yuetongyc/PycharmProjects/multi-attractor-model/data')


a_df = pd.read_csv('a_tau_20.csv', index_col=False)
mu_df = pd.read_csv('mu_tau_20.csv', index_col=False)
var_df = pd.read_csv('var_tau_20.csv', index_col=False)
b_df = pd.read_csv('b_tau_20.csv', index_col=False)

a = np.transpose(a_df.values)
mu = np.transpose(mu_df.values)
var = np.transpose(var_df.values)
b = np.transpose(b_df.values)

v_var_df = pd.read_csv('v_var_tau_20.csv', index_col=False)
v_var = v_var_df.values


a_mean = np.mean(a, axis=0)
mu_mean = np.mean(mu, axis=0)
var_mean = np.mean(var, axis=0)
b_mean = np.mean(b, axis=0)

a_var = np.var(a, axis=0)
mu_var = np.var(mu, axis=0)
var_var = np.var(var, axis=0)
b_var = np.var(b, axis=0)

v_variance = np.mean(v_var, axis=0)
v_variability = np.sqrt(v_variance)


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

exp_term = np.exp((np.cos(x_mtx - mu_mean_mtx) - 1) / var_mean_mtx)
sin_term = np.sin(x_mtx - mu_mean_mtx)
mu_grad_estimate = a_mean_mtx * exp_term * sin_term / var_mean_mtx


font = {'size': 18}
plt.rc('font', **font)

rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False}
plt.rcParams.update(rc)


parameter_pairs = [a_mu_term_avg, a_var_term_avg, a_b_term_avg, mu_var_term_avg, mu_b_term_avg, var_b_term_avg]
pair_labels = [r'$Cov[\hat{a}, \hat{\mu}]}$', r'$Cov[\hat{a}, \hat{\sigma^{2}}]$', r'$Cov[\hat{a}, \hat{b}]$',
               r'$Cov[\hat{\mu}, \hat{\sigma^{2}}]$', r'$Cov[\hat{\mu}, \hat{b}]$', r'$Cov[\hat{\sigma^{2}}, \hat{b}]$']

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
bottom, top = 0.1, 0.9
left, right = 0.1, 0.85
fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
ax1.plot(t_tick, v_variance, color='dimgrey', label='true')
ax1.plot(t_tick, v_variance_estimate, color='orange', label='estimate')
ax1.set_ylabel(r'$\langle{Var_{trials}[V] \rangle }_{neurons}$')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax1.set_ylim(-1, 5)
ax1.legend()
ax2.plot(t_tick, a_term_avg, label=r'$Var[\hat{a}]}$', color='firebrick')
ax2.plot(t_tick, mu_term_avg, label=r'$Var[\hat{\mu}]}$', color='mediumblue')
ax2.plot(t_tick, var_term_avg, label=r'$Var[\hat{\sigma^{2}}]}$', color='darkgreen')
ax2.plot(t_tick, b_term_avg, label=r'$Var[\hat{b}]}$', color='sandybrown')
for i in range(len(parameter_pairs)):
    "if not np.all(np.abs(parameter_pairs[i]) < 0.01):"
    ax2.plot(t_tick, parameter_pairs[i], label=pair_labels[i])
ax2.set_ylabel(r'$\alpha_{i}^{2}w_{i}$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_ylim(-1, 5)
ax2.legend(loc='upper right', ncol=2, fontsize=12)
ax3.plot(t_tick, v_variance_estimate, color='orange', label='variance estimate')
ax3.plot(t_tick, mu_term_avg, color='mediumblue', label=r'$\alpha_{\mu}^{2}w_{\mu}$')
ax3.legend(loc='upper right')
ax3.set_ylabel(r'$\langle{Var_{trials}[V] \rangle }$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.set_ylim(-1, 5)
ax3.set_xlabel('t / ms')
plt.tight_layout()
fig1.savefig('Variance Estimation')

fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
ax1.plot(t_tick, a_var, color='black')
ax1.set_ylabel(r'$Var_{trail}[\hat{a}]}$')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, mu_var, color='black')
ax2.set_ylabel(r'$Var_{trail}[\hat{\mu}]}$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.plot(t_tick, var_var, color='black')
ax3.set_ylabel(r'$Var_{trail}[\hat{\sigma^{2}}]}$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.plot(t_tick, b_var, color='black')
ax4.set_ylabel(r'$Var_{trail}[\hat{b}]}$')
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_xlabel('t / ms')
plt.tight_layout()
fig2.savefig('Parameter Variance')

fig3, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 18))
ax1.plot(t_tick, a_mu_cov, color='black')
ax1.set_ylabel(r'$Cov[\hat{a}, \hat{\mu}]}$')
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, a_var_cov, color='black')
ax2.set_ylabel(r'$Cov[\hat{a}, \hat{\sigma^{2}}]$')
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.plot(t_tick, a_b_cov, color='black')
ax3.set_ylabel(r'$Cov[\hat{a}, \hat{b}]$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.plot(t_tick, mu_var_cov, color='black')
ax4.set_ylabel(r'$Cov[\hat{\mu}, \hat{\sigma^{2}}]$')
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.plot(t_tick, mu_b_cov, color='black')
ax5.set_ylabel(r'$Cov[\hat{\mu}, \hat{b}]$')
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax6.plot(t_tick, var_b_cov, color='black')
ax6.set_ylabel(r'$Cov[\hat{\sigma^{2}}, \hat{b}]$')
ax6.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax6.set_xlabel('t / ms')
plt.tight_layout()
fig3.savefig('Parameter Covariance')

fig4, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(t_tick, np.mean(np.square(a_grad), axis=0), color='firebrick', label='a')
ax.plot(t_tick, np.mean(np.square(mu_grad), axis=0), color='mediumblue', label=r'$\mu$')
ax.plot(t_tick, np.mean(np.square(var_grad), axis=0), color='darkgreen', label=r'$\sigma^{2}$')
ax.plot(t_tick, np.mean(np.square(b_grad), axis=0), color='sandybrown', label='b')
ax.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax.set_ylabel(r'$\langle \alpha_{w}^{2} \rangle$')
ax.set_xlabel('t / ms')
ax.set_ylim(0, 3.5)
ax.legend(loc='upper right', ncol=2)
plt.tight_layout()
fig4.savefig('Parameter Coefficient')

fig5, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))
bottom, top = 0.1, 0.9
left, right = 0.1, 0.85
fig5.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
ax1.plot(t_tick, np.mean(np.square(mu_grad_estimate), axis=0), color='mediumblue')
ax1.set_ylabel(r'$\langle \alpha_{\mu}^{2} \rangle$')
ax1.set_ylim(0, 3.5)
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.plot(t_tick, np.mean(np.square(a_mean_mtx), axis=0), color='black')
ax2.set_ylabel(r'$\langle \bar{a}^{2} \rangle$')
ax3.set_ylim(0.44, 0.47)
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.plot(t_tick, np.mean(np.square(exp_term), axis=0), color='black')
ax3.set_ylabel(r'$\langle (\exp(\frac{\sin(\theta - \bar{\mu}) - 1}{\bar{\sigma}^{2}}))^{2} \rangle$')
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.plot(t_tick, np.mean(np.square(sin_term), axis=0), color='black')
ax4.set_ylabel(r'$\langle \sin^{2}(\theta - \bar{\mu}) \rangle$')
ax4.set_ylim(0.3, 0.7)
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.plot(t_tick, np.mean(np.square(1/var_mean_mtx), axis=0), color='black')
ax5.set_ylabel(r'$\langle (\frac{1}{\bar{\sigma}^{2}})^{2} \rangle$')
ax5.set_ylim(0.25, 0.3)
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.set_xlabel('t / ms')
plt.tight_layout()
fig5.savefig('Mu Parameter Coefficient')
