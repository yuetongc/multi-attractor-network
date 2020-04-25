import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelfit
from scipy.optimize import minimize


import os
"path for simulation data"
os.chdir('/Users/yuetongyc/Desktop/Cambridge/IIB/Project//data')

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


a_df_10 = pd.read_csv('a_tau_10_1.csv', index_col=False)
mu_df_10 = pd.read_csv('mu_tau_10_1.csv', index_col=False)
var_df_10 = pd.read_csv('var_tau_10_1.csv', index_col=False)
b_df_10 = pd.read_csv('b_tau_10_1.csv', index_col=False)
a_df_20 = pd.read_csv('a_tau_20_2.csv', index_col=False)
mu_df_20 = pd.read_csv('mu_tau_20_2.csv', index_col=False)
var_df_20 = pd.read_csv('var_tau_20_2.csv', index_col=False)
b_df_20 = pd.read_csv('b_tau_20_2.csv', index_col=False)
a_df_30 = pd.read_csv('a_tau_30.csv', index_col=False)
mu_df_30 = pd.read_csv('mu_tau_30.csv', index_col=False)
var_df_30 = pd.read_csv('var_tau_30.csv', index_col=False)
b_df_30 = pd.read_csv('b_tau_30.csv', index_col=False)

a_10 = np.transpose(a_df_10.values)
mu_10 = np.transpose(mu_df_10.values)
var_10 = np.transpose(var_df_10.values)
b_10 = np.transpose(b_df_10.values)
a_20 = np.transpose(a_df_20.values)
mu_20 = np.transpose(mu_df_20.values)
var_20 = np.transpose(var_df_20.values)
b_20 = np.transpose(b_df_20.values)
a_30 = np.transpose(a_df_30.values)
mu_30 = np.transpose(mu_df_30.values)
var_30 = np.transpose(var_df_30.values)
b_30 = np.transpose(b_df_30.values)


v_var_10 = pd.read_csv('v_var_tau_10_1.csv', index_col=False)
v_var_20 = pd.read_csv('v_var_tau_20_2.csv', index_col=False)
v_var_30 = pd.read_csv('v_var_tau_30.csv', index_col=False)


v_variance_10 = np.mean(v_var_10, axis=0)
v_variance_20 = np.mean(v_var_20, axis=0)
v_variance_30 = np.mean(v_var_30, axis=0)

a_mean_10 = np.mean(a_10, axis=0)
a_mean_20 = np.mean(a_20, axis=0)
a_mean_30 = np.mean(a_30, axis=0)
mu_mean_10 = np.mean(mu_10, axis=0)
mu_mean_20 = np.mean(mu_20, axis=0)
mu_mean_30 = np.mean(mu_30, axis=0)
var_mean_10 = np.mean(var_10, axis=0)
var_mean_20 = np.mean(var_20, axis=0)
var_mean_30 = np.mean(var_30, axis=0)
b_mean_10 = np.mean(b_10, axis=0)
b_mean_20 = np.mean(b_20, axis=0)
b_mean_30 = np.mean(b_30, axis=0)

mu_var_10 = np.var(mu_10, axis=0)
mu_var_20 = np.var(mu_20, axis=0)
mu_var_30 = np.var(mu_30, axis=0)


N_neuron = 100
N_point = 23000
x = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
t_tick = np.arange(0, t3, 0.1)


x_mtx = np.reshape(np.repeat(x, N_point), (N_neuron, N_point))

a_mean_mtx_10 = np.reshape(np.repeat(a_mean_10, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_10 = np.reshape(np.repeat(mu_mean_10, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_10 = np.reshape(np.repeat(var_mean_10, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_10 = np.reshape(np.repeat(b_mean_10, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_20 = np.reshape(np.repeat(a_mean_20, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_20 = np.reshape(np.repeat(mu_mean_20, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_20 = np.reshape(np.repeat(var_mean_20, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_20 = np.reshape(np.repeat(b_mean_20, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_30 = np.reshape(np.repeat(a_mean_30, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_30 = np.reshape(np.repeat(mu_mean_30, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_30 = np.reshape(np.repeat(var_mean_30, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_30 = np.reshape(np.repeat(b_mean_30, N_neuron), (N_point, N_neuron)).T

mu_var_mtx_10 = np.reshape(np.repeat(mu_var_10, N_neuron), (N_point, N_neuron)).T
mu_var_mtx_20 = np.reshape(np.repeat(mu_var_20, N_neuron), (N_point, N_neuron)).T
mu_var_mtx_30 = np.reshape(np.repeat(mu_var_30, N_neuron), (N_point, N_neuron)).T


var_all_mean_10 = np.mean(var_10)
var_all_mean_20 = np.mean(var_20)
var_all_mean_30 = np.mean(var_30)

b_all_mean_10 = np.mean(b_10)
b_all_mean_20 = np.mean(b_20)
b_all_mean_30 = np.mean(b_30)

var_mtx_all_mean_10 = np.reshape(var_all_mean_10 * np.ones(N_point*N_neuron), (N_neuron, N_point))
var_mtx_all_mean_20 = np.reshape(var_all_mean_20 * np.ones(N_point*N_neuron), (N_neuron, N_point))
var_mtx_all_mean_30 = np.reshape(var_all_mean_30 * np.ones(N_point*N_neuron), (N_neuron, N_point))

b_mtx_all_mean_10 = np.reshape(b_all_mean_10 * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean_20 = np.reshape(b_all_mean_20 * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean_30 = np.reshape(b_all_mean_30 * np.ones(N_point*N_neuron), (N_neuron, N_point))

mu_grad_params_const_10 = modelfit.grad_mu(x_mtx, a_mean_mtx_10, mu_mean_mtx_10, var_mtx_all_mean_10, b_mtx_all_mean_10)
mu_grad_params_const_20 = modelfit.grad_mu(x_mtx, a_mean_mtx_20, mu_mean_mtx_20, var_mtx_all_mean_20, b_mtx_all_mean_20)
mu_grad_params_const_30 = modelfit.grad_mu(x_mtx, a_mean_mtx_30, mu_mean_mtx_30, var_mtx_all_mean_30, b_mtx_all_mean_30)

mu_grad_params_const_10_sqr_mean = np.mean(np.square(mu_grad_params_const_10), axis=0)
mu_grad_params_const_20_sqr_mean = np.mean(np.square(mu_grad_params_const_20), axis=0)
mu_grad_params_const_30_sqr_mean = np.mean(np.square(mu_grad_params_const_30), axis=0)

mu_grad_params_const_10_est = mu_grad_params_const_10_sqr_mean*mu_var_10
mu_grad_params_const_20_est = mu_grad_params_const_20_sqr_mean*mu_var_20
mu_grad_params_const_30_est = mu_grad_params_const_30_sqr_mean*mu_var_30


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

ax1.plot(t_tick, mu_grad_params_const_10_sqr_mean, color='limegreen', label=r'$\tau$=10ms')
ax1.plot(t_tick, mu_grad_params_const_20_sqr_mean, color='forestgreen', label=r'$\tau$=20ms')
ax1.plot(t_tick, mu_grad_params_const_30_sqr_mean, color='darkgreen', label=r'$\tau$=30ms')
ax1.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \rangle_{i}$')
ax1.legend(frameon=False)

ax2.plot(t_tick, mu_var_10, color='dodgerblue', label=r'$\tau$=10ms')
ax2.plot(t_tick, mu_var_20, color='royalblue', label=r'$\tau$=20ms')
ax2.plot(t_tick, mu_var_30, color='mediumblue', label=r'$\tau$=30ms')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')

ax3.plot(t_tick, mu_grad_params_const_10_est, color='lightcoral', label=r'$\tau$=10ms')
ax3.plot(t_tick, mu_grad_params_const_20_est, color='orangered', label=r'$\tau$=20ms')
ax3.plot(t_tick, mu_grad_params_const_30_est, color='firebrick', label=r'$\tau$=30ms')
ax3.plot(t_tick, v_variance_10, color='darkgrey', label=r'$\tau$=10ms')
ax3.plot(t_tick, v_variance_20, color='grey', label=r'$\tau$=20ms')
ax3.plot(t_tick, v_variance_30, color='dimgrey', label=r'$\tau$=30ms')
ax3.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \sigma^{2}_{\mu} \rangle_{i}$')
ax3.set_xlabel('t [ms]')
"ax3.legend(frameon=False, ncol=2, loc='upper right')"

plt.tight_layout()
fig1.savefig('time_const')


start_time = int(rest_time1 / 0.1)
end_time = int((rest_time1+stim_time) / 0.1)
stim_x = np.arange(0, end_time-start_time)
params_guess = np.array([2, 100])
stim_ticks = np.arange(0, stim_time, 0.1)


stim_mu_coeff_10 = mu_grad_params_const_10_sqr_mean[start_time:end_time]
stim_mu_coeff_20 = mu_grad_params_const_20_sqr_mean[start_time:end_time]
stim_mu_coeff_30 = mu_grad_params_const_30_sqr_mean[start_time:end_time]

norm_stim_mu_coeff_10 = stim_mu_coeff_10 - np.min(stim_mu_coeff_10)
norm_stim_mu_coeff_20 = stim_mu_coeff_20 - np.min(stim_mu_coeff_20)
norm_stim_mu_coeff_30 = stim_mu_coeff_30 - np.min(stim_mu_coeff_30)

stim_mu_coeff_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_coeff_10)).x
stim_mu_coeff_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_coeff_20)).x
stim_mu_coeff_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_coeff_30)).x


stim_mu_var_10 = mu_var_10[start_time:end_time]
stim_mu_var_20 = mu_var_20[start_time:end_time]
stim_mu_var_30 = mu_var_30[start_time:end_time]

norm_stim_mu_var_10 = np.max(stim_mu_var_10) - stim_mu_var_10
norm_stim_mu_var_20 = np.max(stim_mu_var_20) - stim_mu_var_20
norm_stim_mu_var_30 = np.max(stim_mu_var_30) - stim_mu_var_30

stim_mu_var_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_10)).x
stim_mu_var_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_20)).x
stim_mu_var_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_30)).x

stim_mu_var_est_10 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_10)) - \
                     modelfit.exp_response(stim_x, stim_mu_var_params_10)
stim_mu_var_est_20 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_20)) - \
                     modelfit.exp_response(stim_x, stim_mu_var_params_20)
stim_mu_var_est_30 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_30)) - \
                     modelfit.exp_response(stim_x, stim_mu_var_params_30)


v_variance_max_index_10 = int(np.argmax(v_variance_10))
v_variance_max_index_20 = int(np.argmax(v_variance_20))
v_variance_max_index_30 = int(np.argmax(v_variance_30))

stim_v_variance_up_10 = v_variance_10[start_time:v_variance_max_index_10]
stim_v_variance_up_20 = v_variance_20[start_time:v_variance_max_index_20]
stim_v_variance_up_30 = v_variance_30[start_time:v_variance_max_index_30]

norm_stim_v_variance_up_10 = stim_v_variance_up_10 - np.min(stim_v_variance_up_10)
norm_stim_v_variance_up_20 = stim_v_variance_up_20 - np.min(stim_v_variance_up_20)
norm_stim_v_variance_up_30 = stim_v_variance_up_30 - np.min(stim_v_variance_up_30)

stim_x_up_10 = np.arange(0, v_variance_max_index_10-start_time)
stim_x_up_20 = np.arange(0, v_variance_max_index_20-start_time)
stim_x_up_30 = np.arange(0, v_variance_max_index_30-start_time)

stim_v_variance_up_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_up_10, norm_stim_v_variance_up_10)).x
stim_v_variance_up_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_up_20, norm_stim_v_variance_up_20)).x
stim_v_variance_up_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_up_30, norm_stim_v_variance_up_30)).x

stim_v_variance_down_10 = v_variance_10[v_variance_max_index_10:end_time]
stim_v_variance_down_20 = v_variance_20[v_variance_max_index_20:end_time]
stim_v_variance_down_30 = v_variance_30[v_variance_max_index_30:end_time]

norm_stim_v_variance_down_10 = v_variance_10[v_variance_max_index_10] - stim_v_variance_down_10
norm_stim_v_variance_down_20 = v_variance_10[v_variance_max_index_20] - stim_v_variance_down_20
norm_stim_v_variance_down_30 = v_variance_10[v_variance_max_index_30] - stim_v_variance_down_30

stim_x_down_10 = np.arange(0, end_time-v_variance_max_index_10)
stim_x_down_20 = np.arange(0, end_time-v_variance_max_index_20)
stim_x_down_30 = np.arange(0, end_time-v_variance_max_index_30)

stim_v_variance_down_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_down_10, norm_stim_v_variance_down_10)).x
stim_v_variance_down_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_down_20, norm_stim_v_variance_down_20)).x
stim_v_variance_down_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x_down_30, norm_stim_v_variance_down_30)).x


var_est_max_index_10 = int(np.argmax(mu_grad_params_const_10_est))
var_est_max_index_20 = int(np.argmax(mu_grad_params_const_20_est))
var_est_max_index_30 = int(np.argmax(mu_grad_params_const_30_est))

stim_var_est_up_10 = mu_grad_params_const_10_est[start_time:var_est_max_index_10]
stim_var_est_up_20 = mu_grad_params_const_20_est[start_time:var_est_max_index_20]
stim_var_est_up_30 = mu_grad_params_const_30_est[start_time:var_est_max_index_30]

norm_stim_var_est_up_10 = stim_var_est_up_10 - np.min(stim_var_est_up_10)
norm_stim_var_est_up_20 = stim_var_est_up_20 - np.min(stim_var_est_up_20)
norm_stim_var_est_up_30 = stim_var_est_up_30 - np.min(stim_var_est_up_30)

stim_est_x_up_10 = np.arange(0, var_est_max_index_10-start_time)
stim_est_x_up_20 = np.arange(0, var_est_max_index_20-start_time)
stim_est_x_up_30 = np.arange(0, var_est_max_index_30-start_time)

stim_var_est_up_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_up_10, norm_stim_var_est_up_10)).x
stim_var_est_up_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_up_20, norm_stim_var_est_up_20)).x
stim_var_est_up_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_up_30, norm_stim_var_est_up_30)).x

stim_var_est_down_10 = mu_grad_params_const_10_est[var_est_max_index_10:end_time]
stim_var_est_down_20 = mu_grad_params_const_20_est[var_est_max_index_20:end_time]
stim_var_est_down_30 = mu_grad_params_const_30_est[var_est_max_index_30:end_time]

norm_stim_var_est_down_10 = mu_grad_params_const_10_est[var_est_max_index_10] - stim_var_est_down_10
norm_stim_var_est_down_20 = mu_grad_params_const_20_est[var_est_max_index_20] - stim_var_est_down_20
norm_stim_var_est_down_30 = mu_grad_params_const_30_est[var_est_max_index_30] - stim_var_est_down_30

stim_est_x_down_10 = np.arange(0, end_time-var_est_max_index_10)
stim_est_x_down_20 = np.arange(0, end_time-var_est_max_index_20)
stim_est_x_down_30 = np.arange(0, end_time-var_est_max_index_30)

stim_var_est_down_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_down_10, norm_stim_var_est_down_10)).x
stim_var_est_down_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_down_20, norm_stim_var_est_down_20)).x
stim_var_est_down_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_est_x_down_30, norm_stim_var_est_down_30)).x


membrane_tau = [10, 20, 30]

coeff_tau = [stim_mu_coeff_params_10[1]*0.1, stim_mu_coeff_params_20[1]*0.1, stim_mu_coeff_params_30[1]*0.1]
true_var_up_tau = [stim_v_variance_up_params_10[1]*0.1, stim_v_variance_up_params_20[1]*0.1, stim_v_variance_up_params_30[1]*0.1]
est_var_up_tau = [stim_var_est_up_params_10[1]*0.1, stim_var_est_up_params_20[1]*0.1, stim_var_est_up_params_30[1]*0.1]

var_tau = [stim_mu_var_params_10[1]*0.1, stim_mu_var_params_20[1]*0.1, stim_mu_var_params_30[1]*0.1]
true_var_down_tau = [stim_v_variance_down_params_10[1]*0.1, stim_v_variance_down_params_20[1]*0.1, stim_v_variance_down_params_30[1]*0.1]
est_var_down_tau = [stim_var_est_down_params_10[1]*0.1, stim_var_est_down_params_20[1]*0.1, stim_var_est_down_params_30[1]*0.1]


fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
line1 = ax1.scatter(membrane_tau, true_var_up_tau, s=40, label='true', color='dimgrey')
line2 = ax1.scatter(membrane_tau, est_var_up_tau, s=40, label=r'$\alpha^{2}\sigma^{2}$', marker='v', color='orangered')
line3 = ax1.scatter(membrane_tau, coeff_tau, s=40, label=r'$\alpha^{2}$', marker='x', color='forestgreen')
ax1.set_ylabel(r'$\tau\ [ms]$')
ax1.set_xlabel(r'$\tau_{m}\ [ms]$')
ax1.set_xticks(membrane_tau)
ax1.spines['bottom'].set_visible(True)
ax2.scatter(membrane_tau, est_var_down_tau, s=40, label='true', color='dimgrey')
ax2.scatter(membrane_tau, true_var_down_tau, s=40, label=r'$\tau_{a}^{2}\sigma^{2}$', marker='v', color='orangered')
line4 = ax2.scatter(membrane_tau, var_tau, s=40, label=r'$\sigma^{2}$', marker='x', color='royalblue')
ax2.set_ylabel(r'$\tau\ [ms]$')
ax2.set_xlabel(r'$\tau_{m}\ [ms]$')
ax2.set_xticks(membrane_tau)
ax2.set_ylim([0, 400])
ax2.spines['bottom'].set_visible(True)
plt.legend(handles = [line1, line2, line3, line4], loc='best', fontsize=18)
plt.tight_layout()
fig2.savefig('time-const-effect')
