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


a_df_08 = pd.read_csv('a_w_08.csv', index_col=False)
a_df_1 = pd.read_csv('a_tau_10_1.csv', index_col=False)
a_df_1h = pd.read_csv('a_w_1.5.csv', index_col=False)
a_df_2 = pd.read_csv('a_w_2.csv', index_col=False)

mu_df_08 = pd.read_csv('mu_w_08.csv', index_col=False)
mu_df_1 = pd.read_csv('mu_tau_10_1.csv', index_col=False)
mu_df_1h = pd.read_csv('mu_w_1.5.csv', index_col=False)
mu_df_2 = pd.read_csv('mu_w_2.csv', index_col=False)

var_df_08 = pd.read_csv('var_w_08.csv', index_col=False)
var_df_1 = pd.read_csv('var_tau_10_1.csv', index_col=False)
var_df_1h = pd.read_csv('var_w_1.5.csv', index_col=False)
var_df_2 = pd.read_csv('var_w_2.csv', index_col=False)

b_df_08 = pd.read_csv('b_w_08.csv', index_col=False)
b_df_1 = pd.read_csv('b_tau_10_1.csv', index_col=False)
b_df_1h = pd.read_csv('b_w_1.5.csv', index_col=False)
b_df_2 = pd.read_csv('b_w_2.csv', index_col=False)

a_08 = np.transpose(a_df_08.values)
a_1 = np.transpose(a_df_1.values)
a_1h = np.transpose(a_df_1h.values)
a_2 = np.transpose(a_df_2.values)

mu_08 = np.transpose(mu_df_08.values)
mu_1 = np.transpose(mu_df_1.values)
mu_1h = np.transpose(mu_df_1h.values)
mu_2 = np.transpose(mu_df_2.values)

var_08 = np.transpose(var_df_08.values)
var_1 = np.transpose(var_df_1.values)
var_1h = np.transpose(var_df_1h.values)
var_2 = np.transpose(var_df_2.values)

b_08 = np.transpose(b_df_08.values)
b_1 = np.transpose(b_df_1.values)
b_1h = np.transpose(b_df_1h.values)
b_2 = np.transpose(b_df_2.values)


v_var_08 = pd.read_csv('v_var_w_08.csv', index_col=False)
v_var_1 = pd.read_csv('v_var_tau_10_1.csv', index_col=False)
v_var_1h = pd.read_csv('v_var_w_1.5.csv', index_col=False)
v_var_2 = pd.read_csv('v_var_w_2.csv', index_col=False)

v_variance_08 = np.mean(v_var_08, axis=0)
v_variance_1 = np.mean(v_var_1, axis=0)
v_variance_1h = np.mean(v_var_1h, axis=0)
v_variance_2 = np.mean(v_var_2, axis=0)


a_mean_08 = np.mean(a_08, axis=0)
a_mean_1 = np.mean(a_1, axis=0)
a_mean_1h = np.mean(a_1h, axis=0)
a_mean_2 = np.mean(a_2, axis=0)
mu_mean_08 = np.mean(mu_08, axis=0)
mu_mean_1 = np.mean(mu_1, axis=0)
mu_mean_1h = np.mean(mu_1h, axis=0)
mu_mean_2 = np.mean(mu_2, axis=0)
var_mean_08 = np.mean(var_08, axis=0)
var_mean_1 = np.mean(var_1, axis=0)
var_mean_1h = np.mean(var_1h, axis=0)
var_mean_2 = np.mean(var_2, axis=0)
b_mean_08 = np.mean(b_08, axis=0)
b_mean_1 = np.mean(b_1, axis=0)
b_mean_1h = np.mean(b_1h, axis=0)
b_mean_2 = np.mean(b_2, axis=0)

mu_var_08 = np.var(mu_08, axis=0)
mu_var_1 = np.var(mu_1, axis=0)
mu_var_1h = np.var(mu_1h, axis=0)
mu_var_2 = np.var(mu_2, axis=0)


N_neuron = 100
N_point = 23000
x = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
t_tick = np.arange(0, t3, 0.1)


x_mtx = np.reshape(np.repeat(x, N_point), (N_neuron, N_point))

a_mean_mtx_08 = np.reshape(np.repeat(a_mean_08, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_1 = np.reshape(np.repeat(a_mean_1, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_1h = np.reshape(np.repeat(a_mean_1h, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_2 = np.reshape(np.repeat(a_mean_2, N_neuron), (N_point, N_neuron)).T

mu_mean_mtx_08 = np.reshape(np.repeat(mu_mean_08, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_1 = np.reshape(np.repeat(mu_mean_1, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_1h = np.reshape(np.repeat(mu_mean_1h, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_2 = np.reshape(np.repeat(mu_mean_2, N_neuron), (N_point, N_neuron)).T

var_mean_mtx_08 = np.reshape(np.repeat(var_mean_08, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_1 = np.reshape(np.repeat(var_mean_1, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_1h = np.reshape(np.repeat(var_mean_1h, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_2 = np.reshape(np.repeat(var_mean_2, N_neuron), (N_point, N_neuron)).T

b_mean_mtx_08 = np.reshape(np.repeat(b_mean_08, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_1 = np.reshape(np.repeat(b_mean_1, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_1h = np.reshape(np.repeat(b_mean_1h, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_2 = np.reshape(np.repeat(b_mean_2, N_neuron), (N_point, N_neuron)).T

mu_var_mtx_08 = np.reshape(np.repeat(mu_var_08, N_neuron), (N_point, N_neuron)).T
mu_var_mtx_1 = np.reshape(np.repeat(mu_var_1, N_neuron), (N_point, N_neuron)).T
mu_var_mtx_1h = np.reshape(np.repeat(mu_var_1h, N_neuron), (N_point, N_neuron)).T
mu_var_mtx_2 = np.reshape(np.repeat(mu_var_2, N_neuron), (N_point, N_neuron)).T


var_all_mean_08 = np.mean(var_08)
var_all_mean_1 = np.mean(var_1)
var_all_mean_1h = np.mean(var_1h)
var_all_mean_2 = np.mean(var_2)

b_all_mean_08 = np.mean(b_08)
b_all_mean_1 = np.mean(b_1)
b_all_mean_1h = np.mean(b_1h)
b_all_mean_2 = np.mean(b_2)

var_mtx_all_mean_08 = np.reshape(var_all_mean_08 * np.ones(N_point*N_neuron), (N_neuron, N_point))
var_mtx_all_mean_1 = np.reshape(var_all_mean_1 * np.ones(N_point*N_neuron), (N_neuron, N_point))
var_mtx_all_mean_1h = np.reshape(var_all_mean_1h * np.ones(N_point*N_neuron), (N_neuron, N_point))
var_mtx_all_mean_2 = np.reshape(var_all_mean_2 * np.ones(N_point*N_neuron), (N_neuron, N_point))

b_mtx_all_mean_08 = np.reshape(b_all_mean_08 * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean_1 = np.reshape(b_all_mean_1 * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean_1h = np.reshape(b_all_mean_1h * np.ones(N_point*N_neuron), (N_neuron, N_point))
b_mtx_all_mean_2 = np.reshape(b_all_mean_2 * np.ones(N_point*N_neuron), (N_neuron, N_point))

mu_grad_params_const_08 = modelfit.grad_mu(x_mtx, a_mean_mtx_08, mu_mean_mtx_08, var_mtx_all_mean_08, b_mtx_all_mean_08)
mu_grad_params_const_1 = modelfit.grad_mu(x_mtx, a_mean_mtx_1, mu_mean_mtx_1, var_mtx_all_mean_1, b_mtx_all_mean_1)
mu_grad_params_const_1h = modelfit.grad_mu(x_mtx, a_mean_mtx_1h, mu_mean_mtx_1h, var_mtx_all_mean_1h, b_mtx_all_mean_1h)
mu_grad_params_const_2 = modelfit.grad_mu(x_mtx, a_mean_mtx_2, mu_mean_mtx_2, var_mtx_all_mean_2, b_mtx_all_mean_2)

mu_grad_params_const_08_sqr_mean = np.mean(np.square(mu_grad_params_const_08), axis=0)
mu_grad_params_const_1_sqr_mean = np.mean(np.square(mu_grad_params_const_1), axis=0)
mu_grad_params_const_1h_sqr_mean = np.mean(np.square(mu_grad_params_const_1h), axis=0)
mu_grad_params_const_2_sqr_mean = np.mean(np.square(mu_grad_params_const_2), axis=0)

mu_grad_params_const_08_est = mu_grad_params_const_08_sqr_mean*mu_var_08
mu_grad_params_const_1_est = mu_grad_params_const_1_sqr_mean*mu_var_1
mu_grad_params_const_1h_est = mu_grad_params_const_1h_sqr_mean*mu_var_1h
mu_grad_params_const_2_est = mu_grad_params_const_2_sqr_mean*mu_var_2


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

ax1.plot(t_tick, mu_grad_params_const_08_sqr_mean, color='lime', label=r'$k_{w}=0.8$')
ax1.plot(t_tick, mu_grad_params_const_1_sqr_mean, color='limegreen', label=r'$k_{w}=1$')
ax1.plot(t_tick, mu_grad_params_const_1h_sqr_mean, color='forestgreen', label=r'$k_{w}=1.5$')
ax1.plot(t_tick, mu_grad_params_const_2_sqr_mean, color='darkgreen', label=r'$k_{w}=2$')
ax1.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \rangle_{i}$')
ax1.legend(frameon=False)

ax2.plot(t_tick, mu_var_08, color='deepskyblue', label=r'$k_{w}=0.8$')
ax2.plot(t_tick, mu_var_1, color='dodgerblue', label=r'$k_{w}=1$')
ax2.plot(t_tick, mu_var_1h, color='royalblue', label=r'$k_{w}=1.5$')
ax2.plot(t_tick, mu_var_2, color='mediumblue', label=r'$k_{w}=2$')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')

ax3.plot(t_tick, mu_grad_params_const_08_est, color='lightcoral', label=r'$k_{w}=0.8$')
ax3.plot(t_tick, mu_grad_params_const_1_est, color='coral', label=r'$k_{w}=1$')
ax3.plot(t_tick, mu_grad_params_const_1h_est, color='orangered', label=r'$k_{w}=1.5$')
ax3.plot(t_tick, mu_grad_params_const_2_est, color='firebrick', label=r'$k_{w}=2$')
ax3.plot(t_tick, v_variance_08, color='lightgrey', label=r'$k_{w}=0.8$')
ax3.plot(t_tick, v_variance_1, color='darkgrey', label=r'$k_{w}=1$')
ax3.plot(t_tick, v_variance_1h, color='grey', label=r'$k_{w}=1.5$')
ax3.plot(t_tick, v_variance_2, color='dimgrey', label=r'$k_{w}=2$')
ax3.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \sigma^{2}_{\mu} \rangle_{i}$')
ax3.set_xlabel('t [ms]')
"ax3.legend(frameon=False, ncol=2, loc='upper right')"

plt.tight_layout()
fig1.savefig('network_weight')


start_time = int(rest_time1 / 0.1)
end_time = int((rest_time1+stim_time) / 0.1)
stim_x = np.arange(0, end_time-start_time)
params_guess = np.array([2, 100, 1])
two_part_params_guess=np.array([2, 100, 1, 100, 2])
stim_ticks = np.arange(0, stim_time, 0.1)


stim_mu_coeff_08 = mu_grad_params_const_08_sqr_mean[start_time:end_time]
stim_mu_coeff_1 = mu_grad_params_const_1_sqr_mean[start_time:end_time]
stim_mu_coeff_1h = mu_grad_params_const_1h_sqr_mean[start_time:end_time]
stim_mu_coeff_2 = mu_grad_params_const_2_sqr_mean[start_time:end_time]

stim_mu_coeff_params_08 = minimize(modelfit.exp_response_up_mse, params_guess, args=(stim_x, stim_mu_coeff_08)).x
stim_mu_coeff_params_1 = minimize(modelfit.exp_response_up_mse, params_guess, args=(stim_x, stim_mu_coeff_1)).x
stim_mu_coeff_params_1h = minimize(modelfit.exp_response_up_mse, params_guess, args=(stim_x, stim_mu_coeff_1h)).x
stim_mu_coeff_params_2 = minimize(modelfit.exp_response_up_mse, params_guess, args=(stim_x, stim_mu_coeff_2)).x

stim_mu_coeff_est_08 = modelfit.exp_response_up(stim_x, stim_mu_coeff_params_08)
stim_mu_coeff_est_1 = modelfit.exp_response_up(stim_x, stim_mu_coeff_params_1)
stim_mu_coeff_est_1h = modelfit.exp_response_up(stim_x, stim_mu_coeff_params_1h)
stim_mu_coeff_est_2 = modelfit.exp_response_up(stim_x, stim_mu_coeff_params_2)


stim_mu_var_08 = mu_var_08[start_time:end_time]
stim_mu_var_1 = mu_var_1[start_time:end_time]
stim_mu_var_1h = mu_var_1h[start_time:end_time]
stim_mu_var_2 = mu_var_2[start_time:end_time]

stim_mu_var_params_08 = minimize(modelfit.exp_response_down_mse, params_guess, args=(stim_x, stim_mu_var_08)).x
stim_mu_var_params_1 = minimize(modelfit.exp_response_down_mse, params_guess, args=(stim_x, stim_mu_var_1)).x
stim_mu_var_params_1h = minimize(modelfit.exp_response_down_mse, params_guess, args=(stim_x, stim_mu_var_1h)).x
stim_mu_var_params_2 = minimize(modelfit.exp_response_down_mse, params_guess, args=(stim_x, stim_mu_var_2)).x

stim_mu_var_est_08 = modelfit.exp_response_down(stim_x, stim_mu_var_params_08)
stim_mu_var_est_1 = modelfit.exp_response_down(stim_x, stim_mu_var_params_1)
stim_mu_var_est_1h = modelfit.exp_response_down(stim_x, stim_mu_var_params_1h)
stim_mu_var_est_2 = modelfit.exp_response_down(stim_x, stim_mu_var_params_2)


var_est_max_index_08 = int(np.argmax(mu_grad_params_const_08_est[start_time:end_time]))
var_est_max_index_1 = int(np.argmax(mu_grad_params_const_1_est)) - start_time
var_est_max_index_1h = int(np.argmax(mu_grad_params_const_1h_est)) - start_time
var_est_max_index_2 = int(np.argmax(mu_grad_params_const_2_est)) - start_time

var_est_params_08 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, mu_grad_params_const_08_est[start_time:end_time], var_est_max_index_08)).x
var_est_params_1 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, mu_grad_params_const_1_est[start_time:end_time], var_est_max_index_1)).x
var_est_params_1h = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, mu_grad_params_const_1h_est[start_time:end_time], var_est_max_index_1h)).x
var_est_params_2 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, mu_grad_params_const_2_est[start_time:end_time], var_est_max_index_2)).x

var_est_est_08 = modelfit.two_part_exp_response(stim_x, var_est_params_08, var_est_max_index_08)
var_est_est_1 = modelfit.two_part_exp_response(stim_x, var_est_params_1, var_est_max_index_1)
var_est_est_1h = modelfit.two_part_exp_response(stim_x, var_est_params_1h, var_est_max_index_1h)
var_est_est_2 = modelfit.two_part_exp_response(stim_x, var_est_params_2, var_est_max_index_2)


v_variance_max_index_08 = int(np.argmax(v_variance_08)) - start_time
v_variance_max_index_1 = int(np.argmax(v_variance_1)) - start_time
v_variance_max_index_1h = int(np.argmax(v_variance_1h)) - start_time
v_variance_max_index_2 = int(np.argmax(v_variance_2)) - start_time

v_variance_params_08 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, v_variance_08[start_time:end_time], v_variance_max_index_08)).x
v_variance_params_1 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, v_variance_1[start_time:end_time], v_variance_max_index_1)).x
v_variance_params_1h = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, v_variance_1h[start_time:end_time], v_variance_max_index_1h)).x
v_variance_params_2 = minimize(modelfit.two_part_exp_response_mse, two_part_params_guess, args=(stim_x, v_variance_2[start_time:end_time], v_variance_max_index_2)).x

v_variance_est_08 = modelfit.two_part_exp_response(stim_x, v_variance_params_08, v_variance_max_index_08)
v_variance_est_1 = modelfit.two_part_exp_response(stim_x, v_variance_params_1, v_variance_max_index_1)
v_variance_est_1h = modelfit.two_part_exp_response(stim_x, v_variance_params_1h, v_variance_max_index_1h)
v_variance_est_2 = modelfit.two_part_exp_response(stim_x, v_variance_params_2, v_variance_max_index_2)


network_weight = [0.8, 1, 1.5, 2]

coeff_tau = [stim_mu_coeff_params_08[1]*0.1, stim_mu_coeff_params_1[1]*0.1, stim_mu_coeff_params_1h[1]*0.1, stim_mu_coeff_params_2[1]*0.1]
true_var_up_tau = [v_variance_params_08[1]*0.1, v_variance_params_1[1]*0.1, v_variance_params_1h[1]*0.1, v_variance_params_2[1]*0.1]
est_var_up_tau = [var_est_params_08[1]*0.1, var_est_params_1[1]*0.1, var_est_params_1h[1]*0.1, var_est_params_2[1]*0.1]

var_tau = [stim_mu_var_params_08[1]*0.1, stim_mu_var_params_1[1]*0.1, stim_mu_var_params_1h[1]*0.1, stim_mu_var_params_2[1]*0.1]
true_var_down_tau = [v_variance_params_08[3]*0.1, v_variance_params_1[3]*0.1, v_variance_params_1h[3]*0.1, v_variance_params_2[3]*0.1]
est_var_down_tau = [var_est_params_08[3]*0.1, var_est_params_1[3]*0.1, var_est_params_1h[3]*0.1, var_est_params_2[3]*0.1]


fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

line1 = ax1.scatter(network_weight, true_var_up_tau, s=40, label='true', color='dimgrey')
line2 = ax1.scatter(network_weight, est_var_up_tau, s=40, label=r'$\alpha^{2}\sigma^{2}$', marker='v', color='red')
line3 = ax1.scatter(network_weight, coeff_tau, s=40, label=r'$\alpha^{2}$', marker='x', color='forestgreen')
ax1.set_ylabel(r'$\tau_{r}\ [ms]$')
ax1.set_xlabel(r'$k_{W}$')
ax1.set_ylim([0, 400])
ax1.set_xticks(network_weight)
ax1.spines['bottom'].set_visible(True)

ax2.scatter(network_weight, est_var_down_tau, s=40, label='true', color='dimgrey')
ax2.scatter(network_weight, true_var_down_tau, s=40, label=r'$\tau_{a}^{2}\sigma^{2}$', marker='v', color='orangered')
line4 = ax2.scatter(network_weight, var_tau, s=40, label=r'$\sigma^{2}$', marker='x', color='royalblue')
ax2.set_ylabel(r'$\tau_{d}\ [ms]$')
ax2.set_xlabel(r'$k_{W}$')
ax2.set_xticks(network_weight)
ax2.set_ylim([0, 400])
ax2.spines['bottom'].set_visible(True)

plt.legend(handles = [line1, line2, line3, line4], loc='best', fontsize=18)
plt.tight_layout()
fig2.savefig('network-weight-effect')


fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 20))

ax1.plot(stim_ticks, stim_mu_coeff_08, color='lime', label=r'$\hat{\tau}_{0.8}=$'+str(int(stim_mu_coeff_params_08[1]*0.08))+'ms')
ax1.plot(stim_ticks, stim_mu_coeff_est_08, color='dimgrey', linestyle='dashed')

ax1.plot(stim_ticks, stim_mu_coeff_1, color='limegreen', label=r'$\hat{\tau}_{1}=$'+str(int(stim_mu_coeff_params_1[1]*0.1))+'ms')
ax1.plot(stim_ticks, stim_mu_coeff_est_1, color='dimgrey', linestyle='dashed')

ax1.plot(stim_ticks, stim_mu_coeff_1h, color='forestgreen', label=r'$\hat{\tau}_{1.5}=$'+str(int(stim_mu_coeff_params_1h[1]*0.1))+'ms')
ax1.plot(stim_ticks, stim_mu_coeff_est_1h, color='dimgrey', linestyle='dashed')

ax1.plot(stim_ticks, stim_mu_coeff_2, color='darkgreen', label=r'$\hat{\tau}_{2}=$'+str(int(stim_mu_coeff_params_2[1]*0.1))+'ms')
ax1.plot(stim_ticks, stim_mu_coeff_est_2, color='dimgrey', linestyle='dashed')

ax1.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \rangle_{i}$')
ax1.legend(frameon=False, loc='lower right')

ax2.plot(stim_ticks, stim_mu_var_08, color='deepskyblue', label=r'$\hat{\tau}_{0.8}=$'+str(int(stim_mu_var_params_08[1]*0.08))+'ms')
ax2.plot(stim_ticks, stim_mu_var_est_08, color='dimgrey', linestyle='dashed')

ax2.plot(stim_ticks, stim_mu_var_1, color='deepskyblue', label=r'$\hat{\tau}_{1}=$'+str(int(stim_mu_var_params_1[1]*0.1))+'ms')
ax2.plot(stim_ticks, stim_mu_var_est_1, color='dimgrey', linestyle='dashed')

ax2.plot(stim_ticks, stim_mu_var_1h, color='royalblue', label=r'$\hat{\tau}_{1.5}=$'+str(int(stim_mu_var_params_1h[1]*0.1))+'ms')
ax2.plot(stim_ticks, stim_mu_var_est_1h, color='dimgrey', linestyle='dashed')

ax2.plot(stim_ticks, stim_mu_var_2, color='mediumblue', label=r'$\hat{\tau}_{2}=$'+str(int(stim_mu_var_params_2[1]*0.1))+'ms')
ax2.plot(stim_ticks, stim_mu_var_est_2, color='dimgrey', linestyle='dashed')

ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax2.legend(frameon=False)

ax3.plot(stim_ticks, mu_grad_params_const_08_est[start_time:end_time], color='lightcoral', label=r'$\hat{\tau}_{0.8,r}=$'+str(int(var_est_params_08[1]*0.1))+'ms,'+r'$\hat{\tau}_{08,d}=$'+str(int(var_est_params_08[3]*0.1))+'ms')
ax3.plot(stim_ticks, var_est_est_08, color='dimgrey', linestyle='dashed')

ax3.plot(stim_ticks, mu_grad_params_const_1_est[start_time:end_time], color='coral', label=r'$\hat{\tau}_{1,r}=$'+str(int(var_est_params_1[1]*0.1))+'ms,'+r'$\hat{\tau}_{1,d}=$'+str(int(var_est_params_1[3]*0.1))+'ms')
ax3.plot(stim_ticks, var_est_est_1, color='dimgrey', linestyle='dashed')

ax3.plot(stim_ticks, mu_grad_params_const_1h_est[start_time:end_time], color='orangered', label=r'$\hat{\tau}_{1.5,r}=$'+str(int(var_est_params_1h[1]*0.1))+'ms,'+r'$\hat{\tau}_{1.5,d}=$'+str(int(var_est_params_1h[3]*0.1))+'ms')
ax3.plot(stim_ticks, var_est_est_1h, color='dimgrey', linestyle='dashed')

ax3.plot(stim_ticks, mu_grad_params_const_2_est[start_time:end_time], color='firebrick', label=r'$\hat{\tau}_{2,r}=$'+str(int(var_est_params_2[1]*0.1))+'ms,'+r'$\hat{\tau}_{2,d}=$'+str(int(var_est_params_2[3]*0.1))+'ms')
ax3.plot(stim_ticks, var_est_est_2, color='dimgrey', linestyle='dashed')

ax3.set_ylabel(r'$\langle (\alpha_{i}^{\mu})^{2} \sigma^{2}_{\hat{\mu}} \rangle_{i}$')
ax3.legend(frameon=False)

ax4.plot(stim_ticks, v_variance_08[start_time:end_time], color='lightgray', label=r'$\hat{\tau}_{0.8,r}=$'+str(int(v_variance_params_08[1]*0.1))+'ms,'+r'$\hat{\tau}_{0.8,d}=$'+str(int(v_variance_params_08[3]*0.1))+'ms')
ax4.plot(stim_ticks, v_variance_est_08, color='dimgrey', linestyle='dashed')

ax4.plot(stim_ticks, v_variance_1[start_time:end_time], color='darkgray', label=r'$\hat{\tau}_{1,r}=$'+str(int(v_variance_params_1[1]*0.1))+'ms,'+r'$\hat{\tau}_{1,d}=$'+str(int(v_variance_params_1[3]*0.1))+'ms')
ax4.plot(stim_ticks, v_variance_est_1, color='dimgrey', linestyle='dashed')

ax4.plot(stim_ticks, v_variance_1h[start_time:end_time], color='grey', label=r'$\hat{\tau}_{1.5,r}=$'+str(int(v_variance_params_1h[1]*0.1))+'ms,'+r'$\hat{\tau}_{1.5,d}=$'+str(int(v_variance_params_1h[3]*0.1))+'ms')
ax4.plot(stim_ticks, v_variance_est_1h, color='dimgrey', linestyle='dashed')

ax4.plot(stim_ticks, v_variance_2[start_time:end_time], color='dimgrey', label=r'$\hat{\tau}_{2,r}=$'+str(int(v_variance_params_2[1]*0.1))+'ms,'+r'$\hat{\tau}_{2,d}=$'+str(int(v_variance_params_2[3]*0.1))+'ms')
ax4.plot(stim_ticks, v_variance_est_2, color='dimgrey', linestyle='dashed')

ax4.set_ylabel(r'$\sigma^{2}$')
ax4.legend(frameon=False)
ax4.set_xlabel('t [ms]')

plt.tight_layout()
fig3.savefig('network-weight-validate')
