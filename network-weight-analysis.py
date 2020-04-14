import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelfit
from scipy.optimize import minimize


import os
os.chdir('/Users/yuetongyc/Desktop/Cambridge/IIB/Project//data')


a_df_1 = pd.read_csv('a_tau_10_1.csv', index_col=False)
mu_df_1 = pd.read_csv('mu_tau_10_1.csv', index_col=False)
var_df_1 = pd.read_csv('var_tau_10_1.csv', index_col=False)
b_df_1 = pd.read_csv('b_tau_10_1.csv', index_col=False)
a_df_1h = pd.read_csv('a_w_1.5.csv', index_col=False)
mu_df_1h = pd.read_csv('mu_w_1.5.csv', index_col=False)
var_df_1h = pd.read_csv('var_w_1.5.csv', index_col=False)
b_df_1h = pd.read_csv('b_w_1.5.csv', index_col=False)
a_df_2 = pd.read_csv('a_w_2.csv', index_col=False)
mu_df_2 = pd.read_csv('mu_w_2.csv', index_col=False)
var_df_2 = pd.read_csv('var_w_2.csv', index_col=False)
b_df_2 = pd.read_csv('b_w_2.csv', index_col=False)

a_1 = np.transpose(a_df_1.values)
mu_1 = np.transpose(mu_df_1.values)
var_1 = np.transpose(var_df_1.values)
b_1 = np.transpose(b_df_1.values)
a_1h = np.transpose(a_df_1h.values)
mu_1h = np.transpose(mu_df_1h.values)
var_1h = np.transpose(var_df_1h.values)
b_1h = np.transpose(b_df_1h.values)
a_2 = np.transpose(a_df_2.values)
mu_2 = np.transpose(mu_df_2.values)
var_2 = np.transpose(var_df_2.values)
b_2 = np.transpose(b_df_2.values)


v_var_1 = pd.read_csv('v_var_tau_10_1.csv', index_col=False)
v_var_1h = pd.read_csv('v_var_w_1.5.csv', index_col=False)
v_var_2 = pd.read_csv('v_var_w_2.csv', index_col=False)

v_variance_1 = np.mean(v_var_1, axis=0)
v_variance_1h = np.mean(v_var_1h, axis=0)
v_variance_2 = np.mean(v_var_2, axis=0)


a_mean_1 = np.mean(a_1, axis=0)
mu_mean_1 = np.mean(mu_1, axis=0)
var_mean_1 = np.mean(var_1, axis=0)
b_mean_1 = np.mean(b_1, axis=0)
a_mean_1h = np.mean(a_1h, axis=0)
mu_mean_1h = np.mean(mu_1h, axis=0)
var_mean_1h = np.mean(var_1h, axis=0)
b_mean_1h = np.mean(b_1h, axis=0)
a_mean_2 = np.mean(a_2, axis=0)
mu_mean_2 = np.mean(mu_2, axis=0)
var_mean_2 = np.mean(var_2, axis=0)
b_mean_2 = np.mean(b_2, axis=0)

a_var_1 = np.var(a_1, axis=0)
mu_var_1 = np.var(mu_1, axis=0)
var_var_1 = np.var(var_1, axis=0)
b_var_1 = np.var(b_1, axis=0)
a_var_1h = np.var(a_1h, axis=0)
mu_var_1h = np.var(mu_1h, axis=0)
var_var_1h = np.var(var_1h, axis=0)
b_var_1h = np.var(b_1h, axis=0)
a_var_2 = np.var(a_2, axis=0)
mu_var_2 = np.var(mu_2, axis=0)
var_var_2 = np.var(var_2, axis=0)
b_var_2 = np.var(b_2, axis=0)


N_neuron = 100
N_point = 23000
x = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
t_tick = np.arange(0, t3, 0.1)


x_mtx = np.reshape(np.repeat(x, N_point), (N_neuron, N_point))
a_mean_mtx_1 = np.reshape(np.repeat(a_mean_1, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_1 = np.reshape(np.repeat(mu_mean_1, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_1 = np.reshape(np.repeat(var_mean_1, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_1 = np.reshape(np.repeat(b_mean_1, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_1h = np.reshape(np.repeat(a_mean_1h, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_1h = np.reshape(np.repeat(mu_mean_1h, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_1h = np.reshape(np.repeat(var_mean_1h, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_1h = np.reshape(np.repeat(b_mean_1h, N_neuron), (N_point, N_neuron)).T
a_mean_mtx_2 = np.reshape(np.repeat(a_mean_2, N_neuron), (N_point, N_neuron)).T
mu_mean_mtx_2 = np.reshape(np.repeat(mu_mean_2, N_neuron), (N_point, N_neuron)).T
var_mean_mtx_2 = np.reshape(np.repeat(var_mean_2, N_neuron), (N_point, N_neuron)).T
b_mean_mtx_2 = np.reshape(np.repeat(b_mean_2, N_neuron), (N_point, N_neuron)).T

mu_grad_1 = modelfit.grad_mu(x_mtx, a_mean_mtx_1, mu_mean_mtx_1, var_mean_mtx_1, b_mean_mtx_1)
mu_grad_1h = modelfit.grad_mu(x_mtx, a_mean_mtx_1h, mu_mean_mtx_1h, var_mean_mtx_1h, b_mean_mtx_1h)
mu_grad_2 = modelfit.grad_mu(x_mtx, a_mean_mtx_2, mu_mean_mtx_2, var_mean_mtx_2, b_mean_mtx_2)

mu_grad_1_sqr_mean = np.mean(np.square(mu_grad_1), axis=0)
mu_grad_1h_sqr_mean = np.mean(np.square(mu_grad_1h), axis=0)
mu_grad_2_sqr_mean = np.mean(np.square(mu_grad_2), axis=0)


font = {'size': 18}
plt.rc('font', **font)

rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False}
plt.rcParams.update(rc)


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

ax1.plot(t_tick, mu_grad_1_sqr_mean, color='limegreen', label='c=1')
ax1.plot(t_tick, mu_grad_1h_sqr_mean, color='forestgreen', label='c=1.5')
ax1.plot(t_tick, mu_grad_2_sqr_mean, color='darkgreen', label='c=2')
ax1.set_ylabel(r'$(\alpha_{\mu})^{2}$')

ax2.plot(t_tick, mu_var_1, color='limegreen')
ax2.plot(t_tick, mu_var_1h, color='forestgreen')
ax2.plot(t_tick, mu_var_2, color='darkgreen')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')

ax3.plot(t_tick, mu_grad_1_sqr_mean*mu_var_1, color='limegreen', label='c=1')
ax3.plot(t_tick, mu_grad_1h_sqr_mean*mu_var_1h, color='forestgreen', label='c=1.5')
ax3.plot(t_tick, mu_grad_2_sqr_mean*mu_var_2, color='darkgreen', label='c=2')
ax3.plot(t_tick, v_variance_1, color='darkgrey', label='c=1')
ax3.plot(t_tick, v_variance_1h, color='grey', label='c=1.5')
ax3.plot(t_tick, v_variance_2, color='dimgrey', label='c=2')
ax3.set_ylabel(r'$(\alpha_{\mu})^{2} \sigma^{2}_{\mu}$')

ax3.set_xlabel('t / ms')
ax1.legend()
ax3.legend(ncol=2, loc='upper right')

plt.tight_layout()

fig1.savefig("network weight")


start_time = int(rest_time1 / 0.1)
end_time = int((rest_time1+stim_time) / 0.1)
stim_x = np.arange(0, end_time-start_time)
params_guess = np.array([2, 100])
stim_ticks = np.arange(0, stim_time, 0.1)


stim_mu_coeff_1 = mu_grad_1_sqr_mean[start_time:end_time]
stim_mu_coeff_1h = mu_grad_1h_sqr_mean[start_time:end_time]
stim_mu_coeff_2 = mu_grad_2_sqr_mean[start_time:end_time]
stim_mu_coeff_params_1 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_1)).x
stim_mu_coeff_params_1h = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_1h)).x
stim_mu_coeff_params_2 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_2)).x


stim_mu_var_1 = mu_var_1[start_time:end_time]
stim_mu_var_1h = mu_var_1h[start_time:end_time]
stim_mu_var_2 = mu_var_2[start_time:end_time]
norm_stim_mu_var_1 = np.max(stim_mu_var_1) - stim_mu_var_1
norm_stim_mu_var_1h = np.max(stim_mu_var_1h) - stim_mu_var_1h
norm_stim_mu_var_2 = np.max(stim_mu_var_2) - stim_mu_var_2
stim_mu_var_params_1 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_1)).x
stim_mu_var_params_1h = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_1h)).x
stim_mu_var_params_2 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_2)).x
stim_mu_var_est_1 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_1)) - modelfit.exp_response(stim_x, stim_mu_var_params_1)
stim_mu_var_est_1h = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_1h)) - modelfit.exp_response(stim_x, stim_mu_var_params_1h)
stim_mu_var_est_2 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_2)) - modelfit.exp_response(stim_x, stim_mu_var_params_2)


fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

ax1.plot(stim_ticks, stim_mu_coeff_1, color='limegreen')
ax1.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_1), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_1[1]*0.1))+'ms')
ax1.get_xaxis().set_visible(False)
ax1.axvline(x=stim_mu_coeff_params_1[1]*0.1, color='lightgrey')
ax1.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax1.legend()

ax2.plot(stim_ticks, stim_mu_var_1, color='limegreen')
ax2.plot(stim_ticks, stim_mu_var_est_1, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_1[1]*0.1))+'ms')
ax2.get_xaxis().set_visible(False)
ax2.axvline(x=stim_mu_var_params_1[1]*0.1, color='lightgrey')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax2.legend()

ax3.plot(stim_ticks, stim_mu_coeff_1h, color='forestgreen')
ax3.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_1h), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_1h[1]*0.1))+'ms')
ax3.get_xaxis().set_visible(False)
ax3.axvline(x=stim_mu_coeff_params_1h[1]*0.1, color='lightgrey')
ax3.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax3.legend()

ax4.plot(stim_ticks, stim_mu_var_1h, color='forestgreen')
ax4.plot(stim_ticks, stim_mu_var_est_1h, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_1h[1]*0.1))+'ms')
ax4.get_xaxis().set_visible(False)
ax4.axvline(x=stim_mu_var_params_1h[1]*0.1, color='lightgrey')
ax4.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax4.legend()

ax5.plot(stim_ticks, stim_mu_coeff_2, color='darkgreen')
ax5.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_2), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_2[1]*0.1))+'ms')
ax5.axvline(x=stim_mu_coeff_params_2[1]*0.1, color='lightgrey')
ax5.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax5.legend()
ax5.set_xlabel('t / ms')

ax6.plot(stim_ticks, stim_mu_var_2, color='darkgreen')
ax6.plot(stim_ticks, stim_mu_var_est_2, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_2[1]*0.1))+'ms')
ax6.axvline(x=stim_mu_var_params_2[1]*0.1, color='lightgrey')
ax6.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax6.legend()
ax6.set_xlabel('t / ms')

plt.tight_layout()

fig2.savefig("fit constant network weight")
