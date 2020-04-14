import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelfit
from scipy.optimize import minimize


import os
"path for simulation data"
os.chdir('/Users/yuetongyc/Desktop/Cambridge/IIB/Project//data')


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
mu_mean_10 = np.mean(mu_10, axis=0)
var_mean_10 = np.mean(var_10, axis=0)
b_mean_10 = np.mean(b_10, axis=0)
a_mean_20 = np.mean(a_20, axis=0)
mu_mean_20 = np.mean(mu_20, axis=0)
var_mean_20 = np.mean(var_20, axis=0)
b_mean_20 = np.mean(b_20, axis=0)
a_mean_30 = np.mean(a_30, axis=0)
mu_mean_30 = np.mean(mu_30, axis=0)
var_mean_30 = np.mean(var_30, axis=0)
b_mean_30 = np.mean(b_30, axis=0)

a_var_10 = np.var(a_10, axis=0)
mu_var_10 = np.var(mu_10, axis=0)
var_var_10 = np.var(var_10, axis=0)
b_var_10 = np.var(b_10, axis=0)
a_var_20 = np.var(a_20, axis=0)
mu_var_20 = np.var(mu_20, axis=0)
var_var_20 = np.var(var_20, axis=0)
b_var_20 = np.var(b_20, axis=0)
a_var_30 = np.var(a_30, axis=0)
mu_var_30 = np.var(mu_30, axis=0)
var_var_30 = np.var(var_30, axis=0)
b_var_30 = np.var(b_30, axis=0)


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

mu_grad_10 = modelfit.grad_mu(x_mtx, a_mean_mtx_10, mu_mean_mtx_10, var_mean_mtx_10, b_mean_mtx_10)
mu_grad_20 = modelfit.grad_mu(x_mtx, a_mean_mtx_20, mu_mean_mtx_20, var_mean_mtx_20, b_mean_mtx_20)
mu_grad_30 = modelfit.grad_mu(x_mtx, a_mean_mtx_30, mu_mean_mtx_30, var_mean_mtx_30, b_mean_mtx_30)

mu_grad_10_sqr_mean = np.mean(np.square(mu_grad_10), axis=0)
mu_grad_20_sqr_mean = np.mean(np.square(mu_grad_20), axis=0)
mu_grad_30_sqr_mean = np.mean(np.square(mu_grad_30), axis=0)


font = {'size': 18}
plt.rc('font', **font)

rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False}
plt.rcParams.update(rc)


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

ax1.plot(t_tick, mu_grad_10_sqr_mean, color='limegreen', label=r'$\tau$=10ms')
ax1.plot(t_tick, mu_grad_20_sqr_mean, color='forestgreen', label=r'$\tau$=20ms')
ax1.plot(t_tick, mu_grad_30_sqr_mean, color='darkgreen', label=r'$\tau$=30ms')
ax1.set_ylabel(r'$(\alpha_{\mu})^{2}$')

ax2.plot(t_tick, mu_var_10, color='limegreen')
ax2.plot(t_tick, mu_var_20, color='forestgreen')
ax2.plot(t_tick, mu_var_30, color='darkgreen')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')

ax3.plot(t_tick, mu_grad_10_sqr_mean*mu_var_10, color='limegreen', label=r'$\tau$=10ms')
ax3.plot(t_tick, mu_grad_20_sqr_mean*mu_var_20, color='forestgreen', label=r'$\tau$=20ms')
ax3.plot(t_tick, mu_grad_30_sqr_mean*mu_var_30, color='darkgreen', label=r'$\tau$=30ms')
ax3.plot(t_tick, v_variance_10, color='darkgrey', label=r'$\tau$=10ms')
ax3.plot(t_tick, v_variance_20, color='grey', label=r'$\tau$=20ms')
ax3.plot(t_tick, v_variance_30, color='dimgrey', label=r'$\tau$=30ms')
ax3.set_ylabel(r'$(\alpha_{\mu})^{2} \sigma^{2}_{\mu}$')
ax3.set_xlabel('t / ms')

ax1.legend()
ax3.legend(ncol=2, loc='upper right')
plt.tight_layout()

fig1.savefig("time constant")


start_time = int(rest_time1 / 0.1)
end_time = int((rest_time1+stim_time) / 0.1)
stim_x = np.arange(0, end_time-start_time)
params_guess = np.array([2, 100])
stim_ticks = np.arange(0, stim_time, 0.1)


stim_mu_coeff_10 = mu_grad_10_sqr_mean[start_time:end_time]
stim_mu_coeff_20 = mu_grad_20_sqr_mean[start_time:end_time]
stim_mu_coeff_30 = mu_grad_30_sqr_mean[start_time:end_time]
stim_mu_coeff_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_10)).x
stim_mu_coeff_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_20)).x
stim_mu_coeff_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, stim_mu_coeff_30)).x


stim_mu_var_10 = mu_var_10[start_time:end_time]
stim_mu_var_20 = mu_var_20[start_time:end_time]
stim_mu_var_30 = mu_var_30[start_time:end_time]
norm_stim_mu_var_10 = np.max(stim_mu_var_10) - stim_mu_var_10
norm_stim_mu_var_20 = np.max(stim_mu_var_20) - stim_mu_var_20
norm_stim_mu_var_30 = np.max(stim_mu_var_30) - stim_mu_var_30
stim_mu_var_params_10 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_10)).x
stim_mu_var_params_20 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_20)).x
stim_mu_var_params_30 = minimize(modelfit.exp_response_mse, params_guess, args=(stim_x, norm_stim_mu_var_30)).x
stim_mu_var_est_10 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_10)) - modelfit.exp_response(stim_x, stim_mu_var_params_10)
stim_mu_var_est_20 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_20)) - modelfit.exp_response(stim_x, stim_mu_var_params_20)
stim_mu_var_est_30 = np.max(modelfit.exp_response(stim_x, stim_mu_var_params_30)) - modelfit.exp_response(stim_x, stim_mu_var_params_30)


fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

ax1.plot(stim_ticks, stim_mu_coeff_10, color='limegreen')
ax1.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_10), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_10[1]*0.1))+'ms')
ax1.get_xaxis().set_visible(False)
ax1.axvline(x=stim_mu_coeff_params_10[1]*0.1, color='lightgrey')
ax1.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax1.legend()

ax2.plot(stim_ticks, stim_mu_var_10, color='limegreen')
ax2.plot(stim_ticks, stim_mu_var_est_10, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_10[1]*0.1))+'ms')
ax2.get_xaxis().set_visible(False)
ax2.axvline(x=stim_mu_var_params_10[1]*0.1, color='lightgrey')
ax2.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax2.legend()

ax3.plot(stim_ticks, stim_mu_coeff_20, color='forestgreen')
ax3.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_20), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_20[1]*0.1))+'ms')
ax3.get_xaxis().set_visible(False)
ax3.axvline(x=stim_mu_coeff_params_20[1]*0.1, color='lightgrey')
ax3.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax3.legend()

ax4.plot(stim_ticks, stim_mu_var_20, color='forestgreen')
ax4.plot(stim_ticks, stim_mu_var_est_20, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_20[1]*0.1))+'ms')
ax4.get_xaxis().set_visible(False)
ax4.axvline(x=stim_mu_var_params_20[1]*0.1, color='lightgrey')
ax4.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax4.legend()

ax5.plot(stim_ticks, stim_mu_coeff_30, color='darkgreen')
ax5.plot(stim_ticks, modelfit.exp_response(stim_x, stim_mu_coeff_params_30), color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_coeff_params_30[1]*0.1))+'ms')
ax5.axvline(x=stim_mu_coeff_params_30[1]*0.1, color='lightgrey')
ax5.set_ylabel(r'$(\alpha_{\mu})^{2}$')
ax5.legend()
ax5.set_xlabel('t / ms')

ax6.plot(stim_ticks, stim_mu_var_30, color='darkgreen')
ax6.plot(stim_ticks, stim_mu_var_est_30, color='dimgrey', label=r'$\hat{\tau}=$'+str(int(stim_mu_var_params_30[1]*0.1))+'ms')
ax6.axvline(x=stim_mu_var_params_30[1]*0.1, color='lightgrey')
ax6.set_ylabel(r'$\sigma^{2}_{\mu}$')
ax6.legend()
ax6.set_xlabel('t / ms')

plt.tight_layout()

fig2.savefig("fit time constant")
