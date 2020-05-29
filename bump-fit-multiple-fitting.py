import model
import modelfit
import time
import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


N_neuron = 100

N_trial = 60
N_iter = 100

prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
N_point = int(t3/0.1)
t_tick = np.arange(0, t3, 0.1)

parameter_matrix = np.zeros([N_trial, 4, N_point])
data_matrix = np.zeros([N_trial, N_neuron, N_point])

font = {'size': 24}
plt.rc('font', **font)

rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False}
plt.rcParams.update(rc)

fig1, (ax11, ax12, ax13, ax14, ax15) = plt.subplots(5, 1, figsize=(12, 18))
fig2, (ax21, ax22, ax23) = plt.subplots(3, 1, figsize=(12, 15))
fig3, (ax31, ax32, ax33) = plt.subplots(3, 1, figsize=(12, 15))
fig4, (ax41, ax42, ax43) = plt.subplots(3, 1, figsize=(12, 15))
fig5, (ax51, ax52, ax53) = plt.subplots(3, 1, figsize=(12, 15))

success_trial = 0
for k in range(N_trial):
    print('{}th trial starts'.format(str(k+1)))
    attractor_model = model.MultiAttractorModel(N=N_neuron)
    V = attractor_model.init_voltage()
    noise = attractor_model.init_noise()
    ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

    t_int = attractor_model.t_int
    n0, n1, n2, n3 = int(t0/t_int), int(t1/t_int), int(t2/t_int), int(t3/t_int)
    t = np.arange(0, t3, t_int)
    N_point = n3
    V_matrix = np.zeros([N_neuron, N_point])

    timer0 = time.time()

    for i in range(n0):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
    c = 0
    for i in range(n1):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        V_matrix[:, c] = V
        c += 1
    for i in range(n2-n1):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 2)
        noise = noise_in
        V_matrix[:, c] = V
        c += 1
    for i in range(n3-n2):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        V_matrix[:, c] = V
        c += 1

    print('- simulation finished')

    N_params = 4
    p_matrix = np.zeros([N_params, N_point])
    p_matrix = modelfit.init_p(p_matrix, [2, 0, 0, 0], 0, n3)

    est_matrix = np.zeros([N_neuron, N_point])

    for n in range(N_iter):
        c = 0
        for i in range(n3):
            est_matrix[:, c] = modelfit.f_v_a(ang_vector, p_matrix[1:3, c], p_matrix[0, c])
            b_est2 = np.mean(V_matrix[:, c] - est_matrix[:, c])
            p_matrix[-1, c] = b_est2
            a_est2 = modelfit.opt_a(ang_vector, V_matrix[:, c], p_matrix[1, c], p_matrix[2, c], b_est2)
            p_matrix[0, c] = a_est2
            est = minimize(modelfit.mse_fv_baseline_a, p_matrix[1:3, c],
                           args=(ang_vector, V_matrix[:, c], a_est2, b_est2))
            p_est2 = est.x
            p_matrix[1:3, c] = p_est2
            c += 1
        print('--- {}th estimation finished, the last a, mean, log var, b estimates are  {}'.
              format(str(n + 1), p_matrix[:, c - 1]))

    timer1 = time.time()
    print('{}th trial finished, took {} seconds'.format(str(k+1), str(timer1-timer0)))

    est_matrix = modelfit.update_p(est_matrix, p_matrix, ang_vector, 0, n3)

    r2 = []
    for i in range(n3):
        r2.append(modelfit.r_squared(V_matrix[:, i], est_matrix[:, i]))

    if np.all(np.exp(p_matrix[2, :]) < 20) and np.all(p_matrix[0, :] > 0):
        parameter_matrix[success_trial, :, :] = p_matrix
        data_matrix[success_trial, :, :] = V_matrix
        print("{}th simulation plotted".format(str(k+1)))
        ax11.plot(t, p_matrix[0, :], color='black')
        ax11.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax11.set_ylabel(r'$\hat{a}\ [mV]$')
        ax11.get_xaxis().set_visible(False)
        ax12.plot(t, 180 * (p_matrix[1, :] / math.pi), color='black')
        ax12.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax12.set_ylim(-math.pi, math.pi)
        ax12.set_yticks([-180, 0, 180])
        ax12.set_ylabel(r'$\hat{\mu}\ [\degree]$')
        ax12.get_xaxis().set_visible(False)
        ax13.plot(t, np.exp(p_matrix[2, :]), color='black')
        ax13.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax13.set_ylabel(r'$\hat{w^{2}}$')
        ax13.get_xaxis().set_visible(False)
        ax14.plot(t, p_matrix[3, :], color='black')
        ax14.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax14.set_ylabel(r'$\hat{b}$\ [mV]')
        ax14.get_xaxis().set_visible(False)
        ax15.plot(t, r2, color='black')
        ax15.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax15.set_ylabel(r'$R^{2}$')
        ax15.set_xlabel('t [ms]')

        ax21.plot(t, p_matrix[0, :], color='black')
        ax21.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax21.set_ylabel(r'$\hat{a}\ [mV]$')
        ax21.get_xaxis().set_visible(False)
        ax31.plot(t, 180 * (p_matrix[1, :] / math.pi), color='black')
        ax31.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax31.set_ylim(-math.pi, math.pi)
        ax31.set_yticks([-180, 0, 180])
        ax31.set_ylabel(r'$\hat{\mu}\ [\degree]$')
        ax31.get_xaxis().set_visible(False)
        ax41.plot(t, np.exp(p_matrix[2, :]), color='black')
        ax41.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax41.set_ylabel(r'$\hat{w^{2}}$')
        ax41.get_xaxis().set_visible(False)
        ax51.plot(t, p_matrix[3, :], color='black')
        ax51.axvspan(t1, t2, alpha=0.5, color='lightgrey')
        ax51.set_ylabel(r'$\hat{b}\ [mV]$')
        ax51.get_xaxis().set_visible(False)

        success_trial += 1

data_matrix = data_matrix[~(data_matrix == 0).all(axis=(1, 2))]
V_var = np.var(data_matrix, axis=0)
V_var_df = pd.DataFrame(V_var)
V_var_df.to_csv('v_var.csv', index=False)

parameter_matrix = parameter_matrix[~(parameter_matrix == 0).all(axis=(1, 2))]

a_df = pd.DataFrame(np.transpose(parameter_matrix[:, 0, :]), index=t_tick)
a_df.to_csv('a.csv', index=False)
mu_df = pd.DataFrame(np.transpose(parameter_matrix[:, 1, :]), index=t_tick)
mu_df.to_csv('mu.csv', index=False)
var_df = pd.DataFrame(np.transpose(np.exp(parameter_matrix[:, 2, :])), index=t_tick)
var_df.to_csv('var.csv', index=False)
b_df = pd.DataFrame(np.transpose(parameter_matrix[:, 3, :]), index=t_tick)
b_df.to_csv('b.csv', index=False)

mu_mean = np.apply_along_axis(modelfit.circular_mean, 0, parameter_matrix[:, 1, :])
mu_var = np.apply_along_axis(modelfit.circular_variance, 0, parameter_matrix[:, 1, :])

plt.tight_layout()
plt.show()
fig1.savefig('Stimulus Parameter Multiple Trial.png')

ax22.plot(t_tick, np.mean(parameter_matrix[:, 0, :], axis=0).tolist(), color='green')
ax22.get_xaxis().set_visible(False)
ax22.set_ylabel(r'$\mu_{trial}$')
ax22.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax23.plot(t_tick, np.var(parameter_matrix[:, 0, :], axis=0).tolist(), color='blue')
ax23.set_ylabel(r'$\sigma^{2}_{trial}$')
ax23.axvspan(t1, t2, alpha=0.5, color='lightgrey')
plt.show()
fig2.savefig('Parameter a.png')

ax32.plot(t_tick, 180 * (mu_mean / math.pi), color='green')
ax32.set_yticks([-180, 0, 180])
ax32.set_ylabel(r'$\mu_{trial}\ [\degree]$')
ax32.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax33.plot(t_tick, mu_var, color='blue')
ax33.set_ylabel(r'$\sigma^{2}_{trial}$')
ax33.axvspan(t1, t2, alpha=0.5, color='lightgrey')
plt.show()
fig3.savefig('Parameter mu.png')

ax42.plot(t_tick, np.mean(np.exp(parameter_matrix[:, 2, :]), axis=0).tolist(), color='green')
ax42.set_ylabel(r'$\mu_{trial}$')
ax42.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax43.plot(t_tick, np.var(np.exp(parameter_matrix[:, 2, :]), axis=0).tolist(), color='blue', label=r'$\sigma^{2}_{trial}$')
ax43.set_ylabel(r'$\sigma^{2}_{trial}$')
ax43.axvspan(t1, t2, alpha=0.5, color='lightgrey')
plt.show()
fig4.savefig('Parameter sig.png')

ax52.plot(t_tick, np.mean(parameter_matrix[:, 3, :], axis=0).tolist(), color='green', label=r'$\mu_{trial}$')
ax52.set_ylabel(r'$\mu_{trial}$')
ax52.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax53.plot(t_tick, np.var(parameter_matrix[:, 3, :], axis=0).tolist(), color='blue', label=r'$\sigma^{2}_{trial}$')
ax53.set_ylabel(r'$\sigma^{2}_{trial}$')
ax53.axvspan(t1, t2, alpha=0.5, color='lightgrey')
plt.show()
fig5.savefig('Parameter b.png')
