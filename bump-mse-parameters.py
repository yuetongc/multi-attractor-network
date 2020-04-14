import model
import modelfit
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

t_int = attractor_model.t_int
prep_time = 30
rest_time1 = 5
stim_time = 10
rest_time2 = 8
total_time = rest_time1 + stim_time + rest_time2
N_point = int(total_time/t_int)

N_params = 3
p_est = np.array([2., 1., 0.])

N_trail = 5
p_matrix = np.zeros([N_trail, N_params, N_point])
est_matrix = np.zeros([N_trail, N_neuron, N_point])
V_matrix = np.zeros([N_trail, N_neuron, N_point])

for n in range(N_trail):
    V = attractor_model.init_voltage()
    noise = attractor_model.init_noise()
    trail_start = time.time()
    for i in range(int(prep_time/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
    c = 0
    for i in range(int(rest_time1/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        V_matrix[n, :, c] = V
        est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
        p_est = est.x
        p_matrix[n, :, c] = p_est
        est_matrix[n, :, c] = modelfit.f_v(ang_vector, p_est)
        c += 1
    for i in range(int(stim_time/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 2)
        noise = noise_in
        V_matrix[n, :, c] = V
        est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
        p_est = est.x
        p_matrix[n, :, c] = p_est
        est_matrix[n, :, c] = modelfit.f_v(ang_vector, p_est)
        c += 1
    for i in range(int(rest_time2/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        V_matrix[n, :, c] = V
        est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
        p_est = est.x
        p_matrix[n, :, c] = p_est
        est_matrix[n, :, c] = modelfit.f_v(ang_vector, p_est)
        c += 1
    print("{}th trial finished, took {} seconds".format(str(n+1), time.time() - trail_start))


fig1, ax = plt.subplots()
idx = np.random.choice(np.arange(N_point*N_trail*N_params), 1000, replace=False)
ax.scatter(V_matrix.flatten()[idx], est_matrix.flatten()[idx])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel(r'$\hat{V}$ / mV')
ax.set_xlabel('V / mV')
plt.show()

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
for n in range(N_trail):
    ax1.plot(p_matrix[n, 0], color='blue')
ax1.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('a')
for n in range(N_trail):
    ax2.plot(p_matrix[n, 1], color='blue')
ax2.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel(r'$\mu$')
for n in range(N_trail):
    ax3.plot(np.exp(p_matrix[n, 2]), color='blue')
ax3.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel(r'$\sigma^{2}$')
ax3.set_xlabel('t / ms')
plt.tight_layout()
plt.show()

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1)
lines1 = []
for n in range(N_trail):
    lines1 += ax1.plot(p_matrix[n, 0][:rest_time1], np.exp(p_matrix[n, 2][:rest_time1]), color='blue', label='before')
    lines1 += ax1.plot(p_matrix[n, 0][rest_time1:rest_time1+stim_time],
                       np.exp(p_matrix[n, 2][rest_time1:rest_time1+stim_time]), color='red', label='during')
    lines1 += ax1.plot(p_matrix[n, 0][rest_time1+stim_time:total_time],
                       np.exp(p_matrix[n, 2][rest_time1+stim_time:total_time]), color='green', label='after')
ax1.legend(lines1, [lines1[0].get_label(), lines1[1].get_label(), lines1[2].get_label()])
ax1.set_xlabel('a')
ax1.set_ylabel(r'$\sigma^{2}$')
lines2 = []
for n in range(N_trail):
    lines2 += ax2.plot(p_matrix[n, 1][:rest_time1], p_matrix[n, 0][:rest_time1], color='blue', label='before')
    lines2 += ax2.plot(p_matrix[n, 1][rest_time1:rest_time1+stim_time], p_matrix[n, 0][rest_time1:rest_time1+stim_time],
                       color='red', label='during')
    lines2 += ax2.plot(p_matrix[n, 1][rest_time1+stim_time:total_time], p_matrix[n, 0][rest_time1+stim_time:total_time],
                       color='green', label='after')
ax2.legend(lines2, [lines2[0].get_label(), lines2[1].get_label(), lines2[2].get_label()])
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel('a')
lines3 = []
for n in range(N_trail):
    lines3 += ax3.plot(p_matrix[n, 1][:rest_time1], np.exp(p_matrix[n, 2][:rest_time1]), color='blue', label='before')
    lines3 += ax3.plot(p_matrix[n, 1][rest_time1:rest_time1+stim_time],
                       np.exp(p_matrix[n, 2][rest_time1:rest_time1+stim_time]), color='red', label='during')
    lines3 += ax3.plot(p_matrix[n, 1][rest_time1+stim_time:total_time],
                       np.exp(p_matrix[n, 2][rest_time1+stim_time:total_time]), color='green', label='after')
ax3.legend(lines3, [lines3[0].get_label(), lines3[1].get_label(), lines3[2].get_label()])
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\sigma^{2}$')
plt.tight_layout()
plt.show()
