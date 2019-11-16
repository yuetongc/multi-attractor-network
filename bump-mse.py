import model
import modelfit
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()
ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

t_int = attractor_model.t_int
prep_time = 300
rest_time1 = 500
stim_time = 1000
rest_time2 = 800
total_time = rest_time1 + stim_time + rest_time2

N_point = int(total_time/t_int)

V_matrix = np.zeros([N_neuron, N_point])

N_params = 3
p_est = np.array([2., 1., 0.])
p_matrix = np.zeros([N_params, N_point])
est_matrix = np.zeros([N_neuron, N_point])

time0 = time.time()
for i in range(int(prep_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
print("end of prep phase, took {} seconds".format(time.time() - time0))

c = 0

time1 = time.time()
for i in range(int(rest_time1/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
print("end of rest phase 1, took {} seconds".format(time.time() - time1))

time2 = time.time()
for i in range(int(stim_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 2)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
print("end of stimulus phase, took {} seconds".format(time.time() - time2))

time3 = time.time()
for i in range(int(rest_time2/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V))
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
print("end of rest phase 2, took {} seconds".format(time.time() - time3))


fig1, (ax1, ax2) = plt.subplots(2, 1)
im1 = ax1.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('True Activity')
ax1.set_ylabel(r'PO / $\degree$')
fig1.colorbar(im1, ax=ax1)
im2 = ax2.imshow(est_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Bump Fit')
ax2.set_ylabel(r'PO / $\degree$')
ax2.set_xlabel('t / ms')
fig1.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(p_matrix[0])
ax1.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('a')
ax2.plot(p_matrix[1])
ax2.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel(r'$\mu$')
ax3.plot(np.exp(p_matrix[2]))
ax3.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel(r'$\sigma^{2}$')
ax3.set_xlabel('t / ms')
plt.tight_layout()
plt.show()
