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
rest_time1 = 1000
stim_time = 2000
rest_time2 = 1500
total_time = rest_time1 + stim_time + rest_time2
N_point = int(total_time/t_int)

V_matrix = np.zeros([N_neuron, N_point])

N_params = 3
p_est = np.array([2., 1., 1.])
p_bounds = ((-10, 10), (-10, 10), (-10, 10))
p_matrix = np.zeros([N_params, N_point])
est_matrix = np.zeros([N_neuron, N_point])

time0 = time.time()
for i in range(int(prep_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
print("end of prep phase, took {} seconds".format(time.time() - time0))

sum_est = 0
V_est = V
for i in range(100):
    V_in = V_est
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V_est = attractor_model.update(V_in, noise, 2)
    noise = noise_in
    sum_est = (sum_est*i + modelfit.firing_rate_app(V_est))/(i+1)
sum_est = int(round(sum_est, -2))
print("end of estimation phase, total firing rate approximated as {}".format(sum_est))

c = 0

time1 = time.time()
for i in range(int(rest_time1/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
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
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
    p_est = est.x
    V_fit = modelfit.f_v(ang_vector, p_est)
    p_est_norm = est.x
    p_est_norm[0] = p_est_norm[0] * (sum_est / modelfit.firing_rate_app(V_fit))
    p_matrix[:, c] = p_est_norm
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est_norm)
    c += 1
print("end of stimulus phase, took {} seconds".format(time.time() - time2))

time3 = time.time()
for i in range(int(rest_time2/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
print("end of rest phase 2, took {} seconds".format(time.time() - time3))


fig1, (ax1, ax2) = plt.subplots(2, 1)
im1 = ax1.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax1.set_title('True Activity')
fig1.colorbar(im1, ax=ax1)
im2 = ax2.imshow(est_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax2.set_title('Bump Fit')
fig1.colorbar(im2, ax=ax2)
fig1.suptitle('Contrained MSE Bump Fit')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(p_matrix[0])
ax1.set_title('a')
ax1.set_xticks([rest_time1/t_int, (rest_time1+stim_time)/t_int])
ax1.set_xticklabels(['stimulus ON', 'stimulus OFF'])
ax1.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax2.plot(p_matrix[1])
ax2.set_title(r'$\mu$')
ax2.set_xticks([rest_time1/t_int, (rest_time1+stim_time)/t_int])
ax2.set_xticklabels(['stimulus ON', 'stimulus OFF'])
ax2.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
ax3.plot(p_matrix[2])
ax3.set_title(r'$\sigma^{2}$')
ax3.set_xticks([rest_time1/t_int, (rest_time1+stim_time)/t_int])
ax3.set_xticklabels(['stimulus ON', 'stimulus OFF'])
ax3.axvspan(rest_time1/t_int, (rest_time1+stim_time)/t_int, alpha=0.5, color='yellow')
plt.tight_layout()
plt.show()

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(p_matrix[0][:rest_time1], p_matrix[2][:rest_time1], color='blue', label='before')
ax1.plot(p_matrix[0][rest_time1:rest_time1+stim_time], p_matrix[2][rest_time1:rest_time1+stim_time],
         color='red', label='during')
ax1.plot(p_matrix[0][rest_time1+stim_time:total_time], p_matrix[2][rest_time1+stim_time:total_time],
         color='green', label='after')
ax1.legend()
ax1.set_xlabel('a')
ax1.set_ylabel(r'$\sigma^{2}$')
ax2.plot(p_matrix[1][:rest_time1], p_matrix[0][:rest_time1], color='blue', label='before')
ax2.plot(p_matrix[1][rest_time1:rest_time1+stim_time], p_matrix[0][rest_time1:rest_time1+stim_time],
         color='red', label='during')
ax2.plot(p_matrix[1][rest_time1+stim_time:total_time], p_matrix[0][rest_time1+stim_time:total_time],
         color='green', label='after')
ax2.legend()
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel('a')
ax3.plot(p_matrix[1][:rest_time1], p_matrix[2][:rest_time1], color='blue', label='before')
ax3.plot(p_matrix[1][rest_time1:rest_time1+stim_time], p_matrix[2][rest_time1:rest_time1+stim_time],
         color='red', label='during')
ax3.plot(p_matrix[1][rest_time1+stim_time:total_time], p_matrix[2][rest_time1+stim_time:total_time],
         color='green', label='after')
ax3.legend()
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\sigma^{2}$')
plt.tight_layout()
plt.show()
