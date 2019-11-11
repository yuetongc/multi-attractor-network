import model
import modelfit
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N_neuron = 100

t_int = 0.1
rest_time1 = 1000
stim_time = 2000
rest_time2 = 1000
total_time = rest_time1 + stim_time + rest_time2
N_point = int(total_time/t_int)

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()
ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

V_matrix = np.zeros([N_neuron, N_point])

N_params = 3
p_est = np.array([2., 1., 1.])
p_bounds = ((-10, 10), (-10, 10), (-10, 10))
p_matrix = np.zeros([N_params, N_point])
est_matrix = np.zeros([N_neuron, N_point])

c = 0
for i in range(int(rest_time1 / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
for i in range(int(stim_time / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 2)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1
for i in range(int(rest_time2 / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    V_matrix[:, c] = V
    est = minimize(modelfit.mse_fv, p_est, args=(ang_vector, V), bounds=p_bounds)
    p_est = est.x
    p_matrix[:, c] = p_est
    est_matrix[:, c] = modelfit.f_v(ang_vector, p_est)
    c += 1


plt.imshow(V_matrix, interpolation='nearest', aspect='auto')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(p_matrix[0])
ax2.plot(p_matrix[1])
ax3.plot(p_matrix[2])
plt.show()

plt.imshow(est_matrix, interpolation='nearest', aspect='auto')
plt.show()
