import model
import numpy as np
import matplotlib.pyplot as plt

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

V_matrix = np.zeros([N_neuron, N_point])

c = 0
for i in range(int(rest_time1 / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(int(stim_time / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 2)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(int(rest_time2 / t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1

plt.imshow(V_matrix, interpolation='nearest', aspect='auto')
plt.show()
