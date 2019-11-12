import model
import numpy as np
import matplotlib.pyplot as plt

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()

t_int = attractor_model.t_int
rest_time1 = 1000
stim_time = 2000
rest_time2 = 1500
total_time = rest_time1 + stim_time + rest_time2
N_point = int(total_time/t_int)

V_matrix = np.zeros([N_neuron, N_point])

c = 0
for i in range(int(rest_time1/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(int(stim_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 2)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(int(rest_time2/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1

fig, ax = plt.subplots()
im = plt.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax.set_title('True Neuron Activity Plot')
ax.set_xlabel('t')
ax.set_ylabel('PO')
fig.colorbar(im, ax=ax)
plt.show()
