import model
import numpy as np
import matplotlib.pyplot as plt


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


N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()

t_int = attractor_model.t_int
prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
total_time = rest_time1 + stim_time + rest_time2
N_point = int(total_time/t_int)

V_matrix = np.zeros([N_neuron, N_point])

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

fig, ax = plt.subplots(figsize=(16, 5))
im = plt.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, total_time, -180, 180))
ax.set_yticks([-180, 0, 180])
ax.set_ylabel(r'PO [$\degree$]')
ax.set_xlabel('t [ms]')
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig('evoked_activity')
