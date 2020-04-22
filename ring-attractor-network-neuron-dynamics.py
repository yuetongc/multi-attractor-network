import time
import numpy as np
import matplotlib.pyplot as plt
import model


rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": True,
      "axes.spines.top": False,
      "lines.linewidth": 2,
      "xtick.labelsize": 24,
      "ytick.labelsize": 24,
      "axes.labelsize": 28,
      }
plt.rcParams.update(rc)


N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)

t_int = attractor_model.t_int
prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

N_trail = 10
val1 = np.zeros([N_trail, N_point])
val2 = np.zeros([N_trail, N_point])
val3 = np.zeros([N_trail, N_point])

for i in range(N_trail):
    running_time = time.time()
    V = attractor_model.init_voltage()
    noise = attractor_model.init_noise()
    for j in range(int(prep_time / t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
    c = 0
    for j in range(int(rest_time1/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        val1[i, c] = V[50]
        val2[i, c] = V[25]
        val3[i, c] = V[0]
        c += 1
    for j in range(int(stim_time/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 2)
        noise = noise_in
        val1[i, c] = V[50]
        val2[i, c] = V[25]
        val3[i, c] = V[0]
        c += 1
    for j in range(int(rest_time2/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
        V = attractor_model.update(V_in, noise, 0)
        noise = noise_in
        val1[i, c] = V[50]
        val2[i, c] = V[25]
        val3[i, c] = V[0]
        c += 1
    print("{}th run finished, took {} seconds".format(str(i+1), time.time() - running_time))


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

for i in range(N_trail):
    ax1.plot(t, val1[i], color='black')
ax1.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='lightgrey')
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel(r'$V_{stim}$ [mV]')

for i in range(N_trail):
    ax2.plot(t, val2[i], color='black')
ax2.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='lightgrey')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel(r'$V_{orth}$ [mV]')

for i in range(N_trail):
    ax3.plot(t, val3[i], color='black')
ax3.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='lightgrey')
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel(r'$V_{opp}$ [mV]')
ax3.set_xlabel('t [ms]')

plt.tight_layout()
fig.savefig('neuron_dynamics')
