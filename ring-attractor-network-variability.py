import time
import numpy as np
import matplotlib.pyplot as plt
import model


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

t_int = attractor_model.t_int
prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

N_stim = 4
N_trial = 150

var_matrix = np.zeros([N_stim, N_point])
for k in range(N_stim):
    V_matrix = np.zeros([N_trial, N_point, N_neuron])
    for i in range(N_trial):
        V = attractor_model.init_voltage()
        noise = attractor_model.init_noise()
        trail_start = time.time()
        c = 0
        for j in range(int(prep_time/t_int)):
            V_in = V
            noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
            V = attractor_model.update(V_in, noise, 0)
            noise = noise_in
        for j in range(int(rest_time1/t_int)):
            V_in = V
            noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
            V = attractor_model.update(V_in, noise, 0)
            noise = noise_in
            V_matrix[i][c] = V
            c += 1
        for j in range(int(stim_time/t_int)):
            V_in = V
            noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
            V = attractor_model.update(V_in, noise, k)
            noise = noise_in
            V_matrix[i][c] = V
            c += 1
        for j in range(int(rest_time2/t_int)):
            V_in = V
            noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
            V = attractor_model.update(V_in, noise, 0)
            noise = noise_in
            V_matrix[i][c] = V
            c += 1
        print("stimulus level {}, {}th trial finished, took {} seconds".format(
            str(k), str(i+1), time.time()-trail_start))
    var = np.var(V_matrix, axis=0)
    var_matrix[k] = np.sqrt(np.mean(var, axis=1))

fig, ax = plt.subplots(figsize=(16, 5.5))
lines = []
lines += ax.plot(t, var_matrix[0], label='c=0', color='dimgrey')
lines += ax.plot(t, var_matrix[1], label='c=1', color='darkgreen')
lines += ax.plot(t, var_matrix[2], label='c=2', color='forestgreen')
lines += ax.plot(t, var_matrix[3], label='c=3', color='limegreen')
ax.legend(lines, [l.get_label() for l in lines], frameon=False)
ax.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='lightgrey')
ax.set_ylabel(r'$\sqrt{ \langle{ Var_{n}[V] \rangle }_{i}}$ [mV]')
ax.set_xlabel('t [ms]')
plt.tight_layout()
fig.savefig('multiple_stim_lvl')
