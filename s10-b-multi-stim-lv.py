import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import model

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)

t_int = attractor_model.t_int
prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

N_stim = 6
N_trial = 200

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

fig, ax = plt.subplots()
lines = []
lines += ax.plot(t, var_matrix[0], label='c=0', color='dimgrey', linewidth=2)
lines += ax.plot(t, var_matrix[1], label='c=1', color='darkgreen', linewidth=2)
lines += ax.plot(t, var_matrix[2], label='c=2', color='green', linewidth=2)
lines += ax.plot(t, var_matrix[3], label='c=3', color='mediumseagreen', linewidth=2)
lines += ax.plot(t, var_matrix[4], label='c=4', color='limegreen', linewidth=2)
lines += ax.plot(t, var_matrix[5], label='c=5', color='lightgreen', linewidth=2)
ax.legend(lines, [l.get_label() for l in lines], frameon=False)
ax.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='lightgrey')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel(r'$\sqrt{ \langle{ Var_{trials}(V) \rangle }_{neurons}}$ / mV', fontsize=14)
ax.set_xlabel('t / ms', fontsize=14)
ax.tick_params(labelsize=12)
plt.show()
