import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import model

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)

t_int = attractor_model.t_int
prep_time = 300
rest_time1 = 500
stim_time = 1000
rest_time2 = 800
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

N_stim = 4
N_trial = 30

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
for k in range(N_stim):
    lines += ax.plot(t, var_matrix[k], label='c={}'.format(str(k)))
ax.legend(lines, [l.get_label() for l in lines])
ax.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='yellow')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel(r'$\sqrt{ \langle{ Var_{trials}(V) \rangle }_{neurons}}$ / mV')
ax.set_xlabel('t / ms')
plt.show()
