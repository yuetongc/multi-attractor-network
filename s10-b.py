import time
import numpy as np
import matplotlib.pyplot as plt
import model

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()

t_int = attractor_model.t_int
rest_time1 = 1000
stim_time = 2000
rest_time2 = 1500
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

N_trial = 10
V_matrix = np.zeros([N_trial, N_point, N_neuron])

N_run = 1
running_avg_var = np.zeros(N_point)

for run in range(N_run):
    for i in range(N_trial):
        trail_start = time.time()
        c = 0
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
            V = attractor_model.update(V_in, noise, 2)
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
        print("{}tn run {}th trial finished, took {} seconds".format(str(run+1), str(i+1), time.time()-trail_start))
    var = np.sqrt(np.var(V_matrix, axis=0))
    avg_var = np.mean(var, axis=1)
    running_avg_var = (running_avg_var*i + avg_var)/(i+1)

fig, ax = plt.subplots()
ax.plot(t, running_avg_var)
ax.set_xticks([rest_time1, rest_time1+stim_time])
ax.set_xticklabels(['stimulus starts', 'stimulus ends'])
ax.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='yellow')
ax.set_ylabel('$\sqrt{Var(V)}$')
plt.show()
