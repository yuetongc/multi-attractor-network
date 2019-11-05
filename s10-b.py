import time
import numpy as np
import matplotlib.pyplot as plt
import model

N_neuron = 100

t_int = 0.1
rest_time1 = 1000
stim_time = 2000
rest_time2 = 1000
total_time = rest_time1 + stim_time + rest_time2
t = np.arange(0, total_time, t_int)
N_point = int(total_time/t_int)

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()

N_trial = 10
V_matrix = np.zeros([N_trial, N_point, N_neuron])

for i in range(N_trial):
    trail_start = time.time()
    c = 0
    for j in range(int(rest_time1/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
        V = attractor_model.sim(V_in, noise, t_int, 0)
        noise = noise_in
        V_matrix[i][c] = V
        c += 1
    for j in range(int(stim_time/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
        V = attractor_model.sim(V_in, noise, t_int, 2)
        noise = noise_in
        V_matrix[i][c] = V
        c += 1
    for j in range(int(rest_time2/t_int)):
        V_in = V
        noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
        V = attractor_model.sim(V_in, noise, t_int, 0)
        noise = noise_in
        V_matrix[i][c] = V
        c += 1
    print("{}th trial finished, took {} seconds".format(str(i+1), time.time()-trail_start))

var = np.sqrt(np.var(V_matrix, axis=0))
avg_var = np.mean(var, axis=1)

plt.plot(t, avg_var)
plt.xlabel('t')
plt.ylabel('$\sqrt{Var(V)}$')
plt.show()
