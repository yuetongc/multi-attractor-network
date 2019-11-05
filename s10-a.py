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

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()

val1 = []
val2 = []
val3 = []

time1 = time.time()
for i in range(int(rest_time1/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of rest phase 1, took {} seconds".format(time.time() - time1))

time2 = time.time()
for i in range(int(stim_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 2)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of stimulus phase, took {} seconds".format(time.time() - time2))

time3 = time.time()
for i in range(int(rest_time2/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise, t_int)
    V = attractor_model.sim(V_in, noise, t_int, 0)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of rest phase 2, took {} seconds".format(time.time() - time3))

plt.plot(t, val1)
plt.title('Neuron tuned to the stimulus direction')
plt.xlabel('t')
plt.ylabel('V')
plt.show()

plt.plot(t, val2)
plt.title('Neuron tuned to the orthogonal direction')
plt.xlabel('t')
plt.ylabel('V')
plt.show()

plt.plot(t, val3)
plt.title('Neuron tuned to the opposite direction')
plt.xlabel('t')
plt.ylabel('V')
plt.show()
