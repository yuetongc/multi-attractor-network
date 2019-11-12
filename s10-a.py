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

val1 = []
val2 = []
val3 = []

time1 = time.time()
for i in range(int(rest_time1/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of rest phase 1, took {} seconds".format(time.time() - time1))

time2 = time.time()
for i in range(int(stim_time/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 2)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of stimulus phase, took {} seconds".format(time.time() - time2))

time3 = time.time()
for i in range(int(rest_time2/t_int)):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    val1.append(V[50])
    val2.append(V[25])
    val3.append(V[0])
print("end of rest phase 2, took {} seconds".format(time.time() - time3))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(t, val1)
ax1.set_title('Neuron tuned to the stimulus direction')
ax1.set_xticks([rest_time1, rest_time1+stim_time])
ax1.set_xticklabels(['stimulus starts', 'stimulus ends'])
ax1.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='yellow')
ax1.set_ylabel('V')

ax2.plot(t, val2)
ax2.set_title('Neuron tuned to the orthogonal direction')
ax2.set_xticks([rest_time1, rest_time1+stim_time])
ax2.set_xticklabels(['stimulus starts', 'stimulus ends'])
ax2.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='yellow')
ax2.set_ylabel('V')

ax3.plot(t, val3)
ax3.set_title('Neuron tuned to the opposite direction')
ax3.set_xticks([rest_time1, rest_time1+stim_time])
ax3.set_xticklabels(['stimulus starts', 'stimulus ends'])
ax3.axvspan(rest_time1, rest_time1 + stim_time, alpha=0.5, color='yellow')
ax3.set_ylabel('V')

plt.tight_layout()
plt.show()
