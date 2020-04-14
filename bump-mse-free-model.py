import model
import modelfit
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N_neuron = 100

attractor_model = model.MultiAttractorModel(N=N_neuron)
V = attractor_model.init_voltage()
noise = attractor_model.init_noise()
ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / N_neuron)

t_int = attractor_model.t_int
prep_time, rest_time1, stim_time, rest_time2 = 300, 500, 1000, 800
t0, t1, t2, t3 = prep_time, rest_time1, rest_time1+stim_time, rest_time1+stim_time+rest_time2
n0, n1, n2, n3 = int(t0/t_int), int(t1/t_int), int(t2/t_int), int(t3/t_int)
t = np.arange(0, t3, t_int)
N_point = n3

V_matrix = np.zeros([N_neuron, N_point])

timer0 = time.time()
for i in range(n0):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
print("prep phase finished")
c = 0
print("start of simulation")
timer1 = time.time()
for i in range(n1):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(n2-n1):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 2)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
for i in range(n3-n2):
    V_in = V
    noise_in = attractor_model.ornstein_uhlenbeck_process(noise)
    V = attractor_model.update(V_in, noise, 0)
    noise = noise_in
    V_matrix[:, c] = V
    c += 1
print("end of simulation")

N_params = 4
p_matrix = np.zeros([N_params, N_point])
p_matrix = modelfit.init_p(p_matrix, [2, 0, 0, 0], 0, n3)

est_matrix = np.zeros([N_neuron, N_point])

for n in range(100):
    c = 0
    for i in range(n3):
        est_matrix[:, c] = modelfit.f_v_a(ang_vector, p_matrix[1:3, c], p_matrix[0, c])
        b_est = np.mean(V_matrix[:, c] - est_matrix[:, c])
        p_matrix[-1, c] = b_est
        a_est = modelfit.opt_a(ang_vector, V_matrix[:, c], p_matrix[1, c], p_matrix[2, c], b_est)
        p_matrix[0, c] = a_est
        est = minimize(modelfit.mse_fv_baseline_a, p_matrix[1:3, c], args=(ang_vector, V_matrix[:, c], a_est, b_est))
        p_est = est.x
        p_matrix[1:3, c] = p_est
        c += 1
    print('{}th estimation finished, the last a, mean, log var, b estimates are  {}'.format(str(n+1), p_matrix[:, c-1]))

est_matrix = modelfit.update_p(est_matrix, p_matrix, ang_vector, 0, n3)

r2 = []
for i in range(n3):
    r2.append(modelfit.r_squared(V_matrix[:, i], est_matrix[:, i]))

r_tot = []
for i in range(n3):
    r_tot.append(modelfit.firing_rate_app(est_matrix[:, i]))


font = {'size': 14}
plt.rc('font', **font)

fig1, (ax1, ax2) = plt.subplots(2, 1)
bottom, top = 0.1, 0.9
left, right = 0.1, 0.85
fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
axes = [ax1, ax2]
im1 = ax1.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, t3, -180, 180))
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('True Activity', fontsize=14)
ax1.set_ylabel(r'PO / $\degree$', fontsize=14)
ax1.set_yticks([-180, 0, 180])
ax1.tick_params(labelsize=12)
im2 = ax2.imshow(est_matrix, interpolation='nearest', aspect='auto', extent=(0, t3, -180, 180))
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Bump Fit', fontsize=14)
ax2.set_ylabel(r'PO / $\degree$', fontsize=14)
ax2.set_yticks([-180, 0, 180])
ax2.set_xlabel('t / ms', fontsize=14)
ax2.tick_params(labelsize=12)
plt.tight_layout()
fig1.colorbar(im2, ax=axes)
plt.show()

fig2, ax = plt.subplots()
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n1-int(0/t_int)], linestyle='--', marker='o',
        markersize=2, color='black', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n1-int(0/t_int)], color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('V / mV', fontsize=14)
ax.set_xlabel(r'PO / $\degree$', fontsize=14)
ax.tick_params(labelsize=12)
plt.show()

fig3, ax = plt.subplots()
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n2-int(0/t_int)], linestyle='--',
        marker='o', markersize=2, color='black', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n2-int(0/t_int)], color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('V / mV', fontsize=14)
ax.set_xlabel(r'PO / $\degree$', fontsize=14)
ax.tick_params(labelsize=12)
plt.show()

fig4, ax = plt.subplots()
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n2+int(0/t_int)], linestyle='--',
        marker='o', markersize=2, color='black', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n2+int(0/t_int)], color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('V / mV', fontsize=14)
ax.set_xlabel(r'PO / $\degree$', fontsize=14)
ax.tick_params(labelsize=12)
plt.show()

fig6, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 20))
ax1.plot(t, p_matrix[0, :])
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('a / mV')
ax2.plot(t, 180 * (p_matrix[1, :] / math.pi))
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(-math.pi, math.pi)
ax2.set_yticks([-180, 0, 180])
ax2.set_ylabel(r'$\mu$ / $\degree$')
ax3.plot(t, np.exp(p_matrix[2, :]))
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylabel(r'$\sigma^{2}$')
ax4.plot(t, p_matrix[3, :])
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_ylabel('b / mV')
ax4.set_xlabel('t / ms')
ax5.plot(t, r2)
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_ylabel(r'$R^{2}$')
ax5.set_xlabel('t / ms')
plt.tight_layout()
plt.show()
fig6.savefig('Figure 6.png')

fig7, ax = plt.subplots()
ax.plot(t, r_tot)
ax.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel(r'$r_{total}$ / $s^{-1}$')
ax.set_xlabel('t / ms')
plt.show()

fig8, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 14))
ax1.scatter(p_matrix[0, 0:t1], np.exp(p_matrix[2, 0:t1]), color='blue', label='before', s=1)
ax1.scatter(p_matrix[0, t1:t2], np.exp(p_matrix[2, t1:t2]), color='red', label='during', s=1)
ax1.scatter(p_matrix[0, t2:t3], np.exp(p_matrix[2, t2:t3]), color='green', label='after', s=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('a')
ax1.set_ylabel(r'$\sigma^{2}$')
ax2.scatter(p_matrix[1, 0:t1], p_matrix[0, 0:t1], color='blue', label='before', s=1)
ax2.scatter(p_matrix[1, t1:t2], p_matrix[0, t1:t2], color='red', label='during', s=1)
ax2.scatter(p_matrix[1, t2:t3], p_matrix[0, t2:t3], color='green', label='after', s=1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel('a')
ax3.scatter(p_matrix[1, 0:t1], np.exp(p_matrix[2, 0:t1]), color='blue', label='before', s=1)
ax3.scatter(p_matrix[1, t1:t2], np.exp(p_matrix[2, t1:t2]), color='red', label='during', s=1)
ax3.scatter(p_matrix[1, t2:t3], np.exp(p_matrix[2, t2:t3]), color='green', label='after', s=1)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\sigma^{2}$')
plt.tight_layout()
handles, labels = ax1.get_legend_handles_labels()
fig8.legend(handles, labels, loc='upper right', fontsize=12)
plt.show()
