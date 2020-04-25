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

for n in range(120):
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


rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False,
      "lines.linewidth": 2,
      "xtick.labelsize": 24,
      "ytick.labelsize": 24,
      'legend.fontsize': 24,
      "axes.labelsize": 28,
      "axes.titlesize": 28,
      }
plt.rcParams.update(rc)


fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
bottom, top = 0.1, 0.9
left, right = 0.1, 0.85
fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
axes = [ax1, ax2]
im1 = ax1.imshow(V_matrix, interpolation='nearest', aspect='auto', extent=(0, t3, -180, 180))
ax1.set_title('True Activity')
ax1.set_ylabel(r'$\theta\ [\degree$]')
ax1.set_yticks([-180, 0, 180])
im2 = ax2.imshow(est_matrix, interpolation='nearest', aspect='auto', extent=(0, t3, -180, 180))
ax2.set_title('Bump Fit')
ax2.set_ylabel(r'$\theta\ [\degree$]')
ax2.set_yticks([-180, 0, 180])
ax2.set_xlabel('t [ms]')
plt.tight_layout()
fig1.colorbar(im2, ax=axes)
fig1.savefig('free_true_data')

fig2, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n1-int(500/t_int)], color='dimgrey', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n1-int(500/t_int)], linestyle='--', marker='o',
        markersize=2, color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['bottom'].set_visible(True)
ax.set_ylabel('V [mV]')
ax.set_xlabel(r'$\theta\ [\degree$]')
ax.set_xticks([-180, 0, 180])
plt.tight_layout()
fig2.savefig('free_fit_before')

fig3, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n2-int(500/t_int)], color='dimgrey', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n2-int(500/t_int)], linestyle='--',
        marker='o', markersize=2, color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['bottom'].set_visible(True)
ax.set_ylabel('V [mV]')
ax.set_xlabel(r'$\theta\ [\degree$]')
ax.set_xticks([-180, 0, 180])
fig3.savefig('free_fit_during')

fig4, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.linspace(-180, 180, N_neuron), V_matrix[:, n2+int(500/t_int)], color='dimgrey', label='actual')
ax.plot(np.linspace(-180, 180, N_neuron), est_matrix[:, n2+int(500/t_int)], linestyle='--',
        marker='o', markersize=2, color='steelblue', label='fitted')
ax.legend(frameon=False)
ax.spines['bottom'].set_visible(True)
ax.set_ylabel('V [mV]')
ax.set_xlabel(r'$\theta\ [\degree$]')
ax.set_xticks([-180, 0, 180])
fig4.savefig('free_fit_after')

fig5, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 14))
ax1.plot(t, p_matrix[0, :])
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax1.set_ylabel('$\hat{a}\ [mV]$')
ax2.plot(t, np.rad2deg(p_matrix[1, :]))
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_ylim(-math.pi, math.pi)
ax2.set_yticks([-180, 0, 180])
ax2.set_ylabel(r'$\hat{\mu}\ [\degree]$')
ax3.plot(t, np.exp(p_matrix[2, :]))
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.set_ylabel(r'$\hat{w}^{2}$')
ax4.plot(t, p_matrix[3, :])
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_ylabel('$\hat{b}\ [mV]$')
ax5.plot(t, r2)
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.set_ylabel(r'$R^{2}$')
ax5.set_xlabel('t [ms]')
plt.tight_layout()
fig5.savefig('free_r_2')
