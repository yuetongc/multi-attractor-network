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
p_matrix1 = np.zeros([N_params, N_point])
p_matrix1 = modelfit.init_p(p_matrix1, [2, 0, 0, 0], 0, n3)
p_matrix2 = np.zeros([N_params, N_point])
p_matrix2 = modelfit.init_p(p_matrix2, [2, 0, 0, 0], 0, n3)

est_matrix1 = np.zeros([N_neuron, N_point])
est_matrix2 = np.zeros([N_neuron, N_point])

for n in range(30):
    c1 = 0
    for i in range(n1):
        est_matrix1[:, c1] = modelfit.f_v_a(ang_vector, p_matrix1[1:3, c1], p_matrix1[0, c1])
        c1 += 1
    b_est = np.mean(V_matrix[:, 0:n1] - est_matrix1[:, 0:n1])
    print("{}th estimation of rest phase 1,  b estimate is {}".format(n+1, b_est))
    p_matrix1[-1, 0:n1] = b_est

    c1 = 0
    for i in range(n1):
        a_est = modelfit.opt_a(ang_vector, V_matrix[:, c1], p_matrix1[1, c1], p_matrix1[2, c1], b_est)
        p_matrix1[0, c1] = a_est
        est = minimize(modelfit.mse_fv_baseline_a, p_matrix1[1:3, c1], args=(ang_vector, V_matrix[:, c1], a_est, b_est))
        p_est = est.x
        p_matrix1[1:3, c1] = p_est
        c1 += 1
    print("{}th estimation of rest phase 1,  the last a, mean, log var estimates are  {}".format(str(n+1),
                                                                                                 p_matrix1[0:3, c1-1]))

for n in range(30):
    c2 = n1
    for i in range(n2-n1):
        est_matrix1[:, c2] = modelfit.f_v_a(ang_vector, p_matrix1[1:3, c2], p_matrix1[0, c2])
        c2 += 1
    b_est = np.mean(V_matrix[:, n1:n2] - est_matrix1[:, n1:n2])
    print("{}th estimation of stimulus phase,  b estimate is {}".format(n+1, b_est))
    p_matrix1[-1, n1:n2] = b_est

    c2 = n1
    for i in range(n2-n1):
        a_est = modelfit.opt_a(ang_vector, V_matrix[:, c2], p_matrix1[1, c2], p_matrix1[2, c2], b_est)
        p_matrix1[0, c2] = a_est
        est = minimize(modelfit.mse_fv_baseline_a, p_matrix1[1:3, c2], args=(ang_vector, V_matrix[:, c2], a_est, b_est))
        p_est = est.x
        p_matrix1[1:3, c2] = p_est
        c2 += 1
    print("{}th estimation of stimulus phase,  the last a, mean, log var estimates are  {}".format(str(n+1),
                                                                                                   p_matrix1[0:3, c2-1]))
for n in range(30):
    c3 = n2
    for i in range(n3-n2):
        est_matrix1[:, c3] = modelfit.f_v_a(ang_vector, p_matrix1[1:3, c3], p_matrix1[0, c3])
        c3 += 1
    b_est = np.mean(V_matrix[:, n2:n3] - est_matrix1[:, n2:n3])
    print("{}th estimation of rest phase 2,  b estimate is {}".format(n+1, b_est))
    p_matrix1[-1, n2:n3] = b_est

    c3 = n2
    for i in range(n3-n2):
        a_est = modelfit.opt_a(ang_vector, V_matrix[:, c3], p_matrix1[1, c3], p_matrix1[2, c3], b_est)
        p_matrix1[0, c3] = a_est
        est = minimize(modelfit.mse_fv_baseline_a, p_matrix1[1:3, c3], args=(ang_vector, V_matrix[:, c3], a_est, b_est))
        p_est = est.x
        p_matrix1[1:3, c3] = p_est
        c3 += 1
    print("{}th estimation of rest phase 2,  the last a, mean, log var estimates are  {}".format(str(n+1),
                                                                                                 p_matrix1[0:3, c3-1]))
est_matrix1 = modelfit.update_p(est_matrix1, p_matrix1, ang_vector, 0, n3)

for n in range(30):
    c = 0
    for i in range(n3):
        est_matrix2[:, c] = modelfit.f_v_a(ang_vector, p_matrix2[1:3, c], p_matrix2[0, c])
        b_est2 = np.mean(V_matrix[:, c] - est_matrix2[:, c])
        p_matrix2[-1, c] = b_est2
        a_est2 = modelfit.opt_a(ang_vector, V_matrix[:, c], p_matrix2[1, c], p_matrix2[2, c], b_est2)
        p_matrix2[0, c] = a_est2
        est = minimize(modelfit.mse_fv_baseline_a, p_matrix2[1:3, c], args=(ang_vector, V_matrix[:, c], a_est2, b_est2))
        p_est2 = est.x
        p_matrix2[1:3, c] = p_est2
        c += 1
    print('{}th estimation finished, the last a, mean, log var, b estimates are  {}'.format(str(n+1), p_matrix2[:, c-1]))

est_matrix2 = modelfit.update_p(est_matrix2, p_matrix2, ang_vector, 0, n3)

r2_1 = []
for i in range(n3):
    r2_1.append(modelfit.r_squared(V_matrix[:, i], est_matrix1[:, i]))

r2_2 = []
for i in range(n3):
    r2_2.append(modelfit.r_squared(V_matrix[:, i], est_matrix2[:, i]))


font = {'size': 20}
plt.rc('font', **font)

rc = {"axes.spines.left": True,
      "axes.spines.right": False,
      "axes.spines.bottom": False,
      "axes.spines.top": False}
plt.rcParams.update(rc)

fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 18))
ax1.plot(t, p_matrix1[0, :])
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax1.set_ylabel(r'$\hat{a}$ / mV')
ax1.get_xaxis().set_visible(False)
ax2.plot(t, 180 * (p_matrix1[1, :] / math.pi))
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_ylim(-math.pi, math.pi)
ax2.set_yticks([-180, 0, 180])
ax2.set_ylabel(r'$\hat{\mu}$ / $\degree$')
ax2.get_xaxis().set_visible(False)
ax3.plot(t, np.exp(p_matrix1[2, :]))
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.set_ylabel(r'$\hat{\sigma^{2}}$')
ax3.get_xaxis().set_visible(False)
ax4.plot(t, p_matrix1[3, :])
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_ylabel(r'$\hat{b}$ / mV')
ax4.get_xaxis().set_visible(False)
ax5.plot(t, r2_1)
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.set_ylabel(r'$R^{2}$')
ax5.set_xlabel('t / ms')
plt.tight_layout()
plt.show()
fig1.savefig('Figure 6.png')

fig2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 18))
ax1.plot(t, p_matrix2[0, :])
ax1.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax1.set_ylabel(r'$\hat{a}$ / mV')
ax1.get_xaxis().set_visible(False)
ax2.plot(t, 180 * (p_matrix2[1, :] / math.pi))
ax2.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax2.set_ylim(-math.pi, math.pi)
ax2.set_yticks([-180, 0, 180])
ax2.set_ylabel(r'$\mu$ / $\degree$')
ax2.get_xaxis().set_visible(False)
ax3.plot(t, np.exp(p_matrix2[2, :]))
ax3.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax3.set_ylabel(r'$\hat{\sigma^{2}}$')
ax3.get_xaxis().set_visible(False)
ax4.plot(t, p_matrix2[3, :])
ax4.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax4.set_ylabel(r'$\hat{b}$ / mV')
ax4.get_xaxis().set_visible(False)
ax5.plot(t, r2_2)
ax5.axvspan(t1, t2, alpha=0.5, color='lightgrey')
ax5.set_ylabel(r'$R^{2}$')
ax5.set_xlabel('t / ms')
plt.tight_layout()
plt.show()
fig2.savefig('Figure 7.png')
