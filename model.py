import math
import scipy
import sklearn
import numpy as np


class MultiAttractorModel:
    def __init__(self, N, max_rate=100, tau_membrane=10, tau_noise=50):
        self.N = N
        self.max_rate = max_rate
        self.tau_membrane = tau_membrane
        self.tau_noise = tau_noise
        self.W = self.build_weight_matrix()
        self.noise_cov = self.build_noise_cov()
        self.cov_term = self.noise_cov_term()

    def build_weight_matrix(self, W_avg=-40, W_mod=33):
        N_matrix = np.reshape(np.repeat(np.arange(self.N), self.N), (self.N, self.N))
        ind_matrix = N_matrix - np.transpose(N_matrix)
        ang_matrix = ind_matrix * 2 * math.pi / self.N
        W_matrix = (np.cos(ang_matrix) * W_mod / self.max_rate + W_avg / self.max_rate) / self.N
        return W_matrix

    def momentary_firing_rate(self, V_in, gain=0.1):
        return self.max_rate * np.tanh(gain * np.maximum(V_in, 0))

    def external_input(self, c, b=2, A=0.1, stim_ang=0):
        ang_vector = np.arange(-math.pi, math.pi, 2 * math.pi / self.N)
        return b + c * (1 - A + A * np.cos(ang_vector - stim_ang))

    def init_voltage(self):
        return np.full(self.N, 0)

    def build_noise_cov(self, sd_noise=0.15, l_noise=math.pi/3):
        N_matrix = np.reshape(np.repeat(np.arange(self.N), self.N), (self.N, self.N))
        ind_matrix = N_matrix - np.transpose(N_matrix)
        ang_matrix = ind_matrix * 2 * math.pi / self.N
        amp = sd_noise * math.sqrt(1 + self.tau_membrane / self.tau_noise)
        cov = amp * amp * np.exp((np.cos(ang_matrix) - 1) / (l_noise ** 2))
        return cov

    def noise_cov_term(self):
        return np.real(scipy.linalg.sqrtm(2 * self.tau_noise * self.noise_cov))

    def init_noise(self):
        return np.random.multivariate_normal(np.zeros(self.N), self.noise_cov)

    def ornstein_uhlenbeck_process(self, noise, t_int):
        ind_rv = np.random.multivariate_normal(np.zeros(self.N), np.identity(self.N))
        step = (-noise * t_int + np.matmul(self.cov_term, ind_rv * math.sqrt(t_int))) / self.tau_noise
        noise = noise + step
        return noise

    def sim(self, V, noise, t_int, c):
        h = self.external_input(c)
        r = self.momentary_firing_rate(V)
        step = (-V + h + noise + np.matmul(self.W, r)) / self.tau_membrane
        V = V + step * t_int
        return V
