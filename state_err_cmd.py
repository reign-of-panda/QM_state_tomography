# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:57:41 2022

@author: therm
"""

import argparse
import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


class state_prep:
    def __init__(self):
        # All of the arguments required passed through the command line.
        self.parser = argparse.ArgumentParser(description="State preparation error")
        self.parser.add_argument('--N', type=int, metavar='',
                                 help='Number of qubits', default=1)
        self.parser.add_argument('--n_runs', type=int, metavar='',
                                 help='Number of norm vs. infidelity points', default=1000)
        self.parser.add_argument('--theta', type=float, metavar='',
                                 help='Distance between states on the bloch sphere', default=2*np.pi/3)
        self.parser.add_argument('--eps_max', type=float, metavar='',
                                 help='eps_max', default=0.1)
        self.parser.add_argument('--file_name', type=str, metavar='',
                                 help='File name for the csv')

        self.args = self.parser.parse_args()  # Stores all of the arguments
        
        # These are the command line inputs
        self.N = self.args.N
        self.n_runs = self.args.n_runs
        self.theta = self.args.theta
        self.eps_max = self.args.eps_max
        self.file_name = self.args.file_name
        
        # Pertains to dynamics
        self.phi = 1
        self.eps = 0.1
        
        # Define Pauli matrices
        pauli_0 = np.identity(2)
        pauli_1 = np.array([[0, 1], [1, 0]])
        pauli_2 = np.array([[0, -1j], [1j, 0]])
        pauli_3 = np.array([[1, 0], [0, -1]])
        self.pauli = np.array([pauli_0, pauli_1, pauli_2, pauli_3])

        # Initial states
        s0 = self.single_qubit(0, 0)
        s1 = self.single_qubit(self.theta, 0)#2 * np.pi / 3
        s2 = self.single_qubit(self.theta, 2 * np.pi / 3)
        s3 = self.single_qubit(self.theta, -2 * np.pi / 3)
        self.s = np.array([s0, s1, s2, s3])

        # Projectors
        P0 = np.array([[1], [0]])
        P1 = np.array([[0], [1]])
        P2 = np.array([[1], [1]]) / np.sqrt(2)
        P3 = np.array([[1], [1j]]) / np.sqrt(2)
        self.P_single = np.array([P0, P1, P2, P3])

    
    def print_latex(self, arr):
    	latex = "\\begin{pmatrix}"
    	dim = arr.shape
    
    	if len(dim) == 2:
    		for i in range(dim[0]):
    			for j in range(dim[1]):
    				str_ij = str(arr[i, j])
    				if str_ij[-1] == ".":
    					str_ij = str_ij[:-1]
    				latex += str_ij + "&"
    				if j == dim[1] - 1:
    					latex = latex[: -1] + "\\\\ "
    		latex = latex[:-3]
    		latex += "\\end{pmatrix}"
    
    	elif len(dim) == 1:
    		for i in range(dim[0]):
    			str_i = str(arr[i])
    			if str_i[-1] == ".":
    				str_i = str_i[:-1]
    			latex += str_i + "\\\\ "
    		latex = latex[:-3]
    		latex += "\\end{pmatrix}"
    
    	print(latex)

    def get_indx(self, N):
    	to_return = []
    	pauli_indx = [0 for i in range(N)]
    	for n in range(4**N):
    		to_return.append(deepcopy(pauli_indx))
    		to_add = -1
    		while True:
    			if to_add == -N - 1:
    				break
    			elif pauli_indx[to_add] == 3:
    				pauli_indx[to_add] = 0
    				to_add -= 1
    			else:
    				pauli_indx[to_add] += 1
    				break
    	return to_return
    
    def get_sigma(self, N):
    	sigma = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	pauli_indx = self.get_indx(N)
    	for n in range(4**N):
    		to_append = np.array([1])
    		for i in range(N):
    			to_append = np.kron(to_append, self.pauli[pauli_indx[n][i]])
    		sigma[n] = to_append
    	return sigma

    # Dynamics
    def dynamics_unitary(self, phi, sigma, N):
    	return np.cos(phi) * np.identity(2**N) + 1j * np.sin(phi) * sigma[2]
    
    def single_qubit(self, theta, phi):
    	return np.array([[np.cos(theta/2)], [np.sin(theta/2) * np.exp(1j * phi)]])
    
    def get_rho(self, N):
    	rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	pauli_indx = self.get_indx(N)
    	for n in range(4**N):
    		to_append = np.array([1])
    		for i in range(N):
    			to_append = np.kron(to_append, self.s[pauli_indx[n][i]])
    		rho[n] = to_append @ to_append.conj().T
    	return rho
    
    def get_projectors(self, N):
    	P = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	pauli_indx = self.get_indx(N)
    	for n in range(4**N):
    		to_append = np.array([1])
    		for i in range(N):
    			to_append = np.kron(to_append, self.P_single[pauli_indx[n][i]])
    		P[n] = to_append @ to_append.conj().T
    	return P
    
    # Finding A
    def get_A(self, P, sigma, rho, N):
    	A = np.zeros((4**N, 4**N, 4**N, 4**N), dtype = np.complex128)
    	A_unpacked = np.zeros((16**N, 16**N), dtype = np.complex128)
    	for p in range(4**N):
    		for q in range(4**N):
    			for i in range(4**N):
    				for j in range(4**N):
    					temp = np.trace(P[q] @ sigma[i] @ rho[p] @ sigma[j].conj().T)
    					A[p, q, i, j] = temp
    					A_unpacked[(4**N) * p + q, (4**N) * i + j] = temp
    	return A, A_unpacked
    
    # Finding rho_final
    def rhof_theoretical(self, U, rho, N):
    	rho_final = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	for i in range(4**N):
    		rho_final[i] = U @ rho[i] @ U.conj().T
    	return rho_final
    
    # Finding P_pq given the projectors and final states
    def get_probs(self, P, rho_final, N):
    	Prob = np.zeros((4**N, 4**N), dtype = np.complex128)
    	Prob_unpacked = np.zeros((16**N, ), dtype = np.complex128)
    	for p in range(4**N):
    		for q in range(4**N):
    			temp = np.trace(P[q] @ rho_final[p])
    			Prob[p, q] = temp
    			Prob_unpacked[(4**N) * p + q] = temp
    	return Prob, Prob_unpacked
    
    # Given P_pq and A_pqij, find eta_ij
    def get_eta(self, A_unpacked, Prob_unpacked, N):
    	eta_unpacked = np.linalg.solve(A_unpacked, Prob_unpacked)
    	eta = np.zeros((4**N, 4**N), dtype = np.complex128)
    	for n in range(16**N):
    		j = n % 4**N
    		i = n // 4**N
    		eta[i, j] = eta_unpacked[n]
    	return eta
    
    # Verification
    def reconstruct(self, eta, sigma, rho, N):
    	calc_rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	for p in range(4**N):
    		to_append = np.zeros((2**N, 2**N), dtype = np.complex128)
    		for i in range(4**N):
    			for j in range(4**N):
    				to_append += eta[i, j] * sigma[i] @ rho[p] @ sigma[j].conj().T
    		calc_rho[p] = to_append
    	return calc_rho
    
    # Single function to do everything
    def find_eta(self, rho, phi, N):
    	sigma = self.get_sigma(N)
    	P = self.get_projectors(N)
    
    	U = self.dynamics_unitary(phi, sigma, N)
    	rho_final = self.rhof_theoretical(U, rho, N)
    
    	_, Prob_unpacked = self.get_probs(P, rho_final, N)
    	_, A_unpacked = self.get_A(P, sigma, rho, N)
    
    	eta = self.get_eta(A_unpacked, Prob_unpacked, N)
    	return eta, rho_final
    
    def rho_experimental(self, eps, N):
    	rho = np.zeros((4**N, 2**N, 2**N), dtype = np.complex128)
    	pauli_indx = self.get_indx(N)
    	for n in range(4**N):
    		to_append = np.array([1])
    		for i in range(N):
    			rand_real = np.random.random((2, 1))
    			rand_imag = np.random.random((2, 1))
    
    			new_state = self.s[pauli_indx[n][i]] + eps * (rand_real + 1j * rand_imag)
    			new_state = new_state / np.linalg.norm(new_state)
    
    			to_append = np.kron(to_append, new_state)
    		rho[n] = to_append @ to_append.conj().T
    	return rho
    
    def fidelity(self, A, B, pure = False):
    	if not pure:
    		return np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A)))**2
    	else:
    		return np.trace(A @ B)
    
    def get_infidelities(self, A, B, N):
    	infid = np.zeros((4**N, ))
    	for i in range(4**N):
    		infid[i] = 1 - np.real(self.fidelity(A[i], B[i], pure = True))
    	return infid
    
    def single_trial(self, eps, N):
    	rho_t = self.get_rho(N)
    	rho_e = self.rho_experimental(eps, N)
    
    	infidelities = self.get_infidelities(rho_t, rho_e, N)
    
    	eta_t, _ = self.find_eta(rho_t, self.phi, N)
    
    	# Find eta_e
    	sigma = self.get_sigma(N)
    	P = self.get_projectors(N)
    
    	U = self.dynamics_unitary(self.phi, sigma, N)
    	rho_final = self.rhof_theoretical(U, rho_e, N)
    
    	_, Prob_unpacked = self.get_probs(P, rho_final, N)
    	_, A_unpacked = self.get_A(P, sigma, rho_t, N)
    
    	eta_e = self.get_eta(A_unpacked, Prob_unpacked, N)
    	print(np.linalg.eigh(eta_e))
    
    	diff = eta_t - eta_e
    	# deta = np.real(np.trace(sqrtm(diff.conj().T @ diff)))
    	deta = np.sqrt(np.real(np.trace(diff.conj().T @ diff)))
    
    	return np.mean(infidelities), deta
    
    def do_runs(self):
        self.single_trial(self.eps, 1)

        n_samples = self.n_runs
        eps_max = self.eps_max
        # eps_max = 0.3
        infids = np.zeros((n_samples, ))
        detas = np.zeros((n_samples, ))
        for i in range(n_samples):
        	if i%100 == 0:
        		print(i)
        	self.eps += i * eps_max / n_samples
        	a, b = self.single_trial(self.eps, self.N)
        	infids[i] = a
        	detas[i] = b
        
        data = np.column_stack((infids, detas))
        np.savetxt(self.file_name, data)
        return infids, detas
        
    def plot_stuff(self, infids, detas):
        plt.title("State preparation errors - {} samples".format(self.n_runs))
        plt.scatter(infids, detas**2, marker = ",")
        plt.grid()
        plt.xlabel("Infidelities")
        plt.ylabel(r"$||\eta^{T} - \eta^{E}||^{2}_{1}$")
        plt.show()

if __name__ == "__main__":
    obj1 = state_prep()
    infids, detas = obj1.do_runs()
    obj1.plot_stuff(infids, detas)
    
    
