import numpy as np

s1 = np.array([[1], [0]])
s2 = np.array([[0], [1]])
s = np.array([s1, s2])

def partial_trace(rho):
	tr = np.zeros((2, 2), dtype = np.complex128)
	for i in range(2):
		for j in range(2):
			for I in range(2):
				tr += s[i] @ s[j].conj().T * (np.kron(s[i], s[I]).conj().T @ rho @ np.kron(s[j], s[I]))
	return tr

if __name__ == "__main__":
	test = np.kron(s2 @ s2.conj().T, s1 @ s1.conj().T)
	print(partial_trace(test))
