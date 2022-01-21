import numpy as np
from QTomography import *

N = 1
phi = 1

rho = get_rho(N)
eta, rho_final = find_eta(rho, phi, N)

sigma = get_sigma(N)
calc_rho = reconstruct(eta, sigma, rho, N)

# Output results
print("eta_ij")
print(np.round(eta, 5))
print("\nLatex code for eta_ij")
print_latex(np.round(eta, 5))
print("\nSum of terms in (rho final - calculated rho)")
print(np.sum(rho_final - calc_rho))