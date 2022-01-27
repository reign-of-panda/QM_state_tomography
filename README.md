# QM_state_tomography
3rd year project by Andres Perez and Swoyam Panda

## Files
- .csv files: These are data files. The naming format is NxSy, where x is the number of qubits used and y is the number of samples taken
- QTomography.py: File in which the functions used to find the $'\eta_{ij}'$ coefficients, imported in other files
- multiple_qubits.py: Calculates $'\eta_{ij}'$ for an arbitrary number of qubits
- multiple_qubits_old.py: Old version of multiple_qubits.py, does not use the functions in QTomography.py
- state_prep_err.py: Calculates the difference between an experimental and a theoretical eta against infidelity. Also plots and saves the data
- trace_test.py: Calculates $'Tr(\sigma_{i} \rho_{p} \sigma_{j}^{\dagger})'$
- verify_eq_21.py: Computational verification of the formula calculated in the PDF in teams
- verify_parametrisation.py: Checks the experimental $'\rho_{p}^{EF}'$ can be written in terms of $'\rho_{p}^{TI}'$
