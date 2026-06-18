"""
HEOM + TCL-DERIVED SURFACE HOPPING
Input Parameter Specification Module

This file defines all physical, numerical, and computational parameters
used throughout the HEOM simulation framework.

It acts as the central configuration layer controlling:

    - electronic structure of the system
    - vibrational (classical and quantum) degrees of freedom
    - molecule-lead coupling
    - bath spectral properties
    - HEOM hierarchy truncation and convergence settings
    - time propagation and numerical tolerances
    - spatial nuclear coordinate grid definition
    - output and diagnostic options

-----------------------------------------------------------------------
GENERAL STRUCTURE
-----------------------------------------------------------------------

The parameters are organized into the following sections:

1. System size and basis definition
2. HEOM hierarchy and bath decomposition settings
3. Numerical propagation and solver controls
4. Bath (lead) physical parameters
5. Vibrational mode definitions (classical and quantum)
6. Electronic Hamiltonian specification
7. Nuclear coordinate grid definition
8. Molecule-lead coupling model
9. Output and runtime control flags

-----------------------------------------------------------------------
UNITS
-----------------------------------------------------------------------

Unless otherwise stated, all quantities are expressed in:

    - electron volts (eV) for energies
    - dimensionless coordinates for vibrational modes
    - atomic units implied where appropriate for dynamical quantities

-----------------------------------------------------------------------
EXECUTION MODEL
-----------------------------------------------------------------------

This file is NOT executed directly.

It is imported by the main driver:

    SHEOM_main.py
    SHEOM_main_parallelized.py

All parameters defined here are read at import time and remain fixed
throughout a simulation run.

-----------------------------------------------------------------------
MODEL FLEXIBILITY
-----------------------------------------------------------------------

The framework supports:

    - single or multi-level electronic systems
    - multiple vibrational modes (classical and quantum)
    - arbitrary lead configurations (number of electrodes = Nleads)
    - tunable molecule-lead coupling matrices
    - flexible spectral decomposition schemes (Pade / barycentric)
    - optional wide-band limit approximation

-----------------------------------------------------------------------
OUTPUT CONTROL

Several flags in this file control simulation behavior:

    n_trajectories: number of surface hopping trajectories

-----------------------------------------------------------------------
COMPUTATIONAL NOTES

- This file strongly influences memory scaling through:
    dim_rho, Nmax, Nmodes, and x_grid resolution.

- HEOM hierarchy size and bath decomposition parameters directly
  determine computational complexity.

- The Fortran backend depends on several parameters defined here
  (e.g. Nleads, Nel, Nmodes, coupling strengths).

-----------------------------------------------------------------------
IMPORTANT

Changing parameters in this file will change all simulation results.
Consistency between physical parameters and numerical convergence
settings is required for reliable results.
"""

import numpy as np
from source.constants import * # pylint disable=unused-wildcard-import
import source.const_ARK as const_ARK
import source.CreAnn as CreAnn
import scipy
import math

# Define molecular system constraints

Nel = 1
N_qu_vib_modes = 0
N_cl_vib_modes = 1
max_occ_qu_vib_modes = [0]                                                                # Maximum number of phonons allowed in each mode
dim_vib_mode_qu = np.prod(np.array(max_occ_qu_vib_modes,dtype=int)+1)                             # Number of bosonic Fock states
dim_el = 2**Nel                                                                     # Number of fermionic Fock states
dim_rho = dim_el*dim_vib_mode_qu
N_el_vib_int_qu = 0
N_el_vib_int_cl = 1
small_polaron_yn = 0

# Define hierarchical constraints

Nmax = 2
Npoles_pade = 10
Npoles_barycentric = 15
Nsupport_points_barycentric = 2000
tol_F = 1e-5
tol_Gamma_barycentric = 1e-5
tol_fermi_symmetrized_barycentric = 1e-3
Nleads = 1
Nsign = 2                                                                         # Types of second-quantization operators. Always = 2: annihilation (d) and creation (ddag).
filtering_YN = 0
if filtering_YN == 1:
    tol = 0.0

# Define constraints of bath spectral functions

wbl_YN = 0
analytic_spectral_function_decomposition = True
pole_choice = "barycentric"
if wbl_YN == 0:                                                                    # If not under the wide-band limit, they are assumed to have a Lorentzian density of states
    Nmodes_pade = (Npoles_pade+1)*Nel*Nleads*Nsign                                            # Calculate number of modes outside of wide-band limit
    Nmodes_barycentric = (Npoles_barycentric+1)*Nel*Nleads*Nsign                                            # Calculate number of modes outside of wide-band limit
    if Nleads == 2:
        specwidth = np.array([10.0,10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
        symmetrized_fermi_specwidth = np.array([10.0,10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
    if Nleads == 1:
        specwidth = np.array([10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
        symmetrized_fermi_specwidth = np.array([10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
    Ncutoff = 1
elif wbl_YN == 1:
    Nmodes_pade = Npoles_pade*Nel*Nleads*Nsign                                            # Calculate number of modes outside of wide-band limit
    Nmodes_barycentric = Npoles_barycentric*Nel*Nleads*Nsign                                            # Calculate number of modes outside of wide-band limit
    if Nleads == 2:
        specwidth = np.array([10.0,10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
        symmetrized_fermi_specwidth = np.array([10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
    if Nleads == 1:
        specwidth = np.array([10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
        symmetrized_fermi_specwidth = np.array([10.0],dtype=float)                                 # Set spectral width of bath Lorentzians outside of wide-band limit
    Ncutoff = 1
    Nwblmodes = Nel*Nleads*Nsign

# Define propagation and derivative constraints

dt_init = 1e-3
dt_min = 1e-2
atol = 1e-6
rtol = 1e-6
fac = 0.38**(1/5)
facmin = 0.2
facmax_init = 10.0
rk_coeff,rk_coeffhat = const_ARK.dvecs('RK4')
rho_0 = np.zeros((dim_rho,dim_rho),dtype=complex) ; rho_0[0,0] = 1
max_expan_order = np.shape(rk_coeff)[1]                                         # Maximum expansion order for Euler's solution of DE
nthreads_liouvillian = 1
max_time = 2*dt_init
n_timesteps = int(max_time/dt_init)
time_vec = np.linspace(0,max_time,n_timesteps)
n_trajectories = 1
trajectory_list = list(np.arange(n_trajectories))
hop_counter_limit = 10

# Define bath parameters

voltage = 0
dv = 0.05
if Nleads == 2:
    muvec = np.array([voltage/2,-voltage/2],dtype=float)                      # Chemical potentials of left (first) and right (second) electrodes
elif Nleads == 1:
    muvec = np.array([voltage/2],dtype=float)                      # Chemical potentials of left (first) and right (second) electrodes

# Kelvin_T = 300
Temp = 0.03#Kelvin_T*k_B                                                     # Electrode temperature, assumed to be same for both electrodes

# Define molecular system classical vibrational parameters

freq_vector_cl_vib_modes = [0.003]
el_vib_int_cl = [0.005]
Vib_Freq_cl = np.zeros(N_cl_vib_modes,dtype=float)
El_Nuclear_Couplings_cl = np.zeros(N_cl_vib_modes,dtype=float)
small_polaron_shift_cl = np.zeros(N_cl_vib_modes,dtype=float)
for itr_cl_vib_modes in range(N_cl_vib_modes):
    Vib_Freq_cl[itr_cl_vib_modes] = freq_vector_cl_vib_modes[itr_cl_vib_modes]
    El_Nuclear_Couplings_cl[itr_cl_vib_modes] = np.sqrt(2)*el_vib_int_cl[itr_cl_vib_modes]
    small_polaron_shift_cl[itr_cl_vib_modes] = (el_vib_int_cl[itr_cl_vib_modes]**2)/Vib_Freq_cl[itr_cl_vib_modes]

ic_type = "wigner_ic" # Possible options are "const_ic", "wigner_ic", or "boltzmann_ic"
dimensionless_coordinates = True
x_vec_initial = np.array([0],dtype=float)
p_vec_initial = np.array([0],dtype=float)

# Define molecular system electronic parameters

energies_vector = [0.0]
hopping_vector = []
elel_interaction_vector = []
degenerate_levels = True
Single_El_Int = np.zeros((Nel,Nel),dtype=float)        # Energies of levels included in molecular system, as well as hopping between levels
Double_El_Int = np.zeros((Nel,Nel),dtype=float)        # Coulomb interactions between fermions - expressed as lower triangular matrix filled with two-particle interactions
hopping_count = 0
if Nel == 1:
    Single_El_Int[0,0] = energies_vector[0]# + small_polaron_shift_cl[0]
elif Nel > 1:
    for itr_el1 in range(Nel):
        Single_El_Int[itr_el1,itr_el1] = energies_vector[itr_el1] + small_polaron_shift_cl[0]
        for itr_el2 in range(itr_el1+1,Nel):
            # Single_El_Int[itr_el1,itr_el2] = hopping_vector[hopping_count] + small_polaron_shift_cl[0]
            # Single_El_Int[itr_el2,itr_el1] = hopping_vector[hopping_count] + small_polaron_shift_cl[0]
            # Double_El_Int[itr_el1,itr_el2] = elel_interaction_vector[hopping_count]
            # Double_El_Int[itr_el2,itr_el1] = elel_interaction_vector[hopping_count]
            hopping_count+=1

# DEFINE GAMMA PARAMETERS ###

Gamma_choice = 3*1e-3
V_Km = np.sqrt(Gamma_choice/(2*np.pi))

def el_lead_couplings_func(Nleads,Nel,V_Km,x_vec):

    el_lead_couplings = np.zeros((Nleads,Nel),dtype=float)
    for itrleads in range(Nleads):
        for itrel in range(Nel):
            el_lead_couplings[itrleads,itrel] = V_Km

    return el_lead_couplings
