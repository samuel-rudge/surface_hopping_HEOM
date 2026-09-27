# ---------------------------------------------------------------------
#
#        SHEOM MAIN DRIVER: HEOM + TCL-DERIVED SURFACE HOPPING
#
# ---------------------------------------------------------------------
#
# This script is the main execution driver for the SHEOM framework.
#
# It performs coupled dynamics between:
#   (i) classical nuclear (vibrational) motion in phase space (x, p)
#   (ii) quantum electronic dynamics of a molecule coupled to metallic leads
#       described exactly via Hierarchical Equations of Motion (HEOM)
#
# The surface hopping mechanism is not heuristic. Transition rates between
# electronic states are derived from a time-local master equation (TCLME),
# which is itself obtained from exact HEOM dynamics evaluated at fixed
# nuclear coordinate x(t).
#
# Electronic surfaces correspond to eigenstates of the isolated molecular
# Hamiltonian H_mol(x), which depends parametrically on the nuclear coordinate.
#
# Nuclear motion evolves classically on a single active surface at a time,
# while electronic transitions are stochastic events driven by HEOM-derived
# rates.
#
# ---------------------------------------------------------------------
#
# REQUIREMENTS
#
# Before running this script, the Fortran-based numerical kernels must be
# compiled using:
#
#     ./compile_f2py.sh
#
# This builds:
#   - sparse HEOM Liouvillian construction routines
#   - x-dependent Liouvillian update kernels
#   - sparse propagation engine (MKL + OpenMP accelerated)
#
# ---------------------------------------------------------------------
#
# CONFIGURATION
#
# All physical and numerical parameters are defined in:
#
#     input_parameters.py
#     system.py
#
# These include:
#   - electronic structure and interactions
#   - vibrational modes and couplings
#   - molecule–lead coupling strengths
#   - HEOM truncation and spectral decomposition parameters
#   - temperature, bias, and bath parameters
#   - number of trajectories and propagation settings
#
# There are no runtime input arguments.
#
# ---------------------------------------------------------------------
#
# USAGE
#
#     python3 SHEOM_main_parallelized.py
#
# ---------------------------------------------------------------------
#
# OUTPUTS
#
# The script writes trajectory and ensemble data to disk:
#
#   mol_pops.dat
#       Electronic state populations (ensemble averaged)
#
#   active_surfaces_tracked.dat
#       Occupation of electronic surfaces vs time
#
#   x_vec.dat
#       Nuclear positions for each trajectory
#
#   p_vec.dat
#       Nuclear momenta for each trajectory
#
# ---------------------------------------------------------------------

import source.generating_quantum_heom_class as generating_quantum_heom_class
import source.sparse_propagation_python as sparse_propagation_python
import source.calculate_quantum_observables as calculate_quantum_observables
import source.vibrational_system_setup as vibrational_system_setup
import source.system as system
from config.input_parameters import *
from scipy import linalg
import os
import joblib # type: ignore 
from pathlib import Path
from matplotlib import pyplot as plt 
import argparse

### OBTAIN ROOT ###

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str, required=True)
parser.add_argument("--slurm_id", type=int, required=True)
args = parser.parse_args()

### GENERATE QUANTUM HEOM INGREDIENTS ###

seed = 42

quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
sparse_heom_ingredients = quantum_heom_ingredients_object.return_sparse_heom_ingredients()
rho_nonzeros_sparse = sparse_heom_ingredients[10]
projected_0 = ((rho_nonzeros_sparse[:,1] == 0) & (rho_nonzeros_sparse[:,2] == 0))
projected_1 = ((rho_nonzeros_sparse[:,1] == 1) & (rho_nonzeros_sparse[:,2] == 1))
molecular_system_ingredients = quantum_heom_ingredients_object.return_molecular_system_ingredients()
quantum_observables_object = calculate_quantum_observables.quantum_observables_class(sparse_heom_ingredients,
                                                                                molecular_system_ingredients,
                                                                                projected_0,projected_1)

### COLLECT/DEFINE NECESSARY INGREDIENTS FOR QUANTUM HEOM PROPAGATION ###

molham_func = system.system_operators(Single_El_Int,Double_El_Int,Nel,N_qu_vib_modes,El_Nuclear_Couplings_cl,
                     max_occ_qu_vib_modes,dim_rho)[4]
molecular_system_ingredients[4] # Function taking vibrational coordinates as input and returning 
                                              # H_mol as output
active_surface_force_funcs = molecular_system_ingredients[11]
d_ops = molecular_system_ingredients[0] # Generate fermionic ann./cre. operators in molecular Hilbert space
pair_info_row_fil = sparse_heom_ingredients[0]        # Rows of quantum HEOM superoperator containing nonzero elements
pair_info_col_fil = sparse_heom_ingredients[1]        # Columns of quantum HEOM superoperator containing nonzero elements
npairs_fil = sparse_heom_ingredients[3]               # Number of filled elements in quantum HEOM superoperator
nnz_elements_sparse_fil = sparse_heom_ingredients[5]  # Number of ADO elements coupled to dynamics
pair_values_this_x = np.zeros(npairs_fil,dtype=np.float64)
# del(sparse_heom_ingredients)
# gc.collect()

rho_deriv = np.zeros(nnz_elements_sparse_fil,dtype=float) # Two arrays necessary for Runge-Kutta algorithm
rho_temp = np.zeros(nnz_elements_sparse_fil,dtype=float)
rho_ic = np.array([[0,0],[0,1]],dtype=float)
rho_input = np.zeros(nnz_elements_sparse_fil,dtype=float) ; rho_input[0] = rho_ic[0,0] ; rho_input[1] = rho_ic[1,1]
# Define initial condition of molecular system

rho_output = np.zeros(nnz_elements_sparse_fil,dtype=np.float64)
mol_pops = np.zeros((n_samples,2),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                 # with \rho_00(t) and \rho_11(t)
current = np.zeros((n_samples,Nleads),dtype=float)

### VIBRATIONAL QUANTITIES ### 

x_array_initial,p_array_initial = vibrational_system_setup.vibrational_initial_conditions(seed=seed) # Define nuclear coordinates/momenta with initial condition
initial_populations = np.diag(rho_ic)
possible_active_surfaces_list = np.arange(dim_rho)
active_surfaces = np.random.choice(possible_active_surfaces_list, size=n_trajectories,p=initial_populations)
# quantum_observables_ic = quantum_observables_object.return_quantum_observables_this_x(
#                                 rho_input,el_lead_couplings_func(Nleads,Nel,V_Km),x_vec)
# mol_pops[0,:] = quantum_observables_ic[2]
# current[0,:] = quantum_observables_ic[0]


def energy_func(El_Nuclear_Couplings_cl,Single_El_Int,x):
    energy_x = Single_El_Int[0,0] + El_Nuclear_Couplings_cl[0]*x
    return energy_x

def fermi_dirac_func(energy,temp):
    fd_occ = 1/(1 + np.exp(energy/temp))
    return fd_occ

def liouvillian(transition_01,transition_10):
    liouvillian_fgr = np.array([[-transition_10,transition_01],[transition_10,-transition_01]],dtype=float)
    return liouvillian_fgr

# def cme_one_step(liouvillian_fgr)
    

sparse_propagation_object = sparse_propagation_python.sparse_propagator(
    pair_info_row_fil, pair_info_col_fil, np.zeros(npairs_fil,dtype=np.float64), nnz_elements_sparse_fil
    )

seed_sequence = np.random.SeedSequence(seed)
trajectory_seeds = seed_sequence.spawn(n_trajectories)

def one_time_loop(
        itr_traj,
        active_surface,
        x_initial, 
        p_initial,
    n_timesteps,Nleads,el_lead_couplings_func,dim_rho,El_Nuclear_Couplings_cl,Nel,V_Km,
    dt_init,max_expan_order,rk_coeff,Vib_Freq_cl,
    quantum_observables_object,rho_deriv,rho_temp,rho_input,active_surface_force_funcs,
    sparse_propagation_object,rho_output,pair_values_this_x,traj_seed
    ):
    rng = np.random.default_rng(traj_seed)
    quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
    quantum_heom_ingredients_object.return_sparse_heom_ingredients()
    ### COLLECT/DEFINE NECESSARY INGREDIENTS FOR QUANTUM HEOM PROPAGATION ###
    print("TRAJECTORY "+str(itr_traj))
    x_vec = np.zeros(n_samples)
    p_vec = np.zeros(n_samples)
    transition_rate_tclme_arr = np.zeros((n_samples,2),dtype=float)
    transition_rate_tclme_cg_arr = np.zeros((n_samples,2),dtype=float)
    transition_rate_cg_arr = np.zeros((n_samples,2),dtype=float)
    x_vec[0] = x_initial
    p_vec[0] = p_initial
    x_final = x_initial
    p_final = p_initial
    mol_pops = np.zeros((n_samples,2),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                    # with \rho_00(t) and \rho_11(t)
    current = np.zeros((n_samples,Nleads),dtype=float)
    quantum_observables_ic = quantum_observables_object.return_quantum_observables_this_x(
                                rho_input,el_lead_couplings_func(Nleads,Nel,V_Km,x_final))
    mol_pops[0,:] = quantum_observables_ic[2]
    current[0,:] = quantum_observables_ic[0]
    nhops = 0
    rho_input_0 = np.concatenate((np.array([1,0],dtype=float),0*rho_input[2:]),axis=0)
    rho_input_1 = np.concatenate((np.array([0,1],dtype=float),0*rho_input[2:]),axis=0)
    rho_output_0 = rho_output.copy()
    rho_output_1 = rho_output.copy()
    rho_deriv_0 = np.zeros(nnz_elements_sparse_fil)
    rho_deriv_1 = np.zeros(nnz_elements_sparse_fil)
    rho_deriv_cg_0 = np.zeros(nnz_elements_sparse_fil)
    rho_deriv_cg_1 = np.zeros(nnz_elements_sparse_fil)
    cg_count = cgt + 1
    next_sample_time = sample_time
    itr_sample = 0
    current_time = 0
    for itrt in range(1,n_timesteps):
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
        ### GENERATE MOLECULAR SYSTEM HAMILTONIAN AND MOLECULE-METAL COUPLING AT THIS VIBRATIONAL COORDINATE
        ham_this_x = molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_final) 
                                            # Return mol. Hamiltonian at this vibrational coordinate
        el_lead_couplings_this_x = el_lead_couplings_func(Nleads,Nel,V_Km,x_final) 
                                            # Return molecule-metal coupling at this point
        ### GENERATE HEOM AT THIS VIBRATIONAL COORDINATE ###
        pair_values_this_x = quantum_heom_ingredients_object.return_sparse_heom_one_x(
                                            ham_this_x,el_lead_couplings_this_x,pair_values_this_x)
                                            # This "basically" returns the values of the nonzero elements
                                            # of the HEOM Liouvillian at this vibrational coordinate. 
        ### DO QUANTUM PART OF PROPAGATION ###
        sparse_propagation_object.update_values(pair_values_this_x)
        rho_output = sparse_propagation_object.propagate(dt_init,rho_input, max_expan_order, 
                                                           rk_coeff,rho_temp,rho_output,rho_deriv)
        U_matrix = np.transpose(np.vstack((rho_output_0[0:2],rho_output_1[0:2])))
        rho_output_0 = sparse_propagation_object.propagate(dt_init,rho_input_0, max_expan_order, 
                                                           rk_coeff,rho_temp,rho_output_0,rho_deriv)
        rho_output_1 = sparse_propagation_object.propagate(dt_init,rho_input_1, max_expan_order, 
                                                           rk_coeff,rho_temp,rho_output_1,rho_deriv)
        rho_deriv_0 = sparse_propagation_object.rho_derivative(rho_input_0,rho_deriv_0)
        rho_deriv_1 = sparse_propagation_object.rho_derivative(rho_input_1,rho_deriv_1)
        U_dot_matrix = np.transpose(np.vstack((rho_deriv_0[0:2],rho_deriv_1[0:2])))
        rate_matrix = U_dot_matrix @ linalg.pinv(U_matrix)
        ### OBTAIN QUANTUM OBSERVABLES FOR THIS VIBRATIONAL COORDINATE ###
        quantum_observables = quantum_observables_object.return_quantum_observables_this_x(rho_output,el_lead_couplings_this_x)
        # transition_rate = quantum_observables[3 + active_surface]
        ### CALCULATE COARSE-GRAINED TRANSITION RATE ###
        if cg_count < cgt:
            cg_count += 1
        else:
            initial_state = quantum_observables_object.split_quantum_state_this_x(rho_output).copy()
            initial_state_10 = initial_state[0].copy()
            initial_state_01 = initial_state[1].copy()
            linear_fit_tr = []
            for itrtau in range(cgt):
                initial_state_10 = sparse_propagation_object.propagate(dt_init,initial_state_10, max_expan_order,
                              rk_coeff,rho_temp,initial_state_10,rho_deriv)
                rho_deriv_cg_1 = sparse_propagation_object.rho_derivative(initial_state_10,rho_deriv_cg_1)
                transition_rate_01_cg = rho_deriv_cg_1[1]/(1 - initial_state_10[1])#/mol_pops[itrt,0]
                initial_state_01 = sparse_propagation_object.propagate(dt_init,initial_state_01, max_expan_order,
                                              rk_coeff,rho_temp,initial_state_01,rho_deriv)
                rho_deriv_cg_0 = sparse_propagation_object.rho_derivative(initial_state_01,rho_deriv_cg_0)
                transition_rate_10_cg = rho_deriv_cg_0[0]/(1 - initial_state_01[0])#/mol_pops[itrt,1]
                U_matrix_cg = np.transpose(np.vstack((initial_state_10[0:2],initial_state_01[0:2])))
                U_dot_matrix_cg = np.transpose(np.vstack((rho_deriv_cg_1[0:2],rho_deriv_cg_0[0:2])))
                rate_matrix_cg = U_dot_matrix_cg @ linalg.pinv(U_matrix_cg)
                transition_rate_01_tclme_cg = rate_matrix_cg[0,1]
                transition_rate_10_tclme_cg = rate_matrix_cg[1,0]
                linear_fit_tr.append(transition_rate_10_cg)
            cg_count = 0
            # plt.plot(dt_init*np.arange(cgt),linear_fit_tr) ; plt.show()
        ### RESET MOLECULAR STATE FOR NEXT ITERATION ###
        rho_input = rho_output
        rho_input_0 = rho_output_0
        rho_input_1 = rho_output_1
        if active_surface == 0:
            transition_rate_tclme_cg = transition_rate_10_cg
        elif active_surface == 1:
            transition_rate_tclme_cg = transition_rate_01_cg
        ### DETERMINE HOPPING ###
        transition_prob = dt_init*transition_rate_tclme_cg
        hop_this_x = rng.uniform() < transition_prob
        if hop_this_x:
            active_surface = 1 - active_surface
            nhops += 1
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        current_time += dt_init
        if (current_time >= next_sample_time) and (itr_sample < n_samples - 1):
            itr_sample += 1
            transition_rate_cg_arr[itr_sample,0] = transition_rate_01_cg
            transition_rate_cg_arr[itr_sample,1] = transition_rate_10_cg
            transition_rate_tclme_cg_arr[itr_sample,0] = transition_rate_01_tclme_cg
            transition_rate_tclme_cg_arr[itr_sample,1] = transition_rate_10_tclme_cg
            transition_rate_tclme_arr[itr_sample,0] = rate_matrix[0,1]
            transition_rate_tclme_arr[itr_sample,1] = rate_matrix[1,0]
            mol_pops[itr_sample,:] += quantum_observables[2]
            current[itr_sample,:] += quantum_observables[0]
            x_vec[itr_sample] = x_final
            p_vec[itr_sample] = p_final
            next_sample_time += sample_time
    # print(nhops)
    return x_vec,p_vec,current,mol_pops,active_surface,transition_rate_tclme_cg_arr,\
            transition_rate_cg_arr,transition_rate_tclme_arr

# results = joblib.Parallel(n_jobs=10)(joblib.delayed(one_time_loop)(itr_traj) for itr_traj in tqdm(range(n_trajectories)))
results = joblib.Parallel(n_jobs=nthreads,max_nbytes=None,batch_size=1)(joblib.delayed(one_time_loop)(itr_traj,active_surfaces[itr_traj],
        x_array_initial[itr_traj],p_array_initial[itr_traj],n_timesteps,Nleads,
        el_lead_couplings_func,dim_rho,El_Nuclear_Couplings_cl,Nel,V_Km,
        dt_init,max_expan_order,rk_coeff,Vib_Freq_cl,
        quantum_observables_object,rho_deriv,rho_temp,rho_input,active_surface_force_funcs,
        sparse_propagation_object,rho_output,pair_values_this_x,trajectory_seeds[itr_traj]) for itr_traj in range(n_trajectories))

x_total = np.zeros(n_samples,dtype=float)
p_total = np.zeros(n_samples,dtype=float)
xsq_total = np.zeros(n_samples,dtype=float)
psq_total = np.zeros(n_samples,dtype=float)
current_total = np.zeros((n_samples,Nleads),dtype=float)
mol_pops_total = np.zeros((n_samples,dim_rho),dtype=float)
transition_rate_total = np.zeros((n_samples,2),dtype=float)
transition_rate_tclme_total = np.zeros((n_samples,2),dtype=float)
transition_rate_cg_total = np.zeros((n_samples,2),dtype=float)
transition_rate_tclme_cg_total = np.zeros((n_samples,2),dtype=float)
active_surfaces_final = np.zeros(n_trajectories,dtype=int)
for itr_traj in range(n_trajectories):
    x_total += results[itr_traj][0]
    p_total += results[itr_traj][1]
    xsq_total += results[itr_traj][0]**2
    psq_total += results[itr_traj][1]**2
    current_total += results[itr_traj][2]
    mol_pops_total += results[itr_traj][3]
    active_surfaces_final[itr_traj] = results[itr_traj][4]
    transition_rate_tclme_cg_total += results[itr_traj][5]
    transition_rate_cg_total += results[itr_traj][6]
    transition_rate_tclme_total += results[itr_traj][7]

x_av = x_total/n_trajectories
p_av = p_total/n_trajectories
xsq_av = xsq_total/n_trajectories
psq_av = psq_total/n_trajectories
current_av = current_total/n_trajectories
mol_pops_av = mol_pops_total/n_trajectories
transition_rate_av = transition_rate_total/n_trajectories
transition_rate_tclme_cg_av = transition_rate_tclme_cg_total/n_trajectories
transition_rate_cg_av = transition_rate_cg_total/n_trajectories
transition_rate_tclme_av = transition_rate_tclme_total/n_trajectories

### SAVING INFORMATION ### 

repo = Path(args.root)
results_dir = repo / f"results/gamma_{Gamma_choice}eV/omega_{Vib_Freq_cl[0]}eV/prop_time_{(max_time * Vib_Freq_cl[0]):.2f}_Omegat/cgt_{(cgt * dt_init * Vib_Freq_cl[0]):.2f}_Omegat/traj_{n_trajectories}/"
results_dir.mkdir(parents=True, exist_ok=True)
transient_dir = results_dir / "transient"
transient_dir.mkdir(parents=True, exist_ok=True)
plots_dir = transient_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
transient_ensemble_av_dir = transient_dir / "ensemble_av"
transient_ensemble_av_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(transient_ensemble_av_dir / "mol_pops_cg_heom.dat",
           np.vstack((time_vec,mol_pops_av.T)).T)
np.savetxt(transient_ensemble_av_dir / "ke_cg_heom.dat",
           np.vstack((time_vec,psq_av * Vib_Freq_cl[0]/2)).T)
np.savetxt(transient_ensemble_av_dir / "pe_cg_heom.dat",
           np.vstack((time_vec,xsq_av * Vib_Freq_cl[0]/2)).T)
# np.savetxt(transient_ensemble_av_dir / "transition_rate_tclme_cg_heom.dat",
#            np.vstack((time_vec,transition_rate_tclme_cg_av.T)).T)
# np.savetxt(transient_ensemble_av_dir / "transition_rate_cg_heom.dat",
#            np.vstack((time_vec,transition_rate_cg_av.T)).T)
# np.savetxt(transient_ensemble_av_dir / "transition_rate_tclme_heom.dat",
#            np.vstack((time_vec,transition_rate_tclme_av.T)).T)

src = repo / f"config/parameters_{args.slurm_id}_cg_sheom.txt"
dst = transient_dir / src.name

os.replace(src, dst)

