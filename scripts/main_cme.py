# ---------------------------------------------------------------------
#
#        DEFINING HEOM AND DOING QUANTUM PROPAGATION EXAMPLE MAIN
#
# ---------------------------------------------------------------------
#
# This Python file generates the quantum HEOM and demonstrates how to use it in a 
# dynamical way within a time propagation. 
# 
# Note that one must create the Python wrappers from the Fortran subroutines first 
# (eta_gamma,sparsity,sparse_propagation). These can be run from the command line as 
# ./compile_f2py.sh
#
# There are no direct inputs, rather, one must first change the input_parameters.py and 
# system.py file to reflect the problem you want to solve. These
# are imported automatically into this code. 
#
# USAGE - :
#
#       python3 SHEOM_main.py
#
# OUTPUT -
#
#       At the moment, there is no output 
#

import source.generating_quantum_heom_class as generating_quantum_heom_class
import source.generate_heom_one_x as generate_heom_one_x
import source.calculate_quantum_observables as calculate_quantum_observables
import source.vibrational_system_setup as vibrational_system_setup
import source.system as system
from importlib import reload
from config.input_parameters import *
from pathlib import Path
import os

import scipy
import gc,joblib,tqdm,pickle # type: ignore
from numba import jit
import argparse

### OBTAIN ROOT ###

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str, required=True)
parser.add_argument("--slurm_id", type=int, required=True)
args = parser.parse_args()

### GENERATE QUANTUM HEOM INGREDIENTS ###

rho_ic = np.array([[0,0],[0,1]],dtype=float)

### VIBRATIONAL QUANTITIES ### 

seed = 42
x_array_initial,p_array_initial = vibrational_system_setup.vibrational_initial_conditions(seed=seed) # Define nuclear coordinates/momenta with initial condition
initial_populations = np.diag(rho_ic)
possible_active_surfaces_list = np.arange(dim_rho)
active_surfaces = np.random.choice(possible_active_surfaces_list, size=n_trajectories,p=initial_populations)

@jit
def energy_func(El_Nuclear_Couplings_cl,Single_El_Int,x):
    energy_x = Single_El_Int[0,0] + El_Nuclear_Couplings_cl[0]*x
    return energy_x

@jit
def fermi_dirac_func(energy,temp):
    fd_occ = 1/(1 + np.exp(energy/temp))
    return fd_occ

@jit
def force_active_surface_0(x_vec):
    force = -Vib_Freq_cl[0]*x_vec
    return force

@jit
def force_active_surface_1(x_vec):
    force = -Vib_Freq_cl[0]*x_vec - El_Nuclear_Couplings_cl[0]#*p1#/Vib_Freq_cl[0]
    return force

def liouvillian(transition_01,transition_10):
    liouvillian_fgr = np.array([[-transition_10,transition_01],[transition_10,-transition_01]],dtype=float)
    return liouvillian_fgr

seed_sequence = np.random.SeedSequence(seed)
trajectory_seeds = seed_sequence.spawn(n_trajectories)

active_surface_force_funcs = [force_active_surface_0,force_active_surface_1]
def one_time_loop(itr_traj,active_surface,x_initial,p_initial,
                  n_timesteps,Gamma_choice,Vib_Freq_cl,initial_populations,traj_seed):
    print("TRAJECTORY "+str(itr_traj))
    rng = np.random.default_rng(traj_seed)
    x_vec = np.zeros(n_samples)
    p_vec = np.zeros(n_samples)
    rho_vec = initial_populations
    rho_vec_cme_0 = np.array([1,0],dtype=float)
    rho_vec_cme_1 = np.array([0,1],dtype=float)
    x_vec[0] = x_initial
    p_vec[0] = p_initial
    x_final = x_initial
    p_final = p_initial
    mol_pops = np.zeros((n_samples,2),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                    # with \rho_00(t) and \rho_11(t)
    mol_pops[0,:] = initial_populations
    n_transitions = 0
    transition_rate_arr = np.zeros((n_samples,2),dtype=float)
    transition_rate_tclme_arr = np.zeros((n_samples,2),dtype=float)
    next_sample_time = sample_time
    itr_sample = 0
    current_time = 0
    for itrt in range(1,n_timesteps):
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
        ### GENERATE MOLECULAR SYSTEM HAMILTONIAN AND MOLECULE-METAL COUPLING AT THIS VIBRATIONAL COORDINATE
        energy_x = energy_func(El_Nuclear_Couplings_cl,Single_El_Int,x_final) 
        transition_10 = Gamma_choice*fermi_dirac_func(energy_x,Temp)
        transition_01 = Gamma_choice*(1-fermi_dirac_func(energy_x,Temp))
        liouvillian_fgr = liouvillian(transition_01,transition_10)
        rho_vec_cme_0 = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),rho_vec_cme_0)
        rho_vec_cme_1 = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),rho_vec_cme_1)
        rho_deriv_cme_0 = np.dot(liouvillian_fgr,rho_vec_cme_0)
        rho_deriv_cme_1 = np.dot(liouvillian_fgr,rho_vec_cme_1)
        U_matrix_cme = np.transpose(np.vstack((rho_vec_cme_0,rho_vec_cme_1)))
        U_dot_matrix_cme = np.transpose(np.vstack((rho_deriv_cme_0,rho_deriv_cme_1)))
        rate_matrix_cme = U_dot_matrix_cme @ scipy.linalg.pinv(U_matrix_cme)
        if active_surface == 0:
            transition_rate = transition_10
        elif active_surface == 1:
            transition_rate = transition_01
        liouvillian_fgr = liouvillian(transition_01,transition_10)
        rho_vec = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),rho_vec)
        hop_this_x = rng.uniform() < transition_rate*dt_init
        if hop_this_x:
            n_transitions += 1
            active_surface = 1 - active_surface
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        current_time += dt_init
        if (current_time >= next_sample_time) and (itr_sample < n_samples - 1):
            itr_sample += 1
            transition_rate_arr[itr_sample,0] = transition_01
            transition_rate_arr[itr_sample,1] = transition_10
            transition_rate_tclme_arr[itr_sample,0] = rate_matrix_cme[0,1]
            transition_rate_tclme_arr[itr_sample,1] = rate_matrix_cme[1,0]
            mol_pops[itr_sample,:] += rho_vec
            x_vec[itr_sample] = x_final
            p_vec[itr_sample] = p_final
            next_sample_time += sample_time

    return x_vec,p_vec,active_surface,mol_pops,transition_rate_arr,transition_rate_tclme_arr

# results = joblib.Parallel(n_jobs=10)(joblib.delayed(one_time_loop)(itr_traj) for itr_traj in tqdm(range(n_trajectories)))
results = joblib.Parallel(n_jobs=nthreads,max_nbytes=None,batch_size=1)(joblib.delayed(one_time_loop)(itr_traj,active_surfaces[itr_traj],
                                x_array_initial[itr_traj],p_array_initial[itr_traj],n_timesteps,
                                Gamma_choice,Vib_Freq_cl,initial_populations,trajectory_seeds[itr_traj]) for itr_traj in range(n_trajectories))

x_total = np.zeros(n_samples,dtype=float)
p_total = np.zeros(n_samples,dtype=float)
xsq_total = np.zeros(n_samples,dtype=float)
psq_total = np.zeros(n_samples,dtype=float)
# current_total = np.zeros((n_samples,Nleads),dtype=float)
mol_pops_total = np.zeros((n_samples,dim_rho),dtype=float)
transition_rate_total = np.zeros((n_samples,2),dtype=float)
transition_rate_tclme_total = np.zeros((n_samples,2),dtype=float)
active_surfaces_final = np.zeros(n_trajectories,dtype=int)
for itr_traj in range(n_trajectories):
    x_total += results[itr_traj][0]
    p_total += results[itr_traj][1]
    xsq_total += results[itr_traj][0]**2
    psq_total += results[itr_traj][1]**2
    # current_total += results[itr_traj][2]
    mol_pops_total += results[itr_traj][3]
    active_surfaces_final[itr_traj] = results[itr_traj][2]
    transition_rate_total += results[itr_traj][4]
    transition_rate_tclme_total += results[itr_traj][5]

x_av = x_total/n_trajectories
p_av = p_total/n_trajectories
xsq_av = xsq_total/n_trajectories
psq_av = psq_total/n_trajectories
# current_av = current_total/n_trajectories
mol_pops_av = mol_pops_total/n_trajectories
transition_rate_av = transition_rate_total/n_trajectories
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

np.savetxt(transient_ensemble_av_dir / "mol_pops_cme.dat",
           np.vstack((time_vec,mol_pops_av.T)).T)
np.savetxt(transient_ensemble_av_dir / "ke_cme.dat",
           np.vstack((time_vec,psq_av * Vib_Freq_cl[0]/2)).T)
np.savetxt(transient_ensemble_av_dir / "pe_cme.dat",
           np.vstack((time_vec,xsq_av * Vib_Freq_cl[0]/2)).T)
np.savetxt(transient_ensemble_av_dir / "transition_rate_cme.dat",
           np.vstack((time_vec,transition_rate_av.T)).T)
np.savetxt(transient_ensemble_av_dir / "transition_rate_tclme_cme.dat",
           np.vstack((time_vec,transition_rate_tclme_av.T)).T)

src = repo / f"config/parameters_{args.slurm_id}_cme.txt"
dst = transient_dir / src.name

os.replace(src, dst)
