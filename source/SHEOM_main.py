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
#     python3 SHEOM_main.py
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

import os
import source.generating_quantum_heom_class as generating_quantum_heom_class
import source.sparse_propagation_python as sparse_propagation_python
import source.calculate_quantum_observables as calculate_quantum_observables
import source.vibrational_system_setup as vibrational_system_setup
from source.input_parameters import *
import gc,tracemalloc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

### DEFINE NUMBER OF THREADS USED IN PROPAGATION ### 

os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUM_THREADS"] = "10"

### GENERATE QUANTUM HEOM INGREDIENTS ###

quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
sparse_heom_ingredients = quantum_heom_ingredients_object.return_sparse_heom_ingredients()
molecular_system_ingredients = quantum_heom_ingredients_object.return_molecular_system_ingredients()
quantum_observables_object = calculate_quantum_observables.quantum_observables_class(
    sparse_heom_ingredients,molecular_system_ingredients
)

### COLLECT/DEFINE NECESSARY INGREDIENTS FOR QUANTUM HEOM PROPAGATION ###

molham_func = molecular_system_ingredients[4] # Function taking vibrational coordinates as input and returning 
                                              # H_mol as output
active_surface_force_funcs = molecular_system_ingredients[11]
d_ops = molecular_system_ingredients[0] # Generate fermionic ann./cre. operators in molecular Hilbert space
pair_info_row_fil = sparse_heom_ingredients[0]        # Rows of quantum HEOM superoperator containing nonzero elements
pair_info_col_fil = sparse_heom_ingredients[1]        # Columns of quantum HEOM superoperator containing nonzero elements
npairs_fil = sparse_heom_ingredients[3]               # Number of filled elements in quantum HEOM superoperator
nnz_elements_sparse_fil = sparse_heom_ingredients[5]  # Number of ADO elements coupled to dynamics
del(sparse_heom_ingredients)
gc.collect()

rho_deriv = np.zeros(nnz_elements_sparse_fil,dtype=np.float64) # Two arrays necessary for Runge-Kutta algorithm
rho_temp = np.zeros(nnz_elements_sparse_fil,dtype=np.float64)
rho_ic = np.array([[1,0],[0,0]],dtype=np.float64)
rho_input = np.zeros(nnz_elements_sparse_fil,dtype=np.float64) ; rho_input[0] = rho_ic[0,0] ; rho_input[1] = rho_ic[1,1]
rho_output = np.zeros(nnz_elements_sparse_fil,dtype=np.float64)
pair_values_this_x = np.zeros(npairs_fil,dtype=np.float64)
                                                          # Define initial condition of molecular system
n_print_timesteps = n_timesteps
mol_pops = np.zeros((n_print_timesteps,2),dtype=np.float64) # Example array for 1 level, 1 mode model, which we are going to fill
                                                # with \rho_00(t) and \rho_11(t)
current = np.zeros((n_print_timesteps,Nleads),dtype=np.float64)
active_surfaces_tracked = np.zeros((n_print_timesteps,dim_rho),dtype=np.float64)

### SPARSE PROPAGATION PREPARATION ###

sparse_propagation_object = sparse_propagation_python.sparse_propagator(
    pair_info_row_fil, pair_info_col_fil, np.zeros(npairs_fil,dtype=np.float64), nnz_elements_sparse_fil
    )


### VIBRATIONAL QUANTITIES ### 

x_vec,p_vec = vibrational_system_setup.vibrational_initial_conditions() # Define nuclear coordinates/momenta with initial condition
x_final = x_vec[0,:]
p_final = p_vec[0,:]
initial_populations = np.diag(rho_ic)
possible_active_surfaces_list = np.arange(dim_rho)
active_surfaces = np.random.choice(possible_active_surfaces_list, size=n_trajectories,p=initial_populations)
quantum_observables_ic = quantum_observables_object.return_quantum_observables_this_x(rho_input,el_lead_couplings_func(Nleads,Nel,V_Km,x_vec))
mol_pops[0,:] = quantum_observables_ic[2]
current[0,:] = quantum_observables_ic[0]
active_surfaces_tracked[0,:] = [np.sum(active_surfaces == 0)/n_trajectories,np.sum(active_surfaces == 1)/n_trajectories]

ham_this_x = molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_final[0]) 
                                            # Return mol. Hamiltonian at this vibrational coordinate
el_lead_couplings_this_x = el_lead_couplings_func(Nleads,Nel,V_Km,x_final) 
                                    # Return molecule-metal coupling at this point
### GENERATE HEOM AT THIS VIBRATIONAL COORDINATE ###
pair_values_this_x = quantum_heom_ingredients_object.return_sparse_heom_one_x(
                                            ham_this_x,el_lead_couplings_this_x,pair_values_this_x)
pair_values_this_x_1 = pair_values_this_x

### TIME-PROPAGATION EXAMPLE ###
percentage_complete_old = 0
previous_hop_time_iteration = 0
n_switches_total = 0

tracemalloc.start()

def print_memory_diff(snapshot1, snapshot2, msg="Memory difference"):
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print(f"--- {msg} ---")
    for stat in stats[:5]:  # top 5 lines
        print(stat)
    print()

def get_memory_usage_mib(snapshot):
    stats = snapshot.statistics('filename')
    total = sum(stat.size for stat in stats)
    return total / (1024* 1024)

snapshot_prev = tracemalloc.take_snapshot()
memory_usage = []
rho_heom_temp = np.tile(rho_input,(n_trajectories,1))
for itrt in range(1,n_timesteps):
    percentage_complete = 100*itrt/n_timesteps
    if percentage_complete > (percentage_complete_old + 0.1):
        print_yn = True
        print(percentage_complete)
        percentage_complete_old = percentage_complete
    else:
        print_yn = False
    mol_pops_temp_total = np.zeros(dim_rho,dtype=np.float64)
    current_temp_total = np.zeros(Nleads,dtype=np.float64)
    for itr_traj in range(n_trajectories):
        x_final = x_vec[itrt-1,itr_traj]
        p_final = p_vec[itrt-1,itr_traj]
        rho_input = rho_heom_temp[itr_traj,:]
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surfaces[itr_traj]](x_final)
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        ### GENERATE MOLECULAR SYSTEM HAMILTONIAN AND MOLECULE-METAL COUPLING AT THIS VIBRATIONAL COORDINATE
        ham_this_x = molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_final) 
                                            # Return mol. Hamiltonian at this vibrational coordinate
        el_lead_couplings_this_x = el_lead_couplings_func(Nleads,Nel,V_Km,x_final) 
                                            # Return molecule-metal coupling at this point
        ### GENERATE HEOM AT THIS VIBRATIONAL COORDINATE ###
        pair_values_this_x = quantum_heom_ingredients_object.return_sparse_heom_one_x(
                                            ham_this_x,el_lead_couplings_this_x,pair_values_this_x)
                                            # This returns the values of the nonzero elements
                                            # of the HEOM Liouvillian at this vibrational coordinate. 
        ### DO QUANTUM PART OF PROPAGATION ###
        sparse_propagation_object.update_values(pair_values_this_x)
        rho_output = sparse_propagation_object.propagate(dt_init,rho_input, max_expan_order, rk_coeff,rho_temp,rho_output,rho_deriv)
        ### OBTAIN QUANTUM OBSERVABLES FOR THIS VIBRATIONAL COORDINATE ###
        rho_heom_temp[itr_traj,:] = rho_output
        quantum_observables = quantum_observables_object.return_quantum_observables_this_x(rho_output,el_lead_couplings_this_x)
        mol_pops_temp_total += quantum_observables[2]
        current_temp_total += quantum_observables[0]
        ### RESET MOLECULAR STATE FOR NEXT ITERATION ###
        # rho_input[:] = rho_output
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surfaces[itr_traj]](x_final)
        x_vec[itrt,itr_traj] = x_final
        p_vec[itrt,itr_traj] = p_final
    mol_pops[itrt,:] = mol_pops_temp_total/n_trajectories
    current[itrt,:] = current_temp_total/n_trajectories
    ### CALCULATE TRANSITION RATES ###
    previous_hop_time_iteration = itrt - 1
    mol_pop_difference = mol_pops[itrt,:] - mol_pops[previous_hop_time_iteration,:]
    # print(mol_pop_difference)
    group_classification = np.zeros(dim_rho,dtype=bool)
    group_classification[mol_pop_difference > 0] = True
    transition_prob = ((mol_pop_difference[group_classification]/mol_pops[previous_hop_time_iteration,np.logical_not(group_classification)])*
                    (mol_pop_difference[np.logical_not(group_classification)]/np.sum(mol_pop_difference[np.logical_not(group_classification)])))
    hop_this_x = (np.random.uniform(0,1,size=n_trajectories) < transition_prob)
    active_surfaces[hop_this_x & (active_surfaces == possible_active_surfaces_list[np.logical_not(group_classification)])] = (
    1 - active_surfaces[hop_this_x & (active_surfaces == possible_active_surfaces_list[np.logical_not(group_classification)])]
    )
    n_switches_total += np.sum(hop_this_x)
    snapshot_now = tracemalloc.take_snapshot()
    memory_usage.append(get_memory_usage_mib(snapshot_now))
    snapshot_prev = snapshot_now
    active_surfaces_tracked[itrt,:] = [np.sum(active_surfaces == 0)/n_trajectories,np.sum(active_surfaces == 1)/n_trajectories]


tracemalloc.stop()
plt.figure(figsize=(8,5))
plt.plot(memory_usage, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Memory usage (MB)")
plt.title("Memory usage during sparse_one_step_propagation calls")
plt.grid(True)
plt.show()

np.savetxt("mol_pops.dat",mol_pops)
np.savetxt("active_surfaces_tracked.dat",active_surfaces_tracked)
np.savetxt("x_vec.dat",x_vec) ; np.savetxt("p_vec.dat",p_vec)

print(n_switches_total)


import numpy as np
from matplotlib import pyplot as plt
from source.input_parameters import *
from matplotlib.lines import Line2D

mol_pops = np.genfromtxt("mol_pops.dat")
active_surfaces_tracked = np.genfromtxt("active_surfaces_tracked.dat")
x_vec = np.genfromtxt("x_vec.dat")
p_vec = np.genfromtxt("p_vec.dat")


xsq_av = np.mean(x_vec**2,axis=1)
psq_av = np.mean(p_vec**2,axis=1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', linewidth=2)
plt.rc('text.latex', preamble=r'\boldmath')

plot_yn = True
if plot_yn:
    fig, ax = plt.subplots()
    ax.set_ylabel(r"$\displaystyle \rho_{ii}$",color='black',fontsize=24,fontweight='bold')
    ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
    ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
    ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
    ax.plot(Vib_Freq_cl[0]*time_vec,mol_pops[:,0],color='red',linestyle='-',linewidth=2)
    ax.plot(Vib_Freq_cl[0]*time_vec,mol_pops[:,1],color='blue',linestyle='-',linewidth=2)
    ax.plot(Vib_Freq_cl[0]*time_vec,active_surfaces_tracked[:,0],color='red',linestyle='--',linewidth=2)
    ax.plot(Gamma_choice*time_vec,active_surfaces_tracked[:,1],color='blue',linestyle='--',linewidth=2)
    occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
                    Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
    ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
    ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
    ax.set_ylim(0,1)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots()
    ax.set_ylabel(r"$\displaystyle E_{\mbox{\textbf{vib.}}}/k_{B}T$",color='black',fontsize=24,fontweight='bold')
    ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
    ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
    ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
    ax.plot(Gamma_choice*time_vec,(Vib_Freq_cl[0]*(xsq_av)/2)/Temp,color='blue',linestyle='-',linewidth=2)
    # occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
    #                     Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
    # ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
    ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
    # ax.set_ylim(0,1)
    plt.tight_layout()
    plt.show()
