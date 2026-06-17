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
# USAGE - 
#
#       python3 SHEOM_main.py
#
# OUTPUT -
#
#       At the moment, there is no output 
#

import os
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUM_THREADS"] = "10"

import generating_quantum_heom_class
import generate_heom_one_x
# import sparse_propagation
import sparse_propagation_python
import calculate_quantum_observables
import vibrational_system_setup
from importlib import reload
from input_parameters import *

import gc,random,tracemalloc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

### GENERATE QUANTUM HEOM INGREDIENTS ###

quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
sparse_heom_ingredients = quantum_heom_ingredients_object.return_sparse_heom_ingredients()
molecular_system_ingredients = quantum_heom_ingredients_object.return_molecular_system_ingredients()
quantum_observables_object = calculate_quantum_observables.quantum_observables_class(sparse_heom_ingredients,
                                                                                molecular_system_ingredients)

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
                                            # This "basically" returns the values of the nonzero elements
                                            # of the HEOM Liouvillian at this vibrational coordinate. 
        ### DO QUANTUM PART OF PROPAGATION ###
        sparse_propagation_object.update_values(pair_values_this_x)
        # sparse_propagation_object = sparse_propagation_python.sparse_propagator(
        #     pair_info_row_fil, pair_info_col_fil, pair_values_this_x, nnz_elements_sparse_fil
        #     )
        rho_output = sparse_propagation_object.propagate(dt_init,rho_input, max_expan_order, rk_coeff,rho_temp,rho_output,rho_deriv)
        # rho_output = sparse_propagation.sparse_one_step_propagation(pair_info_row=pair_info_row_fil,
        #                 pair_info_col=pair_info_col_fil,pair_values=pair_values_this_x,dt=dt_init,
        #                 rho_input=rho_input,max_expan_order=max_expan_order,nthreads_liouvillian=nthreads_liouvillian,
        #                 npairs=npairs_fil,nnz_elements=nnz_elements_sparse_fil,rk_coeff=rk_coeff,rho_temp=rho_temp,
        #                 rho_deriv=rho_deriv) # Run one timestep of fourth-order Runge-Kutta HEOM propagation.
        #                                     # rho_output contains rho_mol + all ADOs at this timestep
        ### OBTAIN QUANTUM OBSERVABLES FOR THIS VIBRATIONAL COORDINATE ###
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
    # print_memory_diff(snapshot_prev, snapshot_now, i)
    memory_usage.append(get_memory_usage_mib(snapshot_now))
    snapshot_prev = snapshot_now
    # print(active_surfaces)
    active_surfaces_tracked[itrt,:] = [np.sum(active_surfaces == 0)/n_trajectories,np.sum(active_surfaces == 1)/n_trajectories]
    # n_switches = int(np.floor(transition_prob*n_trajectories)[0])
    # # print(transition_prob*n_trajectories)
    # n_switches_total += n_switches
    # if n_switches > 0:
    #     trajectories_to_switch = random.sample(trajectory_list,n_switches)
    #     print(trajectories_to_switch)
    #     for itr_traj_to_switch in trajectories_to_switch:
    #         active_surfaces[itr_traj_to_switch] = 1 - active_surfaces[itr_traj_to_switch]
    #     previous_hop_time_iteration = itrt


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
from input_parameters import *
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
