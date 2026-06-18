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
from source.input_parameters import *
from scipy import linalg

import joblib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

### GENERATE QUANTUM HEOM INGREDIENTS ###

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
rho_ic = np.array([[1,0],[0,0]],dtype=float)
rho_input = np.zeros(nnz_elements_sparse_fil,dtype=float) ; rho_input[0] = rho_ic[0,0] ; rho_input[1] = rho_ic[1,1]
# Define initial condition of molecular system

rho_output = np.zeros(nnz_elements_sparse_fil,dtype=np.float64)
mol_pops = np.zeros((n_timesteps,2),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                 # with \rho_00(t) and \rho_11(t)
current = np.zeros((n_timesteps,Nleads),dtype=float)

### VIBRATIONAL QUANTITIES ### 

x_array_initial,p_array_initial = vibrational_system_setup.vibrational_initial_conditions() # Define nuclear coordinates/momenta with initial condition
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
np.random.seed(42)
def one_time_loop(itr_traj,active_surface,x_initial,p_initial,
    n_timesteps,Nleads,el_lead_couplings_func,dim_rho,El_Nuclear_Couplings_cl,Nel,V_Km,
    dt_init,max_expan_order,rk_coeff,Vib_Freq_cl,
    quantum_observables_object,rho_deriv,rho_temp,rho_input,active_surface_force_funcs,
    sparse_propagation_object,rho_output,pair_values_this_x):
    quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
    quantum_heom_ingredients_object.return_sparse_heom_ingredients()
    ### COLLECT/DEFINE NECESSARY INGREDIENTS FOR QUANTUM HEOM PROPAGATION ###
    print("TRAJECTORY "+str(itr_traj))
    x_vec = np.zeros(n_timesteps)
    p_vec = np.zeros(n_timesteps)
    transition_rate_arr = np.zeros((n_timesteps,2),dtype=float)
    transition_rate_tclme_arr = np.zeros((n_timesteps,2),dtype=float)
    transition_rate_cme_arr = np.zeros((n_timesteps,2),dtype=float)
    transition_rate_tclme_cme_arr = np.zeros((n_timesteps,2),dtype=float)
    transition_rate_new_cme_arr = np.zeros((n_timesteps,2),dtype=float)
    x_vec[0] = x_initial
    p_vec[0] = p_initial
    x_final = x_initial
    p_final = p_initial
    mol_pops = np.zeros((n_timesteps,2),dtype=float) # Example array for 1 level, 1 mode model, which we are going to fill
                                                    # with \rho_00(t) and \rho_11(t)
    current = np.zeros((n_timesteps,Nleads),dtype=float)
    quantum_observables_ic = quantum_observables_object.return_quantum_observables_this_x(
                                rho_input,el_lead_couplings_func(Nleads,Nel,V_Km,x_final))
    mol_pops[0,:] = quantum_observables_ic[2]
    current[0,:] = quantum_observables_ic[0]
    nhops = 0
    rho_input_0 = np.concatenate((np.array([1,0],dtype=float),0*rho_input[2:]),axis=0)
    rho_input_1 = np.concatenate((np.array([0,1],dtype=float),0*rho_input[2:]),axis=0)
    rho_vec_cme_0 = np.array([1,0],dtype=float)
    rho_vec_cme_1 = np.array([0,1],dtype=float)
    rho_output_0 = rho_output.copy()
    rho_output_1 = rho_output.copy()
    rho_deriv_0 = np.zeros(nnz_elements_sparse_fil)
    rho_deriv_1 = np.zeros(nnz_elements_sparse_fil)
    transition_rate_10 = np.zeros(nnz_elements_sparse_fil)
    transition_rate_01 = np.zeros(nnz_elements_sparse_fil)
    for itrt in range(1,n_timesteps):
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
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
        transition_rate_tclme_arr[itrt,0] = rate_matrix[1,0]
        transition_rate_tclme_arr[itrt,1] = rate_matrix[0,1]
        # print(rate_matrix)
        ### OBTAIN QUANTUM OBSERVABLES FOR THIS VIBRATIONAL COORDINATE ###
        quantum_observables = quantum_observables_object.return_quantum_observables_this_x(rho_output,el_lead_couplings_this_x)
        mol_pops[itrt,:] += quantum_observables[2]
        current[itrt,:] += quantum_observables[0]
        # transition_rate = quantum_observables[3 + active_surface]
        initial_state = quantum_observables_object.split_quantum_state_this_x(rho_output)[active_surface].copy()
        initial_state_10 = quantum_observables_object.split_quantum_state_this_x(rho_output)[0].copy()
        initial_state_01 = quantum_observables_object.split_quantum_state_this_x(rho_output)[1].copy()
        # print(initial_state_10)
        # print(initial_state_01)
        linear_fit_tr = []
        ntausteps = 200000
        for itrtau in range(ntausteps):
            sparse_propagation_object.propagate(dt_init,initial_state_10, max_expan_order,
                          rk_coeff,rho_temp,initial_state_10,rho_deriv)
            linear_fit_tr.append(initial_state_10[1]/(dt_init*itrtau))
        plt.plot(dt_init*np.arange(ntausteps),linear_fit_tr) ; plt.show()
        transition_rate_arr[itrt,0] = transition_rate_10[1]/dt_init#/mol_pops[itrt,0].copy()
        transition_rate_01 = sparse_propagation_object.propagate(dt_init,initial_state_01, max_expan_order,
                          rk_coeff,rho_temp,transition_rate_01,rho_deriv)
        # print(sparse_propagation_object.propagate(dt_init,initial_state_10, max_expan_order,
        #                   rk_coeff,rho_temp,transition_rate_10,rho_deriv)[0:2])
        transition_rate_arr[itrt,1] = transition_rate_01[0]/dt_init#/mol_pops[itrt,1].copy()
        print(transition_rate_arr)
        transition_prob = np.zeros(nnz_elements_sparse_fil)
        transition_prob = sparse_propagation_object.propagate(dt_init,initial_state, max_expan_order,
            rk_coeff,rho_temp,transition_prob,rho_deriv)
        transition_prob = transition_prob[1-active_surface]#/mol_pops[itrt,active_surface]
        if transition_prob < 0:
            print(transition_prob)
        ########## CME ###########
        energy_x = energy_func(El_Nuclear_Couplings_cl,Single_El_Int,x_final) 
        transition_cme_10 = Gamma_choice*fermi_dirac_func(energy_x,Temp)
        transition_cme_01 = Gamma_choice*(1-fermi_dirac_func(energy_x,Temp))
        liouvillian_fgr = liouvillian(transition_cme_01,transition_cme_10)
        rho_vec_cme_0 = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),rho_vec_cme_0)
        rho_vec_cme_1 = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),rho_vec_cme_1)
        transition_rate_new_cme_arr[itrt,0] = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),np.array([1,0]))[1]/dt_init
        print(np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),np.array([1,0]))/dt_init)
        transition_rate_new_cme_arr[itrt,1] = np.dot(scipy.linalg.expm(liouvillian_fgr*dt_init),np.array([0,1]))[0]/dt_init
        rho_deriv_cme_0 = np.dot(liouvillian_fgr,rho_vec_cme_0)
        rho_deriv_cme_1 = np.dot(liouvillian_fgr,rho_vec_cme_1)
        U_matrix_cme = np.transpose(np.vstack((rho_vec_cme_0,rho_vec_cme_1)))
        U_dot_matrix_cme = np.transpose(np.vstack((rho_deriv_cme_0,rho_deriv_cme_1)))
        rate_matrix_cme = U_dot_matrix_cme @ linalg.pinv(U_matrix_cme)
        # mol_pops[itrt,:] = rho_vec
        transition_rate_tclme_cme_arr[itrt,0] = rate_matrix_cme[1,0]#transition_cme_10
        transition_rate_tclme_cme_arr[itrt,1] = rate_matrix_cme[0,1]#transition_cme_01
        transition_rate_cme_arr[itrt,0] = transition_cme_10
        transition_rate_cme_arr[itrt,1] = transition_cme_01
        if active_surface == 0:
            transition_rate_cme = transition_cme_10
        elif active_surface == 1:
            transition_rate_cme = transition_cme_01
        transition_prob_cme = transition_rate_cme*dt_init
        ### DO CLASSICAL VIBRATIONAL MOTION ON ACTIVE SURFACES ###
        x_final += 0.5*dt_init*p_final*Vib_Freq_cl[0]
        p_final += 0.5*dt_init*active_surface_force_funcs[active_surface](x_final)
        ### DETERMINE HOPPING ###
        # transition_prob = dt_init*transition_rate#/mol_pops[itrt,active_surface]
        hop_this_x = np.random.uniform(0,1,size=1) < transition_prob_cme
        if hop_this_x:
            active_surface = 1 - active_surface
            nhops += 1
        ### RESET MOLECULAR STATE FOR NEXT ITERATION ###
        rho_input = rho_output
        rho_input_0 = rho_output_0
        rho_input_1 = rho_output_1
        x_vec[itrt] = x_final
        p_vec[itrt] = p_final
    print(nhops)
    return x_vec,p_vec,current,mol_pops,active_surface,transition_rate_arr,transition_rate_cme_arr,\
            transition_rate_tclme_arr,transition_rate_tclme_cme_arr,transition_rate_new_cme_arr

# results = joblib.Parallel(n_jobs=10)(joblib.delayed(one_time_loop)(itr_traj) for itr_traj in tqdm(range(n_trajectories)))
results = joblib.Parallel(n_jobs=20,max_nbytes=None,batch_size=1)(joblib.delayed(one_time_loop)(itr_traj,active_surfaces[itr_traj],
        x_array_initial[itr_traj],p_array_initial[itr_traj],n_timesteps,Nleads,
        el_lead_couplings_func,dim_rho,El_Nuclear_Couplings_cl,Nel,V_Km,
        dt_init,max_expan_order,rk_coeff,Vib_Freq_cl,
        quantum_observables_object,rho_deriv,rho_temp,rho_input,active_surface_force_funcs,
        sparse_propagation_object,rho_output,pair_values_this_x) for itr_traj in range(n_trajectories))

x_total = np.zeros(n_timesteps,dtype=float)
p_total = np.zeros(n_timesteps,dtype=float)
xsq_total = np.zeros(n_timesteps,dtype=float)
psq_total = np.zeros(n_timesteps,dtype=float)
current_total = np.zeros((n_timesteps,Nleads),dtype=float)
mol_pops_total = np.zeros((n_timesteps,dim_rho),dtype=float)
transition_rate_total = np.zeros((n_timesteps,2),dtype=float)
transition_rate_cme_total = np.zeros((n_timesteps,2),dtype=float)
transition_rate_tclme_total = np.zeros((n_timesteps,2),dtype=float)
transition_rate_tclme_cme_total = np.zeros((n_timesteps,2),dtype=float)
transition_rate_new_cme_total = np.zeros((n_timesteps,2),dtype=float)
active_surfaces_final = np.zeros(n_trajectories,dtype=int)
for itr_traj in range(n_trajectories):
    x_total += results[itr_traj][0]
    p_total += results[itr_traj][1]
    xsq_total += results[itr_traj][0]**2
    psq_total += results[itr_traj][1]**2
    current_total += results[itr_traj][2]
    mol_pops_total += results[itr_traj][3]
    active_surfaces_final[itr_traj] = results[itr_traj][4]
    transition_rate_total += results[itr_traj][5]
    transition_rate_cme_total += results[itr_traj][6]
    transition_rate_tclme_total += results[itr_traj][7]
    transition_rate_tclme_cme_total += results[itr_traj][8]
    transition_rate_new_cme_total += results[itr_traj][9]

x_av = x_total/n_trajectories
p_av = p_total/n_trajectories
xsq_av = xsq_total/n_trajectories
psq_av = psq_total/n_trajectories
current_av = current_total/n_trajectories
mol_pops_av = mol_pops_total/n_trajectories
transition_rate_av = transition_rate_total/n_trajectories
transition_rate_cme_av = transition_rate_cme_total/n_trajectories
transition_rate_tclme_av = transition_rate_tclme_total/n_trajectories
transition_rate_tclme_cme_av = transition_rate_tclme_cme_total/n_trajectories
transition_rate_new_cme_av = transition_rate_new_cme_total/n_trajectories

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', linewidth=2)
plt.rc('text.latex', preamble=r'\boldmath')

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \rho_{ii}$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_cl[0]*time_vec,mol_pops_av[:,0],color='red',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,mol_pops_av[:,1],color='blue',linestyle='-',linewidth=2)

occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
                    Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
ax.set_ylim(0,1)
plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle E_{\mbox{\textbf{vib.}}}/k_{B}T$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_cl[0]*time_vec,Vib_Freq_cl[0]*(psq_av/2)/Temp,color='blue',linestyle='-',linewidth=2)
# occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
#                     Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
# ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \langle T_{ij} \rangle $",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
# ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_cme_av[:,0],color='blue',linestyle='-',linewidth=2)
# ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_cme_av[:,1],color='red',linestyle='-',linewidth=2)
# ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_tclme_cme_av[:,0],color='blue',linestyle='--',linewidth=2)
# ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_tclme_cme_av[:,1],color='red',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_new_cme_av[:,0],color='blue',linestyle=':',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_new_cme_av[:,1],color='red',linestyle=':',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_av[:,0],color='green',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_av[:,1],color='yellow',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_tclme_av[:,0],color='green',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*time_vec,transition_rate_tclme_av[:,1],color='yellow',linestyle='--',linewidth=2)
# occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
#                     Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
# ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# ax.set_ylabel(r"$\displaystyle x_{i},p_{i}$",color='black',fontsize=24,fontweight='bold')
# ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
# ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
# ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
# for itr_traj in range(n_trajectories):
#     ax.plot(Vib_Freq_cl[0]*time_vec,results[itr_traj][0],color='blue',alpha=0.5,linestyle='-',linewidth=2)
#     ax.plot(Vib_Freq_cl[0]*time_vec,results[itr_traj][1],color='red',alpha=0.5,linestyle='-',linewidth=2)
# # occ_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
# #                     Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \rho_{11} $')]
# # ax.legend(handles=occ_handles,loc='upper left',fontsize=18)
# ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
# # ax.set_ylim(0,1)
# plt.tight_layout()
# plt.show()
