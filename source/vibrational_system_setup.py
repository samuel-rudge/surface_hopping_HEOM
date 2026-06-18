import numpy as np
from source.input_parameters import *

def vibrational_initial_conditions():

    # x_vec = np.zeros(n_trajectories,dtype=float)          # Define empty array of vibrational coordinates
    # p_vec = np.zeros(n_trajectories,dtype=float)          # Define empty array of vibrational momenta
    if (ic_type == "const_ic"):
        x_array_initial = np.matmul(np.ones((n_trajectories,1)),x_vec_initial)
        p_array_initial = np.matmul(np.ones((n_trajectories,1)),p_vec_initial)
    elif (ic_type == "wigner_ic"):
        if dimensionless_coordinates:
            x_array_initial = np.random.normal(loc=0.0,scale=np.sqrt(0.5),size=n_trajectories)
            p_array_initial = np.random.normal(loc=0.0,scale=np.sqrt(0.5),size=n_trajectories)
        else:
            raise ValueError("Wigner initial condition for dimensionfull coordinates not yet implemented")
    elif (ic_type == "boltzmann_ic"):
        raise ValueError("Boltzmann initial condition not yet implemented")

    # x_vec[0,:] = x_array_initial
    # p_vec[0,:] = p_array_initial
    # return x_vec,p_vec
    return x_array_initial,p_array_initial