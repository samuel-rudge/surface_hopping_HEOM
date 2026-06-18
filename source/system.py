"""
HEOM + TCL-DERIVED SURFACE HOPPING
System Definition and Fock Space Construction Module

This module defines the full quantum system used in the HEOM simulation
framework, including the construction of:

    - fermionic and bosonic Fock spaces
    - creation and annihilation operators
    - system Hamiltonian (electronic + vibrational contributions)
    - electron-vibration coupling terms
    - coordinate-dependent Hamiltonian representations
    - Hamiltonian derivatives with respect to nuclear coordinates

It acts as the central interface between abstract model parameters
(input_parameters.py) and the operator-level representation required
for HEOM propagation and friction/correlation calculations.

-----------------------------------------------------------------------
CORE FUNCTIONALITY
-----------------------------------------------------------------------

The main routine system_operators(...) performs the following tasks:

1. Fock space construction
   - Generates fermionic (electronic) operators
   - Generates bosonic (vibrational) operators (if present)
   - Builds full system Hilbert space representation

2. Hamiltonian assembly
   - Electronic Hamiltonian (site energies, hopping, interactions)
   - Vibrational Hamiltonian (quantum modes if enabled)
   - Electron-vibration coupling terms
   - Optional small-polaron transformation contributions

3. Operator dressing (if vibrational coupling is present)
   - Franck-Condon transformations
   - Dressed fermionic operators
   - Coordinate-shifted representations of electronic operators

4. Nuclear coordinate dependence
   - Construction of x-dependent Hamiltonian H(x)

5. Output operator set
   - System Hamiltonian
   - Coordinate-dependent Hamiltonians
   - Hamiltonian derivatives with respect to nuclear coordinates
   - Optional vibrational operator structures

-----------------------------------------------------------------------
MODEL FLEXIBILITY
-----------------------------------------------------------------------

The module supports:

    - single or multi-site electronic systems (Nel)
    - multiple quantum vibrational modes
    - classical vibrational coordinates
    - electron-electron interactions
    - electron-vibration coupling in both diagonal and non-diagonal form
    - optional small-polaron transformation

-----------------------------------------------------------------------
DEPENDENCIES

This module depends on:

    - CreAnn: construction of creation/annihilation operators and Fock space
    - Franck_Condon: vibrational dressing and transformation matrices
    - input_parameters.py: global system and numerical configuration
    - numpy / scipy: linear algebra and matrix exponentials

-----------------------------------------------------------------------
COORDINATE REPRESENTATION

For HEOM propagation and friction evaluation, the Hamiltonian is evaluated
on a discrete nuclear coordinate grid x_vec, including finite-difference
shifts used for derivative-based coupling terms.

This enables direct coupling between:
    - nuclear motion (classical coordinate x)
    - electronic dynamics (quantum subsystem)
    - HEOM dissipative environment

-----------------------------------------------------------------------
OUTPUT STRUCTURE

The module returns:

    - creation and annihilation operators
    - Fock space basis states
    - system Hamiltonian H
    - coordinate-dependent Hamiltonian H(x)
    - Hamiltonian derivatives ∂H/∂x
    - optional vibrational operator structures and transformations

These objects form the basis for all subsequent HEOM propagation,
friction tensor evaluation, and correlation function calculations.

-----------------------------------------------------------------------
EXECUTION MODE

When executed directly, this module runs a standalone test example that:

    - constructs a minimal model system
    - builds the Hamiltonian explicitly
    - writes operator matrices to text files

This mode is intended for verification and debugging of the operator
construction pipeline.
"""

import numpy as np
from numpy.core.numeric import ones
from source.constants import *  # pylint: disable=unused-import
from source.input_parameters import * #  pylint: disable=unused-import
import source.CreAnn as CreAnn
import source.Franck_Condon as Franck_Condon
import itertools

def system_operators(Single_El_Int,Double_El_Int,Nel,N_qu_vib_modes,El_Nuclear_Couplings_cl,
                     max_occ_qu_vib_modes,dim_rho):

    """
    Generate all system related operators and associated Fock states
    """

    # Generate raw creation and annihilation operators from CreAnn class
    
    """
    For explanation of parameters, see input_parameters.py. For explanation of creation and annihilation operators, see CreAnn.py
    """

    Constraints = [Nel,N_qu_vib_modes,max_occ_qu_vib_modes] # Define constraints of system to be inputted into CreAnn function
    if bool(N_qu_vib_modes):
        CreAnn1 = CreAnn.CreAnn(Constraints,'Both') # Run program to generate creation and annihilation operators and Fock space
        d_ops,d,ddag,b_ops,b,bdag,Fock_states = CreAnn1.return_operators() # pylint: disable=unbalanced-tuple-unpacking
    else:
        CreAnn1 = CreAnn.CreAnn(Constraints,'Fermi') # Run program to generate creation and annihilation operators and Fock space
        d_ops,d,ddag,Fock_states = CreAnn1.return_operators() # pylint: disable=unbalanced-tuple-unpacking

    # Generate electronic part of Hamiltonian

    Ham_el = np.zeros((dim_rho,dim_rho),dtype=complex) 
    el_occ_op = np.matmul(ddag[:,:,0],d[:,:,0])
    el_occ_op_log = np.array(el_occ_op,dtype=bool)
    if Nel == 1:
        Ham_el += (Single_El_Int[0,0])*np.matmul(ddag[:,:,0],d[:,:,0])
    elif Nel > 1:
        for itr_el1 in range(Nel):
            for itr_el2 in range(Nel):
                Ham_el += (Single_El_Int[itr_el1,itr_el2])*np.matmul(ddag[:,:,itr_el1],d[:,:,itr_el2])
                if (itr_el1 < itr_el2):
                    Ham_el += Double_El_Int[itr_el1,itr_el2]*np.matmul(ddag[:,:,itr_el1],np.matmul(d[:,:,itr_el1],np.matmul(ddag[:,:,itr_el2],d[:,:,itr_el2])))

    # Generate x-dependent molecular Hamiltonian function

    def molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,x_vec):

        # molham_x = np.zeros((dim_rho,dim_rho),dtype=complex)
        molham_x = Ham_el + El_Nuclear_Couplings_cl[0]*x_vec*np.matmul(d_ops[:,:,0,0],d_ops[:,:,0,1])

        return molham_x
    
    def force_active_surface_0(x_vec):

        force = -Vib_Freq_cl[0]*x_vec
        return force

    def force_active_surface_1(x_vec):

        force = -Vib_Freq_cl[0]*x_vec - El_Nuclear_Couplings_cl[0]
        return force

    active_surface_force_funcs = [force_active_surface_0,force_active_surface_1]

    ### Define logical versions of all system operators ###

    d_ops_log = np.array(d_ops,dtype=bool)
    rho_0_log = np.array(rho_0,dtype=bool)
    identity_dim_rho = np.eye(dim_rho)
    molham_log = np.logical_or(np.array(molham_func(dim_rho,d_ops,El_Nuclear_Couplings_cl,np.array([10])),dtype=bool),np.eye(dim_rho,dtype=bool))
    
    if N_qu_vib_modes == 0:
        return (d_ops,d,ddag,Fock_states,molham_func,molham_log,d_ops_log,rho_0_log,identity_dim_rho,el_occ_op,el_occ_op_log,
                active_surface_force_funcs)
    else:
        return (d_ops,d,ddag,Fock_states,molham_func,molham_log,d_ops_log,rho_0_log,identity_dim_rho,el_occ_op,el_occ_op_log,\
                active_surface_force_funcs,b_ops,b,bdag)
            

if __name__=='__main__': # Example parameters if code is run directly from terminal and not called as a function.

    Nel = 2                                                                       
    N_qu_vib_modes = 1 # Must be 1 for small polaron transformation                                                                       
    max_occ_qu_vib_modes = [2]                                                            
    dim_el = 2**Nel # Number of fermionic Fock states
    dim_ph = np.prod(np.array(max_occ_qu_vib_modes,dtype=int)+1) # Number of bosonic Fock states
    dim_rho = dim_el*dim_ph # Number of states in the total fermionic-bosonic Fock space

    Single_El_Int_0 = np.array([[0.1]],dtype=float)                 # Energies of levels included in system, as well as hopping between levels
    Double_El_Int_0 = np.array([[0]],dtype=float)                   # Coulomb interactions between fermions - expressed as lower triangular matrix filled with two-particle interactions
    Vib_Freq_qu = np.array([0.02],dtype=float)                          # Vibrational frequency of phonon modes included in transport
    El_Ph_Int = np.array([[0.01]],dtype=float)                      # Electron-phonon coupling strength for the two types of couplings (bond length and site rigidity)   
    Single_El_Int = Single_El_Int_0 - (El_Ph_Int**2)/Vib_Freq_qu        # Fermionic energies after small polaron transformation

    if Nel > 1:
        Double_El_Int = Double_El_Int_0 - 2*(np.triu(np.matmul(El_Ph_Int.transpose(),El_Ph_Int)) + np.triu(np.matmul(El_Ph_Int.transpose(),El_Ph_Int)).transpose())/Vib_Freq_qu 
                                                                    # Coulomb fermionic interactions after small polaron transformation
    else:
        Double_El_Int = Double_El_Int_0
    
    Constraints = [Nel,N_qu_vib_modes,max_occ_qu_vib_modes]
    if N_qu_vib_modes == 0:  # Test whether the user wants to generate a fermionic or bosonic only Fock space, or a joint one.
        CreAnn1 = CreAnn.CreAnn(Constraints,'Fermi')
        d_ops,d,ddag,Fermionic_Fock_states = CreAnn1.return_operators()
    elif Nel == 0:
        CreAnn1 = CreAnn.CreAnn(Constraints,'Bose')
        a_ops,a,adag,Bosonic_Fock_states = CreAnn1.return_operators()
    elif N_qu_vib_modes != 0 and Nel != 0:
        CreAnn1 = CreAnn.CreAnn(Constraints,'Both') 
        d_ops,d,ddag,a_ops,a,adag,Both_Fock_states = CreAnn1.return_operators()

    Ham = np.zeros((dim_rho,dim_rho),dtype=float) # Initialize Hamiltonian (after small polaron transformation) to be filled
    Ham += Vib_Freq_qu[0]*np.matmul(adag[:,:,0],a[:,:,0]) # Fill Hamiltonian with bosonic energies (after small polaron transformation)
    for itrm1 in range(Nel): # Loop through fermionic levels 
        Ham += Single_El_Int[0,itrm1]*np.matmul(ddag[:,:,itrm1],d[:,:,itrm1])  # Add the energy of that fermionic level to the Hamiltonian
        for itrm2 in range(Nel): # Loop through all fermionic levels again to take into account double electron interactions
            Ham += Double_El_Int[itrm1,itrm2]*np.matmul(np.matmul(ddag[:,:,itrm1],d[:,:,itrm1]),np.matmul(ddag[:,:,itrm2],d[:,:,itrm2]))  # Include double electron interactions in Hamiltonian

    ham_file = open('Hamiltonian.txt',"w") # Open the Hamiltonian file with write access
    ham_file.write("-----------------------------------------------------------------------------------HAMILTONIAN-------------------------------------------------------------------------------\n")
    np.savetxt(ham_file,Ham,fmt='%3.2f') # Input the Hamiltonian
    ham_file.close()

    ph_file = open('Bosonic_Operators.txt',"w") # Do the same with all bosonic and fermionic annihilation and creation operators
    ph_file.write("-----------------------------------------------------------------------------------BOSONIC CREATION OPERATORS----------------------------------------------------------------------\n")
    np.savetxt(ph_file,adag[:,:,0],fmt='%4.2f')
    ph_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    np.savetxt(ph_file,adag[:,:,1],fmt='%4.2f')
    ph_file.write("---------------------------------------------------------------------------------BOSONIC ANNIHILATION OPERATORS--------------------------------------------------------------------\n")
    np.savetxt(ph_file,a[:,:,0],fmt='%4.2f')
    ph_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    np.savetxt(ph_file,a[:,:,1],fmt='%4.2f')
    ph_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    ph_file.close()

    el_file = open('Fermionic_Operators.txt',"w")
    el_file.write("-----------------------------------------------------------------------------------FERMIONIC CREATION OPERATORS----------------------------------------------------------------------\n")
    np.savetxt(el_file,ddag[:,:,0],fmt='%-2.1i')
    el_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    np.savetxt(el_file,ddag[:,:,1],fmt='%-2.1i')
    el_file.write("---------------------------------------------------------------------------------FERMIONIC ANNIHILATION OPERATORS--------------------------------------------------------------------\n")
    np.savetxt(el_file,d[:,:,0],fmt='%-2.1i')
    el_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    np.savetxt(el_file,d[:,:,1],fmt='%-2.1i')
    el_file.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
    el_file.close()
