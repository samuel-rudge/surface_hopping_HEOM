# ---------------------------------------------------------------------
#
#        CREATION AND ANNIHILATION OPERATOR GENERATOR
#        FOR FERMIONIC, BOSONIC, AND JOINT FOCK SPACES
#
# ---------------------------------------------------------------------
#
# This module constructs explicit matrix representations of:
#
#   - fermionic creation and annihilation operators (d†, d)
#   - bosonic creation and annihilation operators (a†, a)
#   - combined fermion–boson tensor-product operators
#
# in finite-dimensional truncated Fock spaces.
#
# ---------------------------------------------------------------------
#
# PHYSICAL ROLE
#
# This class provides a concrete operator representation of many-body
# quantum systems in occupation-number (Fock) basis form.
#
# It is intended for constructing Hamiltonians and Liouvillians in:
#
#   - fermionic lattice / molecular electronic systems
#   - bosonic vibrational or photonic modes
#   - coupled fermion–boson hybrid systems
#
# ---------------------------------------------------------------------
#
# REPRESENTATION STRATEGY
#
# All operators are represented as explicit dense matrices acting on
# Fock-space basis states.
#
# The basis is constructed from:
#
#   - fermionic occupation-number states (antisymmetric algebra)
#   - bosonic occupation-number states (truncated harmonic oscillator)
#   - tensor products for combined systems
#
# ---------------------------------------------------------------------
#
# FERMIONIC SECTOR
#
# For m fermionic modes:
#
#   dim_el = 2^m
#
# The module constructs matrix representations of:
#
#   d[α]    : annihilation operator for fermionic mode α
#   d†[α]   : creation operator for fermionic mode α
#
# Fermionic antisymmetry is enforced explicitly via sign tracking
# in the occupation-number basis construction.
#
# Operator storage format:
#
#   D_ops[:,:,:,0] = d†
#   D_ops[:,:,:,1] = d
#
# ---------------------------------------------------------------------
#
# BOSONIC SECTOR
#
# For Nmodes bosonic modes with truncation Nbosons:
#
# Each mode is represented in a finite harmonic oscillator basis:
#
#   a† |n> = sqrt(n+1) |n+1>
#   a  |n> = sqrt(n)   |n-1>
#
# Multimode operators are constructed via Kronecker products over modes.
#
# Operator storage format:
#
#   A_ops[:,:,:,0] = a†
#   A_ops[:,:,:,1] = a
#
# ---------------------------------------------------------------------
#
# JOINT FOCK SPACE
#
# When both fermionic and bosonic sectors are present, the total space is:
#
#   dim_rho = dim_el × Nstates_boson
#
# Operators are embedded using tensor products:
#
#   fermions \otimes identity_bosons
#   bosons   \otimes identity_fermions
#
# The resulting basis is a direct product of:
#
#   |fermionic occupation⟩ \otimes |bosonic occupation⟩
#
# ---------------------------------------------------------------------
#
# INITIALIZATION MODES
#
# The class supports three construction modes:
#
#   - "Fermi" : fermionic operators only
#   - "Bose"  : bosonic operators only
#   - "Both"  : combined fermion–boson system
#
# ---------------------------------------------------------------------
#
# OUTPUT INTERFACE
#
# Depending on initialization mode, the class provides:
#
# Fermionic:
#   D_ops, d, ddag, Fermionic_Fock_states
#
# Bosonic:
#   A_ops, a, adag, Bosonic_Fock_states
#
# Both:
#   D_ops_joint, d_joint, ddag_joint,
#   A_ops_joint, a_joint, adag_joint,
#   Both_Fock_states
#
# ---------------------------------------------------------------------
#
# NUMERICAL CHARACTERISTICS
#
# - Fully explicit dense matrix representation
# - Fermionic structure built via combinatorial enumeration
# - Bosonic structure built via truncated oscillator algebra
# - Combined spaces constructed via Kronecker products
#
# ---------------------------------------------------------------------
#
# DESIGN INTENT
#
# This implementation prioritizes:
#
#   - transparency of operator construction
#   - exact fermionic antisymmetry
#   - straightforward tensor-product embedding
#
# over computational scalability.
#
# ---------------------------------------------------------------------
#
# Note
#
# The fermionic construction explicitly enumerates occupation states
# and applies sign factors from permutation parity to enforce correct
# anticommutation relations.
#
# ---------------------------------------------------------------------

import numpy as np
import itertools
import scipy.special
from scipy.special import comb

class CreAnn():
    
    def __init__(self,Constraints,Type):
        self.Constraints = Constraints
        self.Type = Type
        if 'Fermi' in self.Type or 'fermi' in self.Type: # Test whether the user wants to generate a fermionic or bosonic only Fock space, or a joint one.
            self.fermionic_operators() # Run the appropriate code 
        elif 'Bose' in self.Type or 'bose' in self.Type:
            self.bosonic_operators()
        elif 'Both' in self.Type or 'both' in self.Type:
            self.fermionic_operators()    
            self.bosonic_operators()
            self.both_operators()

    def fermionic_operators(self):

        self.m = int(self.Constraints[0]) # Define the number of electronic levels
        self.dim_el = 2**self.m           # Calculate the size of the fermionic Fock space
        most_states = int(np.floor(np.median(list(range(self.m+1))))) # Find the total fermionic occupancy with the most Fock states 
        state_indices = np.zeros((int(comb(self.m,most_states)),self.m+1),dtype=int)
        state_index = 0
            
        states = np.zeros((int(comb(self.m,most_states)), self.m,self.m+1)) # Initialize fermionic Fock states (contains indexing of Fock states to be read by program)
        self.Fermionic_Fock_states = np.zeros((self.dim_el,self.m)) # Initialize fermionic Fock states (contains the same information but meant to be viewed for humans)
        self.d = np.zeros((self.dim_el,self.dim_el,self.m)) # Initialize array of fermionic annihilation operators ready to be filled
        
        for itrm in range(1,self.m+1): # Loop through all possible total electron occupancies (e.g. if there are three levels to be filled, the total occupancy can be 0,1,2, or 3)
            Nstates = int(comb(self.m,itrm)) # Calculate the number of fermionic Fock states for that electron occupancy
            states[0:Nstates,0:itrm,itrm] = list(itertools.combinations(range(1,self.m+1),itrm)) # Generate Fock states for this electron occupancy; basically records which levels are filled. 
                                                                                                 # If m = 3 and itrm = 2, for example, the possible states are {1,2},{1,3},{2,3}, where these mean 
                                                                                                 # that the first and second levels are filled, the first and third levels are filled, and so on.
            state_indices[0:Nstates,itrm] = state_index + np.array(range(1,int(comb(self.m,itrm))+1),dtype=int) # Record the index of Fock space
            self.Fermionic_Fock_states[np.transpose(np.repeat(np.array([state_indices[0:Nstates,itrm]]),[itrm],axis=0)),(states[0:Nstates,0:itrm,itrm]-1).astype(int)] = 1 # Generate Fock states, but 
                                                                                                 # with a binary representaiton. If m = 3 and itrm = 2, we get {1,1,0},{1,0,1},{0,1,1}. 
            state_index += int(comb(self.m,itrm))   # Update the Fock space index
            for itri in range(Nstates): # Loop through all states for this electron occupancy
                if itrm == 1: # If the electron occupancy is 1, you do not need to take any commutation properties into account, and can just fill the itri-th annihilation operator with a 1 
                              # in the element that links the (0,0,...,0) state and the (0,0,...,1,...,0) state (where the 1 is in the itri-th position).
                    self.d[0,itri+1,itri] = 1
                else:
                    state_mi = np.array(states[itri,0:itrm,itrm],dtype=int) # Generate array that records which levels are filled for this state
                    ind_mi = state_indices[itri,itrm] # Find the corresponding index of this state
                    BinAry = self.Fermionic_Fock_states[state_indices[itri,itrm],:] # Generate array with ones in the position of each level that is filled (the same as state_mi but a binary representation)
                    count_ann = 0 
                    for indann in state_mi: # Loop through all filled energy levels in this state
                        Perm = int((-1)**(np.sum(BinAry[0:indann-1]))) # Calculate anti-commutation permutation contribution to bring itri-th annihilation operator to act on itri-th level
                        state_mim1 = np.delete(state_mi,count_ann) # Generate array that records which levels are filled AFTER annihilation operator is applied
                        ind_mim1 = int(state_indices[np.where((states[:,0:itrm-1,itrm-1]==state_mim1).all(axis=1))[0],itrm-1]) # Find the corresponding Fock space index for this new state. If 
                                                                                                                               # one had many fermionic levels, this might get long.
                        self.d[ind_mim1,ind_mi,indann-1] = Perm # Fill the corresponding annihilation operator in the correct position with the result of Perm
                        count_ann = count_ann + 1

        self.ddag = np.zeros((self.dim_el,self.dim_el,self.m)) # Initialize creation operators
        self.ddag = np.transpose(self.d,(1,0,2)) # Calculate creation operators as transpose of annihilation operators

        self.D_ops = np.zeros((self.dim_el,self.dim_el,self.m,2)) # Initialize array containing all annihilation and creation operators
        self.D_ops[:,:,:,0] = self.ddag # Fill with creation operators
        self.D_ops[:,:,:,1] = self.d # Fill with annihilation operators

    def bosonic_operators(self):

        self.Nmodes = int(self.Constraints[1]) # Define the number of bosonic species (vibrational modes)
        if (self.Nmodes == 1):
            Nbosons = np.array(self.Constraints[2],dtype=int) # Define the maximum occupancy of each species of boson (maximum number of phonons allowed in each mode)
        else:
            Nbosons = np.array(self.Constraints[2],dtype=int) # Define the maximum occupancy of each species of boson (maximum number of phonons allowed in each mode)

        self.Nstates = np.prod((Nbosons+1)) # Calculate total number of bosonic Fock states
        self.Bosonic_Fock_states = np.zeros((self.Nstates,self.Nmodes)) # Initialize array containing occupancies of all bosonic Fock states
        self.adag = np.zeros((self.Nstates,self.Nstates,self.Nmodes),dtype=float) # Initialize bosonic creation operator to be filled
        for itrmodes in range(self.Nmodes): # Loop through the vibrational modes
            self.Bosonic_Fock_states[:,itrmodes] = np.transpose(np.tile(np.repeat(np.arange(Nbosons[itrmodes]+1),np.prod(Nbosons[itrmodes+1:]+1)),np.prod(Nbosons[:itrmodes]+1))) # Generate 
                                                                                        # occupancy of this mode for each bosonic Fock state
            adag_mode = np.zeros((Nbosons[itrmodes]+1,Nbosons[itrmodes]+1),dtype=float) # Initialize the bosonic creation operator for just that mode (as if Nmodes = 1)
            adagdiag = np.sqrt(np.arange(1,Nbosons[itrmodes]+1)) # Generate vector of sqrt(1:Nbosons) to be put in bosonic creation operator
            np.fill_diagonal(adag_mode[1:,:],adagdiag) # Fill lower diagonal of bosonic creation operator with sqrt(1:Nbosons). This is the definition of a bosonic creation operator.
            for itrmodesdash in range(self.Nmodes): # Now this needs to be put into the larger Fock space of all bosonic modes. Loop through 
                if itrmodesdash < itrmodes:
                    adag_mode = np.kron(np.eye(Nbosons[itrmodesdash]+1),adag_mode) # Use Kronecker product to tensor the creation operator for this mode into the entire Fock space
                elif itrmodesdash > itrmodes:
                    adag_mode = np.kron(adag_mode,np.eye(Nbosons[itrmodesdash]+1))
            self.adag[:,:,itrmodes] = adag_mode # Put the creation operator for this mode into the corresponding position in the creation operator array

        self.a = np.zeros((self.Nstates,self.Nstates,self.Nmodes),dtype=float) # Initialize and calculate the corresponding annihilation operators as the transpose of the creation operators
        self.a = np.transpose(self.adag,(1,0,2))

        self.A_ops = np.zeros((self.Nstates,self.Nstates,self.Nmodes,2),dtype=float) # Initialize and fill array containing all annihilation and creation operators
        self.A_ops[:,:,:,0] = self.adag
        self.A_ops[:,:,:,1] = self.a

    def both_operators(self):

        self.dim_rho = self.dim_el*self.Nstates # Calculate the total number of Fock states for the joint bosonic and fermionic space
        self.ddag_joint = np.zeros((self.dim_rho,self.dim_rho,self.m),dtype=float) # Initialize all arrays for joint fermionic and bosonic creation and annihilation operators
        self.d_joint = np.zeros((self.dim_rho,self.dim_rho,self.m),dtype=float)
        self.adag_joint = np.zeros((self.dim_rho,self.dim_rho,self.Nmodes),dtype=float)
        self.a_joint = np.zeros((self.dim_rho,self.dim_rho,self.Nmodes),dtype=float)
        self.D_ops_joint = np.zeros((self.dim_rho,self.dim_rho,self.m,2),dtype=float)
        self.A_ops_joint = np.zeros((self.dim_rho,self.dim_rho,self.Nmodes,2),dtype=float)

        for itrm in range(self.m): # Loop through all annihilation and creation operators in fermionic Fock space
            self.ddag_joint[:,:,itrm] = np.kron(np.eye(self.Nstates),self.ddag[:,:,itrm]) # Use Kronecker product to tensor fermionic-only creation and annihilation operators to entire 
                                                                                          # bosonic and fermionic Fock space
            self.d_joint[:,:,itrm] = np.kron(np.eye(self.Nstates),self.d[:,:,itrm])
        for itrmodes in range(self.Nmodes): # Loop through all annihilation and creation operators in bosonic Fock space 
            self.adag_joint[:,:,itrmodes] = np.kron(self.adag[:,:,itrmodes],np.eye(self.dim_el)) # Use Kronecker product to tensor bosonic-only creation and annihilation operators to entire
                                                                                                # bosonic and fermionic Fock space`
            self.a_joint[:,:,itrmodes] = np.kron(self.a[:,:,itrmodes],np.eye(self.dim_el))

        self.D_ops_joint[:,:,:,0] = self.ddag_joint # Fill array for fermionic creation and annihilation operators
        self.D_ops_joint[:,:,:,1] = self.d_joint
        self.A_ops_joint[:,:,:,0] = self.adag_joint # Filla array for bosonic creation and annihilation operators
        self.A_ops_joint[:,:,:,1] = self.a_joint

        self.Both_Fock_states = np.concatenate((np.tile(self.Fermionic_Fock_states,[self.Nstates,1]),np.repeat(self.Bosonic_Fock_states,self.dim_el,axis=0)),axis=1) # Create joint 
                                                    # bosonic and fermionic Fock space by concatenating previous Fock spaces. The ordering of states should match the ordering 
                                                    # in the annihilation and creation operators. The first m columns are fermionic occupations, and the latter Nstates columns are
                                                    # bosonic occupancies.  

    def return_operators(self):
        if 'Fermi' in self.Type or 'fermi' in self.Type:
            return self.D_ops,self.d,self.ddag,self.Fermionic_Fock_states # pylint: disable=unbalanced-tuple-unpacking
        elif 'Bose' in self.Type or 'bose' in self.Type:
            return self.A_ops,self.a,self.adag,self.Bosonic_Fock_states # pylint: disable=unbalanced-tuple-unpacking
        elif 'Both' in self.Type or 'both' in self.Type:
            return self.D_ops_joint,self.d_joint,self.ddag_joint,self.A_ops_joint,self.a_joint,self.adag_joint,self.Both_Fock_states

###################################################### JUNK ########################################################

# np.savetxt(adag_file,np.matmul(adag[:,:,0],a[:,:,0]),fmt='%4.2f')
# adag_file.write('------------------------------------------------------------------------------------------------------------')
# adag_file.write('------------------------------------------------------------------------------------------------------------')
# adag_file.write('------------------------------------------------------------------------------------------------------------')
# np.savetxt(adag_file,np.matmul(adag[:,:,1],a[:,:,1]),fmt='%4.2f')
# adag_file.close()
