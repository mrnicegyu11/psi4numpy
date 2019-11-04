"""
A Psi4 input script to compute Full Configuration Interaction from a SCF reference

Requirements:
SciPy 0.13.0+, NumPy 1.7.2+

References:
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Check energy against psi4?
compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
Be
H 1 1.33376
H 1 1.33376 2 180
symmetry c1
""")


psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)
print("SCF energy: " + str(scf_e))
# Grab data from wavfunction class
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvirt = nmo - ndocc

# Compute size of Hamiltonian in GB
from scipy.special import comb
nDet = comb(nmo, ndocc)**2
#H_Size = nDet**2 * 8e-9
#print('\nSize of the Hamiltonian Matrix will be %4.2f GB.' % H_Size)
#if H_Size > numpy_memory:
#    clean()
#    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
#                    limit of %4.2f GB." % (H_Size, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

kind = "FCI"

from helper_CI import Determinant, HamiltonianGenerator
from itertools import combinations
if kind == "FCI":
    print('Generating %d Full CI Determinants...' % (nDet))
    t = time.time()
    detList = []
    for alpha in combinations(range(nmo), ndocc):
        for beta in combinations(range(nmo), ndocc):
            detList.append(Determinant(alphaObtList=alpha, betaObtList=beta))
#######################
if kind == "CISD": #CISD
    nDet_S = ndocc * nvirt * 2
    nDet_D = 2 * comb(ndocc, 2) * comb(nvirt, 2) + ndocc**2 * nvirt**2
    nDet = 1 + nDet_S + nDet_D
    print('Generating %d CISD Determinants...' % (nDet))
    occList = [i for i in range(ndocc)]
    det_ref = Determinant(alphaObtList=occList, betaObtList=occList)
    detList = det_ref.generateSingleAndDoubleExcitationsOfDet(nmo)
    detList.append(det_ref)
#######################
elif kind == "CISDT": #CISDT
    nDet_S = ndocc * nvirt * 2
    nDet_D = 2 * comb(ndocc, 2) * comb(nvirt, 2) + ndocc**2 * nvirt**2
    nDet = 1 + nDet_S + nDet_D
    print('Generating CISDT Determinants...')
    occList = [i for i in range(ndocc)]
    det_ref = Determinant(alphaObtList=occList, betaObtList=occList)
    detList = det_ref.generateSingleAndDoubleAndTripleExcitationsOfDet(nmo)
    detList.append(det_ref)
    print('Generated %d CISDT Determinants...' % (len(detList)))


print('..finished generating determinants in %.3f seconds.\n' % (time.time() - t))

print('Generating Hamiltonian Matrix...')
import copy
import math
t = time.time()
Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)
originalHamiltonMatrix = copy.deepcopy(Hamiltonian_matrix)
print('..finished generating Matrix in %.3f seconds.\n' % (time.time() - t))
#
print('Diagonalizing Hamiltonian Matrix...')

t = time.time()

e_fci, wavefunctions = np.linalg.eigh(originalHamiltonMatrix)
print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))

fci_mol_e = e_fci[0] #+ mol.nuclear_repulsion_energy()

print('# Determinants:     % 16d' % (len(detList)))

print('SCF energy (no nucl. terms):         % 16.10f' % (scf_e + mol.nuclear_repulsion_energy()))
#print('FCI correlation:    % 16.10f' % (fci_mol_e - scf_e))
print('Reference CI energy (no nucl. terms):   % 16.10f' % (fci_mol_e))



#
if (False):
    hamiltonianSize = 20
    Hamiltonian_matrix = np.zeros((hamiltonianSize,hamiltonianSize))
    for c1,v1 in enumerate(Hamiltonian_matrix):
        for c2,v2 in enumerate(Hamiltonian_matrix[c1]):
            if c1 == c2:
                Hamiltonian_matrix[c1][c2] = c1 + 0.5
                Hamiltonian_matrix[c1][c2] += (2*c1+1)/4
            elif c1 == c2 + 2:
                Hamiltonian_matrix[c1][c2] = 0.25 * math.sqrt(c1*(c1-1))
            elif c1 == c2 - 2:
                Hamiltonian_matrix[c1][c2] = 0.25 * math.sqrt((c1+1)*(c1+2))
    #Hamiltonian_matrix[1][3] = 0
    #Hamiltonian_matrix[3][1] = 0
    #Hamiltonian_matrix[5][3] = 0
    #Hamiltonian_matrix[3][5] = 0
    print("Hamiltonian:")
    print(Hamiltonian_matrix)
    originalHamiltonMatrix = copy.deepcopy(Hamiltonian_matrix)    

def NNPT(HamiltonianMatrix_, mol, iterations_ = 10, targetRoot_ = 0,startGuess_ = None, useBeta = True, debugprint_ = False):
    debugprint = debugprint_
    root = targetRoot_
    Hamiltonian_matrix = HamiltonianMatrix_
    originalHamiltonMatrix = copy.deepcopy(HamiltonianMatrix_)
    print("### START NNPT ###")
    currentBestEnergyGuess = Hamiltonian_matrix[root][root]
    if startGuess_ is not None:
        currentBestEnergyGuess = startGuess_
    currentBestEnergyGuessList = [currentBestEnergyGuess]
    summationList = [currentBestEnergyGuess]
    if debugprint:
        print("Hamiltonian (root,root) value at start:")
        print(currentBestEnergyGuess)
        print("Hamiltonian_matrix")
        print(Hamiltonian_matrix)
    for i in range(iterations_):
        t = time.time()
        Hamiltonian_matrix = copy.deepcopy(originalHamiltonMatrix)
        for c1 in range(len(Hamiltonian_matrix)):
            Hamiltonian_matrix[c1][c1] -= currentBestEnergyGuess
        if  debugprint:
            print("Shifted Hamiltonian_matrix")
            print(Hamiltonian_matrix)
        print("################################")
        for c1 in range(len(Hamiltonian_matrix)):
            if c1 is not root : #First root
                if Hamiltonian_matrix[root][c1] < -1e-10 or Hamiltonian_matrix[root][c1] > 1e-10:
                    #print(Hamiltonian_matrix[c1])
                    #eliminate [root][c1]
                    factor = Hamiltonian_matrix[root][c1] / Hamiltonian_matrix[c1][c1]
                    for c2 in range(len(Hamiltonian_matrix[c1])):
                        Hamiltonian_matrix[root][c2] -= factor * Hamiltonian_matrix[c1][c2]
                    #eliminate all further [k][c1] for k > j 
                    for c2 in range(len(Hamiltonian_matrix[c1])):
                        if c2 > c1:
                            factor = Hamiltonian_matrix[c2][c1] / Hamiltonian_matrix[c1][c1]
                            for c3 in range(len(Hamiltonian_matrix[c1])):
                                Hamiltonian_matrix[c2][c3] -= factor * Hamiltonian_matrix[c1][c3]
                            #eliminate all further [k][c1] for k > j 
                    if debugprint:
                        print("Factor")
                        print(factor)
                        print("New Hamiltonian:")
                        print(Hamiltonian_matrix)
                        print("################################")
            if len(Hamiltonian_matrix) > 50:
                if c1 % int(len(Hamiltonian_matrix)/10) == 0:
                    print(str(float(c1)/float(len(Hamiltonian_matrix))) + "% done.")
        print("Iteration: " + str(i))
        if debugprint:
            print("Hamiltonian (0,0) value:")
            print(Hamiltonian_matrix[root][root])
        if i is not iterations_ - 1 or not useBeta:
            currentBestEnergyGuess += Hamiltonian_matrix[root][root]
        else:
            beta = Hamiltonian_matrix[root][root] / currentBestEnergyGuessList[len(currentBestEnergyGuessList) - 1]
            beta = 1.0 - beta
            beta = 1.0 / beta
            print("Beta correction value: " + str(beta))
            currentBestEnergyGuess += Hamiltonian_matrix[root][root] * beta
        print("Current Best Guess Energy: " + str(currentBestEnergyGuess + mol.nuclear_repulsion_energy()))
        currentBestEnergyGuessList.append(currentBestEnergyGuess)
        summationList.append(Hamiltonian_matrix[root][root])
        print('..finished NNPT Iteration in %.3f seconds.\n' % (time.time() - t))

    print("########################")
    print("NNPT FINISHED")
    finalValue = currentBestEnergyGuessList[len(currentBestEnergyGuessList) - 1]
    print("NNPT Energy Value: " + str(finalValue))
    print("########################")
    return finalValue,currentBestEnergyGuessList

final,summationL = NNPT(Hamiltonian_matrix,mol,iterations_=10)
from mpmath import shanks,nprint,richardson,mp
val = 0
print("Summation List: ")
print(summationL)
print("Final: ")
print(final)
for c,v in enumerate(summationL):
    val = v
    print("##########################")
    print("Trivial Summation estimate (i = "  + str(c) +"):" + str(val))
    print("Delta to FCI: " + str(fci_mol_e - val))
    #print(val + mol.nuclear_repulsion_energy())
    if (c > 2):
        T = shanks(summationL[:c])
        print("Shanks Transformation Estimate: " + str(T[-1][-1])) 
        print("Delta to FCI: " + str(fci_mol_e - T[-1][-1]))
        richardson_v,richardson_c = richardson(summationL[:c])
        print("Richardson Estimate: " + str(richardson_v))
        print("Delta to FCI: " + str(fci_mol_e - richardson_v))
        with mp.extraprec(2 * mp.prec): # levin needs a high working precision
            L = mp.levin(method = "levin", variant = "u")
            AC = mp.cohen_alt()
            levin_v, e = L.update_psum(summationL[:c])
            cohen_v, cohen_e = AC.update_psum(summationL[:c])
            print("Levin Sequence Estimate: " + str(levin_v))
            print("Delta to FCI: " + str(fci_mol_e - levin_v))
            print("Cohen Sequence Estimate: " + str(cohen_v))
            print("Delta to FCI: " + str(fci_mol_e - cohen_v))


print(val)
#if compare_psi4:
#    psi4.compare_values(psi4.energy('FCI'), fci_mol_e, 6, 'FCI Energy')
