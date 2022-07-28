# ==============================================================================
# Standard Python modules
# ==============================================================================
from __future__ import print_function
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import constitutive, elements, pyTACS

comm = MPI.COMM_WORLD

bdfFile = os.path.join(os.path.dirname(__file__), '../input files/CRM_box_2nd.bdf')
FEAAssembler = pyTACS(bdfFile, comm=comm)

# Material properties
rho = 2780.0        # density kg/m^3
E = 73.1e9          # Young's modulus (Pa)
nu = 0.33           # Poisson's ratio
ys = 324.0e6        # yield stress

# Shell thickness
t = 0.01            # m

# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)

    return elem

# Set up elements and TACS assembler
FEAAssembler.initialize(elemCallBack)

modalProb = FEAAssembler.createModalProblem("CRMModes", 36.0, 10)
modalProb.solve()
nmodes = modalProb.getNumEigs()
for mode_i in range(nmodes):
    eigVal, eigVector = modalProb.getVariables(mode_i)
    f_hz = np.sqrt(eigVal)/2/np.pi
    print(f"mode {mode_i}: {f_hz} hz")
modalProb.writeSolution()

modalProb.solve()
modalProb.writeSolution()

numEigs = modalProb.getNumEigs()

for i in range(numEigs):
    eigVal, eigVec = modalProb.getVariables(i)
    f_hz = np.sqrt(eigVal) / 2 / np.pi
    print(f"freq {i+1}: {f_hz}")

transientProb = FEAAssembler.createTransientProblem("CRMResonance", 0.0, 5.0, 100, options={"printLevel": 2})

eigVal0, eigVec0 = modalProb.getVariables(0)
eigFreq0 = np.sqrt(eigVal)

timeSteps = transientProb.getTimeSteps()
for step_i, t in enumerate(timeSteps):
    F = 1e3 * eigVec0 * np.sin(eigFreq0*t)
    transientProb.addLoadToRHS(step_i, F)

transientProb.solve()
transientProb.writeSolution()
