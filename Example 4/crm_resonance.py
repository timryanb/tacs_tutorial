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
rho = 2780.0  # density kg/m^3
E = 73.1e9  # Young's modulus (Pa)
nu = 0.33  # Poisson's ratio
ys = 324.0e6  # yield stress

# Shell thickness
t = 0.01  # m


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

#### Finish the rest!
