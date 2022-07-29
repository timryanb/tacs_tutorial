import os
import numpy as np
from tacs import pyTACS, constitutive, elements, functions

bdfFile = os.path.join(os.path.dirname(__file__), '../input files/pinned_beam.bdf')
FEAAssembler = pyTACS(bdfFile)


# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup material properties
    E = 70e9
    nu = 0.3
    ys = 240e6
    rho = 2700.0
    mat = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Setup constitutive properties (one thickness DV per component)
    t = 1.0e-3
    w = 0.1
    con = constitutive.IsoRectangleBeamConstitutive(mat, t=t, w=w, tNum=dvNum)
    # Set thickness to be aligned with y-direction
    transform = elements.BeamRefAxisTransform([0.0, 1.0, 0.0])
    elem = elements.Beam2(transform, con)
    return elem


# Initialize pytacs
FEAAssembler.initialize(elemCallBack)

# Create static problem
staticProblem = FEAAssembler.createStaticProblem("Case1")
# Add loads
F = np.zeros(6)
F[1] = 1e2
staticProblem.addLoadToNodes(27, F, nastranOrdering=True)

# Add eval functions
staticProblem.addFunction("mass", functions.StructuralMass)
staticProblem.addFunction("ks_failure", functions.KSFailure, ksWeight=100.0)

#### Finish the rest!
