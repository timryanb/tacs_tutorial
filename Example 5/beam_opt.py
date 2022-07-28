import os
import numpy as np
from tacs import pyTACS, constitutive, elements, functions
from pyoptsparse import OPT, Optimization

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

# Create transient prob
transientProb = FEAAssembler.createTransientProblem("Case2", 0.0, 0.5, 50)
timeSteps = transientProb.getTimeSteps()
T = np.zeros(3)
for step_i, t in enumerate(timeSteps):
    T[1] = 3.5e2 * np.sin(2 * np.pi * 12.7 * t)
    compIDs = FEAAssembler.selectCompIDs()
    transientProb.addTractionToComponents(step_i, compIDs, T)
allProbs = [staticProblem, transientProb]

# Add eval functions
for problem in allProbs:
    problem.addFunction("mass", functions.StructuralMass)
    problem.addFunction("ks_failure", functions.KSFailure, ksWeight=100.0)


def objfunc(xdict):
    funcs = {}
    for problem in allProbs:
        problem.setDesignVars(xdict)
        problem.solve()
        problem.evalFunctions(funcs)
    staticProblem.writeSolution()
    print(funcs)
    return funcs


def sens(xdict, funcs):
    funcsSens = {}
    for problem in allProbs:
        problem.setDesignVars(xdict)
        problem.solve()
        problem.evalFunctionsSens(funcsSens)
    for func in funcsSens:
        funcsSens[func].pop("Xpts")
    return funcsSens


# Optimization Object
optProb = Optimization("Beam Optimization Problem", objfunc)

# Design Variables
ndvs = FEAAssembler.getNumDesignVars()
optProb.addVarGroup("struct", ndvs, lower=1e-3, upper=8e-3, value=8e-3, scale=1e3)

# Constraints
optProb.addConGroup("Case1_ks_failure", 1, upper=1.0)
optProb.addConGroup("Case2_ks_failure", 1, upper=1.0, scale=0.1)

# Objective
optProb.addObj("Case1_mass")

# Check optimization problem:
print(optProb)

# Optimizer
opt = OPT("SNOPT", options={"Major step limit": 1e-1,
                            "Penalty parameter": 1.0})

# Solution
sol = opt(optProb, sens=sens)

# Check Solution
print(sol)
transientProb.writeSolution()
