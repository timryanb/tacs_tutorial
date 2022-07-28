# Python imports
import os
from tacs import pyTACS

# Instantiate FEAAssembler
bdfFile = os.path.join(os.path.dirname(__file__), '../input files/run_uCRM-9_static_cruise_coarse.bdf')
FEAAssembler = pyTACS(bdfFile)

# Set up elements and TACS assembler
FEAAssembler.initialize()

# Create TACS problems from cases in BDF
allProblems = FEAAssembler.createTACSProbsFromBDF().values()

# Loop through each problem and solve
for problem in allProblems:
    problem.solve()
    problem.writeSolution()
