"""Adapted from Nektar demo at
   https://gitlab.nektar.info/nektar/nektar/-/tree/master/library/Demos/Python/MultiRegions/Helmholtz2D.py
"""

from NekPy.LibUtilities import SessionReader, ReduceOperator
from NekPy.StdRegions import ConstFactorMap, ConstFactorType, VarCoeffMap, VarCoeffType
from NekPy.SpatialDomains import MeshGraph
from NekPy.MultiRegions import ContField

from matplotlib import pyplot as plt
import numpy as np
import os.path
import sys

# Make MPI communicator global to simplify output
comm = None

#======================================== Helper functions ========================================
def get_rank():
    """ Get MPI rank
        If communicator hasn't been initialised yet, assume rank 0
    """
    return 0 if comm is None else comm.GetRank()

def print_master(str):
    """Print on rank 0 only"""
    if get_rank() == 0:
        print(str)

def plot_domain(x, y):
    fout = os.path.join(os.path.dirname(__file__),f"rank{get_rank()}_domain.pdf") 
    plt.scatter(x,y)
    plt.savefig(fout)
#==================================================================================================

# Default input files
default_inputs_dir   = os.path.join(os.path.dirname(__file__),"inputs")
default_session_file = os.path.join(default_inputs_dir,"poisson_config.xml")

if len(sys.argv) == 2:
    args = sys.argv
elif len(sys.argv) == 1:
    args = [sys.argv[0], default_session_file]
else:
    print_master("Usage1: python poisson.py a_session_file.xml")
    print_master(f"Usage2: python poisson.py    (Uses {default_session_file})")
    exit(1)

# Init session
session = SessionReader.CreateInstance(args)
# Get MPI communicator (and set globally)
comm = session.GetComm()
# Report session file
print_master(f"Using session file at {args[1]}")

# Init mesh
graph = MeshGraph.Read(session)

# Create ContField2D
exp = ContField(session, graph, session.GetVariable(0))

# Construct factor map, using lambda from session file.
lamb = session.GetParameter("Lambda")
factors = ConstFactorMap()
factors[ConstFactorType.FactorLambda] = lamb

# Test use of variable coefficients.
coeffs = VarCoeffMap()
coeffs[VarCoeffType.VarCoeffD00] = np.ones(exp.GetNpoints())
coeffs[VarCoeffType.VarCoeffD11] = np.ones(exp.GetNpoints())

# Construct right hand side forcing term.
x, y = exp.GetCoords()
# Uncomment to generate 2D scatters
#plot_domain(x,y)
sol = np.sin(np.pi * x) * np.sin(np.pi * y)
fx = -(lamb + 2*np.pi*np.pi) * sol

# Solve Helmholtz equation.
helm_sol = exp.BwdTrans(exp.HelmSolve(fx, factors, coeffs))
L2_error = exp.L2(helm_sol, sol)
Linf_error = exp.Linf(helm_sol, sol)
Linf_error_comm = comm.AllReduce(np.max(np.abs(helm_sol - sol)), ReduceOperator.ReduceMax)

# Print out some stats for debugging.
print_master("L 2 error (variable nek)     : %.6e" % L2_error)
print_master("L inf error (variable nek)   : %.6e" % Linf_error)
print_master("L inf error (variable nekpy) : %.6e" % Linf_error_comm)

# Clean up!
session.Finalise()
