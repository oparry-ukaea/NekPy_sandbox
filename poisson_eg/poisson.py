"""Adapted from Nektar demo at
   https://gitlab.nektar.info/nektar/nektar/-/tree/master/library/Demos/Python/MultiRegions/Helmholtz2D.py

   N.B. The rhs (source) term returned by choose_sol_and_rhs() has to match the boundary conditions specified in the session file,
   so this script uses one parameter, 'src_type' to choose both 
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
def choose_sol_and_rhs(src_type,session):
    if src_type == "exp": 
        a = session.GetParameter("a")
        b = session.GetParameter("b")
        exact_sol = np.exp(-(a*x**2 + b*y**2) / 2.0)
        rhs = ((a*x)**2 - a + (b*y)**2 - b) * exact_sol
    elif src_type == "trig":
        exact_sol = np.sin(np.pi * x) * np.sin(np.pi * y)
        rhs = -2 * np.pi*np.pi * exact_sol
    else:
        raise ValueError(f"No solution, rhs defined for src type {src_type}")
    return exact_sol,rhs

def get_rank():
    """ Get MPI rank.
        If communicator hasn't been initialised yet, raise RuntimeError
    """
    if comm is None:
        raise RuntimeError("MPI communicator not set.")
    return comm.GetRank()

def print_master(str):
    """Print to stdout on rank 0 only"""
    if get_rank() == 0:
        print(str)

def write_pdf_for_rank(x, y, z=None, fname_suffix="",scale_z=False):
    """Plot y vs x scatter, coloured by z if supplied
       Save to 'rank[rank][fname_suffix].pdf'
    """
    fout = os.path.join(os.path.dirname(__file__),f"rank{get_rank()}{fname_suffix}.pdf") 
    plt.clf()
    if z is None:
        plt.scatter(x,y)
    else:
        # Use z for color, scaling if requested
        color = (z-np.min(z))/(np.max(z)-np.min(z)) if scale_z else z
        plt.scatter(x,y,c=color)
        plt.colorbar()
    plt.savefig(fout)
#==================================================================================================

# Define possible src types and corresponding session file paths
src_types = ["exp", "trig"]
inputs_dir = os.path.join(os.path.dirname(__file__),"inputs")
session_file_paths = {src_type: os.path.join(inputs_dir,f"poisson_{src_type}.xml") for src_type in src_types}

# Parse args
src_type = "not_set"
if len(sys.argv) == 2:
    src_type = sys.argv[1]
elif len(sys.argv) == 1:
    src_type = src_types[0]
if not src_type in src_types:
    print("Usage: python poisson.py <src_type>")
    print(f" where src_type is one of [{','.join(src_types)}] and defaults to {src_types[0]} ")
    exit(1)

# Construct args to pass to session reader
args = [sys.argv[0], session_file_paths[src_type]]

# Init session (including MPI)
session = SessionReader.CreateInstance(args)
# Get MPI communicator (and set globally)
comm = session.GetComm()

# Report session filepath
print_master(f"Session file was {session.GetSessionName()}.xml")

# Init mesh
graph = MeshGraph.Read(session)

# Create ContField2D
exp = ContField(session, graph, session.GetVariable(0))

# Construct factor map, setting lambda = 0 (Helmholtz=>Poisson)
factors = ConstFactorMap()
factors[ConstFactorType.FactorLambda] = 0.0

# Set coefficients such that the mass matrix ('M' in Equation 2 of the README) is the identity matrix
# N.B. In our case M's prefactor, lambda, is zero, so this is superfluous; it's left in for illustrative purposes only. 
coeffs = VarCoeffMap()
coeffs[VarCoeffType.VarCoeffD00] = np.ones(exp.GetNpoints())
coeffs[VarCoeffType.VarCoeffD11] = np.ones(exp.GetNpoints())

# Construct right hand side forcing term.
x, y = exp.GetCoords()

# Uncomment to generate 2D scatters of the domain (one per MPI rank)
#write_pdf_for_rank(x,y,fname_suffix="_domain")

# Choose exact solution and right-hand-side of Poisson eqn
exact_sol,rhs = choose_sol_and_rhs(src_type,session)

# Solve and transform back from expansion space to physical space
calc_sol = exp.BwdTrans(exp.HelmSolve(rhs, factors, coeffs))

# Output L2 error
L2_error_local = exp.L2(calc_sol, exact_sol)
L2_error = comm.AllReduce(L2_error_local, ReduceOperator.ReduceSum) / comm.GetSize()
print_master("L 2 error : %.6e" % L2_error)

# Output max error (Chebyshev distance)
Linf_error_local = exp.Linf(calc_sol, exact_sol)
Linf_error = comm.AllReduce(Linf_error_local, ReduceOperator.ReduceMax)
print_master("L inf error : %.6e" % Linf_error)

# Uncomment to generate 2D scatters of calculated, exact solutions (one per MPI rank)
#write_pdf_for_rank(x,y,calc_sol,fname_suffix="_calc_sol")
#write_pdf_for_rank(x,y,exact_sol,fname_suffix="_exact_sol")

# Clean up MPI
session.Finalise()