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

# Default input files
default_inputs_dir   = os.path.join(os.path.dirname(__file__),"inputs")
default_session_file = os.path.join(default_inputs_dir,"poisson_config.xml")

# Construct args to pass to session reader
if len(sys.argv) == 2:
    args = sys.argv
elif len(sys.argv) == 1:
    args = [sys.argv[0], default_session_file]
else:
    print_master("Usage1: python poisson.py a_session_file.xml")
    print_master(f"Usage2: python poisson.py    (Uses {default_session_file})")
    exit(1)

# Init session (including MPI)
session = SessionReader.CreateInstance(args)
# Get MPI communicator (and set globally)
comm = session.GetComm()

# Report session filepath
print_master(f"Using session file at {args[1]}")
print_master(f"Session file was {session.GetSessionName()}.xml")

# Init mesh
graph = MeshGraph.Read(session)

# Create ContField2D
exp = ContField(session, graph, session.GetVariable(0))

# Construct factor map, setting lambda = 0 (Helmholtz=>Poisson)
factors = ConstFactorMap()
factors[ConstFactorType.FactorLambda] = 0.0

# Test use of variable coefficients.
coeffs = VarCoeffMap()
coeffs[VarCoeffType.VarCoeffD00] = np.ones(exp.GetNpoints())
coeffs[VarCoeffType.VarCoeffD11] = np.ones(exp.GetNpoints())

# Construct right hand side forcing term.
x, y = exp.GetCoords()

# Uncomment to generate 2D scatters of the domain (one per MPI rank)
#write_pdf_for_rank(x,y,fname_suffix="_domain")

exact_sol = np.sin(np.pi * x) * np.sin(np.pi * y)
fx = -2 * np.pi*np.pi * exact_sol

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

# Uncomment to generate 2D scatters of (re-scaled) solution
#write_pdf_for_rank(x,y,calc_sol,fname_suffix="_sol",scale_z=True)

# Clean up MPI
session.Finalise()