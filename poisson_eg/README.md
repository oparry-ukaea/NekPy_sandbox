# Poisson solver example
Example of running a Poisson solver via NekPy, based on library\Demos\Python\MultiRegions\Helmholtz2D.py in the main Nektar repo.

---

The `Helmsolve` function solves the (2D, in this case) Helmholtz equation:
$$
\nabla^2u(\boldsymbol{x})-\lambda u(\boldsymbol{x}) = f(\boldsymbol{x})    \rm{~~~~\{Equation~1\}}
$$


Applying a $C^0=$ continuous Galerkin discretisation, this equation leads to the following linear system:
$$
\left(\boldsymbol{L}+\lambda\boldsymbol{M}\right)\boldsymbol{\hat{u}}_g=\boldsymbol{\hat{f}}    \rm{~~~~\{Equation~2\}}
$$

where $\boldsymbol{L}$ and $\boldsymbol{M}$ are the Laplacian and mass matrices respectively.

In `poisson.py`, we set $\lambda=0$, supply the source function, $f(x)$, then compute the corresponding field, $u(x)$.
