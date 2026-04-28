"""
pyROL driver for the semilinear parameter identification problem.

Optimization variable: w in R^9 (weights of Gaussian ansatz for sigma)
Objective:    J(w) = 0.5 ||u(w) - u_ref||_L^2^2 + 0.5 * alpha * w^T G w
PDE:          -Delta u + sigma(w) u^3 = f
Inner product on R^9: <w, w'>_G = w^T G w'  (H^1 semi-inner-product
                                             induced by the Gaussian basis)
"""

import numpy as np
import pyrol
from pyrol.vectors.NumPyVector import NumPyVector


class NonlinearROLObjective(pyrol.Objective):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fomCounter = 0
        self.gradCounter = 0

    def value(self, x, tol):
        self.fomCounter += 1
        return float(self.model.compute_objective(x.array))

    def gradient(self, g, x, tol):
        self.gradCounter += 1
        g.array[:] = self.model.compute_gradient(x.array).ravel()


def run_rol(model, w0=None, verify=False):
    """
    model : NonlinearModel with u_ref already set.
    w0    : initial guess.
    """
    x = NumPyVector(np.asarray(w0, dtype=float))
    obj = NonlinearROLObjective(model)

    #TODO this only works if this is not a complicated space atm
    bounds = model.parameter_space
    lo_arr = np.broadcast_to(bounds[0], (model.dim,)).astype(float).copy()
    hi_arr = np.broadcast_to(bounds[1], (model.dim,)).astype(float).copy()

    lo = NumPyVector(lo_arr)
    hi = NumPyVector(hi_arr)
    bnd = pyrol.Bounds(lo, hi, True)

    # ---- Solver parameters ------------------------------------
    params = pyrol.ParameterList()
    params.set("General", pyrol.ParameterList())
    params.set("Step", pyrol.ParameterList())
    params.set("Status Test", pyrol.ParameterList())
    
    general = params.sublist("General")
    general.set("Output Level", 1)
    general.set("Projected Gradient Criticality Measure", True)

    general.set("Secant", pyrol.ParameterList())
    secant = general.sublist("Secant")
    secant.set("Type", "Limited-Memory BFGS")
    secant.set("Use as Hessian", True)

    #step = params.sublist("Step")
    #step.set("Type", "Trust Region")
    #step.set("Trust Region", pyrol.ParameterList())

    #tr = step.sublist("Trust Region")
    #tr.set("Initial Radius", 1.0)
    #tr.set("Subproblem Solver", "Lin-More")
    
    #tr.set("Lin-More", pyrol.ParameterList())
    #linmore = tr.sublist("Lin-More")
    #linmore.set("Maximum Number of Minor Iterations", 100)
    
    # Step: line search
    # step = params.sublist("Step")
    # step.set("Type", "Line Search")
    # step.set("Line Search", pyrol.ParameterList())
    # ls = step.sublist("Line Search")

    # # Descent direction: use L-BFGS quasi-Newton
    # ls.set("Descent Method", pyrol.ParameterList())
    # descent = ls.sublist("Descent Method")
    # descent.set("Type", "Quasi-Newton Method")

    # # Line search algorithm: cubic interpolation with strong Wolfe (scipy's default style)
    # ls.set("Line-Search Method", pyrol.ParameterList())
    # lsm = ls.sublist("Line-Search Method")
    # lsm.set("Type", "Cubic Interpolation")

    # # Curvature condition (Wolfe parameters)
    # ls.set("Curvature Condition", pyrol.ParameterList())
    # cc = ls.sublist("Curvature Condition")
    # cc.set("Type", "Strong Wolfe Conditions")
    
    status = params.sublist("Status Test")
    status.set("Gradient Tolerance", 2.5e-4)
    status.set("Step Tolerance", 1e-12)
    status.set("Iteration Limit", 200)

    print(params)

    # Construct TypeB Lin-More algorithm directly
    algo = pyrol.pyrol.ROL.TypeB.LinMoreAlgorithm_double_t(params)

    # Allocate dual workspace and run
    g = x.clone()
    g.zero()

    #print("\n=== Running TypeB Lin-More algorithm ===")
    algo.run(x, obj, bnd)

    # problem = pyrol.Problem(obj, x)
    # problem.addBoundConstraint(bnd)
    # problem.finalize(False, True)

    # solver = pyrol.Solver(problem, params)
    # solver.solve()

    print("so oft fom", obj.fomCounter, obj.gradCounter)

    return x.array.copy()