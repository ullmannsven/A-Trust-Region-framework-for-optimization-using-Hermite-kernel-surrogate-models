import numpy as np
import pyrol
from pyrol.vectors.NumPyVector import NumPyVector
from pymor.tools.random import new_rng


def fom_objective_functional(model, mu):
    """Evaluates the FOM at the given parameter |mu|."""
    if model.dim == 1:
        mu = np.atleast_2d(mu)
        value_FOM = (-1) * np.exp(-mu[0, 0]**2) + 3 * np.exp(-0.001 * mu[0, 0]**2)

    elif model.dim == 2:
        value_FOM = model.fom.output(mu)[0, 0]

    elif model.dim == 9:
        value_FOM = model.compute_objective(mu)

    elif model.dim == 12:
        mu_mor = model.fom.parameters.parse(mu)
        value_FOM = model.fom.output_functional_hat(mu_mor)

    else:
        raise NotImplementedError

    return float(value_FOM)


def fom_gradient_of_functional(model, mu):
    """Evaluates the gradient of the FOM at the given parameter |mu|."""
    if model.dim == 1:
        mu = np.atleast_2d(mu)
        value_FOM_grad = (
            2 * mu[0, 0] * np.exp(-mu[0, 0]**2)
            - ((3 * mu[0, 0] * np.exp(-0.001 * mu[0, 0]**2)) / 500)
        )

    elif model.dim == 2:
        mu = np.atleast_2d(mu)
        value_FOM_grad = model.fom.output_d_mu(model.fom.parameters.parse(mu)).to_numpy()

    elif model.dim == 9:
        mu = np.atleast_2d(mu)
        value_FOM_grad = model.compute_gradient(mu)

    elif model.dim == 12:
        mu_mor = model.fom.parameters.parse(mu)
        value_FOM_grad = model.fom.output_functional_hat_gradient(mu_mor)

    else:
        raise NotImplementedError

    return np.atleast_2d(value_FOM_grad).reshape(1, -1)[0, :]


class FOMROLObjective(pyrol.Objective):
    """
    Wraps the FOM objective for pyROL and tracks evaluation counts in the
    shared `data` dictionary, matching the scipy setup's bookkeeping.
    """

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def value(self, x, tol):
        self.data['counter'] += 1
        return float(fom_objective_functional(self.model, x.array))

    def gradient(self, g, x, tol):
        self.data['jac_counter'] += 1
        g.array[:] = fom_gradient_of_functional(self.model, x.array).ravel()


def prepare_data(amount_of_iters, dim):
    """Creates a dictionary to store optimization results."""
    data = {
        'J_error': np.zeros((1, 1)),
        'foc': np.zeros((1, 1)),
        'J_min': np.zeros((1, amount_of_iters)),
        'mu': np.zeros((amount_of_iters, dim)),
        'counter': 0,
        'jac_counter': 0,
    }
    return data


def _get_bounds(model):
    """Returns (lo, hi) numpy arrays of length model.dim."""
    dim = model.dim

    if dim == 1:
        lo = np.array([-2.0])
        hi = np.array([2.0])

    elif dim == 2:
        lo = np.array([0.5, 0.5])
        hi = np.array([np.pi, np.pi])

    elif dim == 9:
        lo = np.broadcast_to(model.parameter_space[0], (dim,)).astype(float).copy()
        hi = np.broadcast_to(model.parameter_space[1], (dim,)).astype(float).copy()

    elif dim == 12:
        ranges_door = (0.05, 0.2)
        ranges_heater = (0, 100)
        ranges_wall = (0.025, 0.1)
        per_dim = (
            ranges_door, ranges_door,
            ranges_heater, ranges_heater, ranges_heater, ranges_heater,
            ranges_heater, ranges_heater, ranges_heater,
            ranges_wall, ranges_wall, ranges_wall,
        )
        lo = np.array([r[0] for r in per_dim], dtype=float)
        hi = np.array([r[1] for r in per_dim], dtype=float)

    else:
        raise NotImplementedError

    return lo, hi


# ---------------------------------------------------------------------------
# Single-run optimize via pyROL
# ---------------------------------------------------------------------------

def optimize(data, mu, model, gtol=None, max_iter=200):
    """
    Runs one pyROL optimization for the given model starting from `mu`.

    Returns
    -------
    result : dict with keys 'x' and 'fun', mirroring scipy's OptimizeResult
             enough for the rest of the code to work.
    """
    dim = model.dim
    x = NumPyVector(np.asarray(mu, dtype=float))
    obj = FOMROLObjective(model, data)

    lo_arr, hi_arr = _get_bounds(model)
    lo = NumPyVector(lo_arr)
    hi = NumPyVector(hi_arr)
    bnd = pyrol.Bounds(lo, hi, True)

    if gtol is None:
        if dim == 1:
            gtol = 1e-7
        elif dim == 2:
            gtol = 1e-4
        elif dim == 9:
            gtol = 2.5e-4
        elif dim == 12:
            gtol = 5e-4
        else:
            raise NotImplementedError

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

    status = params.sublist("Status Test")
    status.set("Gradient Tolerance", gtol)
    status.set("Step Tolerance", 1e-12)
    status.set("Iteration Limit", max_iter)

    algo = pyrol.pyrol.ROL.TypeB.LinMoreAlgorithm_double_t(params)

    g = x.clone()
    g.zero()
    algo.run(x, obj, bnd)

    # --- final objective value (also bumps the counter) --------------------
    fun = float(fom_objective_functional(model, x.array))
    data['counter'] += 1

    return {'x': x.array.copy(), 'fun': fun}


def optimize_all_iters(amount_of_iters, model):
    """Repeats the optimization for several starting parameters."""
    dim = model.dim
    data = prepare_data(amount_of_iters, dim)

    for i in range(amount_of_iters):

        if dim == 1:
            np.random.seed(i)
            mu_k = np.random.uniform(-2, 2, size=1)

        elif dim == 2:
            np.random.seed(i)
            mu_k = np.random.uniform(0.5, np.pi, size=2)

        elif dim == 9:
            np.random.seed(i)
            mu_k = np.random.uniform(model.parameter_space[0], model.parameter_space[1], size=dim)
        
        elif dim == 12:
            with new_rng(i):
                mu_k = model.parameter_space.sample_randomly(1)[0].to_numpy()

        else:
            raise NotImplementedError

        result = optimize(data, mu_k, model)
        data['J_min'][0, i] = result['fun']
        data['mu'][i, :] = result['x']

        # error metrics matching the scipy setup
        if dim == 1:
            data['J_error'][0, 0] += abs((result['fun'] - 2) / 2)

        elif dim == 2:
            data['J_error'][0, 0] += abs((result['fun'] - 2.3917078761) / 2.3917078761)

        elif dim == 9:
            data['J_error'][0, 0] += abs(result['fun'] - 3.22918e-06)

        elif dim == 12:
            data['J_error'][0, 0] += abs((result['fun'] - 5.813965062384796) / 5.813965062384796)

        else:
            raise NotImplementedError

    return data


def report(data, amount_of_iters):
    """Reports optimization results."""
    print('\n succeeded!')
    print(f'  mu_min:    {data["mu"][-1, :]}')
    print(f'  J(mu_min): {data["J_min"][0, -1]}')
    print(f'  avg. FOM evals: {data["counter"] / amount_of_iters}')
    print(f'  avg. grad evals: {data["jac_counter"] / amount_of_iters}')
    print(f'  avg. error in J: {data["J_error"] / amount_of_iters}')
    print('')