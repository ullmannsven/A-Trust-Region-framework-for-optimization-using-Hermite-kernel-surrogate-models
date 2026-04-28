import numpy                as np
import abc
from pymor.basic import *
from pymor.core.logger import set_log_levels, getLogger
from functions.models.pyMORAuxData import twelve_dim_discretizer  
from functions.models.pyMORAuxData import problems  
from pathlib import Path
from dolfinx import io


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self.fomCounter      = 0
        self.dim             = None
        self.pyMOR           = True
        self.parameter_space = None
        self.RKHS_explicit   = False

    @abc.abstractmethod
    def getFuncAndGradient(self,mu): 
        pass

class Gaussian1D(Model):
    def __init__(self):
        super().__init__()
        self.dim = 1
        self.pyMOR = False
        self.RKHS_explicit = True
        self.parameter_space = [-2, 2]

    def getFuncAndGradient(self, mu):
        mu = np.atleast_2d(mu)
        
        value = (-1) * np.exp(-mu[0,0]**2) + 3 * np.exp(-0.001 * mu[0,0]**2)
        der   =  2*mu[0,0]* np.exp(-mu[0,0]**2) - ((3 * mu[0,0] * np.exp(- 0.001 * mu[0,0]**2)) / 500)
        
        self.fomCounter += 1
        return value, np.atleast_2d(der).reshape(1,)

    def compute_RKHS_norm(self, mu, kernel=None):
        mu = np.atleast_2d(mu).reshape(-1,1)

        if kernel is None:
            kernel_width = mu[-1, 0]
        else: 
            kernel_width = kernel.gamma

        if kernel_width > 1/np.sqrt(2):
            
            enum  = kernel_width * np.sqrt(np.sqrt(2000*kernel_width**2 - 1) * (np.sqrt(1001*kernel_width**2 - 1) - 33541 * (2**(5/2)) * np.sqrt(2*kernel_width**2 - 1)) + 8999989448 * np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1))
            denom = np.sqrt(np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1) * np.sqrt(2000*kernel_width**2 - 1) )

            return enum / denom

        else:
            print("Warning: kernel width below limit, choosing lowest as possible")
            kernel_width = 1/np.sqrt(2) + 1e-8

            enum  = kernel_width * np.sqrt(np.sqrt(2000*kernel_width**2 - 1) * (np.sqrt(1001*kernel_width**2 - 1) - 33541 * (2**(5/2)) * np.sqrt(2*kernel_width**2 - 1)) + 8999989448 * np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1))
            denom = np.sqrt(np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1) * np.sqrt(2000*kernel_width**2 - 1) )

            return enum / denom

class twoDStuff(Model): 
    def __init__(self):
        super().__init__()
        self.dim = 2

        problem = problems.linear_problem()
        mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
        self.fom, _ = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)

        self.parameter_space = self.fom.parameters.space(0.5, np.pi)

    def getFuncAndGradient(self, mu):
        self.fomCounter += 1
        value_FOM = self.fom.output(mu)[0,0]
        gradient_FOM = self.fom.output_d_mu(self.fom.parameters.parse(mu)).to_numpy().reshape(2,)
        
        return value_FOM, gradient_FOM

class buildingFloor(Model):
    def __init__(self):
        super().__init__()
        self.dim = 12

        set_log_levels({'pymor': 'ERROR',
                        'distributed_adaptive_discretizations': 'DEBUG',
                        'notebook': 'INFO'})

        data_path = 'pyMORAuxData/EXC_data'

        # domain of interest
        bounding_box = [[0,0],[2,1]]
        domain_of_interest = BitmapFunction.from_file('{}/Domain_of_interest.png'.format(data_path), bounding_box=bounding_box, range=[1,0])

        parametric_quantities = {'walls': [1,4,9], 'windows': [], 'doors': [6,7], 'heaters': [1,3,5,6,7,8,9]}
        inactive_quantities = {'removed_walls': [], 'open_windows': [], 'open_doors': [1,2,3,4,5,10], 'active_heaters': []}
        summed_quantities = {'walls': [[1,2,3,7,8],[4,5,6]], 'windows': [], 'doors': [], 'heaters': [[1,2],[3,4],[9,10,11,12]]}

        coefficient_expressions = None

        parameters_in_q = True
        input_dict      = problems.set_input_dict(parametric_quantities, inactive_quantities, coefficient_expressions, summed_quantities, parameters_in_q,
                                        ac=0.5, owc=[0.025,0.1], iwc= [0.025,0.1], idc=[0.05,0.2], wc=[0.0005], ht=[0,100],
                                            owc_c=0.001,     iwc_c= 0.025,     idc_c=0.01,     wc_c=0.05,   ht_c=80)

        parameter_scaling = False
        u_out             = 5

        problem, parameter_scales = problems.EXC_problem(input_dict, summed_quantities, outside_temperature=u_out,
                                                data_path = data_path,parameters_in_q=parameters_in_q,
                                                parameter_scaling=parameter_scaling,
                                                coefficient_expressions=coefficient_expressions)

        u_d     = 18
        mu_d    = None
        sigma_d = 100 
        weights = {'walls': 0.1, 'doors': 1, 'heaters': [0.002,0.002,0.0005,0.0005,0.0005,0.0005,0.004], 'windows': 1, 'state': sigma_d}

        diameter              = np.sqrt(2)/200
        self.fom, data, mu_bar = twelve_dim_discretizer.discretize_quadratic_pdeopt_stationary_cg(problem, diameter, weights, parameter_scales,
                                                                domain_of_interest, desired_temperature=u_d,
                                                                mu_for_u_d=mu_d, mu_for_tikhonov=mu_d,
                                                                parameters_in_q=parameters_in_q, product='fixed_energy')

        self.parameter_space = problem.parameter_space
        print("hier", self.parameter_space.parameters.items())
        

    def getFuncAndGradient(self,mu):  
        self.fomCounter += 1
        mu_mor = self.fom.parameters.parse(mu)
        return self.fom.output_functional_hat(mu_mor), self.fom.output_functional_hat_gradient(mu_mor)


from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import ufl



class NonlinearModel(Model):
    def __init__(self, nx=128, ny=128, alpha=1e-8, centers=None, widths=None):
        super().__init__()
        self.dim = 9
        self.alpha = alpha
        self.u_ref = None
        self._last_solved_w = None
        self.pyMOR = False
        self.parameter_space = [0, 10]

        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))

        self.u = fem.Function(self.V)
        self.u.name = "u"
        self.v = ufl.TestFunction(self.V)
        self.x = ufl.SpatialCoordinate(self.domain)

       
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            self.domain,
            fdim,
            lambda x: np.full(x.shape[1], True, dtype=bool),
        )
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        self.bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)


        if centers is None:
            # 4x4 grid on [0, 1]^2, evenly spaced in the interior
            grid_1d = np.linspace(1/6, 5/6, 3)
            centers = [(cx, cy) for cy in grid_1d for cx in grid_1d]

        #if widths is None:
        #    widths = [0.12]*9

        self.centers = centers
        self.widths = widths

        def gaussian(x, cx, cy):
            return ufl.exp(-((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / 0.03)

        self.g = [gaussian(self.x, cx, cy) for (cx, cy) in self.centers]

        # placeholder, overwritten per solve
        self.w = fem.Constant(self.domain, PETSc.ScalarType(tuple([0.0] * self.dim)))
        self.sigma_expr = sum(self.w[i] * self.g[i] for i in range(self.dim))

        # Optional: FE function for visualization/output of sigma
        self.sigma_fun = fem.Function(self.V)
        self.sigma_fun.name = "sigma"

        self.f = 300

        # weak form
        self.F = (
            ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
            + self.sigma_expr * self.u**3 * self.v * ufl.dx
            - self.f * self.v * ufl.dx
        )

        self.J = ufl.derivative(self.F, self.u)

        G = np.zeros((self.dim, self.dim), dtype=float)
        for i in range(self.dim):
            for j in range(self.dim):
                aij = fem.form(ufl.inner(ufl.grad(self.g[i]), ufl.grad(self.g[j])) * ufl.dx)
                G[i, j] = fem.assemble_scalar(aij)
        self.G = 0.5 * (G + G.T)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        self.problem = NonlinearProblem(
            self.F,
            self.u,
            bcs=[self.bc],
            J=self.J,
            petsc_options_prefix="nonlinear_model_",
            petsc_options=petsc_options,
        )



    def solve(self, w, return_copy=False):
        w = np.atleast_2d(np.asarray(w, dtype=np.float64)).reshape(-1)
        
    
        # update parameter
        self.w.value = w.astype(PETSc.ScalarType)

        # solve nonlinear PDE
        self.problem.solve()

        # synchronize ghost values
        self.u.x.scatter_forward()

        # update sigma function for output/visualization
        self.sigma_fun.interpolate(fem.Expression(self.sigma_expr, self.V.element.interpolation_points))
        self.sigma_fun.x.scatter_forward()

        self._last_solved_w = w.copy()

        if return_copy:
            u_copy = fem.Function(self.V)
            u_copy.name = "u"
            u_copy.x.array[:] = self.u.x.array
            u_copy.x.scatter_forward()
            return u_copy

        return self.u

    def solve_adjoint(self, w, return_copy=False):
        """
        Solve the adjoint equation at parameter w. Assumes the state u(w)
        has already been computed.

        The adjoint equation is:
            a'(w_test, p) = -(u - u_ref, w_test)    for all w_test in V
        where
            a'(w_test, p) = (grad w_test, grad p) + (3 sigma u^2 w_test, p)
        """
        if self.u_ref is None:
            raise RuntimeError("Reference solution u_ref has not been set.")

        if self._last_solved_w is None or not np.array_equal(self._last_solved_w, w):
            self.solve(w)

        w = np.atleast_2d(np.asarray(w, dtype=np.float64)).reshape(-1)
        self.w.value = w.astype(PETSc.ScalarType)

        # adjoint trial/test
        p = fem.Function(self.V)
        p.name = "p"
        p_trial = ufl.TrialFunction(self.V)
        q = ufl.TestFunction(self.V)

        # bilinear form: linearized forward operator at current u
        # a'(p, q) = (grad p, grad q) + 3 sigma u^2 p q
        a_adj = (ufl.inner(ufl.grad(p_trial), ufl.grad(q)) * ufl.dx + 3.0 * self.sigma_expr * self.u**2 * p_trial * q * ufl.dx)

        # rhs: -(u - u_ref, q)
        L_adj = -(self.u - self.u_ref) * q * ufl.dx

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        problem = LinearProblem(
            a_adj,
            L_adj,
            bcs=[self.bc],
            u=p,
            petsc_options_prefix="adjoint_",
            petsc_options=petsc_options,
        )
        problem.solve()
        p.x.scatter_forward()

        if return_copy:
            p_copy = fem.Function(self.V)
            p_copy.name = "p"
            p_copy.x.array[:] = p.x.array
            p_copy.x.scatter_forward()
            return p_copy

        self.p = p
        return p

    def set_reference_solution(self, u_ref):
        self.u_ref = fem.Function(self.V)
        self.u_ref.name = "u_ref"
        self.u_ref.x.array[:] = u_ref.x.array
        self.u_ref.x.scatter_forward()

    def compute_reference_from_weights(self, w_ref):
        u_ref = self.solve(w_ref, return_copy=True)
        self.set_reference_solution(u_ref)

    def compute_objective(self, w):
        if self.u_ref is None:
            raise RuntimeError("Reference solution u_ref has not been set.")

        w = np.atleast_2d(np.asarray(w, dtype=np.float64)).reshape(-1,1)
        self.solve(w)

        misfit_form = fem.form(0.5 * (self.u - self.u_ref) ** 2 * ufl.dx)
        misfit_local = fem.assemble_scalar(misfit_form)
        misfit = self.domain.comm.allreduce(misfit_local, op=MPI.SUM)

        reg = 0.5 * self.alpha * (w.T @ self.G @ w)[0, 0]
        objective = misfit + reg

        return objective

    def compute_gradient(self, w):
        w = np.atleast_2d(np.asarray(w, dtype=np.float64)).reshape(-1,1)
        p = self.solve_adjoint(w)

        # d J / d w_i = alpha (G w)_i + integral of g_i * u^3 * p
        gradient = np.zeros((1, self.dim), dtype=np.float64)
        gradient += self.alpha * (w.T @ self.G)

        for i in range(self.dim):
            form_i = fem.form(self.g[i] * self.u**3 * p * ufl.dx)
            val_local = fem.assemble_scalar(form_i)
            gradient[0, i] += self.domain.comm.allreduce(val_local, op=MPI.SUM)

        return gradient 

    def getFuncAndGradient(self, w):
        self.fomCounter += 1
        objective = self.compute_objective(w)
        gradient = self.compute_gradient(w)

        return objective, gradient

    def taylor_test(self, w, direction=None, n=6):
        """First-order Taylor: |J(w + h d) - J(w)| = O(h).
        Gradient-corrected:  |J(w + h d) - J(w) - h * g . d| = O(h^2)."""
        if direction is None:
            rng = np.random.default_rng(0)
            direction = rng.standard_normal(self.dim)
            direction /= np.linalg.norm(direction)

        J0, g0 = self.getFuncAndGradient(w)
        gd = (g0 @ direction)[0]

        print(f"{'h':>10} {'|dJ|':>14} {'|dJ - h g.d|':>16} {'rate0':>8} {'rate1':>8}")
        prev0, prev1 = None, None
        for k in range(n):
            h = 10.0 ** (-k - 1)
            Jh, _ = self.getFuncAndGradient(w + h * direction)
            e0 = abs(Jh - J0)
            e1 = abs(Jh - J0 - h * gd)
            r0 = "" if prev0 is None else f"{np.log10(prev0/e0):.2f}"
            r1 = "" if prev1 is None else f"{np.log10(prev1/e1):.2f}"
            print(f"{h:10.1e} {e0:14.4e} {e1:16.4e} {r0:>8} {r1:>8}")
            prev0, prev1 = e0, e1
        

    def save_current_solution(self, filename="reference_data/solution.xdmf"):
        with io.XDMFFile(self.domain.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.u)

    def save_reference_solution(self, filename="reference_data/reference_solution.xdmf"):
        if self.u_ref is None:
            raise RuntimeError("Reference solution u_ref has not been set.")

        with io.XDMFFile(self.domain.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.u_ref)


# class NonlinearModel(Model):
#     def __init__(self, nx=128, ny=128, alpha=1e-8, beta=0, centers=None,
#                  s_ref=None):
#         super().__init__()
#         self.n_gauss = 9
#         self.dim = 2 * self.n_gauss
#         self.alpha = alpha
#         self.beta = beta
#         self.u_ref = None
#         self._last_solved_mu = None
#         self.pyMOR = False

#         # parameter space: [w_bounds] * 9 + [s_bounds] * 9
#         self.parameter_space = {'coeff': (0, 10), 'width':(0.1, 0.2)}

#         self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
#         self.V = fem.functionspace(self.domain, ("Lagrange", 1))

#         self.u = fem.Function(self.V)
#         self.u.name = "u"
#         self.v = ufl.TestFunction(self.V)
#         self.x = ufl.SpatialCoordinate(self.domain)

#         # Dirichlet BC on whole boundary
#         fdim = self.domain.topology.dim - 1
#         boundary_facets = mesh.locate_entities_boundary(self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#         boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
#         self.bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)

#         # Gaussian centers
#         if centers is None:
#             grid_1d = np.linspace(1/6, 5/6, 3)
#             centers = [(cx, cy) for cy in grid_1d for cx in grid_1d]
#         assert len(centers) == self.n_gauss
#         self.centers = centers

#         # reference width (used as the center of the width-regularization)
#         if s_ref is None:
#             s_ref = 0.175
#         self.s_ref = float(s_ref)

#         # parameter Constants: amplitudes w and widths s
#         self.w = fem.Constant(
#             self.domain, PETSc.ScalarType(tuple([0.0] * self.n_gauss))
#         )
#         self.s = fem.Constant(
#             self.domain, PETSc.ScalarType(tuple([self.s_ref] * self.n_gauss))
#         )

#         # Gaussians, now with parametric widths via self.s
#         def gaussian(x, cx, cy, s_const):
#             # s_const is a single entry of self.s (UFL scalar)
#             return ufl.exp(
#                 -((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / (2.0 * s_const**2)
#             )

#         self.g = [gaussian(self.x, cx, cy, self.s[i]) for i, (cx, cy) in enumerate(self.centers)]

#         # convenient UFL expressions for r_i^2 = ||x - c_i||^2, used in width
#         # derivatives: d g_i / d s_i = g_i * r_i^2 / s_i^3
#         self.r2 = [(self.x[0] - cx) ** 2 + (self.x[1] - cy) ** 2 for (cx, cy) in self.centers]

#         # sigma(x; w, s) = sum_i w_i g_i(x; s_i)
#         self.sigma_expr = sum(self.w[i] * self.g[i] for i in range(self.n_gauss))

#         # visualization
#         self.sigma_fun = fem.Function(self.V)
#         self.sigma_fun.name = "sigma"

#         self.f = 100

#         # weak form
#         self.F = (
#             ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
#             + self.sigma_expr * self.u**3 * self.v * ufl.dx
#             - self.f * self.v * ufl.dx
#         )
#         self.J = ufl.derivative(self.F, self.u)

#         petsc_options = {
#             "snes_type": "newtonls",
#             "snes_linesearch_type": "bt",
#             "snes_rtol": 1e-8,
#             "snes_atol": 1e-8,
#             "snes_max_it": 50,
#             "ksp_type": "preonly",
#             "pc_type": "lu",
#         }
#         self.problem = NonlinearProblem(
#             self.F, self.u, bcs=[self.bc], J=self.J,
#             petsc_options_prefix="nonlinear_model_",
#             petsc_options=petsc_options,
#         )

#     # ---------- parameter packing utilities ----------

#     def _split(self, mu):
#         """Split a length-18 vector into (w, s)."""
#         mu = np.asarray(mu, dtype=np.float64).reshape(-1)
#         assert mu.size == self.dim, f"Expected {self.dim} parameters, got {mu.size}."
#         w = mu[:self.n_gauss]
#         s = mu[self.n_gauss:]
#         return w, s

#     def _set_params(self, mu):
#         """Push (w, s) from a flat vector into the UFL Constants."""
#         w, s = self._split(mu)
#         self.w.value = w.astype(PETSc.ScalarType)
#         self.s.value = s.astype(PETSc.ScalarType)

#     # ---------- forward and adjoint solves ----------

#     def solve(self, mu, return_copy=False):
#         mu = np.asarray(mu, dtype=np.float64).reshape(-1)
#         self._set_params(mu)
#         self.problem.solve()
#         self.u.x.scatter_forward()

#         self.sigma_fun.interpolate(
#             fem.Expression(self.sigma_expr, self.V.element.interpolation_points)
#         )
#         self.sigma_fun.x.scatter_forward()

#         self._last_solved_mu = mu.copy()

#         if return_copy:
#             u_copy = fem.Function(self.V)
#             u_copy.name = "u"
#             u_copy.x.array[:] = self.u.x.array
#             u_copy.x.scatter_forward()
#             return u_copy
#         return self.u

#     def solve_adjoint(self, mu, return_copy=False):
#         """Adjoint equation. Structure unchanged from before -- only the
#         parameter set now includes widths, which enter through self.sigma_expr."""
#         if self.u_ref is None:
#             raise RuntimeError("Reference solution u_ref has not been set.")

#         mu = np.asarray(mu, dtype=np.float64).reshape(-1)
#         if self._last_solved_mu is None or not np.array_equal(self._last_solved_mu, mu):
#             self.solve(mu)
#         else:
#             # make sure constants are current (cheap; safe)
#             self._set_params(mu)

#         p = fem.Function(self.V)
#         p.name = "p"
#         p_trial = ufl.TrialFunction(self.V)
#         q = ufl.TestFunction(self.V)

#         a_adj = (ufl.inner(ufl.grad(p_trial), ufl.grad(q)) * ufl.dx + 3.0 * self.sigma_expr * self.u**2 * p_trial * q * ufl.dx)
#         L_adj = -(self.u - self.u_ref) * q * ufl.dx

#         petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
#         problem = LinearProblem(
#             a_adj, L_adj, bcs=[self.bc], u=p,
#             petsc_options_prefix="adjoint_",
#             petsc_options=petsc_options,
#         )
#         problem.solve()
#         p.x.scatter_forward()

#         if return_copy:
#             p_copy = fem.Function(self.V)
#             p_copy.name = "p"
#             p_copy.x.array[:] = p.x.array
#             p_copy.x.scatter_forward()
#             return p_copy

#         self.p = p
#         return p

#     # ---------- reference handling (unchanged) ----------

#     def set_reference_solution(self, u_ref):
#         self.u_ref = fem.Function(self.V)
#         self.u_ref.name = "u_ref"
#         self.u_ref.x.array[:] = u_ref.x.array
#         self.u_ref.x.scatter_forward()

#     def compute_reference_from_weights(self, mu_ref):
#         u_ref = self.solve(mu_ref, return_copy=True)
#         self.set_reference_solution(u_ref)

#     # ---------- objective and gradient ----------

#     def _reg(self, mu):
#         """Regularization term and its gradient w.r.t. mu."""
#         w, s = self._split(mu)
#         # simple diagonal regularization:
#         #   (alpha/2) * ||w||^2 + (beta/2) * ||s - s_ref||^2
#         val = 0.5 * self.alpha * float(w @ w) \
#             + 0.5 * self.beta * float((s - self.s_ref) @ (s - self.s_ref))
#         grad = np.concatenate([self.alpha * w, self.beta * (s - self.s_ref)])
#         return val, grad

#     def compute_objective(self, mu):
#         if self.u_ref is None:
#             raise RuntimeError("Reference solution u_ref has not been set.")
#         mu = np.asarray(mu, dtype=np.float64).reshape(-1)
#         self.solve(mu)

#         misfit_form = fem.form(0.5 * (self.u - self.u_ref) ** 2 * ufl.dx)
#         misfit_local = fem.assemble_scalar(misfit_form)
#         misfit = self.domain.comm.allreduce(misfit_local, op=MPI.SUM)

#         reg_val, _ = self._reg(mu)
#         return misfit + reg_val

#     def compute_gradient(self, mu):
#         mu = np.asarray(mu, dtype=np.float64).reshape(-1)
#         p = self.solve_adjoint(mu)

#         # regularization contribution
#         _, reg_grad = self._reg(mu)
#         gradient = np.zeros(self.dim, dtype=np.float64)
#         gradient += reg_grad

#         # PDE-adjoint contribution: for each parameter component mu_k,
#         #   dJ/dmu_k += integral of (dsigma/dmu_k) * u^3 * p dx
#         #
#         # For amplitudes:  dsigma/dw_i = g_i(x; s_i)
#         # For widths:      dsigma/ds_i = w_i * g_i(x; s_i) * r_i^2 / s_i^3
#         for i in range(self.n_gauss):
#             # amplitude derivative
#             form_w = fem.form(self.g[i] * self.u**3 * p * ufl.dx)
#             val_w_local = fem.assemble_scalar(form_w)
#             gradient[i] += self.domain.comm.allreduce(val_w_local, op=MPI.SUM)

#             # width derivative
#             dsig_ds = self.w[i] * self.g[i] * self.r2[i] / (self.s[i]**3)
#             form_s = fem.form(dsig_ds * self.u**3 * p * ufl.dx)
#             val_s_local = fem.assemble_scalar(form_s)
#             gradient[self.n_gauss + i] += self.domain.comm.allreduce(
#                 val_s_local, op=MPI.SUM
#             )

#         return gradient

#     def getFuncAndGradient(self, mu):
#         self.fomCounter += 1
#         return self.compute_objective(mu), self.compute_gradient(mu)

#     # ---------- Taylor test (unchanged logic, updated shapes) ----------

#     def taylor_test(self, mu, direction=None, n=6):
#         if direction is None:
#             rng = np.random.default_rng(0)
#             direction = rng.standard_normal(self.dim)
#             direction /= np.linalg.norm(direction)

#         J0, g0 = self.getFuncAndGradient(mu)
#         gd = float(g0 @ direction)

#         print(f"{'h':>10} {'|dJ|':>14} {'|dJ - h g.d|':>16} {'rate0':>8} {'rate1':>8}")
#         prev0, prev1 = None, None
#         for k in range(n):
#             h = 10.0 ** (-k - 1)
#             Jh, _ = self.getFuncAndGradient(mu + h * direction)
#             e0 = abs(Jh - J0)
#             e1 = abs(Jh - J0 - h * gd)
#             r0 = "" if prev0 is None else f"{np.log10(prev0/e0):.2f}"
#             r1 = "" if prev1 is None else f"{np.log10(prev1/e1):.2f}"
#             print(f"{h:10.1e} {e0:14.4e} {e1:16.4e} {r0:>8} {r1:>8}")
#             prev0, prev1 = e0, e1

#     def save_current_solution(self, filename="reference_data/solution.xdmf"):
#         with io.XDMFFile(self.domain.comm, filename, "w") as xdmf:
#             xdmf.write_mesh(self.domain)
#             xdmf.write_function(self.u)

#     def save_reference_solution(self, filename="reference_data/reference_solution.xdmf"):
#         if self.u_ref is None:
#             raise RuntimeError("Reference solution u_ref has not been set.")
#         with io.XDMFFile(self.domain.comm, filename, "w") as xdmf:
#             xdmf.write_mesh(self.domain)
#             xdmf.write_function(self.u_ref)