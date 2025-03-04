# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2020)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np

from pymor.core.base import ImmutableObject
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.operators.constructions import VectorFunctional, VectorOperator
from pymor.discretizers.builtin.cg import (DiffusionOperatorP1, DiffusionOperatorQ1,
                                AdvectionOperatorP1, AdvectionOperatorQ1,
                                L2ProductP1, L2ProductQ1,
                                L2ProductFunctionalP1, L2ProductFunctionalQ1,
                                BoundaryL2ProductFunctional,
                                BoundaryDirichletFunctional, RobinBoundaryOperator,
                                InterpolationOperator)
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.functionals import GenericParameterFunctional, ConstantParameterFunctional
from pymor.parameters.base import ParametricObject
from pymor.vectorarrays.numpy import NumpyVectorSpace

from twelve_dim_model import QuadraticPdeoptStationaryModel
from pymor.discretizers.builtin.grids.rect import RectGrid

def _construct_mu_bar(problem):
    mu_bar = []
    for key, size in sorted(problem.parameter_space.parameters.items()):
        range_ = problem.parameter_space.ranges[key]
        if range_[0] == 0:
            value = 10**(np.log10(range_[1])/2)
        else:
            value = 10**((np.log10(range_[0]) + np.log10(range_[1]))/2)
        for i in range(size):
            mu_bar.append(value)
    return problem.parameters.parse(mu_bar)

def discretize_quadratic_pdeopt_stationary_cg(problem, diameter=np.sqrt(2)/200., weights=None, parameter_scales=None,
                                              domain_of_interest=None, desired_temperature=None, mu_for_u_d=None,
                                              mu_for_tikhonov=False, parameters_in_q=True, product='h1_l2_boundary',
                                              solver_options=None, use_corrected_functional=True,
                                              adjoint_approach=True):
    if use_corrected_functional and adjoint_approach:
        print('I am using the NCD corrected functional!!')
    else:
        print('I am using the non corrected functional!!')

    mu_bar = _construct_mu_bar(problem)
    primal_fom, data = discretize_stationary_cg(problem, diameter=diameter, grid_type=RectGrid, mu_energy_product=mu_bar)

    if solver_options == 'pyamg':
        from pymor.bindings.pyamg import solver_options as pyamg_solver_options
        primal_fom = primal_fom.with_(operator=primal_fom.operator.with_(solver_options={'inverse':
                                                                          pyamg_solver_options(verb=False)['pyamg_solve']}))

    # put the constant part in only one function
    simplified_operators = [ZeroOperator(primal_fom.solution_space,primal_fom.solution_space)]
    simplified_coefficients = [1]
    to_pre_assemble = ZeroOperator(primal_fom.solution_space,primal_fom.solution_space)

    if isinstance(primal_fom.operator, LincombOperator):
        for (i, coef) in enumerate(primal_fom.operator.coefficients):
            if isinstance(coef, ParametricObject):
                simplified_coefficients.append(coef)
                simplified_operators.append(primal_fom.operator.operators[i])
            else:
                to_pre_assemble += coef * primal_fom.operator.operators[i]
    else:
        to_pre_assemble += primal_fom.operator

    simplified_operators[0] += to_pre_assemble
    simplified_operators[0] = simplified_operators[0].assemble()

    lincomb_operator = LincombOperator(simplified_operators,simplified_coefficients,
                                       solver_options=primal_fom.operator.solver_options)

    simplified_rhs = [ZeroOperator(primal_fom.solution_space,NumpyVectorSpace(1))]
    simplified_rhs_coefficients = [1]
    to_pre_assemble = ZeroOperator(primal_fom.solution_space,NumpyVectorSpace(1))

    if isinstance(primal_fom.rhs, LincombOperator):
        for (i, coef) in enumerate(primal_fom.rhs.coefficients):
            if isinstance(coef, ParametricObject):
                simplified_rhs_coefficients.append(coef)
                simplified_rhs.append(primal_fom.rhs.operators[i])
            else:
                to_pre_assemble += coef * primal_fom.rhs.operators[i]
    else:
        to_pre_assemble += primal_fom.rhs

    simplified_rhs[0] += to_pre_assemble
    simplified_rhs[0] = simplified_rhs[0].assemble()
    lincomb_rhs = LincombOperator(simplified_rhs,simplified_rhs_coefficients)

    primal_fom = primal_fom.with_(operator=lincomb_operator, rhs=lincomb_rhs)

    grid = data['grid']
    d = grid.dim

    # prepare data functions
    if desired_temperature is not None:
        u_desired = ConstantFunction(desired_temperature, d)
    if domain_of_interest is None:
        domain_of_interest = ConstantFunction(1., d)
    if mu_for_u_d is not None:
        domain_of_interest = ConstantFunction(1., d)
        modifified_mu = mu_for_u_d.copy()
        for key in mu_for_u_d.keys():
            if len(mu_for_u_d[key]) == 0:
                modifified_mu.pop(key)
        u_d = primal_fom.solve(modifified_mu)
    else:
        assert desired_temperature is not None
        u_d = InterpolationOperator(grid, u_desired).as_vector()
    DoI = InterpolationOperator(grid, domain_of_interest).as_vector()

    if grid.reference_element is square:
        L2_OP = L2ProductQ1
    else:
        L2_OP = L2ProductP1

    empty_bi = EmptyBoundaryInfo(grid)
    Restricted_L2_OP = L2_OP(grid, data['boundary_info'], dirichlet_clear_rows=False, coefficient_function=domain_of_interest)

    l2_u_d_squared = Restricted_L2_OP.apply2(u_d,u_d)[0][0]
    constant_part = 0.5 * l2_u_d_squared

    # assemble output functional
    from twelve_dim_theta import build_output_coefficient

    if weights is not None:
        weight_for_J = weights.pop('state')
    else:
        weight_for_J = 1.

    if isinstance(weight_for_J, dict):
        assert 0, "this functionality is not covered in this publication!!"
        assert len(weight_for_J) == 4, 'you need to give all derivatives including second order'
        state_functional = ExpressionParameterFunctional(weight_for_J['function'], weight_for_J['parameter_type'],
                                                         derivative_expressions=weight_for_J['derivative'],
                                                         second_derivative_expressions=weight_for_J['second_derivatives'])

    elif isinstance(weight_for_J, float) or isinstance(weight_for_J, int):
        #wir fallen hier rein!! J = 100 wegen sigma_d = 100
        state_functional = ConstantParameterFunctional(weight_for_J)
    else:
        assert 0, 'state weight needs to be an integer or a dict with derivatives'

    if mu_for_tikhonov:
        if mu_for_u_d is not None:
            mu_for_tikhonov = mu_for_u_d
        else:
            assert isinstance(mu_for_tikhonov, dict)
        output_coefficient = build_output_coefficient(primal_fom.parameters, weights, mu_for_tikhonov,
                                                      parameter_scales, state_functional, constant_part)
    else:
        output_coefficient = build_output_coefficient(primal_fom.parameters, weights, None, parameter_scales,
                                                      state_functional, constant_part)

    output_functional = {}

    output_functional['output_coefficient'] = output_coefficient
    output_functional['linear_part'] = LincombOperator([VectorOperator(Restricted_L2_OP.apply(u_d))],[-state_functional])   # j(.)
    output_functional['bilinear_part'] = LincombOperator([Restricted_L2_OP],[0.5*state_functional])                           # k(.,.)
    output_functional['d_u_linear_part'] = LincombOperator([VectorOperator(Restricted_L2_OP.apply(u_d))],[-state_functional]) # j(.)
    output_functional['d_u_bilinear_part'] = LincombOperator([Restricted_L2_OP], [state_functional])                           # 2k(.,.)

    l2_boundary_product = RobinBoundaryOperator(grid, data['boundary_info'], robin_data=(ConstantFunction(1,2),ConstantFunction(1,2)),
                                    name='l2_boundary_product')

    # choose product
    if product == 'h1_l2_boundary':
        opt_product = primal_fom.h1_semi_product + l2_boundary_product       # h1_semi + l2_boundary
    elif product == 'fixed_energy':
        opt_product = primal_fom.energy_product                              # energy w.r.t. mu_bar (see above)
    else:
        assert 0, 'product: {} is not known'.format(product)
    print('my product is {}'.format(product))
    print('mu_bar is: {}'.format(mu_bar))

    primal_fom = primal_fom.with_(products=dict(opt=opt_product, l2_boundary=l2_boundary_product,
                                                **primal_fom.products))
    pde_opt_fom = QuadraticPdeoptStationaryModel(primal_fom, output_functional, opt_product=opt_product,
                                                 use_corrected_functional=use_corrected_functional,
                                                 adjoint_approach=adjoint_approach)
    return pde_opt_fom, data, mu_bar
