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
#   Tim Keil (2019 - 2020)
# ~~~

import numpy as np

from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.functionals import GenericParameterFunctional, ConstantParameterFunctional


def build_output_coefficient(parameters, parameter_weights=None, mu_d_=None, parameter_scales=None,
                             state_functional=None, constant_part=None):
    if parameter_weights is None:
        parameter_weights= {'walls': 1, 'heaters': 1, 'doors': 1, 'windows': 1}
    if parameter_scales is None:
        parameter_scales= {'walls': 1, 'heaters': 1, 'doors': 1, 'windows': 1}
    mu_d={}
    if mu_d_ is None:
        mu_d_ = {}
    for key, size in parameters.items():
        if not isinstance(parameter_weights[key], list):
            parameter_weights[key] = [parameter_weights[key] for i in range(size)]
        if key not in mu_d_:
            mu_d_[key] = [0 for i in range(size)]
        if not isinstance(mu_d_[key], list):
            mu_d[key] = [mu_d_[key][i] for i in range(size)]
        else:
            mu_d[key] = mu_d_[key]

    parameter_functionals = []

    if constant_part is not None:
        # sigma_d * constant_part
        parameter_functionals.append(state_functional * constant_part)
        # + 1
        parameter_functionals.append(ConstantParameterFunctional(1))

    def make_zero_expressions(parameters):
        zero_derivative_expression = {}
        for key, size in parameters.items():
            zero_expressions = np.array([], dtype='<U60')
            for l in range(size):
                zero_expressions = np.append(zero_expressions, ['0'])
            zero_expressions = np.array(zero_expressions, dtype='<U60')
            zero_derivative_expression[key] = zero_expressions
        return zero_derivative_expression

    #prepare dict
    def make_dict_zero_expressions(parameters):
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=dict)
            zero_dict[key] = zero_
            for l in range(size):
                zero_dict[key][l] = make_zero_expressions(parameters)
        return zero_dict

    for key, size in parameters.items():
        for i in range(size):
            weight = parameter_weights[key][i]
            derivative_expression = make_zero_expressions(parameters)
            second_derivative_expressions = make_dict_zero_expressions(parameters)
            derivative_expression[key][i] = \
                        '{}*{}**2*({}[{}]-'.format(weight,parameter_scales[key],key,i) + '{}'.format(mu_d[key][i])+')'
            second_derivative_expressions[key][i][key][i]= '{}*{}**2'.format(weight,parameter_scales[key])
            parameter_functionals.append(ExpressionParameterFunctional('{}*{}**2*0.5*({}[{}]'.format( \
                                                                        weight,parameter_scales[key],key,i) \
                                                                        +'-{}'.format(mu_d[key][i])+')**2',
                                                                        parameters,
                                                                        derivative_expressions=derivative_expression,
                                                                        second_derivative_expressions=second_derivative_expressions))
    def mapping(mu):
        ret = 0
        for f in parameter_functionals:
            ret += f.evaluate(mu)
        return ret

    def make_mapping(key, i):
        def sum_derivatives(mu):
            ret = 0
            for f in parameter_functionals:
                ret += f.d_mu(key, i).evaluate(mu)
            return ret
        return sum_derivatives

    def make_second_mapping(key, i):
        def sum_second_derivatives(mu):
            ret = 0
            for f in parameter_functionals:
                ret += f.d_mu(key, i).d_mu(key, i).evaluate(mu)
            return ret
        return sum_second_derivatives

    def make_zero_mappings(parameters):
        zero_derivative_mappings = {}
        for key, size in parameters.items():
            zero_mappings = np.array([], dtype=object)
            zero_mapping = lambda mu: 0.
            for i in range(size):
                zero_mappings = np.append(zero_mappings, [zero_mapping])
            zero_derivative_mappings[key] = zero_mappings
        return zero_derivative_mappings

    #prepare dict
    def make_dict_zero_mapping(parameters):
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=dict)
            zero_dict[key] = zero_
            for l in range(size):
                zero_dict[key][l] = make_zero_mappings(parameters)
        return zero_dict

    derivative_mappings = make_zero_mappings(parameters)
    for key, size in parameters.items():
        for i in range(size):
            derivative_mappings[key][i] = make_mapping(key,i)

    second_derivative_mappings = make_dict_zero_mapping(parameters)
    for key, size in parameters.items():
        for i in range(size):
            second_derivative_mappings[key][i][key][i] = make_second_mapping(key,i)

    output_coefficient = GenericParameterFunctional(mapping, parameters,
                                                    derivative_mappings=derivative_mappings,
                                                    second_derivative_mappings=second_derivative_mappings)
    return output_coefficient
