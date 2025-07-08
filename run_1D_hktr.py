import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyMORAuxData'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

import functions.model as models
import functions.results_analysis as result_analysis


gamma_list = [0.725, 0.75, 1, 2, 10]
amount_of_iters = 5
model = models.Gaussian1D()


TR_parameters={'radius': 1, 'sub_tolerance': 1e-7, 'max_iterations': 100, 'max_iterations_subproblem': 30, 'FOC_tolerance': 1e-7, 'J_tolerance': 1e-14,
               'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'max_amount_interpolation_points': 5, 'cond_threshold': 1e17, 'gamma_adaptive': False}


optim_data = result_analysis.optimize_all(model, gamma_list, TR_parameters, amount_of_iters)
result_analysis.report_kernel_TR(optim_data, gamma_list, amount_of_iters)