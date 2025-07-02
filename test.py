import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyMORAuxData'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

import functions.model as models
import functions.scipy_algos as scipy_algos
import functions.results_analysis as result_analysis


gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
amount_of_iters = 5

#1D
#model = models.Gaussian1D() #maxpoints 5, 1e17

#2D
model  = models.twoDStuff() #maxpoints 10, 1e20

#12D
#model  = models.buildingFloor() #maxpoints 15, 1e21

TR_parameters={'radius': 1, 'max_radius': 500, 'sub_tolerance': 1e-4, 'max_iterations': 100, 'max_iterations_subproblem': 30, 'max_iterations_armijo': 20,
               'initial_step_armijo': 0.5, 'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-4, 'J_tolerance': 1e-12,
               'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'max_amount_interpolation_points': 10, 'rho_1': 0.25, 'rho_2': 0.75, 'accept_eta': 0.1, 'gamma_adaptive': False}


optim_data = result_analysis.optimize_all(model, gamma_list, TR_parameters, amount_of_iters)
result_analysis.report_kernel_TR(optim_data, gamma_list, amount_of_iters)

#optim_data = scipy_algos.optimize_all_iters(amount_of_iters=amount_of_iters, model=model)
#scipy_algos.report(optim_data, amount_of_iters=amount_of_iters)