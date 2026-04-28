
import functions.model as models
import functions.results_analysis as result_analysis
import functions.scipy_algos as scipy_algos
import numpy as np

model = models.NonlinearModel()

w_ref = np.array([0, 3, 7, 10, 0, 1, 2, 0, 10])
u_ref = model.compute_reference_from_weights(w_ref)

J_test_ref = model.compute_objective(w_ref)

print("the misfit for the reference solution is", J_test_ref)

gamma_list = [0.01, 0.011]
amount_of_iters = 5

TR_parameters={'radius': 1, 'sub_tolerance': 7.5e-5, 'max_iterations': 100, 'max_iterations_subproblem': 30, 'FOC_tolerance': 7.5e-5, 'J_tolerance': 1e-12,
               'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'max_amount_interpolation_points': 20, 'cond_threshold': 1e20, 'gamma_adaptive': False}

optim_data = result_analysis.optimize_all(model, gamma_list, TR_parameters, amount_of_iters)
result_analysis.report_kernel_TR(optim_data, gamma_list, amount_of_iters)





