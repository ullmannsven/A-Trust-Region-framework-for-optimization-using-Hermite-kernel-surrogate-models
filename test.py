import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyMORAuxData'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

import functions.kernel as kernels
import functions.model as models
import functions.scipy_algos as scipy_algos
import functions.kernel_width_hermite_TR as kernel_width_hermite_TR
import functions.results_analysis as result_analysis

gamma_list = [0.4]
amount_of_iters = 2

#12D
#model  = models.buildingFloor()
#kernel = kernels.QuadMatern(gamma=gamma)
#mu_start  = np.array([[1.12553301e-01, 1.58048674e-01, 1.14374817e-02, 3.02332573e+01, 1.46755891e+01, 9.23385948e+00, 1.86260211e+01, 3.45560727e+01, 3.96767474e+01, 6.54112551e-02, 5.64395886e-02, 7.63914625e-02, gamma]]).T

#2D
kernel = kernels.InvMulti(gamma=gamma_list[0])
model  = models.twoDStuff()

TR_parameters={'radius': 0.5, 'sub_tolerance': 1e-4, 'max_iterations': 50, 'max_iterations_subproblem': 20, 'max_iterations_armijo': 20,
               'initial_step_armijo': 0.5, 'armijo_alpha': 1e-4, 'FOC_tolerance': 1e-5, 'J_tolerance': 1e-12,
               'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'max_amount_interpolation_points': 20, 'amount_RKHS_FOMs': 30}


#optim_data = result_analysis.optimize_all(model, kernel, gamma_list, TR_parameters, amount_of_iters)
#result_analysis.report_kernel_TR(optim_data, gamma_list, amount_of_iters)

optim_data = scipy_algos.optimize_all_iters(amount_of_iters=amount_of_iters, fom=model.fom)
scipy_algos.report(optim_data, amount_of_iters=amount_of_iters)