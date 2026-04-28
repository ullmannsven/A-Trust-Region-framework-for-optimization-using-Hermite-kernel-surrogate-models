import functions.model as models
import functions.scipy_algos as scipy_algos
import numpy as np

amount_of_iters = 5
model  = models.NonlinearModel()
w_ref = np.array([0, 3, 7, 10, 0, 1, 2, 0, 10])

u_ref = model.compute_reference_from_weights(w_ref)
J_test_ref = model.compute_objective(w_ref)
print("der objective value", J_test_ref)

optim_data = scipy_algos.optimize_all_iters(amount_of_iters=amount_of_iters, method='bfgs', model=model)
scipy_algos.report(optim_data, amount_of_iters=amount_of_iters)
