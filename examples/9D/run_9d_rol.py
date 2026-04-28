import functions.models.model as models
import functions.analysis.rol_setup as rol_setup
import functions.analysis.results_analysis as result_analysis
import functions.analysis.scipy_algos as scipy_algos
import numpy as np

from functions.rol_setup import NonlinearROLObjective

model = models.NonlinearModel()

w_ref = np.array([0, 3, 7, 10, 0, 1, 2, 0, 10])
u_ref = model.compute_reference_from_weights(w_ref)

J_test_ref = model.compute_objective(w_ref)

print("the misfit for the reference solution is", J_test_ref)

# # Your own taylor test
# print("=== own taylor_test ===")
# model.taylor_test(w_ref)

# # pyROL check
# x = GVector(w_ref)
# d = GVector(np.ones(9) / 3.0)

# obj = NonlinearROLObjective(model)
# print("\n=== pyROL checkGradient ===")
# obj.checkGradient(x, d, True)

# Run pyROL
np.random.seed(4)
w0 = np.random.uniform(0, 10, size=9)

w_opt = rol_setup.run_rol(model, w0)

print("Recovered weights:", w_opt)
print("True weights:     ", w_ref)
print("Error in J:",        abs(model.compute_objective(w_opt) - 3.27521e-06))