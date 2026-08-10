
import functions.models.model as models
from functions.HKTR.kernel_width_hermite_TR import computeDataForRKHSNorm
from functions.HKTR.kernel import QuadWendland, QuadMatern, LinearWendland
import matplotlib.pyplot as plt
import numpy as np

model = models.NonlinearModel()
w_ref = np.array([0, 3, 7, 10, 0, 1, 2, 0, 10])
u_ref = model.compute_reference_from_weights(w_ref)
J_test_ref = model.compute_objective(w_ref)

TR_parameters={'radius': 1, 'sub_tolerance': 7.5e-5, 'max_iterations': 100, 'max_iterations_subproblem': 30, 'FOC_tolerance': 7.5e-5, 'J_tolerance': 1e-12,
               'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9, 'max_amount_interpolation_points': 20, 'cond_threshold': 1e20, 'gamma_adaptive': False}

max_amount = 500
amounts = np.unique(np.logspace(np.log10(10), np.log10(max_amount), num=50).astype(int))

RKHS_norms = []

x, y = computeDataForRKHSNorm(model, TR_parameters, amount=max_amount)
kernel = QuadWendland(gamma=0.009, d=model.dim)

for amount in amounts: 
    nested_targets = np.r_[y[:amount], y[max_amount:max_amount + model.dim * amount]]
    RKHS_norm = kernel.getRKHSNorm(x[:, :amount], nested_targets)
    RKHS_norms.append(RKHS_norm)
    print(f"Amount: {amount}, RKHS norm: {RKHS_norm}")

plt.plot(amounts, RKHS_norms, marker="o")
plt.xscale("log")
plt.xlabel("Amount")
plt.ylabel("RKHS norm")
plt.title("RKHS norm estimation")
plt.show()