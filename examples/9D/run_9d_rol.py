import functions.models.model as models
import functions.analysis.rol_algos as rol_algos
import numpy as np

model = models.NonlinearModel()

w_ref = np.array([0, 3, 7, 10, 0, 1, 2, 0, 10])
u_ref = model.compute_reference_from_weights(w_ref)

amount_of_iters = 5
data = rol_algos.optimize_all_iters(amount_of_iters, model)
rol_algos.report(data, amount_of_iters)