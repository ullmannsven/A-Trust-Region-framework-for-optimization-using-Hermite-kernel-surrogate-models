import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyMORAuxData'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

import functions.model as models
import functions.scipy_algos as scipy_algos


amount_of_iters = 5
model = models.Gaussian1D()

optim_data = scipy_algos.optimize_all_iters(amount_of_iters=amount_of_iters, method='bfgs', model=model)
scipy_algos.report(optim_data, amount_of_iters=amount_of_iters)