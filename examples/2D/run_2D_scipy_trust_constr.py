import functions.models as models
import functions.analysis.scipy_algos as scipy_algos

amount_of_iters = 5

model  = models.twoDStuff() 

optim_data = scipy_algos.optimize_all_iters(amount_of_iters=amount_of_iters, method='trust-constr', model=model)
scipy_algos.report(optim_data, amount_of_iters=amount_of_iters)