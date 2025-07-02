import numpy as np
import time 
from functools import partial
from scipy.optimize import minimize, Bounds
from pymor.tools.random import new_rng

def fom_objective_functional(model, mu):
    """ This method evaluates the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    fom
        The FOM that gets evaluated.
    mu 
        The parameter for which the FOM is evaluated.

    Returns 
    -------
    value_FOM
        The value od the FOM at the parameter |mu|.
    """
    #1D 
    if model.dim == 1:
        mu = np.atleast_2d(mu)
        #value_FOM =  - np.exp(- mu[0,0]**2) - np.exp(-(mu[0,0]-1)**2) + 2
        value_FOM = (-1) * np.exp(-mu[0,0]**2) + 3 * np.exp(-0.001 * mu[0,0]**2)

    elif model.dim == 2:
        value_FOM = model.fom.output(mu)[0,0]

    elif model.dim == 12:
        mu_mor = model.fom.parameters.parse(mu)
        value_FOM = model.fom.output_functional_hat(mu_mor)
    
    return value_FOM

def fom_gradient_of_functional(model, mu):
    """ This method evaluates the gradient of the full order model (FOM) at the given parameter |mu|.

    Parameters
    ----------
    fom
        The FOM that gets evaluated.
    mu 
        The parameter for which the gradient of the FOM is evaluated.

    Returns 
    -------
    value_FOM_grad
        The value of the gradient of the FOM at the parameter |mu|.
    """
    if model.dim == 1:
        mu = np.atleast_2d(mu)
        #value_FOM_grad = 2*mu[0,0]* np.exp(-mu[0,0]**2) + (2*(mu[0,0] - 1)) * np.exp( - (mu[0,0] - 1)**2)
        value_FOM_grad = 2*mu[0,0]* np.exp(-mu[0,0]**2) - ((3 * mu[0,0] * np.exp(- 0.001 * mu[0,0]**2)) / 500)

    elif model.dim == 2:
        mu = np.atleast_2d(mu)
        value_FOM_grad = model.fom.output_d_mu(model.fom.parameters.parse(mu)).to_numpy()

    elif model.dim == 12:
        mu_mor = model.fom.parameters.parse(mu)
        value_FOM_grad = model.fom.output_functional_hat_gradient(mu_mor)

    return np.atleast_2d(value_FOM_grad).reshape(1,-1)[0,:]

def record_results(function, data, model, mu=None):
    """ 

    Parameters
    ----------
    function
        The |function| that is evaluated.
    data
        Dictionary |data| to store the results of the optimization algorithm.
    fom 
        The |fom| that is used as an argument of function.
    mu 
        The current iterate |mu| that is used as an argument of function.

    Returns 
    -------
    QoI
        Output of |function|.
    """
    QoI = function(model, mu)
    data['counter'] += 1
    return QoI

def record_results_jac(function, data, model, mu=None):
    QoI = function(model, mu)
    data['jac_counter'] += 1
    return QoI 


def prepare_data(amount_of_iters, dim):
    """
    Creats a dictionary |data| to save relevant information about the optimization algorithm.

    Parameters
    ----------
    amount_of_iters
        Number of different starting parameters we use.

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """
    data = {'J_error': np.zeros((1,1)), 'foc': np.zeros((1,1)), 'J_min': np.zeros((1,amount_of_iters)), 'mu': np.zeros((amount_of_iters,dim)), 'counter': 0, 'jac_counter': 0}
    return data

def optimize_all_iters(amount_of_iters, model):
    """ Repeats the optimization |amount_of_iters| times with different starting parameters. 

    Parameters
    ----------
    J 
        The objective function that gets optimized. 
    ranges
        The |ranges| of the parameters space. 
    amount_of_iters
        Amount of times the optimization is done. 
    fom 
        The full order model. 
    gradient 
        Gradient information about the full order model

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """
    dim = model.dim

    data = prepare_data(amount_of_iters, dim)
    for i in range(amount_of_iters): 

        if dim == 1: 
            np.random.seed(i)
            mu_k = np.random.uniform(-2,2)
        elif dim == 2:
            np.random.seed(i)
            mu_k = np.random.uniform(0.5, np.pi, size=2)
        else: 
            with new_rng(i):
                mu_k = model.parameter_space.sample_randomly(1)[0].to_numpy()

        #mu_k = np.array([1.12553301e-01, 1.58048674e-01, 1.14374817e-02, 3.02332573e+01, 1.46755891e+01, 9.23385948e+00, 1.86260211e+01, 3.45560727e+01, 3.96767474e+01, 6.54112551e-02, 5.64395886e-02, 7.63914625e-02])

        fom_result = optimize(fom_objective_functional, data, mu_k, model=model)
        data['J_min'][0,i] = fom_result.fun
        data['mu'][i,:] = fom_result.x

        #Save Data
        if model.dim == 1: 
            data['J_error'][0,0]  += abs((fom_result.fun - 2)/2)
            #data['foc'][0,0] += abs(fom_result['jac'][0])

        elif model.dim == 2:
            data['J_error'][0,0] += abs((fom_result.fun - 2.3917078761)/(2.3917078761))
            #data['foc'][0,0] += abs(fom_result['jac'][0])

        elif model.dim == 12: 
            data['J_error'][0,0]  += abs((fom_result.fun - 5.813965062384796)/(5.813965062384796))
        else: 
            raise NotImplementedError

    return data 

def optimize(J, data, mu, model=None):
    """ Calls the minimize method from scipy to solve the optimization problem. 

    Parameters
    ----------
    J 
        The objective function that gets optimized. 
    data 
        Dictionary |data| to store results of the optimization algorithm.
    ranges
        The |ranges| of the parameters space. 
    mu 
        The starting parameter |mu|. 
    fom 
        The full order model. 
    gradient 
        Gradient information about the full order model

    Returns 
    -------
    result
        The |result| of one optimization run. 
    """
    
    #jac = partial(fom_gradient_of_functional, fom)
    
    #Three different ways to encode the box constraints (all are needed for different kind of methods)

    #for 1D
    ranges = (-2.0,2.0)

    #for 2D
    #ranges = (0.5, np.pi)
   
    #for 12D
    #ranges_door = (0.05, 0.2)
    #ranges_heater = (0, 100)
    #ranges_wall = (0.025, 0.1)

    def callback(x):
        print("current iterate", x)


    result = minimize(fun = partial(record_results, J, data, model),
                      x0 = mu,
                      method = 'L-BFGS-B',
                      #method = 'trust-constr',
                      #method = 'SLSQP',
                      jac = partial(record_results_jac, fom_gradient_of_functional, data, model),
                      #constraints=con,
                      #bounds = (ranges_door, ranges_door, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_wall, ranges_wall, ranges_wall),
                      bounds = [ranges],
                      #bounds = (ranges, ranges),
                      #bounds = bounds,
                      #callback=callback,
                      options = {'gtol': 1e-7, 'ftol':1e-14})
    
    #'initial_barrier_parameter': 1e-5, 'initial_barrier_tolerance': 1e-5,
    return result

def report(data, amount_of_iters):
    """Reports the results of the optimization algorithm. 

    Parameters
    ----------
    data
        Dictionary |data| to store results of the optimization algorithm.
    amount of iters
        Amount of times the optimization is done. 
    """
    print('\n succeeded!')
    print(f'  mu_min:    {data["mu"][-1,:]}') #takes the last one, assuming that all solution yield the same result.
    print(f'  J(mu_min): {data["J_min"][0,-1]}') #takes the last one, assuming that all solution yield the same result.
    print(f'  avg. FOM evals: {data["counter"]/amount_of_iters}')
    print(f'  avg. error in J: {data["J_error"]/amount_of_iters}')
    #print(f'  avg. FOC: {data["foc"]/amount_of_iters}')
    print('')