import numpy as np
import time 
from functools import partial
from scipy.optimize import minimize, Bounds

def fom_objective_functional(fom, mu):
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
    #mu = np.atleast_2d(mu)
    #value_FOM =  - np.exp(- mu[0,0]**2) - np.exp(-(mu[0,0]-1)**2)

    #2D
    value_FOM = fom.output(mu)[0,0]

    #12D
    #mu_mor = fom.parameters.parse(mu)
    #value_FOM = fom.output_functional_hat(mu_mor)
    
    return value_FOM

def fom_gradient_of_functional(fom, mu):
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
    #1D
    #mu = np.atleast_2d(mu)
    #value_FOM_grad = 2*mu[0,0]* np.exp(-mu[0,0]**2) + (2*(mu[0,0] - 1)) * np.exp( - (mu[0,0] - 1)**2)

    #2D
    mu = np.atleast_2d(mu)
    value_FOM_grad = fom.output_d_mu(fom.parameters.parse(mu)).to_numpy()

    #12D
    #mu_mor = fom.parameters.parse(mu)
    #value_FOM_grad = fom.output_functional_hat_gradient(mu_mor)

    return np.atleast_2d(value_FOM_grad).reshape(1,-1)[0,:]

def record_results(function, data, fom, mu=None):
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
    QoI = function(fom, mu)
    data['counter'] += 1
    return QoI

def record_results_jac(function, data, fom, mu=None):
    QoI = function(fom, mu)
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
    data = {'times': np.zeros((1,amount_of_iters)), 'J_min': np.zeros((1,amount_of_iters)), 'mu': np.zeros((amount_of_iters,dim)), 'counter': 0, 'jac_counter': 0}
    return data

def optimize_all_iters(amount_of_iters, fom=None):
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
    #dim = 1
    dim = 2
    #dim = 12

    data = prepare_data(amount_of_iters, dim)
    for i in range(amount_of_iters): 

        np.random.seed(i)
        #1D
        #mu_k = np.random.uniform(-3,2)

        #2D 
        mu_k = np.random.uniform(0.25, np.pi, size=2)

        #12D
        #mu_k = np.array([1.12553301e-01, 1.58048674e-01, 1.14374817e-02, 3.02332573e+01, 1.46755891e+01, 9.23385948e+00, 1.86260211e+01, 3.45560727e+01, 3.96767474e+01, 6.54112551e-02, 5.64395886e-02, 7.63914625e-02])

        tic = time.time()
        fom_result = optimize(fom_objective_functional, data, mu_k, fom=fom)
        data['times'][0, i] = time.time()-tic
        data['J_min'][0,i] = fom_result.fun
        data['mu'][i,:] = fom_result.x

    return data 

def optimize(J, data, mu, fom=None):
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
    #bounds = Bounds([-2], [3], keep_feasible=True)
    #ranges = (-2.0,3.0)

    #for 2D
    bounds = Bounds([0, 0], [np.pi, np.pi], keep_feasible=True)
    ranges = (0, np.pi)
    # con = [{'type': 'ineq', 'fun': lambda x: x[0]}, 
    #     {'type': 'ineq', 'fun': lambda x: np.pi - x[0]}, 
    #     {'type': 'ineq', 'fun': lambda x: x[1]}, 
    #     {'type': 'ineq', 'fun': lambda x: np.pi - x[1]}
    # ]
    
    #For 12D
    # con = [{'type': 'ineq', 'fun': lambda x: x[0] - 0.05}, 
    #        {'type': 'ineq', 'fun': lambda x: 0.2 - x[0]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[1] - 0.05}, 
    #        {'type': 'ineq', 'fun': lambda x: 0.2 - x[1]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[2]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[2]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[3]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[3]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[4]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[4]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[5]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[5]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[6]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[6]}, 
    #        {'type': 'ineq', 'fun': lambda x: x[7]}, 
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[7]},
    #        {'type': 'ineq', 'fun': lambda x: x[8]},
    #        {'type': 'ineq', 'fun': lambda x: 100 - x[8]},
    #        {'type': 'ineq', 'fun': lambda x: x[9] - 0.025},
    #        {'type': 'ineq', 'fun': lambda x: 0.1 - x[9]},
    #        {'type': 'ineq', 'fun': lambda x: x[10] - 0.025},
    #        {'type': 'ineq', 'fun': lambda x: 0.1 - x[10]},
    #        {'type': 'ineq', 'fun': lambda x: x[11] - 0.025},
    #        {'type': 'ineq', 'fun': lambda x: 0.1 - x[11]}
    #     ]
    
    #ounds = Bounds([0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025], [0.2, 0.2, 100, 100, 100, 100, 100, 100, 100, 0.1, 0.1, 0.1], keep_feasible=True)
    
    #Note the naming is wrong
    #ranges_door = (0.05, 0.2)
    #ranges_heater = (0, 100)
    #ranges_wall = (0.025, 0.1)


    result = minimize(fun = partial(record_results, J, data, fom),
                      x0 = mu,
                      #method = 'L-BFGS-B',
                      method = 'trust-constr',
                      jac = partial(record_results_jac, fom_gradient_of_functional, data, fom),
                      #constraints=con,
                      #bounds = (ranges_door, ranges_door, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_heater, ranges_wall, ranges_wall, ranges_wall),
                      #bounds = [ranges],
                      bounds = (ranges, ranges),
                      #bounds = bounds,
                      options = {'gtol': 1e-10, 'disp': False}) #'ftol' : 1e-12
    #'initial_barrier_parameter': 1e-5, 'initial_barrier_tolerance': 1e-5,
    
    print(result)
 
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
    print(f'  avg. gradFOM evals: {data["jac_counter"]/amount_of_iters}')
    print(f'  avg. time:      {sum(data["times"][0,:])/amount_of_iters} seconds')
    print('')