import numpy as np
from functools import partial
from scipy.optimize import minimize
import scipy as sp
import torch
from torch.func import jacfwd, jacrev
import kernel as kernels

def projection_onto_range(model, X_train):
    """Projects the parameter |mu| onto the given range of the parameter space.

    Parameters
    ----------
    parameter_space
        The |parameter_space| of the full order model which is optimized.
    mu
        The parameter |mu| that is projected onto the given range.

    Returns
    -------
    mu_new
        The projected parameter |mu_new|.
    """

    X_train_new = X_train.copy()

    for j in range(X_train.shape[1]):
        if model.pyMOR:
            index = 0
            for (key, val) in model.parameter_space.parameters.items():
                range_ = model.parameter_space.ranges[key]
                for i in range(index, index + val):
                    if X_train[i,j] < range_[0]:
                        X_train_new[i,j] = range_[0]
                    if X_train[i,j] > range_[1]:
                        X_train_new[i,j] = range_[1]
                    index += 1
        else: #the 1D case
            range_ = model.parameter_space
            if X_train[0,j] < range_[0]: 
                X_train_new[0,j] = range_[0]
            if X_train[0,j] > range_[1]: 
                X_train_new[0,j] = range_[1]
        
    return X_train_new

def active_and_inactive_sets(model, mu, epsilon):

    Act    = []

    if model.pyMOR:

        mu     = model.fom.parameters.parse(mu)
        ranges = model.parameter_space.ranges

        for (key,val) in model.parameter_space.parameters.items():
            range_ = ranges[key]
            for j in range(val):
                if mu[key][j] - range_[0] <= epsilon:
                    Act.append(1.0)
                elif range_[1] - mu[key][j] <= epsilon:
                    Act.append(1.0)
                else:
                    Act.append(0.0)

    else: #the 1D case
        range_ = model.parameter_space
        if mu[0,0] - range_[0] <= epsilon: 
            Act.append(1.0)
        elif range_[1] - mu[0,0] <= epsilon: 
            Act.append(1.0)
        else: 
            Act.append(0.0)

    Act   = np.array(Act)
    Inact = np.ones(Act.shape) - Act

    return Act, Inact


def computeDataForRKHSNorm(model, TR_parameters):
    from pymor.tools.random import new_rng

    amount             = 10
    dim                = model.dim
    
    with new_rng(amount):
        random_samples     = model.parameter_space.sample_randomly(amount)
    
    train_values       = np.zeros((dim, amount))
    target_values      = np.zeros((amount, 1))
    grad_target_values = np.zeros((dim, amount))

    for i in range(amount):
        mu                                           = random_samples[i].to_numpy()
        train_values[:,i]                            = mu
        target_values[i, 0], grad_target_values[:,i] = model.getFuncAndGradient(mu)

    return  train_values, np.r_[target_values, grad_target_values.flatten(order='F').reshape(-1,1)]

#war auf 1e20 für 12D, geändert auf 1e17 für 1D
def remove_similar_points(X_train, y_train, grad_y_train, kernel, gamma, cond_threshold=1e21):
    """
    Remove points from the training set until the condition number of the Gram matrix,
    computed with kernel.getGramHermite on the features (all rows except the last),
    is below cond_threshold.

    Parameters:
      X_train: 2D array with shape (n_features+1, n_points). The last row may hold extra info.
      y_train: 2D or 1D array of target values.
      grad_y_train: gradients corresponding to y_train.
      kernel: object with a method getGramHermite that takes (X1, X2, newGamma)
      rhs: right-hand side vector (not used here but may be required elsewhere)
      gamma: the value to be used for newGamma in the kernel evaluation.
      cond_threshold: desired maximum condition number of the Gram matrix.
      
    Returns:
      Updated (X_train, y_train, grad_y_train) with points removed.
    """
    # Compute initial Gram matrix and condition number.
    gram = kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=gamma)
    cond_num = np.linalg.cond(gram)
    
    # Continue removing points while condition number is too high and we have enough points.
    while cond_num > cond_threshold and X_train.shape[1] > 1:
        num_points = X_train.shape[1]
        min_distance = np.inf
        idx_to_remove = None
        
        # Find the pair of points (using the feature rows only) with the smallest distance.
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.linalg.norm(X_train[:-1, i] - X_train[:-1, j])
                if dist < min_distance:
                    min_distance = dist
                    idx_to_remove = i  # Remove one of the points in the closest pair.
        
        # If a point to remove is found, remove it from all arrays.
        if idx_to_remove is not None:
            print(f"Removed point index {idx_to_remove} (min pairwise distance: {min_distance:.4f}) "
                  f"to improve condition number from {cond_num:.2e}")
            X_train      = np.delete(X_train, idx_to_remove, axis=1)
            y_train      = np.delete(y_train, idx_to_remove, axis=0)
            grad_y_train = np.delete(grad_y_train, idx_to_remove, axis=1)
            
            # Recompute the Gram matrix and its condition number after removal.
            gram = kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=gamma)
            cond_num = np.linalg.cond(gram)
        else:
            # If no candidate is found, break out.
            break

    if cond_num > cond_threshold:
        print(f"Warning: condition number is still high ({cond_num:.2e}) after removals.")
    else:
        print(f"Final condition number: {cond_num:.2e}")

    return X_train, y_train, grad_y_train


def remove_far_away_points(X_train, y_train, grad_y_train, mu_k, TR_parameters):
    """Removes points from the parameter training set |X_train| if they far away from the current iterate |mu_k|. """

    num_to_keep = TR_parameters['max_amount_interpolation_points']

    if num_to_keep > X_train.shape[1]:
        num_to_keep = X_train.shape[1]

    distances   = np.linalg.norm(X_train[:-1,:] - mu_k[:-1,:], axis=0)
    idx_to_keep = np.argsort(distances,)[:num_to_keep]
    idx_to_keep = np.sort(idx_to_keep)

    X_train      =  X_train[:, idx_to_keep]
    y_train      =  y_train[idx_to_keep, :]
    grad_y_train = grad_y_train[:, idx_to_keep]

    return X_train, y_train, grad_y_train


def create_training_dataset(kernel, mu_k, gradient, model, X_train, y_train, grad_y_train, TR_parameters):
    
    num_of_points_old = X_train.shape[1]
    new_point         = np.zeros((model.dim + 1, 1))
    new_point[:-1, 0] = mu_k[:-1, 0] - gradient[:-1, 0]
    new_point[-1, 0]  = mu_k[-1,0]
    X_train           = np.append(X_train, np.atleast_2d(new_point).reshape(-1,1), axis=1)
    X_train           = projection_onto_range(model, X_train)
    num_of_points     = X_train.shape[1]

    for i in range(num_of_points_old, num_of_points):

        new_target_value, grad_target_value = model.getFuncAndGradient(X_train[:-1, i])
        y_train                             = np.append(y_train, np.atleast_2d(new_target_value), axis=0)
        grad_y_train                        = np.append(grad_y_train, np.atleast_2d(grad_target_value).reshape(-1,1), axis=1)
        X_train, y_train, grad_y_train      = remove_similar_points(X_train[:, :i+1], y_train, grad_y_train, kernel, gamma=new_point[-1,0]) #
    
    return X_train, y_train, grad_y_train


def compute_gradientGamma(mu_k, kernel, X_train, rhs):
    mu_k_torch    = torch.from_numpy(np.atleast_2d(mu_k).reshape(-1,1))
    X_train_torch = torch.from_numpy(X_train)
    rhs_torch     = torch.from_numpy(rhs)
    targetFunc    = lambda gamma: kernel.evalFuncTorch(mu_k_torch[:-1, :], X_train_torch[:-1, :], torch.linalg.solve(kernel.getGramHermiteTorch(X_train_torch[:-1,:], X_train_torch[:-1, :], gamma), rhs_torch), gamma)
    #return np.array([[0]])
    gamma         = torch.tensor(mu_k_torch[-1,0], dtype=torch.float32, requires_grad=True)
    #eturn np.atleast_2d(jacrev(targetFunc)(gamma)[0].detach().numpy())
    return np.atleast_2d(torch.autograd.grad(targetFunc(gamma), gamma)[0].detach().numpy())



def solve_subproblem_scipyBFGS(model, kernel, alpha, X_train, rhs, mu_k, TR_parameters, RKHS_train_values, RKHS_rhs):

    current_rad = TR_parameters['radius']
    dim = model.dim 

    def partial_grad_func_noGamma(mu, alpha=alpha, X_train=X_train): 
        if TR_parameters['gamma_adaptive']:
            return kernel.evalGrad(mu[:-1], x=X_train[:-1,:], alpha=np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=mu[-1]), rhs))
        else: 
            return kernel.evalGrad(mu, x=X_train[:-1,:], alpha=alpha)

    if TR_parameters['gamma_adaptive']:
        partial_grad_func_Gamma = partial(compute_gradientGamma, kernel=kernel, X_train=X_train, rhs=rhs)

        def partial_gradient(mu): 
            return np.concatenate((partial_grad_func_noGamma(mu), partial_grad_func_Gamma(mu)), axis=0)
    
    class TerminationException(Exception):
        pass
    
    #Terminate because we are close to the boundary of the TR
    def custom_termination(x,alpha=alpha, X_train=X_train, current_rad=current_rad):
        if TR_parameters['gamma_adaptive']: 
            alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=x[-1]), rhs)
        
        if model.RKHS_explicit:
            if TR_parameters['gamma_adaptive']:
                if (kernel.powerFuncSingle(x[:-1], X_train[:-1, :], newGamma=x[-1]) * model.compute_RKHS_norm(x) >= np.abs(kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1])) * TR_parameters['beta_2'] * current_rad):
                    return True
            else: 
                if (kernel.powerFuncSingle(x, X_train[:-1, :]) * model.compute_RKHS_norm(x, kernel) >= np.abs(kernel.evalFunc(x, X_train[:-1,:], alpha)) * TR_parameters['beta_2'] * current_rad):
                    return True
        else: 
            if TR_parameters['gamma_adaptive']:
                if (kernel.powerFuncSingle(x[:-1], X_train[:-1, :], newGamma=x[-1]) * kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs, newGamma=x[-1]) >= np.abs(kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1])) * TR_parameters['beta_2'] * current_rad):
                    return True
            else: 
                if (kernel.powerFuncSingle(x, X_train[:-1, :]) * kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs) >= np.abs(kernel.evalFunc(x, X_train[:-1,:], alpha)) * TR_parameters['beta_2'] * current_rad):
                    return True
        
        return False

    last_iteration_data = None
    iteration_counter   = 1

    def callback(x, kernel=kernel, alpha=alpha, X_train=X_train, rhs=rhs):
        nonlocal last_iteration_data, iteration_counter

        if TR_parameters['gamma_adaptive']:
            alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=x[-1]), rhs)
            grad = kernel.evalGrad(x[:-1], X_train[:-1, :], alpha, newGamma=x[-1])
            fun_val = kernel.evalFunc(x[:-1], X_train[:-1, :], alpha, newGamma=x[-1])
        else: 
            grad = kernel.evalGrad(x, X_train[:-1, :], alpha)
            fun_val = kernel.evalFunc(x, X_train[:-1, :], alpha)

        last_iteration_data = {
            'x': x.copy(),
            'jac': grad.copy(),
            'fun': fun_val.copy(),
            'nit': iteration_counter,
            'success': True
        }

        iteration_counter += 1

        if custom_termination(x):
            raise TerminationException("Custom termination criteria met.")

    def penalized_objective(x, kernel=kernel, X_train=X_train,alpha=alpha, penalty_factor=1000, rad=current_rad):
        if TR_parameters['gamma_adaptive']: 
            alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :], newGamma=x[-1]), rhs)

        if model.RKHS_explicit:
            if TR_parameters['gamma_adaptive']:
                return kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1]) + penalty_factor*np.abs(min(0,  (rad * np.abs(kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1])) - (kernel.powerFuncSingle(x[:-1], X_train[:-1, :], newGamma=x[-1]) * model.compute_RKHS_norm(x)))))
            else: 
                return kernel.evalFunc(x, X_train[:-1,:], alpha) + penalty_factor*np.abs(min(0,  (rad * np.abs(kernel.evalFunc(x, X_train[:-1,:], alpha)) - (kernel.powerFuncSingle(x, X_train[:-1, :]) * model.compute_RKHS_norm(x, kernel)))))

        else: 
            if TR_parameters['gamma_adaptive']:
                return kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1]) + penalty_factor*np.abs(min(0,  (rad * np.abs(kernel.evalFunc(x[:-1], X_train[:-1,:], alpha, newGamma=x[-1])) - (kernel.powerFuncSingle(x[:-1], X_train[:-1, :], newGamma=x[-1]) * kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs, newGamma=x[-1])))))
            else:
                return kernel.evalFunc(x, X_train[:-1,:], alpha) + penalty_factor*np.abs(min(0,  (rad * np.abs(kernel.evalFunc(x, X_train[:-1,:], alpha)) - (kernel.powerFuncSingle(x, X_train[:-1, :]) * kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs)))))

    
    if TR_parameters['gamma_adaptive']: 
        try:
            if dim == 1: 
                ranges = (-2, 2.0)
                ranges_gamma = (0.725, 100.0)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=(ranges, ranges_gamma), jac=partial_gradient, options = {'maxiter': 1, 'disp': False})

            elif dim == 2: 
                ranges = (0.5, np.pi)
                ranges_gamma = (0.05, 30)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=(ranges, ranges, ranges_gamma), jac=partial_gradient, options = {'maxiter': 1, 'disp': False})

            elif dim == 12: 
                ranges_0 = (0.05, 0.2)
                ranges_1 = (0, 100)
                ranges_2 = (0.025, 0.1)
                ranges_gamma = (0.001, 100)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=(ranges_0, ranges_0, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_2, ranges_2, ranges_2, ranges_gamma), jac=partial_gradient, options = {'maxiter': 1, 'disp': False})

            else: 
                raise NotImplementedError
            
        except TerminationException as e:
            print("Find Cauchy point: ", e)
            result_BFGS_oneiter = last_iteration_data

        try:
            if dim == 1: 
                ranges = (-2, 2.0)
                ranges_gamma = (0.725, 100.0)
                result_BFGS = minimize(penalized_objective,result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=(ranges, ranges_gamma), jac=partial_gradient, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})

            elif dim == 2: 
                ranges = (0.5, np.pi)
                ranges_gamma = (0.05, 30)
                result_BFGS = minimize(penalized_objective, result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=(ranges, ranges, ranges_gamma), jac=partial_gradient, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})

            elif dim == 12: 
                ranges_0 = (0.05, 0.2)
                ranges_1 = (0, 100)
                ranges_2 = (0.025, 0.1)
                ranges_gamma = (0.001, 100)
                result_BFGS = minimize(penalized_objective, result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=(ranges_0, ranges_0, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_2, ranges_2, ranges_2, ranges_gamma), jac=partial_gradient, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})
            
            else: 
                raise NotImplementedError

        except TerminationException as e:
            print("L-BFGS-B minimizer: ", e)
            result_BFGS = last_iteration_data

    else: 
        try:
            if dim == 1: 
                ranges = (-2, 2.0)
                ranges_gamma = (0.725, 100.0)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=[ranges], jac=partial_grad_func_noGamma, options = {'maxiter': 1, 'disp': False})

            elif dim == 2: 
                ranges = (0.5, np.pi)
                ranges_gamma = (0.05, 30)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=(ranges, ranges), jac=partial_grad_func_noGamma, options = {'maxiter': 1, 'disp': False})

            elif dim == 12: 
                ranges_0 = (0.05, 0.2)
                ranges_1 = (0, 100)
                ranges_2 = (0.025, 0.1)
                ranges_gamma = (0.001, 100)
                result_BFGS_oneiter = minimize(penalized_objective, mu_k[:,0], method='L-BFGS-B', bounds=(ranges_0, ranges_0, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_2, ranges_2, ranges_2), jac=partial_grad_func_noGamma, options = {'maxiter': 1, 'disp': False})

            else: 
                raise NotImplementedError
            
        except TerminationException as e:
            print("Find Cauchy point: ", e)
            result_BFGS_oneiter = last_iteration_data

        try:
            if dim == 1: 
                ranges = (-2, 2.0)
                ranges_gamma = (0.725, 100.0)
                result_BFGS = minimize(penalized_objective,result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=[ranges], jac=partial_grad_func_noGamma, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})

            elif dim == 2: 
                ranges = (0.5, np.pi)
                ranges_gamma = (0.05, 30)
                result_BFGS = minimize(penalized_objective, result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=(ranges, ranges), jac=partial_grad_func_noGamma, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})

            elif dim == 12: 
                ranges_0 = (0.05, 0.2)
                ranges_1 = (0, 100)
                ranges_2 = (0.025, 0.1)
                ranges_gamma = (0.001, 100)
                result_BFGS = minimize(penalized_objective, result_BFGS_oneiter['x'], method='L-BFGS-B', bounds=(ranges_0, ranges_0, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_1, ranges_2, ranges_2, ranges_2), jac=partial_grad_func_noGamma, options = {'gtol': TR_parameters['sub_tolerance'], 'maxiter': TR_parameters['max_iterations_subproblem']})
            
            else: 
                raise NotImplementedError
            
        except TerminationException as e:
            print("L-BFGS-B minimizer: ", e)
            result_BFGS = last_iteration_data

    #Access results
    if TR_parameters['gamma_adaptive']:
        mu_kp1 = result_BFGS['x'].reshape(-1,1)
        gradient = result_BFGS['jac'].reshape(-1,1)
    else:
        mu_kp1  = np.atleast_2d(np.r_[result_BFGS['x'], mu_k[-1,0]]).reshape(-1,1)
        gradient = np.atleast_2d(np.r_[result_BFGS['jac'], np.atleast_2d(np.array([0]))]).reshape(-1,1)

    J_kp1   = result_BFGS['fun']
    J_AGC   = result_BFGS_oneiter['fun']
    success  = result_BFGS['success']
    
    return mu_kp1, J_AGC, J_kp1, gradient, success


def tr_Kernel(model, kernel, TR_parameters):
    
    k    = 1
    mu_k = np.atleast_2d(TR_parameters['starting_parameter']).reshape(-1, 1)
    dim  = model.dim

    FOCs            = []
    mu_list = []
    mu_list.append(mu_k[:,0])

    normgrad       = np.inf
    J_diff         = np.inf
    point_rejected = False
    success        = True

    J_FOM_k, grad_J_FOM_k = model.getFuncAndGradient(mu_k[:-1,:])

    X_train           = mu_k
    y_train           = np.zeros((1,1))
    grad_y_train      = np.zeros((dim ,1))
    y_train[0,0]      = J_FOM_k
    grad_y_train[:,0] = grad_J_FOM_k
    gradient          = grad_J_FOM_k

    if TR_parameters['RKHS_norm'] is None:
        if model.RKHS_explicit:
            RKHS_norm                   = model.compute_RKHS_norm(mu_k)
            RKHS_train_values, RKHS_rhs = np.zeros((dim, 1)), np.zeros((dim, 1))
        else:
            RKHS_train_values, RKHS_rhs = computeDataForRKHSNorm(model, TR_parameters)
            RKHS_norm                   = kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs)
    else: 
        RKHS_norm         = TR_parameters['RKHS_norm']
        RKHS_train_values = TR_parameters['RKHS_train_values']
        RKHS_rhs          = TR_parameters['RKHS_rhs']

    alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :]), np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)])
    J_k   = J_FOM_k

    print('\n**************** Getting started with the TR-Algo ***********\n')
    print('Starting value of the functional: {}'.format(J_FOM_k))
    print('Initial parameter: {}'.format(mu_k[:,0]))
    print('Initial gradient: {}'.format(grad_J_FOM_k))
   
    while k <= TR_parameters['max_iterations']:
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < np.finfo(float).eps:
                print('\n TR-radius below machine precision ... stopping')
                break
        else:
            if normgrad < TR_parameters['FOC_tolerance'] or J_diff < TR_parameters['J_tolerance']:
                    print('\n Stopping criteria fulfilled, normgrad = {}, J_diff = {} -> ... stopping'.format(normgrad, J_diff))
                    break

        print("\n *********** starting iteration number {} ***********".format(k))

        rhs = np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)]
        #TODO vllt hier auch einmal alpha ausrechnen, dann spart man sichs für den rest und unten immer?

        print("_________ starting the subproblem __________________")
        mu_kp1, J_AGC, J_kp1, gradient_kp1, success = solve_subproblem_scipyBFGS(model, kernel, alpha, X_train, rhs, mu_k, TR_parameters, RKHS_train_values, RKHS_rhs)
        #mu_kp1, J_AGC, J_kp1, gradient_kp1, success = solve_optimization_subproblem_NewtonMethod(model, alpha, mu_k, TR_parameters, kernel, X_train, rhs, RKHS_train_values, RKHS_rhs)
        #mu_kp1, J_AGC, J_kp1, gradient_kp1, success = optimization_subproblem_BFGS(model, kernel, alpha, X_train, rhs, mu_k, TR_parameters, RKHS_train_values, RKHS_rhs)
        print("_________ done solving the subproblem ______________")

        if not success:
            print("Solving the subproblem failed: Add additional training point and try again")
            X_train, y_train, grad_y_train = create_training_dataset(kernel, mu_kp1, gradient_kp1, model, X_train, y_train, grad_y_train, TR_parameters)

            rhs    = np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)]
            mu_kp1 = np.atleast_2d(X_train[:,-1]).reshape(-1,1)
            J_kp1  = y_train[-1,0]

        
        if model.RKHS_explicit:
            if TR_parameters['gamma_adaptive']:
                estimator_J = model.compute_RKHS_norm(mu_kp1) * kernel.powerFuncSingle(mu_kp1[:-1,:], X_train[:-1,:], newGamma=mu_kp1[-1,0])
            else:
                print(X_train[:-1,:])
                estimator_J = model.compute_RKHS_norm(mu_kp1) * kernel.powerFuncSingle(mu_kp1[:-1,:], X_train[:-1,:])
        else: 
            if TR_parameters['gamma_adaptive']: 
                
                estimator_J = kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs, newGamma=mu_kp1[-1,0]) * kernel.powerFuncSingle(mu_kp1[:-1,:], X_train[:-1,:], newGamma=mu_kp1[-1,0])
            else:
                estimator_J = kernel.getRKHSNorm(RKHS_train_values, RKHS_rhs) * kernel.powerFuncSingle(mu_kp1[:-1,:], X_train[:-1,:])

        if J_kp1 + estimator_J <= J_AGC:
            print("Accepting the new mu {}".format(mu_kp1[:,0]))

            if success:
                print("\nSolving FOM for new interpolation point ...")
                J_FOM_kp1, grad_J_FOM_kp1 = model.getFuncAndGradient(mu_kp1[:-1, :])

                X_train      = np.append(X_train, mu_kp1, axis=1)
                y_train      = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)
                grad_y_train = np.append(grad_y_train, np.atleast_2d(grad_J_FOM_kp1).reshape(-1,1), axis=1)

                X_train, y_train, grad_y_train = remove_far_away_points(X_train, y_train, grad_y_train, mu_kp1, TR_parameters)
                X_train, y_train, grad_y_train = remove_similar_points(X_train, y_train, grad_y_train, kernel, mu_kp1[-1,0])
                
            else: 
                J_FOM_kp1, grad_J_FOM_kp1 = J_kp1, np.atleast_2d(grad_y_train[:,-1]).reshape(-1,1)
        
            print("Updating the kernel model ...\n")
            if TR_parameters['gamma_adaptive']:
                if dim == 1:
                    kernel = kernels.Gauss(gamma=mu_kp1[-1,0])
                elif dim == 2: 
                    #kernel = kernels.InvMulti(gamma=mu_kp1[-1,0])
                    kernel = kernels.QuadMatern(gamma=mu_kp1[-1,0])
                elif dim == 12: 
                    kernel = kernels.QuadWendland(gamma=mu_kp1[-1,0], d=model.dim)
                    #kernel = kernels.QuadMatern(gamma=mu_kp1[-1,0])
                else: 
                    raise NotImplementedError
                 
            alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :]), np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)])

            if len(y_train) >= 2 and abs(y_train[-2] - J_kp1) > np.finfo(float).eps:
                if ((y_train[-2] - y_train[-1])/(y_train[-2] - J_kp1)) >= TR_parameters['rho']:
                        if TR_parameters['radius'] < 1:
                            TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                            print("Enlarging the TR radius to {}".format(TR_parameters['radius']))

            mu_list.append(mu_kp1[0,:])
            
            J_diff   = abs(J_k - J_FOM_kp1) / np.max([abs(J_k), abs(J_FOM_kp1), 1])
            mu_k     = mu_kp1
            J_k      = J_FOM_kp1
            gradient = grad_J_FOM_kp1
            success  = True


        elif J_kp1 - estimator_J > J_AGC:
            print("Rejecting the parameter mu {}".format(mu_kp1[:,0]))
            TR_parameters['radius'] *= TR_parameters['beta_1']
            print("Shrinking the TR radius to {}".format(TR_parameters['radius']))

            #Still use the computed information in the new surrogate model, just dont accept the point
            alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :]), np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)])

            point_rejected = True

        else:
            print("Building new model to check if proposed iterate mu = {} decreases sufficiently.".format(mu_kp1[:,0]))

            if success:
                print("\nSolving FOM for new interpolation points ...")
                J_FOM_kp1, grad_J_FOM_kp1 = model.getFuncAndGradient(mu_kp1[:-1, :])

                X_train      = np.append(X_train, mu_kp1, axis=1)
                y_train      = np.append(y_train, np.atleast_2d(J_FOM_kp1), axis=0)
                grad_y_train = np.append(grad_y_train, np.atleast_2d(grad_J_FOM_kp1).reshape(-1,1), axis=1)

                X_train, y_train, grad_y_train = remove_far_away_points(X_train, y_train, grad_y_train, mu_kp1, TR_parameters)
                X_train, y_train, grad_y_train = remove_similar_points(X_train, y_train, grad_y_train, kernel, mu_kp1[-1,0])
                
            else: 
                J_FOM_kp1, grad_J_FOM_kp1 = J_kp1, np.atleast_2d(grad_y_train[:,-1]).reshape(-1,1)

            if J_kp1 > J_AGC:

                TR_parameters['radius'] *= TR_parameters['beta_1']
                print("Improvement not good enough: Rejecting the point mu = {} and shrinking TR radius to {}".format(mu_kp1[:,0], TR_parameters['radius']))
                
                #Still use the computed information in the new surrogate model, just dont accept the point
                alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :]), np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)])

                point_rejected = True
                
            else:
                print("Improvement good enough: Accpeting the new mu = {}".format(mu_kp1[:,0]))
                print("\nUpdating the kernel model ...\n")
                
                if TR_parameters['gamma_adaptive']:
                    if dim == 1:
                        kernel = kernels.Gauss(gamma=mu_kp1[-1,0])
                    elif dim == 2: 
                        kernel = kernels.QuadMatern(gamma=mu_kp1[-1,0])
                    elif dim == 12: 
                        kernel = kernels.QuadWendland(gamma=mu_kp1[-1,0], d=model.dim)
                    else: 
                        raise NotImplementedError

                alpha = np.linalg.solve(kernel.getGramHermite(X_train[:-1, :], X_train[:-1, :]), np.r_[y_train, grad_y_train.flatten(order='F').reshape(-1,1)])

                if y_train.shape[0] >= 2 and abs(y_train[-2, 0] - J_kp1) > np.finfo(float).eps:
                        if (k-1 != 0) and (y_train[-2, 0] - y_train[-1, 0])/(y_train[-2, 0] - J_kp1) >= TR_parameters['rho']:
                            if TR_parameters['radius'] < 1: #TODO change to max_radius?
                                TR_parameters['radius'] *= 1/(TR_parameters['beta_1'])
                                print("Enlarging the TR radius to {}".format(TR_parameters['radius']))
                
                mu_list.append(mu_kp1[0,:])
                J_diff   = abs(J_k - J_FOM_kp1) / np.max([abs(J_k), abs(J_FOM_kp1), 1])
                mu_k     = mu_kp1
                J_k      = J_FOM_kp1
                gradient = grad_J_FOM_kp1
                success  = True

        mu_box                = mu_k[:-1,:] - gradient.reshape(-1,1)
        first_order_criticity = mu_k[:-1,:] - projection_onto_range(model, mu_box)
        normgrad              = np.linalg.norm(first_order_criticity, ord=np.inf)

        FOCs.append(normgrad)
        print("First order critical condition: {}".format(normgrad))

        if not point_rejected:
            k += 1

    print("\n************************************* \n")

    if k > TR_parameters['max_iterations']:
        print("WARNING: Maximum number of iteration for the TR algorithm reached")
    
    return mu_k, FOCs, J_k, RKHS_norm, RKHS_train_values, RKHS_rhs