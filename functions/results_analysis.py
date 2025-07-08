from pymor.basic import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from kernel_width_hermite_TR import tr_Kernel
import kernel as kernels
from tr_steihaug import trust_region_bfgs


def prepare_data(model, amount_of_iters, gamma_list=None):
    """ Creats a dictionary |data| to save relevant information about the optimization algorithm. """

    if gamma_list is not None:
        len_gamma = len(gamma_list)
    else: 
        len_gamma = 1

    data = {'FOC': np.zeros((len_gamma,1)), 'J_error': np.zeros((len_gamma,1)), 
           'counter': np.zeros((len_gamma, 1)), 'mu_error':  np.zeros((len_gamma,1)), 'mu_list': np.zeros((amount_of_iters, model.dim))}
    
    return data

def optimize_all(model, gamma_list, TR_parameters, amount_of_iters):
    """ Repeats the optimization |amount_of_iters| times with different starting parameters. 

    Parameters
    ----------
    fom 
        The full order model that gets evaluated throughout the optimization. 
    parameter_space
        The allowed set of parameters. 
    TR_Kernel 
        The kernel Trust-Region algorithm.
    kernel_name 
        The name of the kernel that is used in the kernel Trust-Region algorithm. 
    gamma_list 
        List of all kernel widths gamma that are used for the optimization.
    TR_parameters
        The list |TR_parameters| which contains all the parameters of the TR algorithm.
    amount_of_iters 
        Amount of times the optimization is done. 

    Returns
    -------
    data
        Dictionary |data| to store results of the optimization algorithm.
    """ 

    data = prepare_data(model, amount_of_iters, gamma_list)
    save_radius = TR_parameters['radius']

    for j in range(len(gamma_list)):
        if model.dim == 1: 
            kernel = kernels.Gauss(gamma=gamma_list[j])
        elif model.dim == 2: 
            kernel = kernels.QuadMatern(gamma=gamma_list[j])
        elif model.dim == 12: 
            kernel = kernels.QuadWendland(gamma=gamma_list[j], d=model.dim)
        
        TR_parameters['RKHS_norm']         = None
        TR_parameters['RKHS_train_values'] = None
        TR_parameters['RKHS_rhs']          = None
        
        for i in range(amount_of_iters):

            if model.dim == 1: 
                np.random.seed(i)
                mu_k = np.random.uniform(-2, 2, size=1)
                mu_k = np.append(np.atleast_2d(mu_k), np.atleast_2d(gamma_list[j]), axis=1).T

            elif model.dim == 2:
                np.random.seed(i)       
                mu_k = np.random.uniform(0.5, np.pi, size=2)
                mu_k = np.append(np.atleast_2d(mu_k), np.atleast_2d(gamma_list[j]), axis=1).T

            elif model.dim == 12:
                #mu_k = np.array([1.12553301e-01, 1.58048674e-01, 1.14374817e-02, 3.02332573e+01, 1.46755891e+01, 9.23385948e+00, 1.86260211e+01, 3.45560727e+01, 3.96767474e+01, 6.54112551e-02, 5.64395886e-02, 7.63914625e-02])
                with new_rng(i):
                    mu_k = model.parameter_space.sample_randomly(1)[0].to_numpy()
                    mu_k = np.append(np.atleast_2d(mu_k), np.atleast_2d(gamma_list[j]), axis=1).T

            else: 
                raise NotImplementedError
            
            TR_parameters['starting_parameter'] = mu_k

            #Reset
            TR_parameters['radius'] = save_radius 
            model.fomCounter        = 0

            #Run TR 
            mu_k, FOCs, J_k, RKHS_norm, RKHS_train_values, RKHS_rhs = tr_Kernel(model, kernel, TR_parameters)

            #Update RKHS things
            TR_parameters['RKHS_norm']         = RKHS_norm
            TR_parameters['RKHS_train_values'] = RKHS_train_values
            TR_parameters['RKHS_rhs']          = RKHS_rhs
            
            #Save Data
            if model.dim == 1: 
                data['mu_error'][j,0] += np.linalg.norm(mu_k[:-1,:] - np.array([[0]]))
                data['J_error'][j,0]  += abs((J_k - 2) / 2)

            elif model.dim == 2:
                data['mu_error'][j,0] += np.linalg.norm(mu_k[:-1,:] - np.array([[1.4246656], [3.14159265]]))
                data['J_error'][j,0]  += abs((J_k - 2.3917078761)/(2.3917078761))

            elif model.dim == 12: 
                data['mu_error'][j,0] += np.linalg.norm(mu_k[:-1,:] - np.array([5.00000000e-02, 5.00000000e-02, 2.23825471e+01, 2.33965046e+01, 4.87034843e+01, 4.93742278e+01, 5.23627225e+01, 5.41155631e+01, 2.35238008e+01, 2.50000000e-02, 2.50000000e-02, 2.50000000e-02]).reshape(-1,1))
                data['J_error'][j,0]  += abs((J_k - 5.813965062384796)/(5.813965062384796))
            else: 
                raise NotImplementedError

            data['FOC'][j,0] += FOCs[-1]

            if model.RKHS_explicit:
                data['counter'][j,0] += model.fomCounter
            else: 
                if i == 0: 
                    data['counter'][j,0]  += (model.fomCounter - 10) #TODO
                else: 
                    data['counter'][j,0]  += model.fomCounter
        
            data['mu_list'][i,:]   = mu_k[:-1, 0]

    return data

# def optimize_fom_tr(model, TR_parameters, amount_of_iters):
#     """ Repeats the optimization |amount_of_iters| times with different starting parameters. 

#     Parameters
#     ----------
#     fom 
#         The full order model that gets evaluated throughout the optimization. 
#     parameter_space
#         The allowed set of parameters. 
#     TR_Kernel 
#         The kernel Trust-Region algorithm.
#     kernel_name 
#         The name of the kernel that is used in the kernel Trust-Region algorithm. 
#     gamma_list 
#         List of all kernel widths gamma that are used for the optimization.
#     TR_parameters
#         The list |TR_parameters| which contains all the parameters of the TR algorithm.
#     amount_of_iters 
#         Amount of times the optimization is done. 

#     Returns
#     -------
#     data
#         Dictionary |data| to store results of the optimization algorithm.
#     """ 

#     data = prepare_data(model, amount_of_iters)
#     save_radius = TR_parameters['radius']


#     for i in range(amount_of_iters):

#         if model.dim == 2:
#             np.random.seed(i)       
#             mu_k = np.random.uniform(0.25, np.pi, size=2).reshape(-1,1)

#         elif model.dim == 12:
#             #mu_k = np.array([1.12553301e-01, 1.58048674e-01, 1.14374817e-02, 3.02332573e+01, 1.46755891e+01, 9.23385948e+00, 1.86260211e+01, 3.45560727e+01, 3.96767474e+01, 6.54112551e-02, 5.64395886e-02, 7.63914625e-02]).reshape(-1,1)
#             with new_rng(i):
#                 mu_k = model.parameter_space.sample_randomly(1)[0].to_numpy().reshape(-1,1)
#         else: 
#             raise NotImplementedError
        
        
#         TR_parameters['starting_parameter'] = mu_k

#         #Reset
#         TR_parameters['radius'] = save_radius 
#         model.fomCounter        = 0

#         #Run TR 
#         mu_k, FOC, J_k  = trust_region_bfgs(model, TR_parameters)

#         if model.dim == 2:
#                 data['mu_error'][0,0] += np.linalg.norm(mu_k - np.array([[1.4246656], [3.14159265]]))
#                 data['J_error'][0,0]  += abs((J_k - 2.3917078761)/(2.3917078761))

#         elif model.dim == 12: 
#             data['mu_error'][0,0] += np.linalg.norm(mu_k - np.array([5.00000000e-02, 5.00000000e-02, 2.23825471e+01, 2.33965046e+01, 4.87034843e+01, 4.93742278e+01, 5.23627225e+01, 5.41155631e+01, 2.35238008e+01, 2.50000000e-02, 2.50000000e-02, 2.50000000e-02]).reshape(-1,1))
#             data['J_error'][0,0]  += abs((J_k - 5.813965062384796)/(5.813965062384796))
#         else: 
#             raise NotImplementedError

#         data['FOC'][0,0]     += FOC
#         data['counter'][0,0] += model.fomCounter
#         data['mu_list'][i,:]  = mu_k[:, 0]

#     return data


def report_kernel_TR(data, gamma_list, amount_of_iters):
    """Reports the results of the optimization algorithm. 

    Parameters
    ----------
    data
        Dictionary |data| to store results of the optimization algorithm.
    gamma_list 
        List of all kernel widths gamma that are used for the optimization.
    amount of iters
        Amount of times the optimization is done. 
    """
    data_new = {
        'gamma': np.array(gamma_list).T,  
        'avg. FOM evals.': data['counter'][:,0]/amount_of_iters,
        'avg. error in mu': data['mu_error'][:,0]/amount_of_iters,
        'avg. FOC condition': data['FOC'][:,0]/amount_of_iters,
        'avg. error in J': data['J_error'][:,0]/amount_of_iters
    }

    df = pd.DataFrame(data_new)
    print(df)


# def report_fom_tr(data, amount_of_iters):
#     data_new = {  
#         'avg. FOM evals.': data['counter'][:,0]/amount_of_iters,
#         'avg. error in mu': data['mu_error'][:,0]/amount_of_iters,
#         'avg. FOC condition': data['FOC'][:,0]/amount_of_iters,
#         'avg. error in J': data['J_error'][:,0]/amount_of_iters
#     }
#     df = pd.DataFrame(data_new)
#     print(df)
                  
#######################################################################################################################

def draw_TR_advanced(TR_plot_matrix, mu_list): 
    """ Plots the TR and the iterates in the advanced formulation of the kernel TR algorithm.

    Parameters
    ----------
    TR_plot_matrix 
        Dictionary that stores information about the TR in the advanved formulation. 
    mu_list 
        List of mus computed throughout the algorithm. 
    """
    fig, ax = plt.subplots()
    for i in range(len(mu_list)-1):
        array = 'array{}'.format(i)
        hull = ConvexHull(TR_plot_matrix[array])
        TR_plot_matrix[array] = TR_plot_matrix[array][hull.vertices]
        x = TR_plot_matrix[array][:,0]
        y = TR_plot_matrix[array][:,1]
        ax.plot(mu_list[i][0], mu_list[i][1], 'x', color='red')
        ax.fill(x,y, color='blue', alpha=0.15)
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mu_1$')
    ax.set_ylabel(r'$\mu_2$')
    plt.show(block=True)

def draw_TR_standard(list_delta, mu_list):
    """ Plots the TR and the iterates in the standard formulation of the kernel TR algorithm.

    Parameters
    ----------
    list_delta
        List of the TR radius delta throughout the algorithm. 
    mu_list 
        List of mus computed throughout the algorithm. 
    """
    theta = np.linspace(0, 2*np.pi, 500)
    fig, ax = plt.subplots()
    for i in range(len(mu_list)-1):
        circle = plt.Circle((mu_list[i][0], mu_list[i][1]), list_delta[f"{i}"][-1], fill=False, color='blue')
        plt.gca().add_patch(circle)
        x = mu_list[i][0] + list_delta[f"{i}"][-1]*np.cos(theta)
        y = mu_list[i][1] + list_delta[f"{i}"][-1]*np.sin(theta)
        ax.fill(x,y, color='blue', alpha=0.15)
        ax.plot(mu_list[i][0], mu_list[i][1], 'x', color='red')
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0,np.pi)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mu_1$')
    ax.set_ylabel(r'$\mu_2$')
    plt.show(block=True)