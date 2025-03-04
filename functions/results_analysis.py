from pymor.basic import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from kernel_width_hermite_TR import tr_Kernel


def prepare_data(gamma_list, amount_of_iters):
    """ Creats a dictionary |data| to save relevant information about the optimization algorithm.

    Parameters
    ----------
    gamma_list 
        List of all kernel widths gamma that are used for the optimization. 
    
    Returns 
    -------
    data
        Dictionary |data| to store the results of the optimization algorithm.
    """
    len_gamma = len(gamma_list)
    data = {'FOC': np.zeros((len_gamma,1)), 'J_error': np.zeros((len_gamma,1)), 
           'counter': np.zeros((len_gamma, 1)), 'mu_error':  np.zeros((len_gamma,1)), 'mu_list': np.zeros((amount_of_iters, 3))} #TODO 3 is not dim dependend
    return data

def optimize_all(model, kernel, gamma_list, TR_parameters, amount_of_iters):
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

    data = prepare_data(gamma_list, amount_of_iters)
    save_radius = TR_parameters['radius']

    for j in range(len(gamma_list)):

        TR_parameters['RKHS_norm']         = None
        TR_parameters['RKHS_train_values'] = None
        TR_parameters['RKHS_rhs']          = None
        TR_parameters['kernel_width']      = gamma_list[j]

        for i in range(amount_of_iters):

            np.random.seed(i)       
            mu_k                                = np.random.uniform(0.25, np.pi, size=2)
            mu_k                                = np.append(np.atleast_2d(mu_k), np.atleast_2d(gamma_list[j]), axis=1).T
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
            data['mu_error'][j,0] += np.linalg.norm(mu_k[:-1,:] - np.array([[1.4246656], [3.14159265]]))
            data['FOC'][j,0]      += FOCs[-1]

            if i == 0: 
                data['counter'][j,0]  += (model.fomCounter - TR_parameters['amount_RKHS_FOMs'])
            else: 
                data['counter'][j,0]  += model.fomCounter
                
            data['J_error'][j,0]  += abs((J_k - 2.3917078761)/(2.3917078761))
            data['mu_list'][i,:]   = mu_k.reshape(1,-1)

    return data

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