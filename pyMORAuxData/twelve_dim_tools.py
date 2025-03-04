# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2020)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np


def plot_functional(opt_fom, steps, ranges):
    first_component_steps = steps
    second_component_steps = steps
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    mu_first_component = np.linspace(ranges[0][0],ranges[0][1],first_component_steps)
    mu_second_component = np.linspace(ranges[1][0],ranges[1][1],second_component_steps)

    x1,y1 = np.meshgrid(mu_first_component,mu_second_component)
    func_ = np.zeros([second_component_steps,first_component_steps]) #meshgrid shape the first component as column index

    for i in range(first_component_steps):
        for j in range(second_component_steps):
            mu_ = opt_fom.parameters.parse([x1[j][i],y1[j][i]])
            func_[j][i] = opt_fom.output_functional_hat(mu_)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1, y1, func_, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    #fig.savefig('3d', format='pdf', bbox_inches="tight")

    fig2 = plt.figure()

    number_of_contour_levels= 100
    cont = plt.contour(x1,y1,func_,number_of_contour_levels)

    fig2.colorbar(cont, shrink=0.5, aspect=5)
    # fig2.savefig('2d', format='pdf', bbox_inches="tight")
    return x1, y1, func_


def compute_errors(opt_fom, parameter_space, J_start, J_opt,
                   mu_start, mu_opt, mus, Js, times, tictoc, FOC):
    mu_error = [np.linalg.norm(mu_start.to_numpy() - mu_opt)]
    J_error = [J_start - J_opt]
    for mu_i in mus[1:]: # the first entry is mu_start
        if isinstance(mu_i,dict):
            mu_error.append(np.linalg.norm(mu_i.to_numpy() - mu_opt))
        else:
            mu_error.append(np.linalg.norm(mu_i - mu_opt))

    i = 1 if (len(Js) >= len(mus)) else 0
    for Ji in Js[i:]: # the first entry is J_start
        J_error.append(np.abs(Ji - J_opt))
    times_full = [tictoc]
    for tim in times:
        times_full.append(tim + tictoc)

    if len(FOC)!= len(times_full):
        print("Computing only the initial FOC")
        gradient = opt_fom.output_functional_hat_gradient(mu_start)
        mu_box = opt_fom.parameters.parse(mu_start.to_numpy()-gradient)
        from pdeopt.TR import projection_onto_range
        first_order_criticity = mu_start.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)
        FOCs= [normgrad]
        FOCs.extend(FOC)
    else:
        FOCs = FOC

    if len(J_error) > len(times_full):
        # this happens sometimes in the optional enrichment. For this we need to compute the last J error
        # the last entry is zero and only there to detect this case
        assert not Js[-1]
        J_error.pop(-1)
        J_error.pop(-1)
        J_error.append(np.abs(J_opt-Js[-2]))
    return times_full, J_error, mu_error, FOCs


import scipy
def compute_eigvals(A,B):
    print('WARNING: THIS MIGHT BE VERY EXPENSIVE')
    return scipy.sparse.linalg.eigsh(A, M=B, return_eigenvectors=False)

import csv
def save_data(directory, times, J_error, n, mu_error=None, FOC=None, additional_data=None):
    with open('{}/error_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in J_error:
            writer.writerow([val])
    with open('{}/times_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in times:
            writer.writerow([val])
    if mu_error is not None:
        with open('{}/mu_error_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in mu_error:
                writer.writerow([val])
    if FOC is not None:
        with open('{}/FOC_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in FOC:
                writer.writerow([val])
    if additional_data:
        for key in additional_data.keys():
            if not key == "opt_rom":
                with open('{}/{}_{}.txt'.format(directory, key, n), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for val in additional_data[key]:
                        writer.writerow([val])


def get_data(directory, n, mu_error_=False, mu_est_=False, FOC=False, j_list=True):
    J_error = []
    times = []
    mu_error = []
    mu_time = []
    mu_est = []
    FOC_ = []
    j_list_ = []
    if mu_error_ is True:
        f = open('{}mu_error_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_error.append(float(val[0]))
    if FOC is True:
        f = open('{}FOC_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            FOC_.append(float(val[0]))
    if j_list is True:
        f = open('{}j_list_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            j_list_.append(float(val[0]))
    if mu_est_ is True:
        f = open('{}mu_est_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_est.append(float(val[0]))
        f = open('{}mu_time_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_time.append(float(val[0]))
    f = open('{}error_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        J_error.append(abs(float(val[0])))
    f = open('{}times_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        times.append(float(val[0]))
    if mu_error_:
        if mu_est_:
            if FOC:
                return times, J_error, mu_error, mu_time, mu_est, FOC_, j_list_
            else:
                return times, J_error, mu_error, mu_time, mu_est, j_list_
        else:
            if FOC:
                return times, J_error, mu_error, FOC_, j_list_
            else:
                return times, J_error, mu_error, j_list_
    else:
        if FOC:
            return times, J_error, FOC_, j_list_
        else:
            return times, J_error, j_list

def truncated_conj_grad(A_func,b,x_0=None,tol=10e-6, maxiter = None, atol = None):
    if x_0 is None:
        x_0 = np.zeros(b.size)
    if atol is None:
        atol = tol
    if maxiter is None:
        maxiter = 10*b.size

    test = A_func(x_0)
    if len(test) == len(b):
        def action(x):
                return A_func(x)
    else:
        print('wrong input for A in the CG method')
        return

    #define r_0, note that test= action(x_0)
    r_k = b-test
    #defin p_0
    p_k = r_k
    count = 0
    #define x_0
    x_k = x_0
    #cause we need the norm more often than one time, we save it
    tmp_r_k_norm = np.linalg.norm(r_k)
    norm_b = np.linalg.norm(b)
    while count < maxiter and tmp_r_k_norm > max(tol*norm_b,atol):
        #save the matrix vector product
        #print(tmp_r_k_norm)
        tmp = action(p_k)
        p_kxtmp = np.dot(p_k,tmp)
        #check if p_k is a descent direction, otherwise terminate
        if p_kxtmp<= 1.e-10*(np.linalg.norm(p_k))**2:
            print("CG truncated at iteration: {} with residual: {:.5f}, because p_k is not a descent direction".format(count,tmp_r_k_norm))
            if count>0:
                return x_k, count, tmp_r_k_norm, 0
            else:
                return x_k, count, tmp_r_k_norm, 1
        else:
            #calculate alpha_k
            alpha_k = ((tmp_r_k_norm)**2)/(p_kxtmp)
            #calculate x_k+1
            x_k = x_k + alpha_k*p_k
            #calculate r_k+1
            r_k = r_k - alpha_k*tmp
            #save the new norm of r_k+1
            tmp_r_k1 = np.linalg.norm(r_k)
            #calculate beta_k
            beta_k = (tmp_r_k1)**2/(tmp_r_k_norm)**2
            tmp_r_k_norm = tmp_r_k1
            #calculate p_k+1
            p_k = r_k + beta_k*p_k
            count += 1

    if count >= maxiter:
        print("Maximum number of iteration for CG reached, residual= {}".format(tmp_r_k_norm))
        return x_k,count, tmp_r_k_norm, 1
    return x_k, count, tmp_r_k_norm, 0

def truncated_stabilzed_biconj_grad(A_func,b,x_0=None,tol=1e-12, maxiter = None, atol = None):
    if x_0 is None:
        x_0 = np.zeros(b.size)
    if atol is None:
        atol = tol
    if maxiter is None:
        maxiter = b.size*10

    test = A_func(x_0)
    if len(test) == len(b):
        def action(x):
                return A_func(x)
    else:
        print('wrong input for A in the BICGstab method')
        return

    #define r_0, note that test= action(x_0)
    r_k = b-test
    #for r_0_hat we can choose any random vector which is not orthogonal to r_0, for simplicity we take r_0 itself
    r_0_hat = r_k
    #define p_0
    p_k = r_k
    count = 0
    #define x_0
    x_k = x_0
    #cause we need the norm more often than one time, we save it
    tmp_r_k_norm = np.linalg.norm(r_k)
    norm_b = np.linalg.norm(b)
    tmp_r_k_r_0_hat = np.dot(r_k,r_0_hat)
    while count < maxiter and tmp_r_k_norm > max(tol*norm_b,atol):
        #save the matrix vector product
        #print(tmp_r_k_norm)
        Ap_k = action(p_k)
        #check if p_k is a descent direction, otherwise terminate
        if np.dot(p_k,Ap_k)<= 1.e-10*(np.linalg.norm(p_k))**2:
            print("BICGstab truncated at iteration: {} with residual: {:.5f}, because (p_k)'Ap_k <= 0.0".format(count,tmp_r_k_norm))
            if count>0:
                    return x_k, count, tmp_r_k_norm, 0
            else:
                return x_k, count, tmp_r_k_norm, 1
        else:
            #calculate alpha_k
            alpha_k = tmp_r_k_r_0_hat/(np.dot(r_0_hat,Ap_k))
            #calculate s_k
            s_k = r_k-alpha_k*Ap_k
            As_k = action(s_k)
            if np.dot(s_k,As_k)<=1.e-10*(np.linalg.norm(s_k))**2:
                print("BICGstab truncated at iteration: {} with residual: {:.5f}, because (s_k)'As_k <= 0.0".format(count,tmp_r_k_norm))
                if count>0:
                    return x_k, count, tmp_r_k_norm, 0
                else:
                    return x_k, count, tmp_r_k_norm, 1
            else:
                #calculate omega_k
                omega_k = np.dot(As_k,s_k)/np.dot(As_k,As_k)
                #calculate x_k+1
                x_k = x_k + alpha_k*p_k + omega_k*s_k
                #calculate r_k+1
                r_k = s_k - omega_k*As_k
                #save the new norm of r_k+1
                tmp_r_k1 = np.linalg.norm(r_k)
                #save the product r_k+1, r_0_hat
                tmp_r_k1_r_0_hat = np.dot(r_k,r_0_hat)
                #calculate beta_k
                beta_k = (alpha_k/omega_k)*(tmp_r_k1_r_0_hat)/(tmp_r_k_r_0_hat)
                #update the quantities need in the next loop
                tmp_r_k_norm = tmp_r_k1
                tmp_r_k_r_0_hat = tmp_r_k1_r_0_hat
                #calculate p_k+1
                p_k = r_k + beta_k*(p_k-omega_k*Ap_k)
                count += 1

    if count >= maxiter:
        print("Maximum number of iteration for CG reached, residual= {}".format(tmp_r_k_norm))
        return x_k,count, tmp_r_k_norm, 1
    return x_k, count, tmp_r_k_norm, 0


def truncated_conj_grad_Steihaug(A_func, b, TR_radius, x_0=None, tol=10e-6, maxiter=None, atol=None):
    if x_0 is None:
        x_0 = np.zeros(b.size)
    if atol is None:
        atol = tol
    if maxiter is None:
        maxiter = 10 * b.size

    test = A_func(x_0)
    if len(test) == len(b):
        def action(x):
            return A_func(x)
    else:
        print('wrong input for A in the CG method')
        return

    # define r_0, note that test= action(x_0)
    r_k = b - test
    # defin p_0
    p_k = r_k
    count = 0
    # define x_0
    x_k = x_0
    # cause we need the norm more often than one time, we save it
    tmp_r_k_norm = np.linalg.norm(r_k)
    norm_b = np.linalg.norm(b)
    while count < maxiter and tmp_r_k_norm > max(tol * norm_b, atol):
        # save the matrix vector product
        # print(tmp_r_k_norm)
        tmp = action(p_k)
        p_kxtmp = np.dot(p_k, tmp)
        # check if p_k is a descent direction, otherwise terminate
        if p_kxtmp <= 1.e-10 * (np.linalg.norm(p_k)) ** 2:
            print(
                "CG-Steihaug truncated at iteration: {} with residual: {:.5f}, because H_k is not pos def".format(count,
                                                                                                                 tmp_r_k_norm))
            #b = x_k.dot(p_k)
            #a = p_k.dot(p_k)
            #c = x_k.dot(x_k)-TR_radius**2
            #alpha_k = (b+np.sqrt(b**2-a*c))/a

            ### CHECK
            #if np.abs(np.linalg.norm(x_k+alpha_k*p_k) - TR_radius)/TR_radius <= 1e-6:
            #    print("check steig ok")
            #else:
            #    print("error!!!! in STEIG {}".format(np.linalg.norm(x_k+alpha_k*p_k)))

            #return x_k+alpha_k*p_k, count, tmp_r_k_norm, 0
            return x_k+p_k, count, tmp_r_k_norm, 0


        else:
            # calculate alpha_k
            alpha_k = ((tmp_r_k_norm) ** 2) / (p_kxtmp)
            # calculate x_k+1
            x_k = x_k + alpha_k * p_k
            if np.linalg.norm(x_k,np.inf)>= TR_radius:
                #b = x_k.dot(p_k)
                #a = p_k.dot(p_k)
                #c = x_k.dot(x_k) - TR_radius ** 2
                #alpha_k = (b + np.sqrt(b ** 2 - a * c)) / a
                #return  x_k+alpha_k*p_k, count, tmp_r_k_norm, 0
                return  x_k, count, tmp_r_k_norm, 0
            # calculate r_k+1
            r_k = r_k - alpha_k * tmp
            # save the new norm of r_k+1
            tmp_r_k1 = np.linalg.norm(r_k)
            # calculate beta_k
            beta_k = (tmp_r_k1) ** 2 / (tmp_r_k_norm) ** 2
            tmp_r_k_norm = tmp_r_k1
            # calculate p_k+1
            p_k = r_k + beta_k * p_k
            count += 1

    if count >= maxiter:
        print("Maximum number of iteration for CG reached, residual= {}".format(tmp_r_k_norm))
        return x_k, count, tmp_r_k_norm, 1
    return x_k, count, tmp_r_k_norm, 0
