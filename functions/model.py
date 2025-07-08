import numpy                as np
import abc
from   pymor.basic import *
from   pymor.core.logger import set_log_levels, getLogger
from   pyMORAuxData      import twelve_dim_discretizer  
from   pyMORAuxData      import problems  


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self.fomCounter      = 0
        self.dim             = None
        self.pyMOR           = True
        self.parameter_space = None
        self.RKHS_explicit   = False

    @abc.abstractmethod
    def getFuncAndGradient(self,mu): 
        pass

class Gaussian1D(Model):
    def __init__(self):
        super().__init__()
        self.dim = 1
        self.pyMOR = False
        self.RKHS_explicit = True
        self.parameter_space = [-2, 2]

    def getFuncAndGradient(self, mu):
        mu = np.atleast_2d(mu)
        
        value = (-1) * np.exp(-mu[0,0]**2) + 3 * np.exp(-0.001 * mu[0,0]**2)
        der   =  2*mu[0,0]* np.exp(-mu[0,0]**2) - ((3 * mu[0,0] * np.exp(- 0.001 * mu[0,0]**2)) / 500)
        
        self.fomCounter += 1
        return value, np.atleast_2d(der).reshape(1,)

    def compute_RKHS_norm(self, mu, kernel=None):
        mu = np.atleast_2d(mu).reshape(-1,1)

        if kernel is None:
            kernel_width = mu[-1, 0]
        else: 
            kernel_width = kernel.gamma

        if kernel_width > 1/np.sqrt(2):
            
            enum  = kernel_width * np.sqrt(np.sqrt(2000*kernel_width**2 - 1) * (np.sqrt(1001*kernel_width**2 - 1) - 33541 * (2**(5/2)) * np.sqrt(2*kernel_width**2 - 1)) + 8999989448 * np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1))
            denom = np.sqrt(np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1) * np.sqrt(2000*kernel_width**2 - 1) )

            return enum / denom

        else:
            print("Warning: kernel width below limit, choosing lowest as possible")
            kernel_width = 1/np.sqrt(2) + 1e-8

            enum  = kernel_width * np.sqrt(np.sqrt(2000*kernel_width**2 - 1) * (np.sqrt(1001*kernel_width**2 - 1) - 33541 * (2**(5/2)) * np.sqrt(2*kernel_width**2 - 1)) + 8999989448 * np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1))
            denom = np.sqrt(np.sqrt(2*kernel_width**2 - 1) * np.sqrt(1001*kernel_width**2 - 1) * np.sqrt(2000*kernel_width**2 - 1) )

            return enum / denom

class twoDStuff(Model): 
    def __init__(self):
        super().__init__()
        self.dim = 2

        problem = problems.linear_problem()
        mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
        self.fom, _ = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)

        self.parameter_space = self.fom.parameters.space(0.5, np.pi)

    def getFuncAndGradient(self, mu):
        self.fomCounter += 1
        value_FOM = self.fom.output(mu)[0,0]
        gradient_FOM = self.fom.output_d_mu(self.fom.parameters.parse(mu)).to_numpy().reshape(2,)
        
        return value_FOM, gradient_FOM

class buildingFloor(Model):
    def __init__(self):
        super().__init__()
        self.dim = 12

        set_log_levels({'pymor': 'ERROR',
                        'distributed_adaptive_discretizations': 'DEBUG',
                        'notebook': 'INFO'})

        data_path = 'pyMORAuxData/EXC_data'

        # domain of interest
        bounding_box = [[0,0],[2,1]]
        domain_of_interest = BitmapFunction.from_file('{}/Domain_of_interest.png'.format(data_path), bounding_box=bounding_box, range=[1,0])

        parametric_quantities = {'walls': [1,4,9], 'windows': [], 'doors': [6,7], 'heaters': [1,3,5,6,7,8,9]}
        inactive_quantities = {'removed_walls': [], 'open_windows': [], 'open_doors': [1,2,3,4,5,10], 'active_heaters': []}
        summed_quantities = {'walls': [[1,2,3,7,8],[4,5,6]], 'windows': [], 'doors': [], 'heaters': [[1,2],[3,4],[9,10,11,12]]}

        coefficient_expressions = None

        parameters_in_q = True
        input_dict      = problems.set_input_dict(parametric_quantities, inactive_quantities, coefficient_expressions, summed_quantities, parameters_in_q,
                                        ac=0.5, owc=[0.025,0.1], iwc= [0.025,0.1], idc=[0.05,0.2], wc=[0.0005], ht=[0,100],
                                            owc_c=0.001,     iwc_c= 0.025,     idc_c=0.01,     wc_c=0.05,   ht_c=80)

        parameter_scaling = False
        u_out             = 5

        problem, parameter_scales = problems.EXC_problem(input_dict, summed_quantities, outside_temperature=u_out,
                                                data_path = data_path,parameters_in_q=parameters_in_q,
                                                parameter_scaling=parameter_scaling,
                                                coefficient_expressions=coefficient_expressions)

        u_d     = 18
        mu_d    = None
        sigma_d = 100 
        weights = {'walls': 0.1, 'doors': 1, 'heaters': [0.002,0.002,0.0005,0.0005,0.0005,0.0005,0.004], 'windows': 1, 'state': sigma_d}

        diameter              = np.sqrt(2)/200
        self.fom, data, mu_bar = twelve_dim_discretizer.discretize_quadratic_pdeopt_stationary_cg(problem, diameter, weights, parameter_scales,
                                                                domain_of_interest, desired_temperature=u_d,
                                                                mu_for_u_d=mu_d, mu_for_tikhonov=mu_d,
                                                                parameters_in_q=parameters_in_q, product='fixed_energy')

        self.parameter_space = problem.parameter_space
        print("hier", self.parameter_space.parameters.items())
        

    def getFuncAndGradient(self,mu):  
        self.fomCounter += 1
        mu_mor = self.fom.parameters.parse(mu)
        return self.fom.output_functional_hat(mu_mor), self.fom.output_functional_hat_gradient(mu_mor)
    


