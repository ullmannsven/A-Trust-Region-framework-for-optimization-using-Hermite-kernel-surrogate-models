# A trust-region framework for optimization using Hermite kernel surrogate models
In this repository we provide the code for the paper "A trust-region framework for optimization using Hermite kernel surrogate models" by S. Ullmann, T. Ehring, R. Herkert and B. Haasdonk.

# Organisation of the repository
In "\functions" the following files are stored:
- kernel.py: Implementations of the kernels and its derivatives for the Hermite appraoch. Based on VKOGA https://github.com/GabrieleSantin/VKOGA
- kernel_width_hermite_TR.py: Main file of the project. Contains the Hermite kernel trust-region (HKTR) algorithm (Algorithm 2 in the paper). 
- model.py: Provides classes for the three models used in the numerical Examples Section
- result_analysis.py: Run + Analysis of the output of the HKTR algorithm
- scipy_algos.py: Contains the code to run the L-BFGS-B and trust-constr algorithm from scipy.optimize that appear in Table 2, Table 4 and Table 6 of the paper. 

In "\pyMORAuxData" stores files to setup the 2D and 12D problem, which are based on pyMOR (https://github.com/pymor/pymor). Note that the file for the 12D example are part of the project https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt (and therefore contain more code than required for our experiments)

The files run_*_hktr.py reproduce Table 1, Table 3 and Table 5 in the paper. The files run_*_bfgs.py and run_*_trust_constr.py reproduce the rows for L-BFGS-B and trust-constr in Table 2, Table 4 and Table 6. 