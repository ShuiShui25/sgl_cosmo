#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:51:31 2025

@author: gengs
"""

from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd, lenstable_cut_by_z
from mcmc_utils import  lnprob_K_5D_scale
import numpy as np


#it is also possible to give the wanted lnprob 
#from mcmc_utils import lnprob_K_5D_log, lnprob_K_5D_scale

# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the parent directory of the script
path_project =  os.path.dirname(os.path.dirname(script_path))

print(f'Path in which your output will be saved {path_project}')


# lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable.fits')
lens_table_path = r"/home/astrodust/SG/mass_discrepancy_Ola/data/lens_data_full_new.fits"
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-35.txt'
cut_table = False
mcmc_linear=False

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
name_model_out_list = ['ANN-full']

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP_rec = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = name_model_out_list[S0],force_run=False  )
# GP_rec.main()
# #add_fsolve_table( GP_rec.path_output_table)
# print('Done! \n')

output_path = r"/home/astrodust/SG/mass_discrepancy_Ola/ANN_reconstruction"

#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = output_path)
ANN_rec.main()
print('Done! \n')
