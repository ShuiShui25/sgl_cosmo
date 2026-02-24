# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:27:27 2024

@author: poilo
"""

from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd, lenstable_cut_by_z
from mcmc_utils import  lnprob_K_5D_scale
import numpy as np

seed = 42
np.random.seed(seed)



#it is also possible to give the wanted lnprob 
#from mcmc_utils import lnprob_K_5D_log, lnprob_K_5D_scale

# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the parent directory of the script
path_project =  os.path.dirname(os.path.dirname(script_path))

print(f'Path in which your output will be saved {path_project}')


lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable03.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-35.txt'
cut_table = True
mcmc_linear=False

#please write first the GP and then ANN
name_model_list = ['GP']

#this can be None
name_model_out_list = [ 'GP35']

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


## run the GP
print( '\n ************** running the GP reconstruction ************** \n')
GP_rec = GP(lens_table_path =lens_table_path, 
        path_project=path_project, table_CC= table_CC , 
        output_folder = name_model_out_list[0],force_run=False  )
GP_rec.main()
#add_fsolve_table( GP_rec.path_output_table)
print('Done! \n')


# #### run the ann 
# print(' \n  ************** running the ANN reconstruction ************** ' )
# ANN_rec = ANN(path_project=path_project, 
#         lens_table_path=lens_table_path,
#         output_folder = name_model_out_list[0])
# ANN_rec.main()
# print('Done! \n')

if wmean:
    print('Calculating weighted mean of dd from GP and ANN')
    combined_tab = combined_dd(GP_rec.path_output_table, ANN_rec.path_output_table, 
                output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
                return_table_path=True)


table_list = [GP_rec.path_output_table]


if not len(table_list) == len(name_model_out_list) == len(name_model_list):
     raise ValueError('table_list, name_model_out_list, name_model_list need to all have the same lenght')


for name_model, table,model_name_out  in zip(name_model_list, table_list, name_model_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
    print (f' \n  ************** MCMC for {name_model} ************** \n ')
    
    print(f" \n ************** gamma from {table} for every value ************** \n")
    ## run the mcmc 
    mcmc = MCMC(lens_table_path = table , path_project = path_project, 
                mode='1D',
                model=name_model, nwalkers = nwalkers, nsteps = nsteps,
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out)
    mcmc.main()
    print('Done! \n')