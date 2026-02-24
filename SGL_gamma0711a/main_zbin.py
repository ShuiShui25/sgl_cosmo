from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd, lenstable_cut_by_z
from mcmc_utils import lnprob, lnprobdirect, lnproblinear, lnprob_K_5D, lnprob_K_4D, lnprob_K_2D, lnprob_K_5D_log, lnprob_K_5D_scale
from mcmc_utils import lnprob_K_3D
from mcmc_utils import lnprob_K_3D_Guerrini2024, lnprob_K_3D_Bolton2006
# from mcmc_utils import lnprob_K_3D_tri, lnprob_K_3D_Cappellari, lnprob_K_3D_Guerrini




def run_linearfit(table_path, output_folder ):

    mcmc_linear = MCMC(lens_table_path =  table_path ,
                path_project=path_project, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, 
                mode='linear', x_ini=[2.0, 0],
                checkpoint=True, color_points=color_points,
                model_name_out =model_name_out, 
                output_folder = output_folder,
                param_fit = 'bin_center'
                 )
    mcmc_linear.main()
    print('Done! \n')


#it is also possible to give the wanted lnprob 
#from mcmc_utils import lnprob_K_5D_log, lnprob_K_5D_scale

# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the parent directory of the script
path_project =  os.path.dirname(os.path.dirname(script_path))

print(f'Path in which your output will be saved {path_project}')

prior = 'Guerrini2024'

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin01.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin01_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')


lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin02.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin02_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin03.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin03_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin04.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin04_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin05.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin05_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin06.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin06_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin07.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin07_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin08.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin08_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin09.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin09_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable_fullzbin10.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNgood_fullzbin10_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# ## run the GP
# print( '\n ************** running the GP reconstruction ************** \n')
# GP = GP(lens_table_path =lens_table_path, 
#         path_project=path_project, table_CC= table_CC , 
#         output_folder = model_name_out_list[0]  )
# GP.main()
# add_fsolve_table(path_project , GP.output_table)
# print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN_rec = ANN(path_project=path_project, 
        lens_table_path=lens_table_path,
        output_folder = model_name_out_list[0])
ANN_rec.main()

print('Done! \n')

# print('Calculating weighted mean of dd from GP and ANN')

# if wmean:
#     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
#                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
#                 return_table_path=True)


table_list = [ANN_rec.path_output_table ]


for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
    if 'GP' in  name_model :
        color_points = '#d00000'
    if  'ANN' in name_model:
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_Guerrini2024)
    mcmc_K_beta.main()    
    print('Done! \n')

# lens_table_path = os.path.join(path_project, 'Data' , 'SGLTablegood_zlg0.5625.fits')
# nwalkers = 200
# nsteps = 20000
# #if true runs usinf the checkpoint system
# checkpoint = True
# ncpu = None
# wmean = False
# table_CC = 'Hz-34.txt'
# cut_table = False
# mcmc_linear=True

# #please write first the GP and then ANN
# name_model_list = ['ANN']

# #this can be None
# model_name_out_list = ['ANNgood_zlg0.5625_Guerrini2024']

# # shortens the table used for mcmc accordinly to max('zs') of the table_cc
# if cut_table:
#     lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


# # ## run the GP
# # print( '\n ************** running the GP reconstruction ************** \n')
# # GP = GP(lens_table_path =lens_table_path, 
# #         path_project=path_project, table_CC= table_CC , 
# #         output_folder = model_name_out_list[0]  )
# # GP.main()
# # add_fsolve_table(path_project , GP.output_table)
# # print('Done! \n')


# #### run the ann 
# print(' \n  ************** running the ANN reconstruction ************** ' )
# ANN_rec = ANN(path_project=path_project, 
#         lens_table_path=lens_table_path,
#         output_folder = model_name_out_list[0])
# ANN_rec.main()
# add_fsolve_table(path_project , ANN_rec.output_table)
# print('Done! \n')

# # print('Calculating weighted mean of dd from GP and ANN')

# # if wmean:
# #     combined_tab = combined_dd(GP.output_table, ANN_rec.output_table, 
# #                 output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
# #                 return_table_path=True)


# table_list = [ANN_rec.output_table ]


# for name_model, table,model_name_out  in zip(name_model_list, table_list, model_name_out_list) :
#     if 'GP' in  name_model :
#         color_points = '#d00000'
#     if  'ANN' in name_model:
#         color_points = 'royalblue'
#     elif name_model=='wmean':
#         color_points = '#e3b23c'
    
            
#     print(' \n  ************** Koopmans power law 3d  ************** ' ) 
#     mcmc_K_beta = MCMC(lens_table_path = table ,
#                     path_project=path_project,
#                 model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
#                 checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
#                 model_name_out =model_name_out)
#     mcmc_K_beta.main()    
#     print('Done! \n')





    