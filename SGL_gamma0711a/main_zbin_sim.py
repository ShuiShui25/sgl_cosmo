from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd, lenstable_cut_by_z
from mcmc_utils import lnprob, lnprobdirect, lnproblinear, lnprob_K_5D, lnprob_K_4D, lnprob_K_2D, lnprob_K_5D_log, lnprob_K_5D_scale
from mcmc_utils import lnprob_K_3D,lnprob_K_3D_sim
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

prior = 'sim'

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.023-0.122.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin01_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')


lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.122-0.222.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin02_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.222-0.321.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin03_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.321-0.421.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin04_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)





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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.421-0.520.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin05_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)





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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')


lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.520-0.620.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin06_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.620-0.719.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin07_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.719-0.819.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin08_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.819-0.918.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin09_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')


lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_0.918-1.017.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin10_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.017-1.117.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin11_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    
lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.117-1.216.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin12_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)





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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.216-1.316.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin13_'+prior]

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)





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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')


lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.316-1.415.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin14_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.415-1.515.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin15_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')
    

lens_table_path = os.path.join(path_project, r'Data/zbins_LSSTsim' , 'output_bin_1.515-1.614.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = 16
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
mcmc_linear=True

#please write first the GP and then ANN
name_model_list = ['ANN']

#this can be None
model_name_out_list = ['ANNsim_zbin16_'+prior]

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
                model=name_model, nwalkers=nwalkers, nsteps = 20000, mode='Koopmans_3D',  x_ini= [2.0,2.17,0.22],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points,
                model_name_out =model_name_out,lnprob_touse= lnprob_K_3D_sim)
    mcmc_K_beta.main()    
    print('Done! \n')





    
