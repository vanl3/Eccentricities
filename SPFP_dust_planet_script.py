#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import cmasher as cmr
import time


# In[2]:


paths0 = '/home/vle/AST Research Project/multifluid_0ecc_1jm/'
paths05 = '/home/vle/AST Research Project/multifluid_005ecc_1jm/'
paths10 = '/home/vle/AST Research Project/multifluid_010ecc_1jm/'
paths15 = '/home/vle/AST Research Project/multifluid_015ecc_1jm/'
paths20 = '/home/vle/AST Research Project/multifluid_020ecc_1jm/'
paths25 = '/home/vle/AST Research Project/multifluid_025ecc_1jm/'
paths30 = '/home/vle/AST Research Project/multifluid_030ecc_1jm/'


# In[3]:


get_ipython().system('pwd')


# In[4]:


path0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
path05 = '/blue/jbae/vle/multifluid_005ecc_1jm/'
path10 = '/blue/jbae/vle/multifluid_010ecc_1jm/'
path15 = '/blue/jbae/vle/multifluid_015ecc_1jm/'
path20 = '/blue/jbae/vle/multifluid_020ecc_1jm/'
path25 = '/blue/jbae/vle/multifluid_025ecc_1jm/'
path30 = '/blue/jbae/vle/multifluid_030ecc_1jm/'


# In[35]:


def extra_plots():
    """
    Produces a plot of Radius as a function of Density at Different Timesteps, Perturbed Density and Azimuthal Velocity, 
    and two plots of Density as a function of Radius
    """
    tstart = time.time()
    
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

    nx = len(phi)
    ny = len(rad)
    
    planet = np.loadtxt(fpath+'/planet0.dat')
    px = planet[:,1]   # index 1 for x pos
    
    rhog_i   = pl.fromfile(fpath+'gasdens0.dat').reshape(ny,nx) 
    rhod_i   = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    rhog_f   = pl.fromfile(fpath+'gasdens100.dat').reshape(ny,nx) 
    rhod1_f  = pl.fromfile(fpath+'dust1dens100.dat').reshape(ny,nx) 
    rhod2_f  = pl.fromfile(fpath+'dust2dens100.dat').reshape(ny,nx) 
    rhod3_f  = pl.fromfile(fpath+'dust3dens100.dat').reshape(ny,nx) 
    rhod4_f  = pl.fromfile(fpath+'dust4dens100.dat').reshape(ny,nx) 
    rhod5_f  = pl.fromfile(fpath+'dust5dens100.dat').reshape(ny,nx) 
    rhod6_f  = pl.fromfile(fpath+'dust6dens100.dat').reshape(ny,nx) 
    rhod7_f  = pl.fromfile(fpath+'dust7dens100.dat').reshape(ny,nx) 
    
    gas_f = rhog_f/rhog_i
    dust1_f = rhod1_f/rhod_i
    dust2_f = rhod2_f/rhod_i
    dust3_f = rhod3_f/rhod_i
    dust4_f = rhod4_f/rhod_i
    dust5_f = rhod5_f/rhod_i
    dust6_f = rhod6_f/rhod_i
    dust7_f = rhod7_f/rhod_i
    
    fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    
    ax.plot(rad, np.mean(rhog_f,axis=1), label='gas')
    ax.plot(rad, np.mean(rhod1_f,axis=1), label='dust1')
    ax.plot(rad, np.mean(rhod2_f,axis=1), label='dust2')
    ax.plot(rad, np.mean(rhod3_f,axis=1), label='dust3')
    ax.plot(rad, np.mean(rhod4_f,axis=1), label='dust4')
    ax.plot(rad, np.mean(rhod5_f,axis=1), label='dust5')
    ax.plot(rad, np.mean(rhod6_f,axis=1), label='dust6')
    ax.plot(rad, np.mean(rhod7_f,axis=1), label='dust7')
    ax.plot(rad, np.mean(rhog_i,axis=1), label='gas initial', ls=':')
    ax.plot(rad, np.mean(rhod_i,axis=1), label='dust initial', ls=':')
    ax.set_title('Rho vs R for Each ecc', fontsize=14)
    ax.set_xlabel('Density', fontsize=13)
    ax.set_ylabel('Radius', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right', bbox_to_anchor=(-0.15, 0.7))     # 0x 0y lower left conner
    
    ax1.plot(rad, gas_f, label='gas')
    ax1.plot(rad, dust1_f, label='dust1')
    ax1.plot(rad, dust2_f, label='dust2')
    ax1.plot(rad, dust3_f, label='dust3')
    ax1.plot(rad, dust4_f, label='dust4')
    ax1.plot(rad, dust5_f, label='dust5')
    ax1.plot(rad, dust6_f, label='dust6')
    ax1.plot(rad, dust7_f, label='dust7')
    ax1.set_title('Yun 1b Plot', fontsize=14)
    ax1.set_xlabel(r'r / r$_p$', fontsize=13)
    ax1.set_ylabel(r'$\langle \sum/\sum_0 \rangle$', fontsize=13)
    ax1.set_xlim(0.5,2.0)
#     ax1.legend(loc='center left', bbox_to_anchor=(2.1, 0))   # change 1 to whatever for closer or further away the plots at x=2, y=0

    plt.show()
    fig.savefig(f'{path}{e}_yun1b.png', transparent=True, dpi=300) # dots per inch 
    
    particles_f = [gas_f, dust1_f, dust2_f, dust3_f, dust4_f, dust5_f, dust6_f, dust7_f]
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    for i, title in zip(particles_f, titles):
        dlnr = np.log(rad[1:]) - np.log(rad[:-1])
        dlnrho = np.log(i[1:]) - np.log(i[:-1]) 
        dln = dlnrho[:,1]/dlnr
        np.save(f'{fpath}{title}_dln.npy', dln) # save
        dmax = rad[np.argmax(dln)]
        dmin = rad[np.argmin(dln)]  
        w = dmax - dmin
        wh = w/0.05
        mpth = 10**(-3) / 0.05**3
        
        print(f'{title} max point: {dmax:.2f}')
        print(f'{title} min point: {dmin:.2f}')
        print(f'{title} radius differences: {w:.2f}')
        print(f'{title} w/h : {wh:.2f}')
        print(f'{title} mp/mth: {mpth:.2f}')
        
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        
    for ax, i in zip(axes.flat, titles): 
        dln_plot = np.load(f'{fpath}{i}_dln.npy')
        ax.plot(rad_dat[1:-1], dln_plot, label=f'{e}, t=100')
        ax.plot([px[100], px[100]], [-20, 20], label="planet position")
        ax.axhline(y=0.0, color='k', linestyle=':')
#         ax.set_title('Yun 1c Plot', fontsize=14)
        ax.set_xlabel(r'r / r$_p$', fontsize=13)
#         ax.set_ylabel(r'$dln\langle \sum/\sum_0 \rangle / dlnr$', fontsize=13)
        ax.set_xlim(0.5,1.5)
#         ax1.set_ylim(-30,30)
    fig.text(0.08, 0.5, r'Yun 1c $dln\langle \sum/\sum_0 \rangle / dlnr$', va='center', rotation='vertical', fontsize=14)
    plt.savefig(f'{path}{e}_yun1c.png', transparent=True, dpi=300) # dots per inch  
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[19]:


def make_plot(nf):    
    tstart = time.time()
    
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]

    #phi   = 0.5*(phi[:-1] + phi[1:])
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

    nx = len(phi)
    ny = len(rad)

    P, R = np.meshgrid(phi, rad)
    X = R*np.cos(P)
    Y = R*np.sin(P)
    
    rhog_i   = pl.fromfile(fpath+'gasdens0.dat').reshape(ny,nx) 
    rhod_i   = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    rhog   = pl.fromfile(f'{fpath}gasdens{nf}.dat').reshape(ny,nx) 
    rhod1  = pl.fromfile(f'{fpath}dust1dens{nf}.dat').reshape(ny,nx) 
    rhod2  = pl.fromfile(f'{fpath}dust2dens{nf}.dat').reshape(ny,nx) 
    rhod3  = pl.fromfile(f'{fpath}dust3dens{nf}.dat').reshape(ny,nx) 
    rhod4  = pl.fromfile(f'{fpath}dust4dens{nf}.dat').reshape(ny,nx)
    rhod5  = pl.fromfile(f'{fpath}dust5dens{nf}.dat').reshape(ny,nx) 
    rhod6  = pl.fromfile(f'{fpath}dust6dens{nf}.dat').reshape(ny,nx) 
    rhod7  = pl.fromfile(f'{fpath}dust7dens{nf}.dat').reshape(ny,nx) 

    tempg_i   = pl.fromfile(fpath+'gasenergy0.dat').reshape(ny,nx) 
    tempd_i   = pl.fromfile(fpath+'dust1energy0.dat').reshape(ny,nx) 
    
    tempg_f   = pl.fromfile(fpath+'gasenergy100.dat').reshape(ny,nx) 
    tempd1_f  = pl.fromfile(fpath+'dust1energy100.dat').reshape(ny,nx) 
    tempd2_f  = pl.fromfile(fpath+'dust2energy100.dat').reshape(ny,nx)
    tempd3_f  = pl.fromfile(fpath+'dust3energy100.dat').reshape(ny,nx)
    tempd4_f  = pl.fromfile(fpath+'dust4energy100.dat').reshape(ny,nx)
    tempd5_f  = pl.fromfile(fpath+'dust5energy100.dat').reshape(ny,nx)
    tempd6_f  = pl.fromfile(fpath+'dust6energy100.dat').reshape(ny,nx)
    tempd7_f  = pl.fromfile(fpath+'dust7energy100.dat').reshape(ny,nx)

    tempg  = pl.fromfile(f'{fpath}gasenergy{nf}.dat').reshape(ny,nx) 
    tempd1 = pl.fromfile(f'{fpath}dust1energy{nf}.dat').reshape(ny,nx) 
    tempd2 = pl.fromfile(f'{fpath}dust2energy{nf}.dat').reshape(ny,nx) 
    tempd3 = pl.fromfile(f'{fpath}dust3energy{nf}.dat').reshape(ny,nx) 
    tempd4 = pl.fromfile(f'{fpath}dust4energy{nf}.dat').reshape(ny,nx)
    tempd5 = pl.fromfile(f'{fpath}dust5energy{nf}.dat').reshape(ny,nx)
    tempd6 = pl.fromfile(f'{fpath}dust6energy{nf}.dat').reshape(ny,nx)
    tempd7 = pl.fromfile(f'{fpath}dust7energy{nf}.dat').reshape(ny,nx)
    
    gas = rhog/rhog_i
    dust1 = rhod1/rhod_i
    dust2 = rhod2/rhod_i
    dust3 = rhod3/rhod_i
    dust4 = rhod4/rhod_i
    dust5 = rhod5/rhod_i
    dust6 = rhod6/rhod_i
    dust7 = rhod7/rhod_i
    
    r = 1.8
    vmin = -0.4    # decrease: fanter
    vmax = 1.6     # inclrease: darker
    
    levels = np.linspace(vmin,vmax,100)
    
    particles = [gas, dust1, dust2, dust3, dust4, dust5, dust6, dust7]
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, ax, title in zip(particles, axes.flat, titles):        
        ax.contourf(X, Y, i, levels, extend='both', cmap='gist_heat') # copper or gist_heat
        ax.set_xlim(-r,r)
        ax.set_ylim(-r,r)
        ax.set_title(f'particle: {title}, timestep: {nf}')
    plt.savefig(f'{path}density{nf}.png', transparent=True, dpi=300) # dots per inch  
    
#     extra_plots(rad_dat, rad, ny, nx, rhog_i, rhod_i, titles)  
        
    print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")
    
    return rad_dat, rad, ny, nx, rhog_i, rhod_i, titles


# In[7]:


fp = [path0, path05, path10, path15, path20, path25, path30]
ps = [paths0, paths05, paths10, paths15, paths20, paths25, paths30]
ecc = ['0_ecc', '005_ecc', '010_ecc', '015_ecc', '020_ecc', '025_ecc', '030_ecc']


# In[ ]:


for fpath, path, e in zip(fp, ps, ecc):
    
    for nf in range(0, 101):
        make_plot(nf)


# In[36]:


for fpath, path, e in zip(fp, ps, ecc):
    
    extra_plots() 


# In[ ]:




