#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import cmasher as cmr
import time


# In[4]:


# Path to save figures with planet
paths0 = '/home/vle/AST_Research_Project/multifluid_0ecc_1jm/'
paths05 = '/home/vle/AST_Research_Project/multifluid_005ecc_1jm/'
paths10 = '/home/vle/AST_Research_Project/multifluid_010ecc_1jm/'
paths15 = '/home/vle/AST_Research_Project/multifluid_015ecc_1jm/'
paths20 = '/home/vle/AST_Research_Project/multifluid_020ecc_1jm/'
paths25 = '/home/vle/AST_Research_Project/multifluid_025ecc_1jm/'
paths30 = '/home/vle/AST_Research_Project/multifluid_030ecc_1jm/'

# Path to save figures without planet
paths0_no_planet = '/home/vle/AST_Research_Project/no_planet_0ecc_1jm/'
paths05_no_planet = '/home/vle/AST_Research_Project/no_planet_005ecc_1jm/'
paths10_no_planet = '/home/vle/AST_Research_Project/no_planet_010ecc_1jm/'
paths15_no_planet = '/home/vle/AST_Research_Project/no_planet_015ecc_1jm/'
paths20_no_planet = '/home/vle/AST_Research_Project/no_planet_020ecc_1jm/'
paths25_no_planet = '/home/vle/AST_Research_Project/no_planet_025ecc_1jm/'
paths30_no_planet = '/home/vle/AST_Research_Project/no_planet_030ecc_1jm/'


# In[5]:


path0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
path05 = '/blue/jbae/vle/multifluid_005ecc_1jm/'
path10 = '/blue/jbae/vle/multifluid_010ecc_1jm/'
path15 = '/blue/jbae/vle/multifluid_015ecc_1jm/'
path20 = '/blue/jbae/vle/multifluid_020ecc_1jm/'
path25 = '/blue/jbae/vle/multifluid_025ecc_1jm/'
path30 = '/blue/jbae/vle/multifluid_030ecc_1jm/'


# In[6]:


fp = [path0, path05, path10, path15, path20, path25, path30]
ps = [paths0, paths05, paths10, paths15, paths20, paths25, paths30]
ecc = ['0_ecc', '005_ecc', '010_ecc', '015_ecc', '020_ecc', '025_ecc', '030_ecc']
# particles = [gas, dust1, dust2, dust3, dust4, dust5, dust6, dust7]
titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']


# In[7]:


def make_plot(fpath, path, e, nf):    
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
    
    r = 2.5
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
    plt.close()
    
#     extra_plots(rad_dat, rad, ny, nx, rhog_i, rhod_i, titles)  
        
#     print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")
    
    return rad_dat, rad, ny, nx, rhog_i, rhod_i, titles


# In[8]:


# def movie_plot_1000_0ecc(fpath, path, e):
    
#     for nf in range(548, 1001):        
#         make_plot(fpath, path, e, nf)


# In[ ]:


# movie_plot_1000_0ecc(path0, paths0, '0_ecc')


# In[9]:


def movie_plot_1000(fpath, path, e):
    
    for nf in range(101, 1001):        
        make_plot(fpath, path, e, nf)


# In[10]:


def movie_plot_1000_010ecc(fpath, path, e):
    
    for nf in range(456, 1001):        
        make_plot(fpath, path, e, nf)


# In[11]:


# movie_plot_1000_005ecc(path05, paths05, '005_ecc')


# In[ ]:


movie_plot_1000_010ecc(path10, paths10, '010_ecc')


# In[ ]:


movie_plot_1000(path15, paths15, '015_ecc')


# In[ ]:


movie_plot_1000(path20, paths20, '020_ecc')


# In[ ]:


movie_plot_1000(path25, paths25, '025_ecc')


# In[ ]:


movie_plot_1000(path30, paths30, '030_ecc')


# In[ ]:




