#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import cmasher as cmr
import time


# In[10]:


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


# In[11]:


path0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
path05 = '/blue/jbae/vle/multifluid_005ecc_1jm/'
path10 = '/blue/jbae/vle/multifluid_010ecc_1jm/'
path15 = '/blue/jbae/vle/multifluid_015ecc_1jm/'
path20 = '/blue/jbae/vle/multifluid_020ecc_1jm/'
path25 = '/blue/jbae/vle/multifluid_025ecc_1jm/'
path30 = '/blue/jbae/vle/multifluid_030ecc_1jm/'


# In[12]:


fp = [path0, path05, path10, path15, path20, path25, path30]
ps = [paths0, paths05, paths10, paths15, paths20, paths25, paths30]
ecc = ['0_ecc', '005_ecc', '010_ecc', '015_ecc', '020_ecc', '025_ecc', '030_ecc']
# particles = [gas, dust1, dust2, dust3, dust4, dust5, dust6, dust7]
titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']


# ![image.png](attachment:e99ba9bb-83ad-42c5-800e-89b74556a96b.png)

# $\mu = GM_{\star} = 1$

# In[13]:


# def make_ecc_plot(fpath, path, title, e):    
#     tstart = time.time()
    
#     phi_dat = np.loadtxt(fpath+'domain_x.dat')
#     rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
#     phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
#     rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])
    
#     nx = len(phi)
#     ny = len(rad)    

#     P, R = np.meshgrid(phi, rad)
#     X = R*np.cos(P)
#     Y = R*np.sin(P)   
    
#     rad2d = np.tile(rad,(nx,1))
#     rad2d = np.swapaxes(rad2d,0,1)
    
#     ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
#     omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
#     titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))

#     for ax, title in zip(axes.flat, titles): 
#         vphi = pl.fromfile(f'{fpath}{title}vx100.dat').reshape(ny,nx)
#         vphi += rad2d * omegap
        
#         vr   = pl.fromfile(f'{fpath}{title}vy100.dat').reshape(ny,nx)  
        
# #         print(f'vphi {e} {title}: {vphi}')
# #         print(f'vr {e} {title}: {vr}')
        
#         e1 = rad2d * vphi**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
#         e2 = rad2d * vr * vphi                 # e2 corresponds to the contribution from the radial velocity of the gas or dust
#         ecc = np.sqrt(e1**2 + e2**2)
#         ecc = ecc.mean(axis=1)
    
#         ax.plot(rad2d, ecc)
#         ax.set_xlabel(r'$r/r_p$', fontsize=14)
#         ax.set_title(f'particle: {title}, timestep: 100')
#         ax.set_xlim(0,4)
#     fig.text(0.08, 0.5, f'{e} Ecentricity', va='center', rotation='vertical', fontsize=14)
#     plt.savefig(f'{path}ecc.png', transparent=True, dpi=300) # dots per inch  

#     print(f'phi shape: {phi.shape} \nrad shape: {rad.shape}')
# #     print(f'vphi shape: {vphi[:,1].shape} \nvr shape: {vr[:,1].shape}')
# #     print(f'vphi: {vphi[:,1]} \nvr: {vr[:,1]}')
#     print(f'rad2d shape: {rad2d.shape}')
#     print(f'{e} = {ecc}')
#     print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return phi, rad, vphi, vr, e1, e2, ecc


# In[6]:


# def make_ecc1_plot(fpath, path, e):    
#     tstart = time.time()
    
#     phi_dat = np.loadtxt(fpath+'domain_x.dat')
#     rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
#     phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
#     rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

#     nx = len(phi)
#     ny = len(rad)

#     P, R = np.meshgrid(phi, rad)
#     X = R*np.cos(P)
#     Y = R*np.sin(P)
    
#     rad2d = np.tile(rad,(nx,1))
#     rad2d = np.swapaxes(rad2d,0,1)
    
#     ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
#     omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
#     vphi = pl.fromfile(f'{fpath}dust1vx100.dat').reshape(ny,nx)
#     vphi += rad2d * omegap
#     vr   = pl.fromfile(f'{fpath}dust1vy100.dat').reshape(ny,nx)    
    
#     e1 = rad2d * vphi - 1
#     e2 = rad2d * vr * vphi
#     ecc = np.sqrt(e1**2 + e2**2)
#     ecc = ecc.mean(axis=1)
    
#     fig, ax = plt.subplots(figsize=(6,5))
#     ax.plot(rad, vphi.mean(axis=1))
#     ax.set_title(f'{e}', fontsize=14)
#     ax.set_xlabel(r'$r/r_p$', fontsize=14)
#     ax.set_ylabel('vphi mean', fontsize=14)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
# #     plt.savefig(f'{path}ecc.png', transparent=True, dpi=300) # dots per inch 
    
# #     print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return phi, rad, vphi, vr, e1, e2, ecc


# In[7]:


# for fpath, path, e in zip(fp, ps, ecc):
    
#     make_ecc1_plot(fpath, path, e)


# In[7]:


# for fpath, path, title, e in zip(fp, ps, titles, ecc):
    
#     make_ecc_plot(fpath, path, title, e)


# In[28]:


# fpath0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
# path0 = '/home/vle/AST_Research_Project/multifluid_0ecc_1jm/'

# phi_dat = np.loadtxt(fpath0+'domain_x.dat')
# rad_dat  = np.loadtxt(fpath0+'domain_y.dat')[3:-3]

# phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
# rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

# nx = len(phi)
# ny = len(rad)    
# P, R = np.meshgrid(phi, rad)
# X = R*np.cos(P)
# Y = R*np.sin(P)   

# rad2d = np.tile(rad,(nx,1))
# rad2d = np.swapaxes(rad2d,0,1)

# ind = np.where(np.loadtxt(f'{fpath0}planet0.dat')[:,0] == 1000)[0][-1]
# omegap = np.loadtxt(f'{fpath0}planet0.dat')[ind][-1]

# e1 = []
# e2 = []
# ecc = []
# t = list(range(0, 11))

# for i in range(0, 11):
#     vphig = pl.fromfile(f'{fpath0}gasvx{i}.dat').reshape(ny,nx)
#     vphig+= rad2d * omegap
#     vrg    = pl.fromfile(f'{fpath0}gasvy{i}.dat').reshape(ny,nx)  
    
#     e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
#     e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
#     eccg = np.sqrt(e1g**2 + e2g**2)
#     e1g = e1g.mean(axis=1)
#     e2g = e2g.mean(axis=1)
#     eccg = eccg.mean(axis=1)
#     e1.append(e1g.max())
#     e2.append(e2g.max())
#     ecc.append(eccg.max())
    
# np.save(f'{fpath0}e1dat.npy', e1) # save
# np.save(f'{fpath0}e2dat.npy', e1) # save
# np.save(f'{fpath0}ecc_dat.npy', e1) # save
    
# print(f'e1: {e1}, \ne2: {e2}, \necc: {ecc}')
    
# fig, (ax, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(8, 24), sharex=True)

# ax.plot(t, e1)
# ax.set_xlabel('Time', fontsize=15)
# ax.set_ylabel('e1', fontsize=15)
# ax.set_title(f'e1 vs Time of Gas', fontsize=16)
# #       ax.set_xlim(0,4)
# plt.savefig(f'{path0}e1.png', transparent=True, dpi=300) # dots per inch 

# ax1.plot(t, e2)
# ax1.set_xlabel('Time', fontsize=15)
# ax1.set_ylabel('e2', fontsize=15)
# ax1.set_title(f'e2 vs Time of Gas', fontsize=16)
# #       ax.set_xlim(0,4)
# plt.savefig(f'{path0}e2.png', transparent=True, dpi=300) # dots per inch 

# ax2.plot(t, ecc)
# ax2.set_xlabel('Time', fontsize=15)
# ax2.set_ylabel('e1', fontsize=15)
# ax2.set_title(f'ecc vs Time of Gas', fontsize=16)
# #       ax.set_xlim(0,4)
# plt.savefig(f'{path0}eccentricity.png', transparent=True, dpi=300) # dots per inch 

# ax3.plot(t, e1, label='e1')
# ax3.plot(t, e2, label='e2')
# ax3.plot(t, ecc, label='ecc')
# ax3.set_xlabel('Time', fontsize=15)
# ax3.set_ylabel('eccentricity', fontsize=15)
# ax3.set_title(f'e1, e2, and ecc vs Time of Gas', fontsize=16)
# #       ax.set_xlim(0,4)
# plt.savefig(f'{path0}e1_e2_ecc.png', transparent=True, dpi=300) # dots per inch 


# In[7]:


# plt.plot(ecc)
# plt.plot(e1)
# plt.plot(e2)


# In[8]:


def gas_ecc_calculation(fpath, path, e):
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])
    
    nx = len(phi)
    ny = len(rad)    

    P, R = np.meshgrid(phi, rad)
    X = R*np.cos(P)
    Y = R*np.sin(P)   
    
    rad2d = np.tile(rad,(nx,1))
    rad2d = np.swapaxes(rad2d,0,1)
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 1000)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
    e1 = []
    e2 = []
    ecc = []
    t = list(range(0, 1001))
    
    for i in range(0, 1001):
        vphig = pl.fromfile(f'{fpath}gasvx{i}.dat').reshape(ny,nx)
        vphig+= rad2d * omegap
        vrg    = pl.fromfile(f'{fpath}gasvy{i}.dat').reshape(ny,nx)  
        
        e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
        eccg = np.sqrt(e1g**2 + e2g**2)
        e1g = e1g.mean(axis=1)
        e2g = e2g.mean(axis=1)
        eccg = eccg.mean(axis=1)
        e1.append(e1g.max())
        e2.append(e2g.max())
        ecc.append(eccg.max())
        
    np.save(f'{fpath}e1dat.npy', e1) # save
    np.save(f'{fpath}e2dat.npy', e1) # save
    np.save(f'{fpath}ecc_dat.npy', e1) # save
        
    fig, (ax, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(8, 24), sharex=True)
    
    ax.plot(t, e1)
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel('e1', fontsize=15)
    ax.set_title(f'e1 vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}e1.png', transparent=True, dpi=300) # dots per inch 
    
    ax1.plot(t, e2)
    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('e2', fontsize=15)
    ax1.set_title(f'e2 vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}e2.png', transparent=True, dpi=300) # dots per inch 
    
    ax2.plot(t, ecc)
    ax2.set_xlabel('Time', fontsize=15)
    ax2.set_ylabel('ecc', fontsize=15)
    ax2.set_title(f'ecc vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}eccentricity.png', transparent=True, dpi=300) # dots per inch 
    
    ax3.plot(t, e1, label='e1')
    ax3.plot(t, e2, label='e2')
    ax3.plot(t, ecc, label='ecc')
    ax3.set_xlabel('Time', fontsize=15)
    ax3.set_ylabel('eccentricity', fontsize=15)
    ax3.set_title(f'e1, e2, and ecc vs Time of Gas', fontsize=16)
    ax3.legend()
    plt.savefig(f'{path}e1_e2_ecc.png', transparent=True, dpi=300) # dots per inch 


# In[ ]:


for fpath, path, e in zip(fp, ps, ecc):
    
    gas_ecc_calculation(fpath, path, e)


# In[ ]:




