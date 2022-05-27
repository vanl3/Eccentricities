#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import cmasher as cmr
import time
import math
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse


# ### 1JM

# In[10]:


# Path to save figures with planet
paths0 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_1jm/'
paths05 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_1jm/'
paths10 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_1jm/'
paths15 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_1jm/'
paths20 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_1jm/'
paths25 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_1jm/'
paths30 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_1jm/'

# Path to save figures without planet
paths0_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_0ecc_1jm/'
paths05_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_005ecc_1jm/'
paths10_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_010ecc_1jm/'
paths15_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_015ecc_1jm/'
paths20_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_020ecc_1jm/'
paths25_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_025ecc_1jm/'
paths30_no_planet = '/blue/jbae/vle/AST_Research_Project/no_planet_030ecc_1jm/'


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


# ### 0.3 JM

# In[13]:


# Path to load and save figures with planet
paths0_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_03jm/'
paths05_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_03jm/'
paths10_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_03jm/'
paths15_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_03jm/'
paths20_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_03jm/'
paths25_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_03jm/'
paths30_03 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_03jm/'

path0_03 = '/blue/jbae/vle/multifluid_0ecc_03jm/'
path05_03 = '/blue/jbae/vle/multifluid_005ecc_03jm/'
path10_03 = '/blue/jbae/vle/multifluid_010ecc_03jm/'
path15_03 = '/blue/jbae/vle/multifluid_015ecc_03jm/'
path20_03 = '/blue/jbae/vle/multifluid_020ecc_03jm/'
path25_03 = '/blue/jbae/vle/multifluid_025ecc_03jm/'
path30_03 = '/blue/jbae/vle/multifluid_030ecc_03jm/'

fp_03 = [path0_03, path05_03, path10_03, path15_03, path20_03, path25_03, path30_03]
ps_03 = [paths0_03, paths05_03, paths10_03, paths15_03, paths20_03, paths25_03, paths30_03]
ecc_03 = ['0_ecc_03jm', '005_ecc_03jm', '010_ecc_03jm', '015_ecc_03jm', '020_ecc_03jm', '025_ecc_03jm', '030_ecc_03jm']


# ### 3 JM

# In[14]:


# Path to load and save figures with planet
paths0_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_3jm/'
paths05_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_3jm/'
paths10_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_3jm/'
paths15_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_3jm/'
paths20_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_3jm/'
paths25_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_3jm/'
paths30_3 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_3jm/'

path0_3 = '/blue/jbae/vle/multifluid_0ecc_3jm/'
path05_3 = '/blue/jbae/vle/multifluid_005ecc_3jm/'
path10_3 = '/blue/jbae/vle/multifluid_010ecc_3jm/'
path15_3 = '/blue/jbae/vle/multifluid_015ecc_3jm/'
path20_3 = '/blue/jbae/vle/multifluid_020ecc_3jm/'
path25_3 = '/blue/jbae/vle/multifluid_025ecc_3jm/'
path30_3 = '/blue/jbae/vle/multifluid_030ecc_3jm/'

fp_3 = [path0_3, path05_3, path10_3, path15_3, path20_3, path25_3, path30_3]
ps_3 = [paths0_3, paths05_3, paths10_3, paths15_3, paths20_3, paths25_3, paths30_3]
ecc_3 = ['0_ecc_3jm', '005_ecc_3jm', '010_ecc_3jm', '015_ecc_3jm', '020_ecc_3jm', '025_ecc_3jm', '030_ecc_3jm']


# In[15]:


# Path to load and save figures with planet
paths0_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_10jm/'
paths05_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_10jm/'
paths10_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_10jm/'
paths15_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_10jm/'
paths20_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_10jm/'
paths25_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_10jm/'
paths30_10 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_10jm/'

path0_10 = '/blue/jbae/vle/multifluid_0ecc_10jm/'
path05_10 = '/blue/jbae/vle/multifluid_005ecc_10jm/'
path10_10 = '/blue/jbae/vle/multifluid_010ecc_10jm/'
path15_10 = '/blue/jbae/vle/multifluid_015ecc_10jm/'
path20_10 = '/blue/jbae/vle/multifluid_020ecc_10jm/'
path25_10 = '/blue/jbae/vle/multifluid_025ecc_10jm/'
path30_10 = '/blue/jbae/vle/multifluid_030ecc_10jm/'

fp_10 = [path0_10, path05_10, path10_10, path15_10, path20_10, path25_10, path30_10]
ps_10 = [paths0_10, paths05_10, paths10_10, paths15_10, paths20_10, paths25_10, paths30_10]
ecc_10 = ['0_ecc_10jm', '005_ecc_10jm', '010_ecc_10jm', '015_ecc_10jm', '020_ecc_10jm', '025_ecc_10jm', '030_ecc_10jm']


# ![image.png](attachment:e99ba9bb-83ad-42c5-800e-89b74556a96b.png)

# $\mu = GM_{\star} = 1$

# In[31]:


def make_ecc_plot(fpath, path, title, e):    
    tstart = time.time()
    
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
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for ax, title in zip(axes.flat, titles): 
        vphi = pl.fromfile(f'{fpath}{title}vx100.dat').reshape(ny,nx)
        vphi += rad2d * omegap
        vr   = pl.fromfile(f'{fpath}{title}vy100.dat').reshape(ny,nx)  
        
#         print(f'vphi {e} {title}: {vphi}')
#         print(f'vr {e} {title}: {vr}')
        
        e1 = rad2d * vphi**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2 = rad2d * vr * vphi                 # e2 corresponds to the contribution from the radial velocity of the gas or dust
        ecc = np.sqrt(e1**2 + e2**2)
        e1 = np.abs(e1).mean(axis=1)
        e2 = np.abs(e2).mean(axis=1)
        ecc = ecc.mean(axis=1)
    
        ax.plot(rad2d, e1, label='e1')
        ax.plot(rad2d, e2, label='e2')
        ax.plot(rad2d, ecc, label='ecc')
        ax.set_xlabel(r'$r/r_p$', fontsize=14)
        ax.set_title(f'particle: {title}, timestep: 100')
        ax.set_xlim(0,4)
#         ax.legend()
    fig.text(0.08, 0.5, f'{e} Ecentricity', va='center', rotation='vertical', fontsize=14)
#     plt.savefig(f'{path}{e}.png', transparent=True, dpi=300) # dots per inch  

#     print(f'phi shape: {phi.shape} \nrad shape: {rad.shape}')
#     print(f'vphi shape: {vphi[:,1].shape} \nvr shape: {vr[:,1].shape}')
#     print(f'vphi: {vphi[:,1]} \nvr: {vr[:,1]}')
#     print(f'rad2d shape: {rad2d.shape}')
#     print(f'{e} = {ecc}')
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return phi, rad, vphi, vr, e1, e2, ecc


# In[32]:


for fpath, path, title, e in zip(fp, ps, titles, ecc):
    
    make_ecc_plot(fpath, path, title, e)


# In[6]:


def make_ecc1_plot(fpath, path, e):    
    tstart = time.time()
    
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
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
    vphi = pl.fromfile(f'{fpath}dust1vx100.dat').reshape(ny,nx)
    vphi += rad2d * omegap
    vr   = pl.fromfile(f'{fpath}dust1vy100.dat').reshape(ny,nx)    
    
    e1 = rad2d * vphi - 1
    e2 = rad2d * vr * vphi
    ecc = np.sqrt(e1**2 + e2**2)
    ecc = ecc.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rad, vphi.mean(axis=1))
    ax.set_title(f'{e}', fontsize=14)
    ax.set_xlabel(r'$r/r_p$', fontsize=14)
    ax.set_ylabel('vphi mean', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
#     plt.savefig(f'{path}ecc.png', transparent=True, dpi=300) # dots per inch 
    
#     print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")
    
    return phi, rad, vphi, vr, e1, e2, ecc


# In[7]:


for fpath, path, e in zip(fp, ps, ecc):
    
    make_ecc1_plot(fpath, path, e)


# In[ ]:


fpath0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
path0 = '/home/vle/AST_Research_Project/multifluid_0ecc_1jm/'

phi_dat = np.loadtxt(fpath0+'domain_x.dat')
rad_dat  = np.loadtxt(fpath0+'domain_y.dat')[3:-3]

phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

nx = len(phi)
ny = len(rad)    
P, R = np.meshgrid(phi, rad)
X = R*np.cos(P)
Y = R*np.sin(P)   

rad2d = np.tile(rad,(nx,1))
rad2d = np.swapaxes(rad2d,0,1)

ind = np.where(np.loadtxt(f'{fpath0}planet0.dat')[:,0] == 1000)[0][-1]
omegap = np.loadtxt(f'{fpath0}planet0.dat')[ind][-1]

e1 = []
e2 = []
ecc = []
t = list(range(0, 11))

for i in range(0, 11):
    vphig = pl.fromfile(f'{fpath0}gasvx{i}.dat').reshape(ny,nx)
    vphig+= rad2d * omegap
    vrg    = pl.fromfile(f'{fpath0}gasvy{i}.dat').reshape(ny,nx)  
    
    e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
    e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
    eccg = np.sqrt(e1g**2 + e2g**2)
    e1g = e1g.mean(axis=1)
    e2g = e2g.mean(axis=1)
    eccg = eccg.mean(axis=1)
    e1.append(e1g.max())
    e2.append(e2g.max())
    ecc.append(eccg.max())
    
np.save(f'{fpath0}e1dat.npy', e1) # save
np.save(f'{fpath0}e2dat.npy', e2) # save
np.save(f'{fpath0}ecc_dat.npy', ecc) # save
    
print(f'e1: {e1}, \ne2: {e2}, \necc: {ecc}')
    
fig, (ax, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(8, 24), sharex=True)

ax.plot(t, e1)
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('e1', fontsize=15)
ax.set_title(f'e1 vs Time of Gas', fontsize=16)
# plt.savefig(f'{path0}e1.png', transparent=True, dpi=300) # dots per inch 

ax1.plot(t, e2)
ax1.set_xlabel('Time', fontsize=15)
ax1.set_ylabel('e2', fontsize=15)
ax1.set_title(f'e2 vs Time of Gas', fontsize=16)
# plt.savefig(f'{path0}e2.png', transparent=True, dpi=300) # dots per inch 

ax2.plot(t, ecc)
ax2.set_xlabel('Time', fontsize=15)
ax2.set_ylabel('ecc', fontsize=15)
ax2.set_title(f'ecc vs Time of Gas', fontsize=16)
# plt.savefig(f'{path0}eccentricity.png', transparent=True, dpi=300) # dots per inch 

ax3.plot(t, e1, label='e1')
ax3.plot(t, e2, label='e2')
ax3.plot(t, ecc, label='ecc')
ax3.set_xlabel('Time', fontsize=15)
ax3.set_ylabel('eccentricity', fontsize=15)
ax3.set_title(f'e1, e2, and ecc vs Time of Gas', fontsize=16)
ax3.legend()
# plt.savefig(f'{path0}e1_e2_ecc.png', transparent=True, dpi=300) # dots per inch 


# In[7]:


plt.plot(ecc)
plt.plot(e1)
plt.plot(e2)


# In[ ]:


def gas_ecc_calculation(fpath, path, e):
    tstart = time.time()
    
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
    
    e1 = []
    e2 = []
    ecc = []
    t = list(range(0, 1001))
    
    for i in range(0, 1001):
        ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == i)[0][-1]
        omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
        
        vphig = pl.fromfile(f'{fpath}gasvx{i}.dat').reshape(ny,nx)
        vphig+= rad2d * omegap
        vrg    = pl.fromfile(f'{fpath}gasvy{i}.dat').reshape(ny,nx)  
        
        e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
        eccg = np.sqrt(e1g**2 + e2g**2)
        
        eccg = eccg.mean(axis=1)
        e1g = np.abs(e1g).mean(axis=1)
        e2g = np.abs(e2g).mean(axis=1) 
        
        e1.append(e1g.max())
        e2.append(e2g.max())
        ecc.append(eccg.max())
        
    np.save(f'{fpath}e1dat.npy', e1) # save
    np.save(f'{fpath}e2dat.npy', e2) # save
    np.save(f'{fpath}ecc_dat.npy', ecc) # save
        
    fig, (ax, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(8, 24), sharex=True)
    
    ax.plot(t, e1)
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel('e1', fontsize=15)
    ax.set_title(f'{e}: e1 vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}e1.png', transparent=True, dpi=300) # dots per inch 
    
    ax1.plot(t, e2)
    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('e2', fontsize=15)
    ax1.set_title(f'{e}: e2 vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}e2.png', transparent=True, dpi=300) # dots per inch 
    
    ax2.plot(t, ecc)
    ax2.set_xlabel('Time', fontsize=15)
    ax2.set_ylabel('ecc', fontsize=15)
    ax2.set_title(f'{e}: ecc vs Time of Gas', fontsize=16)
#     plt.savefig(f'{path}eccentricity.png', transparent=True, dpi=300) # dots per inch 
    
    ax3.plot(t, e1, label='e1')
    ax3.plot(t, e2, label='e2')
    ax3.plot(t, ecc, label='ecc')
    ax3.set_xlabel('Time', fontsize=15)
    ax3.set_ylabel('eccentricity', fontsize=15)
    ax3.set_title(f'{e}: e1, e2, and ecc vs Time of Gas', fontsize=16)
    ax3.legend()
    plt.savefig(f'{path}{e}_e1_e2_ecc.png', transparent=True, dpi=300) # dots per inch 
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[ ]:


for fpath, path, e in zip(fp, ps, ecc):
    
    gas_ecc_calculation(fpath, path, e)


# ![image.png](attachment:0939f937-0237-4590-a415-b89819dd4a44.png)

# In[5]:


def gas_test(fpath, path, e):
    tstart = time.time()
    
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
    
    e1 = []
    e2 = []
    ecc = []
    ecc_mean = []
    t = list(range(0, 1001))
    
    for i in range(0, 1001):
        ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == i)[0][-1]
        omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
        
        vphig  = pl.fromfile(f'{fpath}gasvx{i}.dat').reshape(ny,nx)
        vphig += rad2d * omegap
        vrg    = pl.fromfile(f'{fpath}gasvy{i}.dat').reshape(ny,nx) 
        rhog   = pl.fromfile(f'{fpath}gasdens{i}.dat').reshape(ny,nx)
        
        e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
        eccg = np.sqrt(e1g**2 + e2g**2)
        gas_e = rhog*eccg
        
        e1g = np.abs(e1g).mean(axis=1)
        e2g = np.abs(e2g).mean(axis=1)
        eccg = eccg.mean(axis=1)
        rhog_mean  = rhog.mean(axis=1)
        gas_e_mean = gas_e.mean(axis=1) 
        
        eccg_mean = gas_e_mean / rhog_mean
        
        e1.append(e1g.max())
        e2.append(e2g.max())
        ecc.append(eccg.max())
        ecc_mean.append(eccg_mean.max())
        
    np.save(f'{fpath}e1dat.npy', e1) # save
    np.save(f'{fpath}e2dat.npy', e2) # save
    np.save(f'{fpath}ecc_dat.npy', ecc) # save without density weighted
    np.save(f'{fpath}ecc_mean_dat.npy', ecc_mean) # save with density weighted        
        
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(21,19), sharex=True)
    
    axes[0, 0].plot(t, e1)
    axes[0, 0].set_xlabel('Time', fontsize=15)
    axes[0, 0].set_ylabel('e1', fontsize=15)
    axes[0, 0].set_title(f'{e}: e1 vs Time of Gas', fontsize=16)
    
    axes[0, 1].plot(t, e2)
    axes[0, 1].set_xlabel('Time', fontsize=15)
    axes[0, 1].set_ylabel('e2', fontsize=15)
    axes[0, 1].set_title(f'{e}: e2 vs Time of Gas', fontsize=16)
    
    axes[1, 0].plot(t, ecc)
    axes[1, 0].set_xlabel('Time', fontsize=15)
    axes[1, 0].set_ylabel('ecc without weighted', fontsize=15)
    axes[1, 0].set_title(f'{e}: ecc vs Time without density weighted', fontsize=16)
    
    axes[1, 1].plot(t, ecc_mean)
    axes[1, 1].set_xlabel('Time', fontsize=15)
    axes[1, 1].set_ylabel('ecc weighted', fontsize=15)
    axes[1, 1].set_title(f'{e}: ecc vs Time with density weighted', fontsize=16)
    
    axes[2, 0].plot(t, e1, label='e1')
    axes[2, 0].plot(t, e2, label='e2')
    axes[2, 0].plot(t, ecc, label='ecc')
    axes[2, 0].set_xlabel('Time', fontsize=15)
    axes[2, 0].set_ylabel('eccentricity without weighted', fontsize=15)
    axes[2, 0].set_title(f'{e}: e1, e2, ecc without dens weighted', fontsize=16)
    axes[2, 0].legend()
    
    axes[2, 1].plot(t, e1, label='e1')
    axes[2, 1].plot(t, e2, label='e2')
    axes[2, 1].plot(t, ecc_mean, label='ecc mean')
    axes[2, 1].set_xlabel('Time', fontsize=15)
    axes[2, 1].set_ylabel('eccentricity weighted', fontsize=15)
    axes[2, 1].set_title(f'{e}: e1, e2, ecc with dens weighted', fontsize=16)
    axes[2, 1].legend()
    plt.savefig(f'{path}{e}_e1_e2_ecc_weight.png', transparent=True, dpi=300) # dots per inch 
    
    fig, ax = plt.subplots()
    
    ax.plot(t, ecc, label='ecc')
    ax.plot(t, ecc_mean, label='ecc mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('ecc weighted and unweighted')
    ax.legend()
    plt.savefig(f'{path}{e}_mean_overplot.png', transparent=True, dpi=300) # dots per inch
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[25]:


gas_test(path0, paths0, '0_ecc')


# In[12]:


gas_test(path0, paths0, '0_ecc')


# In[6]:


gas_test(path05, paths05, '005_ecc')


# In[7]:


gas_test(path10, paths10, '010_ecc')


# In[8]:


gas_test(path15, paths15, '015_ecc')


# In[9]:


gas_test(path20, paths20, '020_ecc')


# In[10]:


gas_test(path25, paths25, '025_ecc')


# In[11]:


gas_test(path30, paths30, '030_ecc')


# In[11]:


def make_ecc3_plot(fpath, path, e):    
    tstart = time.time()
    
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
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
    vphig  = pl.fromfile(f'{fpath}gasvx100.dat').reshape(ny,nx)
    vphig += rad2d * omegap
    vrg    = pl.fromfile(f'{fpath}gasvy100.dat').reshape(ny,nx) 
    rhog   = pl.fromfile(f'{fpath}gasdens100.dat').reshape(ny,nx)
    
    e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
    e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
    eccg = np.sqrt(e1g**2 + e2g**2)
    gas_e = rhog*eccg
    
    e1g = np.abs(e1g).mean(axis=1)
    e2g = np.abs(e2g).mean(axis=1)
#     eccg = eccg.mean(axis=1)
    rhog_mean  = rhog.mean(axis=1)
    gas_e_mean = gas_e.mean(axis=1) 
    
    eccg_mean = gas_e_mean / rhog_mean

    fig, ax = plt.subplots()
    
    ax.plot(rad2d, eccg.mean(axis=1), label='ecc')
    ax.plot(rad2d, eccg_mean, label='ecc mean')
    ax.set_xlabel('Radius')
    ax.set_xlim(0,4)
    ax.set_ylabel('ecc weighted and unweighted')
#     ax.legend()
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[12]:


make_ecc3_plot(path30, paths30, '030_ecc')


# In[7]:


def overplot_ecc_plot(fpath, path, title, e):    
    tstart = time.time()
    
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
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for ax, title in zip(axes.flat, titles): 
        vphi = pl.fromfile(f'{fpath}{title}vx100.dat').reshape(ny,nx)
        vphi += rad2d * omegap
        vr   = pl.fromfile(f'{fpath}{title}vy100.dat').reshape(ny,nx)  
        
        e1 = rad2d * vphi**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2 = rad2d * vr * vphi                 # e2 corresponds to the contribution from the radial velocity of the gas or dust
        ecc = np.sqrt(e1**2 + e2**2)
        e1 = np.abs(e1).mean(axis=1)
        e2 = np.abs(e2).mean(axis=1)
#         ecc = ecc.mean(axis=1)
        
        rho = pl.fromfile(f'{fpath}gasdens100.dat').reshape(ny,nx)
        gas_e = rho*ecc
        rho_mean = rho.mean(axis=1)
        gas_e_mean = gas_e.mean(axis=1)
        ecc_mean = gas_e_mean / rho_mean        
    
        ax.plot(rad2d, ecc.mean(axis=1), 'k-')
        ax.plot(rad2d, ecc_mean, 'r-')
        ax.set_xlabel(r'$r/r_p$', fontsize=14)
        ax.set_title(f'particle: {title}, timestep: 100')
        ax.set_xlim(0,4)
#         ax.legend()
    fig.text(0.08, 0.5, f'{e} Ecentricity', va='center', rotation='vertical', fontsize=14)
    plt.savefig(f'{path}{e}_mean_overplot_100.png', transparent=True, dpi=300) # dots per inch  
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[ ]:


for fpath, path, title, e in zip(fp, ps, titles, ecc):
    
    overplot_ecc_plot(fpath, path, title, e)


# ![image.png](attachment:e08a0b24-7063-47d9-b940-0c3b53ec85fb.png)

# #### where, 
# #### a = Length of semi major axis
# #### b = Length of semi minor axis
# ![image.png](attachment:56378f1c-2d72-4192-a7b0-1ac7b1f6cede.png)

# In[19]:


# https://www.geeksforgeeks.org/program-to-find-the-eccentricity-of-an-ellipse/

def findEccentricity(fpath, path, e, nf):  
    tstart = time.time()
    
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
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

    gas = rhog/rhog_i
    dust1 = rhod1/rhod_i
    dust2 = rhod2/rhod_i
    dust3 = rhod3/rhod_i
    dust4 = rhod4/rhod_i
    dust5 = rhod5/rhod_i
    dust6 = rhod6/rhod_i
    dust7 = rhod7/rhod_i
    
    particles = [gas, dust1, dust2, dust3, dust4, dust5, dust6, dust7]
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, ax, title in zip(particles, axes.flat, titles):        
        ax.imshow(i, vmin=0.5, vmax=2)
        ax.set_xlabel(r'$\phi$', fontsize=20)
        ax.set_title(f'particle: {title}, timestep: {nf}')
        fig.text(0.08, 0.5, r'$rad$', va='center', rotation='vertical', fontsize=20)
    plt.savefig(f'{path}flat_density{nf}.png', transparent=True, dpi=300) # dots per inch  
    
    # calculate A and B here
    
    ## within some range of radius and take max ???
    partial_rad = np.where((rad > 1.2) & (rad < 3.0))
    
    print(partial_rad)
    
    print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[16]:


findEccentricity(path0, paths0, '0_ecc', 100)     #rad2.6


# In[18]:


findEccentricity(path10, paths10, '010_ecc', 100)   # rad 2.8


# In[20]:


findEccentricity(path15, paths15, '015_ecc', 100)   # rad 3.0


# In[54]:


findEccentricity(path20, paths20, '020_ecc', 100)   # rad 2.2


# In[14]:


findEccentricity(path25, paths25, '025_ecc', 100)    # rad 2.6


# In[15]:


findEccentricity(path30, paths30, '030_ecc', 100)    # rad 2.6


# In[20]:


def flat_movie_plot(fpath, path, e):
    
    for nf in range(0, 1001):        
        findEccentricity(fpath, path, e, nf)


# In[21]:


flat_movie_plot(path0, paths0, '0_ecc')


# In[ ]:


flat_movie_plot(path05, paths05, '005_ecc')


# In[ ]:


flat_movie_plot(path10, paths10, '010_ecc')


# In[ ]:


flat_movie_plot(path15, paths15, '015_ecc')


# In[ ]:


flat_movie_plot(path20, paths20, '020_ecc')


# In[ ]:


flat_movie_plot(path25, paths25, '025_ecc')


# In[ ]:


flat_movie_plot(path30, paths30, '030_ecc')


# In[56]:


nx=100

for i in range(nx):
    print(i)


# ### Dust example

# In[8]:


fpath = path20
nf = 100

phi_dat = np.loadtxt(fpath+'domain_x.dat')
rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]

phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])

#  print(f'phi: {phi}, \nrad: {rad}')

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

gas = rhog/rhog_i
dust1 = rhod1/rhod_i
dust2 = rhod2/rhod_i
dust3 = rhod3/rhod_i
dust4 = rhod4/rhod_i
dust5 = rhod5/rhod_i
dust6 = rhod6/rhod_i
dust7 = rhod7/rhod_i

particles = [gas, dust1, dust2, dust3, dust4, dust5, dust6, dust7]
titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']

# fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# for i, ax, title in zip(particles, axes.flat, titles):        
#     ax.imshow(i, vmin=0.5, vmax=2)
#     ax.set_xlabel(r'$\phi$', fontsize=20)
#     ax.set_title(f'particle: {title}, timestep: {nf}')
#     fig.text(0.08, 0.5, r'$rad$', va='center', rotation='vertical', fontsize=20)
#  plt.savefig(f'{path}density{nf}.png', transparent=True, dpi=300) # dots per inch  
    
    # calculate A and B here
    
    ## within some range of radius and take max ???
partial_rad = np.where((rad > 1.2) & (rad < 2.2))[0]

# print(partial_rad)

rad_el = []
phi_el = phi

for m in range(nx):
    maxx = rad[partial_rad[np.argmax(dust1[partial_rad, m])]]
#     print(maxx)
    rad_el.append(maxx)
    # np. max only gives max value, argmax give index
X_el = rad_el*np.cos(phi_el)
Y_el = rad_el*np.sin(phi_el)
# print(X_el)
# print(Y_el)

xy_arr = np.vstack((X_el, Y_el)).T
print(xy_arr.shape)


ell = EllipseModel()
ell.estimate(xy_arr)

xc, yc, a, b, theta = ell.params

print("center = ",  (xc, yc))
print("angle of rotation = ",  theta)
print("axes = ", (a,b))

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(X_el,Y_el)

# axs[1].scatter(X_el, Y_el)
axs[1].scatter(xc, yc, color='red', s=100)
axs[1].set_xlim(X_el.min(), X_el.max())
axs[1].set_ylim(Y_el.min(), Y_el.max())

ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')

axs[1].add_patch(ell_patch)
plt.show()

# # define A and B
A = a
B = b

# # Store the squares of length of semi-major and semi-minor axis
semiMajor = A**2
semiMinor = B**2
# # Calculate the eccentricity
ans = np.sqrt(1 - semiMinor / semiMajor)
print('%.2f' % ans)


# In[13]:


print(xy_arr)


# In[72]:


print(phi_el)


# In[77]:


plt.plot(X_el, Y_el, '.')
plt.axis('equal')


# In[2]:


get_ipython().run_line_magic('pip', 'install scikit-image')


# ### ellipse calculation example

# In[7]:


import numpy as np
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

points = [(0,3),(1,2),(1,7),(2,2),(2,4),(2,5),(2,6),(2,14),(3,4),(4,4),(5,5),(5,14),(6,4),(7,3),(7,7),(8,10),(9,1),(9,8),(9,9),(10,1),(10,2),(10,12),(11,0),(11, 7),(12,7),(12,11),(12,12),(13,6),(13,8),(13,12),(14,4),(14,5),(14,10),(14,13)]

a_points = np.array(points)
x = a_points[:, 0]
y = a_points[:, 1]

ell = EllipseModel()
ell.estimate(a_points)

xc, yc, a, b, theta = ell.params

print("center = ",  (xc, yc))
print("angle of rotation = ",  theta)
print("axes = ", (a,b))

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(x,y)

axs[1].scatter(x, y)
axs[1].scatter(xc, yc, color='red', s=100)
axs[1].set_xlim(x.min(), x.max())
axs[1].set_ylim(y.min(), y.max())

ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')

axs[1].add_patch(ell_patch)
plt.show()


# ### Ecc for gas only (attemp to be fixed for the combination at the end)

# In[13]:


def ecc_gas(fpath, path, e):
    tstart = time.time()
    
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
    
    e1 = []
    e2 = []
    ecc = []
    ecc_mean = []
    t = list(range(0, 11))
    
    for i in range(0, 11):
        ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == i)[0][-1]
        omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
        
        vphig  = pl.fromfile(f'{fpath}gasvx{i}.dat').reshape(ny,nx)
        vphig += rad2d * omegap
        vrg    = pl.fromfile(f'{fpath}gasvy{i}.dat').reshape(ny,nx) 
        rhog   = pl.fromfile(f'{fpath}gasdens{i}.dat').reshape(ny,nx)
        
        e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
        e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
        eccg = np.sqrt(e1g**2 + e2g**2)
        gas_e = rhog*eccg
        
        e1g = np.abs(e1g).mean(axis=1)
        e2g = np.abs(e2g).mean(axis=1)
        eccg = eccg.mean(axis=1)
        rhog_mean  = rhog.mean(axis=1)
        gas_e_mean = gas_e.mean(axis=1) 
        
        eccg_mean = gas_e_mean / rhog_mean
        
        e1.append(e1g.max())
        e2.append(e2g.max())
        ecc.append(eccg.max())
        ecc_mean.append(eccg_mean.max())
        
    np.save(f'{fpath}e1_gas_dat.npy', e1) # save
    np.save(f'{fpath}e2_gas_dat.npy', e2) # save
    np.save(f'{fpath}ecc_gas_dat.npy', ecc) # save without density weighted
    np.save(f'{fpath}ecc_gas_mean_dat.npy', ecc_mean) # save with density weighted        
        
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(21,19), sharex=True)
    
    axes[0, 0].plot(t, e1)
    axes[0, 0].set_xlabel('Time', fontsize=15)
    axes[0, 0].set_ylabel('e1', fontsize=15)
    axes[0, 0].set_title(f'{e}: e1 vs Time of Gas', fontsize=16)
    
    axes[0, 1].plot(t, e2)
    axes[0, 1].set_xlabel('Time', fontsize=15)
    axes[0, 1].set_ylabel('e2', fontsize=15)
    axes[0, 1].set_title(f'{e}: e2 vs Time of Gas', fontsize=16)
    
    axes[1, 0].plot(t, ecc)
    axes[1, 0].set_xlabel('Time', fontsize=15)
    axes[1, 0].set_ylabel('ecc without weighted', fontsize=15)
    axes[1, 0].set_title(f'{e}: ecc vs Time without density weighted', fontsize=16)
    
    axes[1, 1].plot(t, ecc_mean)
    axes[1, 1].set_xlabel('Time', fontsize=15)
    axes[1, 1].set_ylabel('ecc weighted', fontsize=15)
    axes[1, 1].set_title(f'{e}: ecc vs Time with density weighted', fontsize=16)
    
    axes[2, 0].plot(t, e1, label='e1')
    axes[2, 0].plot(t, e2, label='e2')
    axes[2, 0].plot(t, ecc, label='ecc')
    axes[2, 0].set_xlabel('Time', fontsize=15)
    axes[2, 0].set_ylabel('eccentricity without weighted', fontsize=15)
    axes[2, 0].set_title(f'{e}: e1, e2, ecc without dens weighted', fontsize=16)
    axes[2, 0].legend()
    
    axes[2, 1].plot(t, e1, label='e1')
    axes[2, 1].plot(t, e2, label='e2')
    axes[2, 1].plot(t, ecc_mean, label='ecc mean')
    axes[2, 1].set_xlabel('Time', fontsize=15)
    axes[2, 1].set_ylabel('eccentricity weighted', fontsize=15)
    axes[2, 1].set_title(f'{e}: e1, e2, ecc with dens weighted', fontsize=16)
    axes[2, 1].legend()
    plt.savefig(f'{path}{e}_e1_e2_ecc_weight.png', transparent=True, dpi=300) # dots per inch 
    
    fig, ax = plt.subplots()
    
    ax.plot(t, e1, label='e1')
    ax.plot(t, e2, label='e2')
    ax.plot(t, ecc_mean, label='ecc mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('weighted ecc')
    ax.legend()
#     plt.savefig(f'{path}{e}_mean_overplot.png', transparent=True, dpi=300) # dots per inch
    
    print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")


# ### ecc for gas and dust 

# In[41]:


# https://www.geeksforgeeks.org/program-to-find-the-eccentricity-of-an-ellipse/

def findEccentricity_2(fpath, path, e):  
    tstart = time.time()
    
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
    
    #### gas, dust4, 5, 6, 7 ####
#     e1 = []
    e1_gas = []
    e1_dust4 = []
    e1_dust5 = []
    e1_dust6 = []
    e1_dust7 = []
    
#     e2 = []
    e2_gas = []
    e2_dust4 = []
    e2_dust5 = []
    e2_dust6 = []
    e2_dust7 = []
    
#     ecc= []
    ecc_gas = []
    ecc_dust4 = []
    ecc_dust5 = []
    ecc_dust6 = []
    ecc_dust7 = []
    
#     ecc_mean = []
    ecc_gas_mean = []
    ecc_dust4_mean = []
    ecc_dust5_mean = []
    ecc_dust6_mean = []
    ecc_dust7_mean = []
    
    t = list(range(0, 1001))
    nf = range(0, 1001)    
    
    gas_titles = ['gas','dust4', 'dust5', 'dust6', 'dust7'] 
    
    for i in nf:
        for title in gas_titles:
            ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == i)[0][-1]
            omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
            
            vphig  = pl.fromfile(f'{fpath}{title}vx{i}.dat').reshape(ny,nx)
            vphig += rad2d * omegap
            vrg    = pl.fromfile(f'{fpath}{title}vy{i}.dat').reshape(ny,nx) 
            rhog   = pl.fromfile(f'{fpath}{title}dens{i}.dat').reshape(ny,nx)
            
            e1g = rad2d * vphig**2 - 1               # e1 represents the deviation of vφ from the Kepler velocity
            e2g = rad2d * vrg * vphig                # e2 corresponds to the contribution from the radial velocity of the gas or dust
            eccg = np.sqrt(e1g**2 + e2g**2)
            gas_e = rhog*eccg
            e1g_e = rhog*e1g
            e2g_e = rhog*e2g

            
            e1g = np.abs(e1g).mean(axis=1)
            e2g = np.abs(e2g).mean(axis=1)
            eccg = eccg.mean(axis=1)
            rhog_mean  = rhog.mean(axis=1)
            gas_e_mean = gas_e.mean(axis=1)
            e1g_e_mean = e1g_e.mean(axis=1)
            e2g_e_mean = e2g_e.mean(axis=1)
            
            eccg_mean = gas_e_mean / rhog_mean
            e1g_mean = e1g_e_mean / rhog_mean 
            e2g_mean = e2g_e_mean / rhog_mean
            
            if title == 'gas':
                e1_gas.append(e1g.max())
                e2_gas.append(e2g.max())
                ecc_gas.append(eccg.max())
                ecc_gas_mean.append(eccg_mean.max())
            if title == 'dust4':
                e1_dust4.append(e1g.max())
                e2_dust4.append(e2g.max())
                ecc_dust4.append(eccg.max())
                ecc_dust4_mean.append(eccg_mean.max())
            if title == 'dust5':
                e1_dust5.append(e1g.max())
                e2_dust5.append(e2g.max())
                ecc_dust5.append(eccg.max())
                ecc_dust5_mean.append(eccg_mean.max())
            if title == 'dust6':
                e1_dust6.append(e1g.max())
                e2_dust6.append(e2g.max())
                ecc_dust6.append(eccg.max())
                ecc_dust6_mean.append(eccg_mean.max())
            if title == 'dust7':
                e1_dust7.append(e1g.max())
                e2_dust7.append(e2g.max())
                ecc_dust7.append(eccg.max())
                ecc_dust7_mean.append(eccg_mean.max())
                
    np.save(f'{fpath}e1_gas.npy', e1_gas) # save
    np.save(f'{fpath}e2_gas.npy', e2_gas) # save
    np.save(f'{fpath}ecc_gas.npy', ecc_gas) # save without density weighted
    np.save(f'{fpath}ecc_gas_mean.npy', ecc_gas_mean) # save with density weighted 
    
    np.save(f'{fpath}e1_dust4.npy', e1_dust4) 
    np.save(f'{fpath}e2_dust4.npy', e2_dust4) 
    np.save(f'{fpath}ecc_dust4.npy', ecc_dust4) 
    np.save(f'{fpath}ecc_dust4_mean.npy', ecc_dust4_mean)

    np.save(f'{fpath}e1_dust5.npy', e1_dust5) 
    np.save(f'{fpath}e2_dust5.npy', e2_dust5) 
    np.save(f'{fpath}ecc_dust5.npy', ecc_dust5) 
    np.save(f'{fpath}ecc_dust5_mean.npy', ecc_dust5_mean)
    
    np.save(f'{fpath}e1_dust6.npy', e1_dust6) 
    np.save(f'{fpath}e2_dust6.npy', e2_dust6) 
    np.save(f'{fpath}ecc_dust6.npy', ecc_dust6) 
    np.save(f'{fpath}ecc_dust6_mean.npy', ecc_dust6_mean)
    
    np.save(f'{fpath}e1_dust7.npy', e1_dust7) 
    np.save(f'{fpath}e2_dust7.npy', e2_dust7) 
    np.save(f'{fpath}ecc_dust7.npy', ecc_dust7) 
    np.save(f'{fpath}ecc_dust7_mean.npy', ecc_dust7_mean)

    #### dust1, 2, 3 ####
    rhod_i = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    # calculate A and B here
    ## within some range of radius and take max ???
    partial_rad = np.where((rad > 1.2) & (rad < 3.0))[0]
    
    ecc_ans_dust1 = []
    ecc_ans_dust2 = []
    ecc_ans_dust3 = []
        
    dust_titles = ['dust1', 'dust2', 'dust3']  
    
    for i in nf:
        for title in dust_titles:        
            if title == 'dust1':
                rhod1  = pl.fromfile(f'{fpath}dust1dens{i}.dat').reshape(ny,nx) 
                dust1 = rhod1/rhod_i
                rad_el_dust1 = []
                for m in zip(range(nx)):  
                    maxx = rad[partial_rad[np.argmax(dust1[partial_rad, m])]]
                    rad_el_dust1.append(maxx)              # np.max only gives max value, argmax gives indexes
                X_el = rad_el_dust1*np.cos(phi)
                Y_el = rad_el_dust1*np.sin(phi)    
                xy_arr = np.vstack((X_el, Y_el)).T    
                ell = EllipseModel()
                ell.estimate(xy_arr)    
                xc, yc, A, B, theta = ell.params 
                semiMajor = A**2
                semiMinor = B**2
#                 print(semiMajor, semiMinor)
                ecc_ans = np.sqrt(1 - np.min([semiMinor, semiMajor]) / np.max([semiMajor, semiMinor])) 
                ecc_ans_dust1.append(ecc_ans)                               
                    
            if title == 'dust2':
                rhod2  = pl.fromfile(f'{fpath}dust2dens{i}.dat').reshape(ny,nx) 
                dust2 = rhod2/rhod_i
                rad_el_dust2 = []
                for m in zip(range(nx)): 
                    maxx = rad[partial_rad[np.argmax(dust2[partial_rad, m])]]
                    rad_el_dust2.append(maxx)            
                X_el = rad_el_dust2*np.cos(phi)
                Y_el = rad_el_dust2*np.sin(phi)    
                xy_arr = np.vstack((X_el, Y_el)).T    
                ell = EllipseModel()
                ell.estimate(xy_arr)    
                xc, yc, A, B, theta = ell.params 
                semiMajor = A**2
                semiMinor = B**2
                ecc_ans = np.sqrt(1 - np.min([semiMinor, semiMajor]) / np.max([semiMajor, semiMinor])) 
                ecc_ans_dust2.append(ecc_ans)             
            
            if title == 'dust3':
                rhod3  = pl.fromfile(f'{fpath}dust3dens{i}.dat').reshape(ny,nx) 
                dust3 = rhod3/rhod_i
                rad_el_dust3 = []
                for m in zip(range(nx)): 
                    maxx = rad[partial_rad[np.argmax(dust3[partial_rad, m])]]
                    rad_el_dust3.append(maxx) 
                X_el = rad_el_dust3*np.cos(phi)
                Y_el = rad_el_dust3*np.sin(phi)    
                xy_arr = np.vstack((X_el, Y_el)).T    
                ell = EllipseModel()
                ell.estimate(xy_arr)    
                xc, yc, A, B, theta = ell.params 
                semiMajor = A**2
                semiMinor = B**2
                ecc_ans = np.sqrt(1 - np.min([semiMinor, semiMajor]) / np.max([semiMajor, semiMinor])) 
                ecc_ans_dust3.append(ecc_ans) 
            
#           np.save(f'{fpath}rad_el_dust1.npy', rad_el_dust1) # save
#           np.save(f'{fpath}rad_el_dust2.npy', rad_el_dust2) # save
#           np.save(f'{fpath}rad_el_dust3.npy', rad_el_dust3) # save
        
    np.save(f'{fpath}ecc_ans_dust1.npy', ecc_ans_dust1) # save
    np.save(f'{fpath}ecc_ans_dust2.npy', ecc_ans_dust2) # save
    np.save(f'{fpath}ecc_ans_dust3.npy', ecc_ans_dust3) # save
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,8), sharex=True)    
    
    axes[0, 0].plot(t, e1_gas, label='e1')
    axes[0, 0].plot(t, e2_gas, label='e2')
    axes[0, 0].plot(t, ecc_gas_mean, label='ecc mean')
#     axes[0, 0].set_xlabel('Time', fontsize=16)
    axes[0, 0].set_title(f'particle: gas ecc', fontsize=15)
    axes[0, 0].legend()
    
    axes[0, 1].plot(t, ecc_ans_dust1, label='ecc')
#     axes[0, 1].set_xlabel('Time', fontsize=16)
    axes[0, 1].set_title(f'particle: dust1 ecc', fontsize=15)
    axes[0, 1].legend()
    
    axes[0, 2].plot(t, ecc_ans_dust2)
#     axes[0, 2].set_xlabel('Time', fontsize=16)
    axes[0, 2].set_title(f'particle: dust2 ecc', fontsize=15)
#     axes[0, 2].legend()
    
    axes[0, 3].plot(t, ecc_ans_dust3)
#     axes[0, 3].set_xlabel('Time', fontsize=16)
    axes[0, 3].set_title(f'particle: dust3 ecc', fontsize=15)
#     axes[0, 3].legend()
    
    axes[1, 0].plot(t, e1_dust4, label='e1')
    axes[1, 0].plot(t, e2_dust4, label='e2')
    axes[1, 0].plot(t, ecc_dust4_mean, label='ecc mean')
    axes[1, 0].set_xlabel('Time', fontsize=15)
    axes[1, 0].set_title(f'particle: dust4 ecc', fontsize=15)
#     axes[1, 0].legend()
    
    axes[1, 1].plot(t, e1_dust5, label='e1')
    axes[1, 1].plot(t, e2_dust5, label='e2')
    axes[1, 1].plot(t, ecc_dust5_mean, label='ecc mean')
    axes[1, 1].set_xlabel('Time', fontsize=15)
    axes[1, 1].set_title(f'particle: dust5 ecc', fontsize=15)
#     axes[1, 1].legend()

    axes[1, 2].plot(t, e1_dust6, label='e1')
    axes[1, 2].plot(t, e2_dust6, label='e2')
    axes[1, 2].plot(t, ecc_dust6_mean, label='ecc mean')
    axes[1, 2].set_xlabel('Time', fontsize=15)
    axes[1, 2].set_title(f'particle: dust6 ecc', fontsize=15)
#     axes[1, 2].legend()
    
    axes[1, 3].plot(t, e1_dust7, label='e1')
    axes[1, 3].plot(t, e2_dust7, label='e2')
    axes[1, 3].plot(t, ecc_dust7_mean, label='ecc mean')
    axes[1, 3].set_xlabel('Time', fontsize=15)
    axes[1, 3].set_title(f'particle: dust7 ecc', fontsize=15)
#     axes[1, 3].legend()
    
    fig.text(0.08, 0.5, f'{e}', va='center', rotation='vertical', fontsize=18) 
    plt.savefig(f'{path}{e}_ecc_tot.png', transparent=True, dpi=300) # dots per inch   
    
    print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return e1_gas


# In[42]:


findEccentricity_2(path0, paths0, '0_ecc')


# In[93]:


ed1 = np.load(f'{path0}ecc_ans_dust1.npy')
print(ed1)


# In[94]:


egm = np.load(f'{path0}ecc_gas_mean.npy')
print(egm)


# In[70]:


gas_titles = ['gas','dust4', 'dust5', 'dust6', 'dust7'] 

for title in zip(gas_titles):
    print(title)


# In[71]:


print(gas_titles[0])


# In[12]:


def movie_ecc_plot(fpath, path, e):
    
    for nf in range(0, 11):        
        findEccentricity_2(fpath, path, e, nf)


# In[15]:


findEccentricity_2(path0, paths0, '0_ecc')


# In[22]:


findEccentricity_2(path05, paths05, '005_ecc')


# In[23]:


findEccentricity_2(path10, paths10, '010_ecc')


# In[24]:


findEccentricity_2(path15, paths15, '015_ecc')


# In[25]:


findEccentricity_2(path20, paths20, '020_ecc')


# In[26]:


findEccentricity_2(path25, paths25, '025_ecc')


# In[27]:


findEccentricity_2(path30, paths30, '030_ecc')


# ### 0.3 JM

# In[24]:


findEccentricity_2(path0_03, paths0_03, '0_ecc_03jm')


# In[25]:


findEccentricity_2(path05_03, paths05_03, '005_ecc_03jm')


# In[26]:


findEccentricity_2(path10_03, paths10_03, '010_ecc_03jm')


# In[9]:


findEccentricity_2(path15_03, paths15_03, '015_ecc_03jm')


# In[10]:


findEccentricity_2(path20_03, paths20_03, '020_ecc_03jm')


# In[8]:


findEccentricity_2(path25_03, paths25_03, '025_ecc_03jm')


# In[8]:


findEccentricity_2(path30_03, paths30_03, '030_ecc_03jm')


# ### 3 JM

# In[9]:


findEccentricity_2(path0_3, paths0_3, '0_ecc_3jm')


# In[10]:


findEccentricity_2(path05_3, paths05_3, '005_ecc_3jm')


# In[11]:


findEccentricity_2(path10_3, paths10_3, '010_ecc_3jm')


# In[12]:


findEccentricity_2(path15_3, paths15_3, '015_ecc_3jm')


# In[13]:


findEccentricity_2(path20_3, paths20_3, '020_ecc_3jm')


# In[14]:


findEccentricity_2(path25_3, paths25_3, '025_ecc_3jm')


# In[15]:


findEccentricity_2(path30_3, paths30_3, '030_ecc_3jm')


# In[8]:


# https://www.geeksforgeeks.org/program-to-find-the-eccentricity-of-an-ellipse/

def overplotEcc(fpath, path, e):  
    tstart = time.time()
    
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

    nf = 300    

    #### dust1, 2, 3 ####
    rhod_i = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    # calculate A and B here
    ## within some range of radius and take max ???
    partial_rad = np.where((rad > 1.2) & (rad < 3.0))[0]
            
        ## dust1 ##
    rhod1  = pl.fromfile(f'{fpath}dust1dens{nf}.dat').reshape(ny,nx) 
    dust1 = rhod1/rhod_i
    rad_el_dust1 = []
    for m in zip(range(nx)):  
        maxx = rad[partial_rad[np.argmax(dust1[partial_rad, m])]]
        rad_el_dust1.append(maxx)              # np.max only gives max value, argmax gives indexes
    X_el1 = rad_el_dust1*np.cos(phi)
    Y_el1 = rad_el_dust1*np.sin(phi)    
    xy_arr1 = np.vstack((X_el1, Y_el1)).T    
    ell1 = EllipseModel()
    ell1.estimate(xy_arr1)    
    xc1, yc1, A1, B1, theta1 = ell1.params 
    semiMajor1 = A1**2
    semiMinor1 = B1**2
    ecc_ans1 = np.sqrt(1 - np.min([semiMinor1, semiMajor1]) / np.max([semiMajor1, semiMinor1])) 
    ell_patch1 = Ellipse((xc1, yc1), 2*A1, 2*B1, theta1*180/np.pi, edgecolor='blue', facecolor='none')                          
        
        ## dust 2 ##
    rhod2  = pl.fromfile(f'{fpath}dust2dens{nf}.dat').reshape(ny,nx) 
    dust2 = rhod2/rhod_i
    rad_el_dust2 = []
    for m in zip(range(nx)): 
        maxx = rad[partial_rad[np.argmax(dust2[partial_rad, m])]]
        rad_el_dust2.append(maxx)            
    X_el2 = rad_el_dust2*np.cos(phi)
    Y_el2 = rad_el_dust2*np.sin(phi)    
    xy_arr2 = np.vstack((X_el2, Y_el2)).T    
    ell2 = EllipseModel()
    ell2.estimate(xy_arr2)    
    xc2, yc2, A2, B2, theta2 = ell2.params 
    semiMajor2 = A2**2
    semiMinor2 = B2**2
    ecc_ans2 = np.sqrt(1 - np.min([semiMinor2, semiMajor2]) / np.max([semiMajor2, semiMinor2])) 
    ell_patch2 = Ellipse((xc2, yc2), 2*A2, 2*B2, theta2*180/np.pi, edgecolor='blue', facecolor='none')              
        
        ## dust 3 ##
    rhod3  = pl.fromfile(f'{fpath}dust3dens{nf}.dat').reshape(ny,nx) 
    dust3 = rhod3/rhod_i
    rad_el_dust3 = []
    for m in zip(range(nx)): 
        maxx = rad[partial_rad[np.argmax(dust3[partial_rad, m])]]
        rad_el_dust3.append(maxx) 
    X_el3 = rad_el_dust3*np.cos(phi)
    Y_el3 = rad_el_dust3*np.sin(phi)    
    xy_arr3 = np.vstack((X_el3, Y_el3)).T    
    ell3 = EllipseModel()
    ell3.estimate(xy_arr3)    
    xc3, yc3, A3, B3, theta3 = ell3.params 
    semiMajor3 = A3**2
    semiMinor3 = B3**2
    ecc_ans3 = np.sqrt(1 - np.min([semiMinor3, semiMajor3]) / np.max([semiMajor3, semiMinor3])) 
    ell_patch3 = Ellipse((xc3, yc3), 2*A3, 2*B3, theta3*180/np.pi, edgecolor='blue', facecolor='none')     
    
    r = 5.0
    vmin = -0.4    # decrease: fanter
    vmax = 3.0     # inclrease: darker
    
    levels = np.linspace(vmin,vmax,100)
    
    particles = [dust1, dust2, dust3]
    titles = ['dust1', 'dust2', 'dust3']
    xcs = [xc1, xc2, xc3]
    ycs = [yc1, yc2, yc3]
    X_els = [X_el1, X_el2, X_el3]
    Y_els = [Y_el1, Y_el2, Y_el3]
    ell_patchs = [ell_patch1, ell_patch2, ell_patch3]
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))

    for i, ax, xc, yc, X_el, Y_el, ell_patch, title in zip(particles, axes.flat, xcs, ycs, X_els, Y_els, ell_patchs, titles):        
        ax.contourf(X, Y, i, levels, extend='both', cmap='gist_heat') # copper or gist_heat
        ax.scatter(xc, yc, color='red', s=100)
        ax.scatter(X_el,Y_el)
        ax.add_patch(ell_patch)
        ax.set_xlim(-r,r)
        ax.set_ylim(-r,r)
        ax.set_title(f'particle: {title}, timestep: {nf}')  
    
    fig.text(0.08, 0.5, f'{e}', va='center', rotation='vertical', fontsize=18) 
    plt.savefig(f'{path}{e}_dens_ell.png', transparent=True, dpi=300) # dots per inch   
    
    print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")


# In[60]:


for fpath, path, e in zip(fp, ps, ecc):        
    overplotEcc(fpath, path, e)


# In[17]:


for fpath, path, e in zip(fp_03, ps_03, ecc_03):        
    overplotEcc(fpath, path, e)


# In[9]:


for fpath, path, e in zip(fp_3, ps_3, ecc_3):        
    overplotEcc(fpath, path, e)


# In[16]:


for fpath, path, e in zip(fp_10, ps_10, ecc_10):        
    overplotEcc(fpath, path, e)


# In[32]:


dust3


# In[33]:


dust3.shape


# In[11]:


plt.plot(rad, dust3[:,0], label='0')
plt.plot(rad, dust3[:,255], label='255')
plt.plot(rad, dust3[:,511], label='511')
plt.plot(rad, dust3[:,737], label='737')
plt.xlabel('rad', fontsize=15)
plt.ylabel('dust3 density', fontsize=15)
plt.ylim(0,5)
plt.xlim(1,3)
plt.legend()


# In[62]:


dust3[:,511].max()


# In[13]:


plt.plot(rad, dust1[:,0], label='0')
plt.plot(rad, dust1[:,255], label='255')
plt.plot(rad, dust1[:,511], label='511')
plt.plot(rad, dust1[:,737], label='737')
plt.xlabel('rad', fontsize=15)
plt.ylabel('dust1 density', fontsize=15)
plt.legend()
plt.ylim(0,200)
plt.xlim(1,3)


# In[14]:


plt.plot(rad, dust2[:,0], label='0')
plt.plot(rad, dust2[:,255], label='255')
plt.plot(rad, dust2[:,511], label='511')
plt.plot(rad, dust2[:,737], label='737')
plt.xlabel('rad', fontsize=15)
plt.ylabel('dust1 density', fontsize=15)
plt.legend()
plt.ylim(0,50)
plt.xlim(1,3)


# In[17]:


def make_2d_plot(fpath, path, e, nf):    
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
#     plt.savefig(f'{path}density{nf}.png', transparent=True, dpi=300) # dots per inch  
    
#     extra_plots(rad_dat, rad, ny, nx, rhog_i, rhod_i, titles)  
        
    print(f"{e} {nf} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return rad_dat, rad, ny, nx, rhog_i, rhod_i, titles


# In[18]:


make_2d_plot(path0_3, paths0_3, '0_ecc', 100)


# In[21]:


def make_ecc_plot(fpath, path, title, e):    
    tstart = time.time()
    
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
    
    ind = np.where(np.loadtxt(f'{fpath}planet0.dat')[:,0] == 100)[0][-1]
    omegap = np.loadtxt(f'{fpath}planet0.dat')[ind][-1]
    
#     vphig += rad2d
    
#     vphi = pl.fromfile(f'{fpath}{title}vx100.dat').reshape(ny,nx)
#     vr   = pl.fromfile(f'{fpath}{title}vy100.dat').reshape(ny,nx)    
    
#     e1 = (rad * vphi[:,1]**2) - 1             # e1 represents the deviation of vφ from the Kepler velocity
#     e2 =  rad * vr[:,1] * vphi[:,1]                # e2 corresponds to the contribution from the radial velocity of the gas or dust
#     ecc = np.sqrt( (e1)**2 + (e2)**2 )
    
#     fig, ax = plt.subplots(figsize=(6,5))
#     ax.plot(rad, ecc)
#     ax.set_xlabel(r'$r/r_p$')
#     ax.set_ylabel('Ecentricity')
#     plt.savefig(f'{path}ecc.png', transparent=True, dpi=300) # dots per inch 
    
    titles = ['gas', 'dust1', 'dust2', 'dust3', 'dust4', 'dust5', 'dust6', 'dust7']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for ax, title in zip(axes.flat, titles): 
        vphi = pl.fromfile(f'{fpath}{title}vx100.dat').reshape(ny,nx)
        vphi += rad2d * omegap
        
        vr   = pl.fromfile(f'{fpath}{title}vy100.dat').reshape(ny,nx)  
        
#         print(f'vphi {e} {title}: {vphi}')
#         print(f'vr {e} {title}: {vr}')
        
#         e1 = (rad * vphi[:,1]**2) - 1             # e1 represents the deviation of vφ from the Kepler velocity
#         e2 =  rad * vr[:,1] * vphi[:,1]           # e2 corresponds to the contribution from the radial velocity of the gas or dust
#         ecc = np.sqrt( (e1)**2 + (e2)**2 )
        
        e1 = rad2d * vphi**2 - 1
        e2 = rad2d * vr * vphi
        ecc = np.sqrt(e1**2 + e2**2)
        ecc = ecc.mean(axis=1)
    
        ax.plot(rad2d, ecc)
        ax.set_xlabel(r'$r/r_p$', fontsize=14)
        ax.set_title(f'particle: {title}, timestep: 100')
    fig.text(0.08, 0.5, 'e', va='center', rotation='vertical', fontsize=14)
#     plt.savefig(f'{path}ecc.png', transparent=True, dpi=300) # dots per inch  

#     print(f'phi shape: {phi.shape} \nrad shape: {rad.shape}')
#     print(f'vphi shape: {vphi[:,1].shape} \nvr shape: {vr[:,1].shape}')
#     print(f'vphi: {vphi[:,1]} \nvr: {vr[:,1]}')
#     print(f'rad2d shape: {rad2d.shape}')
#     print(f'{e} = {ecc}')
#     print(f"{e} Elapsed time: {time.time()-tstart:.2f} seconds")
    
#     return phi, rad, vphi, vr, e1, e2, ecc


# In[22]:


make_ecc_plot(path0_3, paths0_3, 'gas', '0_ecc')


# In[ ]:




