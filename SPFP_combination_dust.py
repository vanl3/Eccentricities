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
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import StrMethodFormatter


# ### 1JM

# In[2]:


# Path to save figures with planet
paths0 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_1jm/'
paths05 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_1jm/'
paths10 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_1jm/'
paths15 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_1jm/'
paths20 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_1jm/'
paths25 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_1jm/'
paths30 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_1jm/'

path0 = '/blue/jbae/vle/multifluid_0ecc_1jm/'
path05 = '/blue/jbae/vle/multifluid_005ecc_1jm/'
path10 = '/blue/jbae/vle/multifluid_010ecc_1jm/'
path15 = '/blue/jbae/vle/multifluid_015ecc_1jm/'
path20 = '/blue/jbae/vle/multifluid_020ecc_1jm/'
path25 = '/blue/jbae/vle/multifluid_025ecc_1jm/'
path30 = '/blue/jbae/vle/multifluid_030ecc_1jm/'

fp = [path0, path05, path10, path15, path20, path25, path30]
ps = [paths0, paths05, paths10, paths15, paths20, paths25, paths30]
ecc = ['0_ecc', '005_ecc', '010_ecc', '015_ecc', '020_ecc', '025_ecc', '030_ecc']


# ### 0.3 JM

# In[3]:


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

# In[4]:


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


# ### 0.1 JM

# In[5]:


# Path to load and save figures with planet
paths0_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_0ecc_01jm/'
paths05_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_005ecc_01jm/'
paths10_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_010ecc_01jm/'
paths15_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_015ecc_01jm/'
paths20_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_020ecc_01jm/'
paths25_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_025ecc_01jm/'
paths30_01 = '/blue/jbae/vle/AST_Research_Project/multifluid_030ecc_01jm/'

path0_01 = '/blue/jbae/vle/multifluid_0ecc_01jm/'
path05_01 = '/blue/jbae/vle/multifluid_005ecc_01jm/'
path10_01 = '/blue/jbae/vle/multifluid_010ecc_01jm/'
path15_01 = '/blue/jbae/vle/multifluid_015ecc_01jm/'
path20_01 = '/blue/jbae/vle/multifluid_020ecc_01jm/'
path25_01 = '/blue/jbae/vle/multifluid_025ecc_01jm/'
path30_01 = '/blue/jbae/vle/multifluid_030ecc_01jm/'

fp_01 = [path0_01, path05_01, path10_01, path15_01, path20_01, path25_01, path30_01]
ps_01 = [paths0_01, paths05_01, paths10_01, paths15_01, paths20_01, paths25_01, paths30_01]
ecc_01 = ['0_ecc_01jm', '005_ecc_01jm', '010_ecc_01jm', '015_ecc_01jm', '020_ecc_01jm', '025_ecc_01jm', '030_ecc_01jm']


# ### 10 JM

# In[6]:


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


# ### for gas only

# In[7]:


def dust_2d_ecc(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2):  
    tstart = time.time()
    
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])
    
    partial_rad = np.where((rad > 0.5) & (rad < 3.0))[0]
    
    nx = len(phi)
    ny = len(rad)

    P, R = np.meshgrid(phi, rad)
    X = R*np.cos(P)
    Y = R*np.sin(P)
   
    rad2d = np.tile(rad,(nx,1))
    rad2d = np.swapaxes(rad2d,0,1)
    
    i = nf
    
    #### dust1, 2 ####
    rhod_i = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    # calculate A and B here
    ## within some range of radius and take max ???
        
    dust_titles = ['dust1', 'dust2']  
    
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
            ecc_ans = np.sqrt(1 - np.min([semiMinor, semiMajor]) / np.max([semiMajor, semiMinor])) 
            ecc_ans_dust1.append(ecc_ans) 
            A_d1 = A
            B_d1 = B
                
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
            A_d2 = A
            B_d2 = B

    return ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2

# In[7]:


def dust_2d_diffrad(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2):  
    tstart = time.time()
    
    phi_dat = np.loadtxt(fpath+'domain_x.dat')
    rad_dat  = np.loadtxt(fpath+'domain_y.dat')[3:-3]
    
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])
    
    partial_rad = np.where((rad > 1.0) & (rad < 5.0))[0]
    
    nx = len(phi)
    ny = len(rad)

    P, R = np.meshgrid(phi, rad)
    X = R*np.cos(P)
    Y = R*np.sin(P)
   
    rad2d = np.tile(rad,(nx,1))
    rad2d = np.swapaxes(rad2d,0,1)
    
    i = nf
    
    #### dust1, 2 ####
    rhod_i = pl.fromfile(fpath+'dust1dens0.dat').reshape(ny,nx) 
    
    # calculate A and B here
    ## within some range of radius and take max ???
        
    dust_titles = ['dust1', 'dust2']  
    
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
            ecc_ans = np.sqrt(1 - np.min([semiMinor, semiMajor]) / np.max([semiMajor, semiMinor])) 
            ecc_ans_dust1.append(ecc_ans)       
            A_d1 = A
            B_d1 = B
                
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
            A_d2 = A
            B_d2 = B

    return ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2


# In[28]:


def dust_2d_ecc_movie(fpath):
    
    #### dust ####
    ecc_ans_dust1 = []
    ecc_ans_dust2 = []
    A_d1 = []
    A_d2 = []
    B_d1 = []
    B_d2 = []
    
    nmin= 0
    nmax= 1000
    
    for nf in range(nmin, nmax+1): 
        if nf == nmax:
            ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2 = dust_2d_ecc(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2) 
        else:
            dust_2d_ecc(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2)
            
    
#     np.save(f'{fpath}ecc_dust1_1000.npy', ecc_ans_dust1) # save
#     np.save(f'{fpath}ecc_dust2_1000.npy', ecc_ans_dust2) # save
    np.save(f'{fpath}a_d1_1000.npy', A_d1) # save
    np.save(f'{fpath}a_d2_1000.npy', A_d2) # save
    np.save(f'{fpath}b_d1_1000.npy', B_d1) # save
    np.save(f'{fpath}b_d2_1000.npy', B_d2) # save

# In[29]:

def dust_2d_ecc_diffrad_movie(fpath):
    
    #### dust ####
    ecc_ans_dust1 = []
    ecc_ans_dust2 = []
    A_d1 = []
    A_d2 = []
    B_d1 = []
    B_d2 = []
    
    nmin= 0
    nmax= 1000
    
    for nf in range(nmin, nmax+1): 
        if nf == nmax:
            ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2 = dust_2d_diffrad(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2) 
        else:
            dust_2d_diffrad(fpath, nf, ecc_ans_dust1, ecc_ans_dust2, A_d1, A_d2, B_d1, B_d2)
            
    
#     np.save(f'{fpath}ecc_dust1_1000.npy', ecc_ans_dust1) # save
#     np.save(f'{fpath}ecc_dust2_1000.npy', ecc_ans_dust2) # save
    np.save(f'{fpath}a_d1_1000.npy', A_d1) # save
    np.save(f'{fpath}a_d2_1000.npy', A_d2) # save
    np.save(f'{fpath}b_d1_1000.npy', B_d1) # save
    np.save(f'{fpath}b_d2_1000.npy', B_d2) # save

####################### 0 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path0_01)


# In[ ]:


dust_2d_ecc_movie(path0_03)


# In[ ]:


dust_2d_ecc_movie(path0)


# In[ ]:


dust_2d_ecc_diffrad_movie(path0_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path0_10)


# In[30]:


####################### 0.05 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path05_01)


# In[ ]:


dust_2d_ecc_movie(path05_03)


# In[ ]:


dust_2d_ecc_movie(path05)


# In[ ]:


dust_2d_ecc_diffrad_movie(path05_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path05_10)


# In[30]:


####################### 0.10 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path10_01)


# In[ ]:


dust_2d_ecc_movie(path10_03)


# In[ ]:


dust_2d_ecc_movie(path10)


# In[ ]:


dust_2d_ecc_diffrad_movie(path10_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path10_10)


# In[30]:


####################### 0.15 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path15_01)     # *


# In[ ]:


dust_2d_ecc_movie(path15_03)


# In[ ]:


dust_2d_ecc_movie(path15)


# In[ ]:


dust_2d_ecc_diffrad_movie(path15_3)     


# In[ ]:


dust_2d_ecc_diffrad_movie(path15_10)     # *     


# In[30]:


####################### 0.20 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path20_01)        ## *


# In[ ]:


dust_2d_ecc_movie(path20_03)


# In[ ]:


dust_2d_ecc_movie(path20)


# In[ ]:


dust_2d_ecc_diffrad_movie(path20_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path20_10)        ## *


# In[30]:


####################### 0.25 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path25_01)       # *


# In[ ]:


dust_2d_ecc_movie(path25_03)


# In[ ]:


dust_2d_ecc_movie(path25)


# In[ ]:


dust_2d_ecc_diffrad_movie(path25_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path25_10)       ## *


# In[30]:


####################### 0.30 ecc ########################
# In[ ]:

dust_2d_ecc_movie(path30_01)       # *


# In[ ]:


dust_2d_ecc_movie(path30_03)


# In[ ]:


dust_2d_ecc_movie(path30)


# In[ ]:


dust_2d_ecc_diffrad_movie(path30_3)


# In[ ]:


dust_2d_ecc_diffrad_movie(path30_10)


# In[30]:




