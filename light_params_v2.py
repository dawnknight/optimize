# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:34:26 2013

@author: andy
"""

import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize

def chisq(x,*args):
    A = x[0]
    B = x[1]
    C = x[2]
    D = x[3]
    E = x[4]
    # B = -1/(2*(1-lo^2)*sigma_x^2)
    # C = -1/(2*(1-lo^2)*sigma_y^2)
    # D = -lo/((1-lo^2)*sigma_x*sigma_y)

    if A<255.:
        return 1e12        
    if x[5]>0 or x[5]<31 :   #ctr x
        return 1e12
    if x[6]>0 or x[6]<31 :   # ctr y 
        return 1e12
        
        
    M = A * np.exp(
        -B*(args[1]-x[5])**2 - 
         C*(args[2]-x[6])**2 - 
         D*(args[1]-x[5])*(args[2]-x[6])
         ) + E     


    return np.sqrt(((args[0]-M)**2).sum())

def Gaussian_difference(mtx,im_cut):
    h,w = im_cut.shape
    x_axis = np.arange(w)
    y_axis = np.arange(h)
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    A = mtx[0]
    B = mtx[1]
    C = mtx[2]
    D = mtx[3]
    E = mtx[4]
    ctr_x = mtx[5]
    ctr_y = mtx[6]
    M = A * np.exp(
        -B*(x_axis-ctr_x)**2-\
         C*(y_axis-ctr_y)**2-\
         D*(x_axis-ctr_x)*(y_axis-ctr_y)\
         ) + E
    return im_cut-M
    
    
    

def light_params(im,pts):
    params = [];
    for idx in np.arange(len(pts)):
        im_cut = im_ori[pts[idx][1]:pts[idx][1]+31,pts[idx][0]:pts[idx][0]+31]
        #imshow(im, cmap="Greys_r")
        h,w  = im_cut.shape
        y,x  = np.where(im_cut<=250)
        datapts  = np.vstack((x,y)).T
        data = im_cut[datapts[:,1],datapts[:,0]]
        
        x0 = [1000.,0.4,0.4,0.8,20,0.5*w,0.5*h] #initial
        k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
        
        niter=20
        seed=314
        np.random.seed(seed)
        
        sol = np.array(k)
        
        for i in np.arange(niter):
            x0 = k*(0.6*np.random.rand()+0.7)
            k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
            sol = np.vstack([sol,k])
              
        amp,B,C,D,E,ctr_x,ctr_y = k
        
        x_axis = np.arange(w)
        y_axis = np.arange(h)
        x_axis, y_axis = np.meshgrid(x_axis, y_axis)
        
        G = amp*np.exp(-B*(x_axis-ctr_x)**2\
                       -C*(y_axis-ctr_y)**2\
                       -D*(x_axis-ctr_x)*(y_axis-ctr_y)\
                       )+E  
        params.append(k)
    return params


pts = [ [1260,1605],[1305,1670],[1680,1750],[1980,1580],[2490,1600],\
        [2505,1245],[2500,1115],[2695,1040],[3200,1350],[3615,1540]\
      ]

#pts = [ [1260,1605]]

im_ori = np.array(Image.open('aug_22_2013-09-16-122136-207688.jpg').convert('L')).astype(np.float)
result = light_params(im_ori,pts)

cmap = 'gist_heat'
interp = 'nearest'
clim = [-255,255]
#figure(5),
for idx in np.arange(len(result)): 
   im_cut = im_ori[pts[idx][1]:pts[idx][1]+30,pts[idx][0]:pts[idx][0]+30]
   diff = Gaussian_difference(result[idx],im_cut)   
   plt.subplot(2,5,idx)
   plt.imshow(diff, extent=(0,1,0,1), origin='lower', clim=clim, cmap=cmap,\
              interpolation=interp)      
              
'''              
for idx in range(len(result)): 
    im_cut = im_ori[pts[idx][1]:pts[idx][1]+30,pts[idx][0]:pts[idx][0]+30]
    plt.subplot(2,5,idx)
    plt.imshow(im_cut,cmap = cm.Greys_r)              
'''              