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
    B = x[1] # B = 1/(2*(1-lo^2)*sigma_x^2)    B>0
    C = x[2] # C = 1/(2*(1-lo^2)*sigma_y^2)    C>0 
    D = x[3] # D = -lo/((1-lo^2)*sigma_x*sigma_y)   lo>0 => D<0 , lo<0 => D>0
    E = x[4]
    
    if A<255.:
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
        h,w  = im_cut.shape
        y,x  = np.where(im_cut<=250)
        datapts  = np.vstack((x,y)).T
        data = im_cut[datapts[:,1],datapts[:,0]]
        
        x0 = [1000.,8e-5,8e-5,-0.02,20,16,16] #initial
        k = optimize.fmin_powell(chisq,x0,args=[data,x,y])        

        niter=10
        seed=314
        np.random.seed(seed)        
        sol = np.array(k)        

        for i in np.arange(niter):
            x0 = k*(0.6*np.random.rand()+0.7)
            k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
            sol = np.vstack([sol,k])              
        amp,B,C,D,E,ctr_x,ctr_y = k
        params.append(k)
    return params


#main function

pts = [ [1260,1605],[1305,1670],[1680,1750],[1980,1580],[2510,1535],\
        [2505,1245],[2500,1115],[2695,1040],[3200,1350],[3615,1540]\
      ]
pts = [[1980,1580]]
im_ori = np.array(Image.open('aug_22_2013-09-16-122136-207688.jpg').convert('L')).astype(np.float)
result = light_params(im_ori,pts)

cmap = 'gist_heat'
interp = 'nearest'
clim = [-255,255]

for idx in np.arange(len(result)): 
   im_cut = im_ori[pts[idx][1]:pts[idx][1]+30,pts[idx][0]:pts[idx][0]+30]
   diff = Gaussian_difference(result[idx],im_cut) 
   if len(result)>1:
       plt.subplot(2,5,idx+1)           
   plt.imshow(diff, extent=(0,1,0,1), origin='lower', clim=clim, cmap=cmap,\
              interpolation=interp)

'''
A,B,C,D,E,F,G = result[0]
lo = D**2/B/C/4
s_x = (1/B/2/(1-lo**2))**0.5
s_y = (1/C/2/(1-lo**2))**0.5
'''
              
