# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:39:40 2014

@author: atc327

in this code, by changing idx and pic_idx value we can get the curve fitting
parameters in certain light source in certain pic
"""



import os, glob
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize


def chisq(x,*args):
    A = x[0]
    B = x[1] # B = 1/(2*(1-rho^2)*sigma_x^2)    B>0
    C = x[2] # C = 1/(2*(1-rho^2)*sigma_y^2)    C>0 
    D = x[3] # D = -lo/((1-rho^2)*sigma_x*sigma_y)   rho>0 => D<0 , rho<0 => D>0
    E = x[4]
     
    rho_s = D**2./B/C/4.
    sigmax_s = 1./2./B/(1.-rho_s)
    sigmay_s = 1./2./C/(1.-rho_s)
    
#    print B,sigmax_s,C,sigmay_s,D,rho_s     
    
    if A<255.:
        return 1e12 
    if x[5]<0 or x[5]>31:
        return 1e12    
#    if lo_s>1 or lo_s <0:     
#       return 1e12
#    if sigmax <3 or sigmay<3:
#       return 1e12     
    
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
    IG = sum((M-E).flatten())    
     
    return im_cut-M,IG        # IG = intergrated gaussian value

def light_params(im,pts):
    params = [];
    im_cut = im[pts[1]:pts[1]+31,pts[0]:pts[0]+31]
    h,w  = im_cut.shape
    y,x  = np.where(im_cut<=250)
    datapts  = np.vstack((x,y)).T
    data = im_cut[datapts[:,1],datapts[:,0]]
    args=[data,x,y]   
    x0 = [1000.,0.02,0.02,0,20,16,16] #initial
    k = optimize.fmin_powell(chisq,x0,args)        
    min_fval = chisq(list(k),*args)
    min_fval_params =k
        
    niter=3
    seed=314
    np.random.seed(seed)        
    #sol = np.array(k)        

    for i in np.arange(niter):
       x0 = k*(0.6*np.random.rand()+0.7)
       k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
       if min_fval > chisq(list(k),*args):    
          min_fval = chisq(list(k),*args)         
          min_fval_params =k
       #sol = np.vstack([sol,k])

    return min_fval_params
    
if __name__ == '__main__':  
    pts = [ [1260,1605],[1305,1670],[1680,1750],[1980,1580],[2510,1535],\
            [2505,1237],[2500,1115],[2690,1035],[3200,1350],[3615,1540]\
          ]
    
    pic_idx = 2
    idx      = 7    
    PTS = pts[idx]      
    
    
    path = 'C:/Users/atc327/Desktop/images_ggd/'
    img_dir = glob.glob( os.path.join(path, '*.jpg') )
    
    im = np.array(Image.open(img_dir[pic_idx]).convert('L')).astype(np.float)
    result = light_params(im,PTS)
    
    
    cmap = 'gist_heat'
    interp = 'nearest'
    clim = [-255,255]
    
    
    im_cut = im_ori[PTS[1]:PTS[1]+31,PTS[0]:PTS[0]+31]
    #mask = (im_cut<250)*1
    diff,value = Gaussian_difference(result,im_cut) 
    print value/31/31
               
    
    
    plt.imshow(diff, extent=(0,1,0,1), clim=clim, cmap=cmap,\
               interpolation=interp)



    
    
    
    
    
    
    
    
    
