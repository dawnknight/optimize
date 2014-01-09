# -*- coding: utf-8 -*-
"""
Created on Wed Jan 08 17:41:35 2014

@author: atc327
"""

import os, glob
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
    if x[5]<0 or x[5]>31:
        return 1e12    
        
    M = A * np.exp(
        -B*(args[1]-x[5])**2 - 
         C*(args[2]-x[6])**2 - 
         D*(args[1]-x[5])*(args[2]-x[6])
         ) + E     
    return np.sqrt(((args[0]-M)**2).sum())

def Gaussian_intergate(mtx):
    #h,w = im_cut.shape
    h=31;
    w=31;
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
     
    return IG   

def light_params(im,pts):
    params = [];
    for idx in np.arange(len(pts)):
        im_cut = im[pts[idx][1]:pts[idx][1]+31,pts[idx][0]:pts[idx][0]+31]
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

pts = [ [1260,1605],[1305,1670],[1680,1750],[1980,1580],[2510,1535],\
        [2505,1237],[2500,1115],[2695,1040],[3200,1350],[3615,1540]\
      ]
      
      
path = 'C:/Users/atc327/Desktop/images_ggd/'
r_mtx = [];
im_mtx =[];
ing_G = [];

for infile in glob.glob( os.path.join(path, '*.jpg') ):
    v_mtx = [];
    im_ori = np.array(Image.open(infile).convert('L')).astype(np.float)
    result = light_params(im_ori,pts)
    r_mtx.append(result)

    for idx in np.arange(len(pts)): 
       value = Gaussian_intergate(result[idx]) 
       v_mtx.append(value/31/31)        
    ing_G.append(v_mtx)
           
