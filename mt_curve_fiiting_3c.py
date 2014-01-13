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
from multiprocessing import Pool

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
        
        x0 = [1000.,0.02,0.02,0,20,16,16] #initial
        args=[data,x,y] 
        k = optimize.fmin_powell(chisq,x0,args)        
        min_fval = chisq(list(k),*args)
        min_fval_params =k
        
        niter=3
        seed=314
        np.random.seed(seed)              
        
        for i in np.arange(niter):
            x0 = k*(0.6*np.random.rand()+0.7)
            k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
            if min_fval > chisq(list(k),*args):    
               min_fval = chisq(list(k),*args)         
               min_fval_params =k
        params.append(min_fval_params)
    return params

def image_processing(img_idx):
    v_mtx = []
    im_ori = np.array(Image.open(img_dir[img_idx]).convert('L')).astype(np.float)
    result = light_params(im_ori,pts)
    for idx in np.arange(len(pts)): 
       value = Gaussian_intergate(result[idx]) 
       v_mtx.append(value/31/31)        
    return v_mtx

def image_processing_R(img_idx):
    v_mtx_R = []
    im_ori = np.array(Image.open(img_dir[img_idx])).astype(np.float)
    imR = im_ori[0:,0:,0]
    resultR = light_params(imR,pts)
    for idx in np.arange(len(pts)): 
       value_R = Gaussian_intergate(resultR[idx]) 
       v_mtx_R.append(value_R/31/31)        
    return v_mtx_R

def image_processing_G(img_idx):
    v_mtx_G = []
    im_ori = np.array(Image.open(img_dir[img_idx])).astype(np.float)
    imG = im_ori[0:,0:,1]
    resultG = light_params(imG,pts)
    for idx in np.arange(len(pts)): 
       value_G = Gaussian_intergate(resultG[idx]) 
       v_mtx_G.append(value_G/31/31)        
    return v_mtx_G
    
def image_processing_B(img_idx):
    v_mtx_B = []
    im_ori = np.array(Image.open(img_dir[img_idx])).astype(np.float)
    imB = im_ori[0:,0:,2]
    resultB = light_params(imB,pts)
    for idx in np.arange(len(pts)): 
       value_B = Gaussian_intergate(resultB[idx]) 
       v_mtx_B.append(value_B/31/31)        
    return v_mtx_B
    
    
pts = [ [1260,1605],[1305,1670],[1680,1750],[1980,1580],[2510,1535],\
        [2505,1237],[2500,1115],[2690,1035],[3200,1350],[3615,1540]\
          ]  
      

path = 'C:/Users/atc327/Desktop/images_ggd/'
img_dir = glob.glob( os.path.join(path, '*.jpg') )

if __name__ == '__main__':         
    
    pool = Pool(processes=20)
    
#   gray scale    
    ing_G = [] 
    ing_G.append(pool.map(image_processing,range(len(img_dir))))
    ing_G = ing_G[0]
    ing_G = np.transpose(ing_G) 
    for i in arange(len(pts)):
        figure(1),
        plot(arange(len(img_dir)),ing_G[i]) 
    figure(1), 
    title('gray scale')  
    
    
#   R domain      
    ing_G_R = []
    ing_G_R.append(pool.map(image_processing_R,range(len(img_dir))))
    ing_G_R = ing_G_R[0] 
    ing_G_R = np.transpose(ing_G_R)
    for i in arange(len(pts)):   
        figure(2),
        plot(arange(len(img_dir)),ing_G_R[i])        
    figure(2), 
    title('R domain') 
    
    
#   G domain    
    ing_G_G = []
    ing_G_G.append(pool.map(image_processing_G,range(len(img_dir))))
    ing_G_G = ing_G_G[0]
    ing_G_G = np.transpose(ing_G_G)
    ing_G_B = []       
    for i in arange(len(pts)):
        figure(3),
        plot(arange(len(img_dir)),ing_G_G[i])        
    figure(3), 
    title('G domain')
    
    
#   B domain    
    ing_G_B = []
    ing_G_B.append(pool.map(image_processing_B,range(len(img_dir)))) 
    ing_G_B = ing_G_B[0]
    ing_G_B = np.transpose(ing_G_B)             
    for i in arange(len(pts)):
        figure(4),
        plot(arange(len(img_dir)),ing_G_B[i])
    figure(4), 
    title('B domain') 

## plot by point    
# for i in arange(len(pts)):
#    figure(i+1),
#    plot(arange(len(img_dir)),ing_G[i],color = 'black')
#    plot(arange(len(img_dir)),ing_G_R[i],color = 'red')
#    plot(arange(len(img_dir)),ing_G_G[i],color = 'green')
#    plot(arange(len(img_dir)),ing_G_B[i],color = 'blue')
#    name = "Point " + repr(i+1)
#    title(name)
   
    
    
    
    
    
    
## plot intergrate value subtract average value    
#    avg = []
#    avgR = []
#    avgG = []
#    avgB = []
#    
#    for i in arange(len(pts)):
#        avg.append(sum(ing_G[i])/len(ing_G[i]))
#        avgR.append(sum(ing_G_R[i])/len(ing_G_R[i]))
#        avgG.append(sum(ing_G_G[i])/len(ing_G_G[i]))
#        avgB.append(sum(ing_G_B[i])/len(ing_G_B[i]))
#    
#    
#    
#    
#    
#    
#    
#    for i in arange(len(pts)):
#        figure(1),
#        plot(arange(len(img_dir)),ing_G[i]-avg[i])        
#    figure(1), 
#    title('gray scale (substract avg)')
#    
#    for i in arange(len(pts)):
#        figure(2),
#        plot(arange(len(img_dir)),ing_G_R[i]-avgR[i])        
#    figure(2), 
#    title('R domain(substract avg)')
#    
#    for i in arange(len(pts)):
#        figure(3),
#        plot(arange(len(img_dir)),ing_G_G[i]-avgG[i])        
#    figure(3), 
#    title('G domain(substract avg)')
#    
#    for i in arange(len(pts)):
#        figure(4),
#        plot(arange(len(img_dir)),ing_G_B[i]-avgB[i])        
#    figure(4), 
#    title('B domain(substract avg)')
#        