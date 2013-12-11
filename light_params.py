# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:12:43 2013

@author: andy
"""

import scipy as sp
import numpy as np
from PIL import Image
from scipy import optimize

def chisq(x,*args):
    A = x[0]
    B = x[1]
    C = x[2]
    D = x[3]
    E = x[4]

    if A<255.:
        return 1e12

    M = A * np.exp(
        -B*(args[1]-x[5])**2 - 
         C*(args[2]-x[6])**2 - 
         D*(args[1]-x[5])*(args[2]-x[6])
         ) + E     


    return np.sqrt(((args[0]-M)**2).sum())



def light_params(im,pts):
    params = [];
    for idx in arange(len(pts)):
        im = im_ori[pts[idx][0][1]:pts[idx][1][1],pts[idx][0][0]:pts[idx][1][0]]
        #imshow(im, cmap="Greys_r")
        h,w  = im.shape
        y,x  = np.where(im<=250)
        datapts  = np.vstack((x,y)).T
        data = im[datapts[:,1],datapts[:,0]]
        
        x0 = [1000.,0.4,0.4,0.8,20,0.5*w,0.5*h] #initial
        k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
        
        niter=20
        seed=314
        np.random.seed(seed)
        
        sol = np.array(k)
        
        for i in range(niter):
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

# light source coordinate
pts = [ [[1206,1609],[1236,1636]],[[1264,1607],[1292,1634]],\
        [[1210,1645],[1245,1676]],[[1306,1670],[1338,1701]],\
        [[1977,1582],[2016,1613]],[[2500,1115],[2533,1142]],\
        [[2563,1094],[2588,1120]],[[2588,1092],[2611,1118]],\
        [[2613,1092],[2635,1120]],[[2636,1092],[2656,1117]],\
        [[2656,1093],[2679,1116]],[[2505,1242],[2536,1265]],\
        [[3028,1218],[3058,1248]],[[3202,1350],[3235,1384]],\
       ]


im_ori = np.array(Image.open('aug_22_2013-09-16-122136-207688.jpg').convert('L')).astype(np.float)

result = light_params(im_ori,pts)












