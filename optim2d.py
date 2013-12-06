# -*- coding: utf-8 -*-

import time
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib as mpl
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

"""
Created on Thu Dec 05 17:50:52 2013

@author: greg and andy
"""

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


#im   = np.array(Image.open('test.bmp').convert('L')).astype(np.float)
im   = np.array(Image.open('test7.png').convert('L')).astype(np.float)
h,w  = im.shape
y,x  = np.where(im<=250)
pts  = np.vstack((x,y)).T
data = im[pts[:,1],pts[:,0]]

x0 = [1000.,0.4,0.4,0.8,20,0.5*w,0.5*h] # [amp,B,C,D,constant,ctr_x,ctr_y]
# B = -1/(2*(1-lo^2)*sigma_x^2)
# C = -1/(2*(1-lo^2)*sigma_y^2)
# D = -lo/((1-lo^2)*sigma_x*sigma_y)

t1 = time.time()

#x0 = [100.,0.1,0.1,0.2,0.0,0.35*w,0.85*h] # [amp,B,C,D,constant,ctr_x,ctr_y]
k = optimize.fmin_powell(chisq,x0,args=[data,x,y])

niter=20
seed=314
np.random.seed(seed)

sol = np.array(k)

for i in range(niter):
    x0 = k*(0.6*np.random.rand()+0.7)
    k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
    sol = np.vstack([sol,k])

    print("OPTIM2D: interaion = {0} out of {1}".format(i+1,niter))
    print("OPTIM2D: k = {0}".format(k))
    

print("OPTIM2D:  total optimization time = {0}".format(time.time()-t1))


amp,B,C,D,E,ctr_x,ctr_y = k



'''
#G = np.array([G[:] for G in [[1]*w]*h]) #initial an array G
G = np.zeros(im.shape)
for ii in range(w):
    for jj in range(0,h):
        G[jj,ii] = amp*np.exp(-B*(ii-ctr_x)**2\
                              -C*(jj-ctr_y)**2\
                              -D*(ii-ctr_x)*(jj-ctr_y)\
                              )+E  
'''



x_axis = np.arange(w)
y_axis = np.arange(h)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)

G = amp*np.exp(-B*(x_axis-ctr_x)**2\
               -C*(y_axis-ctr_y)**2\
               -D*(x_axis-ctr_x)*(y_axis-ctr_y)\
               )+E
 


fig = mpl.pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x_axis,y_axis,G)     
mpl.pyplot.show()

