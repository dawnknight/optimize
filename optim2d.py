# -*- coding: utf-8 -*-
import scipy as sp

"""
Created on Thu Dec 05 17:50:52 2013

@author: loaner
"""

def chisq(x,*args):
    A = x[0]
    B = x[1]
    C = x[2]
    D = x[3]
    E = x[4]

    M = A*np.exp(-B*(args[1]-x[5])**2-C*(args[2]-x[6])**2-D*(args[1]-x[5])*(args[2]-x[6]))+E     


    return sum((args[0]-M)**2)



import numpy as np
import scipy as sp
from PIL import Image
import matplotlib as mpl
from scipy import optimize

im = np.array(Image.open('test.bmp').convert('L')).astype(np.float)
im_f = im.flatten()
h,w = im.shape
y,x = np.where(im!=255)
pts = np.vstack((x,y)).T
data = im[pts[:,1],pts[:,0]]

x0 = [1000,0.4,0.4,0.8,20,w/2,h/2] # [amp,B,C,D,constant,ctr_x,ctr_y]

# B = -1/(2*(1-lo^2)*sigma_x^2)
# C = -1/(2*(1-lo^2)*sigma_y^2)
# D = -lo/((1-lo^2)*sigma_x*sigma_y)

k = optimize.fmin_powell(chisq,x0,args=[data,x,y])
print k

amp,B,C,D,E,ctr_x,ctr_y = k

G=np.array([G[:] for G in [[1]*w]*h]) #initial an array G
#plot
for ii in range(0,w):
    for jj in range(0,h):
        G[jj,ii] = amp*np.exp(-B*(ii-ctr_x)**2-C*(jj-ctr_y)**2-D*(ii-ctr_x)*(jj-ctr_y))+E  


x_axis = np.arange(0,w,1)   
y_axis = np.arange(0,h,1)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)

 

fig = mpl.pyplot.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(x_axis,y_axis,G)     


mpl.pyplot.show()









