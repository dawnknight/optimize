# -*- coding: utf-8 -*-
import scipy as sp

"""
Created on Thu Dec 05 17:50:52 2013

@author: loaner
"""

def chisq(x,*args):
    amp = x[0]
    a = x[1]
    b = x[2]
    c = x[3]
    k=np.array([args[1],args[2]])
    ctr = np.vstack((np.ones(k.shape[1])*x[4],np.ones(k.shape[1])*x[5]))
    sigma = np.array(([a,b],[b,c]))
    M = amp*np.exp(np.dot(np.dot((k-ctr).T,np.linalg.inv(sigma)),(k-ctr)))+x[6]
    return sum((args[0]-M)**2)


import numpy as np
import scipy as sp
from PIL import Image
import matplotlib as mpl
from scipy import optimize

im = np.array(Image.open('test.bmp').convert('L')).astype(np.float)
h,w = im.shape
y,x = np.where(im!=255)
pts = np.vstack((x,y)).T
data = im[pts[:,1],pts[:,0]]

x0 = [500,1,2,2,w/2,h/2,20]


print optimize.anneal(chisq,x0,args=[data,x,y],lower=0 )






