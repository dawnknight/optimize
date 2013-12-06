# -*- coding: utf-8 -*-
import scipy as sp

"""
Created on Thu Dec 05 17:50:52 2013

@author: loaner
"""

def chisq(x,*args):
    amp = x[0]
    sigma = x[1]
    ctr = x[2]
    k=args[1]  
    M = amp*np.exp((-(k-ctr)**2)/(2*sigma**2))+x[3]
    return sum((args[0]-M)**2)

def Gaussian(x,amp,ctr,sigma,C):
    return amp*np.exp((-(x-ctr)**2)/(2*sigma**2))+C

import numpy as np
import scipy as sp
from PIL import Image
import matplotlib as mpl
from scipy import optimize

im = np.array(Image.open('test.bmp').convert('L')).astype(np.float)
row = im[20]

x = np.where(row!=255)

data = row[x]

x0 = [500,5,17,20] #[amp sigma ctr C]


k = optimize.anneal(chisq,x0,args=[data,x],lower=0.0)[0]
print k
amp =k[0]
ctr = k[1]
sigma = k[2]
C =k[3]
print amp,ctr,sigma,C

plot(arange(0,41),row)
plot(arange(0,41),Gaussian(arange(0,41),amp,ctr,sigma,C))




