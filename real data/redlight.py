# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:28:15 2014

@author: atc327
"""
import os, glob,pickle,sys
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize
from dst13 import *


fl = pickle.load(file('filelist.pkl'))

start = "10/26/13 19:00:00"
end   = "10/27/13 05:00:00"

#f.view_slice(start,end,"./")

paths,files,times = fl.time_slice(start,end)


pts = [[178,543],[256,541],[286,543],[500,25],[1585,365],[2556,547],[3579,409]]

#im = np.array(Image.open('img_4-8960.png').convert('L')).astype(np.float)
R = []
G = []
B = []

for i,(p,f,t) in enumerate(zip(paths[::],files[::],times[::])):
    im = read_raw(f,p)
    result_avg_R = []
    result_avg_G = []
    result_avg_B = [] 
    print("Processing img : {0}, Be patient !!".format(f))    

    if sum(im.flatten())!=0:                     
        for idx in range(len(pts)):
            im_cut =im[pts[idx][1]:pts[idx][1]+17,pts[idx][0]:pts[idx][0]+17] 
            im_cutR = im_cut[0:,0:,0]
            im_cutG = im_cut[0:,0:,1]
            im_cutB = im_cut[0:,0:,2]
    
            avg_R = sum(im_cutR.flatten())/len(im_cutR.flatten())
            avg_G = sum(im_cutG.flatten())/len(im_cutG.flatten())
            avg_B = sum(im_cutB.flatten())/len(im_cutB.flatten())
                
            result_avg_R.append(avg_R)
            result_avg_G.append(avg_G)
            result_avg_B.append(avg_B)
    else:        
        result_avg_R = list(np.zeros(7));    
        result_avg_G = list(np.zeros(7));
        result_avg_B = list(np.zeros(7));
    
    R.append(result_avg_R)    
    G.append(result_avg_G)
    B.append(result_avg_B)
    
R_T = np.transpose(R)
G_T = np.transpose(G)
B_T = np.transpose(B)        

cmap = 'gist_heat'
interp = 'nearest'
clim = [-255,255]
savepath = os.environ['DST_WRITE']
nameR= 'R.png'
nameG= 'G.png'
nameB= 'B.png'

figure(1),
title('R')
for idx in arange(len(pts)):   
    plot(arange(len(paths)),R_T[idx])
plt.savefig(os.path.join(savepath,nameR),clobber=True)
plt.close()
figure(2),
title('G') 
for idx in arange(len(pts)):       
    plot(arange(len(paths)),G_T[idx])
plt.savefig(os.path.join(savepath,nameG),clobber=True) 
plt.close()  
figure(3),
title('B') 
for idx in arange(len(pts)):          
    plot(arange(len(paths)),B_T[idx])   
plt.savefig(os.path.join(savepath,nameB),clobber=True)
plt.close()