# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:13:11 2014

@author: atc327
"""

import os, glob,pickle,sys
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize
from dst13 import *
from multiprocessing import Pool





def red_analysis(idx):
    print("Hi {0}".format(idx) )
    p = paths[idx]  
    f = paths[idx]
   
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
        
    return result_avg_R,result_avg_G,result_avg_B

fl = pickle.load(file('filelist.pkl'))
start = "10/26/13 19:00:00"
end   = "10/27/13 05:00:00"
paths,files,times = fl.time_slice(start,end)

if __name__ == '__main__': 
    pool = Pool(processes=24)

    pts = [[178,543],[256,541],[286,543],[500,25],[1585,365],[2556,547],[3579,409]]
   
    Result = [] 
    Result.append(pool.map(red_analysis,range(len(paths))))
    Result = Result[0]    
    R = np.transpose([row[0] for row in Result])
    G = np.transpose([row[1] for row in Result])
    B = np.transpose([row[2] for row in Result])

    cmap = 'gist_heat'
    interp = 'nearest'
    clim = [-255,255]
    savepath = os.environ['DST_WRITE']
    nameR= 'R.png'
    nameG= 'G.png'
    nameB= 'B.png'
    
    for idx in arange(len(pts)):   
        plot(arange(len(paths)),R[idx])
    title('R')
    plt.savefig(os.path.join(savepath,nameR),clobber=True)
    plt.close()
    
    for idx in arange(len(pts)):       
        plot(arange(len(paths)),G(idx))
    title('G') 
    plt.savefig(os.path.join(savepath,nameG),clobber=True) 
    plt.close()  
    
    for idx in arange(len(pts)):          
        plot(arange(len(paths)),B(idx))
    title('B')    
    plt.savefig(os.path.join(savepath,nameB),clobber=True)
    plt.close()