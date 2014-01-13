# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:54:10 2014

@author: atc327
"""
import os, glob
import scipy as sp
import numpy as np
from PIL import Image

if __name__ == '__main__':  
    
    pts_s = [ [1485,559],[2421,711],[3445,551]]
    pts_e = [ [1490,564],[2426,716],[3453,558]]
    path = 'C:/Users/atc327/Desktop/images_ggd/'
    R = []
    G = []
    B = []
    for infile in glob.glob( os.path.join(path, '*.jpg') ):
        result_avg_R = []
        result_avg_G = []
        result_avg_B = []
        im = np.array(Image.open(infile)).astype(np.float)
        
        for idx in arange(len(pts_s)):

            im_cut = im[pts_s[idx][1]:pts_e[idx][1],pts_s[idx][0]:pts_e[idx][0]]
            im_cutR = im_cut[0:,0:,0]
            im_cutG = im_cut[0:,0:,1]
            im_cutB = im_cut[0:,0:,2]
            
            avg_R = sum(im_cutR.flatten())/len(im_cutR.flatten())
            avg_G = sum(im_cutG.flatten())/len(im_cutG.flatten())
            avg_B = sum(im_cutB.flatten())/len(im_cutB.flatten())
            
            result_avg_R.append(avg_R)
            result_avg_G.append(avg_G)
            result_avg_B.append(avg_B)
    
        R.append(result_avg_R)    
        G.append(result_avg_G)
        B.append(result_avg_B)
    
    R_T = np.transpose(R)
    G_T = np.transpose(G)
    B_T = np.transpose(B)
    
    for idx in arange(len(pts_s)):
        figure(1),
        title('R')
        plot(arange(165),R_T[idx])
       
        figure(2),
        title('G')
        plot(arange(165),G_T(idx))
        
        figure(3),
        title('B')
        plot(arange(165),B_T(idx))