# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:31:32 2014

@author: atc327
"""
from images2gif import writeGif
from PIL import Image
import os

fl = pickle.load(file('filelist.pkl'))
start = "10/26/13 19:00:00"
end   = "10/27/13 05:00:00"
paths,files,times = fl.time_slice(start,end)

if __name__ == '__main__': 
    pts = [[178,543],[256,541],[286,543],[500,25],[1585,365],[2556,547],[3579,409]]
    fn1 = "s1.GIF"
    fn2 = "s2.GIF"
    fn3 = "s3.GIF"
    fn4 = "s4.GIF"
    fn5 = "s5.GIF"
    fn6 = "s6.GIF"
    fn7 = "s7.GIF"
    image=[];
    for idx in range(len(paths))
        p = paths[idx]  
        f = paths[idx]   
        infile = os.path.join(p,f)
        image.append(infile) 
    

        imc_s1 = im[pts[0][1]-21:pts[0][1]+39,pts[0][0]-21:pts[0][0]+39]

        writeGif(fn1, imc_s1, duration=0.2)

