# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:14:09 2022

@author: jevans
"""


from PIL import Image
import os

from skimage.filters import try_all_threshold, threshold_mean,threshold_otsu
from skimage.color import rgb2gray
from skimage.filters import sobel
from scipy import ndimage as ndi

from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  watershed,
                                  mark_boundaries)

from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np
import cv2

import scipy.ndimage as ndimage 

import math   



impath = "C:\\Users\\jevans\\Documents\\GitHub\\leaf_temperature_thermometer\\images"
allimnames=os.listdir(impath)
i=6
imname=allimnames[i]

minsize=750

img = Image.open(os.path.join(impath,imname))
img=np.array(img)
if len(img.shape)>2:
    img = rgb2gray(img)
    
#fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
#plt.show()

thresh=threshold_otsu(img)
threshim=(img<thresh)+0

threshim=sobel(threshim)

threshim=ndimage.binary_fill_holes(threshim)



label_image = label(threshim)

allregions=regionprops(label_image,img)
allresults={}

allregions=[x for x in allregions if x.area>=minsize]    

for i in range(len(allregions)):
    region=allregions[i]
    allresults.update({region.label:{"bbox":region.bbox,"centroid":region.centroid,"area":region.area,"intensity_mean":region.intensity_mean}})



    
        
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(threshim, cmap=plt.cm.gray)

for props in allregions:
    y0, x0 = props.centroid
    #orientation = props.orientation
    #x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    #y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    #x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    #y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
    
    ax.text(y0, x0,color=)

    #ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    #ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

#ax.axis((0, 600, 600, 0))
plt.show()
        





       




