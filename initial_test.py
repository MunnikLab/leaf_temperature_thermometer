# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:14:09 2022

@author: jevans
"""


from PIL import Image
import os
from pandas import DataFrame
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage import filters
from skimage.measure import label, regionprops
from skimage import morphology
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage


def FindLeaves(fullimpath, exportpath, minsize=1000, boundary=20, searchsize=10):
    """
    

    Parameters
    ----------
    fullimpath : string
        Full path of image location.
    exportpath : TYPE
        Full path of export location.
    minsize : TYPE, optional
        Minimum blob area on initial pass. The default is 1000.
    boundary : TYPE, optional
        Minimum distance of centroid away from the image edge. The default is 20.
    searchsize : TYPE, optional
        Search size in pixels when joining seperated areas. The default is 10.

    Returns
    -------
    Returns dataframe containing area (n. pixels) and the mean, min, max 
    intensities.
    Exports images and results to appropriate exportpath.

    """

    picname = os.path.splitext(os.path.split(fullimpath)[1])[0]
    exportpath = os.path.join(exportpath, picname)

    oimg = Image.open(fullimpath)
    img = np.array(oimg)
    if len(img.shape) > 2:
        img = rgb2gray(img)

    #fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    # plt.show()

    thresh = threshold_otsu(img)
    threshim = (img < thresh)+0

    sobelim = sobel(threshim)

    threshim = ndimage.binary_fill_holes(sobelim)

    largerblobs = morphology.binary_closing(
        threshim, np.ones((searchsize, searchsize)))

    label_image = label(largerblobs)

    allregions = regionprops(label_image, img)

    allresults = {}

    allregions = [x for x in allregions if (x.area >= minsize) and  # area is large enough
                  # centroid is far enough from the edge
                  x.centroid[1] > boundary and
                  # centroid is far enough from the edge
                  x.centroid[1] < (img.shape[1]-boundary)
                  # centroid is far enough from the edge
                  and x.centroid[0] > boundary and
                  x.centroid[0] < (img.shape[0]-boundary)]  # centroid is far enough from the edge

    # genereate initial dataframe

    for j in range(len(allregions)):
        region = allregions[j]
        # don't trust the area here because we now
        allresults.update(
            {region.label: {"bbox": region.bbox, "centroid": region.centroid}})

    ally = [allresults[x]["centroid"][1] for x in allresults]
    hist, bins = np.histogram(ally, bins=6)
    indices = np.digitize(ally, bins)
    indices[indices > 6] = 6

    for j in range(len(indices)):
        x = list(allresults.keys())[j]
        allresults[x]["row"] = indices[j]

    xrow = {x: (allresults[x]["centroid"][0], allresults[x]["row"])
            for x in allresults}
    xroworder = [np.argsort([xrow[x][0] for x in xrow if xrow[x][1] == y])
                 for y in np.sort(list(set(indices)))]

    xrowid = [[x for x in xrow if xrow[x][1] == y]
              for y in np.sort(list(set(indices)))]

    xroworder = [item for sublist in xroworder for item in sublist]
    xrowid = [item for sublist in xrowid for item in sublist]

    for i in range(len(xrowid)):
        currid = xrowid[i]
        currcol = xroworder[i]
        allresults[currid]["col"] = currcol

    os.makedirs(exportpath, exist_ok=True)
    os.makedirs(os.path.join(exportpath, "individuals"), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(oimg, cmap=plt.cm.gray)

    for j in range(len(allresults)):
        props = allregions[j]
        currkey = list(allresults.keys())[j]
        y0, x0 = props.centroid
        #orientation = props.orientation
        #x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        #y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        #x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        #y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.text(x0, y0, str(allresults[currkey]["col"]) +
                ","+str(allresults[currkey]["row"]), color='r')
        ax.text(
            x0, y0+10, str(6*allresults[currkey]["col"]+allresults[currkey]["row"]), color='r')

        #ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        #ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, img.shape[1], img.shape[0], 0))
    plt.savefig(os.path.join(exportpath, "segments"), bbox_inches='tight')

    img3 = np.empty(img.shape)
    img3[:] = np.nan

    img2 = img.copy()
    newresults = {}
    # for each plant, pull out just that sections of the image
    for k, v in allresults.items():
        minr, minc, maxr, maxc = v["bbox"]
        currim = img2[minr:maxr, minc:maxc]
        # inverting the image so the plants are white and the bg is black
        # currim=255+util.invert(currim)

        blurredim = filters.gaussian(currim, sigma=1.0)

        currthresh = threshold_otsu(blurredim)

        currthreshim = (blurredim <= currthresh)+0
        newim = currim.copy()
        newim[currthreshim == 0] = 0

        currsobelim = sobel(newim)

        currthreshim = ndimage.binary_fill_holes(currsobelim)

        newim = currim.copy()
        newim[currthreshim == 0] = np.nan
        newv={}
        n = 6*v["col"]+v["row"]
        newv["imname"]=imname
        newv["n"] = n
        newv["row"]=v["col"]
        newv["col"]=v["row"]
        newv["area"] = np.sum(currthreshim > 0)
        newv["mean"] = np.nanmean(newim)
        newv["max"] = np.nanmax(newim)
        newv["min"] = np.nanmin(newim)
        newresults.update({k: newv})

        img3[minr:maxr, minc:maxc] = newim

        # save individual images
        exportim = Image.fromarray((newim*255.9999).astype(np.uint8))
        exportim.save(os.path.join(exportpath, "individuals", str(n)+".png"))

    # save the results
    df = DataFrame.from_dict(newresults, orient='index')
    # save
    df.to_csv(os.path.join(exportpath, "results.csv"))
    # save individual images
    exportim = Image.fromarray((img3*255.9999).astype(np.uint8)).convert('RGB')
    exportim.save(os.path.join(exportpath, "finalimage.png"))
    exportim = Image.fromarray((img*255.9999).astype(np.uint8))
    exportim.save(os.path.join(exportpath, "originalimage.png"))
    return df

impath = "C:\\Users\\jevans\\Documents\\GitHub\\leaf_temperature_thermometer\\images"
exportpath = "C:\\Users\\jevans\\Documents\\GitHub\\leaf_temperature_thermometer\\exportedimages"
allimnames = os.listdir(impath)


for i in range(len(allimnames)):
    imname = allimnames[i]
    fullimpath = os.path.join(impath, imname)
    FindLeaves(fullimpath, exportpath)
