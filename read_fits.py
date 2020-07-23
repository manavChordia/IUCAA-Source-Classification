# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:47:20 2020

@author: ManavChordia
"""
import astropy.visualization
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.exposure as skie
from astropy.visualization import (AsinhStretch, MinMaxInterval, ImageNormalize)
from astropy.wcs import WCS
"""
def normalize(image):
    maxi = image.max()
    mini = image.min()
    
    image = (image - mini)/(maxi-mini)
    return image
"""

# suppose image file name is image.fits
hdulist = pyfits.open("DIASRC-0153442-028.fits")
image = hdulist[1].data

n=0
for i in range(185):
    if int(image['base_SdssCentroid_x'][i]) not in range(int(image['base_SdssCentroid_x'][i-1])-32,int(image['base_SdssCentroid_x'][i-1])+32):
        print(image['base_SdssCentroid_x'][i])
        n=n+1

"""
#image = image[19:4156, 19:2028]if

# Create interval object
interval = MinMaxInterval()
vmin, vmax = interval.get_limits(image)

# Create an ImageNormalize object using a SqrtStretch object
#for i in range 
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(0.2))

# Display the image
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image, origin='lower', norm=norm)
fig.colorbar(im)

stretch_image = image

import math
for i in range(0,image.shape[0]):
    for j in range(0, image.shape[1]):
        stretch_image[i][j] = math.asinh(stretch_image[i][j])
 
       
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image, origin='lower')
fig.colorbar(im)




scale_percent = 15 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image



limg = np.arcsinh(image)
limg = limg / limg.max()
low = np.percentile(limg, 0.25)
high = np.percentile(limg, 99.5)
opt_img  = skie.exposure.rescale_intensity(limg, in_range=(low,high))

scale_percent = 35 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


cv2.imshow("image", resized_image)


image_normalised = image
image_normalised = normalize(image_normalised)

resized_image_n = cv2.resize(image_normalised, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("image1", resized_image_n)


cv2.waitKey(0)
cv2.destroyAllWindows()
#axarr[1].imshow(image_normalised)


cnt = 0
for i in range(0,image.shape[0]):
    for j in range(0, image.shape[1]):
        if(image[i,j]>10):
            cnt=cnt+1
            print(i,j)
"""

# image will be a 2d array

# list with source catalogs
hdulist2 = pyfits.open("DIASRC-0153442-028.fits")
table = hdulist2[1].data

flags = []

for i in range(0,table.shape[0]):
    flags.append(table[i][0])
    
flags_array = np.array(flags)

col = []

for i in range(1, 115):
    temp = []
    for j in range(0, table.shape[0]):
        temp.append(table[j][i])
    col.append(temp)
        
col_array = np.array(col)
col_array = col_array.transpose()

array = np.concatenate((flags_array,col_array),axis=1)


cnt=0

x = table['base_SdssCentroid_x']
x = np.round(x)
x = x.astype(int)
y = table['base_SdssCentroid_y']
y = np.round(y)
y = y.astype(int)


"""
for i in range(0,1039):
    if array[i][88] == 0:
        if str(array[i][89]) != 'nan':
            print(str(array[i][89])+','+str(array[i][90]))
            cnt=cnt+1
        
import numpy as np
ext = np.load('ext.npy')


extplot = plt.imshow(ext)
"""
# These kind of commands will you give you the arrays.
# Check hdulist[1].header.columns for the different column names.
"""
wcs = WCS(hdulist2[0].header)

from astropy.coordinates import SkyCoord
from astropy.wcs import utils
import astropy.units as u

#cor = wcs.wcs_world2pix([[64.0715784,11.618953832]], 0)

c = SkyCoord(64.0715784*u.deg, 11.61895*u.deg, frame='icrs', unit='deg')
target = utils.skycoord_to_pixel(c, wcs)
coord = SkyCoord(1.11802, 0.208, frame="icrs")

target = utils.skycoord_to_pixel([64.0715784,11.618953832], wcs)
"""