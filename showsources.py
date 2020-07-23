#!/usr/bin/env python
from astropy.io import fits
import numpy as np
import sys


#visit, ccd = int(sys.argv[1]), int(sys.argv[2])

#hdulist = fits.open("/mnt/home/faculty/csurhud/P9_redn/Dec_2018/rerun/diffimage_debug8/deepDiff/02529/HSC-R2/DIASRC-%07d-%03d.fits" % (visit, ccd) )
hdulist = fits.open("DIASRC-0153442-000.fits")
data = hdulist[1].data

flux = data["base_PsfFlux_instFlux"]
err = data["base_PsfFlux_instFluxErr"]
nchild = data["deblend_nchild"]
snr = np.absolute(flux/err)

for cols in data.columns:
    print(cols.name)

#fout = open("/mnt/home/faculty/csurhud/sources/src-%07d-%03d.txt" % (visit, ccd), "w")
#fout.write("fk5\n")
ii=0
i=0
for ra, dec in zip(180./np.pi*data["coord_ra"], 180./np.pi*data["coord_dec"]):
    #if data["flags"][ii][60-1]: # or snr[ii]<7.0:
    if nchild[ii] > 0 :
        pass
    else:
        #fout.write("circle %.5f %.5f 2\"\n" % (ra, dec))
        print("circle %.5f %.5f 2\"\n" % (ra, dec))
        i = i + 1
    ii = ii + 1

#fout.close()
