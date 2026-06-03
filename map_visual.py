#prints out the map

from pixell import enmap, enplot, utils, reproject
from astropy.io import fits
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ymap = enmap.read_map("/Users/jerrywang/Documents/Battaglia_research/Project_tsz_stacking/act-planck_dr6.02_nilc_ComptonY_deproj_cib_1.2_24.0.fits") 

def eshow(x,**kwargs):
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "ipython")

eshow(ymap, downgrade=3, ticks=30, colorbar=True, font_size=60)
