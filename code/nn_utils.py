import numpy as np
import scipy.ndimage as ndimage

from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin

def sq_fluxradius(p,patch,fraction):
    """
    Return the square error between the flux ratio in
    for a given radius and the desired 'fraction' 
    """
    flux  = np.sum(patch)
    nx,ny = np.shape(patch)
    x,y   = np.meshgrid(range(nx),range(ny))
    ind   = np.sqrt((x-(nx-1)/2.)**2+(y-(ny-1)/2.)**2) < p
    finr  = np.sum(patch[ind])
    return (finr/flux-fraction)**2.

def sq_nearest(values):
    return np.sum((values-values[len(values)/2])**2.)

def lnlike(modelpatch,data,sig_smooth,sig_L2,sig_one,w_L2):
    """
    Return the negative log-likelihood given a pixel patch across a
    given set of data patches, weighted by regularization priors.
    Uniform noise case.
    """
    
    # Likelihood given current psf model
    lnlike = 0.0
    for ii in range(data.npatches):
        patch = np.ravel(data.patches[ii])
        flux  = np.dot(modelpatch.T,patch)/np.dot(modelpatch.T,modelpatch)
        model = modelpatch*flux
        lnlike += np.sum(0.5*(patch-model) ** 2
                         / data.bkg_sigmas[ii]**2. + \
                         0.5 * np.log(data.bkg_sigmas[ii]**2.))

    # Smoothness constraint
    if sig_smooth!=0:
            filt = np.array([[False,True,False],
                             [True,True,True],
                             [False,True,False]])
            nearest = ndimage.generic_filter(np.reshape(modelpatch,data.patchshape),
                                     sq_nearest, footprint=filt)
            lnlike  += np.sum(nearest) * sig_smooth

    # L2 norm
    if sig_L2!=0:
        lnlike += np.sum((modelpatch*w_L2)**2.) * sig_L2

    # PSF total ~ 1
    if sig_one!=0:
        lnlike += (np.sum(modelpatch)-1)**2. * sig_one

    return lnlike

