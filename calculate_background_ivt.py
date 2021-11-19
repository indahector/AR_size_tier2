""" Calculte the ENSO Longitude Index (ELI) following Williams and Patricola (2018, GRL). 

Written by Travis A. O'Brien <TAOBrien@lbl.gov>
Copyright University of California (2019).  All Rights Reserved.

"""

import numpy as np


def calculate_background_ivt(ivt,
                             lat2d,
                             lon2d,
                             arci,
                             background):
    """ Calculates the background ivt following Inda-Diaz et al. (2020, JGR Atmospheres).

        input:
        ------

            ivt                : the timeseries of IVT data
                                 (dimensions should be [time, lat, lon])

            lat2d              : the latitude field.
                                 Values should be in [-90,90].
                                 (dimensions should be [lat, lon])

            lon2d              : the longitude field
                                 Values should be in [0,360].
                                 (dimensions should be [lat, lon])
                                 
            background         : which version of background to calculate (1-4)
        output:
        -------

            ivt_bk            : the ivt_bk field (dimensions [time,sampled_points]) (sampled points 6570 for now)

        raises:
        -------

            ValueError if any values are outside the expected range

            RuntimeError if any of the arrays have an inconsistent or unexpected shape

            TypeError if any of the arrays don't behave like numpy ndarrays


        This follows the algorithm outlined in Inda-Diaz et al. (2020, JGR Atmospheres).  For each timestep:

        Written by Inda-Diaz H.A. <indahector@gmail.com>
        Copyright University of California (2021).  All Rights Reserved.

    """

#     print(np.shape(ivt))
#     # do some duck type checking on the input values
#     for varname, field in zip(["ivt", "arci", "lat2d", "lon2d"], [ivt, arci, lat2d, lon2d]):
#         try:
#             np.asarray(field)
#             field.shape
#         except:
#             raise TypeError(
#                 "casting `{}` as a numpy array failed; is it a numpy-like array?".format(varname))

#     # check that arrays have the expected ranks
#     if len(ivt.shape) != 3:
#         raise RuntimeError(
#             "ivt has rank {}, but a rank of 3 is required".format(len(ivt.shape)))
#     # check that arrays have the expected ranks
#     if len(arci.shape) != 3:
#         raise RuntimeError(
#             "arci has rank {}, but a rank of 3 is required".format(len(ivt.shape)))
#     if len(lat2d.shape) != 2:
#         raise RuntimeError(
#             "lat2d has rank {}, but a rank of 2 is required".format(len(lat2d.shape)))
#     if len(lon2d.shape) != 2:
#         raise RuntimeError(
#             "lon2d has rank {}, but a rank of 2 is required".format(len(lon2d.shape)))

    # check that array shapes are consistent
    nlat, nlon = ivt.shape
    for varname, field in zip(["arci", "lat2d", "lon2d"], [arci, lat2d, lon2d]):
        if field.shape[0] != nlat or field.shape[1] != nlon:
            raise RuntimeError("`{}` has shape {}, but a shape of [{},{}] is required.".format(
                varname, field.shape, nlat, nlon))

    # check that lat/lon values are in the expected range
    if lat2d.min() < -90 or lat2d.max() > 90:
        raise ValueError(
            "`lat2d` has values in the range [{},{}], but it's values should be in the range [-90,90]".format(lat2d.min(), lat2d.max()))
    if lon2d.min() < 0 or lon2d.max() > 360:
        raise ValueError("`lon2d` has values in the range [{},{}], but it's values should be in the range [0,360]".format(
            lon2d.min(), lon2d.max()))

    # Gaussian band filter for tropics
    W = 15.
    gaussian_band   =  1. - np.exp(-lat2d**2./(2.*W**2))
        
    # **********************************************************
    # 2. Find grid cells where arci is less than 0.5
    #    less than half algorithms found ivt
    # **********************************************************
    
    ny,nx = np.shape(ivt)
    
    # Subsample
#     size=int(5000)
    size=int(1000)
    
    if background==1:
        # Sample IVT from everywhere without filtering
        sample = ivt
        random = np.random.choice(np.arange(len(sample.ravel())), size=size, replace=False)
        ivt_bk = sample.ravel()[random]

    if background==2:
        # Sample IVT without filtering, except where there ar ARs
        # Filter AR locations
        ind = np.where(arci<0.5)
        sample = ivt[ind]
        random = np.random.choice(np.arange(len(sample)), size=size, replace=False)
        ivt_bk = sample[random]

    if background==3:
        # Sample IVT from everywhere after filtering the equator using gaussian band
        indFilt = np.where(np.abs(lat2d)>=20)
        sample = ivt[indFilt]
#         sample = gaussian_band*ivt
        random = np.random.choice(np.arange(len(sample.ravel())), size=size, replace=False)
        ivt_bk = sample.ravel()[random]

    if background==4:
        # Sample IVT from places without AR after filtering the equator using gaussian band
        # Filter AR locations
        ind = arci<0.5
        ind = ind*np.abs(lat2d)>=20
        indFilt = np.where(ind)
        sample = ivt[indFilt]
#         sample = (gaussian_band*ivt)[ind]
        random = np.random.choice(np.arange(len(sample)), size=size, replace=False)
        ivt_bk = sample[random]
    
    return ivt_bk

    
    