import matplotlib as mpl
import matplotlib.pylab as PP
import numpy as np
import sys
import glob
import os
import xarray as xr
import cartopy.crs as ccrs
from fastkde import fastKDE
import fastKDE_plot_modified as fastkde_plot
from scipy import interpolate
import tempfile
import gc
import time
from netCDF4 import date2num,num2date
from KS_stats import *

#--------------------------------------------------------------------------------------
        
#--------------------------------------------------------------------------------------
# Define runs
runs = ['cmip5_CCSM4_historical',         'cmip5_CCSM4_rcp85', \
        'cmip5_CSIRO-Mk3-6-0_historical', 'cmip5_CSIRO-Mk3-6-0_rcp85', \
        'cmip5_CanESM2_historical',       'cmip5_CanESM2_rcp85', \
        'cmip5_IPSL-CM5A-LR_historical',  'cmip5_IPSL-CM5A-LR_rcp85', \
        'cmip5_IPSL-CM5B-LR_historical',  'cmip5_IPSL-CM5B-LR_rcp85', \
        'cmip5_NorESM1-M_historical',     'cmip5_NorESM1-M_rcp85', \
        'cmip6_BCC-CSM2-MR_historical',   'cmip6_BCC-CSM2-MR_ssp585', \
        'cmip6_IPSL-CM6A-LR_historical',  'cmip6_IPSL-CM6A-LR_ssp585', \
        'cmip6_MRI-ESM2-0_historical',    'cmip6_MRI-ESM2-0_ssp585']

#-------------------------------------------------------------------------------------------------------------------------------------
# Functions to find file names of data
output_fName = lambda run: "output/runs/{}.nc".format(run)

#-------------------------------------------------------------------------------------------------------------------------------------
# # Functions to find files and open dataset for ivt backgrouds
# def open_ivtbk(run):
#     files = np.sort(glob.glob("output/ivtbk/*{}*".format(run)))
# #     output=[]
# #     for f in files:
# #         print(f)
# #         output.append(xr.open_dataset(f))
#     output = xr.open_mfdataset(files, combine="nested", concat_dim="realization", decode_times=False)
    
#     root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/{}/*".format(run)
#     f0 = np.sort(glob.glob(root))[0]
    
#     with xr.open_dataset(f0, decode_times=False) as fin:
#         output.time.attrs["calendar"] = fin.time.calendar
    
#     return output
        
#-------------------------------------------------------------------------------------------------------------------------------------
# Functions to find files and open dataset for ivt backgrouds
def open_ivtbk(run, case=1):
    files = np.sort(glob.glob("output/ivtbk/*{}*{}.nc".format(run,case)))[0]
#     output=[]
#     for f in files:
#         print(f)
#         output.append(xr.open_dataset(f))
    output = xr.open_dataset(files, decode_times=False)
    
    root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/{}/*".format(run)
    f0 = np.sort(glob.glob(root))[0]
    
    with xr.open_dataset(f0, decode_times=False) as fin:
        output.time.attrs["calendar"] = fin.time.calendar
    
    return output    
    
#--------------------------------------------------------------------------------------
# Funcrtion to find where ivt contour crosses background ivt value
def xroots_from_ivt(x, ivt, ivtBK):
        
    ind = np.logical_not(np.isnan(ivt))
    ivt = ivt[ind]
    x = x[ind]
    
    roots = x[np.gradient(np.sign(ivt-ivtBK)).astype("bool")]
    left  = roots[roots<0].mean()
    right = roots[roots>0].mean()
     
    return left, right


#--------------------------------------------------------------------------------------
# Funcrtion to calculate the IVT-dist CDF from the collection of points along
# and across the AR sampled over the PC directions
def CDF(X,IVT,cdf_levels=[0.045, 0.16, 0.5, 0.84, 0.955], numPoints=None, subSample=10):
    
    if subSample is not None:
        sample = int(len(X)/subSample)
        ind = np.random.choice(np.arange(len(X)), size=sample, replace=False)
        X = X[ind]
        IVT = IVT[ind]
    
#     print(len(IVT))
#     print(len(np.where(IVT==np.nan)[0]))
#     print(len(np.where(IVT==np.inf)[0]))
#     print(IVT)
#     print()
    
    bivariate_pdfs, bivariate_axes = fastKDE.conditional(IVT, X ,numPoints=numPoints)
    bivX   = bivariate_axes[0]
    bivIVT = bivariate_axes[1]

    conditionX = np.logical_not((bivX>=np.min(X))&(bivX<=np.max(X)))
    conditionI = np.logical_not((bivIVT>=np.min(IVT))& (bivIVT<=np.max(IVT)))
    
    bivX = np.ma.masked_where(conditionX, bivX)
    bivIVT = np.ma.masked_where(conditionI, bivIVT)
    
    conditionX,conditionI = np.meshgrid(conditionX, conditionI)
    condition = np.logical_or(conditionI, conditionX)
    
    bivariate_pdfs = np.ma.masked_where(condition, bivariate_pdfs)
    
    pdf_to_plot = \
       fastkde_plot.cumulative_integral(bivariate_pdfs,\
                                [bivariate_axes[0],bivariate_axes[1]],integration_axes=1)
#     pdf_to_plot = np.ma.masked_where(condition, pdf_to_plot)

    cdf_ivt_x = \
       np.ma.masked_where(np.logical_not(np.isclose(np.ma.ones(pdf_to_plot.shape) * \
                          pdf_to_plot[-1,:][np.newaxis,:],1.0)),pdf_to_plot)
    cdf_ivt_x = np.ma.masked_where(condition, cdf_ivt_x)
    cdf_ivt_x = np.clip(cdf_ivt_x,0,1)

    return bivX, bivIVT, cdf_ivt_x

#------------------------------------------------------------------------------------------------------
def smooth(y, box_pts, mode="valid"):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode=mode)
    return y_smooth

#------------------------------------------------------------------------------------------------------
def extract_one_contour(x, ivt, cdf, val=None):

    import numpy as np
    import scipy.interpolate as interp
    from scipy.signal import savgol_filter

    val0 = val-0.03
    val1 = val+0.03
    ind = np.where((cdf>=val0)&(cdf<=val1)&(ivt>=0))
    x0 = x[ind[1]]
    y0 = ivt[ind[0]]
    
    rand = (np.random.rand(len(x0))-.5)/1000
    x = x0+rand
    y = np.copy(y0)
    
    sort = x.argsort()
    y = y[sort]
    x = x[sort]
    
    if len(x)!=len(np.unique(x)):
        x2 = []
        y2 = []
        for i in range(len(x)):
            if x[i] not in x2:
                x2.append(x[i])
                y2.append(y[i])
        x = x2
        y = y2
    
    win = int(len(x)/10)
#     print(win)
    if win%2==0: win+=1

    y = smooth(y, win)
    x = smooth(x, win)

#     fig,ax=PP.subplots()
#     ax.plot(x0,y0,'.')
#     ax.plot(x,y)

    
    xi = np.linspace(-1e4, 1e4, int(2e4)+1)
#     ius = interp.UnivariateSpline(x, y, k=3, ext=3)
#     yi = ius(xi)

    from scipy.interpolate import interp1d
    f = interp1d(x, y, kind='cubic', bounds_error=False)
    yi = f(xi)
    
    return x0,y0,xi,yi

#------------------------------------------------------------------------------------------------------
# Function to extract the IVT-dist CDF contours at specific cdf levels
def extract_contours(X,IVT,CDF,cdf_levels=[0.045, 0.16, 0.5, 0.84, 0.955]):
    
    x,y = {},{}
    pdf_contours_x,pdf_contours_y = {},{}
    
    for k in cdf_levels:
        x[k],y[k],pdf_contours_x[k],pdf_contours_y[k] = extract_one_contour(X, IVT, CDF, val=k)

    return x, y, pdf_contours_x, pdf_contours_y

# #------------------------------------------------------------------------------------------------------
# # Function to extract the IVT-dist CDF contours at specific cdf levels
# def extract_contours(X,IVT,cdf_levels=[0.045, 0.16, 0.5, 0.84, 0.955]):
    
#     fig, axs, marginal_vals, marginal_pdfs, bivariate_pdfs, \
#     levels, limits = fastkde_plot.pair_plot([IVT, X], \
#                                             conditional=True, var_names=['IVT', 'x'],\
#                                             cdf_levels=cdf_levels,auto_show = False)    
#     ax0,ax1,ax2,ax3 = axs.ravel()
    
#     pdf_contours_x = {}
#     pdf_contours_y = {}
#     keys = cdf_levels

#     for i,col in enumerate(ax1.collections):
#         paths = col.get_paths()
#         vert = np.array([len(paths[i].vertices.ravel()) for i in range(len(paths))]).argmax()
#         v = paths[vert].vertices
#         x = v[:,0]
#         y = v[:,1]
#         pdf_contours_x[keys[i]] = x
#         pdf_contours_y[keys[i]] = y
    
#     return pdf_contours_x, pdf_contours_y


#--------------------------------------------------------------------------------------
# Function to plot IVT-dist CDF contours for along and across the AR directionos
def plot_CDF_contours(xContours0, yContours0, xContours1, yContours1,\
                      cdf_levels=[0.045, 0.16, 0.5, 0.84, 0.955], ivtBK = None):   
    
    IVT, DIST = [yContours0,yContours1],[xContours0,xContours1]
    minIVT, maxIVT = 0, 1300

    xLimits  = [[-4500,4500],[-2000,2000]]
#     xLimits  = [[-4500,4500],[-4500,4500]]
    XTICKS = [[-4000, -3000, -2000, 0, 2000, 3000, 4000], [-2000, -1500, -1000, 0, 1000, 1500,  2000]]
    PC_names = ["Along","Across"]
    measures = ["Length={:.0f} km","Width={:.0f} km"]
    textX = [-4200, -1850]
    textY = 1150
    
    fig,axs=PP.subplots(1,2,figsize=(12,4))
    ax0,ax1=axs
    pCritic = 0.16
    ivtBK = 200
    
    for i,ax in enumerate(axs):
        for k in cdf_levels:
            ax.plot(DIST[i][k],IVT[i][k], label="p="+str(k))

        left, right = xroots_from_ivt(DIST[i][pCritic], IVT[i][pCritic], ivtBK)
        boundaries = [left,right]
        size = np.abs(left-right)
        text = measures[i]
        ax.text(textX[i], textY, text.format(size), fontsize=12)
        
        xticks = np.sort(np.concatenate((XTICKS[i],boundaries))).astype("int")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks,rotation=90)
        
        yticks = np.array([0,ivtBK, 500, 1000, 1500, 2000],dtype="int")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xlim(xLimits[i])
        ax.set_ylim(minIVT,maxIVT)
        
        xdum = np.linspace(xLimits[i][0],xLimits[i][1],11)
        ydum = xdum*0 + ivtBK
       
        ax.plot(xdum, ydum, "--k", alpha = 0.75, linewidth=0.9, label = "Bkg IVT")
        ax.plot([boundaries[0],boundaries[0]],[minIVT,maxIVT], "--k", alpha = 0.4, linewidth=1)
        ax.plot([boundaries[1],boundaries[1]],[minIVT,maxIVT], "--k", alpha = 0.4, linewidth=1)
        
        ax.set_xlabel("Distance to AR centroid [km] ({} AR)".format(PC_names[i]))
        ax.set_ylabel("IVT CDF [kg m$^{-1}$s$^{-1}$]")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=10,loc=1)
  
    fig.tight_layout()

    return

#--------------------------------------------------------------------------------------
# Function to plot IVT conditional to distance as a function of AR life cycle stage
def plot_mean_cdf_stages(xContours0, yContours0, xContours1, yContours1, pCritic=0.5):   
    
    IVT, DIST = [yContours0,yContours1],[xContours0,xContours1]
    
    stage_keys = [10, 25, 50, 75, 90]
    
    indx0 = np.where(np.abs(xContours0[stage_keys[0]][0.5])<2000)[0]
    indx1 = np.where(np.abs(xContours1[stage_keys[0]][0.5])<2000)[0]
    
    maxIVT = 0
    for j,k in enumerate(stage_keys):
        maxIVT = max(maxIVT, max(np.max(yContours1[k][pCritic][indx1]), np.max(yContours0[k][pCritic][indx0])) ) 
    minIVT, maxIVT = 0, maxIVT+100

    #minIVT, maxIVT = 0, 1250
    xLimits  = [[-5000,5000],[-2500,2500]]
    XTICKS = [[-4000, -3000, -2000, 0, 2000, 3000, 4000], [-2000, -1500, -1000, 0, 1000, 1500,  2000]]
    PC_names = ["Along","Across"]
    measures = ["Length={:.0f} km","Width={:.0f} km"]
    textX = [-4200, -1850]
    textY = 1750
    
    ivtBK = 200
    
    fig,axs=PP.subplots(1,2,figsize=(12,4))
    ax0,ax1=axs
#     colors = mpl.cm.tab20b(np.linspace(0,1,len(stage_keys)))
    colors = mpl.cm.jet(np.linspace(0,1,len(stage_keys)))
    
    for i,ax in enumerate(axs):        
        for j,k in enumerate(stage_keys):
            ax.plot(DIST[i][k][pCritic],IVT[i][k][pCritic], label="{} %".format(k), color=colors[j],\
                    linewidth=1.5)
      
        ax.set_xlim(xLimits[i])
        ax.set_ylim(minIVT,maxIVT)
        ax.set_xlabel("Distance to AR centroid [km] ({} AR)".format(PC_names[i]))
        ax.set_ylabel("IVT CDF [kg m$^{-1}$s$^{-1}$]")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=10,loc=1)
        ax.set_title("AR lifecycle stage {}. CDF={}".format(PC_names[i],pCritic))
        ax.set_facecolor([0.01,0.01,0.01,0.085])
            
    fig.tight_layout()

    return

#--------------------------------------------------------------------------------------
# Function to plot CDF along and across AR
def plot_CDF(X0, IVT0, CDF0, X1, IVT1, CDF1):
    
    fig,axs = PP.subplots(1,2,figsize=(12,4))
    ax0,ax1 = axs

    levels = np.linspace(0,1,11)

    cf0 = ax0.contourf(X0, IVT0, CDF0, levels=levels)
    ax0.set_xlim(-7500,7500)
    cb0 = fig.colorbar(cf0, ax=ax0)
    ax0.set_title("Along AR")
    
    cf1 = ax1.contourf(X1, IVT1, CDF1, levels=levels)
    ax1.set_xlim(-5000,5000)
    cb1 = fig.colorbar(cf1, ax=ax1)
    ax1.set_title("Across AR")
    
    for ax in axs:
        ax.set_ylim(0,1500)
        ax.set_xlabel("Distance to AR centroid [km]")
        ax.set_ylabel("IVT [kg m-1 s-1]")

    fig.tight_layout()
    
    return

#--------------------------------------------------------------------------------------
# Function to interpolate CDF to regular grid:
def create_regular_cdf(X, IVT, CDF):
    
    nomaskx = np.logical_not(X.mask)
    nomasky = np.logical_not(IVT.mask)

    x = X.data[nomaskx]
    y = IVT.data[nomasky]
    z = CDF.data[nomaskx,:][:,nomasky]

    nx,ny = np.shape(z)
    if nx>ny:
        x = x[0:len(y)]
        z = z[0:len(y), :]
    if ny>nx:
        y = y[0:len(x)]
        z = z[:, 0:len(x)]

    f = interpolate.interp2d(x, y, z, kind='linear')

#     xnew = np.arange(-10000, 10000, 1)
    xnew = np.linspace(-1e4, 1e4, int(2e4)+1)
#     ynew = np.arange(0, 2000, 1)
    ynew = np.linspace(0, 2e3, int(2e3)+1)
    
    znew = f(xnew, ynew)
    
    return xnew, ynew, znew

#--------------------------------------------------------------------------------------
# Function to create CDF, extract contours, and interpolate them to regular grid
def calculate_full_CDF(d0, ivt0, d1, ivt1, stage):

    # Define AR lifecycle stages keys
    stage_keys = (100*np.arange(0.05, 1.00, 0.05)).astype("int")

    # Define stage keys
    inds = {}
    for k in stage_keys:
        inds[k] = np.where((stage>=float(k)/100-.03)&(stage<=float(k)/100+0.03))[0]
    
    # Create saving dictionaries
    X0, IVT0, CDF0 = {},{},{}
    X1, IVT1, CDF1 = {},{},{}
    PDFX0,PDFY0 = {},{} 
    PDFX1,PDFY1 = {},{}

    # Calculate mean CDF and contours
    X0_mean, IVT0_mean, CDF0_mean = CDF(d0, ivt0, numPoints=513, subSample=10)
    X1_mean, IVT1_mean, CDF1_mean = CDF(d1, ivt1, numPoints=513, subSample=10)

    d,d,PDFX0_mean,PDFY0_mean = extract_contours(X0_mean, IVT0_mean, CDF0_mean)
    d,d,PDFX1_mean,PDFY1_mean = extract_contours(X1_mean, IVT1_mean, CDF1_mean)
    
    X0_mean, IVT0_mean, CDF0_mean = create_regular_cdf(X0_mean, IVT0_mean, CDF0_mean)
    X1_mean, IVT1_mean, CDF1_mean = create_regular_cdf(X1_mean, IVT1_mean, CDF1_mean)
    
    # Loop through stages
    for k in stage_keys:

        # Calculate CDF
        X0[k], IVT0[k], CDF0[k] = CDF(d0[inds[k]], ivt0[inds[k]], numPoints=513, subSample=10)
        X1[k], IVT1[k], CDF1[k] = CDF(d1[inds[k]], ivt1[inds[k]], numPoints=513, subSample=10)

        # Extract contours
        d,d,PDFX0[k],PDFY0[k] = extract_contours(X0[k], IVT0[k], CDF0[k])
        d,d,PDFX1[k],PDFY1[k] = extract_contours(X1[k], IVT1[k], CDF1[k])
    
        # Interpolate CDF
        X0[k], IVT0[k], CDF0[k] = create_regular_cdf(X0[k], IVT0[k], CDF0[k])
        X1[k], IVT1[k], CDF1[k] = create_regular_cdf(X1[k], IVT1[k], CDF1[k])
    
    # Return results
    return X0_mean,    IVT0_mean,  CDF0_mean, CDF0,  \
           X1_mean,    IVT1_mean,  CDF1_mean, CDF1,  \
           PDFX0_mean, PDFY0_mean, PDFX0,     PDFY0, \
           PDFX1_mean, PDFY1_mean, PDFX1,     PDFY1

#--------------------------------------------------------------------------------------
# Function to create CDF, extract contours, and interpolate them to regular grid
def calculate_full_CDF_arrays(d0, ivt0, d1, ivt1, stage, verbose=False):

    # Define AR lifecycle stages keys
    stage_keys = (100*np.arange(0.05, 1.00, 0.05)).astype("int")

    # Define stage keys
    inds = {}
    for k in stage_keys:
        inds[k] = np.where((stage>=float(k)/100-.03)&(stage<=float(k)/100+0.03))[0]
    
    # Calculate mean CDF and contours
#     X0_mean, IVT0_mean, CDF0_mean = CDF(d0, ivt0, numPoints=513, subSample=100)
#     X1_mean, IVT1_mean, CDF1_mean = CDF(d1, ivt1, numPoints=513, subSample=100)
    X0_mean, IVT0_mean, CDF0_mean = CDF(d0, ivt0, numPoints=513, subSample=None)
    X1_mean, IVT1_mean, CDF1_mean = CDF(d1, ivt1, numPoints=513, subSample=None)

    d,d,PDFX0_mean,PDFY0_mean = extract_contours(X0_mean, IVT0_mean, CDF0_mean)
    d,d,PDFX1_mean,PDFY1_mean = extract_contours(X1_mean, IVT1_mean, CDF1_mean)
    
    X0_mean, IVT0_mean, CDF0_mean = create_regular_cdf(X0_mean, IVT0_mean, CDF0_mean)
    X1_mean, IVT1_mean, CDF1_mean = create_regular_cdf(X1_mean, IVT1_mean, CDF1_mean)
    
    nk, nx, ni = len(stage_keys), len(X0_mean), len(IVT0_mean)
    cdf_levels=[0.045, 0.16, 0.5, 0.84, 0.955]
    nct = len(cdf_levels)

    # Create numpy arrays for results
    CDF0 = np.zeros((nk,ni,nx))
    CDF1 = np.zeros((nk,ni,nx))
    PDFY0 = np.zeros((nk,nct,nx))
    PDFY1 = np.zeros((nk,nct,nx))
    
    # Loop through stages
    for i,k in enumerate(stage_keys):

        if verbose: 
            print("{} out of {}".format(k,stage_keys[-1]))
            print(len(inds[k]))
        # Calculate CDF
#         X0, IVT0, cdf0 = CDF(d0[inds[k]], ivt0[inds[k]], numPoints=513, subSample=10)
#         X1, IVT1, cdf1 = CDF(d1[inds[k]], ivt1[inds[k]], numPoints=513, subSample=10)
        X0, IVT0, cdf0 = CDF(d0[inds[k]], ivt0[inds[k]], numPoints=513, subSample=None)
        X1, IVT1, cdf1 = CDF(d1[inds[k]], ivt1[inds[k]], numPoints=513, subSample=None)

        # Extract contours
        d,d,d,pdfy0 = extract_contours(X0, IVT0, cdf0)
        d,d,d,pdfy1 = extract_contours(X1, IVT1, cdf1)
        for j in range(nct):
            PDFY0[i,j] = pdfy0[cdf_levels[j]]
            PDFY1[i,j] = pdfy1[cdf_levels[j]]
    
        # Interpolate CDF
        X0, IVT0, CDF0[i] = create_regular_cdf(X0, IVT0, cdf0)
        X1, IVT1, CDF1[i] = create_regular_cdf(X1, IVT1, cdf1)
    
    # Convert PDFY#_mean dictionaries to arrays
    pdfy0_mean = np.zeros((nct,nx))
    pdfy1_mean = np.zeros((nct,nx))
    for j in range(nct):
        pdfy0_mean[j] = PDFY0_mean[cdf_levels[j]]
        pdfy1_mean[j] = PDFY1_mean[cdf_levels[j]]
    
    # Return results
    return X0, IVT0, CDF0, X1, IVT1, CDF1, PDFY0, PDFY1, \
           CDF0_mean, CDF1_mean, pdfy0_mean, pdfy1_mean

#--------------------------------------------------------------------------------------
# Function to interpolate pdf into regular axis
def interpolate_pdf(x, pdf, newx):

    from scipy.interpolate import interp1d
    f = interp1d(x, pdf, kind='cubic', bounds_error=False)
    return f(newx)

#--------------------------------------------------------------------------------------
# Calculate full pdf values
def calculate_full_PDF(width, length, area, stage):
    
    # Define AR lifecycle stages keys
    stage_keys = (100*np.arange(0.05, 1.00, 0.05)).astype("int")

    # Define stage keys
    inds = {}
    for k in stage_keys:
        inds[k] = np.where((stage>=float(k)/100-.03)&(stage<=float(k)/100+0.03))[0]
        
    # Create saving dictionaries
    w = np.linspace(0,10000,1001)
    l = np.linspace(0,10000,1001)
    a = np.linspace(0,1e14,1001)
    
    pdfw = np.zeros((len(stage_keys),len(w)))
    pdfl = np.zeros((len(stage_keys),len(l)))
    pdfa = np.zeros((len(stage_keys),len(a)))
    
    # Loop through all life cycle stages
    for i,k in enumerate(stage_keys):
        pdf,x = fastKDE.pdf(width[inds[k]])
        pdfw[i,:] = interpolate_pdf(x, pdf, w)
        
        pdf,x = fastKDE.pdf(length[inds[k]])
        pdfl[i,:] = interpolate_pdf(x, pdf, l)
                            
        pdf,x = fastKDE.pdf(area[inds[k]])
        pdfa[i,:] = interpolate_pdf(x, pdf, a)
                            
    # Calculate mean PDFs
                
    pdf,x = fastKDE.pdf(width)
    pdfw_mean = interpolate_pdf(x, pdf, w)
    
    pdf,x = fastKDE.pdf(length)
    pdfl_mean = interpolate_pdf(x, pdf, l)
    
    pdf,x = fastKDE.pdf(area)
    pdfa_mean = interpolate_pdf(x, pdf, a)
    
    # Return results
    return w, pdfw_mean, pdfw, l, pdfl_mean, pdfl, a, pdfa_mean, pdfa


#--------------------------------------------------------------------------------------
# Function to convert dictionaries to arrays 
def convert_dict_to_array_cdfs(X0, IVT0, CDF0, CDF0_mean, X1, IVT1, CDF1, CDF1_mean,\
                               PDFX0, PDFY0, PDFX0_mean, PDFY0_mean,\
                               PDFX1, PDFY1, PDFX1_mean, PDFY1_mean):

    nt = len(PDFX0.keys())
    nl = len(PDFX0[5].keys())
    nx = len(X0)
    ny = len(IVT0)

    pdfy0 = np.zeros((nt,nl,nx))
    pdfy1 = np.zeros((nt,nl,nx))

    cdf0 = np.zeros((nt,ny, nx))
    cdf1 = np.zeros((nt,ny, nx))

    pdfy0_mean = np.zeros((nl,nx))
    pdfy1_mean = np.zeros((nl,nx))

    for i,k in enumerate(PDFY0.keys()):

        cdf0[i] = CDF0[k]
        cdf1[i] = CDF1[k]

        for j,l in enumerate(PDFY0[k].keys()):
            pdfy0[i,j] = PDFY0[k][l]
            pdfy1[i,j] = PDFY1[k][l]

            pdfy0_mean[j] = PDFY0_mean[l]
            pdfy1_mean[j] = PDFY1_mean[l]

    return cdf0, cdf1, CDF0_mean, CDF1_mean, pdfy0, pdfy0_mean, pdfy1, pdfy1_mean

#--------------------------------------------------------------------------------------
# Function to gather all calculations of PDF and CDF and return arrays to be saved
def analyze_run(run, verbose=False):
    
    # Time the complete code
    t0 = time.time()
    
    # Open file with raw data
    if verbose:     print("\nOpening file {}".format(output_fName(run)))
    
    f = xr.open_dataset(output_fName(run))

    # Extract variables for pdf calculations
    if verbose:     print("Extracting variables for PC analysis")
    
    indNoNan = np.where(np.logical_not(np.isnan(f.width.values)))[0]
    lon = f.lon.values[indNoNan]
    lat = f.lat.values[indNoNan]
    width = f.width.values[indNoNan]
    length = f.length.values[indNoNan]
    area = f.area.values[indNoNan]
    stage0 = (f.stage/f.life).values[indNoNan]
    life0 = f.life.values[indNoNan]
    ar_id = f.life.values[indNoNan]
    track_id = f.track_id.values[indNoNan]

    # Calculate width, length and area pdfs
    if verbose:     print("Calculaging PDFs of length, width, and area")
    
    w, pdfw_mean, pdfw, \
    l, pdfl_mean, pdfl, \
    a, pdfa_mean, pdfa =  calculate_full_PDF(width, length, area, stage0)
    
    # Save memory
    del(indNoNan,lon,lat,width,length,area,stage0,life0,ar_id,track_id)
    gc.collect()
    
    # Extract variables for the CDF analysis\
    if verbose:     print("Extracting variables for SO and BK analysis")
                          
    stage1 = (f.stage/f.life).values
    life = f.life.values

    d0,ivt0 = f.d0.values, f.ivt0.values
    d1,ivt1 = f.d1.values, f.ivt1.values
    
    
#     print(len(d0),len(ivt0),len(d1),len(ivt1))

    # Calculate CDF and contours
    if verbose:     print("Calculaging CDFs")
    
    X0, IVT0, CDF0, X1, IVT1, CDF1, PDFY0, PDFY1, \
    CDF0_mean, CDF1_mean, PDFY0_mean, PDFY1_mean = \
               calculate_full_CDF_arrays(d0, ivt0, d1, ivt1, stage1, verbose=verbose)
    
#     print(type(X0), type(IVT0), type(CDF0), type(X1), type(IVT1), type(CDF1))
#     print(type(PDFY0), type(PDFY1), type(CDF0_mean), type(CDF1_mean))
#     print(type(PDFY0_mean), type(PDFY1_mean))
    
#     print(np.shape(X0), np.shape(IVT0), np.shape(CDF0), np.shape(X1), np.shape(IVT1), np.shape(CDF1))
#     print(np.shape(PDFY0), np.shape(PDFY1), np.shape(CDF0_mean), np.shape(CDF1_mean))
#     print(np.shape(PDFY0_mean), np.shape(PDFY1_mean))
    
    # Define prob contours axis
    pContours = [0.045, 0.16, 0.5, 0.84, 0.955]

    # Define AR lifecycle stages keys
    stage_keys = (100*np.arange(0.05, 1.00, 0.05)).astype("int")
    
    # Save data to temporary nc file
    if verbose:     print("Saving data to tmp file")
    coordinates =  X0, IVT0, pContours, stage_keys, w, l, a
    variables   =  CDF0_mean, CDF1_mean, PDFY0_mean, PDFY1_mean, \
                   CDF0, CDF1, PDFY0, PDFY1, pdfw, pdfl, pdfa,   \
                   pdfw_mean, pdfl_mean, pdfa_mean  
    tmpFile = create_temporal_dataset(coordinates, variables)
    
    # Delete variables to save memory
    del(X0, IVT0, pContours, stage_keys, w, l, a, CDF0_mean, CDF1_mean, PDFY0_mean, PDFY1_mean)
    del(CDF0, CDF1, PDFY0, PDFY1, pdfw, pdfl, pdfa, pdfw_mean, pdfl_mean, pdfa_mean)
    gc.collect()
    
    # Open temporary nc file
    if verbose:     print("Assigning attributes and saving final results file")
    ds = xr.open_dataset(tmpFile)
    
    # Assign attributes to coordinates
    ds = assign_coord_attrs(ds)
    
    # Assign attributes to variables
    ds = assign_vars_attrs(ds)
    
    # Save new file with results
    fName = "output/nc_cdf_pdf/{}.nc".format(run)
    ds.to_netcdf(fName)
    ds.close()
    del(ds)
    os.remove(tmpFile)
    print("\nFile written: {}".format(fName))
    seconds = time.time()-t0
    if seconds<=60:
        print("{:.2f} seconds ellapsed.\n".format(seconds))
    else:
        minutes = int(seconds/60)
        seconds = int(seconds%60)
        print("{}m {}s ellapsed.\n".format(minutes, seconds))
        
    return
    
#--------------------------------------------------------------------------------------
# Function to create a temporal file to save data to
def create_temporal_dataset(coordinates, variables):
    
    X0, IVT0, pContours, stage_keys, w, l, a         = coordinates
    CDF0_mean, CDF1_mean, pdfy0_mean, pdfy1_mean, \
    cdf0, cdf1, pdfy0, pdfy1, pdfw, pdfl, pdfa,   \
    pdfw_mean, pdfl_mean, pdfa_mean                  = variables
    
    # Initialize a dataset with multiple dimensions:
    ds = xr.Dataset(
        data_vars=dict(
            cdf_along_mean   =   (["ivt", "pc"], CDF0_mean),
            cdf_across_mean  = (["ivt", "pc"], CDF1_mean),

            cdf_contours_along_mean =  (["pContours", "pc"], pdfy0_mean),
            cdf_contours_across_mean = (["pContours", "pc"], pdfy1_mean),

            cdf_along =  (["stage", "ivt", "pc"], cdf0),
            cdf_across = (["stage", "ivt", "pc"], cdf1),

            cdf_contours_along  = (["stage", "pContours", "pc"], pdfy0),
            cdf_contours_across = (["stage", "pContours", "pc"], pdfy1),

            pdf_width = (["stage", "widths"], pdfw),
            pdf_length = (["stage", "lengths"], pdfl),
            pdf_area = (["stage", "areas"], pdfa),

            pdf_width_mean = (["widths"], pdfw_mean),
            pdf_length_mean = (["lengths"], pdfl_mean),
            pdf_area_mean = (["areas"], pdfa_mean),

        ),
        coords=dict(
            pc = (["pc"], X0),
            ivt = (["ivt"], IVT0),
            pContours = (["pContours"], pContours),
            stage = (["stage"], stage_keys),
            widths = (["widths"], w),
            lengths = (["lengths"], l),
            areas = (["areas"], a),
        ),
        attrs=dict(description="Calculated using the algorithms outlined in Inda-Diaz et al. (2020, JGR Atmospheres)"),
    )
    
    prefix = "/global/cscratch1/sd/indah/TMP/dataset_tmp"
    tmp = tempfile.NamedTemporaryFile(prefix = '{}/tmp'.format(prefix), suffix = ".nc", delete = False)
    ds.to_netcdf(tmp.name)
    
    return tmp.name

#--------------------------------------------------------------------------------------
# Function to assign attributes to the nc file coordinates
def assign_coord_attrs(ds):
    pc_dict =  {'long_name'   : "Principal component coordinate",                  \
                'units'       : "km",                                              \
                'description' : "Distance to AR centroid along PCs for CDF IVT(d)" }

    ivt_dict = {'long_name'   : "Integrated water vapor coordinate",               \
                'units'       : "kg m-1 s-1",                                      \
                'description' : "IVT along AR PCs for CDF IVT(d)"                  }

    pCo_dict = {'long_name'   : "p-contours values",                               \
                'units'       : "dimensionless (probability)",                     \
                'description' : "CDF contours at p-value "                         }

    st_dict  = {'long_name'   : "AR lifecycle stage",                              \
                'units'       : "percentage",                                      \
                'description' : "Lifecycle AR stage (timestep divided by life)"    }  

    w_dict   = {'long_name'   : "Width pdf axis",                                  \
                'units'       : "km",                                              \
                'description' : "Regular axis from fastkde PDF of width"           } 

    l_dict   = {'long_name'   : "Length pdf axis",                                 \
                'units'       : "km",                                              \
                'description' : "Regular axis from fastkde PDF of length"          }  

    a_dict   = {'long_name'   : "Area pdf axis",                                   \
                'units'       : "km",                                              \
                'description' : "Regular axis from fastkde PDF of area "           }

    ds['pc']        = ds.pc.assign_attrs(       pc_dict)
    ds['ivt']       = ds.ivt.assign_attrs(      ivt_dict)
    ds['pContours'] = ds.pContours.assign_attrs(pCo_dict)
    ds['stage']     = ds.stage.assign_attrs(    st_dict)
    ds['widths']    = ds.widths.assign_attrs(   w_dict)
    ds['lengths']   = ds.lengths.assign_attrs(  l_dict)
    ds['areas']     = ds.areas.assign_attrs(    a_dict)
    
    return ds

#--------------------------------------------------------------------------------------
# Function to assign attributes to the nc file variables
def assign_vars_attrs(ds):

    cdf_attrs_along  =  {'long_name'   : "CDF IVT(d) (t) along AR",                 \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) at different life stages along AR"}
    cdf_attrs_m_along = {'long_name'   : "CDF IVT(d) average  along AR",                  \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) along AR for all life stages averaged" }

    cdf_attrs_across  =  {'long_name'   : "CDF IVT(d) (t) across AR",                 \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) at different life stages across AR"}
    cdf_attrs_m_across = {'long_name'   : "CDF IVT(d) average  across AR",                  \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) across AR for all life stages averaged" }

    ds["cdf_along"]      = ds["cdf_along"     ].assign_attrs(cdf_attrs_along)
    ds["cdf_along_mean"] = ds["cdf_along_mean"].assign_attrs(cdf_attrs_m_along)

    ds["cdf_across"]      = ds["cdf_across"     ].assign_attrs(cdf_attrs_across)
    ds["cdf_across_mean"] = ds["cdf_across_mean"].assign_attrs(cdf_attrs_m_across)

    cdf_along_attrs  =  {'long_name'   : "CDF IVT(d) along AR (at specific p-level) (t)",\
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) along AR contour at speficic p-level for different life stages"}
    cdf_along_m_attrs = {'long_name'   : "CDF IVT(d) along AR (at specific p-level) averaged",     \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) along AR contour at speficic p-level averaged over life stages"}

    cdf_across_attrs  =  {'long_name'   : "CDF IVT(d) across AR (at specific p-level) (t)",\
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) across AR contour at speficic p-level for different life stages"}
    cdf_across_m_attrs = {'long_name'   : "CDF IVT(d) across AR (at specific p-level) averaged",     \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "AR CDF IVT(d) across AR contour at speficic p-level averaged over life stages"}

    ds["cdf_contours_along"]      = ds["cdf_contours_along"] .assign_attrs(cdf_along_attrs)
    ds["cdf_contours_along_mean"] = ds["cdf_contours_along_mean"].assign_attrs(cdf_along_m_attrs)

    ds["cdf_contours_across"]      = ds["cdf_contours_across"     ].assign_attrs(cdf_across_attrs)
    ds["cdf_contours_across_mean"] = ds["cdf_contours_across_mean"].assign_attrs(cdf_across_m_attrs)


    pdf_attrs  =  {'long_name'   : "PDF of AR width for specific lifecycle stage",  \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "PDF of AR width for different life stages from PC analysis"}
    pdf_attrs_m = {'long_name'   : "PDF of AR width (averaged for life stage)",  \
                   'units'       : "uniteless (probability)",                       \
                   'description' : "PDF of AR width from PC analysis  (averaged for life stage)"}

    pdf_vars = [var for var in ds.variables if "pdf_" in var]

    for var in pdf_vars:
        if "mean" in var:
            ds[var] = ds[var].assign_attrs(pdf_attrs_m)
        else:
            ds[var] = ds[var].assign_attrs(pdf_attrs)

    return ds   

#--------------------------------------------------------------------------------------
def calculate_ivt_pdfs(ivtBkgr, Nsample = None, numPoints=None):

    if Nsample is not None:
        ivtBkgr = np.random.choice(ivtBkgr, nSample, replace=False)
        
    ivtFKDE = fastKDE.fastKDE(ivtBkgr, numPoints=numPoints)

    x=ivtFKDE.axes[0]
    pdf=ivtFKDE.pdf

    cdf=np.zeros_like(pdf)
    for i in range(1,len(pdf)):
        cdf[i]=(ivtFKDE.deltaX*pdf)[:i].sum()
    pSigma = x[np.where(cdf>=0.84)[0][0]]
    mSigma = x[np.where(cdf>=0.16)[0][0]]
    
    return x, pdf, cdf, mSigma, pSigma


#--------------------------------------------------------------------------------------
def plot_ivt_pdfs(args, title=None):

    x, pdf, cdf, mSigma, pSigma = args
    fig,ax=PP.subplots(figsize=(7,3))

    p = ax.plot(x,cdf,color="C1",linewidth=2)
    ax.set_ylabel("IVT cdf")
    ax.set_xlim(-50,600)
    ax.grid(alpha=.2)

    ax.plot([mSigma,mSigma],[0,1],'--k',alpha=0.20)
    ax.text(mSigma,0.45,u"$\sigma^-$={:.2f}".format(mSigma),\
            horizontalalignment='center',verticalalignment='center',rotation=90,alpha=.85,fontsize=12)
    
    ax.plot([pSigma,pSigma],[0,1],'--k',alpha=0.20)
    ax.text(pSigma,0.45,u"$\sigma^+$={:.2f}".format(pSigma),\
            horizontalalignment='center',verticalalignment='center',rotation=90,alpha=.85,fontsize=12)

    axt=ax.twinx()
    p = axt.plot(x,pdf,color="C0",linewidth=2)
    axt.set_ylabel("IVT pdf")
    
    if title is not None:
        ax.set_title(title)
    
    fig.tight_layout()

#----------------------------------------------------------------------------------------
def get_calendar_units(run):
    root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/{}/*".format(run)
    f0 = np.sort(glob.glob(root))[0]
    with xr.open_dataset(f0, decode_times=False) as fin:
         return fin.time.units, fin.time.calendar

#----------------------------------------------------------------------------------------
def get_years(run):
    root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/{}/*".format(run)
    files = np.sort(glob.glob(root))
    years = np.array([f.split(".")[-2] for f in files], dtype="int")

    return years

#--------------------------------------------------------------------------------------
def get_years_time(run):
    import datetime as dt
    
    units,calendar = get_calendar_units(run)
    years = get_years(run)
    
    size = len(years)
    steps = int(size/5)
    
    output = np.zeros((steps,2))
    
    for i in range(steps):
        
        j = i*5
        
        if i<steps-1:
            output[i,0] = date2num(dt.datetime(years[j],1,1,0), units, calendar)
            output[i,1] = date2num(dt.datetime(years[j+5],12,31,23), units, calendar)
        else:
            output[i,0] = date2num(dt.datetime(years[j],1,1,0), units, calendar)
            output[i,1] = date2num(dt.datetime(years[-1],12,31,23), units, calendar)
    return years, output

#--------------------------------------------------------------------------------------
# Function to analyze the IVT background filed from 1 case (1 of 4)
def analyze_ivt_bk_case(run, case=None):

    #open file
    f = open_ivtbk(run, case=case)
    units,calendar = get_calendar_units(run)
    
    ind = np.where(f.IVT_bk!=np.inf)[0]
    f = f.isel(time=ind)
    
    #extract ivt, years, and data for sampling every five years
    years_unique, years_times = get_years_time(run)
    size = len(years_unique)
    steps = int(size/5)
    yyyy = f.time.values
    ivt  = f.IVT_bk.values

    # Calculate IVT bk pdf and cdf for every five years
    periods = []
    data = []

    xnew = np.linspace(-100,5000,5101)

#     print("Calculating IVT_bk pdf for every 5 years period")
    for i in range(steps):

        ind = np.where((yyyy>=years_times[i,0])&(yyyy<=years_times[i,1]))[0]
#         print("Calculating IVT_bk for step {}. Dates between {} and {}".format(i, years_times[i,0], years_times[i,1]))
#         print(len(ind))
        try:
            x, pdf, cdf, mSigma, pSigma =  calculate_ivt_pdfs(ivt[ind])
            pdfnew = interpolate_pdf(x, smooth(pdf, 2, mode="same"), xnew)
            cdfnew = interpolate_pdf(x, cdf, xnew)
            data.append((xnew, pdfnew, cdfnew, mSigma, pSigma))
        except:
#             print(ivt[ind])
#         x, pdf, cdf, mSigma, pSigma =  calculate_ivt_pdfs(ivt[ind][::500])
            print("step {} failed. Dates between {} and {}".format(i, \
                                                                   num2date(years_times[i,0],units,calendar),\
                                                                   num2date(years_times[i,1],units,calendar)))
            data.append((xnew*np.nan, xnew*np.nan, xnew*np.nan, np.nan, np.nan))

#         periods.append((years_times[i,0],years_times[i,1]))
        periods.append(years_times[i,0])
    
#     print("Calculating IVT_bk PDF for all the timesteps")
    # Calculate IVT bk pdf and cdf for the full model run
    x, pdf, cdf, mSigma, pSigma =  calculate_ivt_pdfs(ivt)
#     x, pdf, cdf, mSigma, pSigma =  calculate_ivt_pdfs(ivt[::5000])
    pdfnew = interpolate_pdf(x, smooth(pdf, 2, mode="same"), xnew)
    cdfnew = interpolate_pdf(x, cdf, xnew)
    
    return data, periods, (xnew, pdfnew, cdfnew, mSigma, pSigma), f.IVT_bk.attrs
    
#--------------------------------------------------------------------------------------
# Function to analyze the IVT background filed for all 4 cases and create a 
# netCDF file with ivt bk results
def analyze_full_ivt_bk(run):

#     data1, periods1, means1 = analyze_ivt_bk_case(run, case=1)
#     data2, periods2, means2 = analyze_ivt_bk_case(run, case=2)
#     data3, periods3, means3 = analyze_ivt_bk_case(run, case=3)
#     data4, periods4, means4 = analyze_ivt_bk_case(run, case=4)

#     return data1, periods1, means1, data2, periods2, means2, \
#            data3, periods3, means3, data4, periods4, means4
    
    t0 = time.time()
    
    print("\nAnalyzing full IVT bk field for {}\n".format(run))
    DATA, MEANS, ATTRS = [],[],[]
    
    for case in np.arange(1,5,1):
        
        print("Analyzing ivt bk {} out of 4".format(case))
        data, period, means, attrs = analyze_ivt_bk_case(run, case=case)
        DATA.append(data)
        MEANS.append(means)
        ATTRS.append(attrs)
    
    variables = (DATA, period, MEANS, ATTRS)
    
    print("\nCreating xarray dataset")
    ds = create_ivtbk_nc(run, variables)
    
    # Save new file with results
    fName = "output/nc_cdf_pdf/ivt_bk_{}.nc".format(run)
#     if os.path.exists(fName):
#         os.remove(fName)
        
    ds.to_netcdf(fName)
    ds.close()
    del(ds)
    print("\nFile written: {}".format(fName))
    seconds = time.time()-t0
    if seconds<=60:
        print("{:.2f} seconds ellapsed.\n".format(seconds))
    else:
        minutes = int(seconds/60)
        seconds = int(seconds%60)
        print("{}m {}s ellapsed.\n".format(minutes, seconds))

#--------------------------------------------------------------------------------------
# Function to create netCDF ivt bk file
def create_ivtbk_nc(run, variables):
    
    units,calendar = get_calendar_units(run)
    
    DATA, PERIOD, MEANS, ATTRS = variables
    ivt = MEANS[0][0]
    
    # Initialize a dataset with multiple dimensions:
    ds = xr.Dataset(coords=dict(
                                ivt = (["ivt"], ivt),
                                time  = (["time"], PERIOD),
                               ),
                    attrs=dict(description="IVT background field from {}".format(run)))
    
#     prefix = "/global/cscratch1/sd/indah/TMP/dataset_tmp"
#     tmp = tempfile.NamedTemporaryFile(prefix = '{}/tmp'.format(prefix), suffix = ".nc", delete = False)
    
    ivt_attrs  =  {'long_name'   : "Integraged vapor transport",  \
                   'units'       : "kg m-1 s-1",                       \
                   'description' : "IVT axis from fastKDE PDF and CDF"}
    ds["ivt"] = ds["ivt"].assign_attrs(ivt_attrs)

    
    period_attrs  =  {'long_name'   : "5-year perdiod starting date",  \
                   'units'       : units,                       \
                   'calendar'       : calendar,                       \
                   'description' : "Analysis for every 5 years, this is the starting date of each period"}
    ds["time"] = ds["time"].assign_attrs(period_attrs)

    for i in np.arange(1,5,1):
        j=i-1
        data, means, attrs = DATA[j], MEANS[j], ATTRS[j]
        
        PDF,CDF,MSIGMA,PSIGMA = [],[],[],[]
        for d in data:
            x, pdf, cdf, mSigma, pSigma = d
            PDF.append(pdf)
            CDF.append(cdf)
            MSIGMA.append(mSigma)
            PSIGMA.append(pSigma)
        
        ds["pdf_{}".format(i)]=(['time','ivt'], np.array(PDF))
        ds["cdf_{}".format(i)]=(['time','ivt'], np.array(CDF))
        ds["mSigma_{}".format(i)]=(['time'], np.array(MSIGMA))
        ds["pSigma_{}".format(i)]=(['time'], np.array(PSIGMA))
        
        x, pdf, cdf, mSigma, pSigma = means
        ds["mean_pdf_{}".format(i)]=(['ivt'], pdf)
        ds["mean_cdf_{}".format(i)]=(['ivt'], cdf)
        ds["mean_mSigma_{}".format(i)]=(mSigma)
        ds["mean_pSigma_{}".format(i)]=(pSigma)
                
    return ds
    
#--------------------------------------------------------------------------------------
# Function to compare 2 CDF fro ivt backgrounnd and CPD of ivt with distance
def compare_cdf2BK(x, ivt, cdf, ivtbk, cdfbk, subsample=None, \
                   alt = "two-sided", mode='auto'):

    left = 100
    right = 650
    
    indx = np.where(cdf[ivt.argmax(),:]>0.95)[0]
    indy = np.where((ivt>=left)&(ivt<=right))[0]
#     print(len(indy), len(indx))
#     print(np.shape(cdf))
    x = x[indx]
    ivt = ivt[indy]
    cdf = cdf[:,indx]
    cdf = cdf[indy,:]

    if subsample is not None:
        x = x[::subsample]
        cdf = cdf[:,::subsample]
        
    ks = np.ones_like(x)*-999
    pv = np.ones_like(x)*-999

    # fig,ax=PP.subplots()
    # ax.plot(ivtbk,cdfbk,'k',linewidth=3)
    # ax.set_xlim(250,850)
    # ax.set_ylim(0.75,1.05)
    for i,xp in enumerate(x):
        # if (np.abs(xp)>=4000 and np.abs(xp)<=6000):
            ys = cdf[:,i]
            ind = np.where((ivtbk>=left)&(ivtbk<=right))[0]
            m,n =  int(5e1),int(5e1)
            ks[i],pv[i] = ks_2samp(ys, cdfbk[ind], m, n, alternative=alt, mode=mode)
            # ax.plot(ivt,ys,alpha=0.5)
        
    return x, ks, pv

# #--------------------------------------------------------------------------------------
# # Function to find the distance (length and width) from ks-test' results
# def find_ar_boudaries(dist, ks, pv, alt="two-sided", pcritic = 0.95, Length=None):
     
#     left  = dist<-10
#     right = dist>10
    
#     condition = ks[left]<pv[left]
#     left = dist[left][condition].max()
    
#     condition = ks[right]<pv[right]
#     right = dist[right][condition].min()
    
#     return left, right

#--------------------------------------------------------------------------------------
# Function to find the distance (length and width) from ks-test' results
def find_ar_boudaries(dist, ks, pv, alt="two-sided", pcritic = 0.95, Length=None):
     
    m,n =  5e1,5e1    
    a = np.array([0.20,0.15,0.10,0.05,0.025,0.01,0.005,0.001])
    c = np.array([1.073,1.138,1.224,1.358,1.48,1.628,1.731,1.949])
    D = np.sqrt(-1*np.log(a/2)*(1+m/n)/(2*m))

    left  = dist<-10
    right = dist>10
    
    condition = ks[left]<D[-2]
    left = dist[left][condition].max()
    
    condition = ks[right]<D[-2]
    right = dist[right][condition].min()
    
    return left, right

#--------------------------------------------------------------------------------------
# Function to plot two-sided and one-sided KS-test results
def plot_kst_final(xs_ls, ks_ls, pv_ls, xs_2s, ks_2s, pv_2s, pcritic = 0.95, Length=""):
    
    fig, axs = PP.subplots(1,2,figsize=(14,3))
    ax0,ax1 = axs
    
    titles = ["One-tailed KS-test","Two-tailed KS-test"]
    ylabs = ["One-tailed KS","Two-tailed KS"]
    clabs = ["One-tailed p-val","Two-tailed p-val"]
    left2s, right2s = find_ar_boudaries(xs_2s, ks_2s, pv_2s)
    left2l, right2l = find_ar_boudaries(xs_ls, ks_ls, pv_ls)
    lefts, rights = [left2s,left2l],[right2s,right2l]
    offset = (max(xs_2s) - min(xs_2s))/10
    
    s = []
    s.append(ax0.scatter(xs_2s, ks_2s, c=pv_2s, s=12))
    s.append(ax1.scatter(xs_ls, ks_ls, c=pv_ls, s=12))
    
    for i,ax in enumerate(axs):
        ax.set_xlabel("Distance to the AR centtoid AR [km]")
        cb = fig.colorbar(s[i], ax=ax, label=clabs[i])
        ax.set_ylabel(ylabs[i])
        ax.set_xlim(lefts[i]-offset,rights[i]+offset)
        ax.grid(alpha=.25)
        ax.plot([lefts[i],   lefts[i]], [0,1.12], "--k", alpha=.5)
        ax.plot([rights[i], rights[i]], [0,1.12], "--k", alpha=.5)
        ax.set_ylim(0,1)
        
        t = ax.set_title("{}: {}={:.2f} km.".\
                        format(titles[i],Length,rights[i]-lefts[i]), loc='left', fontsize=12)
    return

#--------------------------------------------------------------------------------------
# Function to calculate the length, width and area from the SO method
def calculate_SO(run, ivtbk=3, mean=False, ivtFixed=None, stage=None):
    
    fName = "output/nc_cdf_pdf/{}.nc".format(run)
    with xr.open_dataset(fName) as f:
        
        if stage is not None:
            across = f.cdf_contours_across.sel(stage=stage).values
            along = f.cdf_contours_along.sel(stage=stage).values
        else:
            across = f.cdf_contours_across_mean.values
            along = f.cdf_contours_along_mean.values
        x = f.pc.values
        pContours = f.pContours.values
    
    if ivtFixed is not None:
        bk = ivtFixed
    else:
        if mean:
            if "hist" in run:
                Files = np.sort(glob.glob("output/nc_cdf_pdf/ivt*historical*"))
            else:
                Files = np.sort(glob.glob("output/nc_cdf_pdf/ivt*85*"))
            f = xr.open_mfdataset(Files, combine='nested', concat_dim='realization',\
                                   decode_times=False, decode_cf=False).mean(dim='realization')
        else:
            f = xr.open_dataset('output/nc_cdf_pdf/ivt_bk_{}.nc'.format(run))
        if ivtbk==1:
            bk = f.mean_pSigma_1.values
        if ivtbk==2:
            bk = f.mean_pSigma_2.values
        if ivtbk==3:
            bk = f.mean_pSigma_3.values
        if ivtbk==4:
            bk = f.mean_pSigma_4.values
    
    ind = np.where(pContours==0.16)[0]
    left,right = xroots_from_ivt(x, np.squeeze(across[ind,:]), bk)
    width = right-left
    left,right = xroots_from_ivt(x, np.squeeze(along[ind,:]), bk)
    length = right-left
    
    return width, length,  np.pi*width*length/2

#--------------------------------------------------------------------------------------
# Function to calculate the length, width and area from the BK (ks-test) method
def calculate_BK(run, subsample=1, ivtbk=1, stage=None):

    """ Load CPD of IVT with distance """
    fName = "output/nc_cdf_pdf/{}.nc".format(run)
    # print(fName)
    f = xr.open_dataset(fName)
    ivt = f.ivt.values
    x = f.pc.values
    if stage is not None:
        cdfal = f.cdf_along.sel(stage=stage).values
        cdfac = f.cdf_across.sel(stage=stage).values
    else:
        cdfal = f.cdf_along_mean.values
        cdfac = f.cdf_across_mean.values

    """ Load Background IVT CDF """
    fName = "output/nc_cdf_pdf/ivt_bk_{}.nc".format(run)
    # print(fName)
    # print()
    bk = xr.open_dataset(fName)
    
    if ivtbk==1:
        cdfbk = bk.mean_cdf_1.values
    elif ivtbk==2:
        cdfbk = bk.mean_cdf_2.values
    elif ivtbk==3:
        cdfbk = bk.mean_cdf_3.values
    elif ivtbk==4:
        cdfbk = bk.mean_cdf_4.values
    ivtbk = bk.ivt

    xs_ls, ks_ls, pv_ls = compare_cdf2BK(x, ivt, cdfal , ivtbk, cdfbk, alt="less", mode="auto", subsample=subsample)
    xs_2s, ks_2s, pv_2s = compare_cdf2BK(x, ivt, cdfal , ivtbk, cdfbk, alt="two-sided", mode="auto", subsample=subsample)
    l,r = find_ar_boudaries(xs_ls, ks_ls, pv_ls)
    lengthL = r-l
    l,r = find_ar_boudaries(xs_2s, ks_2s, pv_2s)
    length2 = r-l

    xs_ls, ks_ls, pv_ls = compare_cdf2BK(x, ivt, cdfac , ivtbk, cdfbk, alt="less", mode="auto", subsample=subsample)
    xs_2s, ks_2s, pv_2s = compare_cdf2BK(x, ivt, cdfac , ivtbk, cdfbk, alt="two-sided", mode="auto", subsample=subsample)
    l,r = find_ar_boudaries(xs_ls, ks_ls, pv_ls)
    widthL = r-l
    l,r = find_ar_boudaries(xs_2s, ks_2s, pv_2s)
    width2 = r-l
    
    return (widthL, lengthL, np.pi*lengthL*widthL/4.),(width2, length2, np.pi*length2*width2/4.)
    
#--------------------------------------------------------------------------------------
# Function to calculate AR area from ARCI using fastKDE.floodFillSearch module
def arArea(blobVar, lat, lon, \
           blobThreshold = 0.0, \
           timeInd = 0, \
           latBoundsVar = None, \
           lonBoundsVar = None, \
           latLonsAre2D = False, \
           latLonsInDegrees = True, \
           wrapDimensions = None, \
           extraVars = None, \
           beVerbose = True, \
           earthRadius = 6.37122e6, \
           ):
    """

    """

    #**************************************
    # Read the blob file and detect blob
    #**************************************
    
    #Set the slice for reading a 2D lat/lon plane of variable
    varSlice = (timeInd,slice(None,None,None),slice(None,None,None))

    #Open the blob dataset
    blobIndexList = fastKDE.flood.floodFillSearch( \
                                np.array(blobVar[varSlice],dtype=np.float64), \
                                searchThreshold = blobThreshold,   \
                                wrapDimensions = wrapDimensions)
    latunits = lat.units
    lonunits = lon.units

   #*****************************
   # Deal with the grid geometry
   #*****************************
   #Make the lat/lon variables 2D
    if not latLonsAre2D:
        lon,lat = np.meshgrid(lon.values,lat.values)
    else:
        lon,lat = lon.values,lat.values
        
    #Convert to radians
    if latLonsInDegrees:
        lon *= np.pi/180
        lat *= np.pi/180

    #Get the shape of the lat/lon variables
    nlat,nlon = np.shape(lat)

    #Calculate lat/lon bounds as the average of the lat/lons
    latBounds = np.zeros([nlat,nlon,2])
    lonBounds = np.zeros([nlat,nlon,2])

    latBounds[1:,:,0]  = 0.5*(lat[1:,:] + lat[:-1,:])
    latBounds[:-1,:,1] = 0.5*(lat[1:,:] + lat[:-1,:])

    lonBounds[:,1:,0]  = 0.5*(lon[:,1:] + lon[:,:-1])
    lonBounds[:,:-1,1] = 0.5*(lon[:,1:] + lon[:,:-1])

    #Assuming that the spacing of the latitudes continues beyond the latitude bounds
    latBounds[0,:,0]  = lat[0,:] - 0.5*(lat[1,:] - lat[0,:])
    latBounds[-1,:,1] = lat[-1,:] + 0.5*(lat[-1,:] - lat[-2,:])

    #Do the special case for wrapped longitude
    lonBounds[:,0,0] = 0.5*(lon[:,0] + (lon[:,-1] - 2*np.pi))
    lonBounds[:,-1,1] = lonBounds[:,0,0] + 2*np.pi

    #Calculate the cell areas
    dlon = lonBounds[:,:,1]-lonBounds[:,:,0]
    dlat = latBounds[:,:,1]-latBounds[:,:,0]
    #Use abs() here to deal with possibly negative dlat/dlon
    cellAreas = abs(earthRadius**2 * np.cos(lat) * dlat * dlon)

    #Calculate the blob areas
    blobAreas = [np.sum(cellAreas[s]) for s in blobIndexList]
    # print(len(blobAreas))
    blobAreas = np.array(blobAreas)
    blobAreas = blobAreas[np.where(blobAreas>=4*np.mean(cellAreas))[0]]
    # print(len(blobAreas))
    # print(blobAreas
    
    return blobAreas

# #--------------------------------------------------------------------------------------
# # Function to calculate areas at 0.50 and 0.67 for a given run
# def calculate_ARCI_areas(run):
   
#     root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/"
#     dName = lambda r: "{}/{}".format(root,r)
#     fNames = lambda r: np.sort(glob.glob("{}/*{}*.nc4".format(dName(r),r)))

#     areas050 = []
#     areas067 = []

#     for i,fn in enumerate(fNames(run)):
#         if i%25==0:   print(fn)
#         with xr.open_dataset(fn) as f:
#             a = arArea(f.ar_confidence_index.values, f.lat, f.lon, blobThreshold=0.5)
#             a = a[a>1000]
#             areas050.append(a)
#             a = arArea(f.ar_confidence_index.values, f.lat, f.lon, blobThreshold=0.67)
#             a = a[a>1000]
#             areas067.append(a)

#     return np.ravel(np.concatenate(areas050)),np.ravel(np.concatenate(areas067))

#--------------------------------------------------------------------------------------
def calculate_arci_areas(run):

    with open("metadata_tier2.csv", "r") as fin:
        lines = fin.readlines()
        lines = lines[1:]

    for line in lines:
        if run in line:
            rr,ntime,t0,units,calendar = line.split(",")
            ntime,t0,calendar = int(ntime),float(t0),calendar.split("\n")[0]

    times = np.cumsum(0.25*np.ones(ntime)) - 0.25 + t0
    timeinfo = (calendar, units, times)

    root = "/global/homes/i/indah/RESEARCH/ARTMIP/tier2/CMIP56/Means/"
    dName = lambda r: "{}/{}".format(root,r)
    fNames = lambda r: np.sort(glob.glob("{}/*{}*.nc4".format(dName(r),r)))

    areas50 = np.ones((len(times),50))*np.nan
    counts50 = np.ones(len(times))*np.nan
    areas67 = np.ones((len(times),50))*np.nan
    counts67 = np.ones(len(times))*np.nan
    years = []
    
    c = 0
    for j,fn in enumerate(fNames(run)):
        if j%20==0: print(fn)
        year=int(fn.split(".")[-2])
        years.append(year)
        
        with xr.open_dataset(fn) as f:
            for i,t in enumerate(f.time):
                # print(t.values)
                var = f.ar_confidence_index.isel(time=i).values[np.newaxis,...]
                # var = f.ar_confidence_index.values
                a = arArea(var, f.lat, f.lon, blobThreshold=0.5)
                areas50[c,0:len(a)] = a
                counts50[c] = len(a)
                a = arArea(var, f.lat, f.lon, blobThreshold=0.67)
                areas67[c,0:len(a)] = a
                counts67[c] = len(a)
                c = c+1

    return areas50,counts50,areas67,counts50,years
    
#--------------------------------------------------------------------------------------
def create_arci_area_netcdf(run, areas50, counts50, areas67, counts67, years):
    
    # Initialize a dataset with multiple dimensions:
    ds = xr.Dataset(coords=dict(
                                areas = (["areas"], np.arange(len(areas50[0,:]))),
                                year  = (["year"], years),
                               ),
                    attrs=dict(description="AR area calculated with AR confidence index, model={}".format(run)))

    #     prefix = "/global/cscratch1/sd/indah/TMP/dataset_tmp"
    #     tmp = tempfile.NamedTemporaryFile(prefix = '{}/tmp'.format(prefix), suffix = ".nc", delete = False)

    a_atts  =  {'long_name'   : "ar areas",  \
                   'units'       : "",                       \
                   'description' : ""}
    y_atts  =  {'long_name'   : "year",  \
                   'units'       : "year",                       \
                   'description' : ""}
    a50_attrs  =  {'long_name'   : "ARCI areas at 0.50",  \
                   'units'       : "m**2",                       \
                   'description' : "AR areas calculated form ARCI>0.50"}
    a67_attrs  =  {'long_name'   : "ARCI areas at 0.67",  \
                   'units'       : "m**2",                       \
                   'description' : "AR areas calculated form ARCI>0.67"}
    c50_attrs  =  {'long_name'   : "ARCI count at 0.50",  \
                   'units'       : "count",                       \
                   'description' : "AR count calculated form ARCI>0.50"}
    c67_attrs  =  {'long_name'   : "ARCI count at 0.67",  \
                   'units'       : "count",                       \
                   'description' : "AR count calculated form ARCI>0.67"}
    
    ds["areas"] = ds["areas"].assign_attrs(a_atts)
    ds["year"] = ds["year"].assign_attrs(y_atts)

    ds["areas50"]=(['year','areas'], areas50)
    ds["areas67"]=(['year','areas'], areas67)
    ds["count50"]=(['year'], counts50)
    ds["count67"]=(['year'], counts67)
    
    ds["areas50"] = ds["areas50"].assign_attrs(a50_attrs)
    ds["areas67"] = ds["areas67"].assign_attrs(a67_attrs)
    ds["count50"] = ds["count50"].assign_attrs(c50_attrs)
    ds["count67"] = ds["count67"].assign_attrs(c67_attrs)
        
    os.system("rm output/arciAreas/ARCI_areas.{}.nc".format(run))
    ds.to_netcdf("output/arciAreas/ARCI_areas.{}.nc".format(run))
    
    return



#--------------------------------------------------------------------------------------
def median_sigmas(x,pdf):

    deltaX = np.mean(np.diff(x))
    cdf=np.zeros_like(pdf)
    for i in range(1,len(pdf)):
        cdf[i]=(deltaX*pdf)[:i].sum()
    pSigma = x[np.where(cdf>=0.84)[0][0]]
    mSigma = x[np.where(cdf>=0.16)[0][0]]
    median = x[np.where(cdf>=0.50)[0][0]]

    return median, mSigma, pSigma

#--------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------
# Function to calculate grid area 
def cellArea(lat, lon, \
           latBoundsVar = None, \
           lonBoundsVar = None, \
           latLonsAre2D = False, \
           latLonsInDegrees = True, \
           wrapDimensions = None, \
           extraVars = None, \
           beVerbose = True, \
           earthRadius = 6.37122e6, \
           ):
    """

    """

    #**************************************
    # Read the blob file and detect blob
    #**************************************
    
    latunits = lat.units
    lonunits = lon.units

   #*****************************
   # Deal with the grid geometry
   #*****************************
   #Make the lat/lon variables 2D
    if not latLonsAre2D:
        lon,lat = np.meshgrid(lon.values,lat.values)
    else:
        lon,lat = lon.values,lat.values
        
    #Convert to radians
    if latLonsInDegrees:
        lon *= np.pi/180
        lat *= np.pi/180

    #Get the shape of the lat/lon variables
    nlat,nlon = np.shape(lat)

    #Calculate lat/lon bounds as the average of the lat/lons
    latBounds = np.zeros([nlat,nlon,2])
    lonBounds = np.zeros([nlat,nlon,2])

    latBounds[1:,:,0]  = 0.5*(lat[1:,:] + lat[:-1,:])
    latBounds[:-1,:,1] = 0.5*(lat[1:,:] + lat[:-1,:])

    lonBounds[:,1:,0]  = 0.5*(lon[:,1:] + lon[:,:-1])
    lonBounds[:,:-1,1] = 0.5*(lon[:,1:] + lon[:,:-1])

    #Assuming that the spacing of the latitudes continues beyond the latitude bounds
    latBounds[0,:,0]  = lat[0,:] - 0.5*(lat[1,:] - lat[0,:])
    latBounds[-1,:,1] = lat[-1,:] + 0.5*(lat[-1,:] - lat[-2,:])

    #Do the special case for wrapped longitude
    lonBounds[:,0,0] = 0.5*(lon[:,0] + (lon[:,-1] - 2*np.pi))
    lonBounds[:,-1,1] = lonBounds[:,0,0] + 2*np.pi

    #Calculate the cell areas
    dlon = lonBounds[:,:,1]-lonBounds[:,:,0]
    dlat = latBounds[:,:,1]-latBounds[:,:,0]
    #Use abs() here to deal with possibly negative dlat/dlon
    cellAreas = abs(earthRadius**2 * np.cos(lat) * dlat * dlon)

    
    # return [np.min(cellAreas), np.mean(cellAreas), np.max(cellAreas)]
    return  np.mean(cellAreas)










