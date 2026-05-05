from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import stats
from astropy.io import fits
from astropy.time import Time
import os,sys
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad
from scipy.signal import fftconvolve
import ppxf_util_al_lab_wv as util

#define function to create the templates
def nels_templates(lam_gal,logLam2,fwhm):
    FWHM_galx =fwhm #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
    lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
    gas_templates, line_names,line_wave = \
        util.emission_lines(logLam2, lamRange_gal, FWHM_galx)
    return gas_templates, line_names,line_wave
    
    
def bels_templates(lam_gal,logLam2,fwhm):
    FWHM_galx =fwhm #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
    lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
    Broad_templates, Bline_names, Bline_wave = \
        util.Broad_emission_lines(logLam2, lamRange_gal, FWHM_galx)
    return Broad_templates, Bline_names, Bline_wave

def my_wind_templates(lam_gal,logLam2,fwhm):
    FWHM_galx =fwhm #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
    lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
    wind_templates, wind_line_names, wind_line_wave=util.emission_o3_winds(logLam2, lamRange_gal, FWHM_galx)
    #self.wind_templates, self.wind_line_names, self.wind_line_wave=util.emission_lines(logLam2, lamRange_gal, FWHM_galx)
    #When using a second semi-broad component for the NELs pPXF can use it to model part of the broad component in the
    #BELs regions, then we restrict the winds to the prominent [OIII] region

def pl(wv,a,N):
    n2=1/N**a
    y=n2*(wv**a)
    return y

#Function to create gaussians with defined FWHM
def my_fwhm_l(wv,cwv,template,fwhm,my_sigma):
    c=299792.458  # Speed of light in km/s
    my_fwhm=fwhm #fixed fwhm
    lambda_center=cwv
    target_sigma_velocity = my_fwhm / (2 * np.sqrt(2 * np.log(2)))
    target_sigma_wavelength = (target_sigma_velocity / c) * lambda_center
    #print(target_sigma_wavelength)
    # Calculate convolution sigma
    convolution_sigma = np.sqrt(target_sigma_wavelength**2 - my_sigma**2)# 1 is the sigma
    #print(convolution_sigma)
    # Create a Gaussian-like array (example data)
    # Perform convolution
    convolved_gaussian = gaussian_filter1d(template, convolution_sigma / np.mean(np.diff(wv)))
    return convolved_gaussian

#################################### Use one stellar template to later set other templates ########################################           
def set_temp(lam_gal,velscale,fwhm_gals,dir_temps=None,list_st=None):
    #these list of stellar templates are provided
    my_list=['./ssp_temps/Ech1.30Zm0.40T00.5012_iPp0.00_baseFe.fits',
        './ssp_temps/Ech1.30Zm0.40T05.0119_iPp0.00_baseFe.fits',
        './ssp_temps/Ech1.30Zp0.22T11.2202_iPp0.00_baseFe.fits']
    if dir_temps==None and list_st==None:
        list_st=my_list
        str_op_temps=1
    # Read the list of filenames from the Single Stellar Population library
    #str_op_temps=0 #use any other number to use the option with particular stellar models, also use the list_st
    #list_st=[]  #This maybe an array with the stellar population directories that you want to use
    if dir_temps is not None:
        vdk1=glob.glob(dir_temps)
        vazdekis=np.asarray(vdk1[0:3])
        ustemps='all_eMstr'
    
    else:
        ustemps='selc_eMstr'
        vazdekis=[]
        line_ta=list_st
        vazdekis=np.append(vazdekis,line_ta)
    vazdekis.sort()
    fwhm_tem =2.51 #2.51 #Use the same Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
                    #Because I don't know the exact value and apparently is similar

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu2 = fits.open(vazdekis[0])
    ssp = hdu2[0].data
    h2 = hdu2[0].header
    lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    #Fill for blue side for high redshift
    condi_blu=0
    if lam_temp[0]>lam_gal[0]:
        condi_blu=1
        flt=lam_temp[1]-lam_temp[0]
        dlt=np.absolute(lam_gal[0]-lam_temp[0])
        npfs=dlt/flt
        extsspr=[lam_temp[0]-(flt*ik) for ik in range(int(npfs)+1)]
        extssp=extsspr[::-1]
        lam_temp=np.append(np.array(extssp),lam_temp)
        ssp=np.append(np.zeros(len(extssp)),ssp)
        
        
    stwvmk=(lam_temp>=lam_gal[0]-1)&(lam_temp<=lam_gal[-1]+1)
    lam_temp=lam_temp[stwvmk]
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    delt_lam=lam_temp[1]-lam_temp[0]
    ssp=ssp[stwvmk]
    

    sspNew,logLam2,velscale2 = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
    #Define the size for templates
    Templates_size=sspNew.size
    #Define stars_templates for the constructor
    stars_templates = np.empty((sspNew.size, len(vazdekis)))
# Interpolates the galaxy spectral resolution at the location of every pixel
    # of the templates. Outside the range of the galaxy spectrum the resolution
    # will be extrapolated, but this is irrelevant as those pixels cannot be
    # used in the fit anyway.
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gals)

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SDSS and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SDSS
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
    # In principle it should never happen and a higher resolution template should be used.
    #
    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    #fwhm_gal2=2.76
    #fwhm_dif = np.sqrt(fwhm_gal2**2 - fwhm_tem**2)

    sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
    #sigma=sigma*0
    for j, fname in enumerate(vazdekis):
        #print(j)
        hdu2 = fits.open(fname)
        ssp = hdu2[0].data
        if condi_blu==1:
            ssp=np.append(np.zeros(len(extssp)),ssp)
        ssp=ssp[stwvmk]
        ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma        
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
        for yq in [0,1,2,3,4,5,6,7,-3,-2,-1]:
            if yq>=0:
                sspNew[yq]=sspNew[8]
            else:
                sspNew[yq]=sspNew[-4]
        stars_templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    c = 299792.458
    dv = np.log(np.exp(logLam2[0])/lam_gal[0])*c    # km/s
    #logLam2=logLam2[7:-3] #To cut badpixels
    return stars_templates,dv,logLam2

#################################### Set FeII templates ######################################## 

def feii_temps(Templates_size,lamRange_temp,velscale):
    #The template is provided
    fe=['./uvofeii4800kms.txt']

    #Define the FeII shape templates
    fe_templates = np.empty((Templates_size, len(fe)))

    #FWHM from km/s to AA
    #fwhm_bc=4000
    #sig_bc=(fwhm_bc/(c*2.355))*lam_bc

    for j, fnames in enumerate(fe):
        ssp = np.loadtxt(fnames)
        ssbc1=ssp[:,1]
        lam_bc1=ssp[:,0]
        deltbc=lam_bc1[1]-lam_bc1[0]
        #add zeros or cut to work with the same wavelenght as the masked stellar templates
        if lam_bc1[0]>lamRange_temp[0]:
            wzle=np.arange(-lam_bc1[0],-lamRange_temp[0],deltbc)
            fzle=wzle*0
            wzle=np.sort(-wzle)
            lam_bc=np.append(wzle,lam_bc1)
            sspbc=np.append(fzle,ssbc1)
        else:
            lmks=(lam_bc1>=lamRange_temp[0])
            lam_bc=lam_bc1[lmks]
            sspbc=ssbc1[lmks]


        if lam_bc1[-1]<lamRange_temp[-1]:
            wrge=np.arange(lam_bc1[-1]+deltbc,lamRange_temp[-1]+deltbc,deltbc)
            frge=wrge*0
            lam_bc=np.append(lam_bc,wrge)
            sspbc=np.append(sspbc,frge)
        else:
            hmks=(lam_bc<=lamRange_temp[-1])
            lam_bc=lam_bc[hmks]
            sspbc=sspbc[hmks]
    #Define the sigma difference in pixels as zero
    sigma2d=sspbc*0

    sspbc = util.gaussian_filter1d(sspbc, sigma2d)  # perform convolution with variable sigma
    sspNew, logLam3, velscale = util.log_rebin(lamRange_temp, sspbc, velscale=velscale)
    mfm=(sspNew>0) #To avoid ~zero median
    sspfm=sspNew[mfm]
    #In some cases the template size is different
    if len(sspNew)!=Templates_size:
        dftl=int(len(sspNew)-Templates_size)
        if dftl>0:
            sspNew=sspNew[:-dftl]
        else:
            sadf=np.zeros([np.abs(dftl)])
            sspNew=np.append(sspNew,sadf)
    fe_templates[:, j] = sspNew/np.median(sspfm) # Normalized templates
    return fe_templates

###############BALMER and BALMER High Order Continuum Templates ####################
def balmer_templates(self,Balmer_tem,Templates_size,lamRange_temp,velscale):
#Read the files
    #bc1=glob.glob('BalmerCont/BalCont/Bal*')
    #bc2=glob.glob('BalmerCont/BalHiOrd/*M1000.dat') #Use only the template with FWHM1000km/s
    bc1=glob.glob(Balmer_tem[0])#('BalmerCont/BalCont/Bal*')
    bc2=glob.glob(Balmer_tem[1])#('BalmerCont/BalHiOrd/*M1000.dat') #Use only the template with FWHM1000km/s
    self.bc=np.append(bc1,bc2)

    #Define the BC shape templates
    self.bc_templates = np.empty((Templates_size, len(self.bc)))

    #FWHM from km/s to AA
    #fwhm_bc=4000
    #sig_bc=(fwhm_bc/(c*2.355))*lam_bc

    for j, fname in enumerate(self.bc):
        ssp = np.loadtxt(fname)
        ssbc1=ssp[:,1]
        lam_bc1=ssp[:,0]
        deltbc=lam_bc1[1]-lam_bc1[0]
        #add zeros or cut to work with the same wavelenght as the masked stellar templates
        if lam_bc1[0]>lamRange_temp[0]:
            wzle=np.arange(-lam_bc1[0],-lamRange_temp[0],deltbc)
            fzle=wzle*0
            wzle=np.sort(-wzle)
            lam_bc=np.append(wzle,lam_bc1)
            sspbc=np.append(fzle,ssbc1)
        else:
            lmks=(lam_bc1>=lamRange_temp[0])
            lam_bc=lam_bc1[lmks]
            sspbc=ssbc1[lmks]

        if lam_bc1[-1]<lamRange_temp[-1]:
            wrge=np.arange(lam_bc1[-1]+deltbc,lamRange_temp[-1]+deltbc,deltbc)
            frge=wrge*0
            lam_bc=np.append(lam_bc,wrge)
            sspbc=np.append(sspbc,frge)
        else:
            hmks=(lam_bc<=lamRange_temp[-1])
            lam_bc=lam_bc[hmks]
            sspbc=sspbc[hmks]
        #Define the sigma difference in pixels as zero
        sigma2d=sspbc*0
        sspbc = util.gaussian_filter1d(sspbc, sigma2d)  # perform convolution with variable sigma
        sspNew, logLam3, velscale = util.log_rebin(lamRange_temp, sspbc, velscale=velscale)
        mfm=(sspNew>0) #To avoid ~zero median
        sspfm=sspNew[mfm]
        #sspNew=np.append(sspNew,0) #In some cases the template size is different
         #In some cases the template size is different
        if len(sspNew)!=Templates_size:
            dftl=int(len(sspNew)-Templates_size)
            if dftl>0:
                sspNew=sspNew[:-dftl]
            else:
                sadf=np.zeros([np.abs(dftl)])
                sspNew=np.append(sspNew,sadf)

        self.bc_templates[:, j] = sspNew  #No Normalized templates
    return self.bc_templates


def read_sdss_file(dir_file):
    h=fits.open(dir_file)
    dh=h[1].data
    log_lam=dh['loglam']
    lam_gal=10**log_lam
    ivar=dh['ivar']
    wdisp=dh['wdisp']
    #get the velscale
    c = 299792.458                  # speed of light in km/s
    frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
    velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel
    dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
    fwhm_gal = 2.355*wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms
    return lam_gal, ivar, wdisp, velscale, fwhm_gal


###########FIXING LUMINOSITIES############
# Relative Luminosity
#L_Halpha vs L_5100 #ESTIMATING BLACK HOLE MASSES USING H_alpha, Greene & Ho 2005
def L_halpha(L_5100):
    L_Ha=(5.25)*10**42*(L_5100/10**44)**1.157 #ergs/s
    return L_Ha
#L_MgII vs L_3000 #A CATALOG OF QUASAR PROPERTIES FROM SLOAN DIGITAL SKY SURVEY DATA RELEASE 7, Yue Shen 2011
def L_Mgii(L_3000):
    log_L_Mg=2.22+0.909*np.log10(L_3000)
    L_Mg=10**log_L_Mg
    return L_Mg

#Function to find the nearest wavelenth index
def fnearest(a,v):
    a=np.asarray(a)
    idx=(np.abs(a-v)).argmin()
    return idx

# Function to calculate E(z) = 1 / sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
def E_inv(z, Omega_m, Omega_Lambda):
    return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

# Function to calculate luminosity distance
def luminosity(AGN,wave,z,wv=5100):
    # units=='1E-17 erg/cm^2/s/Ang' for densit flux
    uscf=1#e-17
    #Find the index at 5100A
    idx=fnearest(wave,wv)
    #flux
    flux=AGN[idx]*wv*uscf
    
    c=299792.458 # Speed of light in km/s
    H0 = 70.0  # Hubble constant in km/s/Mpc
    Omega_m = 0.3  # Matter density parameter
    Omega_Lambda = 0.7  # Dark energy density parameter

    integral, _ = quad(E_inv, 0, z, args=(Omega_m, Omega_Lambda))
    D_H = c / H0  # Hubble distance in Mpc
    D_C = D_H * integral  # Comoving distance in Mpc
    D_L = (1 + z) * D_C  # Luminosity distance in Mpc
    D_L = D_L*3.086e24
 

    #L=int(4*np.pi*D_L**2*flux)
    L=4*np.pi*D_L**2*flux
    return L

def luminosity_el(eml,wave,z):
    # units=='1E-17 erg/cm^2/s/Ang' for densit flux
    uscf=1#e-17
    #Find the index at 5100A
    #idx=fnearest(wave,wv)
    #flux
    flux=np.trapezoid(eml,x=wave)
    
    c=299792.458 # Speed of light in km/s
    H0 = 70.0  # Hubble constant in km/s/Mpc
    Omega_m = 0.3  # Matter density parameter
    Omega_Lambda = 0.7  # Dark energy density parameter

    integral, _ = quad(E_inv, 0, z, args=(Omega_m, Omega_Lambda))
    D_H = c / H0  # Hubble distance in Mpc
    D_C = D_H * integral  # Comoving distance in Mpc
    D_L = (1 + z) * D_C  # Luminosity distance in Mpc
    D_L = D_L*3.086e24
 

    #L=int(4*np.pi*D_L**2*flux)
    L=4*np.pi*D_L**2*flux
    return L

#####################
def estimate_RfeII(FWHM_Hb_kms=None, edd_ratio=None):
    """
    Estimate R_FeII = Fe II(λ4570)/Hβ_broad based on either FWHM(Hβ)
    or Eddington ratio (L/L_Edd). One of the two inputs must be provided.
    
    Parameters
    ----------
    FWHM_Hb_kms : float, optional
        Full width at half maximum of Hβ broad component in km/s.
    edd_ratio : float, optional
        Dimensionless Eddington ratio (L_bol / L_Edd).
    
    Returns
    -------
    R_feII : float
        Estimated ratio of Fe II λ4570 flux to broad Hβ flux.
    Refs
    -----
    Boroson & Green (1992), ApJS 80, 109 — discovery of Eigenvector 1.

    Sulentic et al. (2000–2007) — the “4DE1” parameter space.

    Dong et al. (2011) — Fe II vs Eddington ratio correlation in SDSS AGNs.

    Marziani et al. (2021) — recent review of Fe II and EV1 physics.

    Swayam Panda ...
    """
    if (FWHM_Hb_kms is None) and (edd_ratio is None):
        raise ValueError("Provide either FWHM_Hb_kms or edd_ratio")
    
    # Empirical scalings (approximate)
    if edd_ratio is not None:
        # use a simple power‐law scaling: R_feII ∝ (edd_ratio)^0.8
        R_feII = 0.3 * (edd_ratio / 0.1)**0.8
    else:
        # anti‐correlation with FWHM: broader lines → smaller R_feII
        # assume typical FWHM of 4000 km/s → R_feII ≈ 0.5
        R_feII = 0.5 * (FWHM_Hb_kms / 4000.0)**(-1.0)
    
    # impose some limits
    R_feII = max(0.05, R_feII)
    R_feII = min(3.0, R_feII)
    
    return R_feII

#function to convolve the components before sum up
def convolve_loglam(spec, loglam, fwhm_inst_kms):
    """
    Convolve a spectrum (in flux units) sampled on a logarithmic wavelength grid
    with a Gaussian LSF of given FWHM (km/s).

    Parameters
    ----------
    spec : array
        Spectrum flux density (same units for all components).
    loglam : array
        Log10(wavelength in Angstroms), e.g., from SDSS.
    fwhm_inst_kms : float
        Instrumental FWHM in km/s (depends on resolution R = c / FWHM).

    Returns
    -------
    spec_conv : array
        Convolved spectrum, same length as input.
    """
    c = 299792.458  # km/s
    delta_loglam = np.median(np.diff(loglam))
    # velocity step per pixel
    dv_pix = c * np.log(10) * delta_loglam  # km/s per pixel
    sigma_inst_pix = (fwhm_inst_kms / 2.3548) / dv_pix  # Gaussian sigma in pixels

    # make Gaussian kernel
    half_size = int(6 * sigma_inst_pix)
    x = np.arange(-half_size, half_size + 1)
    kernel = np.exp(-0.5 * (x / sigma_inst_pix) ** 2)
    kernel /= kernel.sum()

    # convolve (using FFT for speed)
    spec_conv = fftconvolve(spec, kernel, mode='same')
    return spec_conv


