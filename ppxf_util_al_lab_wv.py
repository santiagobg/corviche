#######################################################################
#
# Copyright (C) 2001-2015, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#######################################################################
#
# NAME:
#   LOG_REBIN
#
# PURPOSE:
#   Logarithmically rebin a spectrum, while rigorously conserving the flux.
#   Basically the photons in the spectrum are simply redistributed according
#   to a new grid of pixels, with non-uniform size in the spectral direction.
#
#   This routine makes the `standard' zero-order assumption that the spectrum
#   is *constant* within each pixels. It is possible to perform log-rebinning
#   by assuming the spectrum is represented by a piece-wise polynomial of
#   higher degree, while still obtaining a uniquely defined linear problem,
#   but this reduces to a deconvolution and amplifies noise.
#
# CALLING SEQUENCE:
#   LOG_REBIN, lamRange, spec, specNew, logLam, $
#       OVERSAMPLE=oversample, VELSCALE=velScale, /FLUX
#
# INPUTS:
#   LAMRANGE: two elements vector containing the central wavelength
#       of the first and last pixels in the spectrum, which is assumed
#       to have constant wavelength scale! E.g. from the values in the
#       standard FITS keywords: LAMRANGE = CRVAL1 + [0,CDELT1*(NAXIS1-1)].
#       It must be LAMRANGE[0] < LAMRANGE[1].
#   SPEC: input spectrum.
#
# OUTPUTS:
#   SPECNEW: logarithmically rebinned spectrum.
#   LOGLAM: log(lambda) (*natural* logarithm: ALOG) of the central
#       wavelength of each pixel. This is the log of the geometric
#       mean of the borders of each pixel.
#
# KEYWORDS:
#   FLUX: Set this keyword to preserve total flux. In this case the
#       log rebinning changes the pixels flux in proportion to their
#       dLam so the following command will show large differences
#       beween the spectral shape before and after LOG_REBIN:
#
#           plot, exp(logLam), specNew  # Plot log-rebinned spectrum
#           oplot, range(lamRange[0],lamRange[1],n_elements(spec)), spec
#
#       By defaul, when this keyword is *not* set, the above two lines
#       produce two spectra that almost perfectly overlap each other.
#   OVERSAMPLE: Oversampling can be done, not to loose spectral resolution,
#       especally for extended wavelength ranges and to avoid aliasing.
#       Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
#   VELSCALE: velocity scale in km/s per pixels. If this variable is
#       not defined, then it will contain in output the velocity scale.
#       If this variable is defined by the user it will be used
#       to set the output number of pixels and wavelength scale.
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lamRange[0] < lamRange[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous version.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.2.0: Included gaussian_filter1d routine, which is a replacement for
#       the Scipy routine with the same name, to be used with variable sigma
#       per pixel. MC, Oxford, 10 October 2015
#
#----------------------------------------------------------------------

from __future__ import print_function

import numpy as np

def log_rebin(lamRange, spec, oversample=False, velscale=None, flux=False):
    """
    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.
    
    When the flux keyword is set, this program performs an exact integration 
    of the original spectrum, assumed to be a step function within the 
    linearly-spaced pixels, onto the new logarithmically-spaced pixels. 
    The output was tested to agree with the analytic solution.

    """
    lamRange = np.asarray(lamRange)
    if len(lamRange) != 2:
        raise ValueError('lamRange must contain two elements')
    if lamRange[0] >= lamRange[1]:
        raise ValueError('It must be lamRange[0] < lamRange[1]')
    s = spec.shape
    if len(s) != 1:
        raise ValueError('input spectrum must be a vector')
    n = s[0]
    if oversample:
        m = int(n*oversample)
    else:
        m = int(n)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale

#----------------------------------------------------------------------
#
# PPXF_DETERMINE_GOODPIXELS: Example routine to generate the vector of goodPixels
#     to be used as input keyword for the routine PPXF. This is useful to mask
#     gas emission lines or atmospheric absorptions.
#     It can be trivially adapted to mask different lines.
#
# INPUT PARAMETERS:
# - LOGLAM: Natural logarithm ALOG(wave) of the wavelength in Angstrom
#     of each pixel of the log rebinned *galaxy* spectrum.
# - LAMRANGETEMP: Two elements vectors [lamMin2,lamMax2] with the minimum and
#     maximum wavelength in Angstrom in the stellar *template* used in PPXF.
# - Z: Estimate of the galaxy redshift.
#
# V1.0.0: Michele Cappellari, Leiden, 9 September 2005
# V1.0.1: Made a separate routine and included additional common emission lines.
#   MC, Oxford 12 January 2012
# V2.0.0: Translated from IDL into Python. MC, Oxford, 10 December 2013
# V2.0.1: Updated line list. MC, Oxford, 8 January 2014
# V2.0.2: Use redshift instead of velocity as input for higher accuracy at large z.
#   MC, Lexington, 31 March 2015

def determine_goodpixels(logLam, lamRangeTemp, z):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
#    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
#                    "skyline,---- OI doublet,----- Na doublet
    lines = np.array([5577.0, 6300.20, 6367.67, 5889.950, 5895.924])# PA:en vez de emission lines dejo skylines
    #    dv = lines*0 + 800 # width/2 of masked gas emission region in km/s
    dv = lines*0 + 100 # width/2 of masked sky lines in km/s
    c = 299792.458 # speed of light in km/s
    z=0.0                                           #PA: le quito el redshift, ahora solo sirve para skylines!!!!!
    flag = np.zeros_like(logLam, dtype=bool)
    for line, dvj in zip(lines, dv):
        flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
              & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)   # stellar library

    return np.where(flag == 0)[0]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---------------------Mask some regions as good pixels----------------------------

def region_goodpixels(logLam, lamRangeTemp, z):
    """
    Generates a list of goodpixels to mask a given set of regions using the rest frame wavelenght.
    This is meant to be used as input for PPXF.

    """
    lines = np.array([3650])# PA:en vez de emission lines dejo skylines
    #    dv = lines*0 + 800 # width/2 of masked gas emission region in km/s
    width2=150
    dv = lines*0 + width2 # width/2 of masked sky lines in km/s
    c = 299792.458 # speed of light in km/s
    z=0.0                                           #PA: le quito el redshift, ahora solo sirve para skylines!!!!!
    flag = np.zeros_like(logLam, dtype=bool)
    for line, dvj in zip(lines, dv):
        flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
              & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)   # stellar library

    return np.where(flag == 0)[0]

#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom

    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha  (lab wavelengths)
    line_wave = np.array([4102.89, 4341.68, 4862.68, 6564.61])
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    #                 -----[OII]-----    -----[SII]-----
    #    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85, 5577.0])
    #names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731', 'skyline'])
    lines = np.array([3726.03, 3728.82, 6718.29, 6732.67])          
    names = np.array(['[OII]3726', '[OII]3729', '[SII]6718', '[SII]6733', ])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

    #                 -----[OIII]-----
    lines = np.array([4960.30, 5008.24])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.35*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    #                  -----[OI]-----
    lines = np.array([6365.54, 6302.05])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.33*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    #                 -----[NII]-----
    lines = np.array([6549.85, 6585.28])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.34*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

#                 -----[NI]----- PA: added doublet, equal intensities
    lines = np.array([5197.9, 5200.39])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NI]5200d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])
#Emission Lines added using the VANDEN BERK et al. 2001 (Composite QSO spectra Ref)
#In order to preserv the air wavelength ref we use the vac to air formula from sdss
#vac/(1+2.735182e-4+131.4182/vac**2+2.76249e8/vac**4)
 #                 -----MgII-----
    lines = np.array([2795.528,2802.705])
    singlet = np.exp(-0.5*((lam - lines[0])/sigma)**2)+np.exp(-0.5*((lam - lines[1])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, singlet])
    line_names = np.append(line_names, 'MgIId') # single template #Using ratio 1:1 that is correct for saturated case 
    line_wave = np.append(line_wave, lines[0])
# MgII lines from the web site http://astronomy.nmsu.edu/drewski/tableofemissionlines.html 

#    ------[OII]7321.48----[NiIII]7892.10----#NO[FeVII]6086.29----###NO[FeVII]5400---
    lines = np.array([7321.48, 7892.10]) 
    names = np.array(['[OII]7321','[NiIII]7892'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#      ----[NeIII]3968.58--[NeIII]3869.85------[NeIV]2423.83-----
    lines = np.array([3968.58,3868.75,2423.83])
    names = np.array(['[NeIII]3968','[NeIII]3869','[NeIV]2424'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#      ----[NeV]3346.82---[NeV]3426.84
    lines = np.array([3346.82,3426.84]) 
    names = np.array(['[NeV]3346','[NeV]3426'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
    
#   ---------[OIII]4364.44-------[OII]2471.03----
    lines = np.array([4364.44,2471.03])
    names = np.array(['[OIII]4364','[OII]2471'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------[NeIII]1814.73--[SII]4073.63--[CaV]5310.59--[ArIII]7137.80
    lines = np.array([1814.73,4073.63,5310.59,7137.80])
    names = np.array(['[NeIII]1814','[SII]4073','[CaV]5310','[ArIII]7137'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   -------- CII]2326.44----CIII]1908.73---SiIII]1892.03----NIII]1750.26----semi-forbiden
    lines = np.array([2326.44,1908.73,1892.03,1750.26]) 
    names = np.array(['CII]2326','CIII]1908','SiIII]1892','NIII1750'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#   -------OIV]1402.06--OIII]1663.48--AlII]2669.95 -----semi-forbiden
    lines = np.array([1402.06,1663.48,2669.95]) 
    names = np.array(['OIV]1402','OIII]1663','AlII]2669'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



#       ----OIII2672.04--HeI3188.67--HeI3588.30
    lines = np.array([2672.04,3188.67,3588.30]) 
    names = np.array(['OIII2672','HeI3188','HeI3588'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----OIII3133.70-----HeI3889.74---HII4687.02---HeI5877.29---HeI7067.20
    lines = np.array([3133.70,3889.74,4687.02, 5877.29,7067.20]) 
    names = np.array(['OIII3133','HeI3889','HII4687','HeI5877','HeI7067'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----SiII1306.82-----AlII1670.79---NIV1718.55---AlII1721.89--SiII1816.98--AlIII1857.40
    lines = np.array([1306.82,1670.79,1718.55,1721.89,1816.98,1857.40]) 
    names = np.array(['SiII1306','AlII1670','NIV1718','AlII1721','SiII1816','AlIII1857'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------HeII1640.42----SiIV1396.76---CII1335.30----OI1304.35---SiII1262.59---NV1240.14
    lines = np.array([1640.42,1396.76,1335.30,1304.35,1262.59,1240.14]) 
    names = np.array(['HeII1640','SiIV1396','CII1335','OI1304','SiII1262','NV1240'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#   ---CIV1548.187---CIV1550.772---
    lines = np.array([1547.604,1550.189]) 
    names = np.array(['CIVd']) #Using ratio 1:1 that is correct for saturated case 
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines[0])/sigma)**2)+np.exp(-0.5*((lam[:,np.newaxis] - lines[1])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines[1])

    
    #CIV lines from the web site http://astronomy.nmsu.edu/drewski/tableofemissionlines.html and transform to lab with the
    #function vac_to_lab

#  -------- CIII*1175.70----CIII977.02--- *multiplete?
    lines = np.array([1175.70,977.02]) 
    names = np.array(['CIII1175','CIII977'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


# Lyman series----Lya1215.67----Lyb1025.72---Lye/Lyd937.80----
    lines = np.array([1215.67,1025.72,937.80])
    names = np.array(['Lya1215','Lyb1025','Lye/Lyd937'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

   # print('Emission lines included in gas templates:')
   # print(line_names)

    return emission_lines, line_names, line_wave

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def emission_lines_lft(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom

    # Balmer Series:      Hdelta   Hgamma    Hbeta    (lab wavelengths)
    line_wave = np.array([4102.89, 4341.68, 4862.68 ])
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    #                 -----[OII]-----    -----[SII]-----
    #    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85, 5577.0])
    #names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731', 'skyline'])
    lines = np.array([3726.03, 3728.82])          
    names = np.array(['[OII]3726', '[OII]3729' ])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

    #                 -----[OIII]-----
    lines = np.array([4960.30, 5008.24])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.35*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])


#                 -----[NI]----- PA: added doublet, equal intensities
    lines = np.array([5197.9, 5200.39])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NI]5200d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])
#Emission Lines added using the VANDEN BERK et al. 2001 (Composite QSO spectra Ref)
#In order to preserv the air wavelength ref we use the vac to air formula from sdss
#vac/(1+2.735182e-4+131.4182/vac**2+2.76249e8/vac**4)
 #                 -----MgII-----
    lines = np.array([2798.75])
    singlet = np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, singlet])
    line_names = np.append(line_names, 'MgII2798.75') # single template 
    line_wave = np.append(line_wave, lines[0])

#      ----[NeIII]3968.58--[NeIII]3869.85------[NeIV]2423.83-----
    lines = np.array([3968.58,3868.75,2423.83])
    names = np.array(['[NeIII]3968','[NeIII]3869','[NeIV]2424'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#      ----[NeV]3346.82---[NeV]3426.84
    lines = np.array([3346.82,3426.84]) 
    names = np.array(['[NeV]3346','[NeV]3426'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
    
#   ---------[OIII]4364.44-------[OII]2471.03----
    lines = np.array([4364.44,2471.03])
    names = np.array(['[OIII]4364','[OII]2471'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------[NeIII]1814.73--[SII]4073.63--[CaV]5310.59--[ArIII]7137.80
    lines = np.array([1814.73,4073.63,5310.59,7137.80])
    names = np.array(['[NeIII]1814','[SII]4073','[CaV]5310','[ArIII]7137'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   -------- CII]2326.44----CIII]1908.73---SiIII]1892.03----NIII]1750.26----semi-forbiden
    lines = np.array([2326.44,1908.73,1892.03,1750.26]) 
    names = np.array(['CII]2326','CIII]1908','SiIII]1892','NIII1750'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#   -------OIV]1402.06--OIII]1663.48--AlII]2669.95 -----semi-forbiden
    lines = np.array([1402.06,1663.48,2669.95]) 
    names = np.array(['OIV]1402','OIII]1663','AlII]2669'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



#       ----OIII2672.04--HeI3188.67--HeI3588.30
    lines = np.array([2672.04,3188.67,3588.30]) 
    names = np.array(['OIII2672','HeI3188','HeI3588'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----OIII3133.70-----HeI3889.74---HII4687.02---HeI5877.29---HeI7067.20
    lines = np.array([3133.70,3889.74,4687.02, ]) 
    names = np.array(['OIII3133','HeI3889','HII4687'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----SiII1306.82-----AlII1670.79---NIV1718.55---AlII1721.89--SiII1816.98--AlIII1857.40
    lines = np.array([1306.82,1670.79,1718.55,1721.89,1816.98,1857.40]) 
    names = np.array(['SiII1306','AlII1670','NIV1718','AlII1721','SiII1816','AlIII1857'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------HeII1640.42----CIV1548.187---CIV1550.772----SiIV1396.76---CII1335.30----OI1304.35---SiII1262.59---NV1240.14
    lines = np.array([1640.42,1548.187,1550.772,1396.76,1335.30,1304.35,1262.59,1240.14]) 
    names = np.array(['HeII1640','CIV1548','CIV1550','SiIV1396','CII1335','OI1304','SiII1262','NV1240'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
#CIV lines from the web site http://astronomy.nmsu.edu/drewski/tableofemissionlines.html

#  -------- CIII*1175.70----CIII977.02--- *multiplete?
    lines = np.array([1175.70,977.02]) 
    names = np.array(['CIII1175','CIII977'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


# Lyman series----Lya1215.67----Lyb1025.72---Lye/Lyd937.80----
    lines = np.array([1215.67,1025.72,937.80])
    names = np.array(['Lya1215','Lyb1025','Lye/Lyd937'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def emission_lines_rgt(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom

    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha  (lab wavelengths)
    line_wave = np.array([ 6564.61])
    line_names = np.array(['Halpha'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    #                 -----[OII]-----    -----[SII]-----
    #    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85, 5577.0])
    #names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731', 'skyline'])
    lines = np.array([ 6718.29, 6732.67])          
    names = np.array([ '[SII]6718', '[SII]6733' ])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


    #                  -----[OI]-----
    lines = np.array([6365.54, 6302.05])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.33*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    #                 -----[NII]-----
    lines = np.array([6549.85, 6585.28])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.34*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])


#    ------[OII]7321.48----[NiIII]7892.10----#NO[FeVII]6086.29----###NO[FeVII]5400---
    lines = np.array([7321.48, 7892.10]) 
    names = np.array(['[OII]7321','[NiIII]7892'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)





    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]
    
    if len(line_wave)<1:
        line_wave = np.array([ 6564.61])
        line_names = np.array(['Halpha'])
        emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave





#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

########NNNNNNNNNNNN
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def emission_lines_lft_no_H(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom

    # Balmer Series:      Hdelta   Hgamma    Hbeta    (lab wavelengths)
#    line_wave = np.array([4102.89, 4341.68, 4862.68 ])
#    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta'])
#    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    #                 -----[OII]-----    -----[SII]-----
    #    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85, 5577.0])
    #names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731', 'skyline'])
    line_wave = np.array([3726.03, 3728.82])          
    line_names = np.array(['[OII]3726', '[OII]3729' ])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)
#    emission_lines = np.column_stack([emission_lines, gauss])
#    line_names = np.append(line_names, names)
 #   line_wave = np.append(line_wave, lines)

    #                 -----[OIII]-----
    lines = np.array([4960.30, 5008.24])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.35*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])


#                 -----[NI]----- PA: added doublet, equal intensities
    lines = np.array([5197.9, 5200.39])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NI]5200d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])
#Emission Lines added using the VANDEN BERK et al. 2001 (Composite QSO spectra Ref)
#In order to preserv the air wavelength ref we use the vac to air formula from sdss
#vac/(1+2.735182e-4+131.4182/vac**2+2.76249e8/vac**4)
 #                 -----MgII-----
    lines = np.array([2798.75])
    singlet = np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, singlet])
    line_names = np.append(line_names, 'MgII2798.75') # single template 
    line_wave = np.append(line_wave, lines[0])

#      ----[NeIII]3968.58--[NeIII]3869.85------[NeIV]2423.83-----
    lines = np.array([3968.58,3868.75,2423.83])
    names = np.array(['[NeIII]3968','[NeIII]3869','[NeIV]2424'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#      ----[NeV]3346.82---[NeV]3426.84
    lines = np.array([3346.82,3426.84]) 
    names = np.array(['[NeV]3346','[NeV]3426'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
    
#   ---------[OIII]4364.44-------[OII]2471.03----
    lines = np.array([4364.44,2471.03])
    names = np.array(['[OIII]4364','[OII]2471'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------[NeIII]1814.73--[SII]4073.63--[CaV]5310.59--[ArIII]7137.80
    lines = np.array([1814.73,4073.63,5310.59,7137.80])
    names = np.array(['[NeIII]1814','[SII]4073','[CaV]5310','[ArIII]7137'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   -------- CII]2326.44----CIII]1908.73---SiIII]1892.03----NIII]1750.26----semi-forbiden
    lines = np.array([2326.44,1908.73,1892.03,1750.26]) 
    names = np.array(['CII]2326','CIII]1908','SiIII]1892','NIII1750'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#   -------OIV]1402.06--OIII]1663.48--AlII]2669.95 -----semi-forbiden
    lines = np.array([1402.06,1663.48,2669.95]) 
    names = np.array(['OIV]1402','OIII]1663','AlII]2669'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



#       ----OIII2672.04--HeI3188.67--HeI3588.30
    lines = np.array([2672.04,3188.67,3588.30]) 
    names = np.array(['OIII2672','HeI3188','HeI3588'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----OIII3133.70-----HeI3889.74---HII4687.02---HeI5877.29---HeI7067.20
    lines = np.array([3133.70,3889.74,4687.02,5877.29 ]) 
    names = np.array(['OIII3133','HeI3889','HII4687','HeI5877'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----SiII1306.82-----AlII1670.79---NIV1718.55---AlII1721.89--SiII1816.98--AlIII1857.40
    lines = np.array([1306.82,1670.79,1718.55,1721.89,1816.98,1857.40]) 
    names = np.array(['SiII1306','AlII1670','NIV1718','AlII1721','SiII1816','AlIII1857'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------HeII1640.42----CIV1548.187---CIV1550.772----SiIV1396.76---CII1335.30----OI1304.35---SiII1262.59---NV1240.14
    lines = np.array([1640.42,1548.187,1550.772,1396.76,1335.30,1304.35,1262.59,1240.14]) 
    names = np.array(['HeII1640','CIV1548','CIV1550','SiIV1396','CII1335','OI1304','SiII1262','NV1240'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
#CIV lines from the web site http://astronomy.nmsu.edu/drewski/tableofemissionlines.html

#  -------- CIII*1175.70----CIII977.02--- *multiplete?
    lines = np.array([1175.70,977.02]) 
    names = np.array(['CIII1175','CIII977'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


# Lyman series----Lya1215.67----Lyb1025.72---Lye/Lyd937.80----
    lines = np.array([1215.67,1025.72,937.80])
    names = np.array(['Lya1215','Lyb1025','Lye/Lyd937'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def emission_lines_rgt_all_H_no_Hbeta(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom
    # Balmer Series:      Hdelta   Hgamma      Halpha  (lab wavelengths)
    line_wave = np.array([4102.89, 4341.68, 6564.61])
    line_names = np.array(['Hdelta', 'Hgamma', 'Halpha'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha  (lab wavelengths)
    #line_wave = np.array([ 6564.61])
    #line_names = np.array(['Halpha'])
    #emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    #                 -----[OII]-----    -----[SII]-----
    #    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85, 5577.0])
    #names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731', 'skyline'])
    lines = np.array([ 6718.29, 6732.67])          
    names = np.array([ '[SII]6718', '[SII]6733' ])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


    #                  -----[OI]-----
    lines = np.array([6365.54, 6302.05])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.33*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    #                 -----[NII]-----
    lines = np.array([6549.85, 6585.28])
    doublet = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.34*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])


#    ------[OII]7321.48----[NiIII]7892.10----#NO[FeVII]6086.29----###NO[FeVII]5400---
    lines = np.array([7321.48, 7892.10]) 
    names = np.array(['[OII]7321','[NiIII]7892'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)





    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]
    
    if len(line_wave)<1:
        line_wave = np.array([ 6564.61])
        line_names = np.array(['Halpha'])
        emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave





#------------------------------------------------------------------------------
#------------------------------------------------------------------------------




def emission_o3_winds(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Generates an array of Gaussian emission lines to be used as templates in PPXF.
    Additional lines can be easily added by editing this procedure.

    - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
      logLam_temp should be the same as that of the stellar templates.

    - lamRange_gal is the estimated rest-frame fitted wavelength range
      Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
      where wave is the observed wavelength of the fitted galaxy pixels and
      z is an initial very rough estimate of the galaxy redshift.

    - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
      Angstrom. Here it is assumed constant. It could be a function of wavelength.

    - The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom

    # [OIII]d    (lab wavelengths)
    lines = np.array([4960.30, 5008.24])
    line_wave = np.array([5008.24])
    line_names = np.array(['[OIII]5007dw'])
    emission_lines = np.exp(-0.5*((lam - lines[1])/sigma)**2) + 0.35*np.exp(-0.5*((lam - lines[0])/sigma)**2)
    return emission_lines, line_names, line_wave
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def Broad_emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
        Generates an array of Gaussian emission lines to be used as templates in PPXF.
        Additional lines can be easily added by editing this procedure.
        
        - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
        logLam_temp should be the same as that of the stellar templates.
        
        - lamRange_gal is the estimated rest-frame fitted wavelength range
        Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial very rough estimate of the galaxy redshift.
        
        - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
        Angstrom. Here it is assumed constant. It could be a function of wavelength.
        """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom


    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha  (lab wavelengths)
    line_wave = np.array([4102.89, 4341.68, 4862.68, 6564.61])
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)


 #                 -----MgII-----
    lines = np.array([2798.75])
    singlet = np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, singlet])
    line_names = np.append(line_names, 'MgIId') # # single template 
    line_wave = np.append(line_wave, lines[0])    
  

#New set of BROAD lines

#   -------- CII]2326.44----CIII]1908.73---SiIII]1892.03----NIII]1750.26----semi-forbiden
    lines = np.array([2326.44,1908.73,1892.03,1750.26]) 
    names = np.array(['CII]2326','CIII]1908','SiIII]1892','NIII1750'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#   -------OIV]1402.06--OIII]1663.48--AlII]2669.95 -----semi-forbiden
    lines = np.array([1402.06,1663.48,2669.95]) 
    names = np.array(['OIV]1402','OIII]1663','AlII]2669'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



#       ----OIII2672.04--HeI3188.67--HeI3588.30
    lines = np.array([2672.04,3188.67,3588.30]) 
    names = np.array(['OIII2672','HeI3188','HeI3588'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----OIII3133.70-----HeI3889.74---HII4687.02---HeI5877.29---HeI7067.20
    lines = np.array([3133.70,3889.74,4687.02, 5877.29,7067.20]) 
    names = np.array(['OIII3133','HeI3889','HII4687','HeI5877','HeI7067'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#       ----SiII1306.82-----AlII1670.79---NIV1718.55---AlII1721.89--SiII1816.98--AlIII1857.40
    lines = np.array([1306.82,1670.79,1718.55,1721.89,1816.98,1857.40]) 
    names = np.array(['SiII1306','AlII1670','NIV1718','AlII1721','SiII1816','AlIII1857'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------HeII1640.42----CIV1548.187---CIV1550.772----SiIV1396.76---CII1335.30----OI1304.35---SiII1262.59---NV1240.14
    lines = np.array([1640.42,1548.187,1550.772,1396.76,1335.30,1304.35,1262.59,1240.14]) 
    names = np.array(['HeII1640','CIV1548','CIV1550','SiIV1396','CII1335','OI1304','SiII1262','NV1240'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)
#CIV lines from the web site http://astronomy.nmsu.edu/drewski/tableofemissionlines.html


#  -------- CIII*1175.70----CIII977.02--- *multiplete?
    lines = np.array([1175.70,977.02]) 
    names = np.array(['CIII1175','CIII977'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


# Lyman series----Lya1215.67----Lyb1025.72---Lye/Lyd937.80----
    lines = np.array([1215.67,1025.72,937.80])
    names = np.array(['Lya1215','Lyb1025','Lye/Lyd937'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]
    
    #print('Emission lines included in Broad gas templates:')
    #print(line_names)
    
    return emission_lines, line_names, line_wave
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# V1.0.0: Michele Cappellari, Oxford, 7 January 2014
# V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
# V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numearical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014

def Broad_nBab_emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
        Generates an array of Gaussian emission lines to be used as templates in PPXF.
        Additional lines can be easily added by editing this procedure.
        
        - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
        logLam_temp should be the same as that of the stellar templates.
        
        - lamRange_gal is the estimated rest-frame fitted wavelength range
        Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial very rough estimate of the galaxy redshift.
        
        - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
        Angstrom. Here it is assumed constant. It could be a function of wavelength.
        """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom
    
    # Balmer Series:      Hdelta   Hgamma    (air wavelengths)
    line_wave = np.array([4101.76, 4340.47])
    line_names = np.array(['Hdelta', 'Hgamma'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)
    

#New set of BROAD lines
#   -------- CII]2325.73----CIII]1908.1---SiIII]1891.40----NIII]1749.66----semi-forbiden
    lines = np.array([2325.73,1908.1,1891.40,1749.65]) 
    names = np.array(['CII]2326','CIII]1909','SiIII]1891.40','NIII1749.65'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

#       ----OIII3132.79-----HeI3888.64---HII4685.65---HeI5875.57---HeI7065.25---HI3888.64
    lines = np.array([3132.79,3888.64,4685.65, 5875.57,7065.25,3888.64]) 
    names = np.array(['OIII3132.79','HeI3888.64','HII4685.71','HeI5875.62','HeI7065.25','HI3888.64'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#   ---------HeII1639.83----CIV1548.47----SiIV1396.18---CII1334.72----OI1303.7---SiII1262.00---NV1239.55-
    lines = np.array([1639.83,1548.47,1396.18,1334.72,1303.7,1262.00,1239.55]) 
    names = np.array(['HeII1639.83','CIV1548.47','SiIV1396.18','CII1334.72','OI1303.7','SiII1262.00','NV1239.55'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


#  -------- CIII*1175.1----CIII976.32--- *multiplete?
    lines = np.array([1175.1,976.32]) 
    names = np.array(['CIII*1175.1','CIII976.32'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)


# Lyman series----Lya1215.08----Lyb1025.06---Lye/Lyd937.07----
    lines = np.array([1215.08,1025.06,937.07])
    names = np.array(['Lya1215.08','Lyb1025.06','Lye/Lyd937.07'])
    gauss = np.exp(-0.5*((lam[:,np.newaxis] - lines)/sigma)**2)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)



    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]
    
    print('Emission lines included in Broad gas templates:')
    print(line_names)
    
    return emission_lines, line_names, line_wave

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def Double_Balmer_emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
        Generates an array of Gaussian emission lines to be used as templates in PPXF.
        Additional lines can be easily added by editing this procedure.
        
        - logLam_temp is the natural log of the wavelength of the templates in Angstrom.
        logLam_temp should be the same as that of the stellar templates.
        
        - lamRange_gal is the estimated rest-frame fitted wavelength range
        Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial very rough estimate of the galaxy redshift.
        
        - FWHM_gal is the instrumental FWHM of the galaxy spectrum under study in
        Angstrom. Here it is assumed constant. It could be a function of wavelength.
        """
    lam = np.exp(logLam_temp)
    sigma = FWHM_gal/2.355 # Assumes instrumental sigma is constant in Angstrom
    
    # Balmer Series:     Hbeta   Halpha  (air wavelengths)
    line_wave = np.array([ 4861.33, 6562.80])
    line_names = np.array([ 'Hbeta', 'Halpha'])
    emission_lines = np.exp(-0.5*((lam[:,np.newaxis] - line_wave)/sigma)**2)
    
 #                 -----MgII]-----
    lines = np.array([2797.93])
    singlet = np.exp(-0.5*((lam - lines[0])/sigma)**2)
    emission_lines = np.column_stack([emission_lines, singlet])
    line_names = np.append(line_names, 'MgII2798.75') # single template 
    line_wave = np.append(line_wave, lines[0])



    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the PPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]
    
    print('Emission lines included in Broad double peaked gas templates:')
    print(line_names)
    
    return emission_lines, line_names, line_wave

#------------------------------------------------------------------------------
def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every
    pixel, given by the vector "sigma" with the same size as "spec".
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating  template library for SDSS data, this implementation
    is 60x faster than the naive loop over pixels.

    """

    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

#------------------------------------------------------------------------------
#----------------------------------
# AGN continuum powerlaw

def AGN_old(logLam_temp,lambda_norm):
    #lambda_norm en AA
    lam = np.exp(logLam_temp)/lambda_norm
    slope = -3
    slopes = np.array([slope])
    AGN_continuum = lam**(slope)
    slope = -2
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)
    slope = -1
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)
    slope = 0
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)
    slope = 1
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)
    slope = 2
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)
    slope = 3
    temp = lam**(slope)
    AGN_continuum = np.column_stack([AGN_continuum, temp])
    slopes = np.append(slopes, slope)

    return AGN_continuum, slopes
#------------------------------------------------------------------------------
#----------------------------------
# AGN continuum powerlaw
def AGN(logLam_temp,lambda_norm,stp=1,islope=-3,fslope=3):

    lam = np.exp(logLam_temp)/lambda_norm
    slopes = np.array([islope])
    slopes_a = np.arange(islope+stp,fslope+stp,stp)
    AGN_continuum = lam**(islope)
    for slope in slopes_a:
        temp = lam**(slope)
        AGN_continuum = np.column_stack([AGN_continuum, temp])
        slopes = np.append(slopes, slope)
    return(AGN_continuum, slopes)
#----------------------------------
#----------------------------------
# AGN continuum powerlaw
def AGN_gm(logLam_temp,lambda_norm,stp=1,islope=-3,fslope=3):

    lam = np.exp(logLam_temp)/lambda_norm
    slopes = np.array([islope])
    slopes_a = np.arange(islope+stp,fslope+stp,stp)
    AGN_continuum = lam**(islope)
    for slope in slopes_a:
        temp = lam**(slope)
        AGN_continuum = np.column_stack([AGN_continuum, temp])
        slopes = np.append(slopes, slope)
    return(AGN_continuum, slopes)
#----------------------------------

# AGN continuum powerlaw log_rebin
def AGN_rb(logLam_temp,lambda_norm,stp=1,islope=-3,fslope=3,velscale=None):

    lam = np.exp(logLam_temp)/lambda_norm
    slopes = np.array([islope])
    slopes_a = np.arange(islope+stp,fslope+stp,stp)
    AGN_continuum = lam**(islope)
    AGN_continuum=AGN_continuum/np.mean(AGN_continuum)
    lam_lim=[np.min(lam),np.max(lam)]
    AGN_rb_cont,lg,vs=log_rebin(lam_lim,AGN_continuum,velscale=velscale)
    for slope in slopes_a:
        temp = lam**(slope)
        temp=temp/np.mean(temp)
        AGN_continuum = np.column_stack([AGN_continuum, temp])
        temp_rb,lg,vs=log_rebin(lam_lim,temp,velscale=velscale)
        AGN_rb_cont=np.column_stack([AGN_rb_cont, temp_rb])
        slopes = np.append(slopes, slope)
    return(AGN_rb_cont, slopes)
#----------------------------------

# AGN continuum powerlaw selected slopes
def AGN_ss(logLam_temp,lambda_norm,slopes):

    lam = np.exp(logLam_temp)/lambda_norm
    AGN_continuum = lam**(slopes[0])
    AGN_continuum=AGN_continuum/np.mean(AGN_continuum)
    if len(slopes)>1:
        for slp in slopes[1:]:
            temp = lam**(slp)
            temp=temp/np.mean(temp)
            AGN_continuum = np.column_stack([AGN_continuum, temp])
    return(AGN_continuum, slopes)
# AGN continuum powerlaw selected slopes rebinned
def AGN_ss_rb(logLam_temp,lambda_norm,slopes,velscale=None):

    lam = np.exp(logLam_temp)/lambda_norm
    AGN_continuum = lam**(slopes[0])
    AGN_continuum=AGN_continuum/np.mean(AGN_continuum)
    lam_lim=[np.min(lam),np.max(lam)]
    AGN_rb_cont,lg,vs=log_rebin(lam_lim,AGN_continuum,velscale=velscale)
    if len(slopes)>1:
        for slp in slopes[1:]:
            temp = lam**(slp)
            temp=temp/np.mean(temp)
            AGN_continuum = np.column_stack([AGN_continuum, temp])
            temp_rb,lg,vs=log_rebin(lam_lim,temp,velscale=velscale)
            AGN_rb_cont=np.column_stack([AGN_rb_cont, temp_rb])
    return(AGN_rb_cont, slopes)
##############################################################################
##############################################################################
##############################################################################
# wavelength transformation

def vac_to_lab(vac):
    return round(vac/(1+2.735182e-4+131.4182/vac**2+2.76249e8/vac**4),3)
    
