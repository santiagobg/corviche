CORVICHE: Is a typical Ecuadorian food that you should try!
# CORVICHE
# COde for Reproducing uV-optICal Hybrid AGN spEctra

# Author: Santiago Bernal
# Version 1.0

# If you use this code, please cite: 
Bernal et al. 2026, A&A, Volume 707, id.A206, 19 pp. 


# Use
You must make a copy of all the archives. Then you can add the PATH of the file corviche.py or
You can use the notebook "synthetic_spectrum_example_V1.ipynb" 

# Description

CORVICHE is a code able to generate AGN spectra in the redshift range [0, 2.5]. The resulting
spectrum has the same resolution and observed wavelength range as an SDSS BOSS spectrum.
You can constrain the redshift, continuum power-law slope, continuum power-law luminosity,
BELs intensity, BELs FWHM, NELs FWHM, ratio AGN/total continuum*, signal-to-noise ratio**

*AGN/total continuum ratio is restricted to certain values.
** The median-all signal-to-noise ratio can be set and takes values of 2-5-10-20-30

The models of the spectrum are
Type I: spec= power-law + stars + NELs 
Type II: spec= power-law + stars + NELs + BELs + Fe II

Also, outflows in the [O III]5007 emission lines can be added, as well as 
a second BELs component with a velocity shift and individual FWHM



