############################ CORVICHE ############################
# COde for Reproducing uV-optICal Hybrid AGN spEctra
###############################
# Author: Santiago Bernal
# Date: May 2026
# Version 1.0

# If you use this code for your publications please cite:
# Bernal et al. 2026, A&A, Volume 707, id.A206, 19 pp.
###############################

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
import utils_corviche as util
import warnings

class corviche:


    def __init__(self,my_dir='./',my_spec_name='syntetic_spec',agn_type=2,my_z=0.8,my_nel_fwhm=400,my_bel_fwhm=5000,
                 my_slope=-1.5,agn_st_r=1,fixed_L=10**45,owinds_fc=0.0,
                 two_bels=False,my_bel_fwhm2=10000,fc_bels2=0.3,vel_2bels=-1500,sn_all=20,my_nels_fct=1,my_bels_fct=1):
        '''my_spec_name: name for your spectrum (spectra)
           agn_type: default 2, meaning the inclusion of BELs, if equal to 1 BELs are not included 
           my_z: redshift to fix the observed wavelength
           my_nel_fwhm: fixed FWHM in km/s for NELs before convolution (default 400km/s)
           my_bel_fwhm: fixed FWHM in km/s for BELs before convolution (default 5000km/s)
           my_slope: slope of the AGN power-law continuum, default slope=-1.5
           agn_st_r: Use values (1,2,3,4,5) to define the ratios between agn continuum and total continum f_pl/F_cont,
                    and stars continuum vs total continuum f_stars/F_cont
                    [f_pl/F_cont, f_stars/F_cont]=([0.9,0.1],[0.8,0.2],[0.5,0.5],[0.2,0.8],[0.1,0.9])
           fixed_L: Fixed Luminosity of the AGN continuum at 3000A or 5100A in ergs/s.
           owinds_fc: If >0 represents the flux fraction [O III]winds/[O III] that will be included in the spectrum
           sn_all: the median all pixels signal-to-noise value (only produce values of 2, 5, 10, 20, and 30)
           my_nels_fct: arbitrary factor to aplied to the flux of the Narrrow Emission Lines (NELs)
           my_bels_fct: arbitrary factor to aplied to the flux of the Broad Emission Lines (BELs)
           (optional) two_bels = True add a second BELs component with FWHM <<my_bel_fwhm2>> and relative flux to the BELs <<fc_bels2>>
                      and shifted(+/-) by the velocity <<vel_2bels>>'''
        # Redshift Warning
        if my_z>2.5:
            raise ValueError('redshift limit is 2.5')
        if agn_st_r>2:
            warnings.warn('using a agn_st_r>2 produce less real spectrum')
            

        #read the sdss file
        dir_sdss='./spectra_sdss/spec-10237-58154-0104.fits' #this an example shared, you can change for any sdss good file
        lam_gal, ivar, wdisp, velscale, fwhm_gal=util.read_sdss_file(dir_sdss)

        #Red relative flux for NELs and BELs These tables are provided
        nel_df=pd.read_csv('./Relative_fluxes_NELs_from_Venden_Berk_2001_permited_are_7_percent_161025.csv')
        bel_df=pd.read_csv('./Relative_fluxes_BELs_from_Venden_Berk_2001_161025.csv')

        
        #Initial FWHM for emission lines, use small value to later convolve with a needed value
        if my_z>1.5:
            ini_fwhm=0.08 #for Gaussianfilter1d to work well for redshift between 1.5 to 2.5 
        else:
            ini_fwhm=0.8

        #stars templates
        ssp,dv,loglam2=util.set_temp(lam_gal/(1+my_z),velscale,fwhm_gal)
################# NELs #############################
        gas_templates, line_names,line_wave=util.nels_templates(np.exp(loglam2),loglam2,ini_fwhm)

        #Convolve the templates for a fixed fwhm
        # NELs fwhm after convolution in km/s
        for t in range(gas_templates.shape[1]):
           # print(line_names[t],line_wave[t])
            gauss=gas_templates[:,t]
            cwv=line_wave[t]
            #print(line_names[t],cwv)
            #my_fwhm=300*2.355 #fixed fwhm
            new_g=util.my_fwhm_l(np.exp(loglam2),cwv,gauss,my_nel_fwhm,ini_fwhm)
            new_g=new_g/np.max(new_g)
            if t==0:
                fixed_gas_templates=new_g
            else:
                fixed_gas_templates=np.column_stack([fixed_gas_templates,new_g]) #Templates with the fixed FWHM
        #get the relative flux from table
        my_nel_rel_flux=[]
        for ln in line_names:
            q=nel_df[nel_df['line_names']==ln]
            fct=q['relative_flux'].values[0]
            my_nel_rel_flux.append(fct)
        #product between the templates and relative flux
        new_gas_lines=fixed_gas_templates*my_nel_rel_flux
################# BELs #############################
        bgas_templates, bline_names,bline_wave=util.bels_templates(np.exp(loglam2),loglam2,ini_fwhm)
        #Convolve the templates for a fixed fwhm
        # BELs fwhm after convolution
        #recommendation FWHM= 5000km/s for z<1  7000km/s for z>1
        for t in range(bgas_templates.shape[1]):
            gauss=bgas_templates[:,t]
            cwv=bline_wave[t]
            #print(line_names[t],cwv)
            #my_fwhm=300*2.355 #fixed fwhm
            new_g=util.my_fwhm_l(np.exp(loglam2),cwv,gauss,my_bel_fwhm,ini_fwhm)
            new_g=new_g/np.max(new_g)
            if t==0:
                fixed_bgas_templates=new_g
            else:
                fixed_bgas_templates=np.column_stack([fixed_bgas_templates,new_g]) #Templates with the fixed FWHM
        #get the relative flux from table
        my_bel_rel_flux=[]
        for ln in bline_names:
            q=bel_df[bel_df['line_names']==ln]
            fct=q['relative_flux'].values[0]
            my_bel_rel_flux.append(fct)
        #product between the templates and relative flux
        new_bgas_lines=fixed_bgas_templates*my_bel_rel_flux

################ For AGN as a POWER LAW ###############
        my_pl=util.pl(np.exp(loglam2),my_slope,5000)

    #Got the power law factor to fixe the luminosity at 3000A or 5100A
        my_waves=[3000,5000]
        if my_z>0.5:
            L=util.luminosity(my_pl,np.exp(loglam2),my_z,wv=my_waves[0])
        else:
            L=util.luminosity(my_pl,np.exp(loglam2),my_z,wv=my_waves[1])
        my_L_fc=fixed_L/L #my_pl factor

########################################################
        #Fixe the BELs luminosity following the relations in utils functions
        if my_z>0.5:
            my_L_line='MgIId'
            Ls=util.luminosity(my_L_fc*my_pl,np.exp(loglam2),my_z,wv=my_waves[0])
            my_l_ex_L=util.L_Mgii(Ls)
        else:
            my_L_line='Halpha'
            Ls=util.luminosity(my_L_fc*my_pl,np.exp(loglam2),my_z,wv=my_waves[1])
            my_l_ex_L=util.L_halpha(Ls)
            
        idx_bl=np.where(bline_names==my_L_line)
        my_bl_gas=new_bgas_lines[:,idx_bl]
        my_bl_gas=my_bl_gas.flatten()
        
        my_l_lum=util.luminosity_el(my_bl_gas,np.exp(loglam2),my_z)
        
        my_fc_blines=my_l_ex_L/my_l_lum
        bels_fc1=my_fc_blines*np.ones(len(bline_names))
########################################################

        #My NELs factor relative to BELs
        #Values here are recommended but can be changed for special needs like testing BPT diagrams 
        if my_z<0.8:
            my_fc_nels=8.5*my_fc_blines
        elif my_z>=0.8 and my_z<1.2:
            my_fc_nels=3.5*my_fc_blines
        elif my_z>=1.2:
            my_fc_nels=1.1*my_fc_blines
        nels_fc1=my_fc_nels*np.ones(len(line_names))
########################################################
        #stars factor
        #fix stars templates contribution
        #Notice that if you want to use more or less than three templates you nedd to modify this array ssp_fc1
        ssp_fc1=np.array([0.5,0.3,0.02]) 
        ssp_all=ssp@ssp_fc1
        if my_z<0.8:
            idxs=util.fnearest(np.exp(loglam2),5100)
            #to have f_pl/F_cont=0.9 and f_stars/F_cont=0.1 with F_cont=f_pl+f_stars at 5100A, considering the EMILES templates flux scale
            fpl=my_pl[idxs]/(9*ssp_all[idxs]) 
        else:
            fpl=1
        #ratio factor
        ccs=1 #Used to change arbitrary the fraction
        if agn_st_r==1:
            my_ratio_fc=1*ccs # AGN domination f_pl/F_cont=0.9 and f_stars/F_cont=0.1 with F_cont=f_pl+f_stars
            correction_fc=1
        elif agn_st_r==2:
            my_ratio_fc=ccs*9/4 # AGN domination f_pl/F_cont=0.8 and f_stars/F_cont=0.2 with F_cont=f_pl+f_stars
            correction_fc=0.8
        elif agn_st_r==3:
            my_ratio_fc=ccs*9 # No domination f_pl/F_cont=0.5 and f_stars/F_cont=0.5 with F_cont=f_pl+f_stars
            correction_fc=0.5
        elif agn_st_r==4:
            my_ratio_fc=9*4*ccs # stellar domination f_pl/F_cont=0.2 and f_stars/F_cont=0.8 with F_cont=f_pl+f_stars
            correction_fc=0.2
        elif agn_st_r==5:
            my_ratio_fc=9*9*ccs # stellar domination f_pl/F_cont=0.1 and f_stars/F_cont=0.9 with F_cont=f_pl+f_stars
            correction_fc=0.1
        else:
            my_ratio_fc=1*ccs # AGN domination f_pl/F_cont=0.9 and f_stars/F_cont=0.1 with F_cont=f_pl+f_stars
            correction_fc=1
########################################################
        #Multipliying templates by the factors
        stars=ssp_all*fpl*my_L_fc*my_ratio_fc*correction_fc
        final_pl=my_pl*my_L_fc*correction_fc
        nels=new_gas_lines*nels_fc1[0]
        bels=new_bgas_lines*bels_fc1[0]
##################### FeII ###################################
        fe_temps=util.feii_temps(len(loglam2),[np.exp(loglam2)[0],np.exp(loglam2)[-1]],velscale)
        fe_temps=fe_temps.flatten()

        # factor estimation for Fe II relative to FWHM of BELs
        RFeII=util.estimate_RfeII(FWHM_Hb_kms=my_bel_fwhm)
        #Factor relative to flux of BELs
        
        if 'Hbeta' in bline_names:
            my_fe_wv=4750
            idxfe=util.fnearest(np.exp(loglam2),my_fe_wv)
            fe_4750=fe_temps[idxfe]*my_fe_wv
            idx_blb=np.where(bline_names=='Hbeta')
            my_bl_gashb=bels[:,idx_blb]
            my_bl_gashb=my_bl_gashb.flatten()
            flx_hb=np.trapezoid(my_bl_gashb,x=np.exp(loglam2))
            
            fe_fc1=flx_hb*RFeII/fe_4750
            fe_fcf=fe_fc1*bels_fc1[0]
        else:
            if 'MgIId' in bline_names:
                my_fe_wv=2500
                idxfe=util.fnearest(np.exp(loglam2),my_fe_wv)
                fe_4750=fe_temps[idxfe]*my_fe_wv
                idx_blb=np.where(bline_names=='MgIId')
                my_bl_gasmg=bels[:,idx_blb]
                my_bl_gasmg=my_bl_gasmg.flatten()
                flx_mg=np.trapezoid(my_bl_gasmg,x=np.exp(loglam2))
                
                qhb=bel_df[bel_df['line_names']=='Hbeta']
                fcthb=qhb['relative_flux'].values[0]
                qmg=bel_df[bel_df['line_names']=='MgIId']
                fctmg=qmg['relative_flux'].values[0]
        
                rhbmg=fcthb/fctmg
                flx_hb=rhbmg*flx_mg
        
                fe_fc1=flx_hb*RFeII/fe_4750
                fe_fcf=fe_fc1*bels_fc1[0]
            else:
                fe_fcf=0.0
        #Get the final Fe II flux using flux factor
        fe_final=fe_temps*fe_fcf


############################## Shift C IV --- add winds #############
        if 'CIV1548' in np.array(bline_names):
            # Add blue-shift to the Civ1550
            idx_civa=np.where(bline_names=='CIV1548')
            idx_civb=np.where(bline_names=='CIV1550')
            my_civa=bels[:,idx_civa]
            my_civa=my_civa.flatten()
            my_wva=bline_wave[idx_civa]
            my_civb=bels[:,idx_civb]
            my_civb=my_civa.flatten()
            my_wvb=bline_wave[idx_civb]
            my_civ_fwhm=10000 #New FWHM
            my_civa_g=util.my_fwhm_l(np.exp(loglam2),my_wva,my_civa,my_civ_fwhm,ini_fwhm)
            my_civb_g=util.my_fwhm_l(np.exp(loglam2),my_wvb,my_civb,my_civ_fwhm,ini_fwhm)
            #my_civa_g=my_civa_g/np.max(my_civa_g)
            #my_civb_g=my_civb_g/np.max(my_civb_g)
            #Blueshift
            pixels_civ=15 #this number is relative to the velscale 
            vel_wind_civ=-velscale*pixels_civ #(for sdss velscale ~ -1037.1772)
            pixels_shift_civ=np.zeros(pixels_civ)
            new_civa=np.append(my_civa_g[pixels_civ:],pixels_shift_civ)
            new_civb=np.append(my_civb_g[pixels_civ:],pixels_shift_civ)
            #Replace in the templates
            bels=np.array(bels)
            bels[:,idx_civa]=new_civa[:,None,None]
            bels[:,idx_civb]=new_civb[:,None,None]

############################# save fluxes ###########################

        nels_flux=[]
        for nn in range(len(line_names)):
            flx_nel=np.trapezoid(nels[:,nn],x=np.exp(loglam2))
            nels_flux.append(flx_nel)
        my_nels_flx=pd.DataFrame()
        my_nels_flx['line_name']=line_names
        my_nels_flx['line_wave']=line_wave
        my_nels_flx['flux']=nels_flux
        if owinds_fc>0.0:
            my_nels_flx.loc[len(my_nels_flx)]=['winds[OIII]',5007.00,owinds_fc]
        #Save flux NELs after all templates
        name_nels_flx=my_dir+my_spec_name+'-nels_synt_flux_z'+str(my_z)+'_v2.csv'
        my_nels_flx.to_csv(name_nels_flx,index=False)
                
        bels_flux=[]
        for nn in range(len(bline_names)):
            flx_bel=np.trapezoid(bels[:,nn],x=np.exp(loglam2))
            bels_flux.append(flx_bel)
        my_bels_flx=pd.DataFrame()
        my_bels_flx['line_name']=bline_names
        my_bels_flx['line_wave']=bline_wave
        my_bels_flx['flux']=bels_flux

        #C IV fluxes
        if 'CIV1548' in np.array(bline_names):
            new_civa_flx=np.trapezoid(new_civa,x=np.exp(loglam2))
            new_civb_flx=np.trapezoid(new_civb,x=np.exp(loglam2))
            my_bels_flx.loc[my_bels_flx["line_name"] == "CIV1548", "flux"] =new_civa_flx
            my_bels_flx.loc[my_bels_flx["line_name"] == "CIV1550", "flux"] =new_civb_flx

        
        name_bels_flx=my_dir+my_spec_name+'-bels_synt_flux_z'+str(my_z)+'.csv'
        my_bels_flx.to_csv(name_bels_flx,index=False)


######################## Convolution of spectrum components #################
        # Convolution with instrumental resolution
        # Instrumental FWHM in km/s
        fwhm_inst = 150.0  # e.g., SDSS ~150 km/s
        
        #Get all lines in one template
        nels_1d=nels@np.ones(len(line_names))
        
        bels_1d=bels@np.ones(len(bline_names))
        # Convolve each component
        pl_conv  = util.convolve_loglam(final_pl, loglam2, fwhm_inst)
        bel_conv = util.convolve_loglam(bels_1d, loglam2, fwhm_inst)
        nel_conv = util.convolve_loglam(nels_1d, loglam2, fwhm_inst)

    
   ############## spectrum construction by adding components
# Sum the convolved components to get final synthetic spectrum
        if agn_type==2:
            if my_z<=0.8:
                synthetic_spec=stars+my_nels_fct*nel_conv+my_bels_fct*bel_conv+final_pl+fe_final
            else:
                synthetic_spec=nel_conv+bel_conv+final_pl+fe_final
        elif agn_type==1:
            if my_z<=0.8:
                synthetic_spec=stars+nel_conv+final_pl
            else:
                synthetic_spec=nel_conv+final_pl

############################# [O III] winds ################################
        if owinds_fc>0.0 and my_z<=1.08:
            #[OIII]5007 winds
            idx_o3=np.where(line_names=='[OIII]5007d')
            my_o3_gas=gas_templates[:,idx_o3]
            my_o3_gas=my_o3_gas.flatten()
            o3_wv=line_wave[idx_o3]
            wind_o3_fwhm=1400 #1000
            wind_o3=util.my_fwhm_l(np.exp(loglam2),o3_wv,my_o3_gas,wind_o3_fwhm,ini_fwhm)
            wind_o3=wind_o3/np.max(wind_o3)
            #Blueshift
            #Zhang, Kai, The Blueshifting and Baldwin Effects for the [O III] λ5007 Emission Line in Type 1 Active Galactic Nuclei,2011
            pixels_03=5 #this number is relative to the velscale
            #v_wind=-225 +/- 72 km/s
            vel_wind_o3=-velscale*pixels_03
            pixels_shift=np.zeros(pixels_03)
            new_wind_o3=np.append(wind_o3[pixels_03:],pixels_shift)
            new_wind_o3=new_wind_o3*nels_fc1[0]

            #Got the flux
            flux_wind_ini=np.trapezoid(new_wind_o3,x=np.exp(loglam2)) #for fluxes file
            ##Zhang, Kai, The Blueshifting and Baldwin Effects for the [O III] λ5007 Emission Line in Type 1 Active Galactic Nuclei, 2011 
            ## Core flux is 54% of total flux with std = 0.16 dex
            # We tested the ratios flux_wind/flux_core=1.0;0.5;0.2
            idxo3=np.where(line_names=='[OIII]5007d')
            flux_c=nels_flux[idxo3[0][0]]
            #fc_w0=(1.0*flux_c)/(flux_wind_ini)
            #fc_w1=(0.5*flux_c)/(flux_wind_ini)
            #fc_w2=(0.2*flux_c)/(flux_wind_ini)
            fc_wo3=(owinds_fc*flux_c)/(flux_wind_ini)
            final_wind_fc_flx=fc_wo3*new_wind_o3
            flux_wind_wo3=flux_wind_ini*fc_wo3 #for fluxes file
            wind_o3_conv = util.convolve_loglam(final_wind_fc_flx, loglam2, fwhm_inst)
            synthetic_spec=synthetic_spec+wind_o3_conv
######################## Add a second BELs component to generate asymetries in BELs ##################

        if two_bels is True:
            #Convolve the templates for a fixed fwhm
            # BELs fwhm after convolution
            #my_bel_fwhm2=10000  #km/s fc_bels2
            for t in range(bgas_templates.shape[1]):
                gauss=bgas_templates[:,t]
                cwv=bline_wave[t]
                new_g=util.my_fwhm_l(np.exp(loglam2),cwv,gauss,my_bel_fwhm2,ini_fwhm)
                new_g=new_g/np.max(new_g)
                if t==0:
                    fixed_bgas_templates2=new_g
                else:
                    fixed_bgas_templates2=np.column_stack([fixed_bgas_templates2,new_g]) #Templates with the fixed FWHM
            #new templates
            my_bel_rel_flux2=[]
            for ln in bline_names:
                q=bel_df[bel_df['line_names']==ln]
                fct=q['relative_flux'].values[0]
                my_bel_rel_flux2.append(fct)

            new_bgas_lines2=fixed_bgas_templates2*my_bel_rel_flux2
            bels2=new_bgas_lines2*bels_fc1[0]*fc_bels2
            bels_1d2=bels2@np.ones(len(bline_names))
            bel_conv2 = util.convolve_loglam(bels_1d2, loglam2, fwhm_inst)
            #Blueshift
            pixels_2bels=abs(int(vel_2bels/velscale))
            pixels_shift_2bels=np.zeros(pixels_2bels)
            if vel_2bels<0.:
                new_bels2=np.append(bel_conv2[pixels_2bels:],pixels_shift_2bels)
            else:
                new_bels2=np.append(pixels_shift_2bels,bel_conv2[:pixels_2bels])


            #Save flux
            bels_flux2=[]
            for nn in range(len(bline_names)):
                flx_bel=np.trapezoid(bels2[:,nn],x=np.exp(loglam2))
                bels_flux2.append(flx_bel)
            my_bels_flx2=pd.DataFrame()
            my_bels_flx2['line_name']=bline_names
            my_bels_flx2['line_wave']=bline_wave
            my_bels_flx2['flux']=bels_flux2

            name_bels_flx2='./'+my_spec_name+'-bels_c2_synt_flux_z'+str(my_z)+'.csv'
            my_bels_flx2.to_csv(name_bels_flx2,index=False)                

            synthetic_spec=synthetic_spec+bel_conv2
########################################
        final_spec=synthetic_spec*10**17 ##### The final spec which is to going be saved
        # If no noise
        my_ivar=100*np.ones(len(final_spec))
        my_sn_median_all = 10000
############################# Adding Noise to the spectrum ################
        if sn_all>0. and sn_all<3.5:
            my_syn_sn=2
            print('Using SN_MEDIAN_ALL = 2')
            my_sdss_spec=glob.glob('./spectra_sdss/spec*sn2.fits')
        elif sn_all>=3.5 and sn_all<7.5:
            my_syn_sn=5
            print('Using SN_MEDIAN_ALL = 5')
            my_sdss_spec=glob.glob('./spectra_sdss/spec*sn5.fits')
        elif sn_all>=7.5 and sn_all<15:
            my_syn_sn=10
            print('Using SN_MEDIAN_ALL = 10')
            my_sdss_spec=glob.glob('./spectra_sdss/spec*sn10.fits')
        elif sn_all>=15 and sn_all<25:
            my_syn_sn=20
            print('Using SN_MEDIAN_ALL = 20')
            my_sdss_spec=glob.glob('./spectra_sdss/spec*sn20.fits')
        elif sn_all>=25:
            my_syn_sn=30
            print('Using SN_MEDIAN_ALL = 30')
            my_sdss_spec=glob.glob('./spectra_sdss/spec*sn30.fits')
        else:
            my_syn_sn='inf'
            print('Spectrum without NOISE')

        if sn_all>0:
            
            hs=fits.open(my_sdss_spec[0])
            ds=hs[1].data
            ssflux=ds['flux']
            ssivar=ds['ivar']
            ssivar[0:10]=ssivar[10:20]
            sswave=ds['loglam']
            my_wdisp = ds['wdisp']
            len_dif=len(ssflux)-len(final_spec)
            #for length differences
            my_ivar=(ssflux[:-len_dif]/(final_spec))**2*ssivar[:-len_dif]
            my_wdisp = my_wdisp[:-len_dif]
            my_noise=[]
            for t in my_ivar:
                #ff_nn=np.random.choice(from_rand,size=1)
                #nn=np.random.normal(loc=0.0,scale=t*ff_nn,size=1)
                if t>0:
                    tv=np.sqrt(1/t)
                else:
                    tv=np.median(1/my_ivar[my_ivar>0])
                nn=np.random.normal(loc=0.0,scale=tv,size=1)
                
                my_noise.append(nn[0])

            my_sn_per_pixel = (final_spec+np.array(my_noise))*np.sqrt(my_ivar)
            my_sn_median_all = np.median(my_sn_per_pixel[my_ivar > 0])

            final_spec=final_spec++np.array(my_noise)

##############Save the spectra###########################
        #vals for fit archive
        nm='synthetic_AGN'
        BUNIT = '1E-17 erg/cm^2/s/Ang' 
        #Create the Primary and header
        hdu=fits.PrimaryHDU()
        hdr=hdu.header
        hdr['ID_name']=(nm,'synthetic') #name in fits
        hdr['AGN_type']=(agn_type,'AGN type')
        #Other values
        hdr['Z']=(my_z,'fixed redshift')
        hdr['BUNIT']=(BUNIT, 'units')
        hdr['software']=('CORVICHE','software')
        #Create data extension
        col1s = fits.Column(name='flux', array=final_spec, format='E')
        loglam3=np.log10(np.exp(loglam2)*(1+my_z))
        col2s = fits.Column(name='loglam', array=loglam3, format='E')
        col3s = fits.Column(name='ivar', array=my_ivar, format='E')
        col4s = fits.Column(name='wdisp', array=my_wdisp, format='E')
        colss = fits.ColDefs([col1s, col2s, col3s,col4s])
        bintable_hdu_1 = fits.BinTableHDU.from_columns(colss)
        bintable_hdu_1.header['EXTNAME'] = 'SPEC'  # Give it a name to the exte
        # synthetic data
        # Define the columns
        col1 = fits.Column(name='Z', array=np.array([my_z]), format='E')
        col2 = fits.Column(name='SN_MEDIAN_ALL', array=np.array([my_sn_median_all]), format='E')
        col3 = fits.Column(name='pl_slope', array=np.array([my_slope]), format='E')
        col4 = fits.Column(name='LogLum_cont', array=np.array([np.log10(1.0*fixed_L)]), format='E')
        col5 = fits.Column(name='FWHM_BELs', array=np.array([my_bel_fwhm]), format='E')
        col6 = fits.Column(name='BELs_fct', array=np.array([my_bels_fct]), format='E')
        col7 = fits.Column(name='O3_wind_fct', array=np.array([owinds_fc]), format='E')
        if two_bels==False:
            fc_bels2=0.
            
        col8 = fits.Column(name='BELs2_fct', array=np.array([fc_bels2]), format='E')
        ttt=Time.now()
        mjd= np.round(ttt.mjd,1)
        col9 = fits.Column(name='MJD', array=np.array([mjd]), format='E')
        cols = fits.ColDefs([col1, col2, col3,col4,col5,col6,col7,col8,col9])
        bintable_hdu = fits.BinTableHDU.from_columns(cols)
        bintable_hdu.header['EXTNAME'] = 'PARAM'  # Give it a name tothe exte

        hdutt=fits.HDUList([hdu,bintable_hdu_1,bintable_hdu])
        name_spec_sy = 'spec-'+my_spec_name+'-type'+str(agn_type)+'-z'+str(my_z)+'-sn'+str(my_syn_sn)+'.fits'
        dir_spec=my_dir+name_spec_sy
        hdutt.writeto(dir_spec, overwrite=True)
        print('spectrum saved in '+dir_spec)