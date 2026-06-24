import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.stats as stats
import h5py


#PYTHON MODULES
from spectral_sirens.utils.constants import *
from spectral_sirens.cosmology import gwcosmo
from spectral_sirens.gw_population import gwpop
from spectral_sirens.utils import gwutils
from spectral_sirens.utils import utils
from spectral_sirens.detectors import sensitivity_curves as sc

#Fiducial universe
from fiducial_universe_xg import *

#Detector configuration
fmin = 10.
detector = 'A+' 
detectorSn, fmin_detect, fmax_detect = sc.detector_psd(detector)
based = 'ground'
snr_th = 8.

#Output directory
dir_out = '../data_injections/'

#Injection parameters
#- Uniform comoving volume
#- Power law for detector primary mass 
#- Uniforma mass ratio
params = 'm1z_m2z_dL'
zmin_inj, zmax_inj = 1e-3, 10 #15
mmin_inj, mmax_inj = 1e-3, 100.
alpha_inj, mzmin_inj, mzmax_inj = -0.3, mmin_inj, mmax_inj*(1+zmax_inj)
inj_details = 'Vz_zmax_%s_m1z_power_law_alpha_%s_mmin_%s_mmax_%s' % (zmax_inj,alpha_inj,mmin_inj,mmax_inj)

#Number of injections
n_detections = int(1e3) 
n_sources = n_detections*15

##############
#Injections
##############

##Defining the injected distribution CDFs
#----
#Detector frame primary mass
m1zs = np.linspace(mzmin_inj,mzmax_inj,10000)
cdf_m1z = cumtrapz(utils.powerlaw(m1zs, mzmin_inj,mzmax_inj,alpha_inj),m1zs,initial=0.0)
cdf_m1z = cdf_m1z / cdf_m1z[-1]
#Redshift
zs = np.linspace(zmin_inj,zmax_inj,10000)
cdf_z = cumtrapz(gwcosmo.diff_comoving_volume_approx(zs,H0_fid,Om0_fid)/(1+zs),zs,initial=0.0)
norm_z = cdf_z[-1]
cdf_z = cdf_z / norm_z
#Geometric factor from orientations
ww = np.linspace(0.0,1.0,1000)
cdf_ww = 1.0-sc.pw_hl(ww)

##Computing injected events
#----
#Detector frame primary mass
m1z_mock_pop = utils.inverse_transf_sampling(cdf_m1z,m1zs,n_sources)
m2z_mock_pop = np.random.uniform(mzmin_inj,m1z_mock_pop,n_sources)
#Luminosity distance
z_mock_pop = utils.inverse_transf_sampling(cdf_z,zs,n_sources)
dL_mock_pop = gwcosmo.dL_approx(z_mock_pop,H0_fid,Om0_fid) #Mpc

#Optimal SNR
snr_opt_mock_pop = gwutils.vsnr_from_psd(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs_fid,detectorSn, fmin_detect, fmax_detect,based)
#True SNR
w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
#Observed SNR
snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)

##Computing p_draw
p_draw_m1z = utils.powerlaw(m1z_mock_pop,mzmin_inj,mzmax_inj,alpha_inj)
p_draw_m2z = 1./(m1z_mock_pop-mzmin_inj)
p_draw_z = gwcosmo.diff_comoving_volume_approx(z_mock_pop,H0_fid,Om0_fid)/(1+z_mock_pop)/norm_z
Ez_i = gwcosmo.Ez_inv(z_mock_pop,H0_fid,Om0_fid)
D_H = (Clight/1.0e3)  / H0_fid #Mpc 
jac_logdLz = dL_mock_pop/(1.+z_mock_pop) + (1. + z_mock_pop)*D_H * Ez_i #Mpc
p_draw_mock_pop = p_draw_m1z * p_draw_m2z * p_draw_z / jac_logdLz

#Detected injections
m1z_inj = m1z_mock_pop[snr_obs_mock_pop>snr_th]
m2z_inj = m2z_mock_pop[snr_obs_mock_pop>snr_th]
dL_inj = dL_mock_pop[snr_obs_mock_pop>snr_th]
p_draw_inj = p_draw_mock_pop[snr_obs_mock_pop>snr_th]

Ndet = np.size(m1z_inj)
Ndraws = n_sources
print('Ndet = ',Ndet,', Ndraw = ',Ndraws)

# Saving the data
variables = ['m1z_inj','m2z_inj','dL_inj','p_draw_inj']
with h5py.File(dir_out+'injections_'+detector+'_'+params+'_'+inj_details+'_Ndraws_%s_Ndet_%s.hdf5' % (Ndraws,Ndet), "w") as f:
    for var in variables:
        dset = f.create_dataset(var, data=eval(var))