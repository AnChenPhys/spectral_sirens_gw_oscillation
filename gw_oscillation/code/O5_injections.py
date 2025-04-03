# Create injections for O5

#IMPORT
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from tqdm import tqdm
import time
import scipy.stats as stats
import h5py
from lal import C_SI

Clight = C_SI #2.997e8

#PYTHON MODULES
from spectral_sirens.utils.constants import *
from spectral_sirens.cosmology import gwcosmo
from spectral_sirens.gw_population import gwpop
from spectral_sirens.utils import gwutils
from spectral_sirens.utils import utils
from spectral_sirens.detectors import sensitivity_curves as sc

#Directories
import os 
PATH = os.getcwd()
dir_out = '../data_injections/'

fmin = 10.
based = 'ground'
snr_th = 8.
H0_fid = 67.66
Om0_fid = 0.30966
Tobs_fid = 1

params = 'm1z_m2z_dL'

#zmin_inj, zmax_inj = 1e-3, 15
zmin_inj, zmax_inj = 1e-3, 20
mmin_inj, mmax_inj = 1., 200.
alpha_inj, mzmin_inj, mzmax_inj = -0.2, mmin_inj, mmax_inj*(1+zmax_inj)

# sig_inj, mu_inj, f_peak_inj, deltaM_inj = 3.88, 32.27, 0.03, 4.8
# zp_fid, alpha_z_fid, beta_fid = 2.47, 4.59, 2.86

n_detections = int(1e2)
n_sources = n_detections*50

starttime = time.time()

##Defining the injected distribution CDFs
#----
#Detector frame primary mass
m1zs = np.linspace(mzmin_inj,mzmax_inj,10000)
cdf_m1z = cumtrapz(utils.powerlaw(m1zs,mzmin_inj,mzmax_inj,alpha_inj),m1zs,initial=0.0)
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

##Computing p_draw
p_draw_m1z = utils.powerlaw(m1z_mock_pop,mzmin_inj,mzmax_inj,alpha_inj)
p_draw_m2z = 1./(m1z_mock_pop-mzmin_inj)
p_draw_z = gwcosmo.diff_comoving_volume_approx(z_mock_pop,H0_fid,Om0_fid)/(1+z_mock_pop)/norm_z
Ez_i = gwcosmo.Ez_inv(z_mock_pop,H0_fid,Om0_fid)
D_H = (Clight/1.0e3)  / H0_fid #Mpc 
jac_logdLz = dL_mock_pop/(1.+z_mock_pop) + (1. + z_mock_pop)*D_H * Ez_i #Mpc
p_draw_mock_pop = p_draw_m1z * p_draw_m2z * p_draw_z / jac_logdLz

obs = 'O5'
detector = 'A+'
detectorSn, fmin_detect, fmax_detect = sc.detector_psd(detector)

# detectors = ['O3-H1', 'O3-L1', 'O3-V1']
# snr_tot_sq_list = {'O3-H1':[], 'O3-L1':[], 'O3-V1':[]}

# for det in detectors:
#     detectorSn, fmin_detect, fmax_detect = sc.detector_psd(det)

#     #Optimal SNR
#     snr_opt_mock_pop = gwutils.vsnr_from_psd(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs_fid,detectorSn, fmin_detect, fmax_detect,based)
#     #True SNR
#     w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
#     snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
#     #Observed SNR
#     snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)
#     snr_tot_sq_list[det] = (snr_obs_mock_pop*snr_obs_mock_pop)

# snr_tot_obs_mock_pop = np.sqrt(snr_tot_sq_list['O3-H1']+snr_tot_sq_list['O3-L1']+snr_tot_sq_list['O3-V1'])

# np.savetxt('SNR_injections.txt', [snr_tot_obs_mock_pop, np.sqrt(snr_tot_sq_list['O3-H1']), np.sqrt(snr_tot_sq_list['O3-L1']), np.sqrt(snr_tot_sq_list['O3-V1'])], header='tot\tH1\tL1\tV1')

#Optimal SNR
snr_opt_mock_pop = gwutils.vsnr_from_psd(m1z_mock_pop,m2z_mock_pop,dL_mock_pop,fmin,Tobs_fid,detectorSn, fmin_detect, fmax_detect,based)
#True SNR
w_mock_pop = utils.inverse_transf_sampling(cdf_ww,ww,n_sources) #random draw
snr_true_mock_pop = snr_opt_mock_pop*w_mock_pop
#Observed SNR
snr_obs_mock_pop = gwutils.observed_snr(snr_true_mock_pop)

#Detected injections
m1z_inj = m1z_mock_pop[snr_obs_mock_pop>snr_th]
m2z_inj = m2z_mock_pop[snr_obs_mock_pop>snr_th]
dL_inj = dL_mock_pop[snr_obs_mock_pop>snr_th]
p_draw_inj = p_draw_mock_pop[snr_obs_mock_pop>snr_th]

Ndet = np.size(m1z_inj)
Ndraws = n_sources
print('Ndet = ',Ndet,', Ndraw = ',Ndraws)

print('Time taken = {} seconds'.format(time.time() - starttime))

inj_details = 'Vz_zmax_%s_m1z_power_law_alpha_%s_mmin_%s_mmax_%s' % (zmax_inj,alpha_inj,mmin_inj,mmax_inj)

# Saving the data
variables = ['m1z_inj','m2z_inj','dL_inj','p_draw_inj']
with h5py.File(dir_out+'injections_'+obs+'_'+params+'_'+inj_details+'_Ndraws_%s_Ndet_%s.hdf5' % (Ndraws,Ndet), "w") as f:
    for var in variables:
        dset = f.create_dataset(var, data=eval(var))
