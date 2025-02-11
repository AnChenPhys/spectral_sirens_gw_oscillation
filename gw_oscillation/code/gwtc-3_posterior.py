# Test with GWTC-3 data

import numpy as np

#Numpyro and friends
import numpyro
from numpyro.infer import NUTS,MCMC
import numpyro.distributions as dist
import jax
from jax import random
# from jax.config import config
import jax.numpy as jnp
import arviz as az
import h5py
import json
import corner
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from lal import C_SI

Clight = C_SI #2.997e8
# config.update("jax_enable_x64", True)

#JAX MODULES
from spectral_sirens.utils.constants import *
from spectral_sirens.cosmology import gwcosmo
from spectral_sirens.gw_population import gwpop
from spectral_sirens.bayesian_inference import likelihood as lik
from spectral_sirens.utils import utils

detector = 'O3'

import os 
PATH = os.getcwd()
dir_plots='../plots/'
dir_samples = 'samples/samples_'+detector+'/'
dir_inj = '../data_injections/'
# dir_mock = 'mock_catalogues/mock_catalogues_'+detector+'/'

#Injections
params = 'm1z_m2z_dL'

zmax = 10
zs = np.linspace(1e-3,zmax,10000)
alpha_inj, mmin_inj, mmax_inj = -2, 4.98, 112.5
sig_inj, mu_inj, f_peak_inj, deltaM_inj = 3.88, 32.27, 0.03, 4.8
alpha_z_fid, zp_fid, beta_fid = 4.59, 2.47, 2.86
H0_fid,Om0_fid = 70, 0.3
inj_details = 'Vz_zmax_%s_m1z_power_law_alpha_%s_mmin_%s_mmax_%s' % (zmax,alpha_inj,mmin_inj,mmax_inj)
Ndet = 103697
Ndraw = 50000000
snr_th = 11

#Reading the data
data = h5py.File(dir_inj+'injections_'+detector+'_'+params+'_'+inj_details+'_Ndraws_%s_Ndet_%s.hdf5' % (Ndraw,Ndet), "r")
m1z_inj = np.array(data['m1z_inj'])
m2z_inj = np.array(data['m2z_inj'])
dL_inj = np.array(data['dL_inj'])
p_draw_inj = np.array(data['p_draw_inj'])

# print('Number of found injections: ',Ndet)
# print('Number of draws: ',Ndraw)

# inj_file = h5py.File('/home/ansonchen/spectral_sirens_gw_oscillation/gw_oscillation/data_injections/inj_SNR9_det_frame_2e6.h5', "r")
# m1z_inj = np.array(inj_file['m1d'])[inj_file['snr'][...]>snr_th]
# m2z_inj = np.array(inj_file['m2d'])[inj_file['snr'][...]>snr_th]
# dL_inj = np.array(inj_file['dl'])[inj_file['snr'][...]>snr_th]
# z_inj = gwcosmo.z_at_dl_approx(dL_inj,H0_fid,Om0_fid)

# p_draw_m1z = utils.powerlaw(m1z_inj, mmin_inj,mmax_inj*(1+zmax),alpha_inj)
# p_draw_m2z = 1./(m1z_inj-mmin_inj)
cdf_z = cumtrapz(gwcosmo.diff_comoving_volume_approx(zs,H0_fid,Om0_fid)/(1+zs),zs,initial=0.0)
norm_z = cdf_z[-1]
# p_draw_z = gwcosmo.diff_comoving_volume_approx(z_inj,H0_fid,Om0_fid)*gwpop.rate_z(z_inj,zp_fid,alpha_z_fid,beta_fid)/(1+z_inj)/norm_z
# Ez_i = gwcosmo.Ez_inv(z_inj,H0_fid,Om0_fid)
# D_H = (Clight/1.0e3)  / H0_fid #Mpc 
# jac_logdLz = dL_inj/(1.+z_inj) + (1. + z_inj)*D_H * Ez_i #Mpc
# p_draw_inj = p_draw_m1z * p_draw_m2z * p_draw_z / jac_logdLz
# p_draw_inj = np.array(inj_file['pini'])[inj_file['snr'][...]>snr_th]

# load GWTC-3 event posteriors
json_path = 'posterior_samples_dictionary_O3_BBH.json'
with open(json_path) as json_file:
    O3_events = json.load(json_file)

posteriors = {}
for event in O3_events.keys():
    posteriors[event] = h5py.File(O3_events[event])

n_samples = 20000
rand_perm = np.random.permutation(n_samples)

m1z_samples = np.zeros((len(posteriors),n_samples))
m2z_samples = np.zeros((len(posteriors),n_samples))
dL_samples = np.zeros((len(posteriors),n_samples))
pdraw_samples = np.zeros((len(posteriors),n_samples))
for i,event in enumerate(O3_events.keys()):
    m1z_samples[i,:] = posteriors[event]['C01:IMRPhenomXPHM']['posterior_samples']['mass_1'][rand_perm[:n_samples]]
    m2z_samples[i,:] = posteriors[event]['C01:IMRPhenomXPHM']['posterior_samples']['mass_2'][rand_perm[:n_samples]]
    dL_samples[i,:] = posteriors[event]['C01:IMRPhenomXPHM']['posterior_samples']['luminosity_distance'][rand_perm[:n_samples]]

    p_draw_m1z = utils.powerlaw(m1z_samples[i,:], mmin_inj,mmax_inj*(1+zmax),alpha_inj)
    p_draw_m2z = 1./(m1z_samples[i,:]-mmin_inj)
    z_samples = gwcosmo.z_at_dl_approx(dL_samples[i,:],H0_fid,Om0_fid)
    p_draw_z = gwcosmo.diff_comoving_volume_approx(z_samples,H0_fid,Om0_fid)/(1+z_samples)/norm_z
    Ez_i = gwcosmo.Ez_inv(z_samples,H0_fid,Om0_fid)
    D_H = (Clight/1.0e3)  / H0_fid #Mpc 
    jac_logdLz = dL_samples[i,:]/(1.+z_samples) + (1. + z_samples)*D_H * Ez_i #Mpc
    pdraw_samples[i,:] = p_draw_m1z * p_draw_m2z * p_draw_z / jac_logdLz
    # pdraw_samples[i,:] = dL_samples[i,:]**2 

#MCMC
nChains = 1
numpyro.set_host_device_count(nChains)
num_warmup = 500 #CHANGE THIS TO ADJUST YOUR NEEDS
num_samples = 1000 #CHANGE THIS TO ADJUST YOUR NEEDS

#Priors
#---------------------
#Cosmo priors
h0_min, h0_max = 0.4, 1.
Om0_min, Om0_max = 0.15, 0.45
#p(m1) priors
alpha_min, alpha_max = -5., 0.
mmin_min, mmin_max = 1., 20.
mmax_min, mmax_max = 30., 150.
mu_m1_min, mu_m1_max = 20., 60.
sig_m1_min, sig_m1_max = 1., 10.
f_peak_min, f_peak_max = 0., 1e-6
#p(q) priors
bq_min, bq_max = 0., 1.
#p(z) priors
alpha_z_min, alpha_z_max = 1., 5.
zp_min, zp_max = 0., 4.
beta_min, beta_max = 0., 10.
#---------------------
H0_fid = 70
Om0_fid = 0.3
alpha_z_fid = 4.59
zp_fid = 2.47
beta_fid = 2.86
mmin_pl_fid = 3
mmax_pl_fid = 150
dmMin_filter_fid = 0.5
dmMax_filter_fid = 2.5

def log_probability():
    
    #Prior distributions
    h0 = numpyro.sample("h0",dist.Uniform(h0_min,h0_max))
    #Om0 = numpyro.sample("Om0",dist.Uniform(Om0_min,Om0_max))
    mmin = numpyro.sample("mmin",dist.Uniform(mmin_min,mmin_max))
    mmax = numpyro.sample("mmax",dist.Uniform(mmax_min,mmax_max))
    alpha = numpyro.sample("alpha",dist.Uniform(alpha_min, alpha_max))
    bq =  numpyro.sample("bq",dist.Uniform(bq_min,bq_max)) #when fitting m_2
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(mu_m1_min,mu_m1_max))
    sig_m1 = numpyro.sample("sig_m1",dist.Uniform(sig_m1_min,sig_m1_max))
    f_peak = numpyro.sample("f_peak",dist.Uniform(f_peak_min,f_peak_max))
    alpha_z = numpyro.sample("alpha_z",dist.Uniform(alpha_z_min,alpha_z_max))
    #Fixed parameters
    Om0 = Om0_fid
    zp = zp_fid # numpyro.sample("zp",dist.Uniform(zp_min,zp_max))
    beta = beta_fid # numpyro.sample("beta",dist.Uniform(beta_min,beta_max))
    mmin_pl = mmin_pl_fid
    mmax_pl = mmax_pl_fid
    dmMin_filter = dmMin_filter_fid
    dmMax_filter = dmMax_filter_fid
    
    #Likelihood
    loglik, Neff = lik.log_lik(m1z_samples,m2z_samples,dL_samples,pdraw_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,h0,Om0,mmin_pl,mmax_pl,alpha,sig_m1,mu_m1,f_peak,mmin,mmax,dmMin_filter,dmMax_filter,bq,alpha_z,zp,beta)

    #Convergence check: uncomment to check number of effective samples
    #conv = numpyro.deterministic('conv', Neff/4/n_detections)

    #Likelihood
    numpyro.factor("logp",loglik)

rng_key = random.PRNGKey(2)
rng_key,rng_key_ = random.split(rng_key)

# Set up NUTS sampler over our likelihood
kernel = NUTS(log_probability)
mcmc = MCMC(kernel,num_warmup=num_warmup,num_samples=num_samples,num_chains=nChains,chain_method='parallel',progress_bar=True)

# inference_details = '_Ndet_%s_Nsamples_%s_Nfoundinj_%s_Ninj_%s' % (n_detections,n_samples,Ndet,Ndraw) +'_'+ inj_details

mcmc.run(rng_key_)
mcmc.print_summary()
samples = mcmc.get_samples()

np.save('mcmc_samples.npy', samples)

print('Fiducial values:')
print('H0=',H0_fid,', Om0=',Om0_fid)
print('alpha=',alpha_inj,', f_peak=',f_peak_inj,', mmax=',mmax_inj,', mmin=',mmin_inj,', mu_m1=',mu_inj,', sig_m1=',sig_inj)
print('alpha_z=',alpha_z_fid,', zp=',zp_fid,', beta=',beta_fid)

#Corner plot
fig = corner.corner(samples,quantiles=[0.16, 0.5, 0.84], show_titles=True) #,labels=labels,fontsize=fontsz)
plt.savefig(f'{PATH}/'+dir_plots+'gwtc3_corner.pdf',bbox_inches='tight')