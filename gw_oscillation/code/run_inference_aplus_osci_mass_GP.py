import numpy as np

#Numpyro and friends
import numpyro
from numpyro.infer import NUTS,MCMC
import numpyro.distributions as dist
import jax
from jax import random
from jax import config
import jax.numpy as jnp
import arviz as az
import h5py
from tinygp import kernels, GaussianProcess

Clight = 2.997e8

config.update("jax_enable_x64", True)

#JAX MODULES
from spectral_sirens.utils.constants import *
from spectral_sirens.cosmology import jgwcosmo
from spectral_sirens.gw_population import jgwpop
from spectral_sirens.bayesian_inference import likelihood_mass_GP as lik_GP

#Detector
detector = 'A+'

#Directories
import os 
PATH = os.getcwd()
# dir_plots='plots_xg/plots_'+detector+'/'
# dir_samples = 'samples_xg/samples_'+detector+'/'
# dir_inj = 'injections_xg/injections_'+detector+'/'
# dir_mock = 'data_mock_catalogues_xg/mock_catalogue_'+detector+'/'
dir_plots='../plots/'
dir_samples = '../samples/'
dir_inj = '../data_injections/'
catalog = 'powerlaw_peak'
dir_mock = '../data_mock_catalogues/'+catalog+'_'

#Fiducial universe
from fiducial_universe_gwtc3 import *
model_name = 'powerlaw_peak_smooth'
# bq_fid = 0. #CHANGE THIS
model_name_with_params = model_name+'_alpha_%s_sig_%s_mu_%s_fpeak_%s_mmin_pl_%s_mmax_pl_%s_mmin_filt_%s_mmax_filt_%s_dmmin_%s_dmmax_%s'%(alpha_fid, sig_m1_fid, mu_m1_fid, f_peak_fid, mmin_pl_fid, mmax_pl_fid, mMin_filter_fid,mMax_filter_fid,dmMin_filter_fid,dmMax_filter_fid)
#for old model_name_with_params =
#model_name_with_params = model_name+'_alpha_%s_sig_%s_mu_%s_fpeak_%s_mmin_pl_%s_mmax_pl_%s_mmin_filt_%s_mmax_filt_%s_dmmin_%s_dmmax_%s '%(alpha_fid, sig_m1_fid, mu_m1_fid, 0.04, 5., mmax_pl_fid, mMin_filter_fid,mMax_filter_fid,dmMin_filter_fid,dmMax_filter_fid)


#Injections
params = 'm1z_m2z_dL'

zmax = 10
alpha_inj, mmin_inj, mmax_inj = -0.3, 0.001, 100.
inj_details = 'Vz_zmax_%s_m1z_power_law_alpha_%s_mmin_%s_mmax_%s' % (zmax,alpha_inj,mmin_inj,mmax_inj)
Ndet = 1004058 #335076
Ndraw = 15000000

#Reading the data
data = h5py.File(dir_inj+'injections_'+detector+'_'+params+'_'+inj_details+'_Ndraws_%s_Ndet_%s.hdf5' % (Ndraw,Ndet), "r")
m1z_inj = np.array(data['m1z_inj'])
m2z_inj = np.array(data['m2z_inj'])
dL_inj = np.array(data['dL_inj'])
p_draw_inj = np.array(data['p_draw_inj'])

print('Number of found injections: ',Ndet)
print('Number of draws: ',Ndraw)

#Mock data
n_samples = 200
n_detections = 1000
print('Number of detections: ',n_detections)
print('Number of samples: ',n_samples)

m1z_mock_samples = np.load(dir_mock+'oscillate_m1z_'+detector+'_Ndet_%s_Nsamples_%s_' % (n_detections,n_samples)+model_name_with_params+'.npy')
m2z_mock_samples = np.load(dir_mock+'oscillate_m2z_'+detector+'_Ndet_%s_Nsamples_%s_' % (n_detections,n_samples)+model_name_with_params+'.npy')
dL_mock_samples = np.load(dir_mock+'oscillate_dL_'+detector+'_Ndet_%s_Nsamples_%s_' % (n_detections,n_samples)+model_name_with_params+'.npy')
pdraw_mock_samples= np.load(dir_mock+'oscillate_pdraw_'+detector+'_Ndet_%s_Nsamples_%s_' % (n_detections,n_samples)+model_name_with_params+'.npy')

#MCMC
nChains = 1
numpyro.set_host_device_count(nChains)
num_warmup = 500
num_samples = 1000 #CHANGE THIS

#Priors
h0_min, h0_max = 0.2, 1.2
Om0_min, Om0_max = 0.1, 0.6
alpha_min, alpha_max = 0., 25.
kappa1_min, kappa1_max = -4., 12.
kappa2_min, kappa2_max = -4., 12.
mmin_min, mmin_max = 1., 20.
mmax_min, mmax_max = 30., 100.
b_min, b_max = 0., 1.
beta_min, beta_max = 0., 10.
zp_min, zp_max = 0., 4.
gamma_min, gamma_max = -10, 10
    
#Priors
h0 = H0_fid/100 #numpyro.sample("h0",dist.Uniform(h0_min,h0_max))
Om0 = Om0_fid #numpyro.sample("Om0",dist.Uniform(Om0_min,Om0_max))
mmin = mMin_filter_fid #numpyro.sample("mmin",dist.Uniform(mmin_min,mmin_max))
mmax = mMax_filter_fid #numpyro.sample("mmax",dist.Uniform(mmax_min,mmax_max))
alpha = alpha_fid #numpyro.sample("alpha",dist.Uniform(-5., 0.)) #numpyro.sample("alpha",dist.Normal(0,5))
mu_m1 = mu_m1_fid #numpyro.sample("mu_m1",dist.Uniform(20,60))
sig_m1 = sig_m1_fid #numpyro.sample("sig_m1",dist.Uniform(1,10))
f_peak = f_peak_fid #numpyro.sample("f_peak",dist.Uniform(0,1e-6))
bq = bq_fid #numpyro.sample("bq",dist.Normal(0,5)) #when fitting m_2
#Fixed
mmin_pl = mmin_pl_fid
mmax_pl = mmax_pl_fid
dmMin_filter = dmMin_filter_fid
dmMax_filter = dmMax_filter_fid
Tobs = 0.73

def get_ell_frechet_params(data,dims=1.,alpha=0.05, return_L=False):
    concentration = dims/2.
    # we choose the lower bound for the length scale to be the characteristic
    # distance between datapoints. we could also choose it to be the minimum
    # distance between datapoints, but this choice makes inference faster
    L = jnp.mean(jnp.diff(jnp.sort(data)))
    lam = -jnp.log(alpha) * (L**concentration)
    scale = lam**(2./dims)
    if return_L:
        return scale, concentration, L
    else:
        return scale, concentration

def get_sigma_gamma_params(U,alpha=0.05):
    k=1.
    lam = - jnp.log(alpha)/U # I think the blog post is missing this minus sign
    #theta = 1./lam
    rate = lam
    return k, rate

scale, concentration, L = get_ell_frechet_params(np.log(m1z_mock_samples).mean(axis=1),return_L=True)
conc, lam_sigma = get_sigma_gamma_params(U=2.)
print(scale, concentration, conc, lam_sigma)

rng_key = random.PRNGKey(2)
rng_key,rng_key_ = random.split(rng_key)

kwargs = dict(m1z_mock_samples=m1z_mock_samples,m2z_mock_samples=m2z_mock_samples,dL_mock_samples=dL_mock_samples,pdraw_mock_samples=pdraw_mock_samples,m1z_inj=m1z_inj,m2z_inj=m2z_inj,
              dL_inj=dL_inj,p_draw_inj=p_draw_inj,Ndraw=Ndraw,Tobs=Tobs) #,PC_params=dict(conc=conc,concentration=concentration,scale=scale,lam_sigma=lam_sigma))

# Set up NUTS sampler over our likelihood
kernel = NUTS(lik_GP.log_lik)
mcmc = MCMC(kernel,num_warmup=num_warmup,num_samples=num_samples,num_chains=nChains,chain_method='parallel',progress_bar=True)

mcmc.run(rng_key_, **kwargs)
mcmc.print_summary()
samples = mcmc.get_samples()

idata = az.from_numpyro(mcmc)
idata.to_netcdf("mcmc_samples_mass_GP.nc")

print('Fiducial values:')
print('H0=',H0_fid,', Om0=',Om0_fid)
print('alpha=',alpha_fid,', bq=',bq_fid,', f_peak=',f_peak_fid,', mmax=',mMax_filter_fid,', mmin=',mMin_filter_fid,', mu_m1=',mu_m1_fid,', sig_m1=',sig_m1_fid)
print('alpha_z=',alpha_z_fid,', zp=',zp_fid,', beta=',beta_fid)

inference_details = '_Ndet_%s_Nsamples_%s_Nfoundinj_%s_Ninj_%s' % (n_detections,n_samples,Ndet,Ndraw) +'_'+ inj_details+'_mass_GP' #CHANGE THIS

#Save samples to hdf5
hf = h5py.File(f'{PATH}/'+dir_samples+'samples_'+model_name+inference_details+'.hdf5', 'w')
for key in samples.keys():
    hf.create_dataset(key, data=samples[key])
hf.close()
'''
#PLOTS
import matplotlib.pyplot as plt
import corner
fontSz = 15
fontsz = 13
fontssz = 11
new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

#Corner plot
fig = corner.corner(samples,quantiles=[0.16, 0.5, 0.84],#, labels=labels
                       show_titles=True,fontsize=fontsz)
plt.savefig(f'{PATH}/'+dir_plots+'oscillate_corner_'+model_name+inference_details+'.pdf',bbox_inches='tight')


az.plot_trace(mcmc, compact=True, lines=[('h0',{},H0_fid/100),
                                         ('Om0',{},Om0_fid),
                                         (r'mmin',{},mMin_filter_fid),
                                         (r'mmax',{},mMax_filter_fid),
                                         (r'alpha',{},alpha_fid),
                                         (r'mu_m1',{},mu_m1_fid),
                                         (r'sig_m1',{},sig_m1_fid),
                                         (r'f_peak',{},f_peak_fid)]
                                         )
plt.tight_layout()
plt.savefig(f'{PATH}/'+dir_plots+'oscillate_plot_trace_'+model_name+inference_details+'.pdf')

#Plot mass spectrum
fig,ax = plt.subplots(figsize=(10,5))

m1_grid = np.linspace(mmin_pl_fid,mmax_pl_fid,1000)

random_inds = np.random.choice(np.arange(samples['h0'].size),size=500)
for i in random_inds:
    
    p_m1 = jgwpop.powerlaw_peak_smooth(m1_grid,
                                       mmin_pl_fid,
                                       mmax_pl_fid,                                     
                                       samples['alpha'][i],
                                       samples['sig_m1'][i],
                                       samples['mu_m1'][i],
                                       samples['f_peak'][i],
                                       samples['mmin'][i],
                                       samples['mmax'][i],
                                       dmMin_filter_fid,
                                       dmMax_filter_fid
                                       )
    p_m1 /= jax.scipy.integrate.trapezoid(p_m1,m1_grid)
    
    ax.plot(m1_grid,p_m1,color='black',lw=0.5,alpha=0.5)

massess = np.linspace(mmin_pl_fid,mmax_pl_fid,1000)
pm_true = jgwpop.powerlaw_peak_smooth(massess,mmin_pl_fid,mmax_pl_fid,alpha_fid,sig_m1_fid,mu_m1_fid,f_peak_fid,mMin_filter_fid,mMax_filter_fid,dmMin_filter_fid,dmMax_filter_fid)

pm_true /= jax.scipy.integrate.trapezoid(pm_true,massess)

ax.plot([],'k',alpha=0.5,label='Samples')
ax.semilogy(massess,pm_true,label='True')    

D_H = (Clight/1.0e3)  / H0_fid #Mpc
m1z_mock_O5 = np.median(m1z_mock_samples,axis=1)
m2z_mock_O5 = np.median(m2z_mock_samples,axis=1)
dL_mock_O5 = np.median(dL_mock_samples,axis=1)
m1s_mock, m2s_mock, zs_mock = jgwcosmo.detector_to_source_frame_approx_dLdH(m1z_mock_O5,m2z_mock_O5,dL_mock_O5/D_H,Om0_fid,zmin=1e-3,zmax=100)

ax.hist(m1s_mock,bins=30,density=True, histtype='step',label='detected')

ax.set_yscale('log')
ax.set_ylim(1e-5,1)
ax.set_xlim(0,100)
plt.legend()
plt.savefig(f'{PATH}/'+dir_plots+'oscillate_pm_posteriors_'+model_name+inference_details+'.pdf',bbox_inches='tight')
'''
