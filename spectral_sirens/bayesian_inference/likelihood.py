import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from ..utils.constants import *
from ..cosmology import jgwcosmo
from ..gw_population import jgwpop
from ..utils import jutils

xp = jnp

#Mass likelihood
#---------------
def logpowerlaw(m,mMin,mMax,alpha):
    lognorm = xp.log(1. + alpha) - xp.log(mMax**(alpha+1.) - mMin**(alpha+1.))
    return lognorm + alpha * xp.log(m)

def logpowerlaw_peak_smooth(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter):

    # Power-law and peak
    plp = jgwpop.powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak)

    # Compute low- and high-mass filters
    loglow_filter = jutils.loglowfilter(m1,mMin_filter,dmMin_filter)
    loghigh_filter = jutils.loghighfilter(m1,mMax_filter,dmMax_filter)

    # Apply filters to combined power-law and peak
    return xp.log(plp) + loghigh_filter + loglow_filter

def logpowerlaw_double_peak_smooth(m1,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter):

    # Power-law and peak
    pldp = jgwpop.powerlaw_double_peak(m1,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak)

    # Compute low- and high-mass filters
    loglow_filter = jutils.loglowfilter(m1,mMin_filter,dmMin_filter)
    loghigh_filter = jutils.loghighfilter(m1,mMax_filter,dmMax_filter)

    # Apply filters to combined power-law and peak
    return xp.log(pldp) + loghigh_filter + loglow_filter

#Merger rate likelihood
#----------------------
def log_Rz(z,r0,zp,alpha,beta):
    logc0 = xp.log1p((1. + zp)**(-alpha-beta))
    return xp.log(r0) + logc0  + alpha*xp.log1p(z) - xp.log1p(xp.power((1.+z)/(1.+zp),(alpha+beta)))

#Cosmological likelihood
# ---------------------- 
def log_cosmo_dGW(z,dGW,H0,Om0,w0=-1.,wa=0.,MG_model=None,Xi0=1,n=1.91,cM=0):
    Ez_i = jgwcosmo.Ez_inv(z,Om0,w0,wa)
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    logdiff_comoving_volume = xp.log(1.0e-9) + xp.log(4.0*xp.pi) + 2.0*xp.log(dGW) +xp.log(D_H) +xp.log(Ez_i)-2*xp.log1p(z)
    
    if MG_model=='Xi0n':
        dGW_dL_ratio = jgwcosmo.dGW_dL_ratio_Xi0n(z,Xi0,n)
    elif MG_model=='cM':
        dGW_dL_ratio = jgwcosmo.dGW_dL_ratio_cM(z,cM,Om0)
    elif MG_model==None:
        dGW_dL_ratio = 1
    else:
        raise ValueError('Please choose between \'Xi0n\' and \'cM\' model.')
    ddGWdz = dGW/(1.+z) + (1.+z)*D_H * Ez_i * dGW_dL_ratio + dGW/dGW_dL_ratio*jgwcosmo.dGW_dL_ratio_bydz(z,MG_model,Xi0,n,cM,Om0) #Mpc 
    
    logJacobian_dL_z = - xp.log(xp.abs(ddGWdz)) #Jac has absolute value 
    logJacobian_t_td = - xp.log1p(z)
    return logdiff_comoving_volume + logJacobian_t_td + logJacobian_dL_z

def log_cosmo(z,H0,Om0,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0):
    if method=='approx':
        dL = jgwcosmo.dL_approx(z,H0,Om0)#Mpc
    elif method=='exact':
        dL = jgwcosmo.dL_interp(z,H0,Om0,w0,wa)#Mpc
    else:
        raise ValueError('Please choose dL computation method between \'approx\' and \'exact\'.')

    if MG_model=='Xi0n':
        dGW = dL * jgwcosmo.dGW_dL_ratio_Xi0n(z,Xi0,n)
    elif MG_model=='cM':
        dGW = dL * jgwcosmo.dGW_dL_ratio_cM(z,cM,Om0)
    elif MG_model==None:
        dGW = dL
    else:
        raise ValueError('Please choose between \'Xi0n\' and \'cM\' MG model.')

    return log_cosmo_dGW(z,dGW,H0,Om0,w0,wa,MG_model,Xi0,n,cM)

#Expected number of events
#-------------------------
def logNdet_events_powerlaw_peak(m1,m2,z,p_draw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0):
    #input data (N,M): N detections x M samples
    
    log_pm1 = logpowerlaw_peak_smooth(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q = m2/m1
    log_pq = logpowerlaw(q,0.,1.,bq)
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z)
    logJacobian_m1m2_m1q =  - jnp.log(m1)
    log_pm = log_pm1 + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    logcosmo = log_cosmo(z,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM)
    logRzs = log_Rz(z,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm + logcosmo + logRzs - jnp.log(p_draw)    
    
    return jnp.sum(jax.scipy.special.logsumexp(log_dN,axis=1) - jnp.log(Nsamples))

def logNdet_events_powerlaw_double_peak(m1,m2,z,p_draw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0):
    #input data (N,M): N detections x M samples
    
    log_pm1 = logpowerlaw_double_peak_smooth(m1,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q = m2/m1
    log_pq = logpowerlaw(q,0.,1.,bq)
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z)
    logJacobian_m1m2_m1q =  - jnp.log(m1)
    log_pm = log_pm1 + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    logcosmo = log_cosmo(z,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM)
    logRzs = log_Rz(z,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm + logcosmo + logRzs - jnp.log(p_draw)    
    
    return jnp.sum(jax.scipy.special.logsumexp(log_dN,axis=1) - jnp.log(Nsamples))

#Expected Ndet with injection sesitivity
#---------------------------------------
def logNdet_exp_powerlaw_peak(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmax=100):
    m1_inj, m2_inj, z_inj = jgwcosmo.detector_to_source_frame_dGW(m1z_inj,m2z_inj,dL_inj,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM,zmin=1e-3,zmax=zmax)

    log_pm1 = logpowerlaw_peak_smooth(m1_inj,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q_inj = m2_inj/m1_inj
    log_pq = logpowerlaw(q_inj,0.,1.,bq)
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m1m2_m1q =  - jnp.log(m1_inj)
    log_pm = log_pm1 + log_pq - jnp.log(norm_m1) + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    
    logcosmo = log_cosmo_dGW(z_inj,dL_inj,H0,Om0,w0,wa,MG_model,Xi0,n,cM)
    logRzs = log_Rz(z_inj,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm  + logcosmo + logRzs 
    
    #Expected number of detections
    log_N = jax.scipy.special.logsumexp(log_dN - jnp.log(p_draw_inj)) - jnp.log(Ndraw)
    
    #Effective number of samples
    log_N2 = jax.scipy.special.logsumexp(2.0*log_dN - 2.0*jnp.log(p_draw_inj)) - 2.0*jnp.log(Ndraw)
    log_sigma2 = jutils.logdiffexp(log_N2, 2.0*log_N - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_N - log_sigma2)
    
    return log_N, Neff

def logNdet_exp_powerlaw_double_peak(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmax=100):
    m1_inj, m2_inj, z_inj = jgwcosmo.detector_to_source_frame_dGW(m1z_inj,m2z_inj,dL_inj,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM,zmin=1e-3,zmax=zmax)
    
    log_pm1 = logpowerlaw_double_peak_smooth(m1_inj,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q_inj = m2_inj/m1_inj
    log_pq = logpowerlaw(q_inj,0.,1.,bq)
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_double_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m1m2_m1q =  - jnp.log(m1_inj)
    log_pm = log_pm1 + log_pq - jnp.log(norm_m1) + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    
    logcosmo = log_cosmo_dGW(z_inj,dL_inj,H0,Om0,w0,wa,MG_model,Xi0,n,cM)
    logRzs = log_Rz(z_inj,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm  + logcosmo + logRzs 
    
    #Expected number of detections
    log_N = jax.scipy.special.logsumexp(log_dN - jnp.log(p_draw_inj)) - jnp.log(Ndraw)
    
    #Effective number of samples
    log_N2 = jax.scipy.special.logsumexp(2.0*log_dN - 2.0*jnp.log(p_draw_inj)) - 2.0*jnp.log(Ndraw)
    log_sigma2 = jutils.logdiffexp(log_N2, 2.0*log_N - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_N - log_sigma2)
    
    return log_N, Neff

#Log likelihood
#--------------
@partial(jit, static_argnames=['method','MG_model'])
def log_lik_powerlaw_peak(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,h0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,alpha_z,zp,beta,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmax=100):
    #Fixed rate
    r0 = 1.0 # 10.**log10r0
    Tobs = 1.
    #redefinition
    H0 = h0*100
    
    Nobs, Nsamples = jnp.shape(m1z_mock_samples)
    
    D_H = (Clight/1.0e3)  / H0 #Mpc
    m1_mock, m2_mock, z_mock = jgwcosmo.detector_to_source_frame_dGW(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM,zmin=1e-3,zmax=zmax)

    #Log_lik Events
    p_draw_mock = pdraw_mock_samples
    loglik_E = logNdet_events_powerlaw_peak(m1_mock,m2_mock,z_mock,p_draw_mock,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,w0,wa,method,MG_model,Xi0,n,cM)

    #Total rate normalization only needed when selection effects are neglected because then N does not cancel out (see notes in example notebook)
    #zs_norm = jnp.linspace(0.01,10,1000) 
    #dn_detec = jgwpop.rate_z(zs_norm,zp,alpha_z,beta)*jgwcosmo.diff_comoving_volume_approx(zs_norm,H0,Om0)/(1.+zs_norm)   
    #norm_z = jax.scipy.integrate.trapezoid(dn_detec,zs_norm)
    #loglik_E -= Nobs*jnp.log(norm_z)
    
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    loglik_E -= Nobs*jnp.log(norm_m1)
        
    #Selection effects
    log_Ndet, Neff = logNdet_exp_powerlaw_peak(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,w0,wa,method,MG_model,Xi0,n,cM,zmax)
    loglik_N = -Nobs*log_Ndet
        
    return loglik_N + loglik_E, Neff

@partial(jit, static_argnames=['method','MG_model'])
def log_lik_powerlaw_double_peak(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,h0,Om0,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,alpha_z,zp,beta,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmax=100):
    #Fixed rate
    r0 = 1.0 # 10.**log10r0
    Tobs = 1.
    #redefinition
    H0 = h0*100
    
    Nobs, Nsamples = jnp.shape(m1z_mock_samples)
    
    D_H = (Clight/1.0e3)  / H0 #Mpc
    m1_mock, m2_mock, z_mock = jgwcosmo.detector_to_source_frame_dGW(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM,zmin=1e-3,zmax=zmax)
    
    #Log_lik Events
    p_draw_mock = pdraw_mock_samples
    loglik_E = logNdet_events_powerlaw_double_peak(m1_mock,m2_mock,z_mock,p_draw_mock,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,w0,wa,method,MG_model,Xi0,n,cM)

    #Total rate normalization only needed when selection effects are neglected because then N does not cancel out (see notes in example notebook)
    #zs_norm = jnp.linspace(0.01,10,1000) 
    #dn_detec = jgwpop.rate_z(zs_norm,zp,alpha_z,beta)*jgwcosmo.diff_comoving_volume_approx(zs_norm,H0,Om0)/(1.+zs_norm)   
    #norm_z = jax.scipy.integrate.trapezoid(dn_detec,zs_norm)
    #loglik_E -= Nobs*jnp.log(norm_z)
    
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_double_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    loglik_E -= Nobs*jnp.log(norm_m1)
        
    #Selection effects
    log_Ndet, Neff = logNdet_exp_powerlaw_double_peak(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1_low,mu_m1_low,f_peak_low,sig_m1_high,mu_m1_high,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,w0,wa,method,MG_model,Xi0,n,cM,zmax)
    loglik_N = -Nobs*log_Ndet
        
    return loglik_N + loglik_E, Neff

#Toy models
#----------
def logbox_smooth(x,edge,width,filt):
    low_edge = edge
    high_edge = edge + width

    loglow_filter = -(x-low_edge)**2/(2.*filt**2)
    loglow_filter = xp.where(x<low_edge,loglow_filter,0.)
    loghigh_filter = -(x-high_edge)**2/(2.*filt**2)
    loghigh_filter = xp.where(x>high_edge,loghigh_filter,0.)

    return loglow_filter + loghigh_filter - xp.log(width)

def logtwo_box(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return xp.where(x < switch,logbox_smooth(x,edge_1,width_1,filt),logbox_smooth(x,edge_2,width_2,filt))

def logsigmoid(x,edge,width):
    exponent = (x-edge)/width
    return jax.nn.log_sigmoid(exponent)

def logbox_sig(x,edge,width,filt):
    low_edge = edge
    high_edge = edge + width
    mid_point = edge + width/2.

    loglow_filter = xp.where(x<mid_point,logsigmoid(x,low_edge-2*filt,filt),0.)
    loghigh_filter = xp.where(x>mid_point,logsigmoid(-x,-high_edge-2*filt,filt),0.)

    return loglow_filter + loghigh_filter - xp.log(width)

def logtwo_box_sig(x,edge_1,width_1,edge_2,width_2,filt,switch):
    
    return xp.where(x < switch,logbox_sig(x,edge_1,width_1,filt),logbox_sig(x,edge_2,width_2,filt))

def loggaussian(x,mu,sig):
    return -(x-mu)**2/(2.*sig**2) - xp.log(xp.sqrt(2.*xp.pi*sig**2))

def loguniform_sigmoid(x,high_edge,width,filt):
    low_edge = high_edge - width
    mid_point = high_edge - width/2.

    loglow_filter = xp.where(x<mid_point,logsigmoid(x,low_edge-2*filt,filt),0.)
    loghigh_filter = xp.where(x>mid_point,logsigmoid(-x,-high_edge-2*filt,filt),0.)

    return loglow_filter + loghigh_filter - xp.log(width)