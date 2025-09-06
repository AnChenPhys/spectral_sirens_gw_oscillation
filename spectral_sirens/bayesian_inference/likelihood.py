import jax
import jax.numpy as jnp
from jax import jit
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

#Merger rate likelihood
#----------------------
def log_Rz(z,r0,zp,alpha,beta):
    logc0 = xp.log1p((1. + zp)**(-alpha-beta))
    return xp.log(r0) + logc0  + alpha*xp.log1p(z) - xp.log1p(xp.power((1.+z)/(1.+zp),(alpha+beta)))

#Cosmological likelihood
# ---------------------- 
def log_cosmo_dL(z,dL,H0,Om0):
    Ez_i = jgwcosmo.Ez_inv(z,Om0)
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    logdiff_comoving_volume = xp.log(1.0e-9) + xp.log(4.0*xp.pi) + 2.0*xp.log(dL) +xp.log(D_H) +xp.log(Ez_i)-2*xp.log1p(z)
    ddLdz = dL/(1.+z) + (1. + z)*D_H * Ez_i #Mpc 
    logJacobian_dL_z = - xp.log(xp.abs(ddLdz)) #Jac has absolute value 
    logJacobian_t_td = - xp.log1p(z)
    return logdiff_comoving_volume + logJacobian_t_td + logJacobian_dL_z 

def log_cosmo(z,H0,Om0):
    dL = jgwcosmo.dL_approx(z,H0,Om0)#Mpc
    return log_cosmo_dL(z,dL,H0,Om0)

#Expected number of events
#-------------------------
def logNdet_events(m1,m2,z,p_draw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples):
    #input data (N,M): N detections x M samples
    
    log_pm1 = logpowerlaw_peak_smooth(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q = m2/m1
    log_pq = logpowerlaw(q,0.,1.,bq)
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z)
    logJacobian_m1m2_m1q =  - jnp.log(m1)
    log_pm = log_pm1 + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    logcosmo = log_cosmo(z,H0,Om0)
    logRzs = log_Rz(z,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm + logcosmo + logRzs - jnp.log(p_draw)    
    
    return jnp.sum(jax.scipy.special.logsumexp(log_dN,axis=1) - jnp.log(Nsamples))

#Expected Ndet with injection sesitivity
#---------------------------------------
def logNdet_exp(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq):
    m1_inj, m2_inj, z_inj = jgwcosmo.detector_to_source_frame_approx(m1z_inj,m2z_inj,dL_inj,H0,Om0,zmin=1e-3,zmax=100)
    
    log_pm1 = logpowerlaw_peak_smooth(m1_inj,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q_inj = m2_inj/m1_inj
    log_pq = logpowerlaw(q_inj,0.,1.,bq)
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m1m2_m1q =  - jnp.log(m1_inj)
    log_pm = log_pm1 + log_pq - jnp.log(norm_m1) + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    
    logcosmo = log_cosmo_dL(z_inj,dL_inj,H0,Om0)
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
@jit
def log_lik(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,h0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,alpha_z,zp,beta,r0=1.,Tobs=1.):
    #Fixed rate
    #r0 = 1.0 # 10.**log10r0
    #Tobs = 1.
    #redefinition
    H0 = h0*100
    
    Nobs, Nsamples = jnp.shape(m1z_mock_samples)
    
    D_H = (Clight/1.0e3)  / H0 #Mpc
    m1_mock, m2_mock, z_mock = jgwcosmo.detector_to_source_frame_approx(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,H0,Om0,zmin=1e-3,zmax=100)
    
    #Log_lik Events
    p_draw_mock = pdraw_mock_samples
    loglik_E = logNdet_events(m1_mock,m2_mock,z_mock,p_draw_mock,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples)

    #Total rate normalization only needed when selection effects are neglected because then N does not cancel out (see notes in example notebook)
    #zs_norm = jnp.linspace(0.01,10,1000) 
    #dn_detec = r0*jgwpop.rate_z(zs_norm,zp,alpha_z,beta)*jgwcosmo.diff_comoving_volume_approx(zs_norm,H0,Om0)/(1.+zs_norm)   
    #norm_z = jax.scipy.integrate.trapezoid(dn_detec,zs_norm)
    #loglik_E -= Nobs*jnp.log(norm_z)
    
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    loglik_E -= Nobs*jnp.log(norm_m1)
        
    #Selection effects
    log_Ndet, Neff = logNdet_exp(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,r0,Tobs,zp,alpha_z,beta,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq)
    # loglik_N = -Nobs*log_Ndet
    #Full merger rate
    loglik_N = -jnp.exp(log_Ndet)
        
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


###GW oscillation
#Merger rate likelihood
#----------------------
def log_Rz_osci(z,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10):
    Theta1 = xp.heaviside(z_step-z,1)
    Theta2 = xp.heaviside(2*z_step-z,1)*xp.heaviside(z-z_step,1) 
    Theta3 = xp.heaviside(3*z_step-z,1)*xp.heaviside(z-2*z_step,1) 
    Theta4 = xp.heaviside(4*z_step-z,1)*xp.heaviside(z-3*z_step,1) 
    Theta5 = xp.heaviside(5*z_step-z,1)*xp.heaviside(z-4*z_step,1) 
    Theta6 = xp.heaviside(6*z_step-z,1)*xp.heaviside(z-5*z_step,1) 
    Theta7 = xp.heaviside(7*z_step-z,1)*xp.heaviside(z-6*z_step,1) 
    Theta8 = xp.heaviside(8*z_step-z,1)*xp.heaviside(z-7*z_step,1) 
    Theta9 = xp.heaviside(9*z_step-z,1)*xp.heaviside(z-8*z_step,1) 
    Theta10 = xp.heaviside(10*z_step-z,1)*xp.heaviside(z-9*z_step,1) 
    # return alphaz1*xp.log1p(z)*Theta1 + alphaz2*xp.log1p(z)*Theta2 + alphaz3*xp.log1p(z)*Theta3 + alphaz4*xp.log1p(z)*Theta4 + alphaz5*xp.log1p(z)*Theta5 + alphaz6*xp.log1p(z)*Theta6 + alphaz7*xp.log1p(z)*Theta7 + alphaz8*xp.log1p(z)*Theta8 + alphaz9*xp.log1p(z)*Theta9 + alphaz10*xp.log1p(z)*Theta10
    return alphaz1*Theta1 + alphaz2*Theta2 + alphaz3*Theta3 + alphaz4*Theta4 + alphaz5*Theta5 + alphaz6*Theta6 + alphaz7*Theta7 + alphaz8*Theta8 + alphaz9*Theta9 + alphaz10*Theta10

#Expected number of events
#-------------------------
def logNdet_events_osci(m1,m2,z,p_draw,H0,Om0,Tobs,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10):
    #input data (N,M): N detections x M samples
    
    log_pm1 = logpowerlaw_peak_smooth(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q = m2/m1
    log_pq = logpowerlaw(q,0.,1.,bq)
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z)
    logJacobian_m1m2_m1q =  - jnp.log(m1)
    log_pm = log_pm1 + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    logcosmo = log_cosmo(z,H0,Om0)
    logRzs = log_Rz_osci(z,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10) + jnp.log(Tobs)
    log_dN = log_pm + logcosmo + logRzs - jnp.log(p_draw)    
    
    return jnp.sum(jax.scipy.special.logsumexp(log_dN,axis=1) - jnp.log(Nsamples))

#Expected Ndet with injection sesitivity
#---------------------------------------
def logNdet_exp_osci(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,Tobs,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10):
    m1_inj, m2_inj, z_inj = jgwcosmo.detector_to_source_frame_approx(m1z_inj,m2z_inj,dL_inj,H0,Om0,zmin=1e-3,zmax=100)
    
    log_pm1 = logpowerlaw_peak_smooth(m1_inj,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter)
    q_inj = m2_inj/m1_inj
    log_pq = logpowerlaw(q_inj,0.,1.,bq)
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m1m2_m1q =  - jnp.log(m1_inj)
    log_pm = log_pm1 + log_pq - jnp.log(norm_m1) + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q
    
    logcosmo = log_cosmo_dL(z_inj,dL_inj,H0,Om0)
    logRzs = log_Rz_osci(z_inj,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10) + jnp.log(Tobs)
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
@jit
def log_lik_osci(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,h0,Om0,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10):
    #Fixed rate
    #r0 = 1.0 # 10.**log10r0
    Tobs = 1.
    #redefinition
    H0 = h0*100
    
    Nobs, Nsamples = jnp.shape(m1z_mock_samples)
    
    D_H = (Clight/1.0e3)  / H0 #Mpc
    m1_mock, m2_mock, z_mock = jgwcosmo.detector_to_source_frame_approx(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,H0,Om0,zmin=1e-3,zmax=100)
    
    #Log_lik Events
    p_draw_mock = pdraw_mock_samples
    loglik_E = logNdet_events_osci(m1_mock,m2_mock,z_mock,p_draw_mock,H0,Om0,Tobs,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,Nsamples,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10)

    #Total rate normalization only needed when selection effects are neglected because then N does not cancel out (see notes in example notebook)
    #zs_norm = jnp.linspace(0.01,10,1000) 
    #dn_detec = jgwpop.rate_z(zs_norm,zp,alpha_z,beta)*jgwcosmo.diff_comoving_volume_approx(zs_norm,H0,Om0)/(1.+zs_norm)   
    #norm_z = jax.scipy.integrate.trapezoid(dn_detec,zs_norm)
    #loglik_E -= Nobs*jnp.log(norm_z)
    
    m1s_norm = jnp.linspace(mMin,mMax,1000)    
    norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    loglik_E -= Nobs*jnp.log(norm_m1)
        
    #Selection effects
    log_Ndet, Neff = logNdet_exp_osci(m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,H0,Om0,Tobs,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter,bq,z_step,alphaz1,alphaz2,alphaz3,alphaz4,alphaz5,alphaz6,alphaz7,alphaz8,alphaz9,alphaz10)
    loglik_N = -Nobs*log_Ndet
        
    return loglik_N + loglik_E, Neff
