### Likelihood with Gaussian Process for merger rate
import jax
import jax.numpy as jnp
from jax import jit, core, lax
from jax.scipy.special import gammaln
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample, promote_shapes, is_prng_key
from tinygp import kernels, GaussianProcess
from tinygp.solvers import QuasisepSolver
from ..utils.constants import *
from ..cosmology import jgwcosmo
from ..gw_population import jgwpop
from ..utils import jutils
from fiducial_universe_gwtc3 import *

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

class Frechet(Distribution):
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.positive
    reparametrized_params = ["scale", "concentration"]

    def __init__(self, scale, concentration, *, validate_args=None):
        self.concentration, self.scale = promote_shapes(concentration, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jax.random.weibull_min(
            key,
            scale=self.scale,
            concentration=-self.concentration,
            shape=sample_shape + self.batch_shape,
        )

    @validate_sample
    def log_prob(self, value):
        """https://en.wikipedia.org/wiki/Weibull_distribution#Related_distributions"""
        k = self.concentration
        ll = -jnp.power(value / self.scale, -k)
        ll += jnp.log(k)
        ll -= (k + 1.0) * jnp.log(value)
        ll += k * jnp.log(self.scale)
        return ll

    def cdf(self, value):
        return jnp.exp(-((value / self.scale) ** -self.concentration))

    @property
    def mean(self):
        return self.scale * jnp.exp(gammaln(1.0 - 1.0 / self.concentration))

    @property
    def median(self):
        return self.scale/jnp.log(2)**(1/self.concentration)

    @property
    def variance(self):
        var = jnp.where(self.concentration >2.,
                        self.scale**2 * (
                            jnp.exp(gammaln(1.0 - 2.0 / self.concentration))
                            - jnp.exp(gammaln(1.0 - 1.0 / self.concentration)) ** 2
                        ),
                        jnp.inf)
        return var

#Log likelihood
#--------------
# @jit
def log_lik(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,pdraw_mock_samples,m1z_inj,m2z_inj,dL_inj,p_draw_inj,Ndraw,Tobs,PC_params=None):
    #Fixed rate
    r0 = 1.0 # 10.**log10r0
    #Tobs = 1.
    #redefinition
    H0 = numpyro.sample("H0",dist.Uniform(20,140)) #H0_fid
    Om0 = Om0_fid
    bq = bq_fid
    zp = zp_fid
    alpha_z = numpyro.sample("alpha_z",dist.Uniform(0,10)) #alpha_z_fid
    beta = beta_fid

    """ Non-parametric population inference """
    mean = numpyro.deterministic("mean",0) #numpyro.sample("mean",dist.Normal(0,3))
    sigma = numpyro.deterministic("sigma",2.5)
    # sigma = numpyro.sample("sigma",dist.Gamma(concentration=PC_params["conc"], rate=PC_params["lam_sigma"]))
    rho = numpyro.deterministic("rho",0.5)
    # rho = numpyro.sample("rho",Frechet(concentration=PC_params["concentration"],scale=PC_params["scale"]))
    
    Nobs, Nsamples = jnp.shape(m1z_mock_samples)
    
    D_H = (Clight/1.0e3)  / H0 #Mpc
    m1_mock, m2_mock, z_mock = jgwcosmo.detector_to_source_frame_approx(m1z_mock_samples,m2z_mock_samples,dL_mock_samples,H0,Om0,zmin=1e-3,zmax=100)

    TEST_M1S = jnp.linspace(0.1,150.,num=300)
    logtestm1s = jnp.log(TEST_M1S)
    kernel = sigma**2 * kernels.quasisep.Matern52(rho) # can change kernel type
    gp = GaussianProcess(kernel,logtestm1s,mean=mean,diag=0.001,
                         solver=QuasisepSolver,assume_sorted=True)
    log_rate_test = numpyro.sample("log_rate_test",gp.numpyro_dist())

    #Expected number of events
    logm1source = jnp.log(m1_mock)
    log_rate_m1s_data = jnp.interp(logm1source,logtestm1s,log_rate_test-logtestm1s,left=-jnp.inf,right=-jnp.inf)
    q = m2_mock/m1_mock
    log_pq = logpowerlaw(q,0.,1.,bq)
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_mock)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_mock)
    logJacobian_m1m2_m1q =  - jnp.log(m1_mock)
    log_pm = log_rate_m1s_data + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q

    logcosmo = log_cosmo(z_mock,H0,Om0)
    logRzs = log_Rz(z_mock,r0,zp,alpha_z,beta) + jnp.log(Tobs)    
    log_dN = log_pm + logcosmo + logRzs - jnp.log(pdraw_mock_samples)    
    
    loglik_E = jnp.sum(jax.scipy.special.logsumexp(log_dN,axis=1) - jnp.log(Nsamples))
    # m1s_norm = jnp.linspace(mMin,mMax,1000)    
    # norm_m1 = jax.scipy.integrate.trapezoid(jgwpop.powerlaw_peak_smooth(m1s_norm,mMin,mMax,alpha,sig_m1,mu_m1,f_peak,mMin_filter,mMax_filter,dmMin_filter,dmMax_filter),m1s_norm)
    # loglik_E -= Nobs*jnp.log(norm_m1)
        
    #Expected Ndet with injection sesitivity
    m1_inj, m2_inj, z_inj = jgwcosmo.detector_to_source_frame_approx(m1z_inj,m2z_inj,dL_inj,H0,Om0,zmin=1e-3,zmax=100)
    
    logm1_injs = jnp.log(m1_inj)
    log_rate_m1s_injs = jnp.interp(logm1_injs,logtestm1s,log_rate_test-logtestm1s,left=-jnp.inf,right=-jnp.inf)
    q_inj = m2_inj/m1_inj
    log_pq = logpowerlaw(q_inj,0.,1.,bq)
    
    logJacobian_m1z_m1 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m2z_m2 = - 1.0*jnp.log1p(z_inj)
    logJacobian_m1m2_m1q =  - jnp.log(m1_inj)
    log_pm = log_rate_m1s_injs + log_pq + logJacobian_m1z_m1 + logJacobian_m2z_m2 + logJacobian_m1m2_m1q

    logcosmo = log_cosmo_dL(z_inj,dL_inj,H0,Om0)
    logRzs = log_Rz(z_inj,r0,zp,alpha_z,beta) + jnp.log(Tobs)
    log_dN = log_pm  + logcosmo + logRzs 
    
    #Expected number of detections
    log_Ndet = jax.scipy.special.logsumexp(log_dN - jnp.log(p_draw_inj)) - jnp.log(Ndraw)
    
    #Effective number of samples
    log_N2 = jax.scipy.special.logsumexp(2.0*log_dN - 2.0*jnp.log(p_draw_inj)) - 2.0*jnp.log(Ndraw)
    log_sigma2 = jutils.logdiffexp(log_N2, 2.0*log_Ndet - jnp.log(Ndraw))
    Neff = jnp.exp(2.0*log_Ndet - log_sigma2)
        
    #Selection effects
    # loglik_N = -Nobs*log_Ndet
    #Full merger rate
    loglik_N = -jnp.exp(log_Ndet)
        
    loglik = loglik_N + loglik_E

    #Likelihood
    numpyro.factor("logp",loglik)


