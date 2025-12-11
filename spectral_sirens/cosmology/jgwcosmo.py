#import numpyro
import jax.numpy as jnp
from ..utils.constants import *

def Ez_inv(z,Om0,w0=-1.,wa=0.):
    OL0 = (1-Om0)
    return 1/jnp.sqrt(Om0*jnp.power((1+z),3) + OL0*jnp.power((1+z),3*(1+w0+wa)) * jnp.exp(-3*wa*z/(1+z)))

@jnp.vectorize
def d_L(z,H0,Om0,w0=-1.,wa=0.):
    zs = jnp.linspace(0,z,100)
    integral = jnp.trapezoid(Ez_inv(zs,Om0,w0,wa),zs)
    return (1+z)*integral * (Clight/1e3)  / H0 #Mpc

def diff_comoving_volume(z,H0,Om0,w0=-1.,wa=0.):
    dL = d_L(z,H0,Om0,w0,wa) #Mpc
    Ez_i = Ez_inv(z,Om0,w0,wa)
    D_H = (Clight/1e3)  / H0 #Mpc
    
    return 1.0e-9 * (4.*jnp.pi) * jnp.power(dL,2) * D_H * Ez_i / jnp.power(1.+z,2.)

def dL_interp(z,H0,Om0,w0=-1.,wa=0.,zmin=1e-3,zmax=100):
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),200)
    return jnp.interp(z, zs, d_L(zs,H0,Om0,w0,wa), left=zmin, right=zmax, period=None)

def z_at_dl(dl,H0,Om0,w0=-1.,wa=0.,zmin=1e-3,zmax=100):
    #dl in Mpc
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),200)
    return jnp.interp(dl, d_L(zs,H0,Om0,w0,wa),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame(m1z,m2z,dL,H0,Om0,w0=-1.,wa=0.,zmin=1e-3,zmax=100):
    z = z_at_dl(dL,H0,Om0,w0,wa,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z

"""Approximate luminosity distance"""
#https://arxiv.org/pdf/1111.6396.pdf
def Phi(x):
    num = 1 + 1.320*x + 0.4415* jnp.power(x,2) + 0.02656*jnp.power(x,3)
    den = 1 + 1.392*x + 0.5121* jnp.power(x,2) + 0.03944*jnp.power(x,3)
    return num/den

def xx(z,Om0):
    return (1.0-Om0)/Om0/jnp.power(1.0+z,3)

def dL_approx(z,H0,Om0):
    D_H = (Clight/1.0e3)  / H0 #Mpc
    
    return 2.*D_H * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/jnp.sqrt(1.+z))/jnp.sqrt(Om0)

def z_at_dl_approx(dl,H0,Om0,zmin=1e-3,zmax=100):
    #dl in Mpc
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),1000)
    return jnp.interp(dl, dL_approx(zs,H0,Om0),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame_approx(m1z,m2z,dL,H0,Om0,zmin=1e-3,zmax=100):
    z = z_at_dl_approx(dL,H0,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z

def diff_comoving_volume_approx(z,H0,Om0):
    dL = dL_approx(z,H0,Om0) #Mpc
    Ez_i = Ez_inv(z,Om0)
    D_H = (Clight/1e3)  / H0 #Mpc
    
    return 1.0e-9 * (4.*jnp.pi) * jnp.power(dL,2) * D_H * Ez_i / jnp.power(1.+z,2.)

def dLdH_approx(z,Om0):    
    return 2. * (1.+z) * (Phi(xx(0.,Om0)) - Phi(xx(z,Om0))/jnp.sqrt(1.+z))/jnp.sqrt(Om0)

def z_at_dldH_approx(dl,Om0,zmin=1e-3,zmax=100):
    #dldH = H0*d_L/c is dimensionless
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),1000)
    return jnp.interp(dl, dLdH_approx(zs,Om0),zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame_approx_dLdH(m1z,m2z,dLdH,Om0,zmin=1e-3,zmax=100):
    z = z_at_dldH_approx(dLdH,Om0,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z


"""Modified GW distance"""
def dGW_dL_ratio_Xi0n(z,Xi0,n):
    return Xi0 + (1-Xi0) / jnp.power(1+z,n)

def dGW_dL_ratio_cM(z,cM,Om0):
    OL0 = 1-Om0
    return jnp.exp(cM/(2*OL0) * jnp.log((1+z)/jnp.power(Om0*jnp.power(1+z,3)+OL0,1./3)))

def z_at_dGW(dGW,H0,Om0,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmin=1e-3,zmax=1e3):
    #dl in Mpc
    zs = jnp.logspace(jnp.log10(zmin),jnp.log10(zmax),200)
    if MG_model=='Xi0n':
        dGW_dL_ratio = dGW_dL_ratio_Xi0n(zs,Xi0,n)
    elif MG_model=='cM':
        dGW_dL_ratio = dGW_dL_ratio_cM(zs,cM,Om0)
    elif MG_model==None:
        dGW_dL_ratio = 1
    else:
        raise ValueError('Please choose between \'Xi0n\' and \'cM\' model.')
    
    if method=='approx':
        dL = dL_approx(zs,H0,Om0)#Mpc
    elif method=='exact':
        dL = dL_interp(zs,H0,Om0,w0,wa,zmin,zmax)#Mpc
    else:
        raise ValueError('Please choose dL computation method between \'approx\' and \'exact\'.')

    return jnp.interp(dGW, dL*dGW_dL_ratio, zs, left=zmin, right=zmax, period=None)

def detector_to_source_frame_dGW(m1z,m2z,dGW,H0,Om0,w0=-1.,wa=0.,method='approx',MG_model=None,Xi0=1,n=1.91,cM=0,zmin=1e-3,zmax=1e3):
    z = z_at_dGW(dGW,H0,Om0,w0,wa,method,MG_model,Xi0,n,cM,zmin,zmax)
    m1 = m1z / (1. + z)
    m2 = m2z / (1. + z)
    return m1, m2, z

def dGW_dL_ratio_bydz(z,MG_model=None,Xi0=1,n=1.91,cM=0,Om0=0.308):
    if MG_model=='Xi0n':
        return -1 * n*(1-Xi0)/jnp.power((1+z),(n+1))
    elif MG_model=='cM':
        OL0 = 1-Om0
        return dGW_dL_ratio_cM(z,cM,Om0) * cM/2/OL0*(1/(1+z)-Om0*jnp.power(1+z,2)/(Om0*jnp.power(1+z,3)+OL0))
    elif MG_model==None:
        return 0
    else:
        raise ValueError('Please choose between \'Xi0n\' and \'cM\' model.')
