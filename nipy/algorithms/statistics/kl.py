import numpy as np
import scipy.stats as scistat


def norm_kl_divergence(samples,drange=[-1,1],nbins=101,mn=None,vr=None):
    """compute the KL divergence of the samples to the normal distribution 
    with mean and variance set to empirical values"""
    # allow for precomputed mean/variance or custom estimators
    if mn == None:
        mn = samples.mean(1)
    mn = mn[:,np.newaxis]
    if vr == None:
        vr = samples.var(1)
    vr = vr[:,np.newaxis]
    #compute normal distributions center in mean and var
    x = np.linspace(drange[0],drange[1],nbins)
    ndists = np.exp(-0.5*np.square(x-mn)/vr)/np.sqrt(2*np.pi*vr)
    ndists = ndists/ndists.sum(1)[:,np.newaxis]

    nsamp = samples.shape[0]
    #compute distribution function for data
    sdists = np.empty((nsamp,nbins))
    for si,sd in enumerate(samples):
        sdists[si] = np.histogram(sd, nbins, drange,density=True)[0]
    sdists = sdists/sdists.sum(1)[:,np.newaxis]
    #compute KL divergence
    kl = sdists*np.log(sdists/ndists)
    kl[np.isnan(kl)] = 0
    kl = kl.sum(1)
    return kl
