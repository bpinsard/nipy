import numpy as np


def generic_bootstrap(data, estimation_function, nsamples, alpha=0.05):
    
    estimate = estimation_function(data)
    samples = np.empty((nsamples,) + estimate.shape, data.dtype)
    orig_size = data.shape[-1]
    rnd = np.empty((orig_size,),np.int)

    for it in xrange(nsamples):
        rnd[:] = np.random.random_integers(0,orig_size-1,(orig_size,))
        samples[it] = estimation_function(data[...,rnd])

    samples.sort(0)
    low_bound = samples[round(nsamples*alpha/2)]
    high_bound = samples[round(nsamples*(1-alpha/2))]
    std = samples.std(0)
    
    del rnd
    return std, low_bound, high_bound, samples
