import numpy as np


def generic_bootstrap(data, estimation_function, nsamples, alpha=0.05):
    
    estimate = estimation_function(data)
    samples = np.empty((nsamples,) + estimate.shape)
    orig_size = data.shape[-1]


    for it in xrange(nsamples):
        samples[it] = estimation_function(data[...,np.random.random_integers(0,orig_size-1,(orig_size,))])

    samples.sort(0)
    low_bound = samples[round(nsamples*alpha/2)]
    high_bound = samples[round(nsamples*(1-alpha/2))]
    std = samples.std(0)
    
    return std, low_bound, high_bound, samples
