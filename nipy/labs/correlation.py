import numpy as np
import scipy.signal as sig
    

def all_distances(coords, coords2=None):
    """ computes all distances between coordinates """
    if coords2 == None:
        dists = coords.reshape(coords.shape+(1,)).repeat(coords.shape[0],
                                                         axis=2)
        dists = dists-dists.transpose((2,1,0))
    else:
        dists = coords.reshape(coords.shape+(1,)).repeat(coords2.shape[0],
                                                         axis=2)
        coords2 = coords2.reshape(coords2.shape+(1,)).repeat(coords.shape[0],
                                                             axis=2)
        dists = dists-coords2.transpose((2,1,0))
        
    dists = np.sqrt(np.sum(dists*dists,axis=1).squeeze())
#    dists = np.sqrt(dists[np.tri(dists.shape[0],k=-1,dtype=bool)])

    return dists

def all_correlations(data1,data2):
#    corr = np.empty((data1.shape[0],data2.shape[0]),np.float)
#    for its1,ts1 in enumerate(data1):
#        for its2,ts2 in enumerate(data2):
#            corr[its1,its2]=np.corrcoef(ts1,ts2)[0,1]
    #significant speedup !!! be carefull with memory
    corr = np.corrcoef(data1,data2)[-(data2.shape[0]+1):-1,:data1.shape[0]]
    return corr

def sample_correlation(data, masks,
                       frequency_bands=[], voxel_size=(1,1,1),
                       nsamples=1000, tr=2):
    """
    samples the voxel correlation in the data using masks of interest
    intra-masks correlations and inter-masks correlations are sampled as well as the distance between the voxels.
    
    Parameters
    ----------
    data : the 4D run data from which to sample signal
    mask : the masks of interest in which we sample voxels
    frequency_bands : a list of (lower_bound, upper_bound) tuples which are used to bandpass-filter the timeseries prior to compute correlation, all results are indexed with the first one [0] being the all-frequencies correlation (no filtering) and the next being the one listed here : ex [(0.08,0.12)]
    voxel_size : a 3-tuple which contains the voxel size, this is used to compute the distance between sampled voxels
    nsamples : the number of voxels to be sampled per mask
    tr : the time of repetition (sampling interval) of the data, used for filtering 

    Returns
    -------
    intra_corr  : intra mask correlation samples indexed by frequency band and mask
    inter_corr  : inter mask correlation samples indexed by frequency band and mask
    intra_dists : intra mask distances samples indexed by mask
    inter_dists : inter mask distances samples indexed by mask

    """

    nfbands = len(frequency_bands)
    #convert cutoffs
    frequency_bands=[[freq*2*tr for freq in band] for band in frequency_bands]
    
    mdata = data.mean(axis=3)
    if masks.ndim == 4:
        nmasks = masks.shape[3]
    elif masks.ndim == 3:
        nmasks = 1
        masks = masks.reshape(masks.shape+(1,))

    gmask = (mdata != 0) * (mdata != np.nan)* (masks.sum(axis=3)>0)
    masks[gmask==False]=False

    n_vox = np.count_nonzero(gmask)
    s = gmask.shape

    sampled_data = np.empty((nmasks,nsamples,data.shape[-1]),np.float16)
    coords = np.zeros((nmasks,nsamples,3),np.float16)
    #the distance set to 0 correspond to non-sampled data due to limited mask
    intra_dists = np.zeros((nmasks,nsamples*(nsamples-1)/2), np.float16)
    inter_dists = np.zeros((nmasks*(nmasks-1)/2,nsamples*nsamples), np.float16)
    intra_corr  = np.empty((nfbands+1,)+intra_dists.shape, np.float16)
    inter_corr  = np.empty((nfbands+1,)+inter_dists.shape, np.float16)

    ntissamples = [0]*nmasks

    for mi in np.arange(nmasks):
        n_vox = np.count_nonzero(masks[...,mi])
        indices = np.nonzero(masks[...,mi])
        ntissamples[mi] = min(nsamples,n_vox)
        ncomb = ntissamples[mi]*(ntissamples[mi]-1)/2
        randind = np.random.permutation(n_vox)[0:ntissamples[mi]]
        indices = [ind[randind] for ind in indices]
        coords[mi,:ntissamples[mi]] = np.array([indices[0]*voxel_size[0],
                               indices[1]*voxel_size[1],
                               indices[2]*voxel_size[2]]).T

        tridx = np.tri(ntissamples[mi],k=-1,dtype=bool)

        sampled_data[mi,:ntissamples[mi]] = data[indices]
        intra_corr[0,mi,:ncomb] = np.corrcoef(sampled_data[mi])[tridx]

        for bi,band in enumerate(frequency_bands):
            bb, ba = sig.butter(5, band,'bandpass')
            fdata = np.empty(sampled_data[mi,:ntissamples[mi]].shape)
            for ti,ts in enumerate(sampled_data[mi,:ntissamples[mi]]):
                fdata[ti] = sig.filtfilt(bb,ba,ts)
            intra_corr[bi+1,mi,:ncomb] = np.corrcoef(fdata)[tridx]
                
        #compute intra mask distances
        dists = all_distances(coords[mi])
        intra_dists[mi] = dists[np.tri(dists.shape[0],k=-1,dtype=bool)]
        
        for mii in np.arange(mi):
            interind=(mi)*(mi-1)/2+mii
            inter_dists[interind] = all_distances(
                coords[mi],coords[mii]).flatten()
            inter_corr[0,interind] = all_correlations(
                sampled_data[mi],
                sampled_data[mii]).flatten()

            for bi,band in enumerate(frequency_bands):
                bb, ba = sig.butter(5, band,'bandpass')
                fdata1 = np.empty(sampled_data[mi,:ntissamples[mi]].shape)
                for ti,ts in enumerate(sampled_data[mi,:ntissamples[mi]]):
                    fdata1[ti] = sig.filtfilt(bb,ba,ts)
                fdata2 = np.empty(sampled_data[mii,:ntissamples[mii]].shape)
                for ti,ts in enumerate(sampled_data[mii,:ntissamples[mii]]):
                    fdata2[ti] = sig.filtfilt(bb,ba,ts)
                ncomb=ntissamples[mi]*ntissamples[mii]
                inter_corr[bi+1,interind,:ncomb] = all_correlations(
                    fdata1,fdata2).flatten()
    
    return intra_corr, inter_corr, intra_dists, inter_dists


def mutli_sample_correlation(datas, masks, voxel_size=(1,1,1),
                             nsamples=1000):
    """
    samples the voxel correlation in the data using masks of interest
    intra-masks correlations are sampled as well as the distance between the same voxels in the different datasets provided
    
    Parameters
    ----------
    datas : the list of  4D runs data from which to sample signal
    mask : the masks of interest in which we sample voxels
    voxel_size : a 3-tuple which contains the voxel size, this is used to compute the distance between sampled voxels
    nsamples : the number of voxels to be sampled per mask

    Returns
    -------
    intra_corr  : intra mask correlation samples indexed by data and mask
    inter_corr  : inter mask correlation samples indexed by data and mask
    intra_dists : intra mask distances samples indexed by mask
    inter_dists : inter mask distances samples indexed by mask

    """
    
    if not isinstance(datas,list) and isinstance(datas,np.array):
        datas = [datas]
    ndata = len(datas)
    mdata = np.mean([data.mean(axis=3) for data in datas], axis=0)
    if masks.ndim == 4:
        nmasks = masks.shape[3]
    elif masks.ndim == 3:
        nmasks = 1
        masks = masks.reshape(masks.shape+(1,))

    gmask = (mdata != 0) * (mdata != np.nan)* (masks.sum(axis=3)>0)
    masks[gmask==False]=False

    n_vox = np.count_nonzero(gmask)
    s = gmask.shape

    sampled_data = np.empty((nmasks,nsamples,data.shape[-1]),np.float16)
    coords = np.zeros((nmasks,nsamples,3),np.float16)
    intra_dists = np.empty((nmasks,nsamples*(nsamples-1)/2), np.float16)
    intra_corr  = np.empty((ndata,)+intra_dists.shape, np.float16)

    ntissamples = [0]*nmasks

    for mi in xrange(nmasks):
        n_vox = np.count_nonzero(masks[...,mi])
        indices = np.nonzero(masks[...,mi])
        ntissamples[mi] = min(nsamples,n_vox)
        ncomb = ntissamples[mi]*(ntissamples[mi]-1)/2
        randind = np.random.permutation(n_vox)[0:ntissamples[mi]]
        indices = [ind[randind] for ind in indices]
        coords[mi,:ntissamples[mi]] = np.array([indices[0]*voxel_size[0],
                               indices[1]*voxel_size[1],
                               indices[2]*voxel_size[2]]).T

        tridx = np.tri(ntissamples[mi],k=-1,dtype=bool)

        for di in xrange(ndata):
            sampled_data[mi,:ntissamples[mi]] = datas[di][indices]
            intra_corr[di,mi,:ncomb] = np.corrcoef(sampled_data[mi])[tridx]

        #compute intra mask distances
        dists = all_distances(coords[mi])
        intra_dists[mi] = dists[np.tri(dists.shape[0],k=-1,dtype=bool)]

    return intra_corr, intra_dists
