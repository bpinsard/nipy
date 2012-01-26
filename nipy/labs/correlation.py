# emacs: -*- mode: python; py-indent-offset: 2; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

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


def multi_sample_correlation(datas, masks, voxel_size=(1,1,1),
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


def sample_correlation_maps(data, mask, seeds_mask, nseeds=-1, 
                            sampling_method='mask', 
                            rois_sampling_ratio = 0.1):

    datam = data[mask]

    #normalizing the timeseries
    for ts in datam:
        mean = ts.mean()
        std = ts.std()
        if std==0:
            ts[...] = 0
        else:
            ts[...] = (ts-mean)/std
    
    samp_map = np.zeros(np.count_nonzero(mask), bool)
    seeds_mask = seeds_mask[mask]
    if sampling_method == 'rois':
        indices = np.empty(0,int)
        roi_ids = np.unique(seeds_mask)[1:]
        for roi_id in roi_ids:
            nvox = np.count_nonzero(seeds_mask==roi_id)
            rois_ind = np.nonzero(seeds_mask==roi_id)[0]
            nsamp = int(np.floor(rois_sampling_ratio*nvox))
            randind = rois_ind[np.random.permutation(rois_ind.size)[:nsamp]]
            indices = np.concatenate((indices,randind))
        del randind
        nseeds = indices.size
        print 'sampling %d seeds in %d rois' % (nseeds, roi_ids.size)
    else:
        indices = np.nonzero(seeds_mask[mask])[0]
        if nseeds < 0:
            nseeds = indices.shape[0]

        randind = np.random.permutation(indices.shape[0])[:nseeds]
        indices = indices[randind]
        del randind

    sample_map=np.zeros(mask.shape, np.int8)
    sample_map[[ind[indices] for ind in np.nonzero(mask)]] = 1
    cmaps = np.empty((datam.shape[0],nseeds),np.float16)
    nsamples = datam.shape[1] - 1.0
    for k,l in enumerate(indices):
        cmaps[:,k] = datam.dot(datam[l])/nsamples
    return cmaps, sample_map


def seed_correlation(s,m):

    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray

    
    block_size = int(2**np.floor(np.log2(s.shape[0])))
    print block_size
    out_type='float'
    in_type='float'
    s = (s-s.mean())/s.std()
    neutral = 0
    kernel_code_template = """
#define BLOCK_SIZE %(block_size)d
typedef %(out_type)s out_type;
typedef %(in_type)s in_type;

__device__ void warpReduce(volatile out_type *sdata, unsigned int tid)
{
	if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
	if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
	if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
	if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
	if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
	if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
}
extern "C"
__global__
void seedCorr(out_type *out, in_type *s, in_type *m, int n)
{
	__shared__ out_type sum[BLOCK_SIZE];
	__shared__ out_type sqsum[BLOCK_SIZE];
	__shared__ out_type covsum[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*n + tid;
	
	if (tid+blockDim.x < n){
		unsigned int ofst=i+blockDim.x;
		sum[tid] = m[i] + m[ofst];
		sqsum[tid] = m[i]*m[i] + m[ofst]*m[ofst];
		covsum[tid] = m[i]*s[tid] + m[ofst]*s[ofst];
	}
	else{
		sum[tid]  = m[i];
		sqsum[tid] = m[i]*m[i];
		covsum[tid] = m[i]*s[tid];
	}
	__syncthreads();
	for(unsigned int k=BLOCK_SIZE/4; k>32; k>>=1){
		if (tid < k){
			sum[tid] = sum[tid] + sum[tid + k];
			sqsum[tid] = sqsum[tid] + sqsum[tid + k];
			sqsum[tid] = sqsum[tid] + sqsum[tid + k];
		}
		__syncthreads();
	}
	if(tid < 32){
		warpReduce(sum,tid);
		warpReduce(sqsum,tid);
		warpReduce(covsum,tid);
	}
	if (tid == 0)
		out[blockIdx.x] =  1.0; /*sum[0];  covsum/sumsq-sum*sum/(BLOCK_SIZE-1);*/
}"""
    src = kernel_code_template % {
        "out_type": out_type,
        "in_type": in_type,
        "block_size": block_size,
        "neutral": neutral
        }

    print src
    mod = SourceModule(src)
    func = mod.get_function('seedCorr')
    cc = gpuarray.empty((m.shape[0]), np.float32)
    nframe = np.uint32(m.shape[-1])
    func(cc, drv.In(s.astype(np.float32)), drv.In(m.astype(np.float32)), nframe,
         block = (block_size,1,1), grid=(m.shape[0],1))
    return cc

def cuda_correlation():
    
    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy as np
    from pycuda.compiler import SourceModule
    
    mod = SourceModule("""

__global__ void
gpuRho(float *out, float *in, unsigned int n, unsigned int m){
    __shared__ float Xs[16][16];
    __shared__ float Ys[16][16];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int xBegin = bx * 16 * m;
    int yBegin = by * 16 * m;
    int yEnd = yBegin + m - 1;
    int x, y, k, o;
    float a1, a2, a3, a4, a5;
    float avgX, avgY, varX, varY, cov, rho;
    a1 = a2 = a3 = a4 = a5 = 0.0;
    for(y=yBegin,x=xBegin;y<=yEnd; y+=16,x+=16){
        Ys[ty][tx] = in[y + ty*m + tx];
        Xs[tx][ty] = in[x + ty*m + tx];
        //*** note the transpose of Xs
        __syncthreads();
        for(k=0;k<16;k++){
            a1 += Xs[k][tx];
            a2 += Ys[ty][k];
            a3 += Xs[k][tx] * Xs[k][tx];
            a4 += Ys[ty][k] * Ys[ty][k];
            a5 += Xs[k][tx] * Ys[ty][k];
        }
        __syncthreads();
    }
    avgX = a1/m;
    avgY = a2/m;
    varX = (a3-avgX*avgX*m)/(m-1);
    varY = (a4-avgY*avgY*m)/(m-1);
    cov = (a5-avgX*avgY*m)/(m-1);
    rho = cov/sqrtf(varX*varY);
    o = by*16*n + ty*n + bx*16 + tx;
    out[o] = rho;
}""")

    gpucorr = mod.get_function("gpuRho")
    return gpucorr
#    corr = np.array((in_data.shape[0],)*2)
#    n,m = in_data.shape
#    gpucorr(drv.Out(corr.astype(np.float32)), drv.In(in_data),
#            drv.In(n), drv.In(m))
#    return corr
