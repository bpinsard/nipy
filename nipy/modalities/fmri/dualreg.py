import numpy as np


def dual_regression(data, templates, mask, data2=None, mask2=None):
    mask = mask>0
    
    ntpl = templates.shape[3]
    mtpl = templates[mask,:].astype(np.float32)
    mtpl[np.isnan(mtpl)]=0
    #reshape and mask data
    mdata = data[mask,:].astype(np.float32)
    #normalize data
    mdata = mdata-np.tile(mdata.mean(axis=1),(data.shape[3],1)).T
    mdata = mdata/np.tile(mdata.std(axis=1),(data.shape[3],1)).T
    mdata[np.isnan(mdata)] = 0
    tcs = np.linalg.pinv(mtpl).dot(mdata)
    
    if data2==None or mask2==None:
        mapv = np.linalg.pinv(tcs.T).dot(mdata.T)
        #rectonstruct
        maps = np.zeros(templates.shape)
        for ti in range(0,ntpl):
            tmp = np.zeros(mask.shape)
            tmp[mask] = mapv[ti,:]
            maps[:,:,:,ti] = tmp
    else: #reconstruct from data in different space
        mask2 = mask>0
        mdata2 = data2[mask2,:].astype(np.float32)
        #normalize data
        mdata2 = mdata2-np.tile(mdata2.mean(axis=1),(data2.shape[3],1)).T
        mdata2 = mdata2/np.tile(mdata2.std(axis=1),(data2.shape[3],1)).T
        mdata2[np.isnan(mdata2)] = 0
        
        mapv = np.linalg.pinv(tcs.T).dot(mdata2.T)
        #reconstruct
        maps = np.zeros(mask2.shape+(ntpl,))
        for ti in range(0,ntpl):
            tmp = np.zeros(mask2.shape)
            tmp[mask2] = mapv[ti,:]
            maps[:,:,:,ti] = tmp
    return maps,tcs
