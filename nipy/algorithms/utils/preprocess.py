import numpy as np
import nipy.algorithms.registration.affine as regaff

def motion_parameter_standardize(motion_in,motion_source='spm'):

    if motion_source == 'spm':
        return motion_in
    elif motion_source == 'fsl':
        return np.concatenate((motion_in[:,3:],motion_in[:,:3]),axis=1)
    elif motion_source == 'afni':
        return np.concatenate((motion_in[:,3:6],motion_in[:,:3]/180*np.pi),
                                axis=1)

def regress_out_motion_parameters(nii,motion_in,mask,
                                  regressors_type = 'global',
                                  regressors_transform = 'bw_derivatives',
                                  slicing_axis = 2,
                                  global_signal = False):
    nt = motion_in.shape[0]

    data = nii.get_data()
    mask = mask>0 
    data = data[mask]
    m = np.isnan(data).sum(1)

    #correct for isolated nan values in mask timeseries due to realign
    # linearly interpolate in ts and extrapolate at ends of ts
    # TODO:optimize
    y = lambda z: z.nonzero()[0]
    for i in m.nonzero()[0]:
        nans = np.isnan(data[i])
        data[i] = np.interp(y(nans),y(~nans),data[i,~nans])
    
    if global_signal:
        gs_tc = data.mean(0).reshape((nt,1))
        gs_tc -= gs_tc.mean() #center
        gs_tc /= gs_tc.var() #reduce
        
    if regressors_type == 'global':

        if regressors_transform == 'bw_derivatives':
            regressors = np.concatenate((
                np.ones((1,6)),
                np.diff(motion_in,axis=0)))
        elif regressors_transform == 'fw_derivatives':
            regressors = np.concatenate((
                np.diff(motion_in,axis=0),
                np.ones((1,6))))
        else:
            regressors = motion_in

        if global_signal:
            regressors = np.concatenate((regressors,gs_tc),1)

        reg_pinv = np.linalg.pinv(np.concatenate((regressors,np.ones((nt,1))),
                                                 axis=1))

        betas = np.empty((data.shape[0], regressors.shape[1]))
        for idx,ts in enumerate(data):
            betas[idx] = reg_pinv.dot(ts)[:-1]
            ts -= regressors.dot(betas[idx])
            
    else:
        
        voxels_motion = compute_voxels_motion(motion_in,mask,nii.get_affine())

#        motion_mats = np.array([regaff.matrix44(m) for m in motion_in])
#        indices = np.nonzero(mask>0)
        #homogeneous indices
#        indices = np.concatenate((indices,np.ones((1,indices[0].shape[0]))))
#        world_coords = nii.get_affine().dot(indices)
#        voxels_motion = np.array([m.dot(world_coords) for m in motion_mats])
#        voxels_motion = voxels_motion.transpose((2,0,1))
        
        if regressors_type == 'voxelwise_translation':
            regressors = voxels_motion[...,:3]
        elif regressors_type == 'voxelwise_drms':
            regressors = np.sqrt(np.square(voxels_motion[...,:3]).sum(axis=2))
        elif regressors_type == 'voxelwise_outplane':
            regressors = np.dot(np.linalg.inv(nii.get_affine()),
                                voxels_motion.transpose((0,2,1)))[slicing_axis]
        if regressors.ndim < 3:
            regressors = regressors[:,:,np.newaxis]
        regsh = regressors.shape
        if regressors_transform == 'bw_derivatives':
            regressors = np.concatenate(
                (np.zeros((regsh[0],1,regsh[2])),
                 np.diff(regressors, axis=1)), axis=1)
        elif regressors_transform == 'fw_derivatives':
            regressors = np.concatenate(
                (np.diff(regressors, axis=1),
                 np.zeros((regsh[0],1,regsh[2]))), axis=1)

        betas=np.empty((data.shape[0],regressors.shape[2]+int(global_signal)))

        for beta,ts,reg in zip(betas,data,regressors):
            if global_signal:
                reg = np.concatenate((reg,gs_tc),1)
            if np.count_nonzero(np.isnan(reg))>0:
                raise ValueError("regressors contains NaN")
            reg_pinv = np.linalg.pinv(np.concatenate((reg,np.ones((nt,1))),1))
            beta[...] = reg_pinv.dot(ts)[:-1]
            ts -= reg.dot(beta)

    cdata = np.empty(nii.shape, np.float)
    cdata.fill(np.nan)
    cdata[mask] = data
    betamaps = np.empty(nii.shape[:-1]+(betas.shape[-1],), np.float)
    betamaps.fill(np.nan)
    betamaps[mask] = betas
    return cdata, regressors, betamaps


def compute_voxels_motion(motion, mask, affine):
    motion_mats = np.array([regaff.matrix44(m) for m in motion])
    
    indices = np.nonzero(mask>0)
    #homogeneous indices
    indices = np.concatenate((indices,np.ones((1,indices[0].shape[0]))))
    world_coords = affine.dot(indices)
    voxels_motion = np.array([m.dot(world_coords) for m in motion_mats])
    voxels_motion = voxels_motion.transpose((2,0,1))
    
    return voxels_motion

def scrubbing_badframes(data, motion, mask,
                        head_radius = 50):
    data = data[mask].astype(np.float32)
    
    dvars=np.empty(data.shape[-1])
    dvars[0]=0
    fd = np.empty(data.shape[-1])
    fd[0]=0
    dvars[1:] = np.sqrt(np.square(np.diff(data,axis=-1)).mean(0))
    rotation_mm = np.sin(motion[:,3:6])*head_radius
    fd[1:] = np.diff(np.concatenate((motion[:,0:3],rotation_mm),1),axis=0).sum(1)  
    return fd,dvars

def otsu(data):
    # 1d Otsu , is that really Otsu??
    sdata=np.sort(data)
    n=sdata.size
    amin = np.argmin(np.linspace(0,1,n)*(((sdata**2).cumsum()-sdata.cumsum()**2/(n**2))/n) + np.linspace(1,0,n)*(((sdata[::-1]**2).cumsum()-sdata[::-1].cumsum()**2/(n**2))/n)[::-1])
    return sdata[amin]

def scrub_data(nii,motion,mask,
               head_radius = 50,
               fd_threshold = -1,
               drms_threshold = -1,
               extend_mask = True):
    data = nii.get_data()
    fd,dvars = scrubbing_badframes(data, motion, mask, head_radius)

    if fd_threshold < 0:
        vox_size = np.sqrt(np.square(nii.get_affine()[:3,:3]).sum(0)).mean()
        fd_threshold = vox_size/2
    if drms_threshold < 0:
        print dvars.std(), dvars.mean()
        drms_threshold = otsu(dvars)
        print 'Threshold DRMS %f' % drms_threshold
        #TODO: check how this perform in good runs
    fd_mask = (fd<fd_threshold)
    drms_mask = (dvars<drms_threshold)
    
    # as in Power et al. extend mask to 1 back and 2 forward frames
    if extend_mask:
        fd_mask[np.where(fd_mask[1:])]=True
        fd_mask[np.where(fd_mask[:-2])[0]+2]=True
        drms_mask[np.where(drms_mask[1:])]=True
        drms_mask[np.where(drms_mask[:-2])[0]+2]=True

    scrub_mask = fd_mask & drms_mask

    scrubbed = data[...,scrub_mask]
    return scrubbed, scrub_mask, fd_threshold, drms_threshold
