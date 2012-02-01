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
    for i in n.nonzero()[0]:
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
        for ts in data:
            beta = reg_pinv.dot(ts)
            ts -= regressors.dot(beta[:-1])
            
    else:
        motion_mats = np.array([regaff.matrix44(m) for m in motion_in])
        
        indices = np.nonzero(mask>0)
        #homogeneous indices
        indices = np.concatenate((indices,np.ones((1,indices[0].shape[0]))))
        world_coords = nii.get_affine().dot(indices)
        voxels_motion = np.array([m.dot(world_coords) for m in motion_mats])
        voxels_motion = voxels_motion.transpose((2,0,1))
        
        if regressors_type == 'voxelwise_translation':
            regressors = voxels_motion[...,:3]
        elif regressors_type == 'voxelwise_drms':
            regressors = np.sqrt(np.square(voxels_motion[...,:3]).sum(axis=2))
        elif regressors_type == 'voxelwise_outplane':
            regressors = np.dot(np.linalg.inv(nii.get_affine()),
                                voxels_motion.transpose((0,2,1)))[slicing_axis]
        if regressors.ndim < 3:
            regressors = regressors.reshape(regressors.shape+(1,))
        regsh = regressors.shape
        if regressors_transform == 'bw_derivatives':
            regressors = np.concatenate(
                (np.zeros((regsh[0],1,regsh[2])),
                 np.diff(regressors, axis=1)), axis=1)
        elif regressors_transform == 'fw_derivatives':
            regressors = np.concatenate(
                (np.diff(regressors, axis=1),
                 np.zeros((regsh[0],1,regsh[2]))), axis=1)

        for ts,reg in zip(data,regressors):
            if global_signal:
                reg = np.concatenate((reg,gs_tc),1)
            if np.count_nonzero(np.isnan(reg))>0:
                raise ValueError("regressors contains NaN")
            reg_pinv = np.linalg.pinv(np.concatenate((reg,np.ones((nt,1))),1))
            beta = reg_pinv.dot(ts)
            ts -= reg.dot(beta[:-1])

    cdata = np.empty(nii.shape)
    cdata.fill(np.nan)
    cdata[mask] = data
    return cdata, regressors


def scrubbing_badframes(data,motion,mask,
                        head_radius = 50):
    data = data[mask]
    
    dvars = np.square(np.diff(data,1)).mean(0)
    rotation_mm = np.sin(motion[:,3:6])*head_radius
    fd = np.diff(np.concatenate((motion[:,0:3],rotation_mm),1),axis=0).sum(1)
    
    return fd,dvars

def scrub_data(nii,motion,mask,
               head_radius = 50,
               motion_thresh = -1,
               variance_thresh = -1):
    data = nii.get_data()
    fd,dvars=scrubbing_badframes(data,motion,mask,head_radius)

    if motion_thresh < 0:
        vox_size = np.sqrt(np.square(nii.get_affine()[:3,:3]).sum(0)).mean()
        motion_thresh = vox_size/2
#    if variance_thresh < 0:
        
