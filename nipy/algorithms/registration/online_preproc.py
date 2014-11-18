import numpy as np

import nibabel as nb, dicom
from nibabel.affines import apply_affine
from ...fixes.nibabel import io_orientation
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .affine import Rigid, rotation_vec2mat

from .optimizer import configure_optimizer, use_derivatives
from scipy.ndimage import convolve1d, gaussian_filter, binary_erosion, binary_dilation
import scipy.stats, scipy.sparse
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import LinearNDInterpolator
from .slice_motion import surface_to_samples, compute_sigloss, intensity_factor


import time
import itertools

# Module globals
SLICE_ORDER = 'ascending'
INTERLEAVED = None
OPTIMIZER = 'powell'
XTOL = 1e-5
FTOL = 1e-5
GTOL = 1e-5
STEPSIZE = 1e-2
SMALL = 1e-20
MAXITER = 64
MAXFUN = None
EXTRAPOLATE_SPACE = 'zero'
EXTRAPOLATE_TIME = 'nearest'

class EPIOnlineResample(object):
    def __init__(self,
                 fieldmap,
                 fieldmap_reg,
                 mask=None,
                 phase_encoding_dir = 1,
                 repetition_time = 3.0,
                 slice_repetition_time = None,
                 echo_time = 0.03,
                 echo_spacing = 0.005,
                 slice_order = None,
                 interleaved = 0,
                 slice_trigger_times=None,
                 slice_thickness=None,
                 slice_axis=2,):
        self.fmap, self.mask = fieldmap, mask
        self.fieldmap_reg = fieldmap_reg
        if self.fmap is not None:
            if self.fieldmap_reg is None:
                self.fieldmap_reg = np.eye(4)
            self.fmap2world = np.dot(self.fieldmap_reg, self.fmap.get_affine())
            self.world2fmap = np.linalg.inv(self.fmap2world)
        self.slice_axis = slice_axis
        self.slice_order = slice_order
        self.pe_sign = int(phase_encoding_dir > 0)*2-1
        self.pe_dir = abs(phase_encoding_dir)-1
        self.repetition_time = repetition_time
        self.echo_time = echo_time
        self.slice_tr = slice_repetition_time
        self.interleaved = int(interleaved)
        self.slice_trigger_times = slice_trigger_times
        self.slice_thickness = slice_thickness

        self.fmap_scale = self.pe_sign*echo_spacing/2.0/np.pi
        self._resample_fmap_values = None
        self.st_ratio = 1


    def resample(self, data, out, voxcoords):
        out[:] = map_coordinates(data, np.rollaxis(voxcoords,-1,0)).reshape(voxcoords.shape[:-1])
        return out

    def scatter_resample_volume(self, data, out, slabs, transforms, target_transform, mask=False):
        coords = apply_affine(
            target_transform,
            np.rollaxis(np.mgrid[[slice(0,d) for d in out.shape]],0,4))
        self.scatter_resample(data, out, transforms, coords, mask=mask)
        del coords

    def scatter_resample(self, data, out, slabs, transforms, coords, mask=True):
        nslices = data[0].shape[2]*len(slabs)
        vol = np.empty(data[0].shape[:2]+(nslices,))
        print vol.shape
        points = np.empty(vol.shape+(3,))
        voxs = np.rollaxis(np.mgrid[[slice(0,d) for d in vol.shape]],0,4)
        print voxs.shape
        phase_vec = np.zeros(3)
        for sl, d in zip(slabs, data):
            vol[...,sl] = d
        for sl, t in zip(slabs, transforms):
            points[:,:,sl] = apply_affine(t, voxs[:,:,sl])
            phase_vec+= t[:3,self.pe_dir]
        phase_vec /= len(slabs)
        epi_mask = slice(0, None)
        if mask:
            epi_mask = self.inv_resample(self.mask, transforms[0], vol.shape, 0)>0
            epi_mask[:] = binary_dilation(epi_mask, iterations=2)
            points = points[epi_mask]
        if not self.fmap is None:
            self._precompute_sample_fmap(coords, vol.shape)
            coords += self._resample_fmap_values[:,np.newaxis].dot(phase_vec[np.newaxis])
        print 'create interpolator'
        lndi = LinearNDInterpolator(points.reshape(-1,3), vol[epi_mask].ravel())
        print 'interpolate'
        out[:] = lndi(coords.reshape(-1,3)).reshape(out.shape)

    def _precompute_sample_fmap(self, coords, shape):
        if self.fmap is None:
            return
        # if coords or shape changes, recompute values
        if not self._resample_fmap_values is None and \
                shape == self._resample_shape_cache and \
                coords is self._resample_coords_cache:
            return
        del self._resample_fmap_values
        interp_coords = apply_affine(self.world2fmap, coords)
        self._resample_coords_cache = coords
        self._resample_shape_cache = shape
        self._resample_fmap_values = map_coordinates(
            self.fmap.get_data(),
            interp_coords.reshape(-1,3).T,
            order=1).reshape(interp_coords.shape[:-1])
        self._resample_fmap_values *= self.fmap_scale*shape[self.pe_dir]
        del interp_coords

    def resample_coords(self, data, affines, coords, out):

        if not hasattr(self,'nslices'):
            self.nslices = data.shape[self.slice_axis]

        self._precompute_sample_fmap(coords,data.shape)
        interp_coords = np.empty(coords.shape)
            
        tmp = np.empty(coords.shape[:-1])
        if len(affines) == 1: #easy, one transform per volume
            wld2epi = np.linalg.inv(affines[0][1])
            interp_coords[:] = apply_affine(wld2epi, coords)
            if not self._resample_fmap_values is None:
                interp_coords[...,self.pe_dir] += self._resample_fmap_values
        else: # we have to solve which transform we sample with
            t = affines[0][0][1][0]
            tmp_coords = np.empty(coords.shape)
            subset = np.ones(coords.shape[:-1], dtype=np.bool)
            interp_coords.fill(np.nan) # just to check, to be removed
            for slab,trans in affines:
                wld2epi = np.linalg.inv(trans)
                tmp_coords[:] = apply_affine(wld2epi, coords)
                if self.fmap != None:
                    tmp_coords[...,self.pe_dir] += self._resample_fmap_values
                    
                if slab[0][0]==t and slab[1][0]==t:
                    times = np.arange(slab[0][1],slab[1][1])
                elif slab[0][0]==t:
                    times = np.arange(slab[0][1], self.nslices)
                elif slab[1][0]==t:
                    times = np.arange(0, slab[1][1]+1)
                else:
                    times = np.arange(0, self.nslices)
                interp_coords[subset] = tmp_coords[subset]
                subset[np.any(
                    np.abs(tmp_coords[...,self.slice_axis,np.newaxis]-\
                           self.slice_order[times][np.newaxis]) <\
                        self.st_ratio+.1, -1)] = False
            del tmp_coords, subset
        if np.count_nonzero(np.isnan(interp_coords)) > 0:
            raise RuntimeError # just to check, to be removed
        self.resample(data, out, interp_coords)
        del interp_coords, tmp

    def _epi_inv_shiftmap(self, affine, shape):
        # compute inverse shift map using approximate nearest neighbor
        #
        # caching
        if hasattr(self,'_inv_shiftmap'):
            if self._inv_shiftmap.shape == shape\
                    and np.allclose(self._invshiftmap_affine,affine):
                return self._inv_shiftmap
            else:
                del self._inv_shiftmap
        self._invshiftmap_affine = affine

        fmap2fmri = np.linalg.inv(affine).dot(self.fmap2world)
        coords = nb.affines.apply_affine(
            fmap2fmri,
            np.rollaxis(np.mgrid[[slice(0,s) for s in self.fmap.shape]],0,4))
        shift = self.fmap_scale * self.fmap.get_data() * shape[self.pe_dir]
        coords[...,self.pe_dir] += shift
        self._inv_shiftmap = np.empty(shape)
        #inv_shiftmap_dist = np.empty(shape)
        self._inv_shiftmap.fill(np.inf)
        #inv_shiftmap_dist.fill(np.inf)
        includ = np.logical_and(
            np.all(coords>-.5,-1),
            np.all(coords<np.array(shape)[np.newaxis]-.5,-1))
        coords = coords[includ]
        rcoords = np.round(coords).astype(np.int)
        #dists = np.sum((coords-rcoords)**2,-1)
        shift = shift[includ]
        self._inv_shiftmap[(rcoords[...,0],rcoords[...,1],rcoords[...,2])] = -shift
        for x,y,z in zip(*np.where(np.isinf(self._inv_shiftmap))):
            ngbd = self._inv_shiftmap[
                max(0,x-1):x+1,max(0,y-1):y+1,max(0,z-1):z+1]
            self._inv_shiftmap[x,y,z] = ngbd.ravel()[np.argmin(np.abs(ngbd.ravel()))]
            del ngbd
        del includ, coords, rcoords, shift
        return self._inv_shiftmap
            
    def inv_resample(self, vol, affine, shape, order=0, mask=slice(None)):
        ## resample a undistorted volume to distorted EPI space
        # order = map_coordinates order, if -1, does integral of voxels in the
        # higher resolution volume (eg. for partial volume map downsampling)
        if order < 0:
            grid = np.mgrid[[slice(0,s) for s in vol.shape[:3]]][:,mask].reshape(3,-1).T
            vol2epi = np.linalg.inv(affine).dot(vol.get_affine())
            voxs = nb.affines.apply_affine(vol2epi, grid)
            if self.fmap is not None:
                vol2fmap = self.world2fmap.dot(vol.get_affine())
                fmap_voxs = nb.affines.apply_affine(vol2fmap, grid)
                fmap_values = self.fmap_scale * map_coordinates(
                    self.fmap.get_data(),
                    fmap_voxs.T,
                    order=1).reshape(fmap_voxs.shape[:-1])
                voxs[:, self.pe_dir] += fmap_values
                del fmap_voxs, fmap_values
            np.round(voxs, out=voxs)
            nvols = (vol.shape+(1,))[:4][-1]
            rvol = np.empty(shape+(nvols,))
            bins = [np.arange(-.5,d+.5) for d in shape]
            data = vol.get_data()[mask]
            if len(vol.shape) < 4:
                data = data[...,np.newaxis]
            for v in range(nvols):
                rvol[...,v] = np.histogramdd(voxs, bins, weights=data[...,v].ravel())[0]
            if order == -1: #normalize
                counts, _ = np.histogramdd(voxs,bins)
                rvol /= counts[...,np.newaxis]
                del counts
            rvol[np.isnan(rvol)] = 0
            rvol = np.squeeze(rvol)
            del bins
        else:
            grid = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]], 0, 4)
            if self.fmap is not None:
                inv_shift = self._epi_inv_shiftmap(affine, shape)
                grid[...,self.pe_dir] += inv_shift
            epi2vol = np.linalg.inv(vol.get_affine()).dot(affine)
            voxs = nb.affines.apply_affine(epi2vol, grid)
            rvol = map_coordinates(
                vol.get_data(),
                voxs.reshape(-1,3).T, order=order).reshape(shape)
        del grid, voxs
        return rvol
    

class EPIOnlineRealign(EPIOnlineResample):

    def __init__(self,

                 bnd_coords,
                 class_coords,

                 fieldmap = None,
                 fieldmap_reg=None,
                 mask = None,

                 phase_encoding_dir = 1,
                 repetition_time = 3.0,
                 slice_repetition_time = None,
                 echo_time = 0.03,
                 echo_spacing = 0.0005,
                 slice_order = None,
                 interleaved = 0,
                 slice_trigger_times = None,
                 slice_thickness = None,
                 slice_axis = 2,
                 
                 detection_threshold=.99,
                 motion_regularization = 0,#1e-3,
                 init_reg = None,
                 affine_class=Rigid,
                 optimizer=OPTIMIZER,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 nsamples_per_slab=2000,
                 min_nsamples_per_slab=100,
                 bbr_slope=1,
                 bbr_offset=0):

        super(EPIOnlineRealign,self).__init__(
                 fieldmap,fieldmap_reg,
                 mask,
                 phase_encoding_dir ,
                 repetition_time,
                 slice_repetition_time,
                 echo_time,
                 echo_spacing,
                 slice_order,
                 interleaved,
                 slice_trigger_times,
                 slice_thickness,
                 slice_axis)


        self.bnd_coords, self.class_coords = bnd_coords, class_coords
        self.border_nvox = self.bnd_coords.shape[0]

        self.detection_threshold = detection_threshold
        self.nsamples_per_slab = nsamples_per_slab
        self.min_sample_number = min_nsamples_per_slab        
        self.affine_class = affine_class
        self.init_reg = init_reg
        self.bbr_slope, self.bbr_offset = bbr_slope, bbr_offset

        self._motion_regularization = motion_regularization

        # compute fmap values on the surface used for realign
        if self.fmap != None:
            fmap_vox = apply_affine(self.world2fmap,
                                    self.class_coords.reshape(-1,3))
            self.fmap_values = self.fmap_scale * map_coordinates(
                self.fmap.get_data(), fmap_vox.T,
                order=1).reshape(2,self.border_nvox)
            del fmap_vox
        else:
            self.fmap_values = None


        # Set the minimization method
        self.set_fmin(
            optimizer, stepsize,
            xtol=xtol, ftol=ftol, gtol=gtol,
            maxiter=maxiter, maxfun=maxfun)


        self.slab_class_voxels = np.empty(self.class_coords.shape,np.double)


    def center_of_mass_init(self, data1):
        # simple center of mass alignment for initialization
        mdata = self.mask.get_data()
        c_ref = np.squeeze(
            np.apply_over_axes(
                np.sum,(np.mgrid[[slice(0,s) for s in mdata.shape]]*
                        mdata[np.newaxis]),[1,2,3])/float(mdata.sum()))
        c_frame1 = np.squeeze(
            np.apply_over_axes(
                np.sum,(np.mgrid[[slice(0,s) for s in data1.shape]]*
                        data1[np.newaxis]),[1,2,3])/float(data1.sum()))
        tr = self.mask.get_affine().dot(c_ref.tolist()+[1])-\
            self.affine.dot(c_frame1.tolist()+[1])
        return np.hstack([tr[:3],[0]*3])

    def process(self, stack, yield_raw=False):

        self.slabs = []
        self.transforms = []
        self._last_subsampling_transform = self.affine_class(np.ones(12)*5)
        
        # dicom list : slices must be provided in acquisition order

        # register first frame
        frame_iterator = stack.iter_frame(queue_dicoms=True)
        nvol, self.affine, data1 = frame_iterator.next()
        data1 = data1.astype(np.float)
        self.slice_order = stack._slice_order
        inv_slice_order = np.argsort(self.slice_order)
        self.nslices = stack.nslices
        last_reg = self.affine_class()
        if self.init_reg is not None:
            if self.init_reg == 'auto':
                last_reg.param = self.center_of_mass_init(data1)
            else:
                last_reg.from_matrix44(self.init_reg)
        self._n_samples = self.slab_class_voxels.shape[1]
        self._slab_slice_mask = np.empty(self._n_samples, np.int8)
#        self._reliable_samples = np.ones(self._n_samples, np.bool)
        self._reg_samples = np.empty((2,self._n_samples))
        self._cost = np.empty(self._n_samples)
        self._samples_mask = np.empty(self._n_samples, dtype=np.bool)

        last_reg.param = self._register_slab(range(self.nslices), data1, last_reg, whole_frame=True)
        # compute values for initial registration
        self.apply_transform(
            last_reg,
            self.class_coords, self.slab_class_voxels,
            self.fmap_values, phase_dim=data1.shape[self.pe_dir])

        
        self.resample(data1, self._reg_samples, self.slab_class_voxels)

        # remove samples that does not have expected contrast (ie sigloss)
        # if fieldmap provided does it using the computed sigloss
        # otherwise remove the samples with inverted contrast
        """
        if not self.fmap is None:
            grid = apply_affine(
                np.linalg.inv(self.mask.get_affine()).dot(self.fmap2world),
                np.rollaxis(np.mgrid[[slice(0,n) for n in self.fmap.shape]],0,4))
            fmap_mask = map_coordinates(
                self.mask.get_data(),
                grid.reshape(-1,3).T, order=0).reshape(self.fmap.shape)
            self._samples_sigloss = compute_sigloss(
                self.fmap, self.fieldmap_reg,
                fmap_mask,
                last_reg.as_affine(), self.affine,
                self.class_coords.reshape(-1,3),
                self.echo_time, slicing_axis=self.slice_axis).reshape(2,-1)

            self._reliable_samples[:] = np.logical_and(
                np.all(self._samples_sigloss>.8,0),
                np.all(self._reg_samples>0,0))
        else:
            self._reliable_samples[:] = np.logical_and(
                np.squeeze(np.diff(self._reg_samples,1,0)) < 0,
                np.all(self._reg_samples>0,0))
        """

#        last_reg.param = self._register_slab(range(self.nslices), data1, last_reg, whole_frame=True)
                
#        self._reg_samples = self._reg_samples[:,self._reliable_samples]
#        self.slab_class_voxels = self.slab_class_voxels[:,self._reliable_samples]
#        self.fmap_values = self.fmap_values[:,self._reliable_samples]
#        del self._slab_slice_mask, self._cost, self._samples_mask
#        self._n_samples = np.count_nonzero(self._reliable_samples)
#        self._slab_slice_mask = np.empty(self._n_samples, np.int8)
        self._samples = np.empty((7,3,self._n_samples))
        self._samples_mask = np.empty(self._n_samples,dtype=np.bool)
        self._samples_dist = np.empty((7,2,self._n_samples))
        self._reg_cost = np.empty(self._n_samples)
        self._cost = np.empty((7,self._n_samples))
        self.optimizer='cg'


#        n_samples_total = np.count_nonzero(self._reliable_samples)
        # reestimate first frame registration  with only reliable samples
 
        self.full_frame_reg = last_reg.copy()
        
        slice_axes = np.ones(3, np.bool)
        slice_axes[self.slice_axis] = False
        slice_spline = np.empty(stack._shape[:2])

        ndim_state = 6
        transition_matrix = np.eye(ndim_state)
        transition_covariance = np.diag([.01]*6+[.1]*6) # change in position should first occur by change in speed !?
        transition_covariance = np.eye(6)*1e-3
        if ndim_state>6:
            transition_matrix[:6,6:] = np.eye(6) # TODO: set speed
            # transition_covariance[:6,6:] = np.eye(6)
        transition_covariance = transition_covariance[:ndim_state,:ndim_state]

        initial_state_mean = np.hstack([last_reg.param.copy(), np.zeros(6)])
        initial_state_covariance = np.eye(ndim_state)*.001 # let say we are quite confident about init volume registration
        initial_state_mean = initial_state_mean[:ndim_state]
        initial_state_covariance = initial_state_covariance[:ndim_state,:ndim_state]
        
        # R the (co)variance (as we suppose white observal noise)
        # this could be used to weight samples
        observation_variance_inv = np.ones(self._n_samples)#/self._n_samples

        observation_variance_inv = (1-np.diff(self._reg_samples,1,0)[0]/self._reg_samples.sum(0))*.1
        observation_variance_inv[np.isnan(observation_variance_inv)]=0
        observation_variance_inv[observation_variance_inv<0]=0

        stack_it = stack.iter_slabs()
        stack_has_data = True
        current_volume = data1.copy()
        fr,sl,aff,tt,sl_data = stack_it.next()
        
        self.filtered_state_means = [initial_state_mean]
        self.filtered_state_covariances = [initial_state_covariance]

        self.raw_motion = [last_reg.param[:6].copy()]
        
        new_reg = last_reg.copy()

        convergence_threshold = 1e-10
        niter_max = 1
        mm = self._samples_mask

        while stack_has_data:
            
#            for si,s in enumerate(sl):
#                current_volume[...,s] = sl_data[...,si]
            
            # forward prediction, in the 6 param case, identity
            pred_state = transition_matrix.dot(self.filtered_state_means[-1])
            estim_state = pred_state.copy()
            pred_covariance = self.filtered_state_covariances[-1] + transition_covariance
            state_covariance = pred_covariance.copy()

            convergence, niter = np.inf, 0
            self.sample_cost_jacobian(sl, sl_data, new_reg, force_recompute_subset=True)
            if self._n_slab_samples < 50:
                print 'not enough points, skipping slab'
            else:
                while convergence > convergence_threshold and niter < niter_max:
                    new_reg.param = estim_state[:6]
                    self.sample_cost_jacobian(sl, sl_data, new_reg)
#                    mm[:] = self._slab_slice_mask>=0
                    cost, jac = self._cost[0], self._cost[1:]
                    mm[:] = self._slab_slice_mask>=0 # this might be redundant, allready updated in sample_cost_jacobian
                    mm[mm] = observation_variance_inv[mm]!=0
                    mm[mm] = np.any(jac[:,mm]!=0,0)
#                    wjac = jac[:,mm]*observation_variance_inv[mm]
                    #wjac = jac*observation_variance_inv
                    #kalman_gain = np.linalg.inv(wjac.dot(jac.T) + state_covariance).dot(wjac)
                    kalman_gain = state_covariance.dot(jac[:,mm]).dot(np.linalg.inv(jac[:,mm].T.dot(state_covariance).dot(jac[:,mm])+np.eye(np.count_nonzero(mm))*observation_variance_inv[mm]))
                    print kalman_gain.shape
                    estim_state_old = estim_state.copy()
                    estim_state[:] = pred_state + kalman_gain.dot(cost[mm]-jac[:,mm].T.dot(pred_state-estim_state))
#                    estim_state[:] = pred_state + kalman_gain.dot(cost - jac.T.dot(pred_state-estim_state) )
#                    state_covariance[:] = (np.eye(ndim_state)-kalman_gain.dot(jac.T)).dot(state_covariance)
                    convergence = np.sqrt(((estim_state_old-estim_state)**2).sum())
                    niter += 1
                    print ("%.12f : "+"%.5f,\t"*6)%((convergence,) +tuple(estim_state[:6]))
                    if np.abs(convergence)>10:
                        raise RuntimeError
                if niter==niter_max:
                    print "maximum iteration number exceeded"

                state_covariance[:] = (np.eye(ndim_state)-kalman_gain.dot(jac[:,mm].T)).dot(self.filtered_state_covariances[-1])
#                state_covariance[:] = (np.eye(ndim_state)-kalman_gain.dot(jac.T)).dot(state_covariance)

            self.filtered_state_means.append(estim_state)
            self.filtered_state_covariances.append(state_covariance)
            print '_'*80
            print ('%5f :\t' + '%.5f,'*6)%((self._cost[0].mean(),)+tuple(estim_state))
            print '_'*80
            new_reg.param = estim_state[:6]
            yield fr, sl, new_reg.as_affine().dot(aff), sl_data
            try:
                fr,sl,aff,tt,sl_data = stack_it.next()
            except StopIteration:
                stack_has_data = False

    def _register_slab(self, slab, slab_data, init_transform, whole_frame=False):
        print 'register slab', slab
        
        transform = self.affine_class(init_transform.as_affine())
        self.sample_cost(slab, slab_data, transform,
                         force_recompute_subset=True,
                         whole_frame=whole_frame)
        if not whole_frame:
            print " %d samples "%self._n_slab_samples
            if self._n_slab_samples < 30:
                print 'not enough points, skipping slab'
                return transform.param

        def f(pc):
            transform.param = pc
            nrgy = self.sample_cost(slab, slab_data, transform, whole_frame=whole_frame)
            print 'f %.10f : %f %f %f %f %f %f'%tuple([nrgy] + pc.tolist())
            return nrgy
        self._pc = None
        fmin, args, kwargs = configure_optimizer(self.optimizer,
                                                 **self.optimizer_kwargs)
        pc = fmin(f, transform.param, *args, **kwargs)

        ## TODO: update the self._reg_samples
        if not whole_frame:
            transform.param = pc
            #nrgy = self.sample_cost(slab, slab_data, transform,
            #                        force_recompute_subset=True,
            #                        update_reg_samples=True)
        return pc

    def apply_transform(self, transform, in_coords, out_coords,
                        fmap_values=None, subset=slice(None), phase_dim=64):
        ref2fmri = np.linalg.inv(transform.as_affine().dot(self.affine))
        #apply current slab transform
        out_coords[...,subset,:] = apply_affine(ref2fmri, in_coords[...,subset,:])
        #add shift in phase encoding direction
        if fmap_values != None:
            out_coords[...,subset,self.pe_dir]+=fmap_values[...,subset]*phase_dim
    
    def sample_cost_jacobian(self, slab, data, transform, sigma=.2, force_recompute_subset=False):
        # compute the cost and the jacobian for a given set of params

        sa = self.slice_axis
        slice_axes = np.ones(3, np.bool)
        slice_axes[sa] = False

        epsilon = 1e-5
        extend, resolution = 2,1000

        mm = self._samples_mask
        mm[:] = self._slab_slice_mask>=0

        test_points = np.array([(x,y,z) for x in (0,data.shape[0]) for y in (0, data.shape[1]) for z in (0,self.nslices)])
        recompute_subset = np.abs(
            self._last_subsampling_transform.apply(test_points) -
            transform.apply(test_points))[:,sa].max() > 0.01
        
        if recompute_subset or force_recompute_subset:
            self.apply_transform(
                transform,
                self.class_coords, self.slab_class_voxels,
                self.fmap_values, phase_dim=data.shape[self.pe_dir])
            self._last_subsampling_transform = transform.copy()
            self._slab_slice_mask.fill(-1)
            self._samples_dist.fill(0)
            for s in slab:
                mm[:] = np.any(np.abs(self.slab_class_voxels[..., self.slice_axis]-s)<.5,0)
                self._slab_slice_mask[mm] = s                
            mm[:] = self._slab_slice_mask >= 0
            sm = self._reg_samples.sum(0)
            self._cost.fill(0)
            self._cost[0] = 200*np.diff(self._reg_samples,1,0)/sm
            self._cost[0,np.abs(sm)<1e-6] = 0
            self._cost[0] = 1+np.tanh(self.bbr_slope*self._cost[0]-self.bbr_offset)
            self._reg_cost[:] = self._cost[0]

            del sm
            self._n_slab_samples = np.count_nonzero(mm)
            print 'new subset %d'%self._n_slab_samples
        else:
            self.apply_transform(
                transform,
                self.class_coords,
                self.slab_class_voxels,
                self.fmap_values, mm,
                phase_dim=data.shape[self.pe_dir])
            
        if not hasattr(self,'_precomp_negexp'):
            self._precomp_negexp = np.exp(-np.arange(0,extend,1./resolution)**2/(2*sigma**2))

        ofst = 0
        for si, s in enumerate(slab):
            mm[:] = self._slab_slice_mask==s
            cnt = np.count_nonzero(mm)
            if cnt>0:
                crds = np.empty((3,7,2,cnt))
                crds[:,0] = self.slab_class_voxels[:,mm].transpose(2,0,1)
                for i in range(6):
                    if i<3:
                        crds[:,i+1] = crds[:,0]
                        crds[i,i+1] += epsilon * transform.precond[i]
                    if i>2:
                        vec = np.zeros(3)
                        vec[i-3] = epsilon * transform.precond[i]
                        m = rotation_vec2mat(vec)
                        crds[:,i+1] = m.dot(crds[:,0].transpose(1,0,2))
                dist2slice = (np.abs(crds[sa]-s)*resolution).astype(np.int)
                dist2slice[dist2slice>=self._precomp_negexp.size] = -1
                self._samples_dist[:,:,ofst:ofst+cnt] = self._precomp_negexp[dist2slice]
                self._samples[:,:2,ofst:ofst+cnt] = (
                    map_coordinates(data[...,si].astype(np.float),
                                    crds[slice_axes].reshape(2,-1)).reshape(7,2,-1)*self._samples_dist[...,ofst:ofst+cnt] +
                    self._reg_samples[:,mm]*(1-self._samples_dist[...,ofst:ofst+cnt]))
                self._samples[:,2,ofst:ofst+cnt] = (
                    map_coordinates(
                        data[...,si].astype(np.float),
                        crds[slice_axes].mean(2).reshape(2,-1)).reshape(7,-1)*self._samples_dist[...,ofst:ofst+cnt].mean(1) +
                    self._reg_samples[:,mm].mean(1)*(1-self._samples_dist[...,ofst:ofst+cnt].mean(1)))

                ofst += cnt
                del crds

        mm[:] = self._slab_slice_mask>=0
        sm = self._samples[...,:ofst].sum(1)
        self._cost[:,mm] = 200*np.diff(self._samples[...,:ofst],1,1)[0]/sm
        mm2 = mm.copy()
        mm2[mm2] = np.abs(sm)<1e-6
        self._cost[:,mm2] = 0
        del mm2
        self._cost[:,mm] = 1+np.tanh(self.bbr_slope*self._cost[:,mm]-self.bbr_offset)
        #compute partial derivatives
        self._cost[1:,mm] -= self._cost[0,mm]
        self._cost[1:,mm] /= epsilon
        del sm
    
    def sample_cost(self, slab, data, transform, 
                    force_recompute_subset=False, whole_frame=False,
                    update_reg_samples = False,sigma=.2):

        sa = self.slice_axis
        slice_axes = np.ones(3, np.bool)
        slice_axes[sa] = False
        mm = self._samples_mask

        # if change of test points z is above threshold recompute subset
        test_points = np.array([(x,y,z) for x in (0,data.shape[0]) for y in (0, data.shape[1]) for z in (0,self.nslices)])
        recompute_subset = np.abs(
            self._last_subsampling_transform.apply(test_points) -
            transform.apply(test_points))[:,sa].max() > 0.05
        
        if recompute_subset or force_recompute_subset or whole_frame:
            self.apply_transform(
                transform,
                self.class_coords, self.slab_class_voxels,
                self.fmap_values, phase_dim=data.shape[self.pe_dir])
            self._last_subsampling_transform = transform.copy()

            if whole_frame:
                self.resample(data, self._reg_samples, self.slab_class_voxels)
                self._init_energy()
                return self._cost.mean()+1
            else:
                self._slab_slice_mask.fill(-1)
                self._samples_dist.fill(0)
                for s in slab:
                    mm[:] = np.any(np.abs(self.slab_class_voxels[..., self.slice_axis]-s)<1,0)
                    gw = np.exp(-(self.slab_class_voxels[:,mm,self.slice_axis]-s)**2/(2*sigma**2))
#                    closer = np.all(gw > self._samples_dist[:,mm], 0)
#                    mm[mm] = closer
                    self._slab_slice_mask[mm] = s
                    self._samples_dist[:,mm] = gw#[:,closer]
                
                mm[:] = self._slab_slice_mask >= 0
                self._n_slab_samples = np.count_nonzero(mm)
                print 'new subset %d'%self._n_slab_samples
                self._reg_cost = np.sum(self._cost[np.logical_not(mm)])
        else:
            # old_coords = self.slab_class_voxels[:,self._slab_slice_mask>=0].copy()
            self.apply_transform(
                transform,
                self.class_coords,
                self.slab_class_voxels,
                self.fmap_values, self._slab_slice_mask>=0,
                phase_dim=data.shape[self.pe_dir])
            #print np.abs(old_coords - self.slab_class_voxels[:,self._slab_slice_mask>=0]).mean(1)

        #
        ofst = 0
#        old_samples = self._samples.copy()
        for si, s in enumerate(slab):
            mm[:] = self._slab_slice_mask==s
            cnt = np.count_nonzero(mm)
            if cnt>0:
                crds = self.slab_class_voxels[:,mm][...,slice_axes]
                self._samples_dist[:,mm] = np.exp(-(self.slab_class_voxels[:,mm,self.slice_axis]-s)**2/(2*sigma**2))
                self._samples[:,ofst:ofst+cnt] = map_coordinates(data[...,si].astype(np.float), crds.reshape(-1,2).T).reshape(2,-1)
                ofst += cnt
        
        mm[:] = self._slab_slice_mask>=0
        weighted_samples = (self._samples_dist[:,mm]*self._samples[:,:ofst])+((1-self._samples_dist[:,mm])*self._reg_samples[:,mm])
        sm = weighted_samples.sum(0)
        cost = 200*np.diff(weighted_samples,1,0)[0]/sm
        cost[np.abs(sm)<1e-6] = 0
        cost[:] = np.tanh(self.bbr_slope*cost-self.bbr_offset)
        cur_cost = 1+(cost.sum()+self._reg_cost)/self._reg_samples.shape[1]

        if update_reg_samples:
            print 'updated cost ', (self._reg_samples[:,mm]-weighted_samples).std(1)
            self._reg_samples[:,mm] = weighted_samples
            self._init_energy()
            
        return cur_cost


    def _init_energy(self):
        sm = self._reg_samples.sum(0)
        self._cost[:] = 200*np.diff(self._reg_samples,1,0)/sm
        self._cost[np.abs(sm)<1e-6] = 0
        self._cost[:] = np.tanh(self.bbr_slope*self._cost-self.bbr_offset)
        del sm        

    def set_fmin(self, optimizer, stepsize, **kwargs):
        """
        Return the minimization function
        """
        self.stepsize = stepsize
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        self.optimizer_kwargs.setdefault('xtol', XTOL)
        self.optimizer_kwargs.setdefault('ftol', FTOL)
        self.optimizer_kwargs.setdefault('gtol', GTOL)
        self.optimizer_kwargs.setdefault('maxiter', MAXITER)
        self.optimizer_kwargs.setdefault('maxfun', MAXFUN)
        self.use_derivatives = use_derivatives(self.optimizer)
        
    def explore_cost(self, slab, transform, data, values, force_recompute_subset=False, sigma=.3):
        tt = transform.copy()
        costs = np.empty((len(transform.param),len(values)))
        for p in range(len(transform.param)):
            self.sample_cost(slab, data,transform, force_recompute_subset=True, sigma=sigma)
            mmm=np.zeros(len(transform.param))
            mmm[p]=1
            for idx,delta in enumerate(values):
                tt.param = transform.param+(mmm*delta)
                costs[p,idx]  = self.sample_cost(slab, data, tt, force_recompute_subset, sigma=sigma)
        return costs                


class EPIOnlineRealignFilter(EPIOnlineResample):
    
    def correct(self, realigned, pvmaps, frame_shape, sig_smth=12, white_idx=1,
                maxiter = 32, residual_tol = 1e-3):
        
        float_mask = nb.Nifti1Image(
            self.mask.get_data().astype(np.float32),
            self.mask.get_affine())

        ext_mask = self.mask.get_data()>0
        ext_mask[pvmaps.get_data()[...,:2].sum(-1)>0.1] = True

        sig_smth = 12
        
        cdata = None
        for fr, slab, reg, data in realigned:
            if cdata is None: # init all
                cdata = np.zeros(data.shape)
                cdata2 = np.zeros(data.shape)
                epi_pvf = np.empty(frame_shape+(pvmaps.shape[-1],))
                epi_mask = np.zeros(frame_shape, dtype=np.bool)
                prev_reg = np.inf
                white_wght = np.empty(data.shape[:2])
                smooth_white_wght = np.empty(data.shape[:2])
                res = np.empty(data.shape[:2])
                corr_fac = np.empty(data.shape[:2])
            if not np.allclose(prev_reg, reg, 1e-4):
                print 'sample pvf and mask'
                epi_pvf[:] = self.inv_resample(
                    pvmaps, reg, frame_shape, -1,
                    mask = ext_mask)
#                epi_mask[:] = self.inv_resample(self.mask, reg, frame_shape, 0)
                epi_mask[:] = self.inv_resample(self.mask, reg, frame_shape, order=-1)>0
                # temporary fix, should fix the mask problem
                #epi_mask[:] = epi_pvf[...,:].sum(-1) > 0
                #epi_mask[epi_pvf[...,:].sum(-1) <=0] = False
            cdata[:] = data
            cdata2.fill(0)
            for sli,sln in enumerate(slab):
                sl_mask = epi_mask[...,sln]
                sl_mask[data[...,sli]<=0] = False
                n_sl_samples = np.count_nonzero(sl_mask)
                if np.count_nonzero(sl_mask) < 30:
                    print 'not enough samples (%d) skipping slice %d'%(epi_mask[...,sln].sum(),sln)
                    continue
                niter = 0
                res.fill(0)
                regs_subset = epi_pvf[sl_mask,sln].sum(0) > 5
                regs = epi_pvf[sl_mask,sln][...,regs_subset]
                regs[:,-1] = 1
                regs_pinv = np.linalg.pinv(regs)
                data_mask_mean = data[sl_mask,sli].mean()
                white_wght[:] = epi_pvf[..., sln, white_idx]*sl_mask
                smooth_white_wght[:] = scipy.ndimage.filters.gaussian_filter(white_wght,sig_smth,mode='constant')
                smooth_white_wght[np.logical_and(smooth_white_wght==0,sl_mask)] = 1e-8
                tmp_res = np.empty(n_sl_samples)
                while niter<maxiter:
                    betas = regs_pinv.dot(cdata[sl_mask,sli].ravel())
                    tmp_res[:] = np.log(cdata[sl_mask,sli]/betas.dot(regs.T))
                    if np.count_nonzero(np.isnan(tmp_res))>0 or  np.count_nonzero(np.isinf(tmp_res))>0:
                        raise RuntimeError
                    res.fill(0)
                    res[sl_mask] = tmp_res
                    res[:] = scipy.ndimage.filters.gaussian_filter(res*white_wght,sig_smth,mode='constant')/\
                        smooth_white_wght
                    res_std = res[sl_mask].std()
                    if res_std < residual_tol:
                        break
                    corr_fac[:] = np.exp(-res)
                    corr_fac /= corr_fac[sl_mask].mean()
                    cdata[...,sli] *= corr_fac
                    print betas, res_std
                    niter+=1
                betas = regs_pinv.dot(cdata[sl_mask,sli].ravel())
                cdata2[sl_mask,sli] = cdata[sl_mask,sli] - betas.dot(regs.T)
            yield fr, slab, reg, cdata2.copy()
        return

    def process(self, stack, *args, **kwargs):
        stack._init_dataset()
        self.correct(
            super(EPIOnlineRealignFilter,self).process(stack, yield_raw=True),
            *args, **kwargs)
    
            
def filenames_to_dicoms(fnames):
    for f in fnames:
        yield dicom.read_file(f)

class NiftiIterator():

    def __init__(self, nii):
        
        self.nii = nii
        self.nslices,self.nframes = self.nii.shape[2:4]
        self._affine = self.nii.get_affine()
        self._voxel_size = np.asarray(self.nii.header.get_zooms()[:3])
        self._slice_order = np.arange(self.nslices)
        self._shape = self.nii.shape
        self._slice_trigger_times = np.arange(self.nslices)*self.nii.header.get_zooms()[3]/float(self.nslices)
        
    def iter_frame(self, data=True):
        data = self.nii.get_data()
        for t in range(data.shape[3]):
            yield t, self.nii.get_affine(), data[:,:,:,t]
        del data

    def iter_slabs(self, data=True):
        data = self.nii.get_data()
        for t in range(data.shape[3]):
            for s,tt in enumerate(self._slice_trigger_times):
                yield t, [s], self.nii.get_affine(), tt, data[:,:,s,t,np.newaxis]
        del data

def resample_mat_shape(mat,shape,voxsize):
    old_voxsize = np.sqrt((mat[:3,:3]**2).sum(0))
    k = old_voxsize*np.array(shape)
    newshape = np.round(k/voxsize)
    res = k-newshape*voxsize
    newmat = np.eye(4)
    newmat[:3,:3] = np.diag((voxsize/old_voxsize)).dot(mat[:3,:3])
    newmat[:3,3] = mat[:3,3]+newmat[:3,:3].dot(res/voxsize/2)
    return newmat,tuple(newshape.astype(np.int32).tolist())
