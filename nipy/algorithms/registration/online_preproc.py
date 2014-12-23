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
from scipy.interpolate import LinearNDInterpolator, interpn, Rbf
from .slice_motion import surface_to_samples, vertices_normals, compute_sigloss, intensity_factor


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


    def resample(self, data, out, voxcoords, order=3):
        out[:] = map_coordinates(data, np.rollaxis(voxcoords,-1,0),order=order).reshape(voxcoords.shape[:-1])
#        out[:] = interpn(tuple([np.arange(d) for d in data.shape]),data,voxcoords,bounds_error=False,fill_value=0)
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
        points = np.empty(vol.shape+(3,))
        voxs = np.rollaxis(np.mgrid[[slice(0,d) for d in vol.shape]],0,4)
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

    def rbf_pve_scatter(self,data,out,slabs,transforms,coords,normals,pvmaps):
        nslices = data[0].shape[2]*len(slabs)
        vol = np.empty(data[0].shape[:2]+(nslices,))
        points = np.empty(vol.shape+(3,))
        voxs = np.rollaxis(np.mgrid[[slice(0,d) for d in vol.shape]],0,4)
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

#        # TODO: optimally should compute for each transform
#        epi_pvf = self.inv_resample(pvmaps, t, frame_shape, -1, mask = ext_mask)

        def pve_dist(x1,x2):
            ## compute euclidian distance between two points
            eucdist = np.sqrt(((x1-x2)**2).sum(0))
            # the smaller the pve, the larger the distance is, so that voxel is not taken into account
            pve = epi_pvf[0][(x1[0],x1[1],x1[2])] # suppose that the x1 is the voxel coordinates ???
            eucdist /= pve
            return eucdist

        rbfi = Rbf(points[...,0].ravel(),points[...,1].ravel(),points[...,1].ravel(),vol[epi_mask].ravel())
                   #,norm=pve_dist)
        out[:] = rbfi(coords[:,0],coords[:,1],coords[:,2]).reshape(out.shape)
        

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
                 normals,

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
                 
                 init_reg = None,
                 affine_class=Rigid,
                 optimizer=OPTIMIZER,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,

                 nsamples_per_slab=1000,
                 min_nsamples_per_slab=50,

                 bbr_shift=1.5,
                 bbr_slope=1,
                 bbr_offset=0,

                 iekf_jacobian_epsilon=1e-6,
                 iekf_convergence=1e-4,
                 iekf_max_iter=8,
                 iekf_observation_var=1e4,
                 iekf_transition_cov=1e-2,
                 iekf_init_state_cov=1e-1):

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


        self.bnd_coords, self.normals = bnd_coords, normals
        self.border_nvox = self.bnd_coords.shape[0]

        self.nsamples_per_slab = nsamples_per_slab
        self.min_nsamples_per_slab = min_nsamples_per_slab
        self.affine_class = affine_class
        self.init_reg = init_reg
        self.bbr_shift, self.bbr_slope, self.bbr_offset = bbr_shift, bbr_slope, bbr_offset

        self.iekf_jacobian_epsilon = iekf_jacobian_epsilon
        self.iekf_convergence = iekf_convergence
        self.iekf_max_iter = iekf_max_iter
        self.iekf_observation_var = iekf_observation_var
        self.iekf_transition_cov = iekf_transition_cov
        self.iekf_init_state_cov = iekf_init_state_cov

        self._interp_order = 3

        # compute fmap values on the surface used for realign
        if self.fmap != None:
            fmap_vox = apply_affine(self.world2fmap,
                                    self.bnd_coords.reshape(-1,3))
            self.fmap_values = self.fmap_scale * map_coordinates(
                self.fmap.get_data(), fmap_vox.T,
                order=1)
            del fmap_vox
        else:
            self.fmap_values = None


        # Set the minimization method
        self.set_fmin(
            optimizer, stepsize,
            xtol=xtol, ftol=ftol, gtol=gtol,
            maxiter=maxiter, maxfun=maxfun)



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

        self._n_samples = self.bnd_coords.shape[0]

        self.slab_bnd_coords = np.empty((3,self._n_samples,3),np.double)
        self.slab_bnd_normals = np.empty((self._n_samples,3),np.double)

        self._slab_slice_mask = np.empty(self._n_samples, np.int8)
        self._reg_samples = np.empty((3,self._n_samples))
        self._samples_mask = slice(0,None)
        self._samples = np.empty((7,3,self._n_samples))
        self._samples.fill(np.nan)
        self._samples_dist = np.empty((13,self._n_samples))
        self._cost = np.empty((7,self._n_samples))

        last_reg.param = self._register_slab(range(self.nslices), data1, last_reg)
        print last_reg.param
        # compute values for initial registration
        self.apply_transform(
            last_reg,
            self.bnd_coords, self.normals, self.slab_bnd_coords[0], self.slab_bnd_normals,
            self.fmap_values, phase_dim=data1.shape[self.pe_dir])
        
        self.slab_bnd_coords[1] = self.slab_bnd_coords[0] + self.slab_bnd_normals
        self.slab_bnd_coords[2] = self.slab_bnd_coords[0] - self.slab_bnd_normals
        self.resample(data1, self._reg_samples, self.slab_bnd_coords)

        self._samples_mask = np.ones(self._n_samples,dtype=np.bool)
        mm = self._samples_mask
        
        # samples with intensity far below mean are less reliable
        reg_samp_means = self._reg_samples.mean(1)[:,np.newaxis]
        reg_samp_var = self._reg_samples.var(1)[:,np.newaxis]
#        self._samples_reliability = np.exp(-((self._reg_samples-reg_samp_means)*
#                                             (self._reg_samples<reg_samp_means))**2/(2*reg_samp_var)).mean(0)

#        self._samples_reliability = 2-np.diff(self._reg_samples[1:],0)[0]/self._reg_samples[1]
#        self._samples_reliability[np.isnan(self._samples_reliability)] = 0
#        self._samples_reliability[self._samples_reliability<0] = 0
#        self._samples_reliability[self._samples_reliability>1] = 1
        self._samples_reliability = np.all(self._reg_samples>0,0).astype(np.float)
        
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
                self.bnd_coords,
                self.echo_time, slicing_axis=self.slice_axis)
            # multiply by sigloss with saturation
            self._samples_reliability *= self._samples_sigloss*1.2
            self._samples_reliability[self._samples_reliability>1]=1
            
        # remove the samples with low reliability
        mm[:] = self._samples_reliability > .5

        # reestimate first frame registration  with only reliable samples using cg optimizer
        self.optimizer = 'cg'
        last_reg.param = self._register_slab(range(self.nslices), data1, last_reg)

        self.full_frame_reg = last_reg.copy()
        
        ndim_state = 6
        transition_matrix = np.eye(ndim_state)
        transition_covariance = np.diag([.01]*6+[.1]*6) # change in position should first occur by change in speed !?
        transition_covariance = np.eye(6)*self.iekf_transition_cov
        if ndim_state>6:
            transition_matrix[:6,6:] = np.eye(6) # TODO: set speed
            # transition_covariance[:6,6:] = np.eye(6)
        self.transition_covariance = transition_covariance[:ndim_state,:ndim_state]

        initial_state_mean = np.hstack([last_reg.param.copy(), np.zeros(6)])
        initial_state_covariance = np.eye(ndim_state)*self.iekf_init_state_cov
        initial_state_mean = initial_state_mean[:ndim_state]
        initial_state_covariance = initial_state_covariance[:ndim_state,:ndim_state]
        
        # R the (co)variance (as we suppose white observal noise)
        # this could be used to weight samples
        self.observation_variance = self._n_samples*self.iekf_observation_var/self._samples_reliability

        stack_it = stack.iter_slabs()
        stack_has_data = True
        current_volume = data1.copy()
        fr,sl,aff,tt,sl_data = stack_it.next()
        
        self.filtered_state_means = [initial_state_mean]
        self.filtered_state_covariances = [initial_state_covariance]
        
        new_reg = last_reg.copy()
        mm = self._samples_mask

        self.tmp_states=[]
        while stack_has_data:
            
            # forward prediction, in the 6 param case, identity
            pred_state = transition_matrix.dot(self.filtered_state_means[-1])
            estim_state = pred_state.copy()
            pred_covariance = self.filtered_state_covariances[-1] + self.transition_covariance
            state_covariance = pred_covariance.copy()

            convergence, niter = np.inf, 0
            new_reg.param = estim_state[:6]
            self.sample_cost_jacobian(sl, sl_data, new_reg, force_recompute_subset=True)
            if self._n_slab_samples < self.min_nsamples_per_slab:
                print 'not enough points, skipping slab'
            else:
                while convergence > self.iekf_convergence and niter < self.iekf_max_iter:
                    if niter>0:
                        new_reg.param = estim_state[:6]
                        self.sample_cost_jacobian(sl, sl_data, new_reg)
                    cost, jac = self._cost[0], self._cost[1:]
                    mm[mm] = self.observation_variance[mm] < 1e16
                    kalman_gain = np.dot(
                        pred_covariance.dot(jac[:,mm]),
                        np.linalg.inv(jac[:,mm].T.dot(pred_covariance).dot(jac[:,mm])+
                                      np.diag(self.observation_variance[mm])))
                    estim_state_old = estim_state.copy()
#                    print 'bias', self._cost[:,mm].mean(1)
#                    print 'gain', kalman_gain.dot(cost[mm] + jac[:,mm].T.dot(pred_state-estim_state))
                    estim_state[:] = estim_state + kalman_gain.dot(cost[mm] + jac[:,mm].T.dot(pred_state-estim_state))
                    state_covariance[:] = (np.eye(ndim_state)-kalman_gain.dot(jac[:,mm].T)).dot(pred_covariance)

                    self.tmp_states.append(estim_state-pred_state)
                    convergence = np.sqrt(((estim_state_old-estim_state)**2).sum())
                    niter += 1
                    print ("%.12f %.5f : "+"%.5f,\t"*6)%((convergence,np.abs(cost[mm]).mean()) +tuple(estim_state[:6]))
                if niter==self.iekf_max_iter:
                    print "maximum iteration number exceeded"

            self.filtered_state_means.append(estim_state)
            self.filtered_state_covariances.append(state_covariance)

            # recompute just for display of the cost function
            new_reg.param = estim_state[:6]
#            self._init_energy()
#            self.sample_cost_jacobian(sl, sl_data, new_reg)

            print '_'*80
            print ('%5f :\t' + '%.5f,'*6)%((np.abs(self._cost[0]).mean(),)+tuple(estim_state))
            print 'update : ', ('%.5f,'*6)%tuple(estim_state-self.filtered_state_means[-2])
            print '_'*80
            new_reg.param = estim_state[:6]
            yield fr, sl, new_reg.as_affine().dot(aff), sl_data
            try:
                fr,sl,aff,tt,sl_data = stack_it.next()
            except StopIteration:
                stack_has_data = False

    def _register_slab(self, slab, slab_data, init_transform):
        print 'register slab', slab
        
        transform = self.affine_class(init_transform.as_affine())
        self.sample_cost(slab, slab_data, transform)

        def f(pc):
            transform.param = pc
            nrgy = self.sample_cost(slab, slab_data, transform)
            if np.isnan(nrgy):
                raise RuntimeError
            print 'f %.10f : %f %f %f %f %f %f'%tuple([nrgy] + pc.tolist())
            return nrgy
        self._pc = None
        fmin, args, kwargs = configure_optimizer(self.optimizer,
                                                 **self.optimizer_kwargs)
        pc = fmin(f, transform.param, *args, **kwargs)

        return pc

    def apply_transform(self, transform, in_coords, in_vec, out_coords, out_vec,
                        fmap_values=None, subset=slice(None), phase_dim=64):
        ref2fmri = np.linalg.inv(transform.as_affine().dot(self.affine))
        #apply current slab transform
        out_coords[...,subset,:] = apply_affine(ref2fmri, in_coords[...,subset,:])
        rot = np.diag(1/np.sqrt((ref2fmri[:3,:3]**2).sum(1))).dot(ref2fmri[:3,:3])
        out_vec[...,subset,:] = in_vec[...,subset,:].dot(rot.T)
        #add shift in phase encoding direction
        if isinstance(fmap_values, np.ndarray):
            out_coords[...,subset,self.pe_dir]+=fmap_values[...,subset]*phase_dim
    
    def sample_cost_jacobian(self, slab, data, transform, force_recompute_subset=False):
        # compute the cost and the jacobian for a given set of params

        sa = self.slice_axis
        slice_axes = np.ones(3, np.bool)
        slice_axes[sa] = False

        mm = self._samples_mask
        mm[:] = self._slab_slice_mask>=0
        
        epsilon = self.iekf_jacobian_epsilon
        
        self._update_subset(slab, transform, data.shape, force_recompute_subset=force_recompute_subset)

        for si, s in enumerate(slab):
            mm[:] = self._slab_slice_mask==s
            cnt = np.count_nonzero(mm)
            if cnt>0:
#                crds = np.empty((13,3,cnt,3))
                crds = np.empty((7,3,cnt,3))
                crds[0,:] = self.slab_bnd_coords[0,mm][np.newaxis]
                tan_arcsin_z = self.slab_bnd_normals[mm,sa]/np.sqrt(1-self.slab_bnd_normals[mm,sa]**2)
                #  normal projected on 2D slice, unit vector
                projnorm = self.slab_bnd_normals[mm][:,slice_axes]
                projnorm /= np.sqrt((projnorm**2).sum(-1))[:,np.newaxis]
                # depending on the normal z the 2 ends are projected reversed
                proj_z = (tan_arcsin_z>0)-.5
                crds[0,1,:,:2] += projnorm*(tan_arcsin_z*((crds[0,0,:,sa]-proj_z)-s))[:,np.newaxis]
                crds[0,2,:,:2] += projnorm*(tan_arcsin_z*((crds[0,0,:,sa]+proj_z)-s))[:,np.newaxis]
                # copy 
                crds[1:4] = crds[0]
                for i in range(6):
                    if i<3:
                        # translate the ith dimension, minus because we compute forward finite difference of the inverse transform
                        crds[i+1,...,i] -= epsilon * transform.precond[i]
                        if i==2:
                            # change in z is merely a translation in 2D
                            crds[i+1,1:,:,:2] -= ((projnorm * tan_arcsin_z[:,np.newaxis]) * epsilon * transform.precond[i])[np.newaxis]
                    else:
                        vec = np.zeros(3)
                        # rotate by epsilon, minus because we compute forward finite difference of the inverse transform
                        vec[i-3] = -epsilon * transform.precond[i]
                        m = rotation_vec2mat(vec)
                        if i < 5:
                            # rotate coordinates and normals by epsilon
                            crds[i+1,:] = m.dot(crds[0,0].T).T[np.newaxis,:]
                            rnormals = m.dot(self.slab_bnd_normals[mm].T).T
                            # same as above
                            tan_arcsin_z[:] = rnormals[:,sa]/np.sqrt(1-rnormals[:,sa]**2)
                            rnormals /= np.sqrt((rnormals[:,slice_axes]**2).sum(-1))[:,np.newaxis]
                            crds[i+1,1,:,:2] += rnormals[:,:2]*(tan_arcsin_z*((crds[i+1,0,:,sa]-proj_z)-s))[:,np.newaxis]
                            crds[i+1,2,:,:2] += rnormals[:,:2]*(tan_arcsin_z*((crds[i+1,0,:,sa]+proj_z)-s))[:,np.newaxis]
                            del rnormals
                        else:
                            # this is only a rotation in 2D
                            crds[i+1] = m.dot(crds[0,:].reshape(-1,3).T).T.reshape(3,-1,3)
                # set to top or bottom of slice thickness, could be removed, only for debug
                crds[:,1,:,2] = s+proj_z
                crds[:,2,:,2] = s-proj_z
                # coordinates between 2 projected coordinates
                crds[:,0] = crds[:,1:].sum(1)/2.
                # add a minimum of space between samples
                crds[:,1,:,:2] -= projnorm*proj_z[:,np.newaxis]*.2
                crds[:,2,:,:2] += projnorm*proj_z[:,np.newaxis]*.2
                # resample 
                self._samples[:,:,mm] = map_coordinates(
                    data[...,si].astype(np.float),
                    crds[...,slice_axes].reshape(-1,2).T,order=self._interp_order).reshape(7,3,-1)
                self.crds=crds
                #del crds
                del projnorm,proj_z,tan_arcsin_z

        mm[:] = self._slab_slice_mask>=0
        if np.count_nonzero(mm) < self.min_nsamples_per_slab:
            return
#        sm = self._samples[...,mm].mean(1)
        sm = np.abs(np.squeeze(np.diff(self._samples[:,1:,mm],1,1)))+1
        self._cost[:,mm] = np.tanh(self.bbr_slope*(
                self._samples[:7,1:,mm].sum(1)-2*self._samples[:7,0,mm])/sm[:7]-self.bbr_offset)
        #compute partial derivatives
#        self._cost[1:,mm] -= np.tanh(self.bbr_slope*(
#                self._samples[7:,1:,mm].sum(1)-2*self._samples[7:,0,mm])/sm[7:]-self.bbr_offset)
        self._cost[1:,mm] -= self._cost[0,mm]
        self._cost[1:,mm] /= epsilon
        del sm#, excl

    def _update_subset(self, slab, transform, shape, force_recompute_subset=False):
        sa = self.slice_axis
        mm = self._samples_mask

        test_points = np.array([(x,y,z) for x in (0,shape[0]) for y in (0, shape[1]) for z in (0,self.nslices)])
        recompute_subset = np.abs(
            self._last_subsampling_transform.apply(test_points) -
            transform.apply(test_points))[:,sa].max() > 0.01
        
        if recompute_subset or force_recompute_subset:
            self.apply_transform(
                transform,
                self.bnd_coords,
                self.normals,
                self.slab_bnd_coords,
                self.slab_bnd_normals,
                self.fmap_values,
                phase_dim=shape[self.pe_dir])
            self._last_subsampling_transform = transform.copy()

            self._slab_slice_mask.fill(-1)
            self._samples_dist.fill(0)
            for s in slab:
                mm[:] = np.abs(self.slab_bnd_coords[0,..., self.slice_axis]-s) < .5
                # remove vertex with normal perpendicular to slice
                mm[np.abs(self.slab_bnd_normals[...,sa])>.85] = False
                nsamp = np.count_nonzero(mm)
                nsamples_per_slice = self.nsamples_per_slab/float(len(slab))
                prop = nsamples_per_slice/float(nsamp+1)
                #mm[mm] = np.random.random(nsamp) < prop
                ## use the more reliable samples in subset
                if prop < 1:
                    mm[mm] = self._samples_reliability[mm] >= np.percentile(self._samples_reliability[mm],100*(1-prop))
                nsamp = np.count_nonzero(mm)
                prop = nsamples_per_slice/float(nsamp+1)
                if prop <1:
                    mm[mm] = np.random.random(nsamp) < prop
                self._slab_slice_mask[mm] = s
            mm[:] = self._slab_slice_mask >= 0

            self._n_slab_samples = np.count_nonzero(mm)
            print 'new subset %d'%self._n_slab_samples
            if self._n_slab_samples < 1:
                return
        else:
            self.apply_transform(
                transform,
                self.bnd_coords,
                self.normals,
                self.slab_bnd_coords,
                self.slab_bnd_normals,
                self.fmap_values,
                mm,
                phase_dim=shape[self.pe_dir])
        self.slab_bnd_coords[1,mm] -= self.slab_bnd_normals[mm]
        self.slab_bnd_coords[2,mm] += self.slab_bnd_normals[mm]

        
    def sample_cost(self, slab, data, transform):

        self.apply_transform(
            transform,
            self.bnd_coords,
            self.normals,
            self.slab_bnd_coords,
            self.slab_bnd_normals,
            self.fmap_values,
            phase_dim=data.shape[self.pe_dir])

        self.slab_bnd_coords[1] -= self.slab_bnd_normals*self.bbr_shift
        self.slab_bnd_coords[2] += self.slab_bnd_normals*self.bbr_shift
        self.resample(data, self._reg_samples, self.slab_bnd_coords,order=self._interp_order)
        self._init_energy()
        return np.abs(self._cost[0,self._samples_mask]).mean()

    def _init_energy(self):
#        sm = self._reg_samples.mean(0)+1
        sm = np.abs(np.diff(self._reg_samples[1:],1,0))[0]+1
        self._cost[0] = (self._reg_samples[1:].sum(0)-2*self._reg_samples[0])/sm
        self._cost[0] = np.tanh(self.bbr_slope*self._cost[0]-self.bbr_offset)
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
        
    def explore_cost(self, slab, transform, data, values):
        tt = transform.copy()
        costs = np.empty((len(transform.param),len(values)))
        for p in range(len(transform.param)):
            self.sample_cost(slab, data,transform)
            mmm=np.zeros(len(transform.param))
            mmm[p]=1
            for idx,delta in enumerate(values):
                tt.param = transform.param+(mmm*delta)
                costs[p,idx]  = self.sample_cost(slab, data, tt)
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
