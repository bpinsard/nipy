import numpy as np

import nibabel as nb, dicom
from nibabel.affines import apply_affine
from ...fixes.nibabel import io_orientation
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .affine import Rigid, Affine, rotation_vec2mat, to_matrix44

from .optimizer import configure_optimizer, use_derivatives
from scipy.optimize import fmin_slsqp
from scipy.ndimage import convolve1d, gaussian_filter, gaussian_filter1d, binary_erosion, binary_dilation
import scipy.stats, scipy.sparse
from scipy.ndimage.interpolation import map_coordinates
#from pykdtree.kdtree import KDTree ## seems slower in practice
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import LinearNDInterpolator, interpn
from .slice_motion import surface_to_samples, vertices_normals, compute_sigloss, intensity_factor


import time
import itertools

# Module globals
RADIUS = 100
DTYPE = np.float32
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
                 slice_axis=2,
                 recenter_fmap_data=True):

        self.fmap, self.mask = fieldmap, mask
        self.fieldmap_reg = fieldmap_reg

        self.slice_axis = slice_axis
        self.in_slice_axes = np.ones(3, np.bool)
        self.in_slice_axes[self.slice_axis] = False


        self.slice_order = slice_order
        self.pe_sign = int(phase_encoding_dir > 0)*2-1
        self.pe_dir = abs(phase_encoding_dir)-1
        self.repetition_time = repetition_time
        self.echo_time = echo_time
        self.slice_tr = slice_repetition_time
        self.interleaved = int(interleaved)
        self.slice_trigger_times = slice_trigger_times
        self.slice_thickness = slice_thickness

        if self.fmap is not None:
            self.recenter_fmap_data = recenter_fmap_data
            self._preproc_fmap()

        self.fmap_scale = self.pe_sign*echo_spacing/2.0/np.pi
        self._resample_fmap_values = None
        self.st_ratio = 1

    def resample(self, data, out, voxcoords, order=1):
        out[:] = map_coordinates(data, np.rollaxis(voxcoords,-1,0),order=order).reshape(voxcoords.shape[:-1])
#        out[:] = interpn(tuple([np.arange(d) for d in data.shape]),data,voxcoords,bounds_error=False,fill_value=0)
        return out

    def scatter_resample_volume(self, data, out, slabs, transforms, target_transform, mask=False):
        coords = apply_affine(
            target_transform,
            np.rollaxis(np.mgrid[[slice(0,d) for d in out.shape]],0,4))
        self.scatter_resample(data, out, slabs, transforms, coords, mask=mask)
        del coords

    def scatter_resample_rbf(self, data, out, slabs, transforms, coords,
                             pve_map = None,
                             rbf_sigma=3, kneigh_dens=256, mask=True):
        nslices = data[0].shape[2]*len(slabs)
        vol_shape = data[0].shape[:2]+(nslices,)
        phase_vec = np.zeros(3)
        for sl, t in zip(slabs, transforms):
            phase_vec+= t[:3,self.pe_dir]
        phase_vec /= len(slabs)
        if (not hasattr(self,'_scat_resam_rbf_coords') or
            not self._scat_resam_rbf_coords is coords
            or not np.allclose(self._scat_resam_rbf_phase_vec, phase_vec, rtol=0, atol=1e-2)):
            self._scat_resam_rbf_coords = coords
            self._scat_resam_rbf_phase_vec = phase_vec
            ## TODO: if fieldmap recompute kdtree for each iteration, the fieldmap evolves!!
            if not self.fmap is None:
                self._precompute_sample_fmap(coords, vol_shape)
                coords = coords + self._resample_fmap_values[:,np.newaxis].dot(phase_vec[np.newaxis])
            print('recompute KDTree')
            self._coords_kdtree = KDTree(coords)
        coords_kdtree = self._coords_kdtree 
        out.fill(0)
        out_weights = np.zeros(len(coords))

        if mask:
            epi_mask = self.inv_resample(self.mask, transforms[len(transforms)/2],
                                         vol_shape, -1, self.mask.get_data()>0)>0

        if not pve_map is None:
            epi_pvf = self.inv_resample(
                pve_map, transforms[len(transforms)/2], vol_shape, -1, mask = self.mask_data)

        voxs = np.rollaxis(np.mgrid[[slice(0,d) for d in vol_shape]],0,4)
        ## could/shoud we avoid loop here and do all slices in the meantime
        for sl, d, t in zip(slabs, data, transforms):
            if mask:
                slab_mask = epi_mask[...,sl]
                d = d[slab_mask]
                points = apply_affine(t, voxs[...,sl,:][slab_mask,:])
                if len(points) < 1:
                    continue
            else:
                points = apply_affine(t, voxs[...,sl,:]).reshape(-1,3)
                d = d.ravel()
            dists, idx = coords_kdtree.query(points, k=kneigh_dens, distance_upper_bound=16)
            not_inf = np.logical_not(np.isinf(dists))
            idx2 = (np.ones(kneigh_dens,dtype=np.int)*np.arange(len(d))[:,np.newaxis])[not_inf]
            idx = idx[not_inf]
            dists = dists[not_inf]
            weights = np.exp(-(dists/rbf_sigma)**2)
            if not pve_map is None:
                weights *= epi_pvf[...,sl][slab_mask][idx2]
            weights[weights<.05] = 0 # truncate
            np.add.at(out, idx, d[idx2]*weights)
            np.add.at(out_weights, idx, weights)
        out /= out_weights
        print np.count_nonzero(out_weights==0)
        out[np.isinf(out)] = np.nan


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
        # get unit norm of mean phase orientation in world space
        phase_vec /= np.linalg.norm(phase_vec)
        epi_mask = slice(0, None)
        if mask:
            epi_mask = self.inv_resample(self.mask, transforms[0], vol.shape, -1, self.mask.get_data()>0)>0
            #epi_mask[:] = binary_dilation(epi_mask, iterations=1)
            points = points[epi_mask]
        if not self.fmap is None:
            self._precompute_sample_fmap(coords, vol.shape)
            coords = coords.copy()
            coords += self._resample_fmap_values[:,np.newaxis].dot(phase_vec[np.newaxis])
        print( 'create interpolator datarange %f'%(vol[epi_mask].ptp()) )
        lndi = LinearNDInterpolator(points.reshape(-1,3), vol[epi_mask].ravel())
        print( 'interpolate', len(points), len(coords) )
        out[:] = lndi(coords.reshape(-1,3)).reshape(out.shape)

    def ribbon_resample(self, data, out, slabs, transforms, inner_surf, outer_surf, mask=True):
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
        # get unit norm of mean phase orientation in world space
        phase_vec /= np.linalg.norm(phase_vec)

        #epi_vox_kdtree = scipy.spatial.KDTree(points.reshape(-1,3))        
        
        pass

    def allpoints(self, stack_iter, out, gm_pve_idx=0, gm_thr=.1):
        idx=0
        for fr, sl, reg, tt, data, pve in stack_iter:
            gm = pve[...,gm_pve_idx]
            mask = gm > gm_thr
            nsamp = np.count_nonzero(mask)
            rng = slice(idx,idx+nsamp)
            out[rng,:3] = apply_affine(reg, np.argwhere(mask))
            out[rng,3] = tt
            out[rng,4] = gm[mask]
            out[rng,5] = data[mask]
            idx += nsamp

    def _preproc_fmap(self):
        if self.fieldmap_reg is None:
            self.fieldmap_reg = np.eye(4)
        self.fmap2world = np.dot(self.fieldmap_reg, self.fmap.affine)
        self.world2fmap = np.linalg.inv(self.fmap2world)
        grid = apply_affine(
            np.linalg.inv(self.mask.affine).dot(self.fmap2world),
            np.rollaxis(np.mgrid[[slice(0,n) for n in self.fmap.shape]],0,4))
        self.fmap_mask = map_coordinates(
            self.mask.get_data(),
            grid.reshape(-1,3).T, order=0).reshape(self.fmap.shape) > 0
        fmap_data = self.fmap.get_data()
        if self.recenter_fmap_data: #recenter the fieldmap range to avoid shift
            fmap_data -= fmap_data[self.fmap_mask].mean()
        ## extend fmap values out of mask
        #fmap_data[~self.fmap_mask] = 0
        #self.pe_dir
        self.fmap = nb.Nifti1Image(fmap_data, self.fmap.affine)

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
            np.rollaxis(np.mgrid[[slice(0,s) for s in self.fmap.shape]],0,4))[self.fmap_mask]
        shift = self.fmap_scale * self.fmap.get_data()[self.fmap_mask] * shape[self.pe_dir]
        coords[...,self.pe_dir] += shift
        self._inv_shiftmap = np.empty(shape)
        self._inv_shiftmap.fill(np.inf)
        includ = np.logical_and(
            np.all(coords>-.5,-1),
            np.all(coords<np.array(shape)[np.newaxis]-.5,-1))
        coords = coords[includ]
        rcoords = np.round(coords).astype(np.int)
        shift = shift[includ]
        self._inv_shiftmap[(rcoords[...,0],rcoords[...,1],rcoords[...,2])] = -shift
        for x,y,z in zip(*np.where(np.isinf(self._inv_shiftmap))):
            ngbd = self._inv_shiftmap[
                max(0,x-1):x+1,max(0,y-1):y+1,max(0,z-1):z+1]
            self._inv_shiftmap[x,y,z] = ngbd.ravel()[np.argmin(np.abs(ngbd.ravel()))] # np.nanmedian(ngbd)  #TODO: check fastest
            del ngbd
        del includ, coords, rcoords, shift
        return self._inv_shiftmap
            
    def inv_resample(self, vol, affine, shape, order=0, mask=None):
        ## resample a undistorted volume to distorted EPI space
        # order = map_coordinates order, if -1, does integral of voxels in the
        # higher resolution volume (eg. for partial volume map downsampling)
        nvols = (vol.shape+(1,))[:4][-1]
        rvol = np.empty(shape+(nvols,))
        vol_data = vol.get_data()
        if vol_data.ndim < 4:
            vol_data = vol_data[...,np.newaxis]
        if order < 0:
            if mask is None:
                grid = np.mgrid[[slice(0,s) for s in vol.shape[:3]]].reshape(3,-1).T
            else:
                grid = np.argwhere(mask)
            vol2epi = np.linalg.inv(affine).dot(vol.affine)
            voxs = nb.affines.apply_affine(vol2epi, grid)
            if self.fmap is not None:
                vol2fmap = self.world2fmap.dot(vol.affine)
                fmap_voxs = nb.affines.apply_affine(vol2fmap, grid)
                fmap_values = shape[self.pe_dir] * self.fmap_scale * map_coordinates(
                    self.fmap.get_data(),
                    fmap_voxs.T,
                    order=1).reshape(fmap_voxs.shape[:-1])
                voxs[:, self.pe_dir] -= fmap_values
                del fmap_voxs, fmap_values
            voxs = np.rint(voxs).astype(np.int)
            steps = np.asarray([shape[1]*shape[2],shape[2],1])
            voxs_subset = np.logical_and(np.all(voxs>=0,1),np.all(voxs<shape,1))
            indices = (voxs[voxs_subset]*steps).sum(1)
            data = vol_data[mask]
            for v in range(nvols):
                rvol[...,v].flat[:] = np.bincount(indices, data[...,v].ravel()[voxs_subset], np.prod(shape))
            if order == -1: #normalize
                counts = np.bincount(indices, minlength=np.prod(shape))
                rvol /= counts.reshape(shape+(1,))
                del counts
            rvol[np.isnan(rvol)] = 0
            rvol = np.squeeze(rvol)
        else:
            grid = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]], 0, 4).astype(np.float)                
            if self.fmap is not None:
                inv_shift = self._epi_inv_shiftmap(affine, shape)
                grid[..., self.pe_dir] -= inv_shift
            epi2vol = np.linalg.inv(vol.affine).dot(affine)
            voxs = nb.affines.apply_affine(epi2vol, grid)

            for v in range(nvols):
                rvol[...,v] = map_coordinates(
                    vol_data[...,v],
                    voxs.reshape(-1,3).T, order=order).reshape(shape)
        del grid, voxs
        return np.squeeze(rvol)


class OnlineRealignBiasCorrection(EPIOnlineResample):
    
    def __init__(self,
                 mask,
                 anat_reg,
                 wm_weight = None,
                 bias_correction = True,
                 bias_sigma = 8,
                 register_gradient = False,
                 dog_sigmas = [1,2],
                 fieldmap = None,
                 fieldmap_reg = None,

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

                 iekf_min_nsamples_per_slab = 200,
                 iekf_jacobian_epsilon = 1e-3,
                 iekf_convergence = 1e-3,
                 iekf_max_iter = 8,
                 iekf_observation_var = 1,
                 iekf_transition_cov = 1e-3,
                 iekf_init_state_cov = 1e-3):

        self.iekf_min_nsamples_per_slab = iekf_min_nsamples_per_slab
        self.iekf_jacobian_epsilon = iekf_jacobian_epsilon
        self.iekf_convergence = iekf_convergence
        self.iekf_max_iter = iekf_max_iter
        self.iekf_observation_var = iekf_observation_var
        self.iekf_transition_cov = iekf_transition_cov
        self.iekf_init_state_cov = iekf_init_state_cov

        super(OnlineRealignBiasCorrection,self).__init__(
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
        self._anat_reg = anat_reg
        print(('init_reg params:\t' + '%.3f\ '*12)% tuple(Affine(self._anat_reg).param))
        self.mask_data = self.mask.get_data()>0

        self._bias_correction = bias_correction
        self._bias_sigma = bias_sigma
        self._register_gradient = register_gradient
        self._dog_sigmas = dog_sigmas
        self.wm_weight = wm_weight

        if self.wm_weight is None:
            if self._bias_correction:
                raise ValueError
        else:
            self.wm_weight_data = self.wm_weight.get_data()
            self.wm_weight_data_mask = self.wm_weight_data[self.mask_data]

        self._ref_kdtree = None

    def sample_ref(self, vol_data_masked, coords, data_out, mask, order=0, rbf_sigma=1, fill_value=0):
        if self._ref_kdtree is None:
            ref_coords = apply_affine(self.mask.affine, np.argwhere(self.mask_data))
            self._ref_kdtree = KDTree(ref_coords)
        if order==0: # NN
            dists, idx = self._ref_kdtree.query(coords, k=1, distance_upper_bound=1)
            mask[:] = np.isfinite(dists)
            if not data_out is None:
                data_out.fill(fill_value)
                data_out[mask] = vol_data_masked[idx[mask]]
        else: # RBF
            data_out.fill(fill_value)
            dists, idx = self._ref_kdtree.query(coords, k=int((rbf_sigma*2)**3), distance_upper_bound=rbf_sigma*2)
            tmp_mask = np.isinf(dists)
            idx[tmp_mask] = 0 # we keep them but their weights whould be zeros
            weights = np.exp(-(dists/rbf_sigma)**2)
            weights[weights<.05] = 0
            data_out[:] = (weights*vol_data_masked[idx]).sum(-1)/weights.sum(-1)
            data_out[np.isnan(data_out)] = fill_value
            mask[:] = dists.min(-1)<1
            
            
    def sample_shift(self, vox_in, aff):
        vox = apply_affine(np.linalg.inv(self.fieldmap_reg.dot(self.fmap.affine)).dot(aff), vox_in)
        shift = map_coordinates(
            self.fmap.get_data(),
            vox.reshape(-1,3).T,
            mode='reflect',
            order=1).reshape(vox_in.shape[:-1])
        return shift

    def process(self, stack, ref_frame=None, yield_raw=False):

        # to check if allocation is done in _sample_cost_jacobian
        self._slab_vox_idx = None

        frame_iterator = stack.iter_frame(queue_dicoms=True)
        #frame_iterator = stack.iter_frame()
        nvol, self.affine, self._first_frame = frame_iterator.next()
        self._epi2anat = np.linalg.inv(self.mask.affine).dot(self._anat_reg).dot(self.affine)

        self._first_frame = self._first_frame.astype(DTYPE)
        
        if self._register_gradient:
            #self.register_refvol = ref_vol - convolve1d(convolve1d(ref_vol, [1/3.]*3,0),[1/3.]*3,1)
            self.register_refvol = reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[0],d), [0,1], self._first_frame)-\
                                   reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[1],d), [0,1], self._first_frame)
        else:
            self.register_refvol = self._first_frame

            """
            anat_slab_coords = apply_affine(
                self._anat_reg.dot(self.affine),
                np.rollaxis(np.mgrid[[slice(None,d) for d in self.register_refvol.shape]],0,4))
            wm_weight = np.zeros_like(self.register_refvol, dtype=DTYPE)
            epi_mask = np.zeros_like(self.register_refvol, dtype=np.bool)
            self.sample_ref(self.wm_weight_data_mask, anat_slab_coords, wm_weight, epi_mask, order=1)
            wm_weight += epi_mask*1e-8 # takes average in slices without white matter
            self._ref_norm = np.exp(
                reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d, truncate=20), [0,1],
                       np.log(self.register_refvol+1e-8)*wm_weight)/
                reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d,truncate=20), [0,1], wm_weight))
            self._ref_norm[np.isnan(self._ref_norm)] = np.nanmean(self._ref_norm)
            self.register_refvol /= self._ref_norm
            """
            #self.register_refvol = reduce(lambda i,d: gaussian_filter1d(i,.5,d), [0,1], self._first_frame)

        self.slice_order = stack._slice_order
        inv_slice_order = np.argsort(self.slice_order)
        self.nslices = stack.nslices


        ndim_state = 6
        transition_matrix = np.eye(ndim_state, dtype=DTYPE)
        #transition_covariance = np.diag([.01]*6+[.1]*6
        transition_covariance = np.eye(ndim_state, dtype=DTYPE)*self.iekf_transition_cov
        if ndim_state>6:
            transition_matrix[:6,6:] = np.eye(6) # TODO: set speed
            # transition_covariance[:6,6:] = np.eye(6)
            self.transition_covariance = transition_covariance[:ndim_state,:ndim_state]

        initial_state_mean = np.zeros(ndim_state, dtype=DTYPE)
        initial_state_covariance = np.eye(ndim_state, dtype=DTYPE) * self.iekf_init_state_cov

        stack_it = stack.iter_slabs()
        stack_has_data = True
        fr,sl,aff,tt,sl_data = stack_it.next()
        self.sl_data = sl_data = sl_data.astype(DTYPE)

        # inv(R) the (co)variance (as we suppose white observal noise)
        # this could be used to weight samples 
        self.observation_variance = np.ones(sl_data.size, dtype=DTYPE)*self.iekf_observation_var #TODO change per voxel
        
        self.filtered_state_means = [initial_state_mean]
        self.filtered_state_covariances = [initial_state_covariance]
        self.niters = []
        self.matrices = []
        self.all_biases = []
        
        new_reg = Rigid(radius=RADIUS)

        self.slices_pred_covariance = dict()
        self.tmp_states=[]
        while stack_has_data:            
            
            pred_state = transition_matrix.dot(self.filtered_state_means[-1])
            estim_state = pred_state.copy()
            pred_covariance = self.filtered_state_covariances[-1] + transition_covariance
            #if not str(sl) in self.slices_pred_covariance:
            #    self.slices_pred_covariance[str(sl)] = np.eye(ndim_state, dtype=DTYPE) * self.iekf_init_state_cov
            #pred_covariance = self.slices_pred_covariance[str(sl)] + transition_covariance
            state_covariance = pred_covariance.copy()

            print 'frame %d slab %s'%(fr,str(sl)) + '_'*80

            convergence, niter = np.inf, 0
            self.all_convergences = []
            new_reg.param = estim_state[:6]

            mean_cost = np.inf
            if self._register_gradient:
                #slice_data_reg = sl_data - convolve1d(convolve1d(sl_data, [1/3.]*3,0),[1/3.]*3,1)
                # 2D DOG
                slice_data_reg = reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[0],d), [0,1], sl_data)-\
                                 reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[1],d), [0,1], sl_data)
            else:
                slice_data_reg = sl_data
                #slice_data_reg = reduce(lambda i,d: gaussian_filter1d(i,.5,d), [0,1], sl_data)
            while convergence > self.iekf_convergence and niter < self.iekf_max_iter:
                new_reg.param = estim_state[:6]
                self._sample_cost_jacobian(sl, slice_data_reg, new_reg, bias_corr=self._bias_correction)
                if self._nvox_in_slab_mask < self.iekf_min_nsamples_per_slab:
                    print 'not enough point'
                    break

                mask = self._slab_mask
                cost, jac = self._cost[0,mask], self._cost[1:,mask]

                S = jac.T.dot(pred_covariance).dot(jac) + np.diag(self.observation_variance[mask.flatten()])

                kalman_gain = np.dual.solve(S, pred_covariance.dot(jac).T, check_finite=False).T
                
                estim_state_old = estim_state.copy()
                estim_state[:] = estim_state + kalman_gain.dot(cost)

                mean_cost = (cost**2).mean()
                self.tmp_states.append(estim_state-pred_state)
                convergence = np.abs(estim_state_old-estim_state).max()
                self.all_convergences.append(convergence)
                niter += 1
                print ("%.5f %.5f : "+"% 2.5f,"*6)%((convergence, mean_cost) +tuple(estim_state[:6]))
                if niter==self.iekf_max_iter:
                    print "maximum iteration number exceeded"

            if niter==0:
                state_covariance[:] = self.filtered_state_covariances[-1]

            if niter>0:
                I_KH = np.eye(ndim_state) - np.dot(kalman_gain, jac.T)
                state_covariance[:] = np.dot(I_KH, state_covariance)
                
            self.niters.append(niter)
            self.filtered_state_means.append(estim_state)
            self.filtered_state_covariances.append(state_covariance)
            self.slices_pred_covariance[str(sl)] = state_covariance

            update = estim_state[:6]-self.filtered_state_means[-2][:6]
            new_reg.param = estim_state[:6]
            self.matrices.append(new_reg.as_affine())
            print 'nvox',self._nvox_in_slab_mask,'_'*100 + '\n'
            print 'update : ', ('% 2.5f,'*6)%tuple(update)
            print ('%.5f %.5f :' + '% 2.5f,'*6)%((convergence, mean_cost,)+tuple(estim_state[:6]))
            print '_'*100 + '\n'
            self._sample_cost_jacobian(sl, sl_data, new_reg, bias_corr=True)
            slab2anat = self._anat_reg.dot(aff).dot(self.matrices[-1])
            if yield_raw:
                yield fr, sl, slab2anat, sl_data/self._bias
            else:
                yield fr, sl, slab2anat
            last_frame = fr
            self._bias.fill(1)
            try:
                fr,sl,aff,tt,sl_data[:] = stack_it.next()
            except StopIteration:
                stack_has_data = False

        
    def _sample_cost_jacobian(self, sl, sl_data, new_reg, bias_corr=False):
        in_slice_axes = [d for d in range(sl_data.ndim) if d!= self.slice_axis]

        if self._slab_vox_idx is None or sl_data.shape[self.slice_axis]!= self._slab_vox_idx.shape[-2]:
            self._slab_vox_idx = np.empty(sl_data.shape+(sl_data.ndim,), dtype=np.int32)
            ## set vox idx for in-plane, does not change with slab
            for d in in_slice_axes:
                self._slab_vox_idx[...,d] = np.arange(sl_data.shape[d])[[
                    (slice(0,None) if d==d2 else None) for d2 in range(sl_data.ndim)]]

            self._slab_shift = np.zeros(sl_data.shape)
            self._slab_sigloss = np.zeros(sl_data.shape)
            self._anat_slab_coords = np.zeros(self._slab_vox_idx.shape, dtype=DTYPE)
            self._slab_coords = np.zeros((7,)+self._slab_vox_idx.shape, dtype=DTYPE)
            if self._register_gradient and False:
                self._interp_data = np.zeros((2,7,)+sl_data.shape, dtype=DTYPE)
            else:
                self._interp_data = np.zeros((7,)+sl_data.shape, dtype=DTYPE)
            self._bias = np.ones(sl_data.shape, dtype=DTYPE)
            self._cost = np.zeros((7,)+sl_data.shape, dtype=DTYPE)
            self._slab_mask = np.zeros(sl_data.shape, dtype=np.bool)
            self._slab_wm_weight = np.zeros(sl_data.shape, dtype=DTYPE)
        # set slice dimension vox idx (slice number)
        self._slab_vox_idx[...,self.slice_axis] = np.asarray(sl)[[
            (slice(0,None) if self.slice_axis==d2 else None) for d2 in range(sl_data.ndim)]]


        # compute interpolation coordinates for registration + jacobian delta step
        self._slab_coords[0] = apply_affine(new_reg.as_affine(), self._slab_vox_idx)
        for pi in range(6):
            reg_delta = Rigid(radius=RADIUS)
            params = new_reg.param.copy()
            params[pi] += self.iekf_jacobian_epsilon
            reg_delta.param = params
            self._slab_coords[pi+1] = apply_affine(reg_delta.as_affine(), self._slab_vox_idx)

        # compute coordinates in anat reference
        slab2anat = self._anat_reg.dot(self.affine).dot(new_reg.as_affine())
        self._anat_slab_coords[:] = apply_affine(slab2anat, self._slab_vox_idx)
        self._slab_shift = self.sample_shift(self._slab_vox_idx, slab2anat)
        phase_vec = slab2anat[:3,self.pe_dir] * self.fmap_scale * sl_data.shape[self.pe_dir]
        self._anat_slab_coords -= self._slab_shift[...,np.newaxis] * phase_vec
        # samplee epi pvf and mask
        if bias_corr:
            self.sample_ref(self.wm_weight_data_mask, self._anat_slab_coords, self._slab_wm_weight, self._slab_mask, order=1)
        else:
            self.sample_ref(None, self._anat_slab_coords, None, self._slab_mask, order=0)


        """
        slice1 = [slice(1,-1) if i==self.slice_axis else slice(None) for i in range(3)]
        slice2 = [slice(None,-2) if i==self.slice_axis else slice(None) for i in range(3)]
        slice3 = [slice(2,None) if i==self.slice_axis else slice(None) for i in range(3)]
        self._slab_shift *= 2 * np.pi # to  rad/sec... 
        lrgradients = np.asarray([
            self._slab_shift[slice1]-self._slab_shift[slice2],
            self._slab_shift[slice3]-self._slab_shift[slice1]])

        gbarte_2 = self.echo_time / 4.0 / np.pi
        sinc = np.sinc(gbarte_2*lrgradients)
        theta = np.pi * gbarte_2 * lrgradients
        re = 0.5 * (sinc[0]*np.cos(theta[0])+
                    sinc[1]*np.cos(theta[1]))
        im = 0.5 * (sinc[0]*np.sin(theta[0])+
                    sinc[1]*np.sin(theta[1]))
        self._slab_sigloss[slice1] = np.sqrt(re**2+im**2)

        self._slab_mask[self._slab_sigloss<.9] = False
        """


        if self._register_gradient and False:

            self._interp_data[:] = map_coordinates(
                self.register_refvol, 
                self._slab_coords.reshape(-1,3).T,
                mode='constant',
                order=1,
                cval=np.nan).reshape(self._interp_data.shape)


            data_mask = np.logical_not(np.any(np.isnan(self._interp_data[:,self._slab_mask]),0))
            self._slab_mask[self._slab_mask] = data_mask

            sl_data_grad = np.zeros(sl_data.shape+(2,))
            sl_data_grad[1:-1,:,0] = sl_data[2:]-sl_data[:-2]
            sl_data_grad[:,1:-1,0] = sl_data[:,2:]-sl_data[:,:-2]

            interp_data_grad = np.zeros(self._interp_data.shape+(2,))
            interp_data_grad[:,1:-1,:,0] = self._interp_data[:,2:]-self._interp_data[:,:-2]
            interp_data_grad[:,:,1:-1,0] = self._interp_data[:,:,2:]-self._interp_data[:,:,:-2]
            
            sl_data_grad_mag = np.sqrt((sl_data_grad**2).sum(-1)+1)
            interp_data_grad_mag =  np.sqrt((interp_data_grad**2).sum(-1)+1)
            
            self._cost[:] = np.minimum(sl_data_grad_mag,interp_data_grad_mag)*\
                            (sl_data_grad*interp_data_grad).sum(-1)/(sl_data_grad_mag*interp_data_grad_mag)
            self._cost[1:] -= self._cost[0]
            self._cost[1:] /= self.iekf_jacobian_epsilon

        else:
            self._interp_data[:, self._slab_mask] = map_coordinates(
                self.register_refvol, 
                self._slab_coords[:,self._slab_mask,:].reshape(-1,3).T,
                mode='constant',
                order=1,
                cval=np.nan).reshape(7,-1)

            data_mask = np.logical_not(np.any(np.isnan(self._interp_data[:,self._slab_mask]),0))
            self._slab_mask[self._slab_mask] = data_mask
            self._interp_data[:,~self._slab_mask] = 0
            if bias_corr:
                #self._slab_wm_weight *= self._slab_sigloss
                #"""
                weight_per_slice = np.apply_over_axes(np.sum, self._slab_wm_weight, in_slice_axes)
                if weight_per_slice.sum() > weight_per_slice.size*10:
                    sl_data_smooth = sl_data * self._slab_wm_weight
                    interp_data_smooth = self._interp_data[0] * self._slab_wm_weight
                    # use separability of gaussian filter
                    for d in in_slice_axes:
                        sl_data_smooth[:] = gaussian_filter1d(sl_data_smooth, self._bias_sigma, d,
                                                              mode='constant', truncate=10)
                        interp_data_smooth[:] = gaussian_filter1d(interp_data_smooth, self._bias_sigma, d, 
                                                                  mode='constant', truncate=10)
                    self._bias[:] = sl_data_smooth/interp_data_smooth
                    self._bias[interp_data_smooth<=0] = 1
                    self._bias[np.logical_or(np.isnan(self._bias),np.isinf(self._bias))] = 1
                    self._bias[...,weight_per_slice<20] = 1
                """
                self._slab_wm_weight += 1e-8*self._slab_mask
                self._bias[:] = np.exp(
                    reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d, truncate=10), [0,1], 
                           np.log(sl_data+1e-8)*self._slab_wm_weight)/\
                    reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d,truncate=10), [0,1], 
                           self._slab_wm_weight))
                """
            self._cost[0,self._slab_mask] = (sl_data[self._slab_mask]/self._bias[self._slab_mask] - 
                                             self._interp_data[0,self._slab_mask])
            self._cost[1:,self._slab_mask] = (self._interp_data[1:,self._slab_mask] - self._interp_data[0,self._slab_mask])/\
                                             self.iekf_jacobian_epsilon
            
        if np.any(~np.isfinite(self._cost)):
            raise RuntimeError
        self._nvox_in_slab_mask = self._slab_mask.sum()
            

    def correct(self, realigned, pvmaps, frame_shape, sig_smth=16, white_idx=1,
                maxiter = 16, residual_tol = 2e-3, n_samples_min = 30):
        
        float_mask = nb.Nifti1Image(
            self.mask.get_data().astype(np.float32),
            self.mask.get_affine())

        ext_mask = self.mask.get_data()>0
        ext_mask[pvmaps.get_data()[...,:2].sum(-1)>0.1] = True

        cdata = None
        for fr, slab, reg, data in realigned:
            if cdata is None: # init all
                cdata = np.zeros(data.shape)
                cdata2 = np.zeros(data.shape)
                sigloss = np.zeros(frame_shape)
                epi_pvf = np.empty(frame_shape+(pvmaps.shape[-1],))
                epi_mask = np.zeros(frame_shape, dtype=np.bool)
                prev_reg = np.inf
                white_wght = np.empty(data.shape[:2])
                smooth_white_wght = np.empty(data.shape[:2])
                res = np.empty(data.shape[:2])
                bias = np.empty(data.shape)
            if not np.allclose(prev_reg, reg, 1e-6):
                prev_reg = reg
                #print 'sample pvf and mask'
                epi_pvf[:] = self.inv_resample(
                    pvmaps, reg, frame_shape, -1,
                    mask = ext_mask)
#                epi_mask[:] = self.inv_resample(self.mask, reg, frame_shape, 0)
                epi_mask[:] = self.inv_resample(self.mask, reg, frame_shape, order=-2, mask=ext_mask)>0
                """
                epi_coords = nb.affines.apply_affine(
                    reg,
                    np.rollaxis(np.mgrid[[slice(0,n) for n in frame_shape]],0,4))
                sigloss[:] = compute_sigloss(
                    self.fmap, self.fieldmap_reg,
                    fmap_mask,
                    np.eye(4), reg,
                    epi_coords,
                    self.echo_time,
                    slicing_axis=self.slice_axis)
                """

            cdata.fill(0)
            cdata2.fill(1)
            bias.fill(1)
            for sli,sln in enumerate(slab):
                sl_mask = epi_mask[...,sln]
                sl_proc_mask = sl_mask.copy()
                sl_proc_mask[data[...,sli]<=0] = False
                
                niter = 0
                res.fill(0)
                #print epi_pvf[sl_mask,sln].sum(0)
                regs_subset = epi_pvf[sl_proc_mask,sln].sum(0) > 10
                sl_proc_mask[epi_pvf[...,sln,regs_subset].sum(-1)<=0] = False

                n_sl_samples = np.count_nonzero(sl_proc_mask)
                if n_sl_samples < n_samples_min:
                    print 'not enough samples (%d) skipping slice %d'%(epi_mask[...,sln].sum(),sln)
                    if n_sl_samples > 0:
                        cdata2[sl_mask,sli] = data[sl_mask,sli] / cdata[sl_mask,sli].mean()
                    else:
                        cdata2[sl_mask,sli] = 1
                    continue

                cdata[...,sli] = data[...,sli]
                bias[...,sli].fill(1)
                tmp_res = np.empty(n_sl_samples)

                white_wght[:] = epi_pvf[..., sln, white_idx]*sl_proc_mask

                smooth_white_wght[:] = scipy.ndimage.filters.gaussian_filter(white_wght, sig_smth, mode='constant')
                smooth_white_wght[np.logical_and(smooth_white_wght==0,sl_mask)] = 1e-8

                while niter<maxiter:
                    means = (cdata[sl_proc_mask,sli,np.newaxis]*epi_pvf[sl_proc_mask,sln]).sum(0)/\
                            epi_pvf[sl_proc_mask,sln].sum(0)
                    tmp_res[:] = np.log(cdata[sl_proc_mask,sli]/(means*epi_pvf[sl_proc_mask,sln])[...,regs_subset].sum(-1))
                    if np.count_nonzero(np.isnan(tmp_res))>0 or np.count_nonzero(np.isinf(tmp_res))>0:
                        raise RuntimeError
                    res_var = tmp_res.var()
                    print ('%d\t%s\t'+'% 4.5f\t'*len(means)+'%.5f\t%d')%((fr,str(slab))+tuple(means)+(res_var,niter))
                    if res_var < residual_tol:
                        break
                    res.fill(0)
                    res[sl_proc_mask] = tmp_res
                    res[:] = scipy.ndimage.filters.gaussian_filter(res*white_wght,sig_smth,mode='constant')/smooth_white_wght

                    bias[...,sli] *= np.exp(-res+res[sl_proc_mask].mean())
                    cdata[...,sli] = data[...,sli] * bias[...,sli]
                    niter+=1
                means = (cdata[sl_proc_mask,sli,np.newaxis]*epi_pvf[sl_proc_mask,sln]).sum(0)/\
                        epi_pvf[sl_proc_mask,sln].sum(0)
                cdata[sl_proc_mask,sli] /= (means*epi_pvf[sl_proc_mask,sln])[...,regs_subset].sum(-1)
                cdata[~sl_proc_mask,sli] = 1

            print cdata.min(),cdata.max()
            yield fr, slab, reg, cdata.copy(),bias.copy()
        return
            
def filenames_to_dicoms(fnames):
    for f in fnames:
        yield dicom.read_file(f)

class NiftiIterator():

    def __init__(self, nii, mb=1):
        
        self.nii = nii
        self.nslices,self.nframes = self.nii.shape[2:4]
        self._affine = self.nii.affine
        self._voxel_size = np.asarray(self.nii.header.get_zooms()[:3])
        self._slice_order = np.arange(self.nslices)
        self._shape = self.nii.shape
        if mb == 1: 
            self._nshots = self.nslices
            self._slabs = [[s] for s in np.arange(self.nslices)]
        else:
            self._nshots = self.nslices / mb
            nincr = 2
            if self._nshots%2 == 0:
                nincr = self._nshots/2 -1
                if nincr%2 == 0:
                    nincr -= 1
            self._slabs = [(np.arange(mb)*self._nshots+sl).tolist() for sl in (np.arange(self._nshots)*nincr)%self._nshots]
                
        self._slab_trigger_times = np.arange(self._nshots)*self.nii.header.get_zooms()[3]/float(self._nshots)
        
    def iter_frame(self, data=True, queue_dicoms=False):
        data = self.nii.get_data()
        for t in range(data.shape[3]):
            yield t, self.nii.affine, data[:,:,:,t]
        del data

    def iter_slabs(self, data=True, queue_dicoms=False):
        data = self.nii.get_data()
        for fr in range(data.shape[3]):
            for sl,tt in zip(self._slabs, self._slab_trigger_times):
                yield fr, sl, self.nii.affine, tt, data[:,:,sl,fr]
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
