import numpy as np
import numpy.linalg as npl

import nibabel as nb, dicom
from nibabel.affines import apply_affine
from ...fixes.nibabel import io_orientation
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .affine import Rigid, Affine, rotation_vec2mat, to_matrix44

from .optimizer import configure_optimizer, use_derivatives
from scipy.optimize import fmin_slsqp
from scipy import interp
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


def apply_affine2(aff, pts):
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((shape[0], -1))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1, :-1]
    trans = aff[:-1, -1]
    res = np.dot(rzs, pts) + trans[:, np.newaxis]
    return res.reshape(shape)

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
                 recenter_fmap_data=False,
                 unmask_fmap=False):


        self.fmap, self.mask = fieldmap, mask
        self.mask_data = self.mask.get_data()>0
        self.fieldmap_reg = fieldmap_reg

        self.slice_axis = slice_axis
        self.in_slice_axes = [d for d in range(3) if d!= self.slice_axis]


        self.slice_order = slice_order
        self.pe_sign = int(phase_encoding_dir > 0)*2-1
        self.pe_dir = abs(phase_encoding_dir)-1
        self.fe_dir = [d for d in range(3) if d!=self.slice_axis and d!=self.pe_dir][0]
        self.repetition_time = repetition_time
        self.echo_time = echo_time
        self.slice_tr = slice_repetition_time
        self.interleaved = int(interleaved)
        self.slice_trigger_times = slice_trigger_times
        self.slice_thickness = slice_thickness

        if self.fmap is not None:
            self.recenter_fmap_data = recenter_fmap_data
            self._unmask_fmap = unmask_fmap
            self._preproc_fmap()

        self.fmap_scale = self.pe_sign*echo_spacing/2.0/np.pi
        self._resample_fmap_values = None
        self.st_ratio = 1

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
            data_out[~mask] = fill_value
            
            
    def sample_shift(self, vox_in, aff):
        vox = apply_affine(npl.inv(self.fieldmap_reg.dot(self.fmap.affine)).dot(aff), vox_in)
        shift = map_coordinates(
            self.fmap.get_data(),
            vox.reshape(-1,3).T,
            mode='reflect',
            order=1).reshape(vox_in.shape[:-1])
        return shift

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
                             normals=None,
                             pve_map = None,
                             rbf_sigma=1, 
                             kneigh_dens=None,
                             mask=True):
        dist_ub = rbf_sigma*3 # 3 std in all directions
        if kneigh_dens is None:
            kneigh_dens = int((2*dist_ub)**3) # all points enclosed in a cube

        nslices = data[0].shape[2]*len(slabs)
        vol_shape = data[0].shape[:2]+(nslices,)
        if (not hasattr(self,'_scat_resam_rbf_coords') or
            not self._scat_resam_rbf_coords is coords):
            self._scat_resam_rbf_coords = coords
            print('recompute KDTree')
            self._coords_kdtree = KDTree(coords)
        coords_kdtree = self._coords_kdtree 
        out.fill(0)
        out_weights = np.zeros(len(coords))

        voxs = np.rollaxis(np.mgrid[[slice(0,d) for d in vol_shape]],0,4)
        slab_mask = np.zeros_like(data[0], dtype=np.bool)
        if not pve_map is None:
            pve_data = pve_map.get_data()
            slab_pve = np.zeros_like(slab_mask, dtype=pve_data.dtype)
        ## could/should we avoid loop here and do all slices in the meantime ?
        for sl, d, t in zip(slabs, data, transforms):
            points = apply_affine(t, voxs[...,sl,:])
            if not self.fmap is None:
                slab_shift = self.sample_shift(voxs[...,sl,:], t)
                phase_vec = t[:3,self.pe_dir] * self.fmap_scale * d.shape[self.pe_dir]
                points -= slab_shift[...,np.newaxis] * phase_vec

            if mask:
                if not pve_map is None:
                    self.sample_ref(pve_data[self.mask_data], points, slab_pve, slab_mask, order=1)
                else:
                    self.sample_ref(None, points, None, slab_mask, order=0)
                points = points[slab_mask]
                d = d[slab_mask]
                if np.count_nonzero(slab_mask) < 1:
                    continue
            else:
                d = d.ravel()
            dists, idx = coords_kdtree.query(points.reshape(-1,3), k=kneigh_dens, distance_upper_bound=dist_ub)
            not_inf = np.isfinite(dists)
            idx2 = (np.ones(kneigh_dens,dtype=np.int)*np.arange(len(d))[:,np.newaxis])[not_inf]
            idx = idx[not_inf]
            dists = dists[not_inf]
            weights = np.exp(-(dists/rbf_sigma)**2)
            if not normals is None:
                # perform anisotropic kernel RBF
                # the kernel is configure so that 2SD is reached at surface
                normal_norm_sq = (normals[idx]**2).sum(-1)
                constrained_mask = normal_norm_sq > 0
                dists_proj_norm = (((coords[idx]-points[idx2])*(normals[idx])).sum(-1)/normal_norm_sq)[constrained_mask]
                weights[constrained_mask] *= np.exp(-(dists_proj_norm/.5)**2)
            if not pve_map is None:
                weights *= slab_pve[slab_mask][idx2]
            non0 = weights > 1e-5 # truncate
            np.add.at(out, idx[non0], d[idx2[non0]]*weights[non0])
            np.add.at(out_weights, idx[non0], weights[non0])

        out /= out_weights
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
        phase_vec /= npl.norm(phase_vec)
        epi_mask = slice(0, None)
        if mask:
            epi_mask = self.inv_resample(self.mask, transforms[0], vol.shape, -1, self.mask_data)>0
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
    
    def _slice(self, pe_dir=slice(None), fe_dir=slice(None), slice_axis=slice(None)):
        slices = [None,]*3
        slices[self.pe_dir] = pe_dir
        slices[self.fe_dir] = fe_dir
        slices[self.slice_axis] = slice_axis
        return tuple(slices)
    
    def _preproc_fmap(self):
        if self.fieldmap_reg is None:
            self.fieldmap_reg = np.eye(4)
        self.fmap2world = np.dot(self.fieldmap_reg, self.fmap.affine)
        self.world2fmap = npl.inv(self.fmap2world)
        grid = apply_affine(
            npl.inv(self.mask.affine).dot(self.fmap2world),
            np.rollaxis(np.mgrid[[slice(0,n) for n in self.fmap.shape]],0,4))
        self.fmap_mask = map_coordinates(
            self.mask_data,
            grid.reshape(-1,3).T, order=0).reshape(self.fmap.shape) > 0
        fmap_data = self.fmap.get_data()
        if self.recenter_fmap_data: #recenter the fieldmap range to avoid shift
            fmap_data -= fmap_data[self.fmap_mask].mean()
        if self._unmask_fmap:
            fmap_unmask = np.empty_like(fmap_data)
            fmap_unmask.fill(np.nan)
            for sl in range(fmap_data.shape[self.slice_axis]):
                for fe in range(fmap_data.shape[self.fe_dir]):
                    line_slice = self._slice(slice_axis=sl,fe_dir=fe)
                    line_mask = self.fmap_mask[line_slice]
                    if np.count_nonzero(line_mask)>0:
                        fmap_unmask[line_slice] = interp(
                            range(fmap_data.shape[self.pe_dir]),
                            np.argwhere(line_mask).ravel(),
                            fmap_data[self._slice(slice_axis=sl,fe_dir=fe,pe_dir=line_mask)])
                for pe in range(fmap_data.shape[self.pe_dir]):
                    line_slice = self._slice(slice_axis=sl,pe_dir=pe)
                    line_mask = np.isfinite(fmap_unmask[line_slice])
                    if np.count_nonzero(line_mask)>0:
                        fmap_unmask[line_slice] = interp(
                            range(fmap_data.shape[self.fe_dir]),
                            np.argwhere(line_mask).ravel(),
                            fmap_unmask[self._slice(slice_axis=sl,pe_dir=pe,fe_dir=line_mask)])
            sa_mask = np.argwhere(np.apply_over_axes(np.sum,self.fmap_mask,self.in_slice_axes).ravel()>0).ravel()
            fmap_unmask[self._slice(slice_axis=slice(None,sa_mask[0]))] = \
                fmap_unmask[self._slice(slice_axis=sa_mask[:1])]
            fmap_unmask[self._slice(slice_axis=slice(sa_mask[-1]+1,None))] = \
                fmap_unmask[self._slice(slice_axis=sa_mask[-1:])]
            fmap_data[:] = fmap_unmask
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
            wld2epi = npl.inv(affines[0][1])
            interp_coords[:] = apply_affine(wld2epi, coords)
            if not self._resample_fmap_values is None:
                interp_coords[...,self.pe_dir] += self._resample_fmap_values
        else: # we have to solve which transform we sample with
            t = affines[0][0][1][0]
            tmp_coords = np.empty(coords.shape)
            subset = np.ones(coords.shape[:-1], dtype=np.bool)
            interp_coords.fill(np.nan) # just to check, to be removed
            for slab,trans in affines:
                wld2epi = npl.inv(trans)
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

        fmap2fmri = npl.inv(affine).dot(self.fmap2world)
        coords = apply_affine(
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
            vol2epi = npl.inv(affine).dot(vol.affine)
            voxs = apply_affine(vol2epi, grid)
            if self.fmap is not None:
                vol2fmap = self.world2fmap.dot(vol.affine)
                fmap_voxs = apply_affine(vol2fmap, grid)
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
            epi2vol = npl.inv(vol.affine).dot(affine)
            voxs = apply_affine(epi2vol, grid)

            for v in range(nvols):
                rvol[...,v] = map_coordinates(
                    vol_data[...,v],
                    voxs.reshape(-1,3).T, order=order).reshape(shape)
        del grid, voxs
        return np.squeeze(rvol)


import dipy.align
import dipy.align.parzenhist
import dipy.align.scalespace
import dipy.core.optimize
from dipy.align import floating
import dipy.align.vector_fields as vfu
from dipy.align.imwarp import get_direction_and_spacings

class EPIOnlineRealignUndistort(EPIOnlineResample):
    
    def __init__(self,
                 ref,
                 init_reg,
                 cc_radius = 1,

                 gradient_descent_tol = 1e-3,
                 
                 orkf_init_covariance = 1e-1,
                 orkf_transition_cov = 1e-1,
                 orkf_observation_covariance = 1e-1,
                 orkf_convergence = 1e-3,
                 orkf_jacobian_epsilon = 1e-8,

                 level_iters = [8,8,4],
                 step_length = .25,
                 def_field_convergence = 1e-5,
                 def_field_min_level = 1, # by default field is subsampled by 2
                 ss_sigma_factor = .5,
                 inv_iter = 20,
                 inv_tol = 1e-3,
                 reg_sigma_diff = 2.,
                 def_update_n_iter = 16,
                 def_update_inv_iter = 10,
                 **kwargs):
        super(EPIOnlineRealignUndistort, self).__init__(**kwargs)

        self.dim = 3
        self._ref = ref
        self._ref_data = self._ref.get_data().astype(DTYPE)

        self.mask_data_dil = binary_dilation(self.mask_data, None, 2)
        self.mask_data_int = self.mask_data.astype(np.int32)
        self._init_reg = init_reg #Rigid(radius=RADIUS)

        self._cc_radius = cc_radius
        self._level_iters = level_iters
        self._levels = len(level_iters)
        self._ss_sigma_factor = ss_sigma_factor
        self.energy_window = 12
        self.step_length = step_length
        self.inv_iter = inv_iter
        self.inv_tol = inv_tol
        self.reg_sigma_diff = reg_sigma_diff
        self.def_field_convergence = def_field_convergence
        self.def_field_min_level = def_field_min_level
        self.def_update_n_iter = def_update_n_iter
        self.def_update_inv_iter = def_update_inv_iter

        self._gradient_descent_tol = gradient_descent_tol

        self._nmin_samples_per_slab = 30

        self.orkf_convergence = orkf_convergence
        self.orkf_transition_cov = orkf_transition_cov
        self.orkf_init_covariance = orkf_init_covariance
        self.orkf_observation_covariance = orkf_observation_covariance
        self.orkf_jacobian_epsilon = orkf_jacobian_epsilon

        self._ref_scale_space = dipy.align.scalespace.ScaleSpace(
            self._ref_data.astype(floating), self._levels, self._ref.affine,
            self._ref.header.get_zooms(), self._ss_sigma_factor,
            False)

        
    def _diffeo_blip_map(self, blip_up_stack, blip_down_stack, blip_up_reg, blip_down_reg, reg=.8):

        _, blip_up_affine, blip_up_data = blip_up_stack.iter_frame(queue_dicoms=True).next()
        _, blip_down_affine, blip_down_data = blip_down_stack.iter_frame(queue_dicoms=True).next()

        blip_up_grid2world = blip_up_reg.dot(blip_up_affine)
        blip_down_grid2world = blip_down_reg.dot(blip_down_affine)

        ref_direction, ref_spacing = get_direction_and_spacings(self._ref.affine, self.dim)
        blip_up_direction, blip_up_spacing = get_direction_and_spacings(blip_up_grid2world, self.dim)
        blip_down_direction, blip_down_spacing = get_direction_and_spacings(blip_down_grid2world, self.dim)
        

        ref_direction_mtx = np.eye(self.dim + 1)
        blip_up_direction_mtx = np.eye(self.dim + 1)
        blip_down_direction_mtx = np.eye(self.dim + 1)
        ref_direction_mtx[:self.dim, :self.dim] = ref_direction
        blip_up_direction_mtx[:self.dim, :self.dim] = blip_up_direction
        blip_down_direction_mtx[:self.dim, :self.dim] = blip_down_direction

        # get pe_dir unit vector in ref space
        blip_up_pedir_ref = blip_up_grid2world[:3,self.pe_dir].copy()
        blip_up_pedir_ref /= npl.norm(blip_up_pedir_ref)
        blip_down_pedir_ref = blip_down_grid2world[:3,self.pe_dir].copy()
        blip_down_pedir_ref /= npl.norm(blip_down_pedir_ref)
                
        blip_up_ss = dipy.align.scalespace.ScaleSpace(
            blip_up_data.astype(floating), self._levels, blip_up_grid2world,
            blip_up_stack._voxel_size, self._ss_sigma_factor,
            False)

        blip_down_ss = dipy.align.scalespace.ScaleSpace(
            blip_down_data.astype(floating), self._levels, blip_down_grid2world,
            blip_down_stack._voxel_size, self._ss_sigma_factor,
            False)

        self.full_energy_profile = []

        for level in range(self._levels-1,self.def_field_min_level-1,-1):
            print('levels %d'%level)

            current_disp_shape = tuple(self._ref_scale_space.get_domain_shape(level))
            current_disp_shape_ar = np.asarray(current_disp_shape)
            current_disp_grid2world = self._ref_scale_space.get_affine(level)
            current_disp_world2grid = npl.inv(current_disp_grid2world)
            current_disp_spacing = np.asarray(self._ref_scale_space.get_spacing(level),dtype=np.float)

            ref2blip_up = npl.inv(blip_up_grid2world).dot(current_disp_grid2world)
            ref2blip_down = npl.inv(blip_down_grid2world).dot(current_disp_grid2world)
            blip_up2ref = npl.inv(ref2blip_up)
            blip_down2ref = npl.inv(ref2blip_down)
                    
            if level < self._levels-1:
                expand_factors = self._ref_scale_space.get_expand_factors(level+1, level).astype(np.double)
                new_shift_up = vfu.resample_displacement_field_3d(shift_up, expand_factors, current_disp_shape_ar)
                new_shift_down = vfu.resample_displacement_field_3d(shift_down, expand_factors, current_disp_shape_ar)
                new_inv_shift_up = vfu.resample_displacement_field_3d(inv_shift_up ,expand_factors, current_disp_shape_ar)
                new_inv_shift_down = vfu.resample_displacement_field_3d(inv_shift_down, expand_factors,
                                                                        current_disp_shape_ar)

                del shift_up, shift_down, inv_shift_up, inv_shift_down
                shift_up = new_shift_up.astype(floating)
                shift_down = new_shift_down.astype(floating)
                inv_shift_up = new_inv_shift_up.astype(floating)
                inv_shift_down = new_inv_shift_down.astype(floating)

            else:
                shift_up = np.zeros(current_disp_shape+(3,), dtype=floating)
                shift_down = np.zeros_like(shift_up)
                inv_shift_up = np.zeros_like(shift_up)
                inv_shift_down = np.zeros_like(shift_up)
                
            step_up = np.empty_like(shift_up, dtype=floating)
            step_down = np.empty_like(step_up)
            step_up_eqcon = np.empty_like(step_up)
            step_down_eqcon = np.empty_like(step_up)
        
            dxi4_dk = np.empty(current_disp_shape, dtype=floating)

            jac_blip_up = np.empty((3,3,)+current_disp_shape, dtype=floating)
            jac_blip_down = np.empty((3,3,)+current_disp_shape, dtype=floating)
            jac_det_blip_up = np.empty_like(dxi4_dk)
            jac_det_blip_down = np.empty_like(dxi4_dk)

            blip_up_down_sum = np.empty_like(dxi4_dk)
            blip_up_down_geomean = np.empty_like(dxi4_dk)

            corr = np.empty_like(dxi4_dk)

            blip_up_interp = np.empty_like(dxi4_dk)
            blip_down_interp = np.empty_like(dxi4_dk)

            ref2subsamp = npl.inv(self._ref.affine).dot(current_disp_grid2world)
            ref_data = vfu.transform_3d_affine(self._ref_scale_space.get_image(level),
                                               current_disp_shape_ar, ref2subsamp)
            msk = vfu.transform_3d_affine_nn(self.mask_data_int,
                                             current_disp_shape_ar, ref2subsamp)>0
            
            ks_fact = np.empty(current_disp_shape+(5,), dtype=floating)

            derivative = np.inf
            self.energy_list = []

            niter = 0
            #while conv > 1e-5: #self.def_field_convergence:
            while ((niter < self._level_iters[-level-1]) and
                   (self.def_field_convergence < derivative)):
                niter += 1

                jac(shift_up, current_disp_spacing, jac_blip_up)
                jac(shift_down, current_disp_spacing, jac_blip_down)            
                jac_det(jac_blip_up, jac_det_blip_up)
                jac_det(jac_blip_down, jac_det_blip_down)

                blip_up_interp[:] = vfu.warp_3d(
                    blip_up_ss.get_image(level), shift_up,
                    np.eye(4), ref2blip_up, npl.inv(blip_up_affine))
                # np.eye(4): field and target have same affine

                blip_down_interp[:] = vfu.warp_3d(
                    blip_down_ss.get_image(level), shift_down,
                    np.eye(4), ref2blip_down, npl.inv(blip_down_affine))

                # geomean
                blip_up_down_sum[:] = blip_up_interp + blip_down_interp
                blip_up_down_geomean[:] = 2*(blip_up_interp*blip_down_interp)/blip_up_down_sum
                blip_up_down_geomean[blip_up_down_sum < 1e-5] = 0

                for gi,gg in enumerate(np.gradient(blip_up_interp)):
                    step_up[...,gi] = gg
                for gi,gg in enumerate(np.gradient(blip_down_interp)):
                    step_down[...,gi] = gg

                step_up /= current_disp_spacing
                vfu.reorient_vector_field_3d(step_up, ref_direction_mtx)
                step_down /= current_disp_spacing
                vfu.reorient_vector_field_3d(step_down, ref_direction_mtx)                    

                cost = 34
                # cost using anatomical image with redistributed epi data
                if cost==3:
                    ks_fact[:] = dipy.align.crosscorr.precompute_cc_factors_3d(
                        ref_data,
                        blip_up_down_geomean,
                        self._cc_radius)

                    dxi4_dk[:] = ks_fact[...,2] / (ks_fact[...,3] * ks_fact[...,4]) * \
                                  (ks_fact[...,0] - ks_fact[...,2]/ ks_fact[...,4] * ks_fact[...,1])
                    step_up *= (np.square(blip_down_interp/blip_up_down_sum) * jac_det_blip_up * dxi4_dk)[...,np.newaxis]
                    step_down *= (np.square(blip_up_interp/blip_up_down_sum) * jac_det_blip_down * dxi4_dk)[...,np.newaxis]

                    corr[:] = np.square(ks_fact[...,2])/(ks_fact[...,3]*ks_fact[...,4])
                # cost not using anatomical image except for def space
                if cost==4:
                    ks_fact[:] = dipy.align.crosscorr.precompute_cc_factors_3d(
                        blip_up_interp,
                        blip_down_interp,
                        self._cc_radius)
                    corr[:] = ks_fact[...,2]/(ks_fact[...,3]*ks_fact[...,4])
                    
                    step_up *= (corr * jac_det_blip_up *
                                (ks_fact[...,1]-ks_fact[...,2]/ks_fact[...,3]*ks_fact[...,0]))[...,np.newaxis]
                    step_down *= (corr * jac_det_blip_down *
                                  (ks_fact[...,0]-ks_fact[...,2]/ks_fact[...,4]*ks_fact[...,1]))[...,np.newaxis]
                    corr *= ks_fact[...,2]
                # cost that combines the 2 costs above
                if cost==34:
                    ks_fact[:] = dipy.align.crosscorr.precompute_cc_factors_3d(
                        ref_data,
                        blip_up_down_geomean,
                        self._cc_radius)

                    dxi4_dk[:] = ks_fact[...,2] / (ks_fact[...,3] * ks_fact[...,4]) * \
                                 (ks_fact[...,0] - ks_fact[...,2]/ ks_fact[...,4] * ks_fact[...,1])

                    ks_fact[:] = dipy.align.crosscorr.precompute_cc_factors_3d(
                        blip_up_interp,
                        blip_down_interp,
                        self._cc_radius)
                    
                    corr[:] = ks_fact[...,2]/(ks_fact[...,3]*ks_fact[...,4])
                    step_up *= (jac_det_blip_up * (
                        corr * (ks_fact[...,1]-ks_fact[...,2]/ks_fact[...,3]*ks_fact[...,0])#+\
                        #np.square(blip_down_interp/blip_up_down_sum) * dxi4_dk
                    ))[...,np.newaxis]
                    step_down *= (jac_det_blip_down * (
                        corr * (ks_fact[...,0]-ks_fact[...,2]/ks_fact[...,4]*ks_fact[...,1])#+
                        #np.square(blip_up_interp/blip_up_down_sum) * dxi4_dk
                    ))[...,np.newaxis]

                    corr *= ks_fact[...,2]
                    
                    
                step_up[~np.isfinite(step_up)] = 0
                step_down[~np.isfinite(step_down)] = 0
                corr[~np.isfinite(corr)] = 0

                for a in [step_up, step_down, corr]:
                    for b in [slice(None,self._cc_radius),slice(-self._cc_radius,None)]:
                        a[b] = 0
                        a[:,b] = 0
                        a[:,:,b] = 0
                
                step_up *= msk[...,np.newaxis]
                step_down *= msk[...,np.newaxis]
                corr *= msk

                for di in range(3):
                    gaussian_filter1d(step_up, self.reg_sigma_diff, axis=di, output=step_up)
                    gaussian_filter1d(step_down, self.reg_sigma_diff, axis=di, output=step_down)

                for a in [step_up, step_down]:
                    for b in [0,-1]:
                        a[b] = 0
                        a[:,b] = 0
                        a[:,:,b] = 0
                ## Soft equality constraint
                if reg==0:
                    step_up_eqcon[:] = step_up
                    step_down_eqcon[:] = step_down
                else:
                    step_up_eqcon[:] = (1-reg/2.) * step_up
                    step_down_eqcon[:] = (1-reg/2.) * step_down
                    #rtup*rdown to take into account registration in deformation equality reg
                    vfu.reorient_vector_field_3d(step_up, npl.inv(blip_down_direction_mtx).dot(blip_up_direction_mtx))
                    vfu.reorient_vector_field_3d(step_down, npl.inv(blip_up_direction_mtx).dot(blip_down_direction_mtx))
                    step_up_eqcon -= reg/2. * step_down
                    step_down_eqcon -= reg/2. * step_up
                    
                # project on pedir
                step_up_eqcon[:] = (step_up_eqcon*blip_up_pedir_ref).sum(-1)[...,np.newaxis]*blip_up_pedir_ref
                step_down_eqcon[:] = (step_down_eqcon*blip_down_pedir_ref).sum(-1)[...,np.newaxis]*blip_down_pedir_ref

                norm_step_up = np.sqrt(np.sum((step_up_eqcon/current_disp_spacing) ** 2, -1)).max()
                norm_step_down = np.sqrt(np.sum((step_down_eqcon/current_disp_spacing) ** 2, -1)).max()

                if norm_step_up > 0:
                    step_up_eqcon /= norm_step_up
                if norm_step_down > 0:
                    step_down_eqcon /= norm_step_down

                vfu.compose_vector_fields_3d(shift_up, step_up_eqcon, None, current_disp_world2grid,
                                             self.step_length, shift_up)
                vfu.compose_vector_fields_3d(shift_down, step_down_eqcon, None, current_disp_world2grid,
                                             self.step_length, shift_down)
                
                inv_shift_up[:] = vfu.invert_vector_field_fixed_point_3d(
                    shift_up,
                    current_disp_world2grid, current_disp_spacing,
                    self.inv_iter, self.inv_tol,
                    inv_shift_up)
                
                inv_shift_down[:] = vfu.invert_vector_field_fixed_point_3d(
                    shift_down,
                    current_disp_world2grid, current_disp_spacing,
                    self.inv_iter, self.inv_tol,
                    inv_shift_down)

                shift_up[:] = vfu.invert_vector_field_fixed_point_3d(
                    inv_shift_up,
                    current_disp_world2grid, current_disp_spacing,
                    self.inv_iter, self.inv_tol,
                    shift_up)
                
                shift_down[:] = vfu.invert_vector_field_fixed_point_3d(
                    inv_shift_down,
                    current_disp_world2grid, current_disp_spacing,
                    self.inv_iter, self.inv_tol,
                    shift_down)

                corr[np.logical_or(corr>1,corr<-1)] = 0
                energy = -(corr[msk].sum())

                if len(self.energy_list) >= self.energy_window:
                    derivative = self._get_energy_derivative()

                self.energy_list.append(energy)
                print(energy, derivative,
                      np.abs(shift_up).max(), np.abs(shift_down).max(),
                      jac_det_blip_up.max(), jac_det_blip_up.min())
                
            del ks_fact, step_up, step_down, step_up_eqcon, step_down_eqcon, blip_up_down_sum, dxi4_dk

            if level > self.def_field_min_level:
                del blip_up_down_geomean, corr, blip_up_interp, blip_down_interp
            self.full_energy_profile.extend(self.energy_list)
        
        # last jacobian update for output
        jac(shift_up, current_disp_spacing, jac_blip_up)
        jac(shift_down, current_disp_spacing, jac_blip_down)        
        jac_det(jac_blip_up, jac_det_blip_up)
        jac_det(jac_blip_down, jac_det_blip_down)

        return shift_up, shift_down, inv_shift_up, inv_shift_down, blip_up_down_geomean,\
            blip_up_interp, blip_down_interp,\
            corr, jac_det_blip_up, jac_det_blip_down, current_disp_grid2world

    # from dipy imwarp
    def _approximate_derivative_direct(self, x, y):
        r"""Derivative of the degree-2 polynomial fit of the given x, y pairs
        Directly computes the derivative of the least-squares-fit quadratic
        function estimated from (x[...],y[...]) pairs.
        Parameters
        ----------
        x : array, shape (n,)
            increasing array representing the x-coordinates of the points to
            be fit
        y : array, shape (n,)
            array representing the y-coordinates of the points to be fit
        Returns
        -------
        y0 : float
            the estimated derivative at x0 = 0.5*len(x)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        X = np.row_stack((x**2, x, np.ones_like(x)))
        XX = (X).dot(X.T)
        b = X.dot(y)
        beta = npl.solve(XX, b)
        x0 = 0.5 * len(x)
        y0 = 2.0 * beta[0] * (x0) + beta[1]
        return y0

    def _get_energy_derivative(self):
        r"""Approximate derivative of the energy profile
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            raise ValueError('Not enough data to fit the energy profile')
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if(ss > 0):
            ss *= -1
        y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x, y)
        return der

    def _update_def_field(self, df, sl, param, conv, max_iter):
        print 'update def field'
        if np.count_nonzero(self._slab_mask) < self._nmin_samples_per_slab:
            print 'not enough sample'
            return
        nrgy_list = []
        self._df_epi_step_up.fill(0)

        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(param * self._precond)
        static_grid2world = current_affine.dot(self._init_reg).dot(self.affine)
        static_world2grid = npl.inv(static_grid2world)
#        slab_grid2world = current_affine.dot(self._init_reg).dot(self.affine).dot(self._slab_affine)
#        slab_world2grid = npl.inv(slab_grid2world)
        moving_grid2world = self._ref.affine
        shift_grid2world = df.affine
        shift_world2grid = npl.inv(shift_grid2world)
        shift_spacing = np.asarray(df.header.get_zooms()[:3], dtype=np.double)
        
        shift_grid2static_grid = static_world2grid.dot(shift_grid2world)
#        shift_grid2slab_grid = slab_world2grid.dot(shift_grid2world)
        
        fw_step = np.zeros(self.sl_data.shape+(self.dim,), dtype=floating)

        derivative = np.inf
        niter = 0

        slab_direction, slab_spacing = get_direction_and_spacings(static_grid2world, self.dim)
        slab_direction_mtx = np.eye(self.dim + 1)
        slab_direction_mtx[:self.dim, :self.dim] = slab_direction

        static_pedir_ref = static_grid2world[:3,self.pe_dir].copy()
        static_pedir_ref /= np.sqrt(np.square(static_pedir_ref).sum())

        self.energy_list = []
        derivative = np.inf

        while ((niter < max_iter) and
               (conv < derivative)):
            niter += 1

            corr, temp = self._compute_cc_cost_gradient(param, df, rigid_gradient=False)

            self._slab_mask[~np.isfinite(temp)] = 0

            # reorient gradient from slab to world space
            for gi,gg in enumerate(np.gradient(self._ref_interp[0], axis=self.in_slice_axes)):
                fw_step[...,gi] = gg
            fw_step /= slab_spacing
            vfu.reorient_vector_field_3d(fw_step, slab_direction_mtx)
            
            fw_step *= temp[...,np.newaxis]
            fw_step[~self._slab_mask] = 0

            nrgy = -corr[self._slab_mask].sum()
            self.energy_list.append(nrgy)

            #"""
            for d in self.in_slice_axes:
                gaussian_filter1d(fw_step, self.reg_sigma_diff, d, output=fw_step, mode='constant')

            fw_step[self._slice(pe_dir=[0,-1])] = 0
            fw_step[self._slice(fe_dir=[0,-1])] = 0

            #project on pedir
            fw_step[:] = (fw_step * static_pedir_ref).sum(-1)[...,np.newaxis] * static_pedir_ref
            nrm = np.abs(fw_step).max()
            if nrm > 0:
                fw_step /= nrm
            #"""

            self._df_epi_step_up[self._slice(slice_axis=sl)] = fw_step

            """
            for d in self.in_slice_axes:
                gaussian_filter1d(self._df_epi_step_up, self.reg_sigma_diff, d, output=self._df_epi_step_up, mode='constant')
            gaussian_filter1d(self._df_epi_step_up, 1, self.slice_axis, output=self._df_epi_step_up, mode='constant')

            for b in [0,-1]:
                self._df_epi_step_up[b] = 0
                self._df_epi_step_up[:,b] = 0
                self._df_epi_step_up[:,:,b] = 0
            #project on pedir
            self._df_epi_step_up[:] = (self._df_epi_step_up * static_pedir_ref).sum(-1)[...,np.newaxis] * static_pedir_ref
            nrm = np.abs(self._df_epi_step_up).max()
            if nrm > 0:
                self._df_epi_step_up /= nrm
            """
            
            vfu.compose_vector_fields_3d(self._df_data, self._df_epi_step_up,
                                         shift_grid2static_grid, static_world2grid,
                                         self.step_length, self._df_data)

            #vfu.compose_vector_fields_3d(self._df_data, fw_step,
            #                              shift_grid2slab_grid, slab_world2grid,
            #                              self.step_length, self._df_data)

            self._inv_df_data[:] = vfu.invert_vector_field_fixed_point_3d(
                self._df_data,
                shift_world2grid, shift_spacing,
                self.inv_iter, self.inv_tol,
                self._inv_df_data)
            self._df_data[:] = vfu.invert_vector_field_fixed_point_3d(
                self._inv_df_data,
                shift_world2grid, shift_spacing,
                self.inv_iter, self.inv_tol,
                self._df_data)

            if len(self.energy_list) >= self.energy_window:
                derivative = self._get_energy_derivative()
            print nrgy, derivative, np.abs(self._df_data).max()

            
    def process(self, stack, df, inv_df=None, yield_raw=False):
        ndim_state = 6
        
        frame_iterator = stack.iter_frame(queue_dicoms=True)
        self._voxel_size = stack._voxel_size
        self._cc_factors = None
        
        nvol, self.affine, first_frame = frame_iterator.next()
        self._first_frame = first_frame.astype(DTYPE)
        self._epi2ref = npl.inv(self._ref.affine).dot(self._init_reg).dot(self.affine)
        
        
        self.transform = dipy.align.transforms.RigidTransform3D()
        self._precond = np.asarray([1e-2]*3+[1.]*3)

        self._df_data = df.get_data().copy()
        # optionally specify the invert field, otherwise invert from scratch
        if inv_df is None:
            self._inv_df_data = vfu.invert_vector_field_fixed_point_3d(
                self._df_data,
                npl.inv(df.affine), np.asarray(df.header.get_zooms()[:3], dtype=np.double),
                self.inv_iter, self.inv_tol)
        else: 
            self._inv_df_data = inv_df.get_data().copy()

        """
        self._ref_warped = vfu.warp_3d(
            self._ref_scale_space.get_image(self.def_field_min_level),
            self._df_data,
            self._ref_scale_space.get_affine_inv(1).dot(self._ref.affine),
            None,
            npl.inv(self._ref.affine),
            out_shape = np.asarray(self._ref.shape, dtype=np.int32))
        """
        # register first frame
        self.sl_data = self._first_frame.astype(DTYPE)
        self.sl_data_smooth = self.sl_data.copy() 
        for d in self.in_slice_axes:
            gaussian_filter1d(self.sl_data_smooth, 1, d, output=self.sl_data_smooth)
        self.sl_data_smooth -= self.sl_data_smooth.min()
        self.sl_data_smooth /= self.sl_data_smooth.max()

        self.histogram = dipy.align.parzenhist.ParzenJointHistogram(32)
        self.histogram.setup(self.sl_data_smooth, self._ref_scale_space.get_image(self.def_field_min_level))

        self._slab_affine = np.eye(4)
        first_frame_reg = self._register_slab(
            np.arange(stack.nslices),
            self._first_frame, df, np.zeros(ndim_state),
            gtol=self._gradient_descent_tol)

        self._df_epi_step_up = np.empty(self.sl_data.shape+(3,), dtype=floating)
        ## Bias field update
        
        self._update_def_field(df, range(stack.nslices), first_frame_reg,
                               self.def_field_convergence, self.def_update_n_iter)

        stack_it = stack.iter_slabs()
        stack_has_data = True
        fr,sl,aff,tt,sl_data = stack_it.next()
        self.sl_data = sl_data = sl_data.astype(DTYPE)
        self.sl_data_smooth = np.empty_like(self.sl_data)

        ## init ORKF
        transition_covariance = np.eye(ndim_state, dtype=DTYPE) * self.orkf_transition_cov
        initial_state_covariance = np.eye(ndim_state, dtype=DTYPE) * self.orkf_init_covariance
        observation_covariance = np.eye(ndim_state, dtype=DTYPE) * self.orkf_observation_covariance

        s = ndim_state

        self.filtered_state_means = [first_frame_reg]
        self.filtered_state_covariances = [initial_state_covariance]
        self.niters = []
        self.matrices = []
        self.all_biases = []
        
        ## loop through slabs
        while stack_has_data:
            print 'frame %d slab %s'%(fr,str(sl)) + '_'*80

            self.sl_data_smooth[:] = self.sl_data
            for d in self.in_slice_axes:
                gaussian_filter1d(self.sl_data_smooth, 1, d, output=self.sl_data_smooth)
            self.sl_data_smooth -= self.sl_data_smooth.min()
            mm = self.sl_data_smooth.max()
            if mm > 0: self.sl_data_smooth /= mm

            self._slab_affine = slab_affine(sl, self.slice_axis)
            
            pred_params = self.filtered_state_means[-1]

            ## register slab
            params = self._register_slab(sl, self.sl_data, df, pred_params, gtol=self._gradient_descent_tol)
            print ('reg nrgy+gradient:'+' %f'*7)%((self._nrgy,)+tuple(self._rigid_gradient))

            ## ORKF
            update_params = pred_params.copy()
            last_update = update_params.copy()
            pred_cov = self.filtered_state_covariances[-1] + transition_covariance
            update_cov = pred_cov.copy()
            
            print ('pred:'+' %f'*6)%tuple(pred_params)
            print ('reg:'+' %f'*6)%tuple(params)
            
            conv = np.inf
            while conv > self.orkf_convergence:
                delta = np.matrix(params - update_params)
                obs_cov = (s*observation_covariance + delta.T.dot(delta) + update_cov)/(s+1)
                kalman_gain = npl.inv(pred_cov + obs_cov).dot(pred_cov)
                update_params[:] = pred_params + kalman_gain.dot(params - pred_params)
                IK = np.eye(len(params)) - kalman_gain
                update_cov[:] = kalman_gain.T.dot(obs_cov).dot(kalman_gain) + IK.T.dot(pred_cov).dot(IK)
                
                conv = np.max(np.abs(update_params-last_update))
                last_update[:] = update_params
                print ('orkf:'+' %f'*6)%tuple(update_params)

            #self._compute_mi_cost_gradient(update_params)
            print ('orkf nrgy+gradient:'+' %f'*7)%((self._nrgy,)+tuple(self._rigid_gradient))
            
            #self.filtered_state_means.append(params)
            self.filtered_state_means.append(update_params)
            self.filtered_state_covariances.append(update_cov)            

            ## Bias field update
            self._update_def_field(df, sl, update_params,
                                  self.def_field_convergence, self.def_update_n_iter)
            
            reg = self.transform.param_to_matrix(update_params*self._precond)
            reg_grid2ref = reg.dot(self.affine)

            if yield_raw:
                yield fr, sl, reg_grid2ref, sl_data
            else:
                yield fr, sl, reg_grid2ref
            try:
                fr,sl,aff,tt,sl_data[:] = stack_it.next()
            except StopIteration:
                stack_has_data = False
        del self._df_epi_step_up

    def _register_slab(self, sl, sl_data, df, init_params, gtol=1e-3):
        # register the T1 to the slab
        
        if self._cc_factors is None or sl_data.shape[self.slice_axis]!= self._cc_factors.shape[self.slice_axis]:
            self._ref_interp = np.zeros((7,)+sl_data.shape, dtype=DTYPE)
            self._cc_factors = np.zeros(sl_data.shape+(5,), dtype=DTYPE)
            self._slab_mask = np.zeros(sl_data.shape, dtype=np.bool)

        self._last_param = np.zeros_like(init_params)+np.inf
        self._rigid_gradient = np.zeros_like(init_params)
        
        def mi_cost_and_gradient(p):
            self._compute_mi_cost_gradient(p, df)
            print ('%.8f  ' * 7)%((self._nrgy,)+tuple(p))
            return self._nrgy, self._rigid_gradient
        
        def cc_cost_and_gradient(p):
            if np.any(self._last_param != p):
                self._last_param[:] = p
                self._compute_cc_cost_rigid_gradient(p, df)
            print ('%.8f  ' * 7)%((self._nrgy,)+tuple(p))
            return self._nrgy, self._rigid_gradient

        def cc_cost(p):
            if np.any(self._last_param != p):
                self._last_param[:] = p
                self._compute_cc_cost_rigid_gradient(p, df)
            print ('%.8f  ' * 7)%((self._nrgy,)+tuple(p))
            return self._nrgy

        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(init_params)
        
        opt = dipy.core.optimize.Optimizer(
            #mi_cost_and_gradient,
            cc_cost_and_gradient,
            #cc_cost,
            init_params,
            method='CG',
            jac=True,
            options=dict(disp=True, gtol=gtol, maxiter=8)
        )
        return opt.xopt

    def _compute_mi_cost_gradient(self, param, df):
        self._sample_ref_new(param, df, rigid_gradient=True)

        if np.count_nonzero(self._slab_mask) < self._nmin_samples_per_slab:
            self._nrgy=2
            self._rigid_gradient[:] = 0
            return

        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(param*self._precond)
        slab_grid2world = current_affine.dot(self._init_reg).dot(self.affine).dot(self._slab_affine)

        H = self.histogram
        slab_mask = self._slab_mask.astype(np.int32)
        #H.update_pdfs_sparse(self.sl_data[self._slab_mask].astype(np.double), self._ref_interp[0,self._slab_mask].astype(np.double))
        H.update_pdfs_dense(self.sl_data_smooth.astype(np.double), self._ref_interp[0].astype(np.double),
                            slab_mask, slab_mask)
           
        """
        H.update_gradient_sparse(
            param,
            transform,
            self.sl_data[self._slab_mask],
            self._ref_interp[0,self._slab_mask],
            slab_grid2world.dot(np.mgrid[[slice(0,d) for d in for self.sl_data.shape]]),
            self._ref_interp[1:4,self._slab_mask])
        """
        """
        mgrad, inside = vfu.gradient(
            self._ref_scale_space.get_image(self.def_field_min_level),
            npl.inv(self._ref.affine),
            np.asarray(self._ref.header.get_zooms()[:3],dtype=np.double),
            np.asarray(self.sl_data.shape,dtype=np.int),
            slab_grid2world
        )
        """
        
        H.update_gradient_dense(
            param*self._precond, dipy.align.transforms.RigidTransform3D(),
            self.sl_data_smooth.astype(np.double), self._ref_interp[0].astype(np.double),
            slab_grid2world,
            np.rollaxis(self._ref_interp[4:],0,4).astype(np.double),
            smask=slab_mask,
            mmask=slab_mask
        )

        self._nrgy = dipy.align.parzenhist.compute_parzen_mi(
            H.joint, H.joint_grad, H.smarginal, H.mmarginal,
            self._rigid_gradient)

        # neg MI
        self._nrgy *= -1
        self._rigid_gradient *= -self._precond
            
    def _compute_cc_cost_rigid_gradient(self, param, df):
        
        corr, temp = self._compute_cc_cost_gradient(param, df, rigid_gradient=True)
        
        temp_mask = np.logical_and(np.isfinite(temp), self._slab_mask)
        if np.count_nonzero(temp_mask) < self._nmin_samples_per_slab:
            print 'not enough sample'
            self._nrgy=2
            self._rigid_gradient[:] = 0
            return

        self._rigid_gradient[:] = (temp[temp_mask] * self._ref_interp[1:,temp_mask]).mean(1)
        self._rigid_gradient /= -self.orkf_jacobian_epsilon
        
        self._nrgy = -corr[temp_mask].mean()

    def _compute_cc_cost_gradient(self, param, df, rigid_gradient=False):
        self._sample_ref_new(param, df, rigid_gradient=rigid_gradient)
        #self._compute_cc_factors(self.sl_data, self._ref_interp[0])
        self._compute_cc_factors(self.sl_data_smooth, self._ref_interp[0])

        Ii = self._cc_factors[..., 0]
        Ji = self._cc_factors[..., 1]
        sfm = self._cc_factors[..., 2]
        sff = self._cc_factors[..., 3]
        smm = self._cc_factors[..., 4]

        # factor some of the computation
        temp = sfm / smm
        corr = temp / sff # sfm / (sff * smm)
        temp[:] = 2.0 * corr * (Ii - temp * Ji) # 2.0 * sfm/(sff*smm) * (Ii - sfm / smm * Ji)
        #temp = sfm / sff
        #corr = temp / smm # sfm / (sff * smm)
        #temp[:] = 2.0 * corr * (Ji - temp * Ii)
        corr *= sfm
        corr[np.isnan(corr)] = 0
        temp[np.isnan(temp)] = 0
        return corr, temp

    def _compute_cc_factors(self, static, moving):
        for si in range(static.shape[self.slice_axis]):
            self._cc_factors[self._slice(slice_axis=si)] = dipy.align.crosscorr.precompute_cc_factors_2d(
                static[self._slice(slice_axis=si)],
                moving[self._slice(slice_axis=si)],
                self._cc_radius)

    def _sample_ref_new(self, param, df, rigid_gradient=False):

        moving_world2grid = npl.inv(self._ref.affine)
        mask_world2grid = npl.inv(self.mask.affine)
        shift_world2grid = npl.inv(df.affine)
        
        param = param * self._precond
        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(param)
        slab_grid2world = current_affine.dot(self._init_reg).dot(self.affine).dot(self._slab_affine)

        slab_shape = np.asarray(self.sl_data.shape, dtype=np.int32)
        self._slab_mask[:] = vfu.warp_3d_nn(
            self.mask_data_int, self._df_data,
            mask_world2grid.dot(slab_grid2world),
            mask_world2grid.dot(slab_grid2world),
            mask_world2grid,
            out_shape = slab_shape)
        self._ref_interp[0] = vfu.warp_3d(
            self._ref_scale_space.get_image(0), self._df_data,
            shift_world2grid.dot(slab_grid2world),
            moving_world2grid.dot(slab_grid2world),
            moving_world2grid,
            out_shape = slab_shape)

        """
        self.grad = vfu.gradient(
            ref_warped,
            moving_world2grid,
            np.asarray(self._ref_scale_space.get_spacing(0)),
            slab_shape,
            slab_grid2world)

        self._ref_interp[0] = vfu.transform_3d_affine(self._ref_warped, slab_shape,
                                                      moving_world2grid.dot(slab_grid2world))
        if rigid_gradient:            
            for pi in range(6):
                param_delta = param.copy()
                param_delta[pi] += self.orkf_jacobian_epsilon * self._precond[pi]
                slab_grid2world_delta = np.dot(
                    transform.param_to_matrix(param_delta).dot(self._init_reg),
                    self.affine.dot(self._slab_affine))
                self._ref_interp[pi+1] = vfu.transform_3d_affine(self._ref_warped, slab_shape,
                                                                 moving_world2grid.dot(slab_grid2world_delta))
            self._ref_interp[1:] -= self._ref_interp[0]
            #self._ref_interp[1:] /= self.orkf_jacobian_epsilon
            for b in [0,-1]:
                self._ref_interp[1:,b] = 0
                self._ref_interp[1:,:,b] = 0
            self._ref_interp[:,np.any(np.isnan(self._ref_interp),0)] = 0
        """
        if rigid_gradient:
            for pi in range(6):
                param_delta = param.copy()
                param_delta[pi] += self.orkf_jacobian_epsilon * self._precond[pi]
                slab_grid2world_delta = np.dot(
                    transform.param_to_matrix(param_delta).dot(self._init_reg),
                    self.affine.dot(self._slab_affine))
                self._ref_interp[pi+1] = vfu.warp_3d(
                    self._ref_scale_space.get_image(0), self._df_data,
                    shift_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid,
                    out_shape = slab_shape)
            
            self._ref_interp[1:] -= self._ref_interp[0]
            #self._ref_interp[1:] /= self.orkf_jacobian_epsilon
            for b in [0,-1]:
                self._ref_interp[1:,b] = 0
                self._ref_interp[1:,:,b] = 0
            self._ref_interp[:,np.any(np.isnan(self._ref_interp),0)] = 0

        """
        if rigid_gradient:
            grad = np.empty(tuple(self._ref_scale_space.get_domain_shape(0))+(3,), dtype=np.float32)
            for gi,gg in enumerate(np.gradient(self._ref_scale_space.get_image(self.def_field_min_level))):
                grad[...,gi] = gg
            vfu.reorient_vector_field_3d(grad, self._ref.affine)
            for pi in range(3):
                self._ref_interp[pi+1] = vfu.warp_3d(
                    grad[...,pi], self._df_data,
                    shift_world2grid.dot(slab_grid2world),
                    moving_world2grid.dot(slab_grid2world),
                    moving_world2grid,
                    out_shape = slab_shape)

                
        if rigid_gradient:
            for pi in range(6):
                param_delta = param.copy()
                param_delta[pi+3] += .5
                slab_grid2world_delta = np.dot(
                    transform.param_to_matrix(param_delta).dot(self._init_reg),
                    self.affine.dot(self._slab_affine))
                self._ref_interp[pi+1] = vfu.warp_3d(
                    self._ref_scale_space.get_image(self.def_field_min_level), self._df_data,
                    shift_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid,
                    out_shape = slab_shape)

                param_delta[pi+3] -= 1
                slab_grid2world_delta = np.dot(
                    transform.param_to_matrix(param_delta).dot(self._init_reg),
                    self.affine.dot(self._slab_affine))
                self._ref_interp[pi+4] = vfu.warp_3d(
                    self._ref_scale_space.get_image(self.def_field_min_level), self._df_data,
                    shift_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid.dot(slab_grid2world_delta),
                    moving_world2grid,
                    out_shape = slab_shape)
    
            self._ref_interp[1:4] -= self._ref_interp[4:]
        """
        
class OnlineRealignBiasCorrection(EPIOnlineResample):
    
    def __init__(self,
                 anat_reg,
                 wm_weight = None,
                 bias_correction = True,
                 bias_sigma = 8,
                 register_gradient = False,
                 dog_sigmas = [1,2],

                 iekf_min_nsamples_per_slab = 200,
                 iekf_jacobian_epsilon = 1e-3,
                 iekf_convergence = 1e-3,
                 iekf_max_iter = 8,
                 iekf_observation_var = 1,
                 iekf_transition_cov = 1e-3,
                 iekf_init_state_cov = 1e-3,
                 **kwargs):

        print kwargs

        self.iekf_min_nsamples_per_slab = iekf_min_nsamples_per_slab
        self.iekf_jacobian_epsilon = iekf_jacobian_epsilon
        self.iekf_convergence = iekf_convergence
        self.iekf_max_iter = iekf_max_iter
        self.iekf_observation_var = iekf_observation_var
        self.iekf_transition_cov = iekf_transition_cov
        self.iekf_init_state_cov = iekf_init_state_cov

        super(OnlineRealignBiasCorrection,self).__init__(**kwargs)
        self._anat_reg = anat_reg
        print(('init_reg params:' + '\t%.3f'*12)% tuple(Affine(self._anat_reg).param))

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


    def process(self, stack, ref_frame=None, yield_raw=False):

        # to check if allocation is done in _sample_cost_jacobian
        self._slab_vox_idx = None
        frame_iterator = stack.iter_frame(queue_dicoms=True)
        self._voxel_size = stack._voxel_size
        #frame_iterator = stack.iter_frame()
        nvol, self.affine, self._first_frame = frame_iterator.next()
        self._epi2anat = npl.inv(self.mask.affine).dot(self._anat_reg).dot(self.affine)

        self._first_frame = self._first_frame.astype(DTYPE)
        
        if self._register_gradient:
            #self.register_refvol = ref_vol - convolve1d(convolve1d(ref_vol, [1/3.]*3,0),[1/3.]*3,1)
            self.register_refvol = reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[0],d),
                                          self.in_slice_axes, self._first_frame)-\
                                   reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[1],d), 
                                          self.in_slice_axes, self._first_frame)
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
            #self.register_refvol = reduce(lambda i,d: gaussian_filter1d(i,.5,d), self.in_slice_axes, self._first_frame)

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
                slice_data_reg = reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[0],d), self.in_slice_axes, sl_data)-\
                                 reduce(lambda i,d: gaussian_filter1d(i,self._dog_sigmas[1],d), self.in_slice_axes, sl_data)
            else:
                slice_data_reg = sl_data
                #slice_data_reg = reduce(lambda i,d: gaussian_filter1d(i,.5,d), self.in_slice_axes, sl_data)
            while convergence > self.iekf_convergence and niter < self.iekf_max_iter:
                new_reg.param = estim_state[:6]
                self._sample_cost_jacobian(sl, slice_data_reg, new_reg, bias_corr=self._bias_correction)
                if self._nvox_in_slab_mask < self.iekf_min_nsamples_per_slab:
                    print 'not enough point'
                    break

                mask = self._slab_mask
                cost, jac = self._cost[0,mask], self._cost[1:,mask]

                S = jac.T.dot(pred_covariance.dot(jac))
                S[np.diag_indices_from(S)] += self.iekf_observation_var#self.observation_variance[mask.ravel()]

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
            self._bias.fill(1.0)
            try:
                fr,sl,aff,tt,sl_data[:] = stack_it.next()
            except StopIteration:
                stack_has_data = False

        
    def _sample_cost_jacobian(self, sl, sl_data, new_reg, bias_corr=False):

        if self._slab_vox_idx is None or sl_data.shape[self.slice_axis]!= self._slab_vox_idx.shape[-2]:
            self._slab_vox_idx = np.empty(sl_data.shape+(sl_data.ndim,), dtype=np.int32)
            ## set vox idx for in-plane, does not change with slab
            for d in self.in_slice_axes:
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
        self._slab_shift[:] = self.sample_shift(self._slab_vox_idx, slab2anat)
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

            data_mask = np.any(np.isnan(self._interp_data),0)
            if np.count_nonzero(data_mask) > 0:
                self._slab_mask *= ~data_mask
                self._interp_data[:,data_mask] = 0
                self._slab_wm_weight *= self._slab_mask
            if bias_corr:
                #self._slab_wm_weight *= self._slab_sigloss
                #"""
                self._fit_bias_gaussian(sl_data)
                #self._fit_bias_spline(sl, sl_data)
                #self._fit_bias_poly(sl, sl_data)
            self._cost[0,self._slab_mask] = (sl_data[self._slab_mask]/self._bias[self._slab_mask] - 
                                             self._interp_data[0,self._slab_mask])
            self._cost[1:,self._slab_mask] = (self._interp_data[1:,self._slab_mask] - self._interp_data[0,self._slab_mask])/\
                                             self.iekf_jacobian_epsilon
            
        if np.any(~np.isfinite(self._cost)):
            raise RuntimeError
        self._nvox_in_slab_mask = self._slab_mask.sum()
    
    def _fit_bias_spline(self, slab, sl_data):
        ratio = sl_data/self._interp_data[0]
        self._slab_wm_weight[~np.isfinite(ratio)]=0
        for sli, sln in enumerate(slab):
            wm_voxs = np.where(self._slab_wm_weight[...,sli]>0)
            if len(wm_voxs[0])<32:
                self._bias[...,sli] = 1
                continue
            brain_voxs = np.where(self._slab_mask[...,sli]>0)
            from scipy.interpolate import bisplrep, bisplev, SmoothBivariateSpline
            sbvs = SmoothBivariateSpline(
                wm_voxs[0],wm_voxs[1], ratio[...,sli][wm_voxs],
                self._slab_wm_weight[...,sli][wm_voxs],
                [0,sl_data.shape[0],0,sl_data.shape[1]],
                kx=2,ky=2,
                s=32)
            self._bias[...,sli] = sbvs(np.arange(sl_data.shape[0]),np.arange(sl_data.shape[1]))
            

    def _fit_bias_poly(self, slab, sl_data, bias_order=3):
        ### not working
        if not hasattr(self, '_polyvander'):
            self._polyvander = np.polynomial.polynomial.polyvander2d(
                *np.mgrid[[slice(-(sl_data.shape[a]-1)/2,sl_data.shape[a]/2) for a in self.in_slice_axes ]],
                deg=[bias_order]*2)
            self._polyvander /= np.sqrt(np.square(self._polyvander.reshape(-1,self._polyvander.shape[-1])).sum(0))
        for sli,sln in enumerate(slab):
            if self._slab_wm_weight[...,sli].sum() < 50:
                self._bias[...,sli] = 1
                continue
            reg = self._polyvander * np.sqrt(self._slab_wm_weight[...,sli,np.newaxis])
            y = sl_data[...,sli]/self._interp_data[0,...,sli] * np.sqrt(self._slab_wm_weight[...,sli])
            c, resids, rank, s = npl.lstsq(reg[self._slab_mask[...,sli]], y[self._slab_mask[...,sli]])
            self._bias[...,sli] = self._polyvander.dot(c)


    def _fit_bias_gaussian(self, sl_data):
        weight_per_slice = np.atleast_1d(np.squeeze(np.apply_over_axes(np.sum, self._slab_wm_weight, self.in_slice_axes)))
        sl_data_smooth = sl_data * self._slab_wm_weight
        interp_data_smooth = self._interp_data[0] * self._slab_wm_weight
        # use separability of gaussian filter
        for d in self.in_slice_axes:
            bias_sigma_vox = self._bias_sigma/self._voxel_size[d]
            truncate = sl_data.shape[d]/bias_sigma_vox
            sl_data_smooth[:] = gaussian_filter1d(sl_data_smooth, bias_sigma_vox, d,
                                                  mode='constant', truncate=truncate)
            interp_data_smooth[:] = gaussian_filter1d(interp_data_smooth, bias_sigma_vox, d, 
                                                      mode='constant', truncate=truncate)
        self._bias[:] = sl_data_smooth / interp_data_smooth
        self._bias[~np.isfinite(self._bias)] = 1
        self._bias[...,weight_per_slice<self.iekf_min_nsamples_per_slab] = 1
        """
        self._slab_wm_weight += 1e-8*self._slab_mask
        self._bias[:] = np.exp(
        reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d, truncate=10), [0,1], 
        np.log(sl_data+1e-8)*self._slab_wm_weight)/\
        reduce(lambda i,d: gaussian_filter1d(i,self._bias_sigma,d,truncate=10), [0,1], 
        self._slab_wm_weight))
        """
        

    def correct(self, realigned, pvmaps, frame_shape, sig_smth=16, white_idx=1,
                maxiter = 16, residual_tol = 2e-3, n_samples_min = 30):
        
        float_mask = nb.Nifti1Image(
            self.mask_data.astype(np.float32),
            self.mask.affine)

        ext_mask = self.mask_data
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
                epi_coords = apply_affine(
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
        self._shape = nii.shape
        if len(self._shape)<4:
            self._shape = self._shape + (1,)
        self.nslices,self.nframes = self._shape[2:]
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
                
        self._slab_trigger_times = np.arange(self._nshots)*(self.nii.header.get_zooms()+(1,))[3]/float(self._nshots)
        
    def iter_frame(self, data=True, queue_dicoms=False):
        data = self.nii.get_data()
        if data.ndim < 4:
            data = data[...,None]
        for t in range(data.shape[3]):
            yield t, self.nii.affine, data[:,:,:,t]
        del data

    def iter_slabs(self, data=True, queue_dicoms=False):
        data = self.nii.get_data()
        if data.ndim < 4:
            data = data[...,None]
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


def jac(shift, disp_spacing, jac_out):
    for di in range(3):
        for gi,gg in enumerate(np.gradient(shift[...,di], *disp_spacing)):
            jac_out[di,gi] = gg
        jac_out[di,di] += 1

def jac_det(jac, jac_det):
    jac_det[:] = (jac[0,0])*((jac[1,1])*(jac[2,2])-(jac[1,2]*jac[2,1])) -\
        jac[0,1]*((jac[1,0]*jac[2,2])-(jac[1,2]*jac[2,0])) +\
        jac[0,2]*((jac[1,0]*jac[2,1])-(jac[1,1]*jac[2,1]))


def slab_affine(sl, slice_axis):
    # compute the premultiplying scaling and shift related to the new grid with slices subset
    a = np.eye(4)
    if len(sl)>1:
        a[slice_axis,slice_axis] = np.diff(sl)[0]
    a[slice_axis,-1] = sl[0]
    return a
