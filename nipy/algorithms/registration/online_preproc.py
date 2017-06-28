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
DTYPE = np.float64
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
        vox = apply_affine(np.linalg.inv(self.fieldmap_reg.dot(self.fmap.affine)).dot(aff), vox_in)
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
        ## could/shoud we avoid loop here and do all slices in the meantime ?
        slab_mask = np.zeros_like(data[0], dtype=np.bool)
        if not pve_map is None:
            pve_data = pve_map.get_data()
            slab_pve = np.zeros_like(slab_mask, dtype=pve_data.dtype)
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
        phase_vec /= np.linalg.norm(phase_vec)
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
        self.world2fmap = np.linalg.inv(self.fmap2world)
        grid = apply_affine(
            np.linalg.inv(self.mask.affine).dot(self.fmap2world),
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
            vol2epi = np.linalg.inv(affine).dot(vol.affine)
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
            epi2vol = np.linalg.inv(vol.affine).dot(affine)
            voxs = apply_affine(epi2vol, grid)

            for v in range(nvols):
                rvol[...,v] = map_coordinates(
                    vol_data[...,v],
                    voxs.reshape(-1,3).T, order=order).reshape(shape)
        del grid, voxs
        return np.squeeze(rvol)


import dipy.align
import dipy.align.parzenhist
import dipy.core.optimize

class EPIOnlineRealignUndistort(EPIOnlineResample):
    
    def __init__(self,
                 ref,
                 init_reg,
                 cc_radius = 1,

                 gradient_descent_tol = 1e-3,
                 
                 orkf_init_covariance = 1e-1,
                 orkf_transition_cov = 1e-1,
                 orkf_observation_covariance = 1e-1,
                 orkf_convergence = 1e-2,
                 orkf_jacobian_epsilon = 1e-8,
                 
                 def_field_convergence = 1e-2,
                 **kwargs):
        super(EPIOnlineRealignUndistort, self).__init__(**kwargs)

        self._ref = ref
        self._ref_data = self._ref.get_data().astype(DTYPE)
        #self._ref_data_smooth = gaussian_filter(self._ref_data, 1)
        #self._ref_data -= self._ref_data.min()
        #self._ref_data /= self._ref_data.max()
        self.mask_data_dil = binary_dilation(self.mask_data, None, 2)
        #self._ref_data *= self.mask_data_dil
        #self._ref_reg = ref_reg
        self._init_reg = init_reg #Rigid(radius=RADIUS)
        #self._init_reg.from_matrix44(self._ref_reg)

        self._cc_radius = cc_radius

        self._gradient_descent_tol = gradient_descent_tol 

        self.orkf_convergence = orkf_convergence
        self.orkf_transition_cov = orkf_transition_cov
        self.orkf_init_covariance = orkf_init_covariance
        self.orkf_observation_covariance = orkf_observation_covariance
        self.orkf_jacobian_epsilon = orkf_jacobian_epsilon
        
        self.def_field_convergence = def_field_convergence

        
    def _init_diffeo_map(self, affine, shape):

        vol2fmap = self.world2fmap.dot(self._init_reg.dot(affine))
        grid = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]], 0, 4)
        fmap_voxs = apply_affine(vol2fmap, grid)
        self._diffeo_map = shape[self.pe_dir] * self.fmap_scale * map_coordinates(
            self.fmap.get_data(),
            fmap_voxs.reshape(-1,3).T,
            order=1).reshape(shape)
        del fmap_voxs, grid
 
    def _diffeo_blip_map(self, blip_up_stack, blip_down_stack, bup_reg, bdown_reg, reg=.8):
        
        _, blip_up_affine, blip_up_data = blip_up_stack.iter_frame(queue_dicoms=True).next()
        _, blip_down_affine, blip_down_data = blip_down_stack.iter_frame(queue_dicoms=True).next()
        blip_up_data = blip_up_data.astype(DTYPE)
        blip_down_data = blip_down_data.astype(DTYPE)

        """
        aff_reg = dipy.align.imaffine.AffineRegistration()
        tx = dipy.align.transforms.RigidTransform3D()
        bup_reg = aff_reg.optimize(self._ref_data, blip_up_data, tx,
                                   static_grid2world = self._ref.affine,
                                   moving_grid2world = blip_up_affine,
                                   starting_affine = 'mass')
        bdown_reg = aff_reg.optimize(self._ref_data, blip_down_data, tx,
                                     static_grid2world = self._ref.affine,
                                     moving_grid2world = blip_down_affine,
                                     starting_affine = 'mass')
        """

        # maybe compute gradient from interpolated data, in higher resolution?
        blip_up_grad = np.gradient(blip_up_data, axis=self.pe_dir)
        blip_down_grad = np.gradient(blip_down_data, axis=self.pe_dir)

        grid = np.mgrid[[slice(0,d) for d in self._ref.shape]]
        ## TODO init reg different for bup/bdown
        ref2blip_up = np.linalg.inv(blip_up_affine).dot(bup_reg).dot(self._ref.affine)
        ref2blip_down = np.linalg.inv(blip_down_affine).dot(bdown_reg).dot(self._ref.affine)
        blip_up2ref = np.linalg.inv(ref2blip_up)
        blip_down2ref = np.linalg.inv(ref2blip_down)
        blip_up_pedir_ref = blip_up2ref[:3,self.pe_dir]
        blip_up_pedir_ref /= np.sqrt(np.square(blip_up_pedir_ref).sum())
        blip_down_pedir_ref = blip_down2ref[:3,self.pe_dir]
        blip_down_pedir_ref /= np.sqrt(np.square(blip_down_pedir_ref).sum())
        coords_blip_up = apply_affine2(ref2blip_up, grid)
        coords_blip_down = apply_affine2(ref2blip_down, grid)
        
        conv = np.inf
        
        nrgy_list = []
        
        shift_up = np.zeros(self._ref.shape)
        shift_down = np.zeros_like(shift_up)
        jac_blip_up = np.ones_like(shift_up)
        jac_blip_down = np.ones_like(shift_up)

        blip_up_interp = np.empty_like(shift_up)
        blip_down_interp = np.empty_like(shift_up)
        blip_up_pe_grad = np.empty((2,)+shift_up.shape)
        blip_down_pe_grad = np.empty((2,)+shift_up.shape)

        # reference scale space setting
        self.ref_ss = ScaleSpace(self._ref_data, self.levels, self.ref.affine,
                                 self.reg.get_zooms(), self.ss_sigma_factor,
                                 False)

        ref_smooth = gaussian_filter(self._ref_data, 2)
        #ref_smooth = self._ref_data
        
        #msk = ref_smooth > 1e-3
        msk = self.mask_data_dil
        

        all_shift_up = []
        
        #while conv > 1e-5: #self.def_field_convergence:
        for i in range(16):
            # apply shift
            coords_blip_up[self.pe_dir] += shift_up
            coords_blip_down[self.pe_dir] += shift_down
            
            # interpolate data and gradient
            map_coordinates(blip_up_data, coords_blip_up, blip_up_interp, order=1)
            map_coordinates(blip_down_data, coords_blip_down, blip_down_interp, order=1)
            #gaussian_filter(blip_up_interp, 1.2, output=blip_up_interp)
            #gaussian_filter(blip_down_interp, 1.2, output=blip_down_interp)
            #blip_up_interp *= jac_blip_up
            #blip_down_interp *= jac_blip_down

            # maybe compute gradient from interpolated data, in higher resolution?
            #map_coordinates(blip_up_grad, coords_blip_up, blip_up_pe_grad[0], order=1)
            #map_coordinates(blip_down_grad, coords_blip_down, blip_down_pe_grad[0], order=1)
            
            #"""
            coords_blip_up[self.pe_dir] += .5
            coords_blip_down[self.pe_dir] += .5
            map_coordinates(blip_up_data, coords_blip_up, blip_up_pe_grad[0], order=1)
            map_coordinates(blip_down_data, coords_blip_down, blip_down_pe_grad[0], order=1)

            coords_blip_up[self.pe_dir] -= 1
            coords_blip_down[self.pe_dir] -= 1
            map_coordinates(blip_up_data, coords_blip_up, blip_up_pe_grad[1], order=1)
            map_coordinates(blip_down_data, coords_blip_down, blip_down_pe_grad[1], order=1)
            blip_up_pe_grad[0] -= blip_up_pe_grad[1]
            blip_down_pe_grad[0] -= blip_down_pe_grad[1]
            #"""

            # unshift remove for next iterations
            coords_blip_up[self.pe_dir] -= shift_up-.5
            coords_blip_down[self.pe_dir] -= shift_down-.5

            blip_up_down_geomean = 2*(blip_up_interp*blip_down_interp)/(blip_up_interp+blip_down_interp)
            blip_up_down_geomean[blip_up_interp * blip_down_interp < 1e-3] = 0

            ks_fact = np.asarray(dipy.align.crosscorr.precompute_cc_factors_3d(
                #self._ref_data,
                ref_smooth,
                blip_up_down_geomean,
                self._cc_radius))
            blip_up_down_fact = np.asarray(dipy.align.crosscorr.precompute_cc_factors_3d(
                blip_up_interp,
                blip_down_interp,
                self._cc_radius))

            msk2 = np.logical_and(msk, np.prod(ks_fact[...,3:],-1) > 1e-5)
                        
            dxi4_dk = 2.0 * ks_fact[...,2] / (ks_fact[...,3] * ks_fact[...,4]) * \
                      (ks_fact[...,1] - ks_fact[...,2]/ ks_fact[...,4] * ks_fact[...,0])
            dxi4_dk[~msk2] = 0
            #i_j = blip_up_down_fact[...,3:].sum(-1)
            i_j = blip_up_interp + blip_down_interp
            
            step_up = np.square(blip_down_interp/i_j) * jac_blip_up * blip_up_pe_grad[0] * dxi4_dk * msk2 # *2.0 not useful
            step_down = np.square(blip_up_interp/i_j) * jac_blip_down * blip_down_pe_grad[0] * dxi4_dk * msk2 # *2.0
            step_up[~np.isfinite(step_up)] = 0
            step_down[~np.isfinite(step_down)] = 0

            def_field_sigma_vox = 8
            gaussian_filter(step_up, def_field_sigma_vox, output=step_up, mode='constant')
            gaussian_filter(step_down, def_field_sigma_vox, output=step_down, mode='constant')
            for a in [step_up, step_down]:
                for b in [0,-1]:
                    a[b] = 0
                    a[:,b] = 0
                    a[:,:,b] = 0

            ## Soft equality constraint
            step_up_eqcon = step_up + reg/2.*(-step_down-step_up)
            step_down_eqcon = step_down + reg/2.*(-step_up-step_down)

            norm_step_up = np.abs(step_up_eqcon).max()
            norm_step_down = np.abs(step_down_eqcon).max()

            step_up_eqcon /= norm_step_up
            step_down_eqcon /= norm_step_down
            
            shift_up += step_up_eqcon * .5
            shift_down += step_down_eqcon * .5
            
            #project gradient of shift on pedir unit vector
            #jac_blip_up[:] = 1 + reduce(lambda b,a: a[0]*a[1]+b, zip(np.gradient(shift_up), blip_up_pedir_ref), 0)
            #jac_blip_down[:] = 1 + reduce(lambda b,a: a[0]*a[1]+b, zip(np.gradient(shift_down), blip_down_pedir_ref), 0)

            corr = np.square(ks_fact[...,2])/(ks_fact[...,3]*ks_fact[...,4])
            corr[np.logical_or(corr>1,corr<-1)] = 0
            #nrgy = -corr[np.isfinite(corr)].mean()
            nrgy = -corr[msk2].mean()
            
            all_shift_up.append(shift_up.copy())

            del blip_up_down_fact, ks_fact
            
            nrgy_list.append(nrgy)
            if len(nrgy_list)>1:
                conv = nrgy_list[-2] - nrgy_list[-1]
            print nrgy, conv
            #if conv<0:
            #    raise RuntimeError
        return shift_up, shift_down, blip_up_down_geomean, blip_up_interp, blip_down_interp, corr, all_shift_up



            
    def process(self, stack, yield_raw=False):
        ndim_state = 6

        frame_iterator = stack.iter_frame(queue_dicoms=True)
        self._voxel_size = stack._voxel_size
        self._slab_vox_idx = None
        
        nvol, self.affine, first_frame = frame_iterator.next()
        self._first_frame = first_frame.astype(DTYPE)
        #self._epi2ref = np.linalg.inv(self._ref.affine).dot(self._ref_reg).dot(self.affine)

        self.histogram = dipy.align.parzenhist.ParzenJointHistogram(32)
        self.histogram.setup(self._first_frame, self._ref_data[self.mask_data_dil])
        
        self.transform = dipy.align.transforms.RigidTransform3D()

        self._init_diffeo_map(self.affine, self._first_frame.shape)

        self._precond = np.asarray([2e-2]*3+[1.]*3)

        # register first frame
        self.sl_data = self._first_frame.astype(DTYPE)
        #first_frame_reg = np.zeros(ndim_state)
        first_frame_reg = self._register_slab(np.arange(stack.nslices), self._first_frame, np.zeros(6),gtol=1e-4)
        first_frame_mtx = self.transform.param_to_matrix(first_frame_reg*self._precond)
        grid = np.mgrid[[slice(0,d) for d in self._ref.shape]]
        ref2epi =  np.linalg.inv(self._init_reg.dot(first_frame_mtx).dot(self.affine)).dot(self._ref.affine)
        coords_ref2epi = apply_affine2(ref2epi, grid)
        self._ref_data[:] = map_coordinates(self._first_frame, coords_ref2epi, order=3)
        
        stack_it = stack.iter_slabs()
        stack_has_data = True
        fr,sl,aff,tt,sl_data = stack_it.next()
        self.sl_data = sl_data = sl_data.astype(DTYPE)

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

            #for d in self.in_slice_axes:
            #    gaussian_filter1d(self.sl_data, .5, d, 0, self.sl_data)
            pred_params = self.filtered_state_means[-1]
            #pred_params = np.zeros(6)

            ## register slab
            params = self._register_slab(sl, self.sl_data, pred_params, gtol=self._gradient_descent_tol)
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
                kalman_gain = np.linalg.inv(pred_cov + obs_cov).dot(pred_cov)
                update_params[:] = pred_params + kalman_gain.dot(params - pred_params)
                IK = np.eye(len(params)) - kalman_gain
                update_cov[:] = kalman_gain.T.dot(obs_cov).dot(kalman_gain) + IK.T.dot(pred_cov).dot(IK)
                
                conv = np.max(np.abs(update_params-last_update))
                last_update[:] = update_params
                print ('orkf:'+' %f'*6)%tuple(update_params)

            self._compute_mi_cost_gradient(update_params)
            print ('orkf nrgy+gradient:'+' %f'*7)%((self._nrgy,)+tuple(self._rigid_gradient))
            
            #self.filtered_state_means.append(params)
            self.filtered_state_means.append(update_params)
            self.filtered_state_covariances.append(update_cov)            

            ## Bias field update
            #self._update_def_field(sl, self.sl_data, update_params)
            
            reg = self.transform.param_to_matrix(update_params)
            reg_grid2ref = reg.dot(self.affine)

            if yield_raw:
                yield fr, sl, reg_grid2ref, sl_data
            else:
                yield fr, sl, reg_grid2ref
            try:
                fr,sl,aff,tt,sl_data[:] = stack_it.next()
            except StopIteration:
                stack_has_data = False

    def _register_slab(self, sl, sl_data, init_params, gtol=1e-3):
        # register the T1 to the slab
        
        if self._slab_vox_idx is None or sl_data.shape[self.slice_axis]!= self._slab_vox_idx.shape[-2]:
            self._slab_vox_idx = np.empty(sl_data.shape +(sl_data.ndim,), dtype=np.int32)
            ## set vox idx for in-plane, does not change with slab
            for d in self.in_slice_axes:
                self._slab_vox_idx[...,d] = np.arange(sl_data.shape[d])[[
                    (slice(0,None) if d==d2 else None) for d2 in range(sl_data.ndim)]]
            self._anat_slab_coords = np.zeros((3,)+self._slab_vox_idx.shape, dtype=DTYPE)
            self._ref_interp = np.zeros(sl_data.shape, dtype=DTYPE)
            self._ref_gradient = np.zeros(sl_data.shape+(3,), dtype=DTYPE)
            self._cc_factors = np.zeros(sl_data.shape+(5,), dtype=DTYPE)
            self._slab_shift = np.zeros(sl_data.shape, dtype=DTYPE)
        # set slice dimension vox idx (slice number)
        self._slab_vox_idx[...,self.slice_axis] = np.asarray(sl)[[
            (slice(0,None) if self.slice_axis==d2 else None) for d2 in range(sl_data.ndim)]]
        
#        self._slab_shift[:] = self._diffeo_map[self._slice(slice_axis=sl)]

        self._last_param = np.zeros_like(init_params)+np.inf
        
        def mi_cost_and_gradient(p):
            #self._compute_nrgy_gradient_rigid(param, sl_data)
            self._compute_mi_cost_gradient(p)
            #print self._rigid_gradient
            return self._nrgy, self._rigid_gradient

        self._rigid_gradient = np.zeros_like(init_params)

        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(init_params)
        
        #self._prealign = current_affine.dot(self._init_reg)
         
        opt = dipy.core.optimize.Optimizer(
            mi_cost_and_gradient,
            init_params,
            method='CG',
            jac=True,
            options=dict(disp=True, gtol=gtol, maxiter=16)
        )
        #opt.xopt[:3] /= 100
        return opt.xopt
        #reg_param = scipy.optimize.fmin_ncg(cost, new_reg.param, gradient, args=(self.sl_data,))
        #reg_params = scipy.optimize.fmin_cg(cost, init_params, gradient, args=(self.sl_data,), gtol=1e-2, maxiter=32)

        #if warnflags:
        #    raise RuntimeError
        #return reg_params



    def _sample_ref_gradient(param):
        transform = dipy.align.transforms.RigidTransform3D()
        current_affine = transform.param_to_matrix(param)

        static_grid2world = self.affine
        moving_world2grid = np.linalg.inv(self._ref.affine)
        current_init_reg = current_affine.dot(self._init_reg)
        static_grid2moving_grid = moving_world2grid.dot(current_init_reg).dot(static_grid2world)

        vol2fmap = self.world2fmap.dot(current_init_reg).dot(static_grid2world)
        
        self._slab_shift[:] = self.sl_data.shape[self.pe_dir] * self.fmap_scale * map_coordinates(
            self.fmap.get_data(),
            apply_affine(vol2fmap, self._slab_vox_idx).reshape(-1,3).T,
            order=1).reshape(self.sl_data.shape)

        self._anat_slab_coords[0] = apply_affine(static_grid2moving_grid, self._slab_vox_idx)
        self._anat_slab_coords[0] -= static_grid2moving_grid[:3,self.pe_dir] * self._slab_shift[...,np.newaxis]

        self._anat_slab_coords[1] = apply_affine(static_grid2current, self._slab_vox_idx)
        self._anat_slab_coords[1] -= static_grid2current[:3,self.pe_dir] * self._slab_shift[...,np.newaxis]

        self._ref_interp[:] = dipy.align.vector_fields.interpolate_scalar_3d(
            self._ref_data,
            self._anat_slab_coords[0].reshape(-1,3))[0].reshape(self.sl_data.shape)


        self._ref_gradient[:], inside = dipy.align.vector_fields.sparse_gradient(
            self._ref_data,
            moving_world2grid,
            np.asarray(self._ref.header.get_zooms()),
            self._anat_slab_coords[1].reshape(-1,3))
        

    def _compute_mi_cost_gradient(self, param):
        if np.all(self._last_param == param):
            return
        self._last_param[:] = param

        param = param * self._precond
        self._sample_ref_gradient(param)

        H = self.histogram
        H.update_pdfs_sparse(self.sl_data.ravel(), self._ref_interp.ravel())

        H.update_gradient_sparse(
            param,
            transform,
            self.sl_data.ravel(),
            self._ref_interp.ravel(),
            self._anat_slab_coords[1].reshape(-1,3),
            self._ref_gradient)

        self._nrgy = dipy.align.parzenhist.compute_parzen_mi(
            H.joint, H.joint_grad, H.smarginal, H.mmarginal,
            self._rigid_gradient)

        # neg MI
        self._nrgy *= -1 
        self._rigid_gradient *= -self._precond

        print ('%.8f '*7)%((self._nrgy,)+tuple(self._last_param))

    def compute_cc_cost_gradient(self, param):
        if np.all(self._last_param == param):
            return
        self._last_param[:] = param

        param = param * self._precond

        self._sample_ref_gradient(param)

        for si in range(static.shape[self.slice_axis]):
            self._cc_factors[self._slice(slice_axis=si)] = dipy.align.crosscorr.precompute_cc_factors_2d(
                self.sl_data[self._slice(slice_axis=si)],
                self._ref_interp[self._slice(slice_axis=si)],
                self._cc_radius)

        Ii = self._cc_factors[..., 0]
        Ji = self._cc_factors[..., 1]
        sfm = self._cc_factors[..., 2]
        sff = self._cc_factors[..., 3]
        smm = self._cc_factors[..., 4]
        temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)

        corr = np.square(sfm)/(sff*smm)
        corr_mask = np.isfinite(corr)
        self._nrgy = -corr[corr_mask].mean()
        if np.isnan(self._nrgy):
            self._nrgy = 2 # set to higher value in case a better registration was found??
        
        temp_mask = np.isfinite(temp)
        self._rigid_gradient = -(temp[temp_mask]*self._ref_interp[1:,temp_mask]).mean(1)
        self._rigid_gradient[np.isnan(self._rigid_gradient)] = 0        
    



    def _compute_cc_factors(self, static, moving):
        for si in range(static.shape[self.slice_axis]):
            self._cc_factors[self._slice(slice_axis=si)] = dipy.align.crosscorr.precompute_cc_factors_2d(
                static[self._slice(slice_axis=si)],
                moving[self._slice(slice_axis=si)],
                self._cc_radius)

    def _sample_ref(self, param, rigid_gradient=False):
        reg = Rigid(radius=RADIUS)
        reg.param = param
        
        slab2anat = np.linalg.inv(self._ref.affine).dot(reg.as_affine()).dot(self.affine)
        #slab2anat = np.linalg.inv(self.affine).dot(reg.as_affine()).dot(self.affine)
        self._anat_slab_coords[0] = apply_affine2(slab2anat, self._slab_vox_idx)
        self._anat_slab_coords[0] -= slab2anat[:3,self.pe_dir, np.newaxis, np.newaxis, np.newaxis] * self._slab_shift
        
        if rigid_gradient:
            for pi in range(6):
                reg_delta = Rigid(radius=RADIUS)
                param_delta = param.copy()
                param_delta[pi] += self.orkf_jacobian_epsilon
                reg_delta.param = param_delta
                slab2anat_delta = np.linalg.inv(self._ref.affine).dot(reg_delta.as_affine()).dot(self.affine)
                #slab2anat_delta = np.linalg.inv(self.affine).dot(reg_delta.as_affine()).dot(self.affine)
                self._anat_slab_coords[pi+1] = apply_affine2(slab2anat_delta, self._slab_vox_idx)
                self._anat_slab_coords[pi+1] -= slab2anat_delta[:3,self.pe_dir,np.newaxis, np.newaxis, np.newaxis] * self._slab_shift

            map_coordinates(self._ref_data, np.rollaxis(self._anat_slab_coords,1,0), self._ref_interp, order=1)
            #map_coordinates(self._first_frame, np.rollaxis(self._anat_slab_coords,1,0), self._ref_interp, order=1, cval=np.nan)

            self._ref_interp[1:] -= self._ref_interp[0]
            self._ref_interp[1:] /= self.orkf_jacobian_epsilon
            self._ref_interp[1:,0] = 0
            self._ref_interp[1:,-1] = 0
            self._ref_interp[1:,:,0] = 0
            self._ref_interp[1:,:,-1] = 0
            self._ref_interp[:,np.any(np.isnan(self._ref_interp),0)] = 0

        else: #phase encoding direction gradient
            subvox_delta = .5
            self._anat_slab_coords[1] = self._anat_slab_coords[0] + slab2anat[:3,self.pe_dir, np.newaxis, np.newaxis, np.newaxis] * subvox_delta
            map_coordinates(self._ref_data, np.rollaxis(self._anat_slab_coords[:2],1,0), self._ref_interp[:2], order=1)
            self._ref_interp[1] -= self._ref_interp[0]


    def _compute_nrgy_gradient_rigid(self, param, sl_data):
        if np.all(self._last_param == param):
            #print 'do nothing'
            return
        self._last_param[:] = param
        self._sample_ref(param, rigid_gradient=True)

        self._compute_cc_factors(sl_data, self._ref_interp[0])
        Ii = self._cc_factors[..., 0]
        Ji = self._cc_factors[..., 1]
        sfm = self._cc_factors[..., 2]
        sff = self._cc_factors[..., 3]
        smm = self._cc_factors[..., 4]
        temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)

        corr = np.square(sfm)/(sff*smm)
        corr_mask = np.isfinite(corr)
        self._nrgy = 1-corr[corr_mask].mean()
        if np.isnan(self._nrgy):
            self._nrgy = 2 # set to higher value in case a better registration was found??
        
        temp_mask = np.isfinite(temp)
        self._rigid_gradient = -(temp[temp_mask]*self._ref_interp[1:,temp_mask]).mean(1)
        self._rigid_gradient[np.isnan(self._rigid_gradient)] = 0        
        #print ('%.8f '*7)%((self._nrgy,)+tuple(param))
        
    def _update_def_field(self, sl, sl_data, param):
        self._slab_shift[:] = self._diffeo_map[self._slice(slice_axis=sl)]
        def_field_sigma_vox = 2
        print 'update def field'
        conv = np.inf
        nrgy_list = []
        #tmp_slsh = []
        while conv > self.def_field_convergence:
            self._sample_ref(param, rigid_gradient=False)
            self._compute_cc_factors(sl_data, self._ref_interp[0])
            Ii = self._cc_factors[..., 0]
            Ji = self._cc_factors[..., 1]
            sfm = self._cc_factors[..., 2]
            sff = self._cc_factors[..., 3]
            smm = self._cc_factors[..., 4]
            temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
            temp[~np.isfinite(temp)] = 0
            fw_step = temp * self._ref_interp[1]
            corr = np.square(sfm)/(sff*smm)
            nrgy = 1-corr[np.isfinite(corr)].mean()
            nrgy_list.append(nrgy)
            for d in self.in_slice_axes:
                fw_step[:] = gaussian_filter1d(fw_step, def_field_sigma_vox, d, mode='constant')
            nrm = np.abs(fw_step).max()
            if nrm > 0:
                fw_step /= nrm

            self._slab_shift += fw_step*.25
            if len(nrgy_list)>1:
                conv = nrgy_list[-2] - nrgy_list[-1]
            print conv, nrgy_list
            #tmp_slsh.append(self._slab_shift.copy())
        if len(nrgy_list)>5:
            raise RuntimeError
        self._diffeo_map[self._slice(slice_axis=sl)] = self._slab_shift
            
        
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
        self._epi2anat = np.linalg.inv(self.mask.affine).dot(self._anat_reg).dot(self.affine)

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
            c, resids, rank, s = np.linalg.lstsq(reg[self._slab_mask[...,sli]], y[self._slab_mask[...,sli]])
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
