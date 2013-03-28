# Emacsrepl: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings
import numpy as np

from nibabel.affines import apply_affine
import nibabel as nb

from ...fixes.nibabel import io_orientation

from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .optimizer import configure_optimizer, use_derivatives
from .affine import Rigid
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_sample4d)
from scipy.ndimage import convolve1d, gaussian_filter, binary_erosion
import scipy.stats
from scipy.ndimage.interpolation import map_coordinates


# Module globals
VERBOSE = True  # enables online print statements
SLICE_ORDER = 'ascending'
INTERLEAVED = None
OPTIMIZER = 'powell'
XTOL = 1e-5
FTOL = 1e-5
GTOL = 1e-5
STEPSIZE = 1e-6
SMALL = 1e-20
MAXITER = 64
MAXFUN = None
BORDERS = 1, 1, 1
REFSCAN = 0
EXTRAPOLATE_SPACE = 'reflect'
EXTRAPOLATE_TIME = 'nearest'

# extract boundary points and tissue class points from a segmentation
# probability map, could be improved by using the other class (gray matter)
# but gray and csf are both brighter than white matter so it is ok
# threshold is where to consider frontier of the class
# margin is to remove the boundaries for which class points probabilities 
# are too close to threshold
def extract_boundaries(wmseg,bbr_dist,subsample=1,exclude=None,
                       threshold=.5,margin=.25):
    
    wmdata = wmseg.get_data().astype(np.float32)
    wmmask = (wmdata>threshold)
    gradient = np.empty(wmdata.shape+(3,),dtype=np.float32)
    for axis in xrange(wmdata.ndim): #convolve separable
        order = [0]*wmdata.ndim
        order[axis] = 1
        gaussian_filter(wmdata,0.2,order,gradient[...,axis])
    boundaries = wmmask - binary_erosion(wmmask)
    # allow to remove some boundaries points as trunk for pulsatility
    if exclude != None: 
        boundaries[exclude] = False
    if subsample > 1:
        #subsample only in slice plane not in slicing axis
        aux = np.zeros(boundaries.shape,boundaries.dtype)
        aux[::subsample,::subsample] = boundaries[::subsample,::subsample]
        boundaries[:] = aux
    n_bnd_pts = np.count_nonzero(boundaries)
    coords = np.asarray(np.where(boundaries),dtype=np.double)
    voxsize = np.sqrt((wmseg.get_affine()[:3,:3]**2).sum(0))
    gradient_mm = wmseg.get_affine()[:3,:3].dot(gradient[boundaries].T).T
    gradient_mm /= np.sqrt((gradient_mm**2).sum(-1))[:,np.newaxis]
    
    coords_mm = apply_affine(wmseg.get_affine(),coords.T)
    class_coords = np.empty((2,)+coords_mm.shape)
    class_coords[1] = coords_mm + gradient_mm*bbr_dist #climb gradient white
    class_coords[0] = coords_mm - gradient_mm*bbr_dist #go downhill gray

    # remove the points that would fall out of the aimed class
    wm_splines = _cspline_transform(wmseg.get_data())
    sample_values = np.empty(class_coords.shape[:2])
    class_voxs = apply_affine(np.linalg.inv(wmseg.get_affine()),
                              class_coords.reshape((-1,3)))
    _cspline_sample3d(sample_values,wm_splines,
                      class_voxs[:,0],class_voxs[:,1],class_voxs[:,2])
    valid_subset = np.logical_and(sample_values[0] < threshold+margin,
                                  sample_values[1] > threshold-margin)
    del class_voxs, sample_values, wm_splines
    
    return coords_mm[valid_subset],class_coords[:,valid_subset]


def fieldmap_to_sigloss(fieldmap,mask,echo_time,slicing_axis=2):
    tmp = fieldmap.copy()
    tmp[np.logical_not(mask)] = np.nan
    sel, sel2 = [slice(None)]*3,[slice(None)]*3
    sel[slicing_axis], sel2[slicing_axis] = slice(0,-1), slice(1,None)
    lrgradients = np.empty((2,)+mask.shape)
    lrgradients.fill(np.nan)
    lrgradients[[slice(0,1)]+sel2] = tmp[sel2]-tmp[sel]
    lrgradients[[slice(1,2)]+sel] = tmp[sel2]-tmp[sel]
    nans = np.logical_and(np.isnan(lrgradients),mask[np.newaxis])
    lrgradients[0,nans[0]] = lrgradients[1,nans[0]]
    lrgradients[1,nans[1]] = lrgradients[0,nans[1]]
    lrgradients[:,np.logical_not(mask)] = np.nan
    gbarte_2 = echo_time / 4.0 / np.pi
    sinc = np.sinc(gbarte_2*lrgradients)
    theta = np.pi * gbarte_2 * lrgradients
    re = 0.5 * (sinc[0]*np.cos(theta[0])+sinc[1]*np.cos(theta[1]))
    im = 0.5 * (sinc[0]*np.sin(theta[0])+sinc[1]*np.sin(theta[1]))
    sigloss = np.sqrt(re**2+im**2).astype(fieldmap.dtype)
    del lrgradients, nans, sinc, theta, re, im
    sigloss[np.isnan(sigloss)]=0
    return sigloss

class SliceImage4d(object):
    """
    Class to represent a sequence of 3d scans (possibly acquired on a
    slice-by-slice basis).

    Object remains empty until the data array is actually loaded in memory.

    Parameters
    ----------
      data : nd array or proxy (function that actually gets the array)
    """
    def __init__(self, data, affine, tr, tr_slices=None, start=0.0,
                 slice_order=SLICE_ORDER, interleaved=INTERLEAVED,
                 slice_trigger_times=None, slice_thickness=None,
                 slice_info=None):
        """
        Configure fMRI acquisition time parameters.
        """
        self.affine = np.asarray(affine)
        self.tr = float(tr)
        self.start = float(start)
        self.interleaved = bool(interleaved)

        # guess the slice axis and direction (z-axis)
        if slice_info == None:
            orient = io_orientation(self.affine)
            self.slice_axis = int(np.where(orient[:, 0] == 2)[0])
            self.slice_direction = int(orient[self.slice_axis, 1])
        else:
            self.slice_axis = int(slice_info[0])
            self.slice_direction = int(slice_info[1])

        # unformatted parameters
        self._tr_slices = tr_slices
        self._slice_order = slice_order
        self._slice_trigger_times = slice_trigger_times
        self._slice_thickness = slice_thickness

        self._vox_size = np.sqrt((self.affine[:3,:3]**2).sum(0))
        if self._slice_thickness == None:
            self._slice_thickness = self._vox_size[self.slice_axis]

        if isinstance(data, np.ndarray):
            self._data = data
            self._shape = data.shape
            self._get_data = None
            self._init_timing_parameters()
        else:
            self._data = None
            self._shape = None
            self._get_data = data

    def _load_data(self):
        self._data = self._get_data()
        self._shape = self._data.shape
        self._init_timing_parameters()

    def get_data(self):
        if self._data == None:
            self._load_data()
        return self._data
    
    def get_shape(self):
        if self._shape == None:
            self._load_data()
        return self._shape

    def _init_timing_parameters(self):
        # Number of slices
        nslices = self.get_shape()[self.slice_axis]
        self.nslices = nslices
        # Default slice repetition time (no silence)
        if self._tr_slices == None:
            self.tr_slices = self.tr / float(nslices)
        else:
            self.tr_slices = float(self._tr_slices)
        # Set slice order
        if isinstance(self._slice_order, str):
            if not self.interleaved:
                aux = range(nslices)
            else:
                aux = range(nslices)[0::2] + range(nslices)[1::2]
            if self._slice_order == 'descending':
                aux.reverse()
            self.slice_order = np.array(aux)
        else:
            # Verify correctness of provided slice indexes
            provided_slices = np.array(sorted(self._slice_order))
            if np.any(provided_slices != np.arange(nslices)):
                raise ValueError(
                    "Incorrect slice indexes were provided. There are %d "
                    "slices in the volume, indexes should start from 0 and "
                    "list all slices. "
                    "Provided slice_order: %s" % (nslices, self._slice_order))
            self.slice_order = np.asarray(self._slice_order)
        if self._slice_trigger_times == None:
            self._slice_trigger_times = np.arange(
                0,self.tr*self._shape[3],self.tr)[:,np.newaxis].repeat(
                nslices,axis=1)+self.slice_order[np.newaxis,:]*self.tr_slices

    def z_to_slice(self, z):
        """
        Account for the fact that slices may be stored in reverse
        order wrt the scanner coordinate system convention (slice 0 ==
        bottom of the head)
        """
        if self.slice_direction < 0:
            return self.nslices - 1 - z
        else:
            return z

    def slice_to_z(self, z):
        return self.slice_order[z]

    def scanner_time(self, zv, t):
        """
        tv = scanner_time(zv, t)
        zv, tv are grid coordinates; t is an actual time value.
        """
        corr = self.tr_slices * interp_slice_order(self.z_to_slice(zv),
                                                   self.slice_order)
        return (t - self.start - corr) / self.tr

    def free_data(self):
        if not self._get_data == None:
            self._data = None

class RealignSliceAlgorithm(object):

    def __init__(self,
                 im4d,
                 wmseg,
                 exclude_boundaries_mask=None,
                 fmap=None,
                 mask=None,
                 pe_dir=1,
                 echo_spacing=0.005,
                 echo_time=0.03,
                 bbr_dist=2.0,
                 affine_class=Rigid,
                 slice_groups=None,
                 transforms=None,
                 optimizer=OPTIMIZER,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 nsamples_per_slicegroup=2000):

        self.im4d = im4d
        self.dims = im4d.get_data().shape
        self.nscans = 1
        if len(self.dims) > 3:
            self.nscans = self.dims[3]
        self.reference = wmseg
        self.fmap = fmap
        self.mask = mask
        if self.mask != None:
            self.mask_data = self.mask.get_data()>0
        self.bnd_coords,self.class_coords = extract_boundaries(
            wmseg,bbr_dist,1,exclude_boundaries_mask)
        self.border_nvox = self.bnd_coords.shape[0]
        self.min_sample_number = 100

        self.slg_class_voxels = np.empty(self.class_coords.shape,np.double)

        self.echo_time = echo_time
        self.nsamples_per_slicegroup = nsamples_per_slicegroup
        self.pe_sign = int(pe_dir > 0)*2-1
        self.pe_dir = abs(pe_dir)
        self.fmap_scale=self.pe_sign*echo_spacing*self.dims[self.pe_dir]/2.0/np.pi
        self.affine_class = affine_class


        # Initialize space/time transformation parameters
        self.affine = im4d.affine
        self.inv_affine = np.linalg.inv(self.affine)
        #defines bunch of slices for witch same affine is estimated
        if slice_groups == None:
            self.slice_groups=[((t,0),(t,self.dims[im4d.slice_axis])) for t in range(self.nscans)]
            self.nparts=self.nscans
        else:
            self.slice_groups=slice_groups
            self.nparts=len(slice_groups)
        if transforms == None:
            self.transforms = []
        else:
            self.transforms = transforms

        self.scanner_time = im4d.scanner_time
        self.timestamps = im4d.tr * np.arange(self.nscans)
        self.st_ratio = self.im4d._slice_thickness/self.im4d._vox_size[im4d.slice_axis]/2.0

        # Compute the 3d cubic spline transform
        self.cbspline = np.zeros(self.dims, dtype='double')
        for t in range(self.dims[3]):
            self.cbspline[:, :, :, t] =\
                _cspline_transform(im4d.get_data()[:, :, :, t])
        # Compute the 4d cubic spline transform
#        self.cbspline = _cspline_transform(im4d.get_data())

        if self.fmap != None:
            self.sigloss = fieldmap_to_sigloss(
                self.fmap.get_data(),self.mask.get_data(),
                self.echo_time,self.im4d.slice_axis)
            fmap_inv_aff = np.linalg.inv(self.fmap.get_affine())
            fmap_vox = apply_affine(fmap_inv_aff,
                                    self.class_coords.reshape(-1,3))
            self.fmap_values = map_coordinates(
                self.fmap.get_data(), fmap_vox.T, order=1).reshape(2,self.border_nvox)
            self.sigloss_values = map_coordinates(
                self.sigloss, fmap_vox.T, order=1).reshape(2,self.border_nvox)
            del fmap_vox
        else:
            self.fmap_values = None
            self.sigloss_values = None
            
        # Set the minimization method
        self.set_fmin(optimizer, stepsize,
                      xtol=xtol,
                      ftol=ftol,
                      gtol=gtol,
                      maxiter=maxiter,
                      maxfun=maxfun)
        
        self.data = np.array([[]])
        self._pc = None
        self._last_subsampling_transform = affine_class(np.ones(12)*5)
        self._subset = np.ones(self.border_nvox,dtype=np.bool)
        self._subsamp = np.ones(self.border_nvox,dtype=np.bool)
        self._first_vol_subset = np.zeros(self.border_nvox, dtype=np.bool)
        self._last_vol_subset = np.zeros(self.border_nvox, dtype=np.bool)
        self._first_vol_subset_ssamp=np.zeros(self.border_nvox, dtype=np.bool)
        self._last_vol_subset_ssamp=np.zeros(self.border_nvox, dtype=np.bool)

        # store all data for more complex energy function with intensity prior
        self._all_data = np.zeros((self.nscans,self.border_nvox,2),np.float32)+np.nan
        self._allcosts=list()

    def apply_transform(self,transform,in_coords,out_coords,
                        fmap_values=None,subset=slice(None)):
        inv_trans = np.linalg.inv(transform.as_affine())
        ref2fmri = np.dot(self.inv_affine,inv_trans)
        #apply current slice group transform
        out_coords[...,subset,:]=apply_affine(ref2fmri,in_coords[...,subset,:])
        #add shift in phase encoding direction
        if fmap_values != None:
            out_coords[...,subset,self.pe_dir]+=fmap_values[...,subset]*self.fmap_scale
            
    def resample_slice_group(self, sgi):
        """
        Resample a particular slice group on the (sub-sampled) working grid.
        """
        tst = self.timestamps
        sa = self.im4d.slice_axis
        sg = self.slice_groups[sgi]

        inv_trans = np.linalg.inv(self.transforms[sgi].as_affine())
        ref2fmri = np.dot(self.inv_affine,inv_trans)
        
        # if change of test points z is above threshold recompute subset
        test_points = np.array([[0,0,sg[0][1]],[0,0,sg[1][1]]])
        recompute_subset = np.abs(
            self._last_subsampling_transform.apply(test_points)-
            self.transforms[sgi].apply(test_points))[:,sa].max() > 0.1
        
        if recompute_subset:
            self.apply_transform(self.transforms[sgi],
                                 self.class_coords,self.slg_class_voxels,
                                 self.fmap_values)

            self._last_subsampling_transform = self.transforms[sgi].copy()
            # adapt subsampling to keep regular amount of points in each slice
            nslices = self.dims[sa]
            zs = self.slg_class_voxels[...,sa].sum(0)/2.
            samples_slice_hist = np.histogram(zs,np.arange(nslices+1)-self.st_ratio)
            # this computation is wrong 
            self._subsamp[:] = False
            """
            maxsamp_per_slice = self.nsamples_per_slicegroup/nslices
            for sl,nsamp in enumerate(samples_slice_hist[0]):
                stp = max(int(nsamp/maxsamp_per_slice),1)
                tmp = np.where(np.abs(zs-sl)<0.5)[0]
                if tmp.size > 0:
                    self._subsamp[tmp[::stp]] = True
                    """
            step = np.floor(self._subsamp.shape[0]/float(self.nsamples_per_slicegroup))
            self._subsamp[::step] = True

            self._first_vol_subset[:] = (np.abs(zs[:,np.newaxis]-self.im4d.slice_to_z(np.arange(sg[0][1],nslices))[np.newaxis]) < self.st_ratio).sum(1) > 0
            self._last_vol_subset[:] = (np.abs(zs[:,np.newaxis]-self.im4d.slice_to_z(np.arange(0,sg[1][1]))[np.newaxis]) < self.st_ratio).sum(1) > 0


            if sg[0][0] == sg[1][0]:
                np.logical_and(self._last_vol_subset,
                               self._first_vol_subset,
                               self._first_vol_subset)
                np.logical_and(self._first_vol_subset,self._subsamp,
                               self._first_vol_subset_ssamp)
                self._last_vol_subset_ssamp.fill(False)
                self._last_vol_subset.fill(False)
                self._subset[:] = self._first_vol_subset[:]
                np.logical_and(self._subset, self._subsamp, self._subsamp)
            else:
                np.logical_and(self._last_vol_subset,self._subsamp,
                               self._last_vol_subset_ssamp)
                if sg[1][0] - sg[0][0] > 1:
                    self._subset.fill(True)
                else:
                    np.logical_and(self._first_vol_subset, self._subsamp, self._subsamp)
                    np.logical_and(self._last_vol_subset, self._subsamp, self._subsamp)
                
            print 'new subset %d samples'%self._subsamp.sum()

        else:
            self.apply_transform(self.transforms[sgi],
                                 self.class_coords,self.slg_class_voxels,
                                 self.fmap_values,self._subsamp)
        nsamples_1vol = np.count_nonzero(self._first_vol_subset_ssamp)
        self.skip_sg=False
        if nsamples_1vol < self.min_sample_number:
            print 'skipping slice group, only %d samples'%nsamples_1vol
            self.skip_sg = True
            return

        n_samples = np.count_nonzero(self._subsamp)
        n_samples_lvol = np.count_nonzero(self._last_vol_subset_ssamp)
        n_samples_total = nsamples_1vol + n_samples_lvol +\
            n_samples * max( sg[1][0]-sg[0][0]-1, 0)
            
        # if subsampling changes
        if self.data.shape[1] != n_samples_total:
            del self.data
            self.data = np.empty((2,n_samples_total))

        # resample per volume, split not optimal for many volumes ???
        self.resample(
            self.data[:,:nsamples_1vol],
            self.slg_class_voxels[:,self._first_vol_subset_ssamp],sg[0][0])
        for i in range(sg[0][0]+1, sg[1][0]):
            seek = nsamples_1vol + n_samples * (i - sg[0][0] -1)
            self.resample(self.data[:,seek:seek+n_samples],
                          self.slg_class_voxels[:,self._subsamp],i)
        if sg[0][0] < sg[1][0]:
            self.resample(
                self.data[:,-n_samples_lvol:None],
                self.slg_class_voxels[:,self._last_vol_subset_ssamp],sg[1][0])

        if False:
        #else:
            n_samples = self._first_vol_subset_ssamp.sum() +\
                self._subset.sum()*(sg[1][0]-sg[0][0]) +\
                self._last_vol_subset_ssamp.sum()
            t = np.empty()
            """
            t_gm = np.concatenate(
            [self.scanner_time(self.slg_gm_vox[self._first_vol_subset_ssamp,sa],
                               tst[sg[0][0]])]+
            [self.scanner_time(self.slg_gm_vox[self._subset,sa],
                               tst[t]) for t in xrange(sg[0][0]+1,sg[1][0])]+
            [self.scanner_time(self.slg_gm_vox[self._last_vol_subset_ssamp,sa],
                               tst[sg[1][0]])])
            """

    def resample(self,out,coords,time=None):
        if not isinstance(time,np.ndarray):
            _cspline_sample3d(
                out,self.cbspline[...,time],
                coords[...,0], coords[...,1], coords[...,2],
                mx=EXTRAPOLATE_SPACE,
                my=EXTRAPOLATE_SPACE,
                mz=EXTRAPOLATE_SPACE,)
        else:
            _cspline_sample4d(
                out,self.cbspline,
                coords[...,0], coords[...,1], coords[...,2], time,
                mx=EXTRAPOLATE_SPACE,
                my=EXTRAPOLATE_SPACE,
                mz=EXTRAPOLATE_SPACE,
                mt=EXTRAPOLATE_TIME)

    def resample_full_data(self, voxsize=None, reference=None):
        # TODO, time interpolation, slice group, ...
        if VERBOSE:
            print('Gridding...')
        mat=self.reference.get_affine()
        shape = self.reference.shape
        if reference != None:
            mat,shape = reference.get_affine(),reference.shape
            voxsize = reference.get_header().get_zooms()[:3]
        elif voxsize!=None:
            mat,shape=resample_mat_shape(self.reference.get_affine(),
                                         self.reference.shape,voxsize)
        new_to_t1 = np.linalg.inv(self.fmap.get_affine()).dot(mat)
        xyz = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]],0,4)
        interp_coords = apply_affine(new_to_t1,xyz)
        res = np.zeros(shape+(self.nscans,), dtype=np.float32)
        if self.mask != None:
            mask = map_coordinates(self.mask.get_data(),
                                   interp_coords.reshape(-1,3).T,
                                   order=0).reshape(shape)
            mask = mask>.5
            xyz = xyz[mask]
            interp_coords = interp_coords[mask]
        tmp = np.zeros(xyz.shape[:-1])
        sa = self.im4d.slice_axis
        subset = np.zeros(shape,dtype=np.bool)
        if self.fmap != None:
            fmap_values = map_coordinates(self.fmap.get_data(),
                                          interp_coords.reshape(-1,3).T,
                                          order=1).reshape(xyz.shape[:-1])
            fmap_values *= self.fmap_scale
            print 'fieldmap ranging from %f to %f'%(fmap_values.max(),
                                                    fmap_values.min())
        for t in range(self.nscans):
            sgs_trs = [(sg,trans) for sg,trans in zip(self.slice_groups,self.transforms) if sg[0][0]<=t and t<= sg[1][0]]
            if len(sgs_trs)==1: #trivial case
                print 'easy peasy'
                interp_coords[...] = apply_affine(np.dot(
                    np.dot(self.inv_affine,np.linalg.inv(trans.as_affine())),
                    mat),xyz)
                if self.fmap != None:
                    interp_coords[...,self.pe_dir] += fmap_values
            else: # we have to solve from which transform we sample
                print 'more tricky'
                for sg,trans in sgs_trs:
                    coords = apply_affine(np.dot(
                      np.dot(self.inv_affine,np.linalg.inv(trans.as_affine())),
                      mat), xyz)
                    if self.fmap != None:
                        coords[...,self.pe_dir] += fmap_values
                    subset.fill(False)
                    
                    if sg[0][0]==t and sg[1][0]==t:
                        times = np.arange(sg[0][1],sg[1][1])
                    elif sg[0][0]==t:
                        times = np.arange(sg[0][1],nslices)
                    elif sg[1][0]==t:
                        times = np.arange(sg[0][1],nslices)
                    else:
                        times = np.arange(0,nslices)
                    subset = (np.abs(coords[...,sa,np.newaxis]-self.im4d.slice_to_z(times)[np.newaxis]) < self.st_ratio+.1).sum(-1) > 0
                        
                    interp_coords[subset,:] = coords[subset]
            self.resample(tmp,interp_coords,t)
            if self.mask !=None:
                res[mask,t] = tmp
            else:
                res[...,t] = tmp
            print t
        out_nii = nb.Nifti1Image(res,mat)
        out_nii.get_header().set_xyzt_units('mm','sec')
        out_nii.get_header().set_zooms(voxsize + (self.im4d.tr,))
        return out_nii

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
        

    def set_transform(self, t, pc):
        self.transforms[t].param = pc
        self.resample_slice_group(t)

    def _init_energy(self, pc):
        if pc is self._pc:
            return
        self.set_transform(self._t, pc)
        self._pc = pc

        if self.use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            self._aux = self.data
            nrgy = self._energy()
            print 'energy %f'% nrgy
            basis = np.eye(pc.size,dtype=np.bool)
            A=np.zeros((pc.size,pc.size))
            A2=np.zeros(pc.size)
            for j in range(pc.size):
                for k in range(j,pc.size):
                    self.set_transform(self._t, 
                                       pc + self.stepsize*(basis[j]+basis[k]))
                    A[j,k]=self._energy()
                self.set_transform(self._t, pc - self.stepsize*basis[j])
                A2[j]=self._energy()
            
            self.transforms[self._t].param = pc
            self.data = self._aux
            # pre-compute gradient and hessian of numerator and
            # denominator
            tril = np.tri(pc.size, k=-1,dtype=np.bool)
            self._dV = (A.diagonal() - nrgy)/self.stepsize*2.0
            self._H = ((A-A.diagonal()-A.diagonal()[:,np.newaxis]+nrgy)*tril.T+
                       np.diag(((A.diagonal()-A2)/2-(A.diagonal()-nrgy))) )* 2.0/self.stepsize

            self._H[tril] = self._H.T[tril]

    def _energy(self):
        percent_contrast = 200*np.diff(self.data,1,0)/self.data.sum(0)
        percent_contrast[np.abs(percent_contrast)<1e-6] = 0
        bbr_offset=0 # TODO add as an option, and add weighting
        cost = (1.0+np.tanh(percent_contrast-bbr_offset)).mean()
        return cost

    def _energy_gradient(self):
        print 'gradient', self._dV
        return self._dV

    def _energy_hessian(self):
        print 'hessian',self._H
        return self._H

    def estimate_instant_motion(self, sg):
        """
        Estimate motion parameters at a particular time.
        """
        if VERBOSE:
            print('Estimating motion at slice group %d/%d...'
                  % (sg + 1, self.nparts))
        if len(self.transforms) <= sg:
            if sg > 0 and len(self.transforms) == sg:
                self.transforms.append(self.transforms[sg-1].copy())
            else:
                self.transforms.append(self.affine_class())
        self._last_subsampling_transform = self.affine_class(np.ones(12)*5)
        self.resample_slice_group(sg)

        self.set_transform(sg,self.transforms[sg].param)
        if self.skip_sg:
            return

        def f(pc):
            self._init_energy(pc)
            nrgy = self._energy()
            print 'f %f : %f %f %f %f %f %f'%tuple([nrgy] + pc.tolist())
            return nrgy

        def fprime(pc):
            self._init_energy(pc)
            return self._energy_gradient()

        def fhess(pc):
            print 'fhess'
            self._init_energy(pc)
            return self._energy_hessian()

        self._pc = None
        self._t = sg
        fmin, args, kwargs =\
            configure_optimizer(self.optimizer,
                                fprime=fprime,
                                fhess=fhess,
                                **self.optimizer_kwargs)

        # With scipy >= 0.9, some scipy minimization functions like
        # fmin_bfgs may crash due to the subroutine
        # `scalar_search_armijo` returning None as a stepsize when
        # unhappy about the objective function. This seems to have the
        # potential to occur in groupwise registration when using
        # strong image subsampling, i.e. at the coarser levels of the
        # multiscale pyramid. To avoid crashes, we insert a try/catch
        # instruction.
#        try:
        pc = fmin(f, self.transforms[sg].param, *args, **kwargs)
        self.set_transform(sg, pc)

        # resample all points in slice group
        self.apply_transform(self.transforms[sg],self.class_coords,
                             self.slg_class_voxels,self.fmap_values)
        tmp = np.empty(self.class_coords.shape[:2])
        sgp = self.slice_groups[sg]
        for t in range(sgp[0][0],sgp[1][0]+1):
            sbst=self._subset
            nsamp = sbst.sum(0)
            if t==sgp[0][0]:
                sbst = self._first_vol_subset
                nsamp=sbst.sum(0)
            elif t==sgp[1][0]:
                sbst = self._last_vol_subset
                nsamp=sbst.sum(0)
            self.resample(
                tmp[:,:nsamp],self.slg_class_voxels[:,sbst],sgp[0][0])
            self._all_data[sgp[0][0],sbst,:] = tmp[:,:nsamp].T
        #except:
        #warnings.warn('Minimization failed')

    def estimate_motion(self):
        """
        Optimize motion parameters for the whole sequence. All the
        time frames are initially resampled according to the current
        space/time transformation, the parameters of which are further
        optimized sequentially.
        """
        
        for sg in range(self.nparts):
            print 'slice group %d' % sg
            self.estimate_instant_motion(sg)
            if VERBOSE:
                print(self.transforms[sg])


def resample_mat_shape(mat,shape,voxsize):
    k=np.diag(mat[:3,:3].dot(np.diag(np.array(shape)-1.0)))
    newshape = np.round(k/voxsize)
    res = k-newshape*voxsize
    old_voxsize=np.sqrt((mat[:3,:3]**2).sum(0))
    newmat=np.eye(4)
    newmat[:3,:3] = np.diag((voxsize/old_voxsize)).dot(mat[:3,:3])
    newmat[:3,3] = mat[:3,3]+res/2
    newshape = np.abs(newshape).astype(np.int32)
    return newmat,tuple(newshape.tolist())
