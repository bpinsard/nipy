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
    class_voxs = apply_affine(np.linalg.inv(wmseg.get_affine()),
                              class_coords.reshape((-1,3)))
    # linear interpolation to avoid negative values
    sample_values = map_coordinates(wmseg.get_data(), class_voxs.T, order=1
                                    ).reshape(class_coords.shape[:2])
    valid_subset = np.logical_and(sample_values[0] < threshold+margin,
                                  sample_values[1] > threshold-margin)
    del class_voxs, sample_values
    
    return coords_mm[valid_subset],class_coords[:,valid_subset]

#old sigloss computation : wrong if epi/fmao orientation are really different
def fieldmap_to_sigloss(fieldmap,mask,echo_time,slicing_axis=2,scaling=1):
    tmp = fieldmap.copy()
    tmp[np.logical_not(mask)] = np.nan
    sel, sel2 = [slice(None)]*3,[slice(None)]*3
    sel[slicing_axis], sel2[slicing_axis] = slice(0,-1), slice(1,None)
    lrgradients = np.empty(mask.shape+(2,))
    lrgradients.fill(np.nan)
    lrgradients[sel2+[slice(0,1)]] = tmp[sel2]-tmp[sel]
    lrgradients[sel+[slice(1,2)]] = tmp[sel2]-tmp[sel]
    nans = np.logical_and(np.isnan(lrgradients),mask[...,np.newaxis])
    lrgradients[nans[...,0],0] = lrgradients[nans[...,0],1]
    lrgradients[nans[...,1],1] = lrgradients[nans[...,1],0]
    lrgradients[np.logical_not(mask),:] = np.nan
    if scaling != 1 :
        lrgradients*=scaling
    gbarte_2 = echo_time / 4.0 / np.pi
    sinc = np.sinc(gbarte_2*lrgradients)
    theta = np.pi * gbarte_2 * lrgradients
    re = 0.5 * (sinc[...,0]*np.cos(theta[...,0])+
                sinc[...,1]*np.cos(theta[...,1]))
    im = 0.5 * (sinc[...,0]*np.sin(theta[...,0])+
                sinc[...,1]*np.sin(theta[...,1]))
    sigloss = np.sqrt(re**2+im**2).astype(fieldmap.dtype)
    del lrgradients, nans, sinc, theta, re, im
    sigloss[np.isnan(sigloss)]=0
    return sigloss

# estime sigloss in EPI space using fieldmap with another sampling.
# modified from FSL sigloss
def compute_sigloss(fieldmap,mask,
                    reg,fmat,pts,
                    echo_time,slicing_axis=2,order=0):
    fmri2wld = reg.dot(fmat)
    shift_points = np.empty(pts.shape+(2,))
    sv = np.zeros(3)
    sv[slicing_axis] = 1
    world_shift = fmri2wld[:3,:3].dot(sv)
    shift_points[...,0] = nb.affines.apply_affine(
        np.linalg.inv(fieldmap.get_affine()),
        pts-world_shift[np.newaxis,np.newaxis,np.newaxis,:])
    shift_points[...,1] = nb.affines.apply_affine(
        np.linalg.inv(fieldmap.get_affine()), 
        pts+world_shift[np.newaxis,np.newaxis,np.newaxis,:])
    tmp = fieldmap.get_data()
    tmp[mask==0] = np.nan
    pts = nb.affines.apply_affine(np.linalg.inv(fieldmap.get_affine()),pts)
    fmap_values = map_coordinates(
        tmp, np.concatenate((
                np.rollaxis(pts,-1,0)[...,np.newaxis],
                np.rollaxis(shift_points,-2,0)),-1).reshape(3,-1),
        order=order,cval=np.nan).reshape(pts.shape[:-1]+(3,))
    lrgradients = np.empty(pts.shape[:-1]+(2,))
    lrgradients[...,0] = fmap_values[...,1] - fmap_values[...,0]
    lrgradients[...,1] = fmap_values[...,2] - fmap_values[...,0]
    gbarte_2 = echo_time / 4.0 / np.pi
    sinc = np.sinc(gbarte_2*lrgradients)
    theta = np.pi * gbarte_2 * lrgradients
    re = 0.5 * (sinc[...,0]*np.cos(theta[...,0])+
                sinc[...,1]*np.cos(theta[...,1]))
    im = 0.5 * (sinc[...,0]*np.sin(theta[...,0])+
                sinc[...,1]*np.sin(theta[...,1]))
    sigloss = np.sqrt(re**2+im**2).astype(fieldmap.get_data_dtype())
    del lrgradients, sinc, theta, re, im, pts, fmap_values
    sigloss[np.isnan(sigloss)]=0
    return sigloss

class EPIInterpolation(object):

    """ Class to handle interpolation on a slice group basis,
    optional fieldmap based epi unwarping ... """

    def __init__(self,
                 epi,
                 slice_groups,
                 transforms,
                 fieldmap,
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
        self.epi, self.fmap, self.mask = epi, fieldmap, mask
        self.slice_groups = slice_groups
        self.transforms = transforms
        self.inv_affine = np.linalg.inv(self.epi.get_affine())

        self.slice_axis = slice_axis
        self.slice_order = slice_order
        self.pe_sign = int(phase_encoding_dir > 0)*2-1
        self.pe_dir = abs(phase_encoding_dir)
        self.repetition_time = repetition_time
        self.slice_tr = slice_repetition_time
        self.interleaved = int(interleaved)
        self.slice_trigger_times = slice_trigger_times
        self.slice_thickness = slice_thickness
        self.voxsize = np.sqrt((self.epi.get_affine()[:3,:3]**2).sum(0))

        if self.fmap != None:
            self.fmap_scale = self.pe_sign*echo_spacing* \
                self.epi.shape[self.pe_dir]/2.0/np.pi
        
        self.nslices = self.epi.shape[self.slice_axis]
        self.nvolumes = self.epi.shape[-1]

        #defines bunch of slices for witch same affine is estimated
        if slice_groups == None:
            self.slice_groups = [((t,0),(t,self.nslices)) \
                                     for t in range(self.nvolumes)]
            self.nslice_groups=self.nvolumes
        else:
            self.slice_groups = slice_groups
            self.nslice_groups=len(slice_groups)
        if transforms == None:
            self.transforms = []
        else:
            self.transforms = transforms

        self.st_ratio = 1
        if self.slice_thickness != None:
            self.st_ratio = self.slice_thickness/self.voxsize[self.slice_axis]/2.0

        # Default slice repetition time (no silence)
        if self.slice_tr == None:
            self.slice_tr = self.repetition_time / float(self.nslices)
        else:
            self.slice_tr = float(self.slice_tr)
        # Set slice order
        if isinstance(self.slice_order, str):
            if self.interleaved < 2:
                aux = range(self.nslices)
            else:
                aux = reduce(
                    lambda l,s: l+range(self.nslices)[s::self.interleaved],
                    range(self.interleaved),[])
            if self.slice_order == 'descending':
                aux.reverse()
            self.slice_order = np.array(aux)
        else:
            # Verify correctness of provided slice indexes
            provided_slices = np.array(sorted(self.slice_order))
            if np.any(provided_slices != np.arange(self.nslices)):
                raise ValueError(
                    "Incorrect slice indexes were provided. There are %d "
                    "slices in the volume, indexes should start from 0 and "
                    "list all slices. "
                    "Provided slice_order: %s"%(self.nslices,self.slice_order))
            self.slice_order = np.asarray(self.slice_order)
        if self.slice_trigger_times == None:
            self.slice_trigger_times = np.arange(
                0,self.repetition_time*self.nslices,
                self.repetition_time)[:,np.newaxis].repeat(
                self.nslices,axis=1)+self.slice_order[np.newaxis,:]*self.slice_tr
                
        self.cbspline = np.zeros(self.epi.shape, dtype='double')
        for t in range(self.epi.shape[-1]):
            self.cbspline[:, :, :, t] =\
                _cspline_transform(epi.get_data()[:, :, :, t])


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

    def resample_volume(self, voxsize=None, reference=None):
        if reference != None:
            if voxsize == None:
                voxsize = reference.get_header().get_zooms()[:3]
            mat,shape = resample_mat_shape(
                reference.get_affine(),reference.shape, voxsize)
        elif voxsize!=None:
            mat,shape=resample_mat_shape(self.mask.get_affine(),
                                         self.mask.shape,voxsize)
        xyz = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]],0,4)
        if self.mask != None:
            ref2mask = np.linalg.inv(self.mask.get_affine()).dot(mat)
            interp_coords = apply_affine(ref2mask, xyz)
            mask = map_coordinates(self.mask.get_data(),
                                   interp_coords.reshape(-1,3).T,
                                   order=0).reshape(shape)
            mask = mask>.5
            xyz = xyz[mask]
        interp_coords = apply_affine(mat, xyz)
        if self.mask != None:
            res = np.zeros(shape+(self.nvolumes,))
            res[mask>0] = self.resample_coords(interp_coords)
        else:
            res = self.resample_coords(interp_coords)
        out_nii = nb.Nifti1Image(res,mat)
        out_nii.get_header().set_xyzt_units('mm','sec')
        out_nii.get_header().set_zooms(voxsize + (self.repetition_time,))
        return out_nii

        """
        if rois!=None:
            ref2roi = np.linalg.inv(rois.get_affine()).dot(mat)
            interp_coords = apply_affine(ref2roi, xyz)
            resam_rois = map_coordinates(rois.get_data(),
                                         interp_coords.reshape(-1,3).T,
                                         order=0)
            xyz = xyz[resam_rois>0]
            rois_idx = np.sort(np.unique(resam_rois[resam_rois>0]))
            res = [np.empty((np.count_nonzero(resam_rois==i),self.nvolumes),
                            self.epi.get_data_dtype()) for i in rois_idx]
        else:
            res = np.empty(shape+(self.nvolumes,),self.epi.get_data_dtype())
            """

    def resample_coords(self, coords):

        if self.fmap != None:
            ref2fmap = np.linalg.inv(self.fmap.get_affine())
            interp_coords = apply_affine(ref2fmap, coords)
            fmap_values = self.fmap_scale * \
                map_coordinates(self.fmap.get_data(),
                                interp_coords.reshape(-1,3).T,
                                order=1).reshape(interp_coords.shape[:-1])

        subset = np.zeros(coords.shape[:-1], dtype=np.bool)
        tmp = np.empty(coords.shape[:-1])
        res = np.empty(tmp.shape+(self.nvolumes,),np.float32)
        for t in range(self.nvolumes):
            # select slice groups containing volume slices
            sgs_trs = [ (sg,trans) for sg,trans \
                            in zip(self.slice_groups,self.transforms) \
                            if sg[0][0] <= t and t <= sg[1][0] ]
            if len(sgs_trs) == 1: #easy, one transform per volume
                inv_reg = np.linalg.inv(sgs_trs[0][1].as_affine())
                wld2epi = self.inv_affine.dot(inv_reg)
                interp_coords[...] = apply_affine(wld2epi, coords)
                if self.fmap != None:
                    interp_coords[...,self.pe_dir] += fmap_values
            else: # we have to solve which transform we sample with
                for sg,trans in sgs_trs:
                    inv_reg = np.linalg.inv(trans.as_affine())
                    wld2epi = self.inv_affine.dot(inv_reg)
                    interp_coords = apply_affine(wld2epi, coords)
                    if self.fmap != None:
                        interp_coords[...,self.pe_dir] += fmap_values
                    subset.fill(False)
                    
                    if sg[0][0]==t and sg[1][0]==t:
                        times = np.arange(sg[0][1],sg[1][1])
                    elif sg[0][0]==t:
                        times = np.arange(sg[0][1],nslices)
                    elif sg[1][0]==t:
                        times = np.arange(sg[0][1],nslices)
                    else:
                        times = np.arange(0,nslices)
                    subset = np.any(
                        np.abs(coords[...,self.slicing_axis,np.newaxis]-
                               self.slice_order[times][np.newaxis]) \
                            < self.st_ratio+.1, -1)
                    interp_coords[subset,:] = coords[subset]
            self.resample(tmp,interp_coords,t)
            res[...,t] = tmp
            print t
        return res
    

    def _epi_inv_shiftmap(self, t):
        # compute inverse shift map using approximate nearest neighbor
        #
        fmap2fmri = np.linalg.inv(t.as_affine().dot(
                self.epi.get_affine())).dot( self.fmap.get_affine())
        coords = nb.affines.apply_affine(
            fmap2fmri,
            np.rollaxis(np.mgrid[[slice(0,s) for s in self.fmap.shape]],0,4))
        shift = self.fmap_scale * self.fmap.get_data()
        coords[...,self.pe_dir] += shift
        coords = coords[shift!=0]
        shift = shift[shift!=0]
        inv_shiftmap = np.empty(self.epi.shape[:-1])
        inv_shiftmap_dist = np.empty(self.epi.shape[:-1])
        inv_shiftmap.fill(np.inf)
        inv_shiftmap_dist.fill(np.inf)
        rcoords = np.round(coords)
        dists = np.sum((coords-rcoords)**2,-1)
        for c,d,s in zip(rcoords,dists,shift):
            if d < inv_shiftmap_dist[c[0],c[1],c[2]] \
                    and s < inv_shiftmap[c[0],c[1],c[2]]:
                inv_shiftmap[c[0],c[1],c[2]] = -s
        return inv_shiftmap

    def inv_resample(self, vol, ti=0, order=0):
        grid = np.rollaxis(np.mgrid[[slice(0,s) for s in self.epi.shape[:-1]]], 0, 4)
        if self.fmap != None:
            inv_shift = self._epi_inv_shiftmap(self.transforms[ti])
            grid[...,self.pe_dir] += inv_shift
        epi2vol = np.linalg.inv(vol.get_affine()).dot(
            self.transforms[ti].as_affine().dot(self.epi.get_affine()))
        voxs = nb.affines.apply_affine(epi2vol, grid)
        rvol = map_coordinates(
            vol.get_data(),
            voxs.reshape(-1,3).T, order=order).reshape(self.epi.shape[:-1])
        return rvol


    def resample_surface(self, vertices, triangles,thickness,project_frac=.5):
        normals = vertices_normals(vertices,triangles)
        interp_coords = vertices + \
            (normals * thickness[:,np.newaxis] * project_frac)
        return self.resample_coords(interp_coords)

        

class RealignSliceAlgorithm(EPIInterpolation):

    def __init__(self,
                 epi,

                 bnd_coords,
                 class_coords,

                 slice_groups,
                 transforms,
                 fieldmap=None,
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

                 affine_class=Rigid,
                 optimizer=OPTIMIZER,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 nsamples_per_slicegroup=2000,
                 min_nsamples_per_slicegroup=100):

        super(RealignSliceAlgorithm,self).__init__(
            epi, slice_groups, transforms,
            fieldmap, mask,
            phase_encoding_dir,
            repetition_time, slice_repetition_time, echo_time, echo_spacing,
            slice_order, interleaved, slice_trigger_times, 
            slice_thickness, slice_axis)

        if self.mask != None:
            self.mask_data = self.mask.get_data()>0
        self.bnd_coords,self.class_coords = bnd_coords, class_coords
        self.border_nvox = self.bnd_coords.shape[0]
        
        self.nsamples_per_slicegroup = nsamples_per_slicegroup
        self.min_sample_number = min_nsamples_per_slicegroup

        self.slg_class_voxels = np.empty(self.class_coords.shape,np.double)
        self.affine_class = affine_class

        self.affine = epi.get_affine()
        self.inv_affine = np.linalg.inv(self.affine)

        # Compute the 3d cubic spline transform
        self.cbspline = np.zeros(self.epi.shape, dtype='double')
        for t in range(self.cbspline.shape[-1]):
            self.cbspline[:, :, :, t] =\
                _cspline_transform(epi.get_data()[:, :, :, t])
        # Compute the 4d cubic spline transform
#        self.cbspline = _cspline_transform(im4d.get_data())

        if self.fmap != None:
            fmap_inv_aff = np.linalg.inv(self.fmap.get_affine())
            fmap_vox = apply_affine(fmap_inv_aff,
                                    self.class_coords.reshape(-1,3))
            self.fmap_values = self.fmap_scale * map_coordinates(
                self.fmap.get_data(), fmap_vox.T,
                order=1).reshape(2,self.border_nvox)
            del fmap_vox
        else:
            self.fmap_values = None
            
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
        self._all_data=np.empty((self.nvolumes,self.border_nvox,2),np.float32)
        self._all_data.fill(np.nan)
#        self._allcosts=list()

    def apply_transform(self,transform,in_coords,out_coords,
                        fmap_values=None,subset=slice(None)):
        inv_trans = np.linalg.inv(transform.as_affine())
        ref2fmri = np.dot(self.inv_affine,inv_trans)
        #apply current slice group transform
        out_coords[...,subset,:]=apply_affine(ref2fmri,in_coords[...,subset,:])
        #add shift in phase encoding direction
        if fmap_values != None:
            out_coords[...,subset,self.pe_dir]+=fmap_values[...,subset]
            
    def resample_slice_group(self, sgi):
        """
        Resample a particular slice group on the (sub-sampled) working grid.
        """
        sa = self.slice_axis
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
            nslices = self.epi.shape[sa]
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

            self._first_vol_subset[:] = np.any(
                np.abs(zs[:,np.newaxis]-self.slice_order[np.arange(sg[0][1],nslices)][np.newaxis]) < self.st_ratio, 1) 
            self._last_vol_subset[:] = np.any(np.abs(zs[:,np.newaxis]-self.slice_order[np.arange(0,sg[1][1])][np.newaxis]) < self.st_ratio,1)

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

        n_samples = np.count_nonzero(self._subsamp)
        n_samples_lvol = np.count_nonzero(self._last_vol_subset_ssamp)
        n_samples_total = nsamples_1vol + n_samples_lvol +\
            n_samples * max( sg[1][0]-sg[0][0]-1, 0)

        self.skip_sg=False
        if n_samples_total < self.min_sample_number:
            print 'skipping slice group, only %d samples'%n_samples_total
            self.skip_sg = True
            return
            
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
        if sg[0][0] < sg[1][0] and n_samples_lvol > 0:
            self.resample(
                self.data[:,-n_samples_lvol:None],
                self.slg_class_voxels[:,self._last_vol_subset_ssamp],sg[1][0])
        if sg[0][0]>0 and False:
            diff = self.data[:nsamples_1vol]-self._all_data[:sg[0][0],self._first_vol_subset_ssamp].mean(0).T
            motion = self._previous_class_voxel[:,self._first_vol_subset_ssamp] - self.slg_class_voxels[:,self._first_vol_subset_ssamp]
            drms = np.sqrt((motion**2).sum(-1))
            print 'diffstd %f %f'%(diff[0].std(),diff[1].std())

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
                  % (sg + 1, self.nslice_groups))
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
        self._previous_class_voxel = self.slg_class_voxels.copy()
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
        
        for sg in range(self.nslice_groups):
            print 'slice group %d' % sg
            self.estimate_instant_motion(sg)
            if VERBOSE:
                print(self.transforms[sg])


def resample_mat_shape(mat,shape,voxsize):
    old_voxsize = np.sqrt((mat[:3,:3]**2).sum(0))
    k = old_voxsize*np.array(shape)
    newshape = np.round(k/voxsize)
    res = k-newshape*voxsize
    newmat = np.eye(4)
    newmat[:3,:3] = np.diag((voxsize/old_voxsize)).dot(mat[:3,:3])
    newmat[:3,3] = mat[:3,3]+newmat[:3,:3].dot(res/voxsize/2)
    return newmat,tuple(newshape.astype(np.int32).tolist())


def intensity_sd_heuristic(alg,mask):
    # return split of run in slice and also stable vs. motion slices
    ddata = np.diff(
        alg.epi.get_data().reshape((-1,)+alg.epi.get_shape()[2:]), 1, -1)
    slice_std = ddata.std(0)
    global_std = ddata.std()
    peaks = slice_std > 1.96*global_std
    timepeaks = peaks[im4d.slice_order]
    np.where(np.logical_not(timepeaks))[0]
    
    return runstd
    

def vertices_normals(vertices,triangles):
    norm = np.zeros(vertices.shape,dtype=vertices.dtype)
    tris = vertices[triangles]
    n = np.cross(tris[::,1 ]-tris[::,0], tris[::,2 ]-tris[::,0])
    n /= np.sqrt((n**2).sum(-1))[:,np.newaxis]
    norm[triangles.ravel()] += n.repeat(3,0)
    norm /= np.sqrt((norm**2).sum(-1))[:,np.newaxis]
    return norm
    
def surface_to_samples(vertices, triangles, bbr_dist):
    normals = vertices_normals(vertices,triangles)
    class_coords = np.empty((2,)+vertices.shape)
    class_coords[0] = vertices + normals*bbr_dist
    class_coords[1] = vertices - normals*bbr_dist
    return class_coords

def vox2ras_tkreg(voldim, voxres):
    return np.array([
            [-voxres[0], 0, 0, voxres[0]*voldim[0]/2],
            [0, 0, voxres[2], -voxres[2]*voldim[2]/2],
            [0,-voxres[1],0,voxres[1]*voldim[1]/2],
            [0,0,0,1]])
