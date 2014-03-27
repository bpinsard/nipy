import numpy as np

import nibabel as nb, dicom
from nibabel.affines import apply_affine
from ...fixes.nibabel import io_orientation
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .affine import Rigid

from .optimizer import configure_optimizer, use_derivatives
from ._registration import (_cspline_transform,
                            _cspline_sample2d,
                            _cspline_sample3d,
                            _cspline_sample4d)
from scipy.ndimage import convolve1d, gaussian_filter, binary_erosion
import scipy.stats
from scipy.ndimage.interpolation import map_coordinates
from .slice_motion import surface_to_samples, compute_sigloss, intensity_factor


import time

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

    def resample(self, data, out, voxcoords,time=None, splines=None):
        if splines is None:
            splines = _cspline_transform(data)
        _cspline_sample3d(
            out,splines,
            voxcoords[...,0], voxcoords[...,1], voxcoords[...,2],
            mx=EXTRAPOLATE_SPACE,
            my=EXTRAPOLATE_SPACE,
            mz=EXTRAPOLATE_SPACE,)
        return splines


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
        if hasattr(self,'_inv_shiftmap') and self._inv_shiftmap.shape == shape\
                and np.allclose(self._invshiftmap_affine,affine):
            return self._inv_shiftmap
        self._invshiftmap_affine = affine

        fmap2fmri = np.linalg.inv(affine).dot(self.fmap2world)
        coords = nb.affines.apply_affine(
            fmap2fmri,
            np.rollaxis(np.mgrid[[slice(0,s) for s in self.fmap.shape]],0,4))
        shift = self.fmap_scale * self.fmap.get_data()
        coords[...,self.pe_dir] += shift
#        coords = coords[shift!=0]
#        shift = shift[shift!=0]
        if hasattr(self,'_inv_shiftmap'):
            del self._inv_shiftmap
        self._inv_shiftmap = np.empty(shape)
        inv_shiftmap_dist = np.empty(shape)
        self._inv_shiftmap.fill(np.inf)
        inv_shiftmap_dist.fill(np.inf)
        includ = np.logical_and(
            np.all(coords>-.5,-1),
            np.all(coords<np.array(shape)[np.newaxis]-.5,-1))
        coords = coords[includ]
        rcoords = np.round(coords).astype(np.int)
        dists = np.sum((coords-rcoords)**2,-1)
        shift = shift[includ]
        self._inv_shiftmap[(rcoords[...,0],rcoords[...,1],rcoords[...,2])] = shift
#        for c,d,s in zip(rcoords,dists,shift):
#            if  d < inv_shiftmap_dist[c[0],c[1],c[2]] \
#                    and s < self._inv_shiftmap[c[0],c[1],c[2]]:
#                self._inv_shiftmap[c[0],c[1],c[2]] = -s
        for x,y,z in zip(*np.where(np.isinf(self._inv_shiftmap))):
            ngbd = self._inv_shiftmap[
                max(0,x-1):x+1,max(0,y-1):y+1,max(0,z-1):z+1]
            self._inv_shiftmap[x,y,z] = ngbd.ravel()[np.argmin(np.abs(ngbd.ravel()))]
            del ngbd
        return self._inv_shiftmap
            
    def inv_resample(self, vol, affine, shape, order=0, mask=slice(None)):
        ## resample a undistorted volume to distorted EPI space
        # order = map_coordinates order, if -1, does integral of voxels in the
        # higher resolution volume (eg. for partial volume map downsampling)
        if order == -1:
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
                del fmap_voxs
                voxs[:, self.pe_dir] += fmap_values
                del fmap_values
            np.round(voxs, out=voxs)
            nvols = (vol.shape+(1,))[:4][-1]
            rvol = np.empty(shape+(nvols,))
            bins = [np.arange(-.5,d+.5) for d in shape]
            counts, _ = np.histogramdd(voxs,bins)
            for v in range(nvols):
                rvol[...,v] = np.histogramdd(
                    voxs, bins,weights=vol.get_data()[...,v][mask].ravel())[0]/counts
            rvol[np.isnan(rvol)] = 0 
            del counts, bins
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
                 min_nsamples_per_slab=100):

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
        self.st_ratio = 1

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

        self.data = np.array([[]])
        self.slab_class_voxels = np.empty(self.class_coords.shape,np.double)
        self._percent_contrast = None
        self._subset = np.ones(self.border_nvox,dtype=np.bool)
        self._subsamp = np.ones(self.border_nvox,dtype=np.bool)
        self._first_vol_subset = np.zeros(self.border_nvox, dtype=np.bool)
        self._last_vol_subset = np.zeros(self.border_nvox, dtype=np.bool)
        self._first_vol_subset_ssamp=np.zeros(self.border_nvox, dtype=np.bool)
        self._last_vol_subset_ssamp=np.zeros(self.border_nvox, dtype=np.bool)
        self._samples_data = np.empty((2,self.border_nvox))

    def process(self, stack, yield_raw=False):

        self.slabs = []
        self.transforms = []
        self._last_subsampling_transform = self.affine_class(np.ones(12)*5)
        
        # dicom list : slices must be provided in acquisition order

        nvox_min = 128
        slab_min_slice = 5 # arbitrary...

        # register first frame
        frame_iterator = stack.iter_frame()
        nvol, self.affine, data1 = frame_iterator.next()
        data1 = data1.astype(np.float)
        self.slice_order = stack._slice_order
        inv_slice_order = np.argsort(self.slice_order)
        self.nslices = stack.nslices
        last_reg = self.affine_class()
        if self.init_reg is not None:
            last_reg.from_matrix44(self.init_reg)
        self.rslab = ((0,0),(0,self.nslices-1))
        self.estimate_instant_motion(data1[...,np.newaxis], last_reg)
        #suppose first frame was motion free
        self.epi_mask=self.inv_resample(
            self.mask, last_reg.as_affine().dot(self.affine), data1.shape) > 0
        
        # compute values for initial registration
        self.apply_transform(
            last_reg,
            self.class_coords, self.slab_class_voxels,
            self.fmap_values, phase_dim=data1.shape[self.pe_dir])

        self.resample(data1[...,np.newaxis],
                      self._samples_data,
                      self.slab_class_voxels,
                      self.cbspline[...,0])
        # remove samples that does not have expected contrast (ie sigloss)
        # if fieldmap provided does it using the computed sigloss
        # otherwise remove the samples with inverted contrast
        if not self.fmap is None:
            grid = apply_affine(
                np.linalg.inv(self.mask.get_affine()).dot(self.fmap2world),
                np.rollaxis(
                    np.mgrid[[slice(0,n) for n in self.fmap.shape]],0,4))
            fmap_mask = map_coordinates(
                self.mask.get_data(),
                grid.reshape(-1,3).T, order=0).reshape(self.fmap.shape)
            self._samples_sigloss = compute_sigloss(
                self.fmap, self.fieldmap_reg,
                fmap_mask,
                last_reg.as_affine(), self.affine,
                self.class_coords.reshape(-1,3),
                self.echo_time, slicing_axis=self.slice_axis).reshape(2,-1)

            self._reliable_samples = np.logical_and(
                np.all(self._samples_sigloss>.8,0),
                np.all(self._samples_data>0,0))
        else:
            self._reliable_samples = np.logical_and(
                np.squeeze(np.diff(self._samples_data,1,0)) < 0,
                np.all(self._samples_data>0,0))

        self.optimizer='cg'
#        self.use_derivatives = True
#        self.optimizer_kwargs.setdefault('gtol', 1e-7)

        # reestimate first frame registration  with only reliable samples
        self.estimate_instant_motion(data1[...,np.newaxis], last_reg)
#        self.transforms.append(last_reg)
        self.resample(data1[...,np.newaxis],
                      self._samples_data,
                      self.slab_class_voxels,
                      self.cbspline[...,0])
        
        slice_axes = np.ones(3, np.bool)
        slice_axes[self.slice_axis] = False
        slice_spline = np.empty(stack._shape[:2])

        method='detect'
        if method is 'volume':
            yield nvol, [self.slabs[0]], [last_reg.as_affine().dot(self.affine)], data1
            for nvol, self.affine, data1[:] in frame_iterator:
                print 'volume %d'%nvol
                nreg = self.affine_class(last_reg.as_affine())
                self.estimate_instant_motion(data1[...,np.newaxis], nreg)
                self.transforms.append(nreg)
                last_reg = nreg
                slab = ((nvol,0),(nvol,self.nslices))
                self.slabs.append(slab)
                if yield_raw:
                    yield nvol, slab, nreg.as_affine().dot(self.affine), data1
                else:
                    yield nvol, slab, nreg.as_affine().dot(self.affine), None
                self.resample(data1[...,np.newaxis],
                              self._samples_data,
                              self.slab_class_voxels,
                              self.cbspline[...,0])

        elif method is 'detect':
            self.slabs = []            
            stack_it = stack.iter_slabs()
            slab_data = [(0,sl,self.affine,tt,data1[...,sl]) for sl,tt in\
                             zip(self.slice_order,stack._slice_trigger_times)]
            datan = data1.copy()
            
            sl_voxs_m = np.empty(self.slab_class_voxels.shape[1], np.int8)
            self.motion_detection = []
            yield_data = np.empty(stack._shape[:3])

            fr,sl,aff,tt,sl_data = stack_it.next()
            last_frame_full = True

            mot_flags = [False]*self.nslices # suppose no motion in frame 0
            mot, nmot = False, 0
            last_slab_end = -1
            stack_has_data = True
            drift = False
            while stack_has_data:
                sl_mask = self.epi_mask[...,sl]

                sl_voxs_m.fill(-1)
                for s in sl:
                    sl_voxs_m[np.logical_and(
                            np.all(np.abs(self.slab_class_voxels[...,self.slice_axis]-s)<.4,0),
                            self._reliable_samples)] = s
                n_samples = np.count_nonzero(sl_voxs_m >= 0)
                mot=False
                if n_samples > 127: # this should always be true for slabs?
                    sl_samples = np.empty((2, n_samples))
                    d0 = np.empty((2, n_samples))
                    dn = np.empty((2, n_samples))
                    ofst = 0
                    for si,s in enumerate(sl):
                        mm = sl_voxs_m==s
                        cnt = np.count_nonzero(mm)
                        crds = self.slab_class_voxels[:,mm][...,slice_axes].T

                        slice_spline[:] = _cspline_transform(data1[...,s])
                        _cspline_sample2d(d0[:,ofst:ofst+cnt],
                                          slice_spline,*crds)
                        slice_spline[:] = _cspline_transform(datan[...,s])
                        _cspline_sample2d(dn[:,ofst:ofst+cnt],
                                          slice_spline,*crds)
                        slice_spline[:] = _cspline_transform(sl_data[...,si])
                        _cspline_sample2d(sl_samples[:,ofst:ofst+cnt],
                                          slice_spline, *crds)
                        ofst += cnt

                    _,_,corr,pval,_=scipy.stats.linregress(sl_samples[1],dn[1])
                    mot = corr < self.detection_threshold
                    if mot:
                        print 'motion (%d,%s) %f (p=%f)'%(fr,str(sl),corr,pval)

                    _,_,corr,pval,_=scipy.stats.linregress(sl_samples[1],d0[1])
                    drift = drift or (corr < self.detection_threshold)
                    if drift:
                        print 'drift (%d,%s) %f (p=%f)'%(fr,str(sl),corr,pval)
                    
                    self.motion_detection.append((fr,sl,(d0,dn,sl_samples),))

                for si,s in enumerate(sl):
                    datan[...,s] = sl_data[...,si]
                    slab_data.append((fr,s,aff,tt,sl_data[...,si]))
                    mot_flags.append(mot)

                reg_sl1 = (mot_flags[last_slab_end+1:-1]+[False]).index(False) + last_slab_end + 1
                reg_sln = (mot_flags[reg_sl1:-1]+[True]).index(True) + reg_sl1
                nmot = sum(mot_flags[reg_sl1:])
                #if nmot > self.nslices:
                #    print 'register motion frame'
                #    reg_sln = len(mot_flags)-1
                #    reg_sl1 = reg_sln-(mot_flags[::-1]+[False]).index(False)
                
                print last_slab_end, reg_sl1, reg_sln, nmot

                try:
                    fr,sl,aff,tt,sl_data = stack_it.next()
                except StopIteration:
                    stack_has_data = False

                # if motion detected and motion finished or
                # motion slices fills one frame or stack is empty
                if ((not mot and (nmot>0 or ( drift and slab_data[-1][1]==self.slice_order[-1]))) or not stack_has_data):
                    # or nmot>self.nslices
                    fr0, frn = slab_data[reg_sl1][0], slab_data[reg_sln][0]
                    sl0 = inv_slice_order[slab_data[reg_sl1][1]]
                    sln = inv_slice_order[slab_data[reg_sln][1]]
                    rslab = ((fr0,sl0),(frn,sln))
                    nreg = self._register_slab(
                        rslab, slab_data[reg_sl1:reg_sln+1], last_reg)
                    
                    self.transforms.append(nreg)

                    self.epi_mask=self.inv_resample(
                        self.mask, nreg.as_affine().dot(self.affine),
                        data1.shape) > 0

                    self.apply_transform(
                        nreg, self.class_coords, self.slab_class_voxels,
                        self.fmap_values, phase_dim=stack._shape[self.pe_dir])

                    if not stack_has_data:
                        frn = slab_data[-1][0]
                        sln = self.nslices-1
                    if len(self.slabs)>0:
                        if fr0 <= self.slabs[-1][1][0] and \
                                sl0 <= self.slabs[-1][1][1]:
                            raise RuntimeError
                    slab = ((fr0,sl0),(frn,sln))
                    self.slabs.append(slab)
                    first_frame_full = sl0 == 0
                    last_frame_full = sln == self.nslices-1
                    multiple_frames = fr0 < frn
                    regs = []
                    slabs = []

                    n_yielded = 0
                    yield_data.fill(np.nan)# to be removed
                    for sd in slab_data:
                        fr2,sl2,aff2,tt2,slice_data2 = sd
                        if fr2 > fr-2:
                            data1[...,sl2] = slice_data2
                        print fr2, sl2, last_frame_full
                        if fr2 < frn + last_frame_full:
                            n_yielded += 1
                            yield_data[...,sl2] = slice_data2
                            if sl2 == self.slice_order[-1]:
                                if np.count_nonzero(np.isnan(yield_data))>0:
                                    raise RuntimeError # to be removed
                                slabs = [slb for slb in self.slabs if \
                                             slb[0][0]<=fr2 and slb[1][0]>=fr2]
                                regs = [reg.as_affine().dot(self.affine)\
                                            for slb,reg in \
                                            zip(self.slabs,self.transforms) if\
                                            slb[0][0]<=fr2 and slb[1][0]>=fr2]
                                if len(slabs) == 0:
                                    slabs = [self.slabs[-1]]
                                    regs = [self.transforms[-1].as_affine().dot(self.affine)]
                                yield fr2, slabs, regs, yield_data
                                yield_data.fill(np.nan) # to be removed

                    # TODO : handle movement frames 
                    slab_data = slab_data[n_yielded:]
                    mot_flags = mot_flags[n_yielded:]
                    last_slab_end = max(0,reg_sln - n_yielded)
                    last_reg = nreg
                    drift = False


    # register a data slab
    def _register_slab(self,slab,slab_data,transform):
        print 'register slab', slab
        fr1 = slab_data[0][0]
        nframes = slab[1][0] - fr1 + 1
        data = np.zeros(slab_data[0][-1].shape[:2] + (self.nslices,nframes))
#        data.fill(np.nan) # just to check, to be removed?
        for fr,sl,aff,tt,slice_data in slab_data:
            data[...,sl,fr-fr1] = slice_data
#        pos = np.where(self.slice_order==slab_data[0][1])[0][0]
#        self.slice_order[:pos,np.newaxis,]
#        for si in range():
#            data[...,si,0] =             
#        data[...,:slab_data[0][1],0] = data[...,slab_data[0][1],0,np.newaxis]
#        data[...,slab_data[-1][1]:,-1] = data[...,slab_data[-1][1],-1,
#                                                  np.newaxis]
        # fill missing slice with following or previous slice
#        count1=np.squeeze(np.apply_over_axes(np.sum, np.isnan(data),[0,1]))
        if nframes > 1:
            n_missl = slab[0][1]
            if nframes==2:
                n_missl = min(slab[1][1]+1, n_missl)
                for si in range(n_missl, slab[0][1]):
                    so = self.slice_order[si]
                    data[...,so,0] = data[...,so,1]
            for si in range(n_missl):
                so = self.slice_order[si]
                data[...,so,0] = data[...,so,1]

            idx = np.where(self.slice_order==sl)[0]
            count2 = np.squeeze(np.apply_over_axes(
                    np.sum, np.isnan(data),[0,1]))
            idx2 = idx
            if nframes == 2:
                idx2 = max(slab[0][1], idx2)
            for si in range(idx2, self.nslices):
                so = self.slice_order[si]
                data[...,so,-1] = data[...,so,-2]

#        if np.count_nonzero(np.isnan(data)) > 0:
#            raise RuntimeError
        self.rslab = slab
                
        reg = self.affine_class(transform.as_affine())
        self.estimate_instant_motion(data, reg)
        del data
        return reg

    # estimate motion on a data volume
    def estimate_instant_motion(self, data, transform):
        
        if hasattr(self,'cbspline') and \
                data.shape[-1] > self.cbspline.shape[-1]:
            del self.cbspline
        if not hasattr(self,'cbspline'):
            self.cbspline = np.empty(data.shape,np.double)
        for t in range(data.shape[-1]):
            self.cbspline[:, :, :, t] =\
                _cspline_transform(data[:, :, :, t])

        self.sample(data, transform, force_recompute_subset=True)

        def f(pc):
            self._init_energy(pc, data, transform)
            nrgy = self._energy()
            print 'f %f : %f %f %f %f %f %f'%tuple([nrgy] + pc.tolist())
            return nrgy

        def fprime(pc):
            self._init_energy(pc, data, transform)
            return self._energy_gradient()

        def fhess(pc):
            print 'fhess'
            self._init_energy(pc, data, transform)
            return self._energy_hessian()

        self._pc = None
        fmin, args, kwargs =\
            configure_optimizer(self.optimizer,
#                                fprime=fprime,
#                                fhess=fhess,
                                **self.optimizer_kwargs)

        pc = fmin(f, transform.param, *args, **kwargs)
        return pc

    def _init_energy(self, pc, data, transform):
        if pc is self._pc:
            return
        transform.param = pc
        self._pc = pc
        self.sample(data, transform)

        if self.use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            if hasattr(self,'_aux') and self._aux.shape != self.data.shape:
                del self._aux
            if not hasattr(self,'_aux'):
                self._aux = np.empty(self.data.shape)
            self._aux[:] = self.data[:]
            nrgy = self._energy()
            print 'energy %f'% nrgy
            basis = np.eye(pc.size)
            A=np.zeros((pc.size,pc.size))
            t2 = self.affine_class()
            for j in range(pc.size):
                for k in range(j,pc.size):
                    t2.param = pc + self.stepsize*(basis[j]+basis[k])
                    self.sample(data, t2)
                    A[j,k]=self._energy()
                t2.param = pc + self.stepsize*basis[j]
                self.sample(data, t2)
                A[j,j] = self._energy() 
            
            self.data[:] = self._aux[:]
            # pre-compute gradient and hessian of numerator and
            # denominator
            tril = np.tri(pc.size, k=-1,dtype=np.bool)
            self._dV = (A.diagonal() - nrgy)/self.stepsize
            self._H = ((A-A.diagonal()-A.diagonal()[:,np.newaxis]+nrgy)*tril.T+
                       np.diag(((A.diagonal()-A2)/2-(A.diagonal()-nrgy))) )* 2.0/self.stepsize
            self._H[tril] = self._H.T[tril]


    def _energy_gradient(self):
        print 'gradient', self._dV
        return self._dV

    def _energy_hessian(self):
        print 'hessian',self._H
        return self._H

    def apply_transform(self, transform, in_coords, out_coords,
                        fmap_values=None, subset=slice(None), phase_dim=64):
        ref2fmri = np.linalg.inv(transform.as_affine().dot(self.affine))
        #apply current slab transform
        out_coords[...,subset,:]=apply_affine(ref2fmri,in_coords[...,subset,:])
        #add shift in phase encoding direction
        if fmap_values != None:
            out_coords[...,subset,self.pe_dir]+=fmap_values[...,subset]*phase_dim

    def sample(self, data, transform, force_recompute_subset=False):
        """
        sampling points interpolation in EPI data
        """
        sa = self.slice_axis
        nvols = data.shape[-1]
        ref2fmri = np.linalg.inv(transform.as_affine().dot(self.affine))
        slab = self.rslab
        
        # if change of test points z is above threshold recompute subset
        test_points = np.array([[0,0,0],[0,0,self.nslices]])
        recompute_subset = np.abs(
            self._last_subsampling_transform.apply(test_points) -
            transform.apply(test_points))[:,sa].max() > 0.1
        
        if recompute_subset or force_recompute_subset:
            self.apply_transform(
                transform,
                self.class_coords,self.slab_class_voxels,
                self.fmap_values, phase_dim=data.shape[self.pe_dir])

            self._last_subsampling_transform = transform.copy()
            # adapt subsampling to keep regular amount of points in each slice
            zs = self.slab_class_voxels[...,sa].sum(0)/2.
            #samples_slice_hist = np.histogram(zs,np.arange(self.nslices+1)-self.st_ratio)

            np_and_ow = lambda x,y: np.logical_and(x,y,y)

            # this computation is wrong 
            self._subsamp[:] = False
            
            
            if hasattr(self,'_reliable_samples'):
                step = np.floor(np.count_nonzero(self._reliable_samples)/
                                float(self.nsamples_per_slab))
                self._subsamp[np.where(self._reliable_samples)[0][::step]] = True
            else:
                step = np.floor(self._subsamp.shape[0]/
                                float(self.nsamples_per_slab))
                self._subsamp[::step] = True

            excl_slices = np.ones(self.nslices, dtype=np.bool)
            excl_slices[np.arange(slab[0][1],self.nslices)] = False
            self._first_vol_subset[:] = np.all(
                np.abs(zs[:,np.newaxis]-
                       self.slice_order[excl_slices][np.newaxis]) > 
                    self.st_ratio, 1)
            excl_slices.fill(True)
            excl_slices[np.arange(0,slab[1][1])] = False
            self._last_vol_subset[:] = np.all(
                np.abs(zs[:,np.newaxis]-
                       self.slice_order[excl_slices][np.newaxis]) > 
                self.st_ratio,1)
            print np.count_nonzero(self._first_vol_subset), \
                np.count_nonzero(self._last_vol_subset)

            if data.shape[-1] == 1:
                np_and_ow(self._last_vol_subset, self._first_vol_subset)
                np.logical_and(self._first_vol_subset, self._subsamp,
                               self._first_vol_subset_ssamp)
                self._last_vol_subset_ssamp.fill(False)
                self._last_vol_subset.fill(False)
                self._subset[:] = self._first_vol_subset[:]
                np_and_ow(self._subset, self._subsamp)
            else:
                np.logical_and(self._last_vol_subset,self._subsamp,
                               self._last_vol_subset_ssamp)
                if nvols > 2:
                    self._subset.fill(True)
                else:
                    np_and_ow(self._first_vol_subset, self._subsamp)
                    np_and_ow(self._last_vol_subset, self._subsamp)
                
            self._tmp_nsamples = self._subsamp.sum()
            print 'new subset %d samples'%self._tmp_nsamples
        else:
            self.apply_transform(transform,
                                 self.class_coords,self.slab_class_voxels,
                                 self.fmap_values,self._subsamp,
                                 phase_dim = data.shape[self.pe_dir])

        nsamples_1vol = np.count_nonzero(self._first_vol_subset_ssamp)
        n_samples = np.count_nonzero(self._subsamp)
        n_samples_lvol = 0
        if nvols > 1:
            n_samples_lvol = np.count_nonzero(self._last_vol_subset_ssamp)
        n_samples_total = nsamples_1vol + n_samples_lvol +\
            n_samples * max(nvols-2, 0)

        self.skip_slab=False
        if n_samples_total < self.min_sample_number:
            print 'skipping slab, only %d samples'%n_samples_total
            self.skip_slab = True
            return
            

        # if subsampling changes
        if self.data.shape[1] != n_samples_total:
            del self.data
            self.data = np.empty((2,n_samples_total))
            del self._percent_contrast
            self._percent_contrast = np.empty(n_samples_total)

        # resample per volume, split not optimal for many volumes ???
        self.resample(
            data[...,0],
            self.data[:,:nsamples_1vol],
            self.slab_class_voxels[:,self._first_vol_subset_ssamp],
            self.cbspline[...,0])
        for i in range(1, nvols - (nvols > 1) ):
            seek = nsamples_1vol + n_samples * (i -1)
            self.resample(
                data[...,i],
                self.data[:,seek:seek+n_samples],
                self.slab_class_voxels[:,self._subsamp],
                self.cbspline[...,i])
        if n_samples_lvol > 0:
            self.resample(
                data[...,nvols-1],
                self.data[:,-n_samples_lvol:None],
                self.slab_class_voxels[:,self._last_vol_subset_ssamp],
                self.cbspline[...,nvols-1])

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

    def _energy(self):
        sm = self.data.sum(0)
        self._percent_contrast[:] = 200*np.diff(self.data,1,0)/sm
        self._percent_contrast[np.abs(sm)<1e-6] = 0
        bbr_offset = 0 # TODO add as an option, and add weighting
        bbr_slope = 1
        cost = 1.0+np.tanh(bbr_slope*self._percent_contrast-bbr_offset).mean()
        # if there is a previous frame registered
        if len(self.transforms)>0:
            # compute regularization 
            # add it to the cost
#            self.data[:np.count_nonzero(self._first_vol_subset_ssamp)]
#            regl = (self.data[1]-self._samples_data[1,self._first_vol_subset_ssamp]).std()/(self.data[1].std()*self._samples_data[1,self._first_vol_subset_ssamp].std())
#            print regl, np.tanh(100*regl)
#            cost += np.tanh(100*regl)
            # REGULARIZATIOn
            # to be tested for sub volume slabs
            # penalize small motion
            if self._motion_regularization>0:
                param_diff = np.abs(self._pc-self.transforms[-1].param)
                cost += np.tanh(param_diff).sum()*self._motion_regularization
        return cost
    

    def explore_cost(self, data, transform, values,
                     bbr_slope=.5, bbr_offset=0, factor=200):
        costs = np.empty((len(transform.param),len(values)))
        self.sample(data, transform, force_recompute_subset=True)
        t2 = transform.copy()
        for p in range(len(transform.param)):
            for idx,delta in enumerate(values):
                params = transform.param.copy()
                params[p] += delta
                t2.param = params
                n.sample(data, t2)
                sm = n.data.sum(0)
                n._percent_contrast[:] = factor*np.diff(n.data,1,0)/sm
                n._percent_contrast[np.abs(sm)<1e-6] = 0
                costs[p,idx] = 1.0+np.tanh(bbr_slope*n._percent_contrast-bbr_offset).mean()
                print params, costs[p,idx]
        return costs
                


class EPIOnlineRealignFilter(EPIOnlineResample):
    
    def correct(self, realigned, pvmaps, poly_order = 2, do_n4=False):
        
        float_mask = nb.Nifti1Image(
            self.mask.get_data().astype(np.float32),
            self.mask.get_affine())
        
#        do_n4 = True


        ext_mask = self.mask.get_data()>0
        ext_mask[pvmaps.get_data()[...,:2].sum(-1)>0.1] = True
        float_mask = nb.Nifti1Image(
            ext_mask.astype(np.float32),
            self.mask.get_affine())        
        from scipy.interpolate import SmoothBivariateSpline
        spline_order = (3,3)
        min_vox = np.prod(np.asarray(spline_order)+1)
        cdata = None
        for slab, reg, data in realigned:
            if cdata is None:
                pts = np.mgrid[:data.shape[0],:data.shape[1]]
                cdata = np.zeros(data.shape)
                epi_pvf = np.ones(data.shape+(pvmaps.shape[-1]+1,))
                epi_mask = np.zeros(data.shape, dtype=np.bool)
                epi_mask2 = np.zeros(data.shape, dtype=np.bool)
            epi_mask[:] = self.inv_resample(self.mask, reg, data.shape, 0)
            epi_mask2[:] = self.inv_resample(float_mask, reg, data.shape, 0)
            epi_pvf[...,:-1] = self.inv_resample(
                pvmaps, reg, data.shape, -1,
                mask = self.mask.get_data()>0)

            regs = epi_pvf[epi_mask,1:]
#                regs[:-1] = (regs[:-1] - regs[:-1].mean(0))/regs[:-1].std(0)
            regs_pinv = np.linalg.pinv(regs)
            betas = regs_pinv.dot(data[epi_mask])
            data -= epi_pvf[...,1:-1].dot(betas[:-1])

            for sl in range(data.shape[self.slice_axis]):
                sl_mask = epi_mask[...,sl].copy()
                sl_mask2 = epi_mask2[...,sl]

                logdata = np.log(data[...,sl])
                sl_mask[np.isnan(logdata)] = False
                sl_mask[np.isinf(logdata)] = False
                if np.count_nonzero(sl_mask) < min_vox:
                    continue

                """
                regs = epi_pvf[sl_mask,sl,1:]
#                regs[:-1] = (regs[:-1] - regs[:-1].mean(0))/regs[:-1].std(0)
                regs_pinv = np.linalg.pinv(regs)
                betas = regs_pinv.dot(logdata[sl_mask])
                logdata -= epi_pvf[...,sl,1:].dot(betas)
#                print betas
#                logdata = np.log(cdata[...,sl])
"""

                aw = np.argwhere(sl_mask2)
                xmin,ymin = aw.min(0)-5
                xmax,ymax = aw.max(0)+5
                bbox = [xmin,xmax,ymin,ymax]
                spline2d = SmoothBivariateSpline(
                    pts[0,sl_mask],pts[1,sl_mask],
                    logdata[sl_mask],
                    w=epi_pvf[sl_mask,sl,1]+1e-1,
                    bbox = bbox,
                    kx=spline_order[0],
                    ky=spline_order[1])
                fit = spline2d(pts[0,:,0], pts[1,0])
#                cdata[...,sl] = np.exp(fit)
                cdata[sl_mask2,sl] = np.exp(logdata[sl_mask2]-fit[sl_mask2])
                cdata[np.isinf(cdata[...,sl]),sl] = 0
                cdata[cdata[...,sl]>10,sl] = 0
                cdata[-sl_mask2,sl] = 0
#                cdata[cdata[...,sl]>10,sl] = 0
#                regs = epi_pvf[sl_mask2,sl,1:]
#                regs[:] = (regs - regs.mean(0))/regs.std(0)
#                regs_pinv = np.linalg.pinv(regs)
#                betas = regs_pinv.dot(cdata[sl_mask2,sl])
#                cdata[sl_mask2,sl] -= epi_pvf[sl_mask2,sl,1:].dot(betas)
            if (np.count_nonzero(np.isinf(cdata)) > 0 or 
                np.count_nonzero(np.isnan(cdata)) > 0): 
                raise RuntimeError('inf or nan')
            yield slab, reg, cdata
        return


        """
        import SimpleITK as sitk
        n4filt = sitk.N4BiasFieldCorrectionImageFilter()
        cordata2 = None
        pcdata = None 
        init=False
        for slab, reg, data in realigned:

            if cordata2 is None:
                cordata2 = np.zeros(data.shape)
                pcdata = np.zeros(data.shape)
                epi_pvf = np.zeros(data.shape+pvmaps.shape[-1:])
                epi_ppvf = np.zeros(epi_pvf.shape)
                epi_mask = np.zeros(data.shape, dtype=np.bool)
            epi_mask[:] = self.inv_resample(float_mask, reg, data.shape, 1)>.5
            
            if do_n4:
                itkmask = sitk.GetImageFromArray(epi_mask.astype(np.int64))
                itkdata = sitk.GetImageFromArray(data.astype(np.float))
                try:
                    cordata = n4filt.Execute(itkdata, itkmask)
                except RuntimeError:
                    print 'boum crash why??'
                    cordata = n4filt.Execute(itkdata, itkmask)
                cordata2[:] = sitk.GetArrayFromImage(cordata)
                del cordata, itkdata, itkmask
            else:
                cordata2[:] = data   


            #### regress pvmaps
            epi_pvf[:] = self.inv_resample(
                pvmaps, reg, data.shape, -1,
                mask = self.mask.get_data()>0)

            if False:
                regs_pinv = np.linalg.pinv(regs[epi_mask])
                betas = regs_pinv.dot(np.log(cordata2[epi_mask]))
                cordata2[epi_mask] /= np.exp(regs[epi_mask].dot(betas))
                cordata2[np.logical_not(epi_mask)] = 0
            if not init:
                init=True
                pcdata[:] = cordata2
                epi_ppvf[:] = epi_pvf
            else:
                ddata = cordata2 - pcdata
                regs = np.concatenate(
                    [epi_pvf-epi_ppvf,epi_mask[...,np.newaxis]],-1)
                for sl in range(data.shape[self.slice_axis]):
                    if np.count_nonzero(epi_mask[...,sl]) > 0:
                        sl_mask = epi_mask[...,sl].copy()
                        sl_mask[np.isinf(ddata[...,sl])] = False
                        regs_pinv = np.linalg.pinv(regs[sl_mask,sl,:])
                        betas = regs_pinv.dot(ddata[sl_mask,sl])
                        print betas
                        cordata2[sl_mask,sl] = pcdata[sl_mask,sl] + (
                            ddata[sl_mask,sl]-regs[sl_mask,sl,:].dot(betas))
#                        cordata2[np.isinf(cordata2[...,sl]),sl] = 0
                pcdata[:] = cordata2
                epi_ppvf[:] = epi_pvf
            yield slab, reg, cordata2
        return
        """
    
    #### OLD OPTIONS MIGHT HAVE TO COME BACK TO THIS ######
        import itertools
        ext_mask = self.mask.get_data()>0
        ext_mask[pvmaps.get_data()[...,:2].sum(-1)>0] = True
        float_mask = nb.Nifti1Image(
            ext_mask.astype(np.float32),
            self.mask.get_affine())
        init_regs = False        
        for slab, reg, data in realigned:
            shape = data.shape 
            if not init_regs:
                epi_pvf = np.empty(shape+(pvmaps.shape[-1],))
                epi_mask = np.empty(shape, dtype=np.bool)
                epi_sigint = np.empty(shape)
                cdata = np.zeros(shape)
                ij = itertools.product(range(poly_order+1),range(poly_order+1))
                x = ((np.arange(0,shape[0])-shape[0]/2.+.5)/shape[0]*2)[:,np.newaxis]
                y = ((np.arange(0,shape[1])-shape[1]/2.+.5)/shape[1]*2)[np.newaxis]
                slice_regs = np.empty(shape[:2]+((poly_order+1)**2,))
                for k, (i,j) in enumerate(ij):
                    slice_regs[...,k] = x**i * y**j

            # should try to use caching for inv resample coords computation
            print 'compute distorted mask'
            epi_mask[:] = self.inv_resample(float_mask, reg, data.shape, 1) > .1
            print 'compute partial volume maps'
            epi_pvf[:] = self.inv_resample(pvmaps, reg, data.shape, -1,
                                           mask = ext_mask)
            epi_pvf[epi_pvf.sum(-1)==0,-1] = 1 # only useful if fitting whole image
            ############## regressing PV + 2d poly from each slice #######    
            cdata.fill(0)
            for sl in range(data.shape[self.slice_axis]):
                if np.count_nonzero(epi_mask[...,sl]) > 0:
                    sl_mask = epi_mask[...,sl].copy()
                    regs = np.dstack([slice_regs,epi_pvf[...,sl,:]])
                    #regs[np.isinf(regs)] = 0
                    """                    
                    regs = np.dstack([
                            slice_regs, 
                            1-np.exp(-30./(np.array([1331,832,4000])*\
                                               epi_pvf[...,sl,:])),
                            np.exp(-30./(np.array([85,77,2200])*\
                                             epi_pvf[...,sl,:])) ])"""
                    #regs = np.dstack([slice_regs,np.exp(epi_pvf[...,sl,1:])])
                    #regs[np.isinf(regs)] = 0
                    regs = np.dstack([
                       slice_regs, 
                       epi_pvf[...,sl,:]*(1-np.exp(-30./np.array([1331,832,4000]))),
                       epi_pvf[...,sl,:]*np.exp(-30./np.array([85,77,2200]))])
                    logdata = np.log(data[...,sl])
                    sl_mask[np.isinf(logdata)] = False
                    regs_pinv = np.linalg.pinv(regs[sl_mask])
                    betas = regs_pinv.dot(logdata[sl_mask])
                    #print betas
                    cdata[...,sl] = 0
#                    cdata[...,sl] = np.exp(
#                        logdata - regs.dot(betas))
#                    cdata[sl_mask,sl] = np.exp(
#                        logdata[sl_mask] - regs[sl_mask].dot(betas))
                    cdata[...,sl] = np.exp(
                        logdata - regs.dot(betas))
                    regs = epi_pvf[...,sl,1:]
#                    cdata[np.isinf(cdata[...,sl]),sl] = 0
                    del regs, regs_pinv, logdata, betas
            yield slab, reg, cdata
            
        del epi_pvf, float_mask

        """
        self._white_means = []
        for slab, reg, data in realigned:
            cordata = np.empty(data.shape)
            
            epi_mask = self.inv_resample(self.mask, reg, data.shape, order=0)
            epi_white = self.inv_resample(white_mask, reg, data.shape, order=1)>=1
            
            print '%d white, %d mask'% \
                (np.count_nonzero(epi_white),np.count_nonzero(epi_mask))
            
            vol_white_means = np.squeeze(np.divide(
                np.apply_over_axes(np.sum,epi_white*data,[0,1]),
                np.apply_over_axes(np.sum,epi_white,[0,1])))
            self._white_means.append(vol_white_means)
            cordata[:] = data
            if len(self._white_means)>0:
                white_diff = vol_white_means-self._white_means[0] 
                for sl in xrange(data.shape[-1]):
                    if not np.isnan(white_diff[sl]):
                        cordata[...,sl] = data[...,sl]-white_diff[sl]
            yield slab, reg, cordata
            """

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
        self.nslices = self.nii.shape[2]
        self._slice_order = np.arange(self.nslices)
        
    def iter_frame(self, data=True):
        data = self.nii.get_data()

        for t in range(data.shape[3]):
            yield t, self.nii.get_affine(), data[:,:,:,t]
        del data

