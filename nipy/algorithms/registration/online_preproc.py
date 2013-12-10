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
from .slice_motion import surface_to_samples


# Module globals
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
        self.pe_dir = abs(phase_encoding_dir)
        self.repetition_time = repetition_time
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
        if self.fmap != None:
            interp_coords = apply_affine(self.world2fmap, coords)
            self._resample_fmap_values = self.fmap_scale * \
                map_coordinates(self.fmap.get_data(),
                                interp_coords.reshape(-1,3).T,
                                order=1).reshape(interp_coords.shape[:-1])


    def resample_coords(self, data, affines, coords, out):

        self._precompute_sample_fmap(coords,data.shape)
        interp_coords = np.empty(coords.shape)
            
        tmp_coords = np.empty(coords.shape)
        subset = np.zeros(coords.shape[:-1], dtype=np.bool)
        tmp = np.empty(coords.shape[:-1])
        if len(affines) == 1: #easy, one transform per volume
            wld2epi = np.linalg.inv(affines[0][1])
            interp_coords[:] = apply_affine(wld2epi, coords)
            if self._resample_fmap_values != None:
                interp_coords[...,self.pe_dir] += self._resample_fmap_values
        else: # we have to solve which transform we sample with
            for sg,trans in affines:
                wld2epi = np.linalg.inv(trans)
                tmp_coords[:] = apply_affine(wld2epi, coords)
                if self.fmap != None:
                    tmp_coords[...,self.pe_dir] += fmap_values
                subset.fill(False)
                    
                if sg[0][0]==t and sg[1][0]==t:
                    times = np.arange(sg[0][1],sg[1][1])
                elif sg[0][0]==t:
                    times = np.arange(sg[0][1], self.nslices)
                elif sg[1][0]==t:
                    times = np.arange(sg[0][1], self.nslices)
                else:
                    times = np.arange(0, self.nslices)
                subset = np.any(
                    np.abs(tmp_coords[...,self.slice_axis,np.newaxis]-
                           self.slice_order[times][np.newaxis]) \
                        < self.st_ratio+.1, -1)
            interp_coords[subset] = tmp_coords[subset]
        self.resample(data, out,interp_coords)
        del interp_coords, tmp_coords, subset

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
                 
                 motion_regularization = 1e-2,
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
        self._last_subsampling_transform = affine_class(np.ones(12)*5)
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
        
        # dicom list : slices must be provided in acquisition order

        slab_data = []
        mot_flags = []
        nvox_min = 128
        slab_min_slice = 5 # arbitrary...

        self._last_subsampling_transform = self.affine_class(np.ones(12)*5)

        # register first frame
        _, self.affine, data1 = stack.iter_frame().next()
        self.slice_order = stack._slice_order
        self.nslices = stack.nslices
        last_reg = self.affine_class()
        if self.init_reg is not None:
            last_reg.from_matrix44(self.init_reg)
        self.slabs.append(((0,0),(0,self.nslices-1)))

        self.estimate_instant_motion(data1[...,np.newaxis], last_reg)
        #suppose first frame was motion free
        self.epi_mask=self.inv_resample(
            self.mask, last_reg.as_affine(), data1.shape) > 0
        
        # compute values for initial registration
        self.apply_transform(
            last_reg,
            self.class_coords, self.slab_class_voxels,
            self.fmap_values, phase_dim=data1.shape[self.pe_dir])

        self.resample(data1[...,np.newaxis],
                      self._samples_data,
                      self.slab_class_voxels)
        # remove samples that does not have expected contrast (ie sigloss)
        self._reliable_samples = np.logical_and(
            np.squeeze(np.diff(self._samples_data,1,0)) < 0,
            np.all(self._samples_data>0,0))

        # reestimate first frame registration  with only reliable samples
        self.estimate_instant_motion(data1[...,np.newaxis], last_reg)
        self.transforms.append(last_reg)
        self.resample(data1[...,np.newaxis],
                      self._samples_data,
                      self.slab_class_voxels)
        yield self.slabs[0], last_reg.as_affine().dot(self.affine), data1
        
        slice_voxs_m = np.empty(self.slab_class_voxels.shape[1], np.bool)
        slice_axes = np.ones(3, np.bool)
        slice_axes[self.slice_axis] = False
        slice_spline = np.empty(data1.shape[:2])

        self.mot_ests=[]
        
        method='volume'
        if method is 'volume':
            for nvol, self.affine, data1 in stack.iter_frame():
                nreg = last_reg.copy()
                self.estimate_instant_motion(data1[...,np.newaxis], nreg)
                self.transforms.append(nreg)
                last_reg = nreg
                slab = ((nvol,0),(nvol,self.nslices))
                if yield_raw:
                    yield slab, nreg.as_affine().dot(self.affine), data1
                else:
                    yield slab, nreg.as_affine().dot(self.affine), None
                self.resample(data1[...,np.newaxis],
                              self._samples_data,
                              self.slab_class_voxels)

        elif method is 'detect':
            for fr,sl,aff,tt,slice_data in stack.iter_slices():
            
                sl_mask = self.epi_mask[...,sl]

                slice_voxs_m[:] = np.logical_and(
                    np.all(np.abs(self.slab_class_voxels[...,self.slice_axis]-sl)<.2,0),
                    self._reliable_samples)
                n_samples = np.count_nonzero(slice_voxs_m)
                if n_samples < 100:
                    data1[...,sl] = slice_data
                    slab_data.append((fr,sl,aff,tt,slice_data))
                    continue
                slice_spline[:] = _cspline_transform(slice_data)
                slice_samples = np.empty((2, n_samples))
                _cspline_sample2d(
                    slice_samples, slice_data.astype(np.float),
                    *self.slab_class_voxels[:,slice_voxs_m][...,slice_axes].T)
                d0 = self._samples_data[:,slice_voxs_m]
                class_nc = np.divide(
                    (slice_samples-self._samples_data[:,slice_voxs_m]).std(1),
                    slice_samples.std(1)*
                    self._samples_data[:,slice_voxs_m].std(1))
                
                print fr, sl, 'class_nc', class_nc, slice_samples.shape
                mot = class_nc[1] > .02

                mot_flags.append(mot)

                # motion detected and there are sufficient slices in slab
                if mot and sum(mot_flags)>1 and len(mot_flags)>=stack.nslices:
                    slab = (
                        (slab_data[0][0],
                         np.where(self.slice_order==slab_data[0][1])[0][0]),
                        (slab_data[-1][0],
                         np.where(self.slice_order==slab_data[-1][1])[0][0]))
                    self.slabs.append(slab)
                    nreg = self._register_slab(
                        self.slabs[-1],
                        slab_data)
#                    [s for s,m in zip(slab_data,mot_flags) if not m])
                    self.transforms.append(nreg)
                    slab_data = []
                    mot_flags = []
                    last_reg = nreg
                    self.epi_mask=self.inv_resample(
                        self.mask, nreg.as_affine(), data1.shape) > 0

                    self.apply_transform(
                        self.transforms[-1],
                        self.class_coords, self.slab_class_voxels,
                        self.fmap_values, phase_dim=data1.shape[self.pe_dir])
                    yield slab, nreg.as_affine().dot(self.affine), data1
                slab_data.append((fr,sl,aff,tt,slice_data))
                data1[...,sl] = slice_data
        if  len(slab_data) > 0:
            slab = ((slab_data[0][0],
                     np.where(self.slice_order==slab_data[0][1])[0][0]),
                    (slab_data[-1][0],
                     np.where(self.slice_order==slab_data[-1][1])[0][0]))
            self.slabs.append(slab)
            nreg = self._register_slab(self.slabs[-1],slab_data)
            self.transforms.append(nreg)
            yield slab, nreg.as_affine().dot(self.affine), data1

    # register a data slab
    def _register_slab(self,slab,slab_data):
        nframes = slab[1][0] - slab[0][0] + 1
        data = np.empty(slab_data[0][-1].shape + \
                            (self.slice_order.max()+1,nframes))
        data.fill(np.nan) # just to check, to be removed?
        fr1 = slab_data[0][0]
        for fr,sl,aff,tt,slice_data in slab_data:
            data[...,sl,fr-fr1] = slice_data
        # fill missing slice with following or previous slice
        count1=np.squeeze(np.apply_over_axes(np.sum, np.isnan(data),[0,1]))
        if nframes > 1:
            n_missl = slab[0][1]
            if nframes==2:
                n_missl = min(slab[1][1]+1, n_missl)
            for si in range(n_missl):
                so = self.slice_order[si]
                data[...,so,0] = data[...,so,1]
            pos = np.where(self.slice_order==sl)[0]
            count2=np.squeeze(np.apply_over_axes(np.sum, np.isnan(data),[0,1]))
            if nframes == 2:
                pos = max(slab[0][1], pos)
            for si in range(pos, self.nslices):
                so = self.slice_order[si]
                data[...,so,-1] = data[...,so,-2]
        count3=np.squeeze(np.apply_over_axes(np.sum, np.isnan(data),[0,1]))
        if np.count_nonzero(np.isnan(data)) > 0:
            crashboumbang()

        reg = self.transforms[-1].copy()
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
        self._last_subsampling_transform = self.affine_class(np.ones(12)*5)

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
                                fprime=fprime,
                                fhess=fhess,
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
            
            self.data = self._aux
            # pre-compute gradient and hessian of numerator and
            # denominator
            tril = np.tri(pc.size, k=-1,dtype=np.bool)
            self._dV = (A.diagonal() - nrgy)/self.stepsize*2.0
            self._H = ((A-A.diagonal()-A.diagonal()[:,np.newaxis]+nrgy)*tril.T+
                       np.diag(((A.diagonal()-A2)/2-(A.diagonal()-nrgy))) )* 2.0/self.stepsize

            self._H[tril] = self._H.T[tril]



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
        slab = self.slabs[-1]
        
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
            samples_slice_hist = np.histogram(zs,np.arange(self.nslices+1)-self.st_ratio)

            np_and_ow = lambda x,y: np.logical_and(x,y,y)

            # this computation is wrong 
            self._subsamp[:] = False
            
            
            if hasattr(self,'_reliable_samples'):
                self._subsamp[:] = self._reliable_samples
            else:
                step = np.floor(self._subsamp.shape[0]//
                                float(self.nsamples_per_slab))
                self._subsamp[::step] = True
            self._first_vol_subset[:] = np.any(
                np.abs(zs[:,np.newaxis]-self.slice_order[
                        np.arange(slab[0][1],self.nslices)][np.newaxis]
                       ) < self.st_ratio, 1) 
            self._last_vol_subset[:] = np.any(
                np.abs(zs[:,np.newaxis]-self.slice_order[
                        np.arange(0,slab[1][1])][np.newaxis]
                       ) < self.st_ratio,1)

            if data.shape[-1] == 1:
                np_and_ow(self._last_vol_subset,self._first_vol_subset)
                np.logical_and(self._first_vol_subset,self._subsamp,
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

        self.skip_sg=False
        if n_samples_total < self.min_sample_number:
            print 'skipping slab, only %d samples'%n_samples_total
            self.skip_sg = True
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
        bbr_offset=0 # TODO add as an option, and add weighting
        bbr_slope=.5
        cost = 1.0+np.tanh(bbr_slope*self._percent_contrast-bbr_offset).mean()
        # if there is a previous frame registered
        if len(self.transforms)>0 and True:
            # compute regularization 
            # add it to the cost
#            self.data[:np.count_nonzero(self._first_vol_subset_ssamp)]
#            regl = (self.data[1]-self._samples_data[1,self._first_vol_subset_ssamp]).std()/(self.data[1].std()*self._samples_data[1,self._first_vol_subset_ssamp].std())
            #cost += np.tanh(100*regl)
            #parameter regularization 
            param_diff = np.abs(self._pc-self.transforms[-1].param)
            print np.tanh(param_diff).sum()*self._motion_regularization#,  regl, np.tanh(100*regl), 
            cost += np.tanh(param_diff).sum()*self._motion_regularization
        return cost

    def _epi_inv_shiftmap(self, affine, shape):
        # compute inverse shift map using approximate nearest neighbor
        #
        fmap2fmri = np.linalg.inv(affine.dot(self.fmap2world))
        coords = nb.affines.apply_affine(
            fmap2fmri,
            np.rollaxis(np.mgrid[[slice(0,s) for s in self.fmap.shape]],0,4))
        shift = self.fmap_scale * self.fmap.get_data()
        coords[...,self.pe_dir] += shift
        coords = coords[shift!=0]
        shift = shift[shift!=0]
        inv_shiftmap = np.empty(shape)
        inv_shiftmap_dist = np.empty(shape)
        inv_shiftmap.fill(np.inf)
        inv_shiftmap_dist.fill(np.inf)
        rcoords = np.round(coords)
        dists = np.sum((coords-rcoords)**2,-1)
        for c,d,s in zip(rcoords,dists,shift):
            if np.all(c >= 0) and np.all(c < shape) \
                    and d < inv_shiftmap_dist[c[0],c[1],c[2]] \
                    and s < inv_shiftmap[c[0],c[1],c[2]]:
                inv_shiftmap[c[0],c[1],c[2]] = -s
        return inv_shiftmap

    # 
    def inv_resample(self, vol, affine, shape, order=0):
        grid = np.rollaxis(np.mgrid[[slice(0,s) for s in shape]], 0, 4)
        if self.fmap is not None and False:
            inv_shift = self._epi_inv_shiftmap(affine, shape)
            grid[...,self.pe_dir] += inv_shift
        epi2vol = np.linalg.inv(vol.get_affine()).dot(affine)
        voxs = nb.affines.apply_affine(epi2vol, grid)
        rvol = map_coordinates(
            vol.get_data(),
            voxs.reshape(-1,3).T, order=order).reshape(shape)
        return rvol
        

class EPIOnlineRealignFilter(EPIOnlineRealign):
    
    def correct(self, realigned, white_mask):

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
        

    def process(self, stack, *args, **kwargs):
        stack._init_dataset()
        self.correct(
            super(EPIOnlineRealignFilter,self).process(stack, yield_raw=True),
            *args, **kwargs)

    
            
def filenames_to_dicoms(fnames):
    for f in fnames:
        yield dicom.read_file(f)
