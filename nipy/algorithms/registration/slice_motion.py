# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings
import numpy as np

from nibabel.affines import apply_affine

from ...fixes.nibabel import io_orientation

from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .optimizer import configure_optimizer, use_derivatives
from .affine import Rigid
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_sample4d)
from scipy.ndimage import convolve1d, gaussian_filter

def extract_boundaries(wmseg,bbr_dist,subsample=1,fmap=None, exclude=None):
    
    wmdata = wmseg.get_data().astype(np.float32)
    wmsum = (wmdata>0.5).astype(np.int8)
    gradient = np.empty(wmdata.shape+(3,),dtype=np.float32)
    for axis in xrange(wmdata.ndim): #convolve separable
        convolve1d(wmsum,np.ones(3),axis,wmsum)
        order = [0]*wmdata.ndim
        order[axis] = 1
        gaussian_filter(wmdata,0.2,order,gradient[...,axis])
    boundaries = np.logical_and(wmdata>0.5,wmsum<26.5)
    # allow to remove some boundaries points as trunk for pulsatility
    if exclude != None: 
        boundaries[exclude] = False
    if subsample > 1:
        #subsample only in slice plane not in slicing axis
        aux = np.zeros(boundaries.shape,boundaries.dtype)
        aux[::subsample,::subsample] = boundaries[::subsample,::subsample]
        boundaries[:] = aux
    n_bnd_pts = np.count_nonzero(boundaries)
    coords = np.array(np.where(boundaries)+(np.ones(n_bnd_pts,np.double),),
                      dtype=np.double)
    coords_mm = wmseg.get_affine().dot(coords).T
    voxsize = np.sqrt((wmseg.get_affine()[:3,:3]**2).sum(0))
    gradient_mm = wmseg.get_affine()[:3,:3].dot(gradient[boundaries].T).T
    gradient_mm /= np.sqrt((gradient_mm**2).sum(-1))[:,np.newaxis]
    
    wmcoords = coords_mm[:,:3]+gradient_mm*bbr_dist #climb gradient
    gmcoords = coords_mm[:,:3]-gradient_mm*bbr_dist #go downhill
    wm_fmap_values = gm_fmap_values = None
    if fmap != None:
        fmap_spline = _cspline_transform(fmap.get_data())
        fmap_inv_aff = np.linalg.inv(fmap.get_affine())
        fmap_wmvox = np.dot(
            fmap_inv_aff,np.concatenate((wmcoords,np.ones((n_bnd_pts,1))),1).T)
        fmap_gmvox = np.dot(
            fmap_inv_aff,np.concatenate((gmcoords,np.ones((n_bnd_pts,1))),1).T)
        wm_fmap_values = np.empty(n_bnd_pts)
        gm_fmap_values = np.empty(n_bnd_pts)
        _cspline_sample3d(
            wm_fmap_values,fmap_spline,
            fmap_wmvox[0,:],fmap_wmvox[1,:],fmap_wmvox[2,:],
            mx=EXTRAPOLATE_SPACE,my=EXTRAPOLATE_SPACE,mz=EXTRAPOLATE_SPACE)
        _cspline_sample3d(
            gm_fmap_values,fmap_spline,
            fmap_gmvox[0,:],fmap_gmvox[1,:],fmap_gmvox[2,:],
            mx=EXTRAPOLATE_SPACE,my=EXTRAPOLATE_SPACE,mz=EXTRAPOLATE_SPACE)
    return coords_mm,wmcoords,gmcoords,wm_fmap_values,gm_fmap_values


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
EXTRAPOLATE_TIME = 'reflect'


class RealignSliceAlgorithm(object):

    def __init__(self,
                 im4d,
                 wmseg,
                 fmap=None,
                 pe_dir=1,
                 echo_spacing=0.005,
                 bbr_dist=0.5,
                 affine_class=Rigid,
                 slice_groups=None,
                 transforms=None,
                 subsampling=(1, 1, 1),
                 borders=BORDERS,
                 optimizer=OPTIMIZER,
                 optimize_template=True,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 refscan=REFSCAN,
                 slice_thickness=3.0):

        self.im4d = im4d
        self.slice_thickness=slice_thickness
        self.dims = im4d.get_data().shape
        self.nscans = self.dims[3]
        self.bnd_coords,self.wmcoords,self.gmcoords, \
            self.wm_fmap_values,self.gm_fmap_values = extract_boundaries(
            wmseg,bbr_dist,1,fmap)
        self.border_nvox=self.wmcoords.shape[0]
        self.gmcoords = np.concatenate((self.gmcoords,
                                        np.ones((self.border_nvox,1))),1)
        self.wmcoords = np.concatenate((self.wmcoords,
                                        np.ones((self.border_nvox,1))),1)

        self.slg_gm_vox = np.empty(self.gmcoords.shape,np.double)
        self.slg_wm_vox = np.empty(self.wmcoords.shape,np.double)

        self.pe_dir = pe_dir
        self.fmap_scale = echo_spacing * self.dims[pe_dir]/(2.0*np.pi)
        self.affine_class = affine_class

        # Initialize space/time transformation parameters
        self.affine = im4d.affine
        self.inv_affine = np.linalg.inv(self.affine)
        #defines bunch of slice for with same affine is estimated
        if slice_groups == None:
            self.slice_groups=[((t,0),(t,self.dims[im4d.slice_axis])) for t in range(self.nscans)]
            self.parts=self.nscans
        else:
            self.slice_groups=slice_groups
            self.parts=len(slice_groups)
        if transforms == None:
            self.transforms = []
        else:
            self.transforms = transforms

        self.scanner_time = im4d.scanner_time
        self.timestamps = im4d.tr * np.arange(self.nscans)

        # Compute the 4d cubic spline transform
        self.cbspline = _cspline_transform(im4d.get_data())

        # The reference scan conventionally defines the head
        # coordinate system
        self.optimize_template = optimize_template
        if not optimize_template and refscan == None:
            self.refscan = REFSCAN
        else:
            self.refscan = refscan

        # Set the minimization method
        self.set_fmin(optimizer, stepsize,
                      xtol=xtol,
                      ftol=ftol,
                      gtol=gtol,
                      maxiter=maxiter,
                      maxfun=maxfun)

        self.data = np.array([[]])

        # Auxiliary array for realignment estimation
#        self._res = np.zeros(masksize, dtype='double')
#        self._res0 = np.zeros(masksize, dtype='double')
#        self._aux = np.zeros(masksize, dtype='double')
#        self.A = np.zeros((masksize, self.transforms[0].param.size),
#                          dtype='double')
        self._pc = None

    def resample(self, sg):
        """
        Resample a particular slice group on the (sub-sampled) working
        grid.
        
        x,y,z,t are "head" grid coordinates
        X,Y,Z,T are "scanner" grid coordinates
        """
        inv_trans = np.linalg.inv(self.transforms[sg].as_affine())
        ref2fmri = np.dot(self.inv_affine,inv_trans)
        self.slg_gm_vox[:] = np.dot(self.gmcoords,ref2fmri.T) #(ABt)t = (B,At)
        self.slg_wm_vox[:] = np.dot(self.wmcoords,ref2fmri.T)

        #add shift in phase encoding direction
        if self.gm_fmap_values != None:
            self.slg_gm_vox[:,self.pe_dir]+=self.gm_fmap_values*self.fmap_scale
            self.slg_wm_vox[:,self.pe_dir]+=self.wm_fmap_values*self.fmap_scale
        
        sg = self.slice_groups[sg]
        tst = self.timestamps
        sa = self.im4d.slice_axis
        first_vol_subset = np.logical_and(
            self.slg_gm_vox[:,sa]>sg[0][1]-0.5,
            self.slg_wm_vox[:,sa]>sg[0][1]-0.5)

        last_vol_subset = np.logical_and(
            self.slg_gm_vox[:,sa]<sg[1][1]+0.5,
            self.slg_wm_vox[:,sa]<sg[1][1]+0.5)
         if sg[0][0] == sg[1][0]:
            first_vol_subset = np.logical_and(first_vol_subset,last_vol_subset)
            last_vol_subset = np.array([],dtype=np.bool)

        # adapt subsampling to keep regular amount of points in each slice
        if sf[0][0] < sg[1][0]-1:
            samples_slice_hist = np.histogram(
                self.slg_gm_vox[:,sa],
                np.arange(self.dims[sa]+1)-0.5)
        else:
            samples_slice_hist += np.histogram(
                self.slg_gm_vox[first_vol_subset,sa],
                np.arange(self.dims[sa]+1)-0.5)
            if sg[0][0] < sg[1][0]:
                samples_slice_hist += np.histogram(
                    self.slg_gm_vox[last_vol_subset,sa],
                    np.arange(self.dims[sa]+1)-0.5)
        
        

        self.skip_sg=False
        if np.count_nonzero(first_vol_subset) == 0:
            print 'skipping slice group, no boundaries contained'
            self.skip_sg=True
            return
        
        t_gm = np.concatenate(
            [self.scanner_time(self.slg_gm_vox[first_vol_subset,sa],
                               tst[sg[0][0]])]+
            [self.scanner_time(self.slg_gm_vox[:,sa],
                               tst[t]) for t in xrange(sg[0][0]+1,sg[1][0])]+
            [self.scanner_time(self.slg_gm_vox[last_vol_subset,sa],
                               tst[sg[1][0]])])

        t_wm = np.concatenate(
            [self.scanner_time(self.slg_wm_vox[first_vol_subset,sa],
                               tst[sg[0][0]])]+
            [self.scanner_time(self.slg_wm_vox[:,sa],
                               tst[t]) for t in xrange(sg[0][0]+1,sg[1][0])]+
            [self.scanner_time(self.slg_wm_vox[last_vol_subset,sa],
                               tst[sg[1][0]])])
        
        tmp_slg_gm_vox = np.concatenate((
                self.slg_gm_vox[first_vol_subset],
                self.slg_gm_vox.repeat(sg[1][0]-sg[0][0],0),
                self.slg_gm_vox[last_vol_subset]))

        tmp_slg_wm_vox = np.concatenate((
                self.slg_wm_vox[first_vol_subset],
                self.slg_wm_vox.repeat(sg[1][0]-sg[0][0],0),
                self.slg_wm_vox[last_vol_subset]))

        if self.data.shape[1] != t_gm.shape[0]:
            self.data = np.empty((2,t_gm.shape[0]))

        _cspline_sample4d(
            self.data[0],self.cbspline,
            tmp_slg_gm_vox[:,0], tmp_slg_gm_vox[:,1], tmp_slg_gm_vox[:,2],t_gm,
            mx=EXTRAPOLATE_SPACE,
            my=EXTRAPOLATE_SPACE,
            mz=EXTRAPOLATE_SPACE,
            mt=EXTRAPOLATE_TIME)
        _cspline_sample4d(
            self.data[1],self.cbspline,
            tmp_slg_wm_vox[:,0], tmp_slg_wm_vox[:,1], tmp_slg_wm_vox[:,2],t_wm,
            mx=EXTRAPOLATE_SPACE,
            my=EXTRAPOLATE_SPACE,
            mz=EXTRAPOLATE_SPACE,
            mt=EXTRAPOLATE_TIME)


    def resample_full_data(self):
        if VERBOSE:
            print('Gridding...')
        xyz = make_grid(self.dims[0:3])
        res = np.zeros(self.dims)
        for t in range(self.nscans):
            if VERBOSE:
                print('Fully resampling scan %d/%d' % (t + 1, self.nscans))
            X, Y, Z = scanner_coords(xyz, self.transforms[t].as_affine(),
                                     self.inv_affine, self.affine)
            T = self.scanner_time(Z, self.timestamps[t])
            _cspline_sample4d(res[:, :, :, t],
                              self.cbspline,
                              X, Y, Z, T,
                              mt='nearest')
        return res

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
        self.resample(t)

    def _init_energy(self, pc):
        if pc is self._pc:
            return
        self.set_transform(self._t, pc)
        self._pc = pc

        if self.use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            self._aux = self.data
            nrgy = np.diff(self.data,1,0).mean()/self.stepsize
            basis = np.eye(6)
            self._dV=np.zeros(6)
            for j in range(pc.size):
                self.set_transform(self._t, pc + self.stepsize * basis[j])
                self._dV[j] = np.diff(self.data,1,0).mean()/self.stepsize-nrgy
            self.transforms[self._t].param = pc
            self.data = self._aux
            # pre-compute gradient and hessian of numerator and
            # denominator
            self._H = 2.0 * self._dV[:,np.newaxis].dot(self._dV[np.newaxis])
            self._dV *= 2.0

    def _energy(self):
        percent_contrast = 200*np.diff(self.data,1,0)/self.data.sum(0)
        percent_contrast[np.abs(percent_contrast)<1e-6] = 0
        bbr_offset=0 # TODO add as an option, and add weighting
        cost = (1.0+np.tanh(percent_contrast-bbr_offset)).mean()
        print 'mean gradient %f'%cost
        return cost

    def _energy_gradient(self):
        return self._dV

    def _energy_hessian(self):
        return 

    def estimate_instant_motion(self, t):
        """
        Estimate motion parameters at a particular time.
        """
        if VERBOSE:
            print('Estimating motion at time frame %d/%d...'
                  % (t + 1, self.nscans))
        if len(self.transforms) <= t:
            if t > 0:
                self.transforms.append(self.transforms[t-1].copy())
            else:
                self.transforms.append(self.affine_class())
        
        self.set_transform(t,self.transforms[t].param)
        if self.skip_sg:
            return
        

        def f(pc):
            self._init_energy(pc)
            return self._energy()

        def fprime(pc):
            self._init_energy(pc)
            return self._energy_gradient()

        def fhess(pc):
            self._init_energy(pc)
            return self._energy_hessian()

        self._pc = None
        self._t = t
        fmin, args, kwargs =\
            configure_optimizer(self.optimizer,
                                fprime=fprime,
#                                fhess=fhess,
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
        pc = fmin(f, self.transforms[t].param, *args, **kwargs)
        self.set_transform(t, pc)
        #except:
        #warnings.warn('Minimization failed')

    def estimate_motion(self):
        """
        Optimize motion parameters for the whole sequence. All the
        time frames are initially resampled according to the current
        space/time transformation, the parameters of which are further
        optimized sequentially.
        """
        
        for t in range(self.parts):
            print 'slice group %d' % t
            self.estimate_instant_motion(t)
            if VERBOSE:
                print(self.transforms[t])
