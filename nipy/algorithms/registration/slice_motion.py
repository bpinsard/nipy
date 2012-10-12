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

def extract_boundaries(wmseg,bbr_dist,fmap=None):
    
    wmdata = wmseg.get_data()
    wmsum = (wmdata>0.5).astype(np.int16)
    gradient = np.empty(wmdata.shape+(3,))
    for axis in xrange(wmdata.ndim): #convolve separable
        convolve1d(wmsum,np.ones(3),axis,wmsum)
        order = [0]*wmdata.ndim
        order[axis] = 1
        gaussian_filter(wmdata,0.2,order,gradient[...,axis])
    boundaries = np.logical_and(wmdata>0.5,wmsum<26.5)
    n_bnd_pts = np.count_nonzero(boundaries)
    coords = np.array(np.where(boundaries)+(np.ones(n_bnd_pts),))
    coords_mm = wmseg.get_affine().dot(coords).T
    voxsize = np.sqrt((wmseg.get_affine()[:3,:3]**2).sum(0))
    gradient_mm = gradient/voxsize
    gradient_mm /= np.sqrt((gradient**2).sum(-1))[...,np.newaxis]
    
    wmcoords = coords_mm[:,:3]-gradient_mm[boundaries]*bbr_dist
    gmcoords = coords_mm[:,:3]+gradient_mm[boundaries]*bbr_dist
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
    return wmcoords,gmcoords,wm_fmap_values,gm_fmap_values


# Module globals
VERBOSE = True  # enables online print statements
SLICE_ORDER = 'ascending'
INTERLEAVED = None
OPTIMIZER = 'ncg'
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

LOOPS = 5  # loops within each run
BETWEEN_LOOPS = 5  # loops used to realign different runs
SPEEDUP = 5  # image sub-sampling factor for speeding up
"""
# How to tune those parameters for a multi-resolution implementation
LOOPS = 5, 1
BETWEEN_LOOPS = 5, 1
SPEEDUP = 5, 2
"""

class RealignSliceAlgorithm(object):

    def __init__(self,
                 im4d,
                 wmseg,
                 fmap=None,
                 pe_dir=1,
                 bbr_dist=0.5,
                 affine_class=Rigid,
                 slice_groups=None,
                 transforms=None,
                 time_interp=True,
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
                 refscan=REFSCAN):

        self.dims = im4d.get_data().shape
        self.nscans = self.dims[3]
        self.wmcoords,self.gmcoords,self.wm_fmap_values,gm_fmap_values = extract_boundaries(wmseg,bbr_dist,fmap)

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
            self.transforms = [affine_class() for sg in self.slice_groups]
        else:
            self.transforms = transforms

        self.scanner_time = im4d.scanner_time
        self.timestamps = im4d.tr * np.arange(self.nscans)

        # Compute the 4d cubic spline transform
        self.time_interp = time_interp
        if time_interp:
            self.cbspline = _cspline_transform(im4d.get_data())
        else:
            self.cbspline = np.zeros(self.dims, dtype='double')
            for t in range(self.dims[3]):
                self.cbspline[:, :, :, t] =\
                    _cspline_transform(im4d.get_data()[:, :, :, t])

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

        self.wm_values = np.empty(self.wmcoords.shape[0])
        self.gm_values = np.empty(self.gmcoords.shape[0])

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
        print 'resampling....'
        inv_trans = np.linalg.inv(self.transforms[sg].as_affine())
        ref2fmri = np.dot(inv_trans,self.inv_affine)
        slg_gm_vox = np.dot(ref2fmri,self.gmcoords)
        slg_wm_vox = np.dot(ref2fmri,self.wmcoords)
        #add shift in phase encoding direction
        if self.gm_fmap_values != None:
            slg_gm_vox[:,slice_axis] += self.gm_fmap_values*self.fmap_scale
            slg_wm_vox[:,slice_axis] += self.wm_fmap_values*self.fmap_scale
            
        sg = self.slice_groups[sg]
        tst = self.timestamps
        sa = self.slice_axis
        t_gm = np.concatenate(
            [self.scanner_time(
                    slg_gm_vox[slg_gm_vox[:,sa]>sg[0][1],sa],tst[sg[0][0]])]+
            [self.scanner_time(
                    slg_gm_vox[:,self.slice_axis],
                    tst[t]) for t in xrange(sg[0][1]+1,sg[1][1])]+
            [self.scanner_time(
                    slg_gm_vox[slg_gm_vox[:,sa]<=sg[1][1],sa],tst[sg[1][0]])])
        t_wm = np.concatenate(
            [self.scanner_time(
                    slg_wm_vox[slg_wm_vox[:,sa]>sg[0][1],sa],tst[sg[0][0]])]+
            [self.scanner_time(
                    slg_wm_vox[:,self.slice_axis],
                    tst[t]) for t in xrange(sg[0][1]+1,sg[1][1])]+
            [self.scanner_time(
                    slg_wm_vox[slg_wm_vox[:,sa]<=sg[1][1],sa],tst[sg[1][0]])])
        
        _cspline_sample4d(
            self.gm_values,self.cbspline,
            slg_gm_vox[:,0], slg_gm_vox[:,1], slg_gm_vox[:,2],t_gm,
            mx=EXTRAPOLATE_SPACE,
            my=EXTRAPOLATE_SPACE,
            mz=EXTRAPOLATE_SPACE,
            mt=EXTRAPOLATE_TIME)
        _cspline_sample4d(
            self.wm_values,self.cbspline,
            slg_wm_vox[:,0], slg_wm_vox[:,1], slg_wm_vox[:,2],t_wm,
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
            if self.time_interp:
                T = self.scanner_time(Z, self.timestamps[t])
                _cspline_sample4d(res[:, :, :, t],
                                  self.cbspline,
                                  X, Y, Z, T,
                                  mt='nearest')
            else:
                _cspline_sample3d(res[:, :, :, t],
                                  self.cbspline[:, :, :, t],
                                  X, Y, Z)
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
        self._res[:] = self.data[:, self._t] - self.mu[:]
        self._V = np.maximum(self.offset + np.mean(self._res ** 2), SMALL)
        self._res0[:] = self.data[:, self._t] - self.mu0
        self._V0 = np.maximum(self.offset0 + np.mean(self._res0 ** 2), SMALL)

        if self.use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            self._aux[:] = self.data[:, self._t]
            basis = np.eye(6)
            for j in range(pc.size):
                self.set_transform(self._t, pc + self.stepsize * basis[j])
                self.A[:, j] = (self.data[:, self._t] - self._aux)\
                    / self.stepsize
            self.transforms[self._t].param = pc
            self.data[:, self._t] = self._aux[:]
            # pre-compute gradient and hessian of numerator and
            # denominator
            c = 2 / float(self.data.shape[0])
            self._dV = c * np.dot(self.A.T, self._res)
            self._dV0 = c * np.dot(self.A.T, self._res0)
            self._H = c * np.dot(self.A.T, self.A)

    def _energy(self):
        """
        The alignment energy is defined as the log-ratio between the
        average temporal variance in the sequence and the global
        spatio-temporal variance.
        """
        return (self.gm_values-self.wm_values).mean()

    def _energy_gradient(self):
        return self._dV / self._V - self._dV0 / self._V0

    def _energy_hessian(self):
        return (1 / self._V - 1 / self._V0) * self._H\
            - np.dot(self._dV, self._dV.T) / np.maximum(self._V ** 2, SMALL)\
            + np.dot(self._dV0, self._dV0.T) / np.maximum(self._V0 ** 2, SMALL)

    def estimate_instant_motion(self, t):
        """
        Estimate motion parameters at a particular time.
        """
        if VERBOSE:
            print('Estimating motion at time frame %d/%d...'
                  % (t + 1, self.nscans))

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
        try:
            pc = fmin(f, self.transforms[t].param, *args, **kwargs)
            self.set_transform(t, pc)
        except:
            warnings.warn('Minimization failed')

    def estimate_motion(self):
        """
        Optimize motion parameters for the whole sequence. All the
        time frames are initially resampled according to the current
        space/time transformation, the parameters of which are further
        optimized sequentially.
        """
        
        for t in range(self.parts):
            self.estimate_instant_motion(t)
            if VERBOSE:
                print(self.transforms[t])

    def align_to_refscan(self):
        """
        The `motion_estimate` method aligns scans with an online
        template so that spatial transforms map some average head
        space to the scanner space. To conventionally redefine the
        head space as being aligned with some reference scan, we need
        to right compose each head_average-to-scanner transform with
        the refscan's 'to head_average' transform.
        """
        if self.refscan == None:
            return
        Tref_inv = self.transforms[self.refscan].inv()
        for t in range(self.nscans):
            self.transforms[t] = (self.transforms[t]).compose(Tref_inv)

