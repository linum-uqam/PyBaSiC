#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['BaSiC']

from pathlib import Path
import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
import tqdm

class BaSiC(object):
    def __init__(self, input, estimate_darkfield=False, extension=".tif", verbose=False):
        """Input can either be:
         - Path to a directory containing the images to process
         - List of images path to process
         - List of images as ndarray
         - A stack of ndarrays of shape N_Images x Height x Width
        """
        self.input_type = None
        self.extension = extension
        if isinstance(input, str) or isinstance(input, Path): # Directory
            self.directory = input
            self._sniff_input() # Get a list of files
            self.input_type = "directory"
        elif isinstance(input, np.ndarray):
            self.img_stack = input
            self.input_type = "images_stack"
        elif isinstance(input, list) and (isinstance(input[0], str) or isinstance(input[0], Path)):
            self.file_list = input
            self.input_type = "files_list"
        elif isinstance(input, list) and isinstance(input[0], np.ndarray):
            self.img_stack = np.array(input)
            self.input_type = "images_list"
        else:
            raise "input should either be a directory, a list of ndarrays, or a ndarray stack."

        # Optimizer parameters
        self.working_size = 128  # px : image resampling size to accelerate learning.
        self.epsilon = 0.1  # Iterative reweighted L1-norm stability parameter
        self.l_s = None  # flat-field regularization parameter (set automatically if None)
        self.l_d = None  # dark-field regularization parameter (set automatically if None)
        self.reweighting_tolerance = 1e-3
        self.max_reweightingIterations = 10
        self.reweighting_iteration = 0
        self.estimate_darkfield = estimate_darkfield
        self.verbose = verbose

    def _sniff_input(self):
        # Get a list of tiles to process
        directory = Path(self.directory).resolve()
        file_list = list(directory.glob(f"*{self.extension}"))
        file_list.sort()
        self.files = file_list
        assert len(self.files) > 0, "No files were found in the input directory. Make sure you provided the right path and file extension. Aborting."

    def _load_images(self, img_stack = None):
        # Load the stack
        if img_stack is None:
            img_stack = []
            if self.verbose:
                gen = tqdm.tqdm(self.files, "Loading the images")
            else:
                gen = self.files
            for this_file in gen:
                img = cv2.imread(str(this_file), cv2.IMREAD_ANYDEPTH)
                img_stack.append(img)
            self.img_stack = np.array(img_stack)
        else:
            self.img_stack = img_stack
        self.n_images = self.img_stack.shape[0]

        # Resample the images to accelerate learning
        self.image_shape = self.img_stack.shape[1::]
        new_shape = tuple([self.working_size]*2)
        img_stack_p = np.zeros([self.n_images, *new_shape], dtype=self.img_stack.dtype)
        if self.working_size > self.image_shape[0]:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        for i in range(self.n_images):
            img = self.img_stack[i, ...].squeeze()
            img_stack_p[i, ...] = cv2.resize(img.T, new_shape, interpolation=interpolation).T
        self.img_stack_resized = img_stack_p.astype(np.float32)

    def normalize(self, img, clip=True, epsilon=1e-6):
        img_p = (img.astype(np.float32) - self.darkfield_fullsize) / (self.flatfield_fullsize + epsilon)
        if clip and not(img.dtype in [np.float32, np.float64]):

            img_p[img_p < np.iinfo(img.dtype).min] = np.iinfo(img.dtype).min
            img_p[img_p > np.iinfo(img.dtype).max] = np.iinfo(img.dtype).max

        return img_p.astype(img.dtype)

    def write_images(self, directory, epsilon=1e-6):
        # Create the output directory
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Loop over the images
        for i in tqdm.tqdm(range(self.n_images), desc="Shading Correction"):
            this_file = self.files[i]
            this_img = self.img_stack[i, ...]

            # Normalize the image
            img_p = self.normalize(this_img, epsilon=epsilon)

            # Get the output filename
            filename = directory / this_file.name

            # Save the file
            cv2.imwrite(str(filename), img_p)

    def prepare(self, img_stack=None):
        # Load the data
        if img_stack is not None:
            self._load_images(img_stack)
        elif self.input_type in ["directory", "files_list"]:
            self._load_images()
        elif self.input_type in ["images_stack", "images_list"]:
            self._load_images(self.img_stack)

        # Initialize the regularization parameters
        mean_value = self.img_stack_resized.mean(axis=0)
        mean_value /= mean_value.mean() # Normalized pixel-wise mean of all images
        mean_value_dct = dctn(mean_value, norm='ortho')

        if self.l_s is None:
            self.l_s = np.abs(mean_value_dct).sum() / 800.0
        if self.l_d is None:
            self.l_d = np.abs(mean_value_dct).sum() / 2000.0

        # Construct the measurement matrix
        self.img_sort = np.sort(self.img_stack_resized, axis=0)

        # Initialize the darkfield, flatfield, offset, and weights (for the L1 reweighted loss)
        self.offset = np.zeros([self.working_size]*2)
        self.flatfield = np.ones([self.working_size]*2)
        self.flatfield_fullsize = np.ones(self.image_shape)
        self.darkfield = np.random.randn(self.working_size, self.working_size)
        self.darkfield_fullsize = np.zeros(self.image_shape)
        self.W = np.ones_like(self.img_sort)

        # Initialize other parameters
        self.iteration = 0
        self.flag_reweigthing = True

    def update_weights(self):
        """Weighting matrix for the Reweighted L1-norm"""
        # Weight Update formula in the paper
        # self.W = 1.0 / (np.abs(self.Ir / self.Ib) + self.epsilon)

        # Actual Weight update formula in the matlab implementation
        self.W = 1.0 / (np.abs(self.Ir / (self.Ib.mean() + 1e-6)) + self.epsilon)
        self.W = self.W * self.W.size / self.W.sum()
        self.reweighting_iteration += 1

    def update(self):
        last_flatfield = self.flatfield.copy()
        last_darkfield = self.darkfield.copy()

        # Perform LADM optimization
        Ib, Ir, D = inexact_alm_l1(self.img_sort, self.l_s, self.l_d, weight=self.W, estimateDarkField=self.estimate_darkfield, verbose=self.verbose)

        # Reshape the images.
        self.Ib = np.reshape(Ib, (self.n_images, self.working_size, self.working_size)) # Flat-field
        self.Ir = np.reshape(Ir, (self.n_images, self.working_size, self.working_size)) # Residual
        D = np.reshape(D, (self.working_size, self.working_size)) # Dark-field

        # Update the weight matrix
        self.update_weights()

        # Update the flat-field and dark-field
        self.flatfield = self.Ib.mean(axis=0) - D
        self.flatfield = self.flatfield / self.flatfield.mean()
        self.darkfield = D

        # Compute the difference between the new fields and the last ones.
        mad_flatfield = np.abs(self.flatfield - last_flatfield).sum() / np.abs(last_flatfield).sum()
        mad_darkfield = np.abs(self.darkfield - last_darkfield).sum()
        if mad_darkfield < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = mad_darkfield / max(np.abs(last_darkfield).sum(), 1e-6)

        # Check if another L1 reweighting is necessary
        if (max(mad_flatfield, mad_darkfield)<=self.reweighting_tolerance) or (self.reweighting_iteration > self.max_reweightingIterations):
            self.flag_reweigthing = False

    def run(self):
        if self.verbose:
            pbar = tqdm.tqdm(desc="Reweighting Iteration", total=self.max_reweightingIterations)
        while self.flag_reweigthing:
            self.update()
            if self.verbose:
                pbar.update()
            # self.display_fields()
        if self.verbose:
            pbar.close()

        # Reshape the flat and dark fields to the original shape
        self.flatfield_fullsize = cv2.resize(self.flatfield.T, self.image_shape, cv2.INTER_LINEAR).T
        self.flatfield_fullsize = self.flatfield_fullsize / self.flatfield_fullsize.mean()
        self.darkfield_fullsize = cv2.resize(self.darkfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def set_flatfield(self, flatfield):
        self.flatfield_fullsize = cv2.resize(flatfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def set_darkfield(self, darkfield):
        self.darkfield_fullsize = cv2.resize(darkfield.T, self.image_shape, cv2.INTER_LINEAR).T

    def get_flatfield(self):
        return self.flatfield_fullsize.copy()

    def get_darkfield(self):
        return self.darkfield_fullsize.copy()

def shrink(theta, epsilon=1e-3):
    """Scalar Shrink Operator"""
    theta_p = np.sign(theta) * np.maximum(np.abs(theta) - epsilon, 0)
    return theta_p

def inexact_alm_l1(imgs, l_s, l_d, tol=1e-6, maxIter=500, weight=1, estimateDarkField=True, rho=1.5, verbose=False):
    """l1 minimization using the inexact augmented Lagrange multiplier method for Sparse low rank matrix recovery.
    Parameters
    ----------
    imgs : N x P x Q ndarray
        Images stack
    l_s : float
        Flat-field regularization parameter
    l_d : float
        Dark-field regularization parameter
    tol : float
        Convergence tolerance
    maxIter : int
        Maximum iterations number
    weight : N x P x Q ndarray
        Optional weight matrix used for the reweighted L1 norm
    estimateDarkField : bool
        Set to True to estimate the darkfield in addition to the flat field
    darkFieldLimit : float
        Maximum value for the darkfield, use to constrain the minimization
    rho : float
        Lagrange multiplier learning rate

    Returns
    -------
    S : ndarray
        Estimated unnormalised flat-field
    Ib : ndarray

    Ir : ndarray



    Notes
    -----
    % modified from the BaSiC matlab implementation
    % modified from Robust PCA
    % reference:
    % Peng et al. "A BaSiC tool for background and shading correction
    % of optical microscopy images" Nature Communications, 14836(2017)
    % Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal Component
    % Analysis?" J. ACM (58) 2011

    % D - m x m x n matrix of observations/data (required input)
    %

    % while ~converged
    %   minimize (inexactly, update A and E only once)
    %   L(W, E,Y,u) = |E|_1+lambda * |W|_1 + <Y2,D-repmat(QWQ^T)-E> + +mu/2 * |D-repmat(QWQ^T)-E|_F^2;
    %   Y1 = Y1 + \mu * (D - repmat(QWQ^T) - E);
    %   \mu = \rho * \mu;
    % end
    %
    % Tingying Peng (tingying.peng@tum.de)

    %
    % Copyright: CAMP, Technical University of Munich

    """
    ###############################
    # Initialize the optimization #
    ###############################
    N, P, Q = imgs.shape[:]

    # Reshape the image stack as a measurement matrix
    D = np.reshape(imgs, (N, P*Q))
    d_norm = np.linalg.norm(D, "fro")
    W = np.reshape(weight, D.shape)
    B1_uplimit = D.min()
    B1 = 0

    # Initialize Ib, Ir, B, S and D
    S = np.zeros_like(D)  # Flat-field
    Sf = dctn(np.reshape(S, (N,P,Q)).mean(axis=0), norm='ortho') # Flat-field, in Fourier Domain
    Ir = np.zeros_like(D)  # Residual
    B = np.ones((N,1)) # Image baseline
    D_field = np.zeros((1, P*Q)) # Darkfield

    # Initialize Lagrange multiplier
    Y = 0
    D_svd = np.linalg.svd(D, compute_uv=False)
    mu = 12.5 / D_svd[0] # TODO: This can be tuned
    mu_bar = mu * 1e7
    ent2 = 10

    converged = False
    iteration = 0
    if verbose:
        pbar = tqdm.tqdm(desc="Iteration", total=maxIter)
    while not(converged) and (iteration < maxIter): # TODO: Add tqdm
        # 1. Compute DCT of the flat-field Sf(k), and Ib(k)
        S = np.reshape(idctn(Sf, norm='ortho'), (1,P*Q)) # Flat-field in spatial domain
        Ib = S * B + D_field # with broadcasting, as S (NxPQ), B (Nx1), D(1xPQ)

        # 2. Update Sf(k) -> Sf(k+1)
        dS = (D - Ib - Ir + Y / mu) # Shape: N x PQ
        dS = np.reshape(dS, (N, P, Q))
        dS = dS.mean(axis=0) # Averate over all images
        Sf = Sf + dctn(dS, norm='ortho')
        Sf = shrink(Sf, l_s / mu) # Scalar shrinkage operator

        # 3. Get S(k+1,x) and mid-iteration Ib(k+1/2)
        S = np.reshape(idctn(Sf, norm='ortho'), (1, P*Q)) # Flat-field at iteration S(k+1)
        Ib = S * B + D_field # Ib, mid iteration

        # 4. Update the residual Ir(k) -> Ir(k+1)
        dIr = (D - Ib - Ir + Y / mu)
        Ir = Ir + dIr
        Ir = shrink(Ir, W/mu)

        # 5. Update the image baseline B(k) -> B(k+1), and Ib(k+1)
        R = D - Ir
        B = R.mean(axis=1, keepdims=True) / R.mean()
        B[B < 0] = 0 # Enforce non-negativity constraint
        # Ib = S * B + D_field

        # Dark-field optimization
        if estimateDarkField: # TODO: Add the darkfield optimization here
            # 1. Dz estimation with least-mean square
            mask_validB = B < 1
            mask_highS = S > (S.mean() - 1e-6)
            mask_lowS = S < (S.mean() + 1e-6)
            R_high = np.mean(R * (mask_highS * mask_validB), axis=1, keepdims=True)
            R_low = np.mean(R * (mask_lowS * mask_validB), axis=1, keepdims=True)
            B1 = (R_high - R_low)/R.mean()
            k = mask_validB.sum()
            temp1 = np.sum(B[mask_validB]**2)
            temp2 = B[mask_validB].sum()
            temp3 = B1.sum()
            temp4 = np.sum(B[mask_validB] * B1)
            temp5 = temp2 * temp3 - k * temp4
            if temp5 == 0:
                B1 = 0
            else:
                B1 = (temp1*temp3 - temp2*temp4) / temp5

            # Clipping to limit the range of Dz to 0 and min(D)
            B1 = np.maximum(B1, 0) # Non negativity constraing
            B1 = np.minimum(B1, B1_uplimit / S.mean())

            # 2. Dr optimization
            Z = B1 * (S.mean() - S)

            A1_offset = np.ma.masked_array(R, np.tile(~mask_validB, (1, P*Q))).mean(axis=0, keepdims=True) - B[mask_validB].mean() * S
            A1_offset = A1_offset - A1_offset.mean()
            A_offset = A1_offset - A1_offset.mean() - Z

            # Smooth A_offset (Dr)
            Dr_f = dctn(np.reshape(A_offset, (P,Q)), norm='ortho') # Fourier-domain flat-field residual
            Dr_f = shrink(Dr_f, l_d / (ent2 * mu)) # Shrink operator in Fourier-domain for smooth residual
            Dr = idctn(Dr_f, norm='ortho').reshape((1, P*Q))
            Dr = shrink(Dr, l_d / (mu*ent2)) # Shrink operator in spatial domain for sparse redisual
            D_field = Dr + Z

        # 6. Update the Lagrange multiplier Y(k)->Y(k+1), mu(k)->mu(k+1), and k->k+1
        dY = D - Ib - Ir
        Y = Y + mu * dY
        mu = min(mu * rho, mu_bar) # Clip mu
        iteration += 1

        # Evaluate convergence
        stopCriterion = np.linalg.norm(dY, "fro") / d_norm
        if verbose:
            pbar.update()
        # pbar.set_description(f"Iteration (Criterion={stopCriterion:.3e})", refresh=False)
        if stopCriterion < tol:
            converged = True

        # TODO: Add logging

    if iteration == maxIter:
        print("Maximum iterations reached")
    if verbose:
        pbar.close()
    # Update the darkfield, with
    D_field = D_field + B1*S

    return Ib, Ir, D_field

