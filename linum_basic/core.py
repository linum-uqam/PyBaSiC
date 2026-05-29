"""BaSiC shading correction — main estimator class.

Implements the BaSiC algorithm for retrospective estimation of
illumination flat-field and dark-field correction profiles from a stack of
microscopy images.

References
----------
.. [1] Peng, T. *et al.* "A BaSiC tool for background and shading correction
   of optical microscopy images." *Nat. Commun.* **8**, 14836 (2017).
   https://doi.org/10.1038/ncomms14836
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import cv2
import numpy as np
from scipy.fft import dctn
from tqdm.auto import tqdm

from linum_basic._alm import inexact_alm_l1
from linum_basic.backend import get_xp

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["DEFAULT_L_D_DIVISOR", "DEFAULT_L_S_DIVISOR", "BaSiC", "dct_energy"]

# Default divisors for auto-tuning the regularisation weights from DCT energy:
# ``l_s = dct_energy / DEFAULT_L_S_DIVISOR`` and likewise for ``l_d``.  These
# are the single source of truth shared by :meth:`BaSiC.prepare` and
# :func:`linum_basic.tuning.tune`.
DEFAULT_L_S_DIVISOR = 800.0
DEFAULT_L_D_DIVISOR = 2000.0


def dct_energy(mean_image: NDArray) -> float:
    """Return the DCT energy of a normalised mean image.

    The mean image is normalised by its own mean before the 2-D DCT so the
    result is scale-invariant across datasets.  This is the quantity BaSiC
    uses to auto-tune its regularisation weights (``l_s``/``l_d``).

    Parameters
    ----------
    mean_image : numpy.ndarray
        The per-pixel mean image (e.g. ``stack.mean(axis=0)``).

    Returns
    -------
    float
        Sum of the absolute DCT coefficients of the normalised mean image.
    """
    normalised = mean_image / (float(mean_image.mean()) + 1e-9)
    return float(np.abs(dctn(normalised, norm="ortho")).sum())


class BaSiC:
    r"""Retrospective shading-correction estimator based on the BaSiC method.

    Accepts a collection of images in several formats (directory, file list,
    NumPy array stack, or list of arrays), resizes them to a working
    resolution, then jointly estimates a smooth flat-field and an optional
    dark-field via L1-penalised matrix factorisation.

    Parameters
    ----------
    input : str, Path, list of str/Path, list of numpy.ndarray, or numpy.ndarray
        Input images in one of the following forms:

        * A path to a directory containing image files.
        * A list of file paths.
        * A list of 2-D NumPy arrays (each image as an ndarray).
        * A 3-D NumPy array of shape *(N, H, W)*.

    estimate_darkfield : bool
        When ``True`` the dark-field is estimated alongside the flat-field.
    extension : str
        Glob file extension used when *input* is a directory.
    verbose : bool
        Show progress bars during loading and optimisation.
    backend : {"numpy", "torch", "auto"}
        Compute backend for the ALM optimisation loop.  ``"auto"`` picks
        Torch with CUDA/MPS when available, otherwise falls back to NumPy.
    device : str or None
        PyTorch device string (e.g. ``"cuda:0"``).  Ignored for NumPy.

    Attributes
    ----------
    working_size : int
        Side length (pixels) of the square working resolution.  Images are
        resized to ``working_size x working_size`` before optimisation.
        Default ``128``.  Increase to ``256`` for finer flat-field detail.
        Must be set before :meth:`prepare`.
    epsilon : float
        Stability constant $\varepsilon$ in the reweighted-L1 weight update.
        Default ``0.1``.  Smaller values produce sharper weight contrast;
        larger values approximate plain L1.
    l_s : float or None
        Flat-field regularisation weight $\\lambda_s$ (DCT-domain).
        ``None`` triggers auto-tuning in :meth:`prepare`
        (``dct_sum / 800``).  Override after calling :meth:`prepare` but
        before :meth:`run`.
    l_d : float or None
        Dark-field regularisation weight $\\lambda_d$.
        ``None`` triggers auto-tuning in :meth:`prepare`
        (``dct_sum / 2000``).  Only effective when
        ``estimate_darkfield=True``.
    reweighting_tolerance : float
        Convergence threshold for the outer reweighting loop (relative
        change in flat-field and dark-field).  Default ``1e-3``.
    max_reweighting_iterations : int
        Hard cap on outer reweighting iterations.  Default ``10``.

    Raises
    ------
    TypeError
        If *input* is not one of the supported types.

    See Also
    --------
    linum_basic.algorithms.inexact_alm_l1 : The underlying ALM solver.

    Examples
    --------
    Estimate flat-field from a directory of TIFF tiles:

    >>> model = BaSiC("/path/to/tiles", estimate_darkfield=True)
    >>> model.prepare()
    >>> model.run()
    >>> flatfield = model.get_flatfield()

    Use an existing NumPy stack:

    >>> import numpy as np
    >>> stack = np.random.rand(50, 512, 512).astype(np.float32)
    >>> model = BaSiC(stack)
    >>> model.prepare()
    >>> model.run()
    """

    def __init__(
        self,
        input: str | Path | list[str | Path] | list[NDArray] | NDArray,
        *,
        estimate_darkfield: bool = False,
        extension: str = ".tif",
        verbose: bool = False,
        backend: Literal["numpy", "torch", "auto"] = "numpy",
        device: str | None = None,
    ) -> None:
        self.input_type: str | None = None
        self.extension = extension
        self.estimate_darkfield = estimate_darkfield
        self.verbose = verbose
        self._xp = get_xp(backend, device)

        if isinstance(input, (str, Path)):
            self.directory: str | Path = input
            self._sniff_input()
            self.input_type = "directory"
        elif isinstance(input, np.ndarray):
            self.img_stack: NDArray = input
            self.input_type = "images_stack"
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], (str, Path)):
            self.files: list[Path] = [Path(f) for f in cast("list[str | Path]", input)]
            self.input_type = "files_list"
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], np.ndarray):
            self.img_stack = np.array(input)
            self.input_type = "images_list"
        else:
            msg = "input must be a directory path, a list of file paths, a list of ndarrays, or a 3-D ndarray stack."
            raise TypeError(msg)

        # Optimiser hyper-parameters
        self.working_size: int = 128
        self.epsilon: float = 0.1
        self.l_s: float | None = None
        self.l_d: float | None = None
        self.reweighting_tolerance: float = 1e-3
        self.max_reweighting_iterations: int = 10
        self.reweighting_iteration: int = 0
        self.warm_start_reweighting: bool = False

        # State (populated by prepare / run)
        self._flag_reweighting: bool = True
        self._iteration: int = 0
        self.n_images: int = 0
        self.image_shape: tuple[int, int] = (0, 0)
        self.flatfield: NDArray = np.ones((self.working_size, self.working_size), dtype=np.float32)
        self.darkfield: NDArray = np.zeros((self.working_size, self.working_size), dtype=np.float32)
        self.flatfield_fullsize: NDArray = np.ones((1, 1), dtype=np.float32)
        self.darkfield_fullsize: NDArray = np.zeros((1, 1), dtype=np.float32)
        self._alm_state: dict | None = None  # warm-start state for outer reweighting

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, stack: NDArray, **kwargs: Any) -> BaSiC:
        """Construct a :class:`BaSiC` instance directly from an image array.

        Parameters
        ----------
        stack : numpy.ndarray, shape (N, H, W)
            Pre-loaded image stack.
        **kwargs
            Forwarded to :class:`BaSiC.__init__`.
        """
        return cls(stack, **kwargs)

    @classmethod
    def from_directory(cls, path: str | Path, **kwargs: Any) -> BaSiC:
        """Construct a :class:`BaSiC` instance from a directory of images.

        Parameters
        ----------
        path : str or Path
            Directory containing image files.
        **kwargs
            Forwarded to :class:`BaSiC.__init__`.
        """
        return cls(path, **kwargs)

    @classmethod
    def from_files(cls, file_list: list[str | Path], **kwargs: Any) -> BaSiC:
        """Construct a :class:`BaSiC` instance from an explicit file list.

        Parameters
        ----------
        file_list : list of str or Path
            Ordered list of image file paths.
        **kwargs
            Forwarded to :class:`BaSiC.__init__`.
        """
        return cls(file_list, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sniff_input(self) -> None:
        """Glob image files from the input directory.

        Populates ``self.files`` with a sorted list of paths matching
        ``self.extension``.

        Raises
        ------
        AssertionError
            If no matching files are found.
        """
        directory = Path(self.directory).resolve()
        file_list = sorted(directory.glob(f"*{self.extension}"))
        assert len(file_list) > 0, (
            f"No files with extension '{self.extension}' found in '{directory}'.  Check the path and --extension flag."
        )
        self.files = file_list

    def _load_images(self, img_stack: NDArray | None = None) -> None:
        """Load and resize images to the working resolution.

        Parameters
        ----------
        img_stack : numpy.ndarray or None
            If provided the stack is used directly instead of reading from
            disk.  Shape must be *(N, H, W)*.

        Notes
        -----
        Images are resized to ``self.working_size x self.working_size``
        using bilinear (upscale) or area (downscale) interpolation.  The
        resize step runs on CPU/OpenCV regardless of the selected backend
        because it is I/O-bound.
        """
        if img_stack is not None:
            self.img_stack = img_stack
        elif self.input_type in {"directory", "files_list"}:
            # Parallel I/O: cv2.imread releases the GIL, so threads overlap
            # disk reads across multiple files at once.
            n_workers = min(32, len(self.files))

            def _read_one(path: Path) -> NDArray | None:
                return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                raw_iter = pool.map(_read_one, self.files)
            if self.verbose:
                raw_iter = tqdm(raw_iter, desc="Loading images", total=len(self.files), leave=False)
            raw: list[NDArray] = [img for img in raw_iter if img is not None]
            self.img_stack = np.array(raw)

        self.n_images = self.img_stack.shape[0]
        self.image_shape = self.img_stack.shape[1:]

        new_shape = (self.working_size, self.working_size)
        interp = cv2.INTER_LINEAR if self.working_size > self.image_shape[0] else cv2.INTER_AREA
        resized = np.zeros([self.n_images, *new_shape], dtype=np.float32)

        # Parallel resizing: cv2.resize releases the GIL; each worker writes
        # to an independent row of the pre-allocated output array.
        def _resize_one(i: int) -> None:
            img = self.img_stack[i].squeeze()
            resized[i] = cv2.resize(img.T, new_shape, interpolation=interp).T

        with ThreadPoolExecutor(max_workers=min(32, self.n_images)) as pool:
            list(pool.map(_resize_one, range(self.n_images)))

        self.img_stack_resized = resized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, img_stack: NDArray | None = None) -> None:
        """Load data and initialise all optimisation state.

        Must be called before :meth:`run`.  Computes the initial
        regularisation parameters (``l_s``, ``l_d``) from the normalised
        pixel-wise mean of the image stack if they have not been set
        manually.

        Parameters
        ----------
        img_stack : numpy.ndarray or None
            Optional pre-loaded image stack *(N, H, W)*.  If ``None`` the
            stack is loaded according to ``input_type``.

        See Also
        --------
        linum_basic.core.BaSiC.l_s : Flat-field regularisation weight.
        linum_basic.core.BaSiC.l_d : Dark-field regularisation weight.
        """
        if img_stack is not None:
            self._load_images(img_stack)
        elif self.input_type in {"directory", "files_list"}:
            self._load_images()
        else:
            self._load_images(self.img_stack)

        # Auto-tune regularisation from the DCT of the mean image
        dct_sum = dct_energy(self.img_stack_resized.mean(axis=0))
        if self.l_s is None:
            self.l_s = dct_sum / DEFAULT_L_S_DIVISOR
        if self.l_d is None:
            self.l_d = dct_sum / DEFAULT_L_D_DIVISOR

        self.img_sort = np.sort(self.img_stack_resized, axis=0)

        ws = self.working_size
        self.flatfield = np.ones((ws, ws), dtype=np.float32)
        self.darkfield = np.zeros((ws, ws), dtype=np.float32)
        self.flatfield_fullsize = np.ones(self.image_shape, dtype=np.float32)
        self.darkfield_fullsize = np.zeros(self.image_shape, dtype=np.float32)
        self._W: NDArray = np.ones_like(self.img_sort)
        self._Ir: NDArray = np.zeros_like(self.img_sort)
        self._Ib: NDArray = np.zeros_like(self.img_sort)
        self._flag_reweighting = True
        self.reweighting_iteration = 0
        self._alm_state: dict | None = None  # reset warm-start state

    def update_weights(self) -> None:
        """Update the reweighting matrix for the next ALM iteration.

        Implements the reweighted-L1 weight update from the BaSiC paper::

            W ← 1 / (|Ir / (mean(Ib) + ε)| + ε)

        followed by normalisation so that ``mean(W) = 1``.

        Notes
        -----
        The weight is then renormalised so the effective L1 penalty
        magnitude is preserved across reweighting iterations.
        """
        denom = np.abs(self._Ir / (self._Ib.mean() + 1e-6)) + self.epsilon
        self._W = 1.0 / denom
        self._W = self._W * self._W.size / self._W.sum()
        self.reweighting_iteration += 1

    def update(self) -> None:
        """Run one reweighted ALM pass and update flat/dark fields.

        Calls :func:`~linum_basic.algorithms.inexact_alm_l1` on the sorted image
        stack, updates the flat-field, dark-field, and reweighting matrix,
        then checks the convergence criterion to decide whether further
        reweighting iterations are needed.
        """
        last_flatfield = self.flatfield.copy()
        last_darkfield = self.darkfield.copy()

        if self.l_s is None or self.l_d is None:
            msg = "l_s and l_d must be set before calling update(); call prepare() first."
            raise RuntimeError(msg)
        result = inexact_alm_l1(
            self.img_sort,
            self.l_s,
            self.l_d,
            weight=self._W,
            estimate_darkfield=self.estimate_darkfield,
            verbose=self.verbose,
            xp=self._xp,
            warm_start=self._alm_state if self.warm_start_reweighting else None,
            return_state=self.warm_start_reweighting,
        )
        Ib, Ir, D, alm_state = result
        if self.warm_start_reweighting:
            self._alm_state = alm_state

        self._Ib = Ib
        self._Ir = Ir
        D_2d = D.reshape(self.working_size, self.working_size)

        self.update_weights()

        self.flatfield = Ib.mean(axis=0) - D_2d
        self.flatfield = self.flatfield / (self.flatfield.mean() + 1e-9)
        self.darkfield = D_2d

        mad_flat = float(np.abs(self.flatfield - last_flatfield).sum() / (np.abs(last_flatfield).sum() + 1e-9))
        mad_dark_abs = float(np.abs(self.darkfield - last_darkfield).sum())
        last_dark_sum = float(np.abs(last_darkfield).sum())
        if mad_dark_abs < 1e-7:
            mad_dark = 0.0
        elif last_dark_sum < 1e-7:
            mad_dark = 1.0  # previous estimate was zero; relative change is undefined, assume not converged
        else:
            mad_dark = mad_dark_abs / last_dark_sum

        if (
            max(mad_flat, mad_dark) <= self.reweighting_tolerance
            or self.reweighting_iteration >= self.max_reweighting_iterations
        ):
            self._flag_reweighting = False

    def run(self) -> None:
        """Run the full BaSiC optimisation loop.

        Calls :meth:`update` iteratively (reweighted ALM) until the
        convergence criterion is met or :attr:`max_reweighting_iterations`
        is reached.  After convergence the flat-field and dark-field are
        up-sampled back to the original image resolution.

        See Also
        --------
        linum_basic.algorithms.inexact_alm_l1 : Inner ALM solver.
        linum_basic.core.BaSiC.normalize : Apply the estimated correction.
        linum_basic.core.BaSiC.write_images : Write corrected images to disk.

        Notes
        -----
        :meth:`prepare` must be called before this method.
        """
        if self.verbose:
            pbar: tqdm | None = tqdm(desc="Reweighting", total=self.max_reweighting_iterations, leave=False)
        else:
            pbar = None
        while self._flag_reweighting:
            self.update()
            if pbar is not None:
                pbar.update()
        if pbar is not None:
            pbar.close()

        # Up-sample to the full image resolution
        h, w = self.image_shape
        self.flatfield_fullsize = cv2.resize(self.flatfield, (w, h), interpolation=cv2.INTER_LINEAR)
        self.flatfield_fullsize = self.flatfield_fullsize / (self.flatfield_fullsize.mean() + 1e-9)
        self.darkfield_fullsize = cv2.resize(self.darkfield, (w, h), interpolation=cv2.INTER_LINEAR)

    def normalize(self, img: NDArray, *, clip: bool = True, epsilon: float = 1e-6) -> NDArray:
        """Apply the estimated shading correction to a single image.

        Computes ``(img - darkfield) / (flatfield + ε)`` and optionally
        clips the result to the valid range of the input dtype.

        Parameters
        ----------
        img : numpy.ndarray
            2-D image to correct.  Must have the same spatial dimensions as
            the images used during fitting.
        clip : bool
            When ``True`` and the input dtype is an integer type, the result
            is clipped to ``[dtype.min, dtype.max]``.
        epsilon : float
            Small constant added to the flat-field to prevent division by
            zero.

        Returns
        -------
        numpy.ndarray
            Corrected image with the same shape and dtype as *img*.
        """
        corrected = (img.astype(np.float32) - self.darkfield_fullsize) / (self.flatfield_fullsize + epsilon)
        if clip and img.dtype not in (np.float32, np.float64):
            info = np.iinfo(img.dtype)
            corrected = np.clip(corrected, info.min, info.max)
        return corrected.astype(img.dtype)

    def write_images(self, directory: str | Path, epsilon: float = 1e-6) -> None:
        """Save shading-corrected images to *directory*.

        Applies :meth:`normalize` to every image in the loaded stack and
        writes the results to disk with the original filename.

        Parameters
        ----------
        directory : str or Path
            Output directory.  Created if it does not exist.
        epsilon : float
            Stability constant forwarded to :meth:`normalize`.

        Notes
        -----
        Output images are written by OpenCV (``cv2.imwrite``).  The format
        is inferred from the output filename extension (inherited from the
        input filenames).
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Vectorised normalisation over the whole stack at once, then
        # parallel write (cv2.imwrite releases the GIL).
        stack_f32 = self.img_stack.astype(np.float32)
        corrected_stack = (stack_f32 - self.darkfield_fullsize) / (self.flatfield_fullsize + epsilon)
        if self.img_stack.dtype not in (np.float32, np.float64):
            info = np.iinfo(self.img_stack.dtype)
            corrected_stack = np.clip(corrected_stack, info.min, info.max)
        corrected_stack = corrected_stack.astype(self.img_stack.dtype)

        def _write_one(i: int) -> None:
            cv2.imwrite(str(out_dir / self.files[i].name), corrected_stack[i])

        with ThreadPoolExecutor(max_workers=min(32, self.n_images)) as pool:
            list(tqdm(pool.map(_write_one, range(self.n_images)), desc="Shading correction", total=self.n_images, leave=False))

    def set_flatfield(self, flatfield: NDArray) -> None:
        """Override the estimated flat-field with a pre-computed one.

        The supplied array is resized to match the loaded image dimensions.

        Parameters
        ----------
        flatfield : numpy.ndarray
            2-D flat-field image.

        Notes
        -----
        The array is transposed (``flatfield.T``) before resizing to match
        OpenCV's column-major coordinate convention, then transposed back.
        The net effect on a symmetric flat-field is invisible; for
        asymmetric profiles this convention must be kept consistent with
        :meth:`get_flatfield`.
        """
        h, w = self.image_shape
        self.flatfield_fullsize = cv2.resize(flatfield.T, (w, h), interpolation=cv2.INTER_LINEAR).T

    def set_darkfield(self, darkfield: NDArray) -> None:
        """Override the estimated dark-field with a pre-computed one.

        The supplied array is resized to match the loaded image dimensions.

        Parameters
        ----------
        darkfield : numpy.ndarray
            2-D dark-field image.

        Notes
        -----
        The array is transposed (``darkfield.T``) before resizing, matching
        the convention of :meth:`set_flatfield`.
        """
        h, w = self.image_shape
        self.darkfield_fullsize = cv2.resize(darkfield.T, (w, h), interpolation=cv2.INTER_LINEAR).T

    def get_flatfield(self) -> NDArray:
        """Return a copy of the full-resolution estimated flat-field.

        Returns
        -------
        numpy.ndarray
            Flat-field array of shape *(H, W)*.
        """
        return self.flatfield_fullsize.copy()

    def get_darkfield(self) -> NDArray:
        """Return a copy of the full-resolution estimated dark-field.

        Returns
        -------
        numpy.ndarray
            Dark-field array of shape *(H, W)*.
        """
        return self.darkfield_fullsize.copy()
