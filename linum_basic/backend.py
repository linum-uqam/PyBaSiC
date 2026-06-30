"""Array-backend abstraction for linum-basic.

Provides a unified namespace for NumPy and PyTorch operations, allowing
the core ALM optimisation loop to run on CPU (NumPy) or GPU (PyTorch)
without code duplication.
"""

from __future__ import annotations

import enum
import math
from typing import Any

import numpy as np
from scipy.fft import dctn as scipy_dctn
from scipy.fft import idctn as scipy_idctn

__all__ = ["ArrayNamespace", "Backend", "get_xp"]


class Backend(enum.StrEnum):
    """Supported compute backends.

    Attributes
    ----------
    NUMPY : str
        Pure NumPy / SciPy on CPU.
    TORCH : str
        PyTorch — CPU or any accelerator (CUDA, MPS, etc.).
    """

    NUMPY = "numpy"
    TORCH = "torch"


class ArrayNamespace:
    """Thin wrapper providing a common API over NumPy and PyTorch.

    Parameters
    ----------
    backend : Backend
        Which backend to use.
    device : str or None
        PyTorch device string (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
        Ignored when *backend* is ``Backend.NUMPY``.

    Raises
    ------
    ImportError
        If ``Backend.TORCH`` is requested but PyTorch is not installed.
    ValueError
        If an unsupported *backend* value is given.
    """

    def __init__(self, backend: Backend, device: str | None = None) -> None:
        self._backend = backend
        self._torch: Any = None
        self._device: Any = None
        if backend is Backend.TORCH:
            try:
                import torch
            except ImportError as exc:
                msg = "PyTorch is required for the 'torch' backend. Install it with: uv sync --extra gpu"
                raise ImportError(msg) from exc
            self._torch = torch
            self._device = torch.device(device or "cpu")
        elif backend is Backend.NUMPY:
            pass
        else:
            msg = f"Unknown backend: {backend!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def asarray(self, x: np.ndarray, dtype: type | None = None) -> Any:
        """Convert a NumPy array to the backend's native type.

        Parameters
        ----------
        x : numpy.ndarray
            Source array.
        dtype : type or None
            Target dtype.  ``None`` preserves the source dtype mapped to the
            closest backend equivalent.

        Returns
        -------
        object
            Native array on the configured device.
        """
        if self._backend is Backend.NUMPY:
            return np.asarray(x, dtype=dtype)
        # Pass-through: already a tensor on the right device with no dtype coercion needed
        if isinstance(x, self._torch.Tensor) and x.device == self._device and dtype is None:
            return x
        # For dtype resolution, always go through numpy to get a numpy dtype
        np_dtype = np.dtype(dtype) if dtype is not None else np.asarray(x).dtype
        torch_dtype = self._numpy_dtype_to_torch(np_dtype)
        return self._torch.as_tensor(np.asarray(x), dtype=torch_dtype, device=self._device)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert a backend array back to a NumPy array.

        Parameters
        ----------
        x : object
            Backend-native array.

        Returns
        -------
        numpy.ndarray
            CPU NumPy copy.
        """
        if self._backend is Backend.NUMPY:
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)
        return x.detach().cpu().numpy()

    def astype(self, x: Any, dtype: type) -> Any:
        """Cast a backend array to *dtype* without a device round-trip.

        Parameters
        ----------
        x : object
            Backend-native array.
        dtype : type
            Target NumPy dtype (e.g. ``np.float32``, ``np.float64``).

        Returns
        -------
        object
            Array with the requested element type, on the same device as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.asarray(x, dtype=dtype)
        return x.to(self._numpy_dtype_to_torch(np.dtype(dtype)))

    def zeros(self, shape: tuple[int, ...], dtype: type | None = None) -> Any:
        """Return a zero-filled array of the given shape.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        dtype : type or None
            Element type.

        Returns
        -------
        object
            Backend-native zero array.
        """
        if self._backend is Backend.NUMPY:
            return np.zeros(shape, dtype=dtype)
        torch_dtype = self._numpy_dtype_to_torch(np.dtype(dtype or np.float32))
        return self._torch.zeros(shape, dtype=torch_dtype, device=self._device)

    def ones(self, shape: tuple[int, ...], dtype: type | None = None) -> Any:
        """Return a ones-filled array of the given shape.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        dtype : type or None
            Element type.

        Returns
        -------
        object
            Backend-native ones array.
        """
        if self._backend is Backend.NUMPY:
            return np.ones(shape, dtype=dtype)
        torch_dtype = self._numpy_dtype_to_torch(np.dtype(dtype or np.float32))
        return self._torch.ones(shape, dtype=torch_dtype, device=self._device)

    def zeros_like(self, x: Any) -> Any:
        """Return a zero array matching the shape and dtype of *x*.

        Parameters
        ----------
        x : Any
            Reference array.

        Returns
        -------
        Any
            Zero array matching *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.zeros_like(x)
        return self._torch.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        """Return a ones array matching the shape and dtype of *x*.

        Parameters
        ----------
        x : Any
            Reference array.

        Returns
        -------
        Any
            Ones array matching *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.ones_like(x)
        return self._torch.ones_like(x)

    # ------------------------------------------------------------------
    # Element-wise math
    # ------------------------------------------------------------------

    def abs(self, x: Any) -> Any:
        """Element-wise absolute value.

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            ``|x|``.
        """
        if self._backend is Backend.NUMPY:
            return np.abs(x)
        return self._torch.abs(x)

    def sign(self, x: Any) -> Any:
        """Element-wise sign (-1, 0, or 1).

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            Sign of *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.sign(x)
        return self._torch.sign(x)

    def maximum(self, x: Any, y: Any) -> Any:
        """Element-wise maximum of *x* and *y*.

        Parameters
        ----------
        x : Any
            First operand.
        y : Any
            Second operand (may be a scalar).

        Returns
        -------
        Any
            ``max(x, y)`` element-wise.
        """
        if self._backend is Backend.NUMPY:
            return np.maximum(x, y)
        if isinstance(y, (int, float)):
            return self._torch.clamp_min(x, y)
        return self._torch.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        """Element-wise minimum of *x* and *y*.

        Parameters
        ----------
        x : Any
            First operand.
        y : Any
            Second operand (may be a scalar).

        Returns
        -------
        Any
            ``min(x, y)`` element-wise.
        """
        if self._backend is Backend.NUMPY:
            return np.minimum(x, y)
        if isinstance(y, (int, float)):
            return self._torch.clamp_max(x, y)
        return self._torch.minimum(x, y)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Element-wise selection from *x* or *y* depending on *condition*.

        Parameters
        ----------
        condition : Any
            Boolean array or scalar.
        x : Any
            Values used where *condition* is ``True``.
        y : Any
            Values used where *condition* is ``False``.

        Returns
        -------
        Any
            Array of selected values on the same device as the inputs.
        """
        if self._backend is Backend.NUMPY:
            return np.where(condition, x, y)
        return self._torch.where(condition, x, y)

    def clone(self, x: Any) -> Any:
        """Return an independent copy of *x*, stripping view metadata.

        For the NumPy backend this is a no-op since arithmetic operations
        already produce new arrays.  For the Torch backend, ``x.clone()``
        strips the ``ADInplaceOrView`` dispatch key that ``reshape`` and
        same-dtype ``.to()`` calls attach to their outputs, preventing
        spurious ``torch._dynamo`` guard failures when the tensor is fed back
        as a compiled-step input on the next iteration.

        Parameters
        ----------
        x : Any
            Array or tensor to clone.

        Returns
        -------
        Any
            An independent tensor with the same values and device as *x*.
        """
        if self._backend is Backend.NUMPY:
            return x
        return x.clone()

    def copysign(self, magnitude: Any, sign_source: Any) -> Any:
        """Element-wise copysign: return *magnitude* with the sign of *sign_source*.

        Parameters
        ----------
        magnitude : Any
            Array of non-negative magnitudes.
        sign_source : Any
            Array from which the sign information is taken.

        Returns
        -------
        Any
            Array with values from *magnitude* and signs from *sign_source*.
        """
        if self._backend is Backend.NUMPY:
            return np.copysign(magnitude, sign_source)
        return self._torch.copysign(magnitude, sign_source)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the arithmetic mean along an axis.

        Parameters
        ----------
        x : Any
            Input array or tensor.
        axis : int or None
            Axis to reduce along.  ``None`` reduces all elements to a scalar.
        keepdims : bool
            If ``True`` the reduced axis is kept as a dimension of size 1.

        Returns
        -------
        Any
            Reduced result on the same backend as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.mean(x, axis=axis, keepdims=keepdims)
        if axis is None:
            return self._torch.mean(x)
        return self._torch.mean(x, dim=axis, keepdim=keepdims)

    def sum(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the sum of array elements along an optional axis.

        Parameters
        ----------
        x : object
            Backend-native array.
        axis : int or None
            Axis along which to reduce.  If ``None``, reduces over all elements.
        keepdims : bool
            If ``True``, the reduced axis is kept as a size-1 dimension.

        Returns
        -------
        object
            Reduced array (or scalar when *axis* is ``None``).
        """
        if self._backend is Backend.NUMPY:
            return np.sum(x, axis=axis, keepdims=keepdims)
        if axis is None:
            return self._torch.sum(x)
        return self._torch.sum(x, dim=axis, keepdim=keepdims)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def norm_fro(self, x: Any) -> float:
        """Compute the Frobenius norm of *x*.

        Parameters
        ----------
        x : Any
            2-D (or flat) array.

        Returns
        -------
        float
            Frobenius norm ``‖x‖_F``.
        """
        if self._backend is Backend.NUMPY:
            return float(np.linalg.norm(x, "fro"))
        return float(self._torch.linalg.norm(x, ord="fro"))

    def norm_fro_batched(self, x: Any) -> Any:
        """Per-batch Frobenius norm along trailing matrix dimensions.

        Parameters
        ----------
        x : Any
            Array with shape ``(Z, …)`` where the last two dimensions form
            the matrix whose norm is computed for each leading batch index.

        Returns
        -------
        Any
            Shape ``(Z,)`` on the active device (Torch) or NumPy array.
        """
        if self._backend is Backend.NUMPY:
            flat = x.reshape(x.shape[0], -1)
            return np.linalg.norm(flat, axis=1)
        return self._torch.linalg.vector_norm(x.reshape(x.shape[0], -1), ord=2, dim=1)

    def min(self, x: Any) -> float:
        """Return the global minimum value of *x* as a Python float.

        Avoids materialising the full tensor to CPU when on GPU backends.

        Parameters
        ----------
        x : object
            Backend-native array.

        Returns
        -------
        float
            Minimum element.
        """
        if self._backend is Backend.NUMPY:
            return float(np.min(x))
        return float(self._torch.min(x))

    def min_along(self, x: Any, axis: int, *, keepdims: bool = False) -> Any:
        """Minimum along *axis*, keeping result on the active device.

        Returns
        -------
        Any
            Reduced array on the same backend as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.min(x, axis=axis, keepdims=keepdims)
        return self._torch.amin(x, dim=axis, keepdim=keepdims)

    def svd_leading_singular(self, x: Any, *, n_iter: int = 10) -> float:
        """Return the largest singular value of *x*.

        Uses power iteration on ``x @ x.T`` for GPU backends to avoid a full
        SVD and the associated device synchronisation.  NumPy falls back to
        ``numpy.linalg.svd(compute_uv=False)``.

        Parameters
        ----------
        x : object
            2-D input matrix.
        n_iter : int
            Power-iteration steps (GPU path only).

        Returns
        -------
        float
            Largest singular value σ₁.
        """
        if self._backend is Backend.NUMPY:
            return float(np.linalg.svd(x, compute_uv=False)[0])
        return float(self._svd_leading_singular_torch(x, n_iter=n_iter))

    def svd_leading_singular_batched(self, x: Any, *, n_iter: int = 10) -> Any:
        """Return the largest singular value for each batch matrix.

        Parameters
        ----------
        x : object
            3-D array ``(Z, n, m)``.
        n_iter : int
            Power-iteration steps.

        Returns
        -------
        object
            Shape ``(Z,)`` tensor/array on the active device.
        """
        if self._backend is Backend.NUMPY:
            return np.array([float(np.linalg.svd(x[z], compute_uv=False)[0]) for z in range(x.shape[0])])
        return self._svd_leading_singular_torch_batched(x, n_iter=n_iter)

    def _svd_leading_singular_torch(self, x: Any, *, n_iter: int) -> Any:
        """Power iteration for σ₁ of a single 2-D matrix (returns Python float)."""
        sigma = self._svd_leading_singular_torch_batched(x.unsqueeze(0), n_iter=n_iter)
        return float(sigma[0])

    def _svd_leading_singular_torch_batched(self, x: Any, *, n_iter: int) -> Any:
        """Power iteration for σ₁ of batched matrices ``(Z, n, m)``."""
        torch = self._torch
        z, _n, m = x.shape
        v = torch.randn(z, m, 1, dtype=x.dtype, device=x.device)
        v = v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-9)
        for _ in range(n_iter):
            u = torch.bmm(x, v)
            u = u / (torch.linalg.norm(u, dim=1, keepdim=True) + 1e-9)
            v = torch.bmm(x.transpose(1, 2), u)
            v = v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-9)
        return torch.linalg.norm(torch.bmm(x, v), dim=(1, 2))

    def inference_mode(self) -> Any:
        """Return a context manager that disables gradient tracking.

        On the Torch backend this returns ``torch.inference_mode()``.
        On the NumPy backend it returns a no-op context manager so the
        ALM loop can unconditionally use ``with xp.inference_mode()``.

        Returns
        -------
        contextlib.AbstractContextManager
            Context manager that disables gradient computation (Torch) or
            is a no-op (NumPy).
        """
        if self._backend is Backend.TORCH:
            return self._torch.inference_mode()
        import contextlib

        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # DCT
    # ------------------------------------------------------------------

    def dctn(self, x: Any, norm: str = "ortho") -> Any:
        """N-dimensional orthonormal DCT-II.

        Parameters
        ----------
        x : object
            Input array.
        norm : str
            Normalisation mode.  Only ``"ortho"`` is supported.

        Returns
        -------
        object
            DCT-II coefficients with the same shape as *x*.

        Notes
        -----
        The Torch backend uses an FFT-based implementation that matches
        SciPy's ``scipy.fft.dctn`` output to within floating-point precision.
        """
        if self._backend is Backend.NUMPY or isinstance(x, np.ndarray):
            return scipy_dctn(x, norm=norm)
        return _torch_dctn(x, norm=norm)

    def idctn(self, x: Any, norm: str = "ortho") -> Any:
        """N-dimensional orthonormal inverse DCT-II (DCT-III).

        Parameters
        ----------
        x : object
            DCT coefficient array.
        norm : str
            Normalisation mode.  Only ``"ortho"`` is supported.

        Returns
        -------
        object
            Reconstructed array with the same shape as *x*.
        """
        if self._backend is Backend.NUMPY or isinstance(x, np.ndarray):
            return scipy_idctn(x, norm=norm)
        return _torch_idctn(x, norm=norm)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _numpy_dtype_to_torch(self, dtype: np.dtype) -> object:
        """Map a NumPy dtype to the corresponding PyTorch dtype.

        Parameters
        ----------
        dtype : numpy.dtype
            Source dtype.

        Returns
        -------
        torch.dtype
            Closest PyTorch equivalent.  Defaults to ``float32`` for unknown
            types.

        Notes
        -----
        Supported mappings:

        ============== ====================
        NumPy dtype    PyTorch dtype
        ============== ====================
        ``float32``    ``torch.float32``
        ``float64``    ``torch.float64``
        ``int32``      ``torch.int32``
        ``int64``      ``torch.int64``
        ``bool_``      ``torch.bool``
        *other*        ``torch.float32``
        ============== ====================
        """
        _map = {
            np.float32: self._torch.float32,
            np.float64: self._torch.float64,
            np.int32: self._torch.int32,
            np.int64: self._torch.int64,
            np.bool_: self._torch.bool,
        }
        return _map.get(dtype.type, self._torch.float32)


# ---------------------------------------------------------------------------
# PyTorch DCT-II / DCT-III via FFT  (Lee 1984 / Makhoul 1980)
# ---------------------------------------------------------------------------

# Per-(length, dtype, device) cache of precomputed twiddle factors and scale
# vectors.  Populated lazily on first use; avoids O(n) trig recomputation on
# every DCT call inside the hot ALM loop.
_DCT_TWIDDLE_CACHE: dict[tuple, tuple] = {}
_IDCT_TWIDDLE_CACHE: dict[tuple, tuple] = {}

# Per-(length, device) cache of orthonormal DCT-II matrices for matmul-based
# DCT.  Used by _build_alm_step to replace FFT-based DCT inside torch.compile
# regions (Torchinductor cannot generate Triton code for complex ops).
_DCT_MATRIX_CACHE: dict[tuple, Any] = {}


def _get_dct_matrix(n: int, device: Any) -> Any:
    """Return a cached (n, n) orthonormal DCT-II matrix on *device*.

    The matrix ``A`` satisfies ``y = A @ x`` for the length-*n* DCT-II
    with ``norm="ortho"``.  It is built in float64 for accuracy and cached
    per ``(n, device)``.  Callers cast to float32 via ``.float()`` as needed.

    Parameters
    ----------
    n : int
        Length of the DCT.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor, shape (n, n), dtype float64
        Orthonormal DCT-II matrix on *device*.
    """
    import torch

    key = (n, str(device))
    if key not in _DCT_MATRIX_CACHE:
        k = torch.arange(n, dtype=torch.float64, device=device)
        i = torch.arange(n, dtype=torch.float64, device=device)
        A = torch.cos(math.pi * (i[None, :] + 0.5) * k[:, None] / n)
        A[0] *= 1.0 / math.sqrt(n)
        A[1:] *= math.sqrt(2.0 / n)
        _DCT_MATRIX_CACHE[key] = A
    return _DCT_MATRIX_CACHE[key]


def _get_dct_twiddles(n: int, dtype: Any, device: Any) -> tuple:
    """Return cached (cos_k, sin_k, ortho_scale) for a forward DCT of length *n*."""
    import torch

    key = (n, dtype, str(device))
    if key not in _DCT_TWIDDLE_CACHE:
        k = torch.arange(n, dtype=dtype, device=device)
        theta = math.pi * k / (2.0 * n)
        cos_k = torch.cos(theta)
        sin_k = torch.sin(theta)
        scale = torch.empty(n, dtype=dtype, device=device)
        scale[0] = 1.0 / math.sqrt(n)
        scale[1:] = 1.0 / math.sqrt(n / 2)
        _DCT_TWIDDLE_CACHE[key] = (cos_k, sin_k, scale)
    return _DCT_TWIDDLE_CACHE[key]


def _get_idct_twiddles(n: int, dtype: Any, device: Any) -> tuple:
    """Return cached (cos_k, sin_k, ortho_scale) for an inverse DCT of length *n*."""
    import torch

    key = (n, dtype, str(device))
    if key not in _IDCT_TWIDDLE_CACHE:
        k = torch.arange(n, dtype=dtype, device=device)
        cos_k = torch.cos(math.pi * k / (2.0 * n))
        sin_k = torch.sin(math.pi * k / (2.0 * n))
        scale = torch.empty(n, dtype=dtype, device=device)
        scale[0] = math.sqrt(n)
        scale[1:] = math.sqrt(n / 2)
        _IDCT_TWIDDLE_CACHE[key] = (cos_k, sin_k, scale)
    return _IDCT_TWIDDLE_CACHE[key]


def _torch_dct1d(x: Any, norm: str = "ortho") -> Any:
    """Orthonormal 1-D DCT-II along the last axis.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        DCT-II coefficients along the last axis.

    Notes
    -----
    Algorithm reorders the input (even/odd interleaving), applies a real
    FFT, then multiplies by precomputed twiddle factors to obtain the
    DCT-II spectrum.  Twiddle factors are cached per (length, dtype, device)
    to avoid recomputation on every call.
    """
    import torch

    n = x.shape[-1]
    cos_k, sin_k, scale = _get_dct_twiddles(n, x.dtype, x.device)
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    Vc = torch.fft.fft(v, n=n, dim=-1)
    # Re(Vc * exp(-i*theta)) = Vc.real*cos + Vc.imag*sin
    y = Vc.real * cos_k + Vc.imag * sin_k
    if norm == "ortho":
        return y * scale
    return 2.0 * y


def _torch_idct1d(x: Any, norm: str = "ortho") -> Any:
    """Orthonormal 1-D DCT-III (inverse DCT-II) along the last axis.

    Parameters
    ----------
    x : torch.Tensor
        DCT coefficient tensor.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        Reconstructed values along the last axis.
    """
    import torch

    n = x.shape[-1]
    cos_k, sin_k, scale = _get_idct_twiddles(n, x.dtype, x.device)
    xn = x * scale if norm == "ortho" else x / 2
    # Anti-Hermitian imaginary part (exploits real-signal symmetry of forward FFT)
    Vt_i = torch.cat([torch.zeros_like(xn[..., :1]), -xn[..., 1:].flip(-1)], dim=-1)
    V_r = xn * cos_k - Vt_i * sin_k
    V_i = xn * sin_k + Vt_i * cos_k
    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, n=n, dim=-1).real
    y = torch.zeros_like(v)
    y[..., ::2] = v[..., : math.ceil(n / 2)]
    y[..., 1::2] = v[..., math.ceil(n / 2) :].flip(-1)
    return y


def _torch_dctn(x: Any, norm: str = "ortho") -> Any:
    """N-dimensional orthonormal DCT-II (applied to each axis sequentially).

    For the common 2-D case the two passes are fused into a single
    transpose-free contiguous batch to reduce kernel launch overhead.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary rank.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        DCT-II coefficients, same shape as *x*.
    """
    if x.ndim == 2:
        # Axis 1 (last): DCT along rows.
        y = _torch_dct1d(x, norm=norm)
        # Axis 0: transpose so axis 0 becomes the last axis, apply DCT, transpose back.
        return _torch_dct1d(y.t().contiguous(), norm=norm).t().contiguous()
    y = x
    for i in range(y.ndim):
        y = _torch_dct1d(y.transpose(-1, i).contiguous(), norm=norm).transpose(-1, i)
    return y


def _torch_idctn(x: Any, norm: str = "ortho") -> Any:
    """N-dimensional orthonormal inverse DCT-II (DCT-III).

    For the common 2-D case the two passes are fused into a single
    transpose-free contiguous batch.

    Parameters
    ----------
    x : torch.Tensor
        DCT coefficient tensor of arbitrary rank.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        Reconstructed tensor, same shape as *x*.
    """
    if x.ndim == 2:
        y = _torch_idct1d(x, norm=norm)
        return _torch_idct1d(y.t().contiguous(), norm=norm).t().contiguous()
    y = x
    for i in range(y.ndim):
        y = _torch_idct1d(y.transpose(-1, i).contiguous(), norm=norm).transpose(-1, i)
    return y


def get_xp(backend: str | Backend, device: str | None = None) -> ArrayNamespace:
    """Construct an :class:`ArrayNamespace` for the requested backend.

    Parameters
    ----------
    backend : str or Backend
        ``"numpy"``, ``"torch"``, or ``"auto"``.  ``"auto"`` selects the
        Torch backend with CUDA or MPS when available, otherwise falls back
        to NumPy.
    device : str or None
        PyTorch device string (e.g. ``"cuda:0"``).  Ignored for NumPy.

    Returns
    -------
    ArrayNamespace
        Configured array namespace ready for use in the ALM loop.

    Examples
    --------
    >>> xp = get_xp("numpy")
    >>> arr = xp.zeros((4, 4))

    >>> xp_gpu = get_xp("auto")  # picks GPU if available
    """
    if isinstance(backend, str) and backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return ArrayNamespace(Backend.TORCH, device or "cuda")
            # MPS is intentionally excluded: svdvals and float64 are not supported.
        except ImportError:
            pass
        return ArrayNamespace(Backend.NUMPY)

    b = Backend(backend) if isinstance(backend, str) else backend
    if b is Backend.TORCH:
        resolved_device = device or "cpu"
        if resolved_device.startswith("mps"):
            msg = (
                "MPS is not supported as a backend for linum-basic: PyTorch MPS lacks "
                "float64 and svdvals, both required by the ALM solver. "
                "Use backend='torch' with device='cuda' or backend='numpy' instead."
            )
            raise NotImplementedError(msg)
    return ArrayNamespace(b, device)
