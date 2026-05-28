"""Array-backend abstraction for PyBaSiC.

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
        torch_dtype = self._numpy_dtype_to_torch(x.dtype if dtype is None else np.dtype(dtype))
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
            return np.asarray(x)
        return x.detach().cpu().numpy()

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

    def svd_leading_singular(self, x: Any) -> float:
        """Return the largest singular value of *x*.

        Only the leading singular value is computed to avoid the cost of a
        full SVD decomposition.

        Parameters
        ----------
        x : object
            2-D input matrix.

        Returns
        -------
        float
            Largest singular value σ₁.
        """
        if self._backend is Backend.NUMPY:
            return float(np.linalg.svd(x, compute_uv=False)[0])
        return float(self._torch.linalg.svdvals(x)[0])

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
    FFT, then multiplies by twiddle factors to obtain the DCT-II spectrum.
    """
    import torch

    n = x.shape[-1]  # type: ignore[union-attr]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)  # type: ignore[index]
    Vc = torch.fft.fft(v, n=n, dim=-1)
    k = torch.arange(n, dtype=torch.float64, device=x.device)  # type: ignore[union-attr]
    theta = math.pi * k / (2.0 * n)
    cos_k = torch.cos(theta).to(Vc.real.dtype)
    sin_k = torch.sin(theta).to(Vc.real.dtype)
    # Re(Vc * exp(-i*theta)) = Vc.real*cos + Vc.imag*sin
    y = Vc.real * cos_k + Vc.imag * sin_k
    if norm == "ortho":
        y = y.clone()
        y[..., 0] = y[..., 0] / math.sqrt(n)
        y[..., 1:] = y[..., 1:] / math.sqrt(n / 2)
    else:
        y = 2.0 * y
    return y


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

    n = x.shape[-1]  # type: ignore[union-attr]
    xn = x.clone()  # type: ignore[union-attr]
    if norm == "ortho":
        xn[..., 0] = xn[..., 0] * math.sqrt(n)
        xn[..., 1:] = xn[..., 1:] * math.sqrt(n / 2)
    else:
        xn = xn / 2

    k = torch.arange(n, dtype=torch.float64, device=x.device)  # type: ignore[union-attr]
    cos_k = torch.cos(math.pi * k / (2.0 * n)).to(xn.dtype)
    sin_k = torch.sin(math.pi * k / (2.0 * n)).to(xn.dtype)
    # Anti-Hermitian imaginary part (exploits real-signal symmetry of forward FFT)
    Vt_i = torch.cat([xn[..., :1] * 0, -xn[..., 1:].flip(-1)], dim=-1)
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
    y = x
    for i in range(y.ndim):  # type: ignore[union-attr]
        y = _torch_dct1d(y.transpose(-1, i), norm=norm).transpose(-1, i)  # type: ignore[union-attr]
    return y


def _torch_idctn(x: Any, norm: str = "ortho") -> Any:
    """N-dimensional orthonormal inverse DCT-II (DCT-III).

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
    y = x
    for i in range(y.ndim):  # type: ignore[union-attr]
        y = _torch_idct1d(y.transpose(-1, i), norm=norm).transpose(-1, i)  # type: ignore[union-attr]
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
            if torch.backends.mps.is_available():
                return ArrayNamespace(Backend.TORCH, device or "mps")
        except ImportError:
            pass
        return ArrayNamespace(Backend.NUMPY)

    b = Backend(backend) if isinstance(backend, str) else backend
    return ArrayNamespace(b, device)
