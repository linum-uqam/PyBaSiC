(algorithm)=
# Algorithm

## Overview

BaSiC (*Background and Shading Correction*) models the observed image
intensity $D_i$ at pixel $p$ for frame $i$ as:

$$
D_i(p) = S(p) \cdot I_i(p) + B(p)
$$

where:

- $S(p)$ is the **flat-field** — a smooth, spatially varying gain that
  captures optical vignetting and detector non-uniformity.
- $B(p)$ is the **dark-field** — a slowly varying additive offset due to
  fluorescence from optical components or camera dark current.
- $I_i(p)$ is the *true* scene intensity to recover.

The goal is to estimate $S$ and $B$ from a collection of images **without**
knowledge of the true intensities $I_i$.

The figure below illustrates the model in practice.  A synthetic Gaussian
vignette $S(p)$ darkens the image towards the corners.  Linum BaSiC recovers
$S$ by exploiting the fact that the flat-field varies **smoothly** across
pixels and is therefore **sparse in the DCT domain**.

```{figure} _static/demo/demo_comparison.png
:alt: Three-panel demo of BaSiC correction
:align: center
:width: 100%

**Left:** observed (corrupted) tile.  **Centre:** flat-field $S$ estimated by
Linum BaSiC — a smooth, centralised gain map recovered from a stack of 176 such
tiles.  **Right:** corrected tile $\hat{I} = (D - B) / S$.
```

---

## Matrix formulation

Stack all $N$ images as rows of a matrix $\mathbf{D} \in \mathbb{R}^{N \times P}$
(where $P = H \times W$ is the number of pixels). The model becomes:

$$
\mathbf{D} = \mathbf{b} \cdot S^\top + \mathbf{E} + \mathbf{1} \otimes B^\top
$$

where $\mathbf{b} \in \mathbb{R}^N$ is a per-image baseline (intensity
normalisation scalar) and $\mathbf{E}$ is the sparse residual matrix.

The key insight of Peng et al. (2017) is that **the flat-field is smooth**
and therefore sparse in the DCT domain. The optimisation problem is:

$$
\min_{S, \mathbf{b}, B, \mathbf{E}} \;
\lambda_s \|\tilde{S}\|_1 + \|\mathbf{W} \odot \mathbf{E}\|_1
\quad \text{s.t.} \quad
\mathbf{D} = \text{repmat}(S \odot \mathbf{b}) + \mathbf{E} + \mathbf{1} \otimes B^\top
$$

Here $\tilde{S}$ denotes the DCT-II coefficients of the flat-field, $\odot$
is element-wise multiplication, and $\mathbf{W}$ is a per-entry weight
matrix (updated during reweighting; see below).

---

## Augmented Lagrangian solver

The constrained problem is solved via the **Inexact Augmented Lagrangian
Method (ALM / ADMM)** implemented in
{func}`~linum_basic.algorithms.inexact_alm_l1`

At each ALM iteration the following sub-problems are solved in closed form:

1. **Flat-field update** — DCT-domain soft-thresholding (shrinkage) at
   threshold $\lambda_s / \mu$ where $\mu$ is the current Lagrange
   multiplier scale.
2. **Residual update** — pixel-wise soft-thresholding of the weighted
   residual matrix.
3. **Baseline update** — mean projection along the pixel axis.
4. **Dark-field update** (optional) — soft-thresholding in pixel space with
   regularisation $\lambda_d$.
5. **Multiplier update** — gradient ascent step scaled by $\mu$.
6. **$\mu$ update** — $\mu \leftarrow \rho \cdot \mu$ where $\rho = 1.5$
   by default.

Convergence is measured by the relative Frobenius-norm residual:

$$
\frac{\|\mathbf{D} - \text{repmat}(S \odot \mathbf{b}) - \mathbf{E}\|_F}
     {\|\mathbf{D}\|_F} \leq \text{tol}
$$

The dark-field $B$ is initialised to **zero** (not random noise) so that
the very first ALM iterate has a well-defined baseline.  The per-iteration
convergence check for the dark-field uses a *relative* mean-absolute
deviation:

$$
\text{mad\_dark} = \frac{\|B^{(k)} - B^{(k-1)}\|_1}{\|B^{(k-1)}\|_1}
$$

When $\|B^{(k-1)}\|_1 = 0$ (i.e. the previous iterate was still all-zero),
the relative change is undefined.  In that case the solver conservatively
treats the dark-field as *not yet converged* (sets $\text{mad\_dark} = 1$)
rather than clamping the denominator to a small epsilon, which would
artificially report convergence and freeze the dark-field at zero.

---

## Reweighted L1

After the ALM loop converges, Linum BaSiC applies **reweighted L1
regularisation** (Candès et al., 2008) to promote sparsity of the
residual more aggressively than plain L1:

$$
W_{i,p}^{(k+1)} =
\frac{1}{\left| E_{i,p}^{(k)} / (\text{mean}(B^{(k)}) + \varepsilon) \right| + \varepsilon}
$$

The weights are renormalised so that $\text{mean}(\mathbf{W}) = 1$.
The outer loop continues until the flatfield change is below
{attr}`~linum_basic.core.BaSiC.reweighting_tolerance` or
{attr}`~linum_basic.core.BaSiC.max_reweighting_iterations` is reached.

---

## Full execution flow

```{mermaid}
flowchart TD
    A[BaSiC.__init__] --> B[prepare]
    B --> B1[Load & resize images]
    B1 --> B2[Compute mean DCT]
    B2 --> B3[Auto-tune l_s = dct_sum / 800\nl_d = dct_sum / 2000]
    B3 --> B4[Sort image stack]
    B4 --> C[run]
    C --> D[update loop]
    D --> D1[inexact_alm_l1\nALM solver]
    D1 --> D2[update_weights\nreweighted-L1 update]
    D2 --> D3{converged?}
    D3 -- no --> D1
    D3 -- yes --> E[Up-sample flatfield & darkfield]
    E --> F[normalize / write_images]
```

---

## Regularisation auto-tuning

{meth}`~linum_basic.core.BaSiC.prepare` computes the L1 norm of the DCT
coefficients of the normalised pixel-mean image and sets:

$$
\lambda_s = \frac{\|\tilde{\bar{D}}\|_1}{800}, \quad
\lambda_d = \frac{\|\tilde{\bar{D}}\|_1}{2000}
$$

These heuristics originate from the original MATLAB implementation.  They
work well for typical fluorescence stacks but can be tuned manually; see
{ref}`Parameter Tuning <parameters>`.

---

## References

1. Peng, T. *et al.* "A BaSiC tool for background and shading correction
   of optical microscopy images." *Nat. Commun.* **8**, 14836 (2017).
   <https://doi.org/10.1038/ncomms14836>
2. Lin, Z. *et al.* "The Augmented Lagrange Multiplier Method for Exact
   Recovery of Corrupted Low-Rank Matrices." *arXiv* 1009.5055 (2010).
3. Candès, E. J. *et al.* "Enhancing Sparsity by Reweighted ℓ1
   Minimization." *J. Fourier Anal. Appl.* **14**, 877-905 (2008).
