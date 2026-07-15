(streaming)=

# Streaming Zarr Processing

Large microscopy mosaics — a grid of tiles acquired over many focal planes —
can dwarf the available RAM long before the science starts. A single
OME-Zarr volume easily runs to tens of gigabytes once it is tiled across z,
and the default `fit` path materialises the whole per-z tile stack into
memory before running the solver.

**Streaming** is a read-path optimisation that bounds peak memory to a
single focal plane. It does not change the correction math: the estimated
flat-fields and dark-fields are *identical* to what the non-streaming path
produces. It only changes *when* the data is read — one plane at a time,
straight from the on-disk Zarr store, instead of all up front.

```{note}
Streaming is an **MVP**. It covers the most common large-volume case — a
per-z mosaic fit that would otherwise exceed memory — and leaves the rest
of the pipeline eager. See {ref}`streaming-scope-and-limits` for the
explicit boundaries.
```

---

(streaming-when-to-use)=

## When to use it

Reach for streaming when the full mosaic volume is the bottleneck:

- **Peak memory exceeds available RAM.** The eager path holds every per-z
  tile stack live for the duration of the fit; streaming releases each
  plane as soon as its solve finishes.
- **You are on a memory-limited host** (a laptop, a shared login node, or a
  container with a tight cgroup limit) and only need the corrected output
  on disk.

You do **not** need streaming for small volumes, single-tile stacks, or
the {doc}`tuning` and {doc}`gpu` paths — those already fit comfortably in
memory, and streaming adds no speed benefit. Streaming is strictly a
memory trade; it runs sequentially and is often *slower* in wall time than
the parallel {doc}`parallelism` strategies.

---

## Quick start

### Command line

The lowest peak memory comes from combining the two flags — `--lazy` reads
the input volume lazily, and `--streaming` fits one plane at a time:

```bash
basic fit --input subject.ome.zarr --output corrected.ome.zarr \
    --streaming --lazy --verbose
```

Both flags default to off, so every existing invocation behaves exactly as
before.

### Python

The library API mirrors the flags one-to-one. Build the grid lazily and
pass `streaming=True` to `fit_mosaic`:

```python
from linum_basic import fit_mosaic, apply_fit, save_corrected
from linum_basic.mosaic import MosaicGrid

# lazy=True keeps the backing volume on disk (a zarr.Array handle, not an ndarray)
mosaic = MosaicGrid.from_ome_zarr("subject.ome.zarr", overlap_fraction=0.2, lazy=True)

# streaming=True fits z-levels one at a time; numerics are bit-identical
fit = fit_mosaic(mosaic, streaming=True, strategy="sequential", verbose=True)

save_corrected(mosaic, fit, "corrected.ome.zarr", input_path="subject.ome.zarr", overwrite=True)
```

At the lowest level, {func}`linum_basic.io.zarr.load_ome_zarr` takes the
same `lazy=True` switch and returns the level-0 `zarr.Array` handle instead
of a materialised array — useful if you build a {class}`~linum_basic.mosaic.MosaicGrid`
from a custom source.

---

## How it works

Streaming is two cooperating pieces on the read path:

1. **Lazy loading** (`lazy=True`). Instead of reading the whole OME-Zarr
   array into a NumPy array, the loader hands back the level-0
   `zarr.Array`. The {class}`~linum_basic.mosaic.MosaicGrid` stores that
   handle as its backing array and infers tile geometry from the array's
   shape and chunk layout. Individual tiles are only read from disk when a
   z-level is actually fitted, and each read pulls just the slice it needs.

2. **One-plane-at-a-time fitting** (`streaming=True`). `fit_mosaic`
   iterates over z-levels and extracts a single plane's tiles, runs the
   full BaSiC solve on that plane, stores the resulting fields, and drops
   the tile data before moving on. Peak memory therefore tracks a single
   focal plane rather than the entire z-stack.

Crucially, the solver that runs on each plane is the *exact same* routine
the non-streaming sequential path uses — same tiles, same order, same
iterations. The only difference is extraction timing. This is why the
output is guaranteed identical, not merely close (see {ref}`streaming-numerics`).

---

(streaming-scope-and-limits)=

## Scope and limits

This is an MVP. The boundaries below are deliberate, not accidents:

| Concern | Status | Detail |
|---|---|---|
| **Read path** | ✅ Streamed | Plane-by-plane reads from a lazy Zarr store. |
| **Write path** | ❌ Not streamed | Applying the fit and writing the corrected volume still needs the full volume; `save_corrected` / `apply_fit` remain eager. Streaming lowers the *fit's* peak memory, not the correction write-out's. |
| **Strategies** | Sequential only | `streaming=True` works with `strategy="auto"` or `"sequential"`. Passing `strategy="multi"` or `"batched"` together with streaming raises {exc}`ValueError`. |
| **Numerics** | Bit-identical | Identical to the non-streaming sequential path. See {ref}`streaming-numerics`. |
| **Scale** | RAM-bounded, not TB-grade | This bounds peak memory to one plane; it is not a terabyte-scale redesign and adds no disk format or indexing changes. |

```{warning}
Because the write path stays eager, the *overall* process can still need a
full volume's worth of memory at correction time. Streaming makes large
**fits** feasible on modest hosts; it does not by itself make large
**correction write-outs** feasible. If your bottleneck is writing the
corrected output, that is out of MVP scope — and that boundary is
deliberate, not unaddressed. The write path was evaluated against
production-scale peak-memory evidence (M007/S01) and explicitly deferred as
*not yet justified*: at measured scale the corrected output is only a few
hundred KB and the eager write adds no memory pressure. See
`scripts/experiments/m007_s02_artifacts/S02-DECISION.md` for the terminal
go/no-go audit and the exact, falsifiable condition under which the write
path would be revisited.
```

```{note}
The **Scale** boundary — RAM-bounded rather than terabyte-grade — is
likewise a deliberate, *evaluated* deferral, not an open gap. A
terabyte-scale distributed/cluster redesign (a coordination layer and
multi-host worker model) was assessed against the same production-scale
peak-memory evidence (M007/S03): the largest available real per-z mosaic
volume peaks at 2.40 GB host RSS — ~416x below a terabyte and ~4.66% of a
single A6000's 48 GB budget — and the codebase is single-host only (joblib
processes and single-host CUDA multi-GPU fan-out), so a cluster layer would
unlock work that does not otherwise fail. The redesign was re-deferred as
*not justified*. See `scripts/experiments/m007_s03_artifacts/S03-DECISION.md`
for the terminal go/no-go audit and the exact, falsifiable condition under
which it would be revisited.
```

---

(streaming-numerics)=

## Numerics: identical, not approximate

A common worry with "streaming" or "lazy" modes is that they quietly
change the result. They do not here. The streaming path reuses the
non-streaming sequential path's solver component unchanged — the same
Augmented-Lagrangian iterations, the same reweighting, the same convergence
stopping rule, on the same tiles in the same order. Only the *timing* of
tile extraction differs (one plane at a time vs all up front).

The test suite asserts this directly: the flat-fields and dark-fields from
`fit_mosaic(streaming=True)` are compared to the non-streaming sequential
result with an exact array-equality check, not a tolerance. If you ever
suspect a divergence, run the two paths on the same input and compare the
output arrays — they should match exactly.

The algorithm invariants documented in {doc}`algorithm` are untouched by
streaming; no solver internals, DCT kernels, or precision flags change.

---

## Choosing streaming vs other strategies

| If you need… | Use |
|---|---|
| Lowest peak memory, any wall-time cost | `--streaming --lazy` |
| Fastest wall time on multiple GPUs | `--strategy multi` (see {doc}`parallelism`) |
| Fastest on a single CUDA device | `--strategy auto` / `batched` (see {doc}`gpu`) |
| Default, balanced behaviour | (no flags) |

Streaming trades wall time for memory. It is strictly sequential, so it
forgoes the multi-GPU fan-out of `strategy="multi"` and the batched CUDA
solve of `strategy="batched"`. Pick it when memory is the binding
constraint, not when speed is.

---

(streaming-troubleshooting)=

## Troubleshooting

**`ValueError: streaming=True is incompatible with strategy='multi'; streaming is sequential only.`**
Streaming is sequential-only. Drop the `--strategy multi` / `--strategy
batched` flag (or set `--strategy sequential` explicitly), or stop using
`--streaming`. The error names the offending strategy and suggests
`'auto'` or `'sequential'`.

**No memory reduction observed.**
Confirm both `--streaming` *and* `--lazy` are set. Without `--lazy`, the
grid still loads the full volume eagerly before the streaming fit begins,
defeating the point. The lazy switch is what keeps the backing volume on
disk.

**Slower than expected.**
Expected. Streaming is sequential and trades wall time for memory; it will
be slower than the parallel strategies. If memory is not your constraint,
prefer {doc}`parallelism` or {doc}`gpu`.

**Apple Silicon / MPS.**
Streaming is backend-agnostic on the read path and runs fine on NumPy. On
Apple Silicon, the GPU (MPS) backend is unsupported for BaSiC fitting in
any case — see {doc}`gpu`. Streaming on a Mac simply uses the NumPy solver
plane by plane.
