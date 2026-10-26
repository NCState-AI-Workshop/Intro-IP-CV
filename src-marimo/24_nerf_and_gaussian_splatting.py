# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    EXTERNAL_FIGURES_DIR = IMAGES_DIR / "external_figures"
    return (EXTERNAL_FIGURES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # NeRF and 3D Gaussian Splatting: An Overview

    Notebook 23 produced exactly two things from a set of photos: a
    **bundle-adjusted camera pose per image**, and a **sparse 3D point cloud**
    (the by-product of structure from motion). Two classical paths forward
    from there already live in Szeliski's book: **multi-view stereo** (§12.7)
    turns those poses into *dense* per-view depth, and **Chapter 13's 3D
    reconstruction** fuses many depth maps into an explicit mesh, point
    cloud, or volumetric model you can render with ordinary graphics
    rasterization.

    This notebook covers a third path -- one the book calls **neural
    rendering** (§14.6) -- that skips building an explicit 3D model
    altogether: optimize a scene *representation* directly against the input
    photos themselves, through a rendering process that is differentiable
    end to end. **NeRF** (Mildenhall, Srinivasan et al. 2020) is the book's
    own worked example of this idea; **3D Gaussian Splatting** (Kerbl,
    Kopanas et al. 2023) is a newer, much faster alternative that doesn't
    appear in this book's 2021 draft at all -- the first time this course
    has had to go beyond the textbook's own citations. This is a conceptual
    overview with figures from both original papers (cited throughout,
    reproduced for educational use) -- no from-scratch implementation here;
    a follow-up session will cover hands-on use of Nerfstudio directly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The problem: novel view synthesis

    Given a set of photos of a static scene, each with a known camera pose
    (exactly notebook 23's output), **render a brand-new viewpoint that was
    never photographed.** Szeliski's Chapter 14 already surveys several
    classical answers: **view interpolation** with view-dependent texture
    maps (§14.1), **layered depth images** (§14.2), and **light fields /
    Lumigraphs** (§14.3), which densely resample rays rather than building
    any explicit geometry at all. NeRF and Gaussian Splatting both belong to
    a newer family that the book's own §14.6 introduces as **neural
    rendering**: instead of hand-designed interpolation or resampling rules,
    *optimize* a scene representation so that rendering it from each known
    training pose reproduces that training photo -- and trust that a
    representation which explains every training view from its own camera
    also renders sensible *new* views in between.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. NeRF: a scene as a continuous neural field

    NeRF represents a scene as a single continuous function of **position
    and viewing direction**: feed it a 5D coordinate $(x,y,z,\theta,\phi)$,
    and it returns a volume density $\sigma$ (how opaque space is there) and
    a view-dependent emitted color $\mathbf{c}=(r,g,b)$. The function itself
    is just a multilayer perceptron, $F_\Theta:(\mathbf{x},\mathbf{d})\to(\mathbf{c},\sigma)$
    -- 8 fully-connected layers taking only position as input (so density
    stays multiview-consistent), outputting $\sigma$ and a feature vector
    that is then combined with viewing direction through one more layer to
    produce color (so highlights and reflections *can* change with
    viewpoint). There is no mesh, no voxel grid, no point cloud -- just
    network weights.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(8, 2.2))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "nerf_fig1_pipeline_teaser.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 1 from Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, "
            "and Ng, \"NeRF: Representing Scenes as Neural Radiance Fields for "
            "View Synthesis,\" ECCV 2020 (arXiv:2003.08934), reproduced for "
            "educational use. 100 input photos of a synthetic drum kit, randomly "
            "captured on a surrounding hemisphere, optimize one NeRF -- which can "
            "then be rendered from any new viewpoint.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Rendering a ray: volume rendering, not ray-surface intersection

    NeRF never asks "where does this ray hit a surface?" Instead, it borrows
    **classical volume rendering** (§4 of the paper) wholesale: march along
    the ray, treat $\sigma(\mathbf{x})$ as a differential probability of the
    ray terminating at each point, and integrate color weighted by how much
    light actually survives to reach that point:

    $$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt,
       \quad T(t)=\exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right). \qquad (\text{Eq. 1}) $$

    $T(t)$ is accumulated **transmittance** -- the probability the ray
    survives, un-absorbed, from the near bound $t_n$ out to $t$. In practice
    the integral is estimated by **stratified sampling** (one random sample
    per bin along the ray, so the MLP is still queried at genuinely
    continuous positions over the course of training, Eq. 2), then combined
    via a discrete quadrature rule that turns out to be exactly classical
    **alpha compositing**:

    $$ \hat C(\mathbf{r}) = \sum_{i=1}^N T_i\big(1-\exp(-\sigma_i\delta_i)\big)\,\mathbf{c}_i,
       \quad T_i=\exp\!\left(-\sum_{j<i}\sigma_j\delta_j\right). \qquad (\text{Eq. 3}) $$

    Every step of this -- march, query the MLP, composite -- is a plain
    differentiable computation graph, so the whole pipeline (panels a-d
    below) trains with ordinary backpropagation against a simple squared
    pixel-color loss, no 3D supervision needed at all.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(8, 2.6))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "nerf_fig2_pipeline_detail.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 2 from Mildenhall et al. 2020, reproduced for educational "
            "use. (a) sample 5D coordinates along camera rays; (b) feed them to "
            "the MLP $F_\\Theta$ for color+density; (c) volume-render those "
            "samples into a pixel color per ray; (d) compare against the ground "
            "truth photo -- the entire chain is differentiable.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Two tricks that make it actually work

    A direct implementation of the above badly underperforms. Two fixes,
    both ablated in Figure 4 below, are essential:

    - **Positional encoding** (Eq. 4): plain coordinate-based MLPs are
      *spectrally biased* toward low-frequency functions -- they simply
      can't represent fine detail from raw $(x,y,z,\theta,\phi)$ inputs. NeRF
      first maps each coordinate through a fixed, non-learned sinusoidal
      encoding at increasing frequencies,
      $$ \gamma(p)=\big(\sin(2^0\pi p),\cos(2^0\pi p),\ldots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\big), \qquad (\text{Eq. 4}) $$
      *before* the MLP ever sees it ($L{=}10$ for position, $L{=}4$ for
      viewing direction) -- the same frequency-boosting trick behind
      Transformer positional encodings, repurposed here for continuous
      coordinates instead of discrete token positions.
    - **Hierarchical sampling**: densely sampling every ray wastes most of
      its queries on empty space. NeRF trains a "coarse" network first,
      reinterprets its per-sample weights $w_i=T_i(1-\exp(-\sigma_i\delta_i))$
      as a piecewise-constant probability density along the ray (Eq. 5), and
      resamples a "fine" network's query points from *that* distribution --
      concentrating compute where the scene actually has visible content.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(8, 2.4))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "nerf_fig4_ablation.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 4 from Mildenhall et al. 2020, reproduced for educational "
            "use. Removing view-dependence loses the specular highlight on the "
            "bulldozer tread; removing positional encoding collapses high-"
            "frequency geometry and texture into an oversmoothed blob.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Input, cost, and limits

    NeRF's only required input is a set of RGB images *with known camera
    poses and intrinsics* -- for real (non-synthetic) scenes, the paper
    gets these by running **COLMAP**, the same structure-from-motion problem
    notebook 23 just solved from scratch via bundle adjustment. Training
    optimizes one MLP **per scene** from scratch, typically 100-300k
    gradient steps (roughly 1-2 days on a single 2020-era GPU), and
    *rendering* one new frame still requires marching rays through the
    volume and querying the MLP many times per ray -- there's no
    pre-built geometry to simply rasterize, so even a trained NeRF renders
    far from real time. Both the training cost and the rendering cost are
    exactly what 3D Gaussian Splatting was designed to fix.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 3D Gaussian Splatting: an explicit, fast alternative

    Published three years after NeRF (Kerbl, Kopanas, Leimkühler, and
    Drettakis, SIGGRAPH 2023 -- not in Szeliski's 2021 draft), 3D Gaussian
    Splatting keeps NeRF's core idea -- optimize a representation by
    differentiably rendering it against training photos -- but swaps the
    representation itself from an *implicit* MLP field to an **explicit set
    of 3D Gaussian primitives**, rendered by GPU rasterization instead of
    volumetric ray marching.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 2.1))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "gsplat_fig1_teaser_comparison.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Crop of Figure 1 from Kerbl, Kopanas, Leimkühler, and Drettakis, "
            "\"3D Gaussian Splatting for Real-Time Radiance Field Rendering,\" "
            "ACM TOG 42(4), 2023 (arXiv:2308.04079), reproduced for educational "
            "use. At a comparable ~6-minute training budget, their method "
            "already matches InstantNGP's quality while rendering at 135 fps; "
            "given Mip-NeRF360's full 48-hour training budget instead, 51 "
            "minutes of training here reaches slightly higher quality still, at "
            "93 fps -- Mip-NeRF360 itself renders at 0.071 fps.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Representation: a cloud of fuzzy 3D ellipsoids

    Each primitive is a 3D Gaussian "blob" in world space, centered at mean
    $\mu$ with a full 3x3 covariance $\Sigma$:

    $$ G(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}. \qquad (\text{Eq. 4, Kerbl et al.}) $$

    A raw $\Sigma$ isn't safe to optimize directly by gradient descent --
    nothing stops an update from producing an invalid (non positive
    semi-definite) covariance. So instead of storing $\Sigma$, each Gaussian
    stores a scale vector $s$ and a rotation quaternion $q$ and *reconstructs*
    $\Sigma=RSS^TR^T$ (Eq. 6) every step -- always valid by construction,
    exactly the same "parameterize the thing you can safely differentiate,
    reconstruct the thing you actually need" move notebook 23 used for camera
    rotations. Each Gaussian also carries an opacity $\alpha$ and a color
    represented by **spherical harmonic (SH) coefficients**, so color can
    vary smoothly with viewing direction too, just as NeRF's MLP output did.

    **Crucially, initialization is literally notebook 23's output**: start
    from the sparse point cloud structure-from-motion already produces as a
    side effect of camera calibration, and turn every point directly into
    one small 3D Gaussian.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 1.8))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "gsplat_fig2_pipeline.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 2 from Kerbl et al. 2023, reproduced for educational use. "
            "SfM points (notebook 23's output) initialize a set of 3D "
            "Gaussians; a fast differentiable tile rasterizer renders them, "
            "compared against training photos; adaptive density control "
            "(Section 5 below) adds and removes Gaussians as optimization "
            "proceeds.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Rendering: project to 2D, sort, and alpha-blend -- no ray marching

    To render, each 3D Gaussian is projected into camera space via an affine
    approximation of the perspective projection (a Jacobian $J$),
    $\Sigma'=JW\Sigma W^TJ^T$ (Eq. 5), and the result is simply a 2D Gaussian
    "splat" on the image plane. Rendering a pixel is then the *same*
    discrete alpha-compositing form as NeRF's Eq. 3 -- $C=\sum_i c_i\alpha_i
    \prod_{j<i}(1-\alpha_j)$ -- just applied to a handful of depth-sorted
    splats overlapping that pixel, instead of dozens of samples along a
    marched ray. The paper's **tile-based rasterizer** makes this fast: split
    the screen into 16x16-pixel tiles, assign each Gaussian a (tile, depth)
    key, sort *all* Gaussians once with a single GPU radix sort, and
    alpha-blend each tile's pre-sorted list front-to-back until a pixel's
    accumulated opacity saturates. No ray marching, no per-pixel sort, no
    MLP query anywhere in the render path -- which is exactly why this
    renders in real time where NeRF cannot.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Training: the same differentiable-rendering loop, now with adaptive density control

    Optimization still looks like NeRF's: render, compare to a training
    photo, backpropagate -- here through an $L_1$ plus D-SSIM loss (Eq. 7)
    instead of NeRF's squared error, with gradients flowing into each
    Gaussian's position, covariance, opacity, and SH color coefficients.
    What's new is that the *number* of Gaussians also changes during
    training, interleaved with the optimization steps:

    - Gaussians in **under-reconstructed** regions (missing fine detail) are
      **cloned** -- duplicated and nudged along the positional gradient.
    - Gaussians in **over-reconstructed** regions (one blob standing in for
      detail it's too coarse to capture) are **split** into two smaller ones.
    - Both cases are detected the same way: large view-space positional
      gradients, since both correspond to regions the optimizer is still
      visibly straining to fix.
    - Near-transparent Gaussians (low $\alpha$) are periodically **pruned**,
      and $\alpha$ is occasionally reset near zero to curb runaway growth
      from floaters.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "gsplat_fig4_densification.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 4 from Kerbl et al. 2023, reproduced for educational use. "
            "Top: under-reconstruction is fixed by cloning a Gaussian. Bottom: "
            "over-reconstruction is fixed by splitting one large Gaussian into "
            "two smaller ones.*"
        ),
    ])
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 2.2))
    _ax.imshow(plt.imread(EXTERNAL_FIGURES_DIR / "gsplat_fig3_shrunken_gaussians.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "*Figure 3 from Kerbl et al. 2023, reproduced for educational use. "
            "Shrinking the optimized Gaussians 60% (right) reveals the "
            "anisotropic, elongated shapes they settle into -- thin and long "
            "along a surface, compactly representing fine structure with far "
            "fewer primitives than tiny isotropic blobs would need.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Two answers to the same question

    | | **NeRF** | **3D Gaussian Splatting** |
    |---|---|---|
    | Representation | Implicit: MLP weights | Explicit: millions of 3D Gaussians |
    | Rendering | Volumetric ray marching (Eq. 1-3) | Tile-based rasterization + $\alpha$-blending |
    | Render speed | Seconds per frame | Real time (>=30 fps at 1080p) |
    | Training time | ~1-2 days (original paper) | Minutes |
    | Required input | Posed images (+ COLMAP for real scenes) | Posed images + SfM sparse point cloud |
    | View-dependence | Direction-conditioned MLP output | Spherical harmonic coefficients per Gaussian |
    | Editability | Hard -- geometry is implicit in network weights | Easier -- primitives can be directly added/moved/removed |

    The single most important similarity is easy to miss: **both methods
    start from exactly the same input** -- a set of photographed images with
    camera poses recovered by structure from motion, i.e., notebook 23's
    bundle adjustment (or the equivalent inside COLMAP). Gaussian Splatting
    simply goes one step further and reuses SfM's *sparse point cloud* too,
    as literal initialization. Neither method would have anything to
    optimize against without that posed-camera starting point.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Beyond today

    - **These aren't as new as they look.** NeRF's continuous volumetric
      field is a direct, learned descendant of Chapter 13's **volumetric
      representations** (§13.5); Gaussian Splatting's unstructured primitive
      cloud descends from **point-based representations** (§13.4) and from
      classical **splatting** (Zwicker et al. 2001, cited directly by Kerbl
      et al. for the 2D-projection math in Eq. 5 above); both descend from
      **light fields and Lumigraphs** (§14.3) in spirit -- densely enough
      sampled rays eventually look continuous regardless of representation.
    - **Already in the book, cited by name**: Szeliski's own §13.6 and §14.6
      quote the original NeRF paper directly and discuss several extensions
      the book itself flags as active research even in 2021 -- generalizable
      /few-shot radiance fields conditioned on as little as one image
      (pixelNeRF), and deformable/dynamic radiance fields for hand-held video
      of non-rigid scenes (Nerfies, Neural Scene Flow Fields, HyperNeRF).
    - **Since the book's 2021 draft**: anti-aliased and unbounded-scene NeRF
      variants (Mip-NeRF, Mip-NeRF 360 -- the strong baseline Gaussian
      Splatting compares against above), fast grid/hash-based training
      (Instant-NGP, Plenoxels), and -- the subject of this notebook's second
      half -- 3D Gaussian Splatting itself, plus its own fast-growing family
      (Mip-Splatting for anti-aliasing, SuGaR for extracting meshes from a
      trained Gaussian scene, and compression schemes to shrink the several
      millions of Gaussians a typical scene needs on disk).
    - **Practical tools**: COLMAP remains the standard way to get camera
      poses and a sparse point cloud from an unstructured photo set;
      Nerfstudio and the original Inria 3D Gaussian Splatting codebase (and
      its many forks, e.g. `gsplat`) are the standard way to actually train
      and view either representation -- covered hands-on in a follow-up
      session, not in this notebook.

    ## Summary

    - **Novel view synthesis** (§14): render a new camera viewpoint never
      directly photographed, given a set of posed training images -- notebook
      23's bundle-adjusted poses and point cloud are the standard starting
      point for every method in this notebook.
    - **NeRF** (§14.6; Mildenhall et al. 2020) represents a scene as a single
      continuous MLP mapping 5D position+direction to color and density,
      rendered by classical volume rendering (Eq. 1-3) made differentiable,
      and trained end to end against a simple photometric loss -- enabled in
      practice by positional encoding (Eq. 4) and hierarchical sampling
      (Eq. 5). Photorealistic, but slow to train (~1-2 days) and slow to
      render (no real-time display).
    - **3D Gaussian Splatting** (Kerbl et al. 2023) represents a scene as an
      explicit, optimizable cloud of 3D Gaussians -- initialized directly
      from SfM's sparse point cloud -- rendered by projecting to 2D and
      alpha-blending depth-sorted splats via a fast tile-based GPU
      rasterizer, with adaptive density control (clone/split/prune)
      interleaved into training. Comparable or better quality than NeRF, at
      training times measured in minutes and real-time (>=30 fps) rendering.
    - Both methods are optimized **per scene** from differentiable rendering
      against photos, not learned once and generalized across scenes (with
      some newer exceptions, e.g. pixelNeRF) -- and both are direct, modern
      descendants of classical representations (volumetric fields, point
      splatting, light fields) already covered in Szeliski Chapters 13-14.
    """)
    return


if __name__ == "__main__":
    app.run()
