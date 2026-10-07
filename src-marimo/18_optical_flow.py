# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from scipy.signal import correlate2d
    from scipy.ndimage import gaussian_filter, map_coordinates

    return (
        correlate2d,
        gaussian_filter,
        hsv_to_rgb,
        map_coordinates,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    EXTERNAL_FIGURES_DIR = IMAGES_DIR / "external_figures"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return EXTERNAL_FIGURES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Optical Flow

    Notebook 13 asked "does this patch move unambiguously?" at a sparse set of
    corners. **Optical flow** asks the same question at *every* pixel: an
    independent motion estimate $u=(u_x,u_y)$ per pixel, densely over the whole image.
    This notebook covers Szeliski **§9.3, "Optical flow"** (and the Lucas-Kanade
    derivation from §9.1.3, "Incremental refinement," that it builds on) —
    brightness constancy, the optical flow constraint equation, and the
    Lucas-Kanade local method, which turns out to reuse the *exact* auto-correlation
    matrix from notebook 13.

    We stop short of global/variational methods (Horn-Schunck) and layered
    (multi-object, segmented) motion models — real fundamentals, not the full
    research literature. Before any of that, though: what is this actually used
    for, beyond the examples already in the book?
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-17)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Why optical flow matters: applications beyond the book

    The book's own optical-flow application sidebars are worth knowing (video
    stabilization, rolling-shutter correction, frame interpolation, video object
    segmentation/tracking) — but they're not the whole story. A few more, spanning
    very different fields:

    - **Scientific imaging — Particle Image Velocimetry (PIV).** Seed a fluid with
      small tracer particles, photograph it twice in quick succession, and run
      essentially the same windowed-correlation idea as Lucas-Kanade below to
      recover the fluid's velocity field. It's a standard tool in aerodynamics,
      combustion research, and oceanography (Adrian 1991; Raffel, Willert et al.
      2007). The figure below is a real PIV result — notice how similar the seeded
      "tracer" image looks to the synthetic textured image we'll build in §4.
    - **Autonomous driving and robotics.** Per-pixel motion (and its 3D
      generalization, *scene flow*) helps a vehicle or robot separate independently
      moving objects from the background induced by its own motion — part of why
      the KITTI benchmark (Geiger, Lenz, and Urtasun 2012), mentioned in §9.3, was
      built from a driving platform in the first place.
    - **Action recognition and video understanding.** A classic and still-influential
      approach feeds optical flow into a neural network as a second, motion-only
      input stream alongside raw RGB frames, since *how* things move is often a more
      reliable cue for recognizing an action than any single frame's appearance
      (Simonyan and Zisserman 2014).
    - **Bio-inspired robotics.** Flying insects navigate, avoid obstacles, and
      control landing using optic flow almost exclusively, with no direct distance
      sensing at all (Srinivasan, Zhang et al. 1996). This has directly inspired
      lightweight optic-flow-only autopilots for drones (e.g., the BeeRotor
      platform, Expert and Ruffier 2015), useful precisely where cameras are cheap
      but range sensors are heavy.
    - **Sports and broadcast analytics.** Player- and ball-tracking pipelines used
      in broadcast and performance analysis lean on dense motion estimation to
      maintain tracks through fast, blurry motion between frames (Thomas, Gade et
      al. 2017).
    - **Medical motion imaging.** Beyond the book's inter-patient atlas
      registration example (Figure 9.6), optical flow is used to track a *single*
      patient's own tissue motion over time, e.g., estimating heart-wall strain
      from echocardiography or tagged cardiac MRI sequences (Tobon-Gomez, De Craene
      et al. 2013).
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, plt):
    _img = plt.imread(str(EXTERNAL_FIGURES_DIR / "piv_vortex_pair.jpg"))
    _fig, _ax = plt.subplots(figsize=(7.5, 5))
    _ax.imshow(_img)
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md(
            "**Particle Image Velocimetry**: a vortex pair, visualized by seeding "
            "tracer particles into the fluid and computing the velocity field "
            "between two exposures (inset: finer-resolution re-analysis of the "
            "left vortex). *Willa, Wikimedia Commons, CC BY-SA 3.0.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Brightness constancy: the assumption

    Write $I(x,y,t)$ for image intensity at pixel $(x,y)$ at time $t$. Nearly
    all of motion estimation, optical flow included, rests on one assumption:
    a physical point's brightness doesn't change as it moves — **brightness
    constancy**. If that point is at $(x,y)$ in frame $t$ and moves by
    $u=(u_x,u_y)$ by the next frame,

    $$ I(x,y,t) = I(x+u_x,\ y+u_y,\ t+1). $$

    Translational alignment (an earlier part of this chapter) assumes $u$ is
    the *same* for every pixel in a patch, and fits it by minimizing the
    summed squared violation of this equation over the patch,

    $$ E_{SSD}(u) = \sum_i \big[I_1(x_i + u) - I_0(x_i)\big]^2, \qquad (9.1) $$

    writing $I_0, I_1$ for the two frames. **Optical flow** instead asks for
    an independent $u$ at *every* pixel: a full 2D vector field, not one
    global shift — twice as many unknowns as the one brightness-constancy
    equation above provides per pixel. Pinning it down starts with
    linearizing that equation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Linearizing: Taylor expansion to the optical flow constraint equation

    Take a first-order Taylor expansion of the right-hand side of brightness
    constancy around $(x,y,t)$, treating the displacement $u=(u_x,u_y)$ (and
    the one-frame time step) as small:

    $$ I(x+u_x,\ y+u_y,\ t+1) \approx I(x,y,t) + I_x u_x + I_y u_y + I_t, $$

    where $I_x = \partial I/\partial x$ and $I_y=\partial I/\partial y$ are the
    spatial image gradients and $I_t = \partial I/\partial t$ is the temporal
    derivative (the frame-to-frame intensity change), all evaluated at
    $(x,y,t)$. Substitute this expansion back into brightness constancy,
    $I(x,y,t) = I(x+u_x,y+u_y,t+1)$ — the $I(x,y,t)$ term on both sides
    cancels, leaving the **optical flow constraint equation** (Horn and
    Schunck 1981),

    $$ I_x u_x + I_y u_y + I_t = 0. \qquad (9.31) $$

    This is **one scalar equation in two unknowns** $u=(u_x,u_y)$ at every
    pixel — it only pins down the component of motion *along* the gradient
    direction, exactly the aperture problem notebook 13 already met for a
    single patch (Figure 7.4): a slanted edge's true sideways motion is
    invisible to this equation, since sliding along the edge changes nothing
    about $I_x, I_y, I_t$ there.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _img = plt.imread(str(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_4_aperture_problem.png"))
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.imshow(_img)
    _ax.axis("off")
    mo.vstack([mo.md("**Figure 7.4** (Szeliski) — the aperture problem, revisited."), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Lucas-Kanade: the same auto-correlation matrix, applied densely

    One equation per pixel isn't enough, but a small *window* of pixels is: Lucas
    and Kanade (1981) assume $u=(u_x,u_y)$ is constant over a small neighborhood, then
    least-squares fit it against every optical-flow-constraint equation in that
    window at once,

    $$ E_{LK}(u) = \sum_i \big[I_xu_x + I_yu_y + I_t\big]^2, $$

    summed over the window. Minimizing this leads to the normal equations

    $$ A\,\Delta u = b, \qquad (9.32) $$

    $$ A = \begin{bmatrix} \sum I_x^2 & \sum I_xI_y \\ \sum I_xI_y & \sum I_y^2 \end{bmatrix}, \qquad b = -\begin{bmatrix} \sum I_xI_t \\ \sum I_yI_t \end{bmatrix}. \qquad (9.33\text{-}9.35) $$

    Look closely at $A$: it's **exactly** the auto-correlation (structure tensor)
    matrix from notebook 13's Harris corner detector — the same windowed sums of
    gradient outer products, computed here for a different purpose. Everything
    notebook 13 said about its eigenvalues still applies: two large eigenvalues
    (a corner-like, textured patch) means $A$ is well-conditioned and flow is
    reliable; one small eigenvalue (an edge) means the aperture problem strikes
    and only the cross-edge motion component is trustworthy; two small
    eigenvalues (a flat region) means $A$ is singular and there's no information
    at all. This is made precise by the uncertainty model

    $$ \Sigma_u = \sigma_n^2 A^{-1}, \qquad (9.37) $$

    the covariance of the flow estimate under additive noise $\sigma_n^2$ — its
    eigenvalues are the (inverse) confidence in each direction of motion, the
    *least*-certain direction being whichever eigenvector of $A$ has the
    *smallest* eigenvalue.

    Because the Taylor expansion is only a *local* linear approximation, one
    shot isn't always enough — Lucas and Kanade's own fix is to iterate: warp
    $I_1$ by the current estimate, recompute the residual $I_t$ between the
    warped $I_1$ and $I_0$, solve (9.32) again for an update $\Delta u$, and
    repeat (§9.1.3, "Incremental
    refinement"). Try it below on a richly-textured synthetic image (so $A$ is
    well-conditioned everywhere) undergoing a small, exactly-known rotation:
    watch the single-shot estimate improve as you add iterations, and see how a
    larger rotation needs more of them before converging.
    """)
    return


@app.function
def compute_gradients(img_gray, sigma, gaussian_filter, correlate2d, np):
    GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    GY = GX.T
    smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect") if sigma > 0 else img_gray
    Ix = correlate2d(smoothed, GX, mode="same", boundary="symm")
    Iy = correlate2d(smoothed, GY, mode="same", boundary="symm")
    return Ix, Iy


@app.function
def make_textured_image(H, W, texture_sigma, seed, np, gaussian_filter):
    """Band-limited random noise -- lots of 2D texture everywhere, so the
    auto-correlation matrix A is well-conditioned at every pixel (no aperture
    problem to worry about yet)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (H, W))
    texture = gaussian_filter(noise, sigma=texture_sigma)
    return (texture - texture.min()) / (texture.max() - texture.min()) * 255.0


@app.function
def rotation_flow_field(H, W, theta_deg, np):
    """Ground-truth dense flow (dx, dy) for a small rotation about the image
    center -- a genuinely spatially-varying 'optical flow' field, not just one
    global displacement."""
    theta = np.radians(theta_deg)
    cx, cy = W / 2, H / 2
    ct, st = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx = (xx - cx) * ct - (yy - cy) * st - (xx - cx)
    dy = (xx - cx) * st + (yy - cy) * ct - (yy - cy)
    return dx, dy


@app.function
def warp_by_flow(img, dx, dy, map_coordinates, np):
    """Produce I1 from I0 given the forward flow (dx,dy): I1(x) = I0(x - u(x)),
    the standard construction matching brightness constancy I1(x+u) = I0(x)."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return map_coordinates(img, [yy - dy, xx - dx], order=1, mode="reflect")


@app.function
def lucas_kanade_flow(I0, I1, win_sigma, grad_sigma, n_iter, gaussian_filter, correlate2d, map_coordinates, np):
    """Iterative Lucas-Kanade (Eqs. 9.32-9.36): fixed template gradients, warp
    I1 by the running estimate each iteration, and accumulate the update."""
    H, W = I0.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u = np.zeros((H, W))
    v = np.zeros((H, W))
    Ix, Iy = compute_gradients(I0, grad_sigma, gaussian_filter, correlate2d, np)
    Ixx = gaussian_filter(Ix * Ix, win_sigma)
    Iyy = gaussian_filter(Iy * Iy, win_sigma)
    Ixy = gaussian_filter(Ix * Iy, win_sigma)
    detA = Ixx * Iyy - Ixy**2
    detA_safe = np.where(np.abs(detA) < 1e-6, 1e-6, detA)
    trA = Ixx + Iyy
    disc = np.sqrt(np.maximum((Ixx - Iyy) ** 2 + 4 * Ixy**2, 0))
    lam_min = (trA - disc) / 2
    for _ in range(n_iter):
        I1_warped = map_coordinates(I1, [yy + v, xx + u], order=1, mode="reflect")
        It = I1_warped - I0
        Ixt = gaussian_filter(Ix * It, win_sigma)
        Iyt = gaussian_filter(Iy * It, win_sigma)
        du = -(Iyy * Ixt - Ixy * Iyt) / detA_safe
        dv = -(Ixx * Iyt - Ixy * Ixt) / detA_safe
        u = u + du
        v = v + dv
    return u, v, lam_min


@app.function
def flow_to_color(u, v, max_mag, hsv_to_rgb, np):
    """Standard angle-is-hue, magnitude-is-saturation flow visualization (the
    same convention used on the Middlebury flow evaluation page, Figure 9.8)."""
    mag = np.hypot(u, v)
    hue = (np.arctan2(v, u) + np.pi) / (2 * np.pi)
    sat = np.clip(mag / max_mag, 0, 1)
    val = np.ones_like(hue)
    return hsv_to_rgb(np.stack([hue, sat, val], axis=-1))


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the color panels below.** Each pixel's flow vector $u=(u_x,u_y)$
    is mapped to a color in HSV space, with direction as hue and speed as
    saturation:

    $$ \text{hue} = \frac{\operatorname{atan2}(u_y, u_x) + \pi}{2\pi}, \qquad \text{saturation} = \min\!\left(\frac{\|u\|}{u_{max}},\, 1\right), \qquad \text{value} = 1, $$

    where $u_{max}$ is the largest flow magnitude in the frame (so the fastest
    motion present is fully saturated, and everything else is scaled relative
    to it). A vector pointing right is one color, left is its complementary
    color, and a near-zero vector of any direction washes out toward white —
    exactly the convention used on the Middlebury flow evaluation page (Figure
    9.8). The ground-truth panel also overlays a sparse sample of the actual
    arrows so the color-to-direction mapping is concrete before you have to
    read it off color alone.
    """)
    return


@app.cell
def _(mo):
    flow_theta_slider = mo.ui.slider(start=0.5, stop=4.0, value=1.5, step=0.5, label="rotation (degrees)", debounce=True)
    flow_niter_slider = mo.ui.slider(start=1, stop=8, value=1, step=1, label="LK iterations", debounce=True)
    mo.hstack([flow_theta_slider, flow_niter_slider], justify="start", gap=2)
    return flow_niter_slider, flow_theta_slider


@app.cell
def _(
    correlate2d,
    flow_niter_slider,
    flow_theta_slider,
    gaussian_filter,
    hsv_to_rgb,
    map_coordinates,
    mo,
    np,
    plt,
):
    _H, _W = 200, 260
    _texture = make_textured_image(_H, _W, texture_sigma=2.0, seed=0, np=np, gaussian_filter=gaussian_filter)
    _dx, _dy = rotation_flow_field(_H, _W, flow_theta_slider.value, np)
    _frame2 = warp_by_flow(_texture, _dx, _dy, map_coordinates, np)

    _u, _v, _lam = lucas_kanade_flow(
        _texture, _frame2, win_sigma=2.0, grad_sigma=1.0, n_iter=flow_niter_slider.value,
        gaussian_filter=gaussian_filter, correlate2d=correlate2d, map_coordinates=map_coordinates, np=np,
    )
    _err = np.hypot(_u - _dx, _v - _dy)
    _max_mag = max(np.hypot(_dx, _dy).max(), 1e-6)

    _fig, _axes = plt.subplots(2, 2, figsize=(9, 7.4))
    _axes[0, 0].imshow(_texture, cmap="gray")
    _axes[0, 0].set_title("frame 1 (synthetic texture)")
    _axes[0, 1].imshow(flow_to_color(_u, _v, _max_mag, hsv_to_rgb, np))
    _axes[0, 1].set_title(f"estimated flow ({flow_niter_slider.value} iter)")
    _axes[1, 0].imshow(flow_to_color(_dx, _dy, _max_mag, hsv_to_rgb, np))
    _gt_step = 20
    _gt_yy, _gt_xx = np.mgrid[_gt_step // 2 : _H : _gt_step, _gt_step // 2 : _W : _gt_step]
    _axes[1, 0].quiver(
        _gt_xx, _gt_yy, _dx[_gt_yy, _gt_xx], _dy[_gt_yy, _gt_xx],
        color="black", angles="xy", scale_units="xy", scale=0.15, width=0.006,
    )
    _axes[1, 0].set_title("ground-truth flow")
    _im = _axes[1, 1].imshow(_err, cmap="inferno", vmin=0, vmax=max(_max_mag, 1.0))
    _axes[1, 1].set_title(f"endpoint error (median {np.median(_err):.2f}px)")
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.colorbar(_im, ax=_axes[1, 1], fraction=0.046)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "Compare the two color panels: hue encodes direction, saturation "
            "encodes speed. A true rotation field radiates outward with hue "
            "sweeping a full circle -- the estimate should match it almost "
            "everywhere, since this texture never suffers the aperture problem. "
            "Push the rotation up and watch the 1-iteration estimate degrade "
            "(the Taylor approximation breaking down), then add iterations back "
            "and watch the error map darken again -- starting from the center "
            "outward. Even with many iterations, a dim ring of high error "
            "persists near the corners, where true displacement is largest: "
            "iterating fixes a *local* linearization error, but it can't rescue "
            "a corner pixel whose true motion is simply too big for this "
            "single-resolution window to have ever converged correctly. That "
            "gap is exactly what the coarse-to-fine pyramid in §6 is for."
        )
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Real-world flow: the MPI-Sintel benchmark

    Szeliski's own text singles out one dataset: "more recent publications tend
    to focus (both training and evaluation) on the MPI Sintel dataset" (Butler,
    Wulff et al. 2012) — built by rendering an open-source animated film,
    *Sintel*, since computer-generated frames come with perfect, free ground-truth
    flow that a real camera never could. Below are two real frame pairs from that
    same film (not the benchmark's own ground-truth-annotated release, just the
    finished movie itself, one frame apart), chosen because the camera itself
    is panning across a richly-textured scene — real motion parallax, nearby
    objects sliding faster than distant ones, with no synthetic guarantees this
    time. Run through the identical single-shot Lucas-Kanade from §4 above.
    Only pixels with a large minimum eigenvalue of $A$ (confident, §4) are drawn,
    exactly the mask the aperture problem argument says we should trust.
    """)
    return


@app.cell
def _(mo):
    flow_scene_dropdown = mo.ui.dropdown(options=["desert", "overlook"], value="desert", label="scene")
    flow_scene_dropdown
    return (flow_scene_dropdown,)


@app.cell
def _(EXTERNAL_FIGURES_DIR, flow_scene_dropdown, mo):
    mo.vstack([
        mo.video(
            src=str(EXTERNAL_FIGURES_DIR / f"sintel_{flow_scene_dropdown.value}_clip.mp4"),
            controls=True, muted=True, loop=True, autoplay=True, width=480,
        ),
        mo.md(
            "A short real clip spanning the two frames used below (32 frames, "
            "~1.3s at the film's native 24fps) — watch the actual motion here, "
            "then compare it against the single-pair flow estimate underneath."
        ),
    ])
    return


@app.cell
def _(
    EXTERNAL_FIGURES_DIR,
    correlate2d,
    flow_scene_dropdown,
    gaussian_filter,
    map_coordinates,
    mo,
    np,
    plt,
):
    _f1 = plt.imread(str(EXTERNAL_FIGURES_DIR / f"sintel_{flow_scene_dropdown.value}_frame1.png"))[:, :, :3] * 255.0
    _f2 = plt.imread(str(EXTERNAL_FIGURES_DIR / f"sintel_{flow_scene_dropdown.value}_frame2.png"))[:, :, :3] * 255.0
    _g1, _g2 = to_luma(_f1), to_luma(_f2)

    _u, _v, _lam = lucas_kanade_flow(
        _g1, _g2, win_sigma=2.0, grad_sigma=1.0, n_iter=1,
        gaussian_filter=gaussian_filter, correlate2d=correlate2d, map_coordinates=map_coordinates, np=np,
    )
    _H, _W = _g1.shape
    _step = 5
    _yy, _xx = np.mgrid[_step // 2 : _H : _step, _step // 2 : _W : _step]
    _conf_thresh = np.percentile(_lam, 55)
    _mask = _lam[_yy, _xx] > _conf_thresh

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    _axes[0].imshow(_g1, cmap="gray")
    _axes[0].set_title("frame 1")
    _axes[1].imshow(_g1, cmap="gray")
    _axes[1].quiver(
        _xx[_mask], _yy[_mask], _u[_yy, _xx][_mask], _v[_yy, _xx][_mask],
        color="lime", angles="xy", scale_units="xy", scale=0.25, width=0.003,
    )
    _axes[1].set_title("flow (confident pixels only)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Beyond today

    Three directions, deliberately left unimplemented here:

    - **Coarse-to-fine (pyramid) refinement.** The demo above showed the honest
      limit of single-resolution Lucas-Kanade: no amount of iterating rescues a
      pixel whose true displacement is larger than the window can see.
      §9.1.1's fix is to estimate flow on a heavily downsampled (notebook 11
      style) pyramid level first, where the same motion is only a pixel or two,
      then use that as the starting point one level finer, repeating down to
      full resolution — the same "coarse first" idea used for feature detection
      (DoG scale-space, notebook 13) and image alignment, applied here to
      extend Lucas-Kanade to much larger motions without changing the
      per-level math at all.
    - **Global (variational) methods.** Horn and Schunck (1981) minimize a
      single energy over the *entire* flow field at once, trading the local
      window assumption for an explicit smoothness penalty on neighboring flow
      vectors, $E_{HS} = \int (I_xu_x + I_yu_y + I_t)^2\,dx\,dy + \text{(smoothness
      term)}$ (Eq. 9.57) — this better handles motion discontinuities but
      requires solving a large coupled system over the whole image rather than
      one small $2\times2$ system per pixel.
    - **Deep learning.** Modern top-performing methods (FlowNet, PWC-Net, RAFT,
      and successors) replace the hand-derived linearization entirely with a
      trained network, per §9.3.1 — effective, but no longer the kind of
      closed-form math this notebook was after.
    - Not covered either: **layered motion** (§9.4) — segmenting a scene into
      independently-moving regions with their own parametric models, a
      genuinely different (harder) problem from estimating one dense field.

    ## Summary

    - **Brightness constancy** (Eq. 9.1) says corresponding pixels keep their
      value across frames; optical flow asks for an independent $u=(u_x,u_y)$ at
      *every* pixel instead of one global shift.
    - Taylor-expanding it gives the **optical flow constraint equation**
      ($I_xu_x+I_yu_y+I_t=0$, Eq. 9.31) — one equation, two unknowns, the aperture
      problem all over again.
    - **Lucas-Kanade** (Eqs. 9.32-9.37) resolves this with a local window,
      reusing notebook 13's auto-correlation matrix $A$ for an entirely new
      purpose; its eigenvalues are *exactly* what determine confidence in the
      recovered motion.
    - **Incremental refinement** (iterating: warp, re-residual, re-solve) fixes
      up the linearization for larger motions — demonstrated on a synthetic,
      fully-textured image with known ground truth, then tried honestly on two
      real frame pairs from the MPI-Sintel film.
    """)
    return


if __name__ == "__main__":
    app.run()
