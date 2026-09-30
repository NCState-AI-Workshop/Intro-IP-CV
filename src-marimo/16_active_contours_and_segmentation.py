# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
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
    from scipy.signal import correlate2d
    from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt, maximum_filter, label
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.segmentation import felzenszwalb, mark_boundaries

    return (
        cKDTree,
        connected_components,
        coo_matrix,
        correlate2d,
        distance_transform_edt,
        felzenszwalb,
        gaussian_filter,
        label,
        map_coordinates,
        mark_boundaries,
        maximum_filter,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Active Contours and Segmentation

    Notebooks 13–15 found *points* (corners) and *curves* (edges). This notebook
    covers two more ways of describing image content, both from Szeliski
    **Chapter 7**: **active contours** (§7.3, "Contour tracking") lock an initial
    curve estimate onto a real object boundary using the edge map from notebook
    15, and **segmentation** (§7.5) partitions the *whole* image into regions,
    not just their boundaries.

    Several of these techniques (level sets, normalized cuts) are described here
    at a conceptual level rather than implemented from scratch — they require
    numerical machinery (PDE evolution, dense eigendecomposition at scale) beyond
    what's useful to hand-derive in a course notebook, and the book itself notes
    that many classical segmentation methods "are no longer widely used." Where
    that's the case, we say so explicitly.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-15)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(np):
    # Normalized Sobel (notebook 13): dividing by 8 makes the response equal the
    # true intensity-change-per-pixel.
    GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    GY = GX.T

    def compute_gradients(img_gray, sigma, gaussian_filter, correlate2d):
        smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect") if sigma > 0 else img_gray
        Ix = correlate2d(smoothed, GX, mode="same", boundary="symm")
        Iy = correlate2d(smoothed, GY, mode="same", boundary="symm")
        return Ix, Iy

    return (compute_gradients,)


@app.function
def rgb255_to_lab(rgb255, np):
    """Gamma-encoded RGB in [0,255] -> CIELAB, via Szeliski Eq. 2.106-2.109
    (same pipeline as notebook 7's rgb255_to_lab, redefined standalone here).
    Used below as a perceptually-motivated color feature space, in place of the
    book's CIELUV -- the mode-seeking and affinity mechanics are identical."""
    M_RGB_TO_XYZ = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ])
    WHITE = np.array([0.950456, 1.0, 1.088754])
    DELTA = 6 / 29

    c = np.asarray(rgb255, dtype=float) / 255.0
    linear = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = linear @ M_RGB_TO_XYZ.T

    def f(t):
        return np.where(t > DELTA**3, np.cbrt(t), t / (3 * DELTA**2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0] / WHITE[0]), f(xyz[..., 1] / WHITE[1]), f(xyz[..., 2] / WHITE[2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Active contours: snakes (§7.3.1)

    A **snake** is a curve $f(s) = (x(s), y(s))$, sampled at $s = 0, 1, \dots, N-1$
    control points, that evolves to minimize an energy functional. The internal
    (spline) energy penalizes stretching and bending,

    $$ E_{int} = \int \alpha(s)\left\|\frac{df}{ds}\right\|^2 + \beta(s)\left\|\frac{d^2f}{ds^2}\right\|^2 \, ds, \qquad (7.26) $$

    discretized over the control points as

    $$ E_{int} = \sum_i \alpha_i\|f_{i+1}-f_i\|^2 + \beta_i\|f_{i+1}-2f_i+f_{i-1}\|^2. \qquad (7.27) $$

    The first term is **elasticity** (large when points spread apart), the second
    is **curvature** (large when the curve bends sharply). Separately, an
    external image energy pulls the curve toward image features,

    $$ E_{image} = w_{line}E_{line} + w_{edge}E_{edge} + w_{term}E_{term}. \qquad (7.28) $$

    Each term pulls the curve toward a different kind of image structure:

    - $E_{line}$ is just the image intensity itself, $E_{line} = I(x,y)$, which
      attracts the curve to dark (or, with the opposite sign, bright) ridges.
    - $E_{edge} = -\|\nabla I\|^2$ attracts the curve to strong gradients —
      i.e., to real edges — since this term is most negative (lowest energy)
      exactly where the gradient magnitude is largest.
    - $E_{term}$ uses the curvature of level contours of a smoothed image to
      attract the curve specifically to line endings and corners, useful for
      closing small gaps between edge segments.

    We implement just $E_{edge}$ (using notebook 15's gradient magnitude,
    smoothed to widen its basin of attraction), since it does the most work for
    typical object boundaries and is the term everything else in this notebook
    has already built the machinery for. The full energy at each point is then
    $E_{int} + w_{edge}E_{edge}$, and the snake evolves to (locally) minimize it.

    **Minimizing it: the greedy algorithm (Williams and Shah 1992).** Rather than
    solving the variational problem directly, at each iteration every control
    point independently jumps to whichever nearby pixel (within a small search
    window) gives the lowest total energy, using its neighbors' *current*
    positions. Repeated many times, the curve wiggles into a local energy
    minimum — usually a real object boundary, if it started reasonably close.

    Because a snake's internal energy always prefers *shrinking* (points want to
    be close together), the book notes it's best initialized **outside** the
    object of interest, and it will contract onto the boundary. Try it below: the
    dashed blue circle is the initial curve, the solid curve is after
    optimization. Watch what happens if you push $w_{edge}$ too high relative to
    $\beta$ — an individual point can detach and jump onto a *neighboring*
    object's edge instead, a classic failure mode of greedy energy minimization.

    *(Many extensions exist beyond what's implemented here: **elastic nets** solve
    the Traveling Salesman Problem with a snake-like formulation; **B-snakes**
    and **active shape/appearance models** replace individual point freedom with
    a learned low-dimensional shape prior; **Kalman snakes** and **CONDENSATION**
    track a deforming contour across video frames; **intelligent scissors**
    ("live wire") lets a user sketch a rough boundary that snaps to the nearest
    real edge in real time via Dijkstra's algorithm. See §7.3.1 for all of
    these.)*
    """)
    return


@app.function
def init_circle_snake(cx, cy, r, n, np):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


@app.function
def greedy_snake(Eext, pts, alpha, beta, w_edge, window, n_iter, map_coordinates, np):
    """Greedy energy-minimizing snake: at each iteration, move every point to
    whichever nearby candidate location minimizes internal (Eq. 7.27) plus
    external (Eq. 7.28) energy, given its neighbors' positions from the last
    iteration."""
    pts = pts.copy()
    N = len(pts)
    offs = np.array(
        [(dx, dy) for dx in range(-window, window + 1) for dy in range(-window, window + 1)], dtype=float
    )
    for _ in range(n_iter):
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        cands = pts[:, None, :] + offs[None, :, :]
        econt = alpha * np.sum((cands - prev[:, None, :]) ** 2, axis=2)
        ecurv = beta * np.sum((prev[:, None, :] - 2 * cands + nxt[:, None, :]) ** 2, axis=2)
        xs, ys = cands[:, :, 0].ravel(), cands[:, :, 1].ravel()
        eext = w_edge * map_coordinates(Eext, [ys, xs], order=1, mode="nearest").reshape(cands.shape[:2])
        etot = econt + ecurv + eext
        best = np.argmin(etot, axis=1)
        pts = cands[np.arange(N), best]
    return pts


@app.cell
def _(mo):
    snake_image_dropdown = mo.ui.dropdown(
        options=["coins", "astronaut", "coffee", "chelsea", "raccoon"], value="coins", label="image"
    )
    snake_cx_slider = mo.ui.slider(start=0, stop=383, value=335, step=1, label="center x", debounce=True)
    snake_cy_slider = mo.ui.slider(start=0, stop=302, value=48, step=1, label="center y", debounce=True)
    snake_r0_slider = mo.ui.slider(start=10, stop=90, value=45, step=1, label="initial radius", debounce=True)
    snake_alpha_slider = mo.ui.slider(start=0.02, stop=0.6, value=0.2, step=0.02, label="alpha (elasticity)", debounce=True)
    snake_beta_slider = mo.ui.slider(start=0.02, stop=0.6, value=0.3, step=0.02, label="beta (curvature)", debounce=True)
    snake_wedge_slider = mo.ui.slider(start=0.5, stop=8.0, value=3.0, step=0.5, label="w_edge", debounce=True)
    snake_niter_slider = mo.ui.slider(start=10, stop=200, value=100, step=10, label="iterations", debounce=True)
    mo.vstack([
        snake_image_dropdown,
        mo.hstack([snake_cx_slider, snake_cy_slider, snake_r0_slider], justify="start", gap=2),
        mo.hstack(
            [snake_alpha_slider, snake_beta_slider, snake_wedge_slider, snake_niter_slider], justify="start", gap=2
        ),
    ])
    return (
        snake_alpha_slider,
        snake_beta_slider,
        snake_cx_slider,
        snake_cy_slider,
        snake_image_dropdown,
        snake_niter_slider,
        snake_r0_slider,
        snake_wedge_slider,
    )


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    map_coordinates,
    mo,
    np,
    plt,
    snake_alpha_slider,
    snake_beta_slider,
    snake_cx_slider,
    snake_cy_slider,
    snake_image_dropdown,
    snake_niter_slider,
    snake_r0_slider,
    snake_wedge_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, snake_image_dropdown.value, plt, np))
    _Ix, _Iy = compute_gradients(_img_gray, 1.5, gaussian_filter, correlate2d)
    _mag = np.hypot(_Ix, _Iy)
    _Eext = -gaussian_filter(_mag, sigma=2.0)

    _cx, _cy, _r0 = snake_cx_slider.value, snake_cy_slider.value, snake_r0_slider.value
    _pts0 = init_circle_snake(_cx, _cy, _r0, 50, np)
    _pts_final = greedy_snake(
        _Eext, _pts0,
        snake_alpha_slider.value, snake_beta_slider.value, snake_wedge_slider.value,
        window=2, n_iter=snake_niter_slider.value,
        map_coordinates=map_coordinates, np=np,
    )

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.imshow(_img_gray, cmap="gray")
    _ax.plot(np.append(_pts0[:, 0], _pts0[0, 0]), np.append(_pts0[:, 1], _pts0[0, 1]), "b--", label="initial")
    _ax.plot(
        np.append(_pts_final[:, 0], _pts_final[0, 0]), np.append(_pts_final[:, 1], _pts_final[0, 1]),
        "r-", lw=2, label="after optimization",
    )
    _ax.legend(loc="upper right", fontsize=9)
    _margin = _r0 + 25
    _ax.set_xlim(_cx - _margin, _cx + _margin)
    _ax.set_ylim(_cy + _margin, _cy - _margin)
    _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Level sets (§7.3.2)

    Snakes represent a curve *explicitly*, as an ordered list of points — which
    makes it awkward for the curve to split into two pieces or merge back
    together as it evolves (imagine a snake around one object that should split
    in two once the object turns out to be two overlapping objects). Every
    representation used so far in this course — points, patches, point lists —
    has been **explicit**: the data structure directly stores the thing it
    represents. Level sets introduce something different, and worth pausing on:
    an **implicit** representation.

    **The core idea.** Instead of storing the curve itself, store a scalar field
    $\phi(x,y)$ over the *whole plane*, and define the curve as wherever that
    field crosses zero: $\phi < 0$ inside, $\phi > 0$ outside (or vice versa).
    The curve isn't a data structure any more — it's a *derived quantity*, the
    zero **iso-level** of $\phi$. This sounds like a roundabout way to describe
    a curve, but it buys something an explicit point list can't easily give
    you: $\phi$ can be reshaped by an ordinary PDE (a local update rule applied
    everywhere), and however many separate loops its zero level set happens to
    trace out falls out automatically — nothing in the representation itself
    ever has to decide "is this one curve or two?"

    One convenient special case of an implicit field is a **signed distance
    function**: $\phi(x,y) = \pm\,\text{dist}(x, y, \text{curve})$, positive
    outside and negative inside. It satisfies the Eikonal equation
    $|\nabla\phi| = 1$ everywhere — the field's slope is exactly 1 in every
    direction, since moving one unit away from the curve changes your distance
    to it by exactly one unit. Keeping $\phi$ close to a signed distance
    function (by periodically recomputing it exactly from its own current zero
    level set) is what keeps the numerics below well-behaved.

    **Why this generalizes.** The same trick — represent a shape or scene as a
    field over space, and recover the actual geometry only implicitly, as
    wherever that field crosses a threshold — reappears constantly later in
    computer vision and graphics, at increasing levels of abstraction:

    - 3D shape models built from a **signed distance function** or an
      **occupancy field** on a voxel grid, with the surface extracted (e.g., by
      marching cubes) only when needed for rendering.
    - **Neural implicit representations** (DeepSDF, occupancy networks, NeRF):
      replace the grid with a neural network $f_\theta(x)$ evaluated at
      continuous 3D query points. DeepSDF and occupancy networks still define
      a surface as an iso-level (0 for signed distance, 0.5 for occupancy);
      NeRF instead predicts a continuous density and color at each point and
      *integrates* through it along a ray, rather than thresholding a single
      surface — a genuine variant on the idea, not just the same equation
      relabeled, but built on exactly the same premise: don't store the shape,
      store a function that implicitly defines it.

    The payoff in every one of these cases is the same one this section
    demonstrates concretely below: representation complexity is decoupled from
    shape complexity. One field (one grid, or one trained network) represents
    an arbitrary — and potentially *changing* — number of disconnected pieces,
    with no change to the data structure and no bookkeeping.

    **Evolving $\phi$.** Instead of moving points, the whole field evolves
    according to a PDE. Szeliski's geodesic active contour is one example,

    $$ \frac{d\phi}{dt} = g(I)|\nabla\phi|\,\text{div}\!\left(\frac{\nabla\phi}{|\nabla\phi|}\right) + \nabla g(I)\cdot\nabla\phi, \qquad (7.38) $$

    where $g(I)$ is a generalized edge potential (small near strong edges,
    large in flat regions). The first term straightens the curve according to
    its own curvature, modulated by $g(I)$; the second pulls it toward minima
    of $g(I)$, i.e., toward edges.

    **A more robust variant, for the demo below.** $g(I)$ is gradient-based,
    which is fragile on a real, textured photo — a coin's engraved face has
    plenty of internal gradient competing with its true outline. The book's own
    text in this section names the fix: recast the energy in terms of *region*
    statistics instead of raw gradients (Chan and Vese 2001). Concretely, using
    the same $\phi<0$-inside convention, evolve

    $$ \frac{d\phi}{dt} = \delta_\epsilon(\phi)\Big[\mu\,\kappa + \lambda_1(I-c_1)^2 - \lambda_2(I-c_2)^2\Big], $$

    where $c_1, c_2$ are the mean image intensity **currently** inside and
    outside the curve (recomputed every step), $\kappa$ is the curvature term
    from Eq. 7.38, and $\delta_\epsilon(\phi)$ is a smoothed Dirac delta that's
    large only within a few pixels of the *current* zero level set (so, unlike
    Eq. 7.38's raw gradient term, texture far from the boundary can't
    spuriously tug the curve — the same robustness argument, just applied to
    the update rule instead of only the stopping criterion). A pixel that looks
    more like the inside average than the outside average pulls the curve
    toward itself, and vice versa — no gradients involved at all. This
    formulation still changes topology exactly as freely as Eq. 7.38; the
    substitution is purely for robustness on a real photo, not a special
    property of Chan-Vese.

    Try it below: a single initial curve, drawn as one big loop enclosing
    *two* separate objects, evolves under this rule. Watch it pinch at the
    gap between them and split into two independent closed curves — with the
    same update rule running unchanged the entire time.
    """)
    return


@app.function
def signed_distance_from_mask(inside_mask, distance_transform_edt):
    """A true signed-distance embedding for the given inside/outside mask
    (negative inside, per this section's convention) -- also how we
    periodically re-anchor phi to satisfy the Eikonal equation |grad phi|=1."""
    d_in = distance_transform_edt(inside_mask)
    d_out = distance_transform_edt(~inside_mask)
    return d_out - d_in


@app.function
def level_set_curvature(phi, np, eps_grad=1e-6):
    """kappa = div(grad(phi)/|grad(phi)|), expanded via the quotient rule.
    np.gradient uses one-sided differences at the boundary, so this is safe
    even when phi's domain touches a real image edge (unlike a periodic
    np.roll, which would wrap around)."""
    phi_y, phi_x = np.gradient(phi)
    phi_xy = np.gradient(phi_x, axis=0)
    phi_xx = np.gradient(phi_x, axis=1)
    phi_yy = np.gradient(phi_y, axis=0)
    denom = (phi_x**2 + phi_y**2 + eps_grad) ** 1.5
    return (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / denom


@app.function
def chan_vese_evolve(I01, phi0, mu, dt, n_iter, reinit_every, np, distance_transform_edt, eps_delta=1.5):
    """Evolve phi under the region-based (Chan-Vese-style) speed function for
    n_iter steps, periodically re-anchoring phi to a true signed-distance
    function. I01 is the image intensity normalized to [0,1]."""
    phi = phi0.copy()
    for i in range(n_iter):
        inside = phi < 0
        c1 = I01[inside].mean() if inside.any() else 0.0
        c2 = I01[~inside].mean() if (~inside).any() else 0.0
        kappa = level_set_curvature(phi, np)
        delta = (1.0 / np.pi) * eps_delta / (eps_delta**2 + phi**2)
        speed = mu * kappa + (I01 - c1) ** 2 - (I01 - c2) ** 2
        phi = phi + dt * delta * speed
        if i % reinit_every == reinit_every - 1:
            phi = signed_distance_from_mask(phi < 0, distance_transform_edt)
    return phi


@app.function
def make_two_blob_scene(H, W, c1_pos, r1, c2_pos, r2, np):
    """Two separated circular 'objects' of contrasting intensity, for a
    guaranteed-clean synthetic version of the topology-split demo."""
    yy, xx = np.mgrid[0:H, 0:W]
    blob1 = (xx - c1_pos[1]) ** 2 + (yy - c1_pos[0]) ** 2 < r1**2
    blob2 = (xx - c2_pos[1]) ** 2 + (yy - c2_pos[0]) ** 2 < r2**2
    img = np.where(blob1 | blob2, 200.0, 30.0)
    rng = np.random.default_rng(2)
    return np.clip(img + rng.normal(0, 8, img.shape), 0, 255)


@app.cell
def _(mo):
    ls_scene_dropdown = mo.ui.dropdown(
        options=["two coins", "synthetic two blobs"], value="two coins", label="scene"
    )
    ls_mu_slider = mo.ui.slider(start=0.02, stop=0.5, value=0.15, step=0.02, label="mu (curvature weight)", debounce=True)
    ls_iter_slider = mo.ui.slider(start=0, stop=900, value=300, step=20, label="iterations", debounce=True)
    mo.vstack([ls_scene_dropdown, mo.hstack([ls_mu_slider, ls_iter_slider], justify="start", gap=2)])
    return ls_iter_slider, ls_mu_slider, ls_scene_dropdown


@app.cell
def _(
    IMAGES_DIR,
    distance_transform_edt,
    ls_iter_slider,
    ls_mu_slider,
    ls_scene_dropdown,
    mo,
    np,
    plt,
):
    if ls_scene_dropdown.value == "two coins":
        # Measured directly on images/coins.png: the bottom-left pair of coins,
        # separated by ~18px of background, with no other coin in the crop.
        _crop = to_luma(load_rgb255(IMAGES_DIR, "coins", plt, np))[210:300, 0:155]
        _cy0, _cx0, _a, _b = 48, 76, 64, 36
        _dt, _reinit_every = 8.0, 20
    else:
        _crop = make_two_blob_scene(70, 120, (35, 30), 18, (35, 90), 15, np)
        _cy0, _cx0, _a, _b = 35, 60, 55, 22
        _dt, _reinit_every = 5.0, 20

    _H, _W = _crop.shape
    _I01 = _crop / 255.0
    _yy, _xx = np.mgrid[0:_H, 0:_W]
    _inside0 = ((_xx - _cx0) / _a) ** 2 + ((_yy - _cy0) / _b) ** 2 < 1
    _phi0 = signed_distance_from_mask(_inside0, distance_transform_edt)

    _phi = chan_vese_evolve(
        _I01, _phi0, ls_mu_slider.value, _dt, ls_iter_slider.value, _reinit_every, np, distance_transform_edt
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _axes[0].imshow(_phi, cmap="RdBu")
    _axes[0].contour(_phi, levels=[0], colors="black", linewidths=1.5)
    _axes[0].set_title(r"$\phi$ (red < 0 = inside, blue > 0 = outside)")
    _axes[1].imshow(_crop, cmap="gray")
    _axes[1].contour(_phi, levels=[0], colors="lime", linewidths=2)
    _axes[1].set_title(f"zero level set, iteration {ls_iter_slider.value}")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md(
            "Scrub the iteration slider from 0: the single initial loop "
            "contracts, pinches at the gap between the two objects, and "
            "splits into two independent closed curves -- the exact same "
            "update rule the whole time, with no special-casing for the "
            "moment of the split."
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Segmentation (§7.5)

    **Segmentation** asks a different question than active contours: instead of
    refining one curve around one object, partition the *entire* image into
    regions that "go together." The simplest approaches — a single intensity
    threshold, or greedily merging neighboring pixels whose colors are close
    enough — go back to the earliest days of computer vision, and remain too
    fragile for real images (lighting variation alone defeats a single
    threshold). Below are four strategies spanning very different ideas of what
    "go together" should mean: **k-means** (plain color clustering, §5.2.2),
    **watershed** (flooding a gradient/distance landscape, §7.5), **mean shift**
    mode-finding (§7.5.2), and **graph-based** merging (§7.5.1). As the book
    itself notes, several classical segmentation algorithms are no longer
    state-of-the-art for whole-image segmentation, but they remain
    foundational — mean shift and graph-based merging in particular are still
    widely used today as **superpixel** generators, a preprocessing step for
    other algorithms.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. K-means clustering (§5.2.2)

    The simplest way to segment by color is to forget about pixel layout
    entirely and just **cluster** the pixels as plain color vectors — the
    generic vector-space clustering the book covers in §5.2.2, applied here to
    segmentation. Given $K$ (chosen in advance), **k-means** alternates two
    steps to minimize the within-cluster sum of squared distances,

    $$ J = \sum_{k=1}^K \sum_{x_i \in S_k} \|x_i - \mu_k\|^2. $$

    **Assignment** — each point joins its nearest centroid:

    $$ z_i = \arg\min_k \|x_i - \mu_k\|^2. $$

    **Update** — each centroid moves to the mean of its assigned points:

    $$ \mu_k = \frac{1}{|S_k|}\sum_{i \,:\, z_i = k} x_i. $$

    This is Lloyd's algorithm: each step can only decrease $J$, so it always
    converges — to *a* local minimum, not necessarily the global one, which is
    why real implementations restart from several random initializations and
    keep the best.

    As the book's own distinction puts it, clustering "usually ignores pixel
    layout and neighborhoods, while segmentation relies heavily on spatial
    cues." Color-only k-means is a pure instance of the former: two pixels get
    the same label purely because they're similarly colored, however far apart
    they are in the image. Try it below, and compare against mean shift (§6) —
    color-only k-means can leave same-colored pixels scattered across noisy,
    spatially incoherent patches, especially in textured regions, exactly the
    shortcoming mean shift's *joint* spatial+color feature space is built to
    fix.
    """)
    return


@app.function
def kmeans_segment(rgb, k, n_iter, seed, np):
    """Lloyd's algorithm in CIELAB color space only (no spatial coordinates),
    to keep this a pure instance of vector-space clustering per §5.2.2."""
    lab = rgb255_to_lab(rgb, np)
    h, w, _ = lab.shape
    X = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = X[m].mean(axis=0)
    return labels.reshape(h, w)


@app.cell
def _(mo):
    km_image_dropdown = mo.ui.dropdown(
        options=["coffee", "astronaut", "chelsea", "raccoon", "coins"], value="coffee", label="image"
    )
    km_k_slider = mo.ui.slider(start=2, stop=10, value=5, step=1, label="K (number of clusters)", debounce=True)
    mo.vstack([km_image_dropdown, km_k_slider])
    return km_image_dropdown, km_k_slider


@app.cell
def _(IMAGES_DIR, km_image_dropdown, km_k_slider, mo, np, plt):
    _img = load_rgb255(IMAGES_DIR, km_image_dropdown.value, plt, np)
    _small = block_downsample(_img, 60, np)
    _labels = kmeans_segment(_small, km_k_slider.value, 15, 0, np)

    _posterized = _small.copy()
    _flat = _posterized.reshape(-1, 3)
    _lab_flat = _labels.ravel()
    for _c in range(km_k_slider.value):
        _m = _lab_flat == _c
        if _m.any():
            _flat[_m] = _small.reshape(-1, 3)[_m].mean(axis=0)

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.8))
    _axes[0].imshow(_small.astype(np.uint8))
    _axes[0].set_title("downsampled input")
    _axes[1].imshow(_labels, cmap="tab10")
    _axes[1].set_title(f"k-means labels (K={km_k_slider.value})")
    _axes[2].imshow(_posterized.astype(np.uint8))
    _axes[2].set_title("posterized (mean color/cluster)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Watershed

    A completely different strategy: treat an image as a **height field** and
    ask where rain would collect. Every local minimum defines a **catchment
    basin** — the points that drain to it — and the **watershed lines** are the
    ridges separating neighboring basins (Vincent and Soille 1991). Formally,
    the basin of a minimum $m_i$ is the set of points closer to $m_i$ than to
    any other minimum, in a *topographical* distance measuring the highest
    point you'd have to cross along the best path, not straight-line distance:

    $$ C(m_i) = \{p : T(p, m_i) < T(p, m_j) \ \forall j \ne i\}, \qquad T(p,q) = \min_{\pi:\,p\to q}\ \max_{t\in\pi} g(t), $$

    for a landscape $g$. Read $T$ one piece at a time: for one specific path
    $\pi$, $\max_{t\in\pi} g(t)$ is the highest point you're forced to climb
    over somewhere along that route — its worst obstacle, ignoring everything
    else about the path (length, how deep its valleys are). $T(p,q)$ then
    minimizes that worst obstacle over *every* possible route — the elevation
    of the best mountain pass connecting $p$ and $q$, however winding the route
    that finds it. Two points can be geometrically close but topographically
    *far* if a tall ridge stands between them, and geometrically far but
    topographically *close* if a low valley connects them.

    This is exactly the right notion for flooding: if water starts rising from
    minimum $m_i$, it reaches point $p$ precisely when the water level hits the
    lowest pass separating them — i.e., at water level $T(p, m_i)$. So $C(m_i)$
    really does mean "the points whose flood from $m_i$ arrives before any
    other minimum's flood." The efficient way to compute this matches the
    physical picture exactly: flood the landscape from every minimum
    simultaneously, in order of increasing elevation (via a priority queue —
    this processes each pixel exactly at its true $T$ value, without ever
    computing $T$ explicitly), and wherever two different floods would merge,
    build a dam — a watershed line — instead.

    **A classic use: separating touching objects** that a simple threshold
    would merge into one blob. Compute the distance transform $D$ of the
    foreground mask (distance to the nearest background pixel — reusing
    `distance_transform_edt` from the level-set solver, §2), treat $g = -D$ as
    the landscape (each object's own center, the point farthest from any edge,
    becomes a local minimum of $g$), and flood outward from those centers.

    Try the slider below: too small a minimum marker spacing lets a single
    object's noisy boundary create several spurious nearby minima, splitting it
    into extra tiny regions — **over-segmentation** is a real, common failure
    mode here, not just a hypothetical one.
    """)
    return


@app.function
def watershed_from_markers(landscape, markers, domain_mask, heapq, np):
    """Priority-flood watershed (Vincent and Soille 1991): flood outward from
    each marker in order of increasing landscape elevation; whichever marker's
    flood reaches a pixel first claims it."""
    H, W = landscape.shape
    labels = np.where(domain_mask, markers, -1)
    heap = []
    counter = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(H):
        for x in range(W):
            if labels[y, x] > 0:
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0:
                        heapq.heappush(heap, (landscape[ny, nx], counter, ny, nx, labels[y, x]))
                        counter += 1
    while heap:
        _, _, y, x, lab = heapq.heappop(heap)
        if labels[y, x] != 0:
            continue
        labels[y, x] = lab
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0:
                heapq.heappush(heap, (landscape[ny, nx], counter, ny, nx, lab))
                counter += 1
    return np.where(labels < 0, 0, labels)


@app.function
def make_touching_blobs(H, W, centers, np):
    """Several overlapping/touching circular objects -- a simple threshold
    merges these into one blob, which is exactly the case
    watershed-on-the-distance-transform is for."""
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.zeros((H, W), dtype=bool)
    for cy, cx, r in centers:
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
    return mask


@app.cell
def _(mo):
    ws_mindist_slider = mo.ui.slider(start=3, stop=21, value=9, step=2, label="minimum marker spacing", debounce=True)
    ws_mindist_slider
    return (ws_mindist_slider,)


@app.cell
def _(
    distance_transform_edt,
    label,
    maximum_filter,
    mo,
    np,
    plt,
    ws_mindist_slider,
):
    import heapq as _heapq

    _H, _W = 90, 160
    _centers = [(40, 25, 20), (38, 60, 24), (48, 100, 19), (45, 135, 17)]
    _mask = make_touching_blobs(_H, _W, _centers, np)
    _rng = np.random.default_rng(0)
    _noise = _rng.normal(0, 1.2, (_H, _W))
    _mask_noisy = _mask & (distance_transform_edt(_mask) + _noise > 0.5)

    _D = distance_transform_edt(_mask_noisy)
    _local_max = (_D == maximum_filter(_D, size=ws_mindist_slider.value)) & (_D > 2)
    _markers, _n_markers = label(_local_max)
    _labels = watershed_from_markers(-_D, _markers, _mask_noisy, _heapq, np)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _axes[0].imshow(_mask_noisy, cmap="gray")
    _axes[0].set_title("merged mask (touching objects)")
    _axes[1].imshow(_D, cmap="viridis")
    _axes[1].set_title("distance transform + markers")
    _ys, _xs = np.nonzero(_local_max)
    _axes[1].scatter(_xs, _ys, c="red", s=10)
    _axes[2].imshow(_labels, cmap="tab10")
    _axes[2].set_title(f"watershed split (n={_n_markers})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Watershed as a superpixel generator.** The same flooding rule works even
    without a binary mask: seed markers on a plain regular grid across the
    *whole* image, and flood the (smoothed) gradient magnitude as the
    landscape (notebook 15's edge strength, reused once more). Because the
    landscape is real image structure rather than empty space, the resulting
    regions still start out grid-sized but their boundaries snap to actual
    edges wherever the grid falls near one — unlike a plain grid, but more
    regular than graph-based merging's adaptively-sized regions (§7). This is
    essentially the classic "watershed of the gradient image" superpixel
    technique, a direct ancestor of the modern SLIC algorithm (Achanta et al.
    2012), which follows the same recipe with markers refined in a combined
    color+space feature space instead of a fixed grid.

    **Why generate superpixels at all?** Treat each one as a single unit
    instead of hundreds of individual pixels, on the assumption that pixels
    inside one superpixel almost always share the same depth, motion, or
    object label — a very safe bet, since that's exactly what "go together"
    meant when the superpixels were formed. The book cites this as a practical
    speedup for **stereo matching** (estimate one disparity per superpixel,
    not per pixel), **optical flow** (one motion vector per superpixel), and
    **recognition** (classify superpixels instead of every pixel) — each
    becomes both faster and more robust to noise, since a superpixel's
    aggregate statistics are far less noisy than any single pixel's. The demo
    below makes this concrete: the third panel replaces every pixel with its
    superpixel's mean color, and the title reports how many numbers that
    took — a large reduction from one value per pixel to one per region.

    **What the "superpixel size" slider does.** It's the spacing between
    adjacent grid markers, in pixels — directly the side length of each
    starting cell before its boundaries get nudged toward nearby edges. Two
    things follow immediately from that one number: a smaller spacing places
    *more* markers (roughly $(H/\text{step})\times(W/\text{step})$ of them), so
    more, smaller superpixels that track fine detail but each average over
    fewer pixels (noisier per-region color, less compression); a larger
    spacing places fewer, bigger superpixels — more aggressive compression,
    but a superpixel that size can straddle a real object boundary if it falls
    entirely inside one grid cell, since flooding can only bend a cell's edge
    toward a gradient it actually reaches, not invent a split through the
    middle of a flat region.
    """)
    return


@app.cell
def _(mo):
    ws_sp_image_dropdown = mo.ui.dropdown(
        options=["coffee", "astronaut", "chelsea", "raccoon", "coins"], value="coffee", label="image"
    )
    ws_sp_step_slider = mo.ui.slider(start=6, stop=30, value=12, step=2, label="grid spacing (superpixel size)", debounce=True)
    mo.vstack([ws_sp_image_dropdown, ws_sp_step_slider])
    return ws_sp_image_dropdown, ws_sp_step_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    mark_boundaries,
    mo,
    np,
    plt,
    ws_sp_image_dropdown,
    ws_sp_step_slider,
):
    import heapq as _heapq2

    _img = load_rgb255(IMAGES_DIR, ws_sp_image_dropdown.value, plt, np)
    _small = block_downsample(_img, 120, np)
    _gray = to_luma(_small)
    _Ix, _Iy = compute_gradients(_gray, 1.0, gaussian_filter, correlate2d)
    _mag = gaussian_filter(np.hypot(_Ix, _Iy), sigma=1.0)

    _H, _W = _gray.shape
    _step = ws_sp_step_slider.value
    _markers = np.zeros((_H, _W), dtype=int)
    _lbl = 1
    for _gy in range(_step // 2, _H, _step):
        for _gx in range(_step // 2, _W, _step):
            _markers[_gy, _gx] = _lbl
            _lbl += 1
    _n_markers = _lbl - 1

    _labels = watershed_from_markers(_mag, _markers, np.ones((_H, _W), dtype=bool), _heapq2, np)

    _compressed = _small.copy()
    _flat = _compressed.reshape(-1, 3)
    _labels_flat = _labels.ravel()
    for _s in np.unique(_labels):
        _m = _labels_flat == _s
        _flat[_m] = _small.reshape(-1, 3)[_m].mean(axis=0)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _axes[0].imshow(_small.astype(np.uint8))
    _axes[0].set_title("input")
    _axes[1].imshow(mark_boundaries(_small.astype(np.uint8) / 255.0, _labels))
    _axes[1].set_title(f"watershed superpixels (n={_n_markers})")
    _axes[2].imshow(_compressed.astype(np.uint8))
    _axes[2].set_title("superpixel-compressed")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Mean shift (§7.5.2)

    Treat every pixel as a sample point in a joint **feature space**: its
    spatial position $(x, y)$ concatenated with its color. (The book uses
    CIELUV; we reuse notebook 7's **CIELAB** instead — both are perceptually
    motivated color spaces, and the mode-seeking mechanics below don't care
    which one we pick.) The pixels in this 5D space form clusters wherever many
    similarly-colored, nearby pixels pile up — imagine the whole image as a
    scatter of points, most of them jumbled together into a handful of dense
    "clouds."

    **Mean shift** finds these clouds without ever computing the full density
    explicitly, and — unlike k-means (§4) — without being told $K$ in advance.

    Split every feature vector into its two pieces, spatial and range
    (color): $x_i = (x_i^s, x_i^r)$, where $x_i^s = (x_i, y_i) \in \mathbb{R}^2$
    is pixel $i$'s position and $x_i^r \in \mathbb{R}^3$ is its CIELAB color;
    write the current mode estimate the same way, $y = (y^s, y^r)$. Since
    position and color live on totally different scales, each piece gets its
    own bandwidth ($h_s$ for space, $h_r$ for color, exactly as in the
    bilateral filter, notebook 8 §3.3.2), via the normalized offsets

    $$ u_s = \frac{y^s - x_i^s}{h_s} \in \mathbb{R}^2, \qquad u_r = \frac{y^r - x_i^r}{h_r} \in \mathbb{R}^3. $$

    The kernel is then a **product of two separate Gaussians, one per piece**,

    $$ K(u_s, u_r) = K_s(u_s)\,K_r(u_r), \qquad K_s(u_s) = e^{-\|u_s\|^2/2}, \quad K_r(u_r) = e^{-\|u_r\|^2/2}, $$

    so that a pair of points only gets a large weight when they're close in
    *both* pieces at once — this is exactly what lets $h_s$ and $h_r$ be tuned
    independently below. Starting each point $y$ at its own position,
    repeatedly replace it with this kernel's weighted mean of *every* original
    point $x_i$ (with $u_{s,i}, u_{r,i}$ denoting $u_s, u_r$ evaluated against
    that $x_i$):

    $$ y \leftarrow \frac{\sum_i x_i\,K_s(u_{s,i})\,K_r(u_{r,i})}{\sum_i K_s(u_{s,i})\,K_r(u_{r,i})}. $$

    This is **gradient ascent** on the (implicit) density estimate: each step
    climbs uphill toward the nearest mode. Points that converge to (nearly) the
    same location are grouped into one segment — and because position is
    baked directly into $x^s$, unlike k-means's color-only feature vector, the
    resulting segments are spatially coherent by construction.

    Try the sliders below: a larger color bandwidth $h_r$ merges more distinct
    colors into one segment; a larger spatial bandwidth $h_s$ favors big,
    spatially compact regions over small, scattered ones. (For speed, the image
    is heavily downsampled first — full mean shift is a "brute force" $O(n^2)$
    per iteration algorithm over the *number of pixels*, exactly the same
    tradeoff notebook 14 made for patch matching and notebook 13 made for the
    auto-correlation surface.)
    """)
    return


@app.function
def block_downsample(img, target_w, np):
    """Crop to a size divisible by the target grid, then average each block --
    an anti-aliased alternative to naive pixel striding."""
    H, W = img.shape[:2]
    target_h = max(1, int(round(H * (target_w / W))))
    bs_h, bs_w = H // target_h, W // target_w
    cropped = img[: bs_h * target_h, : bs_w * target_w]
    if img.ndim == 3:
        return cropped.reshape(target_h, bs_h, target_w, bs_w, img.shape[2]).mean(axis=(1, 3))
    return cropped.reshape(target_h, bs_h, target_w, bs_w).mean(axis=(1, 3))


@app.function
def mean_shift_segment(small_rgb, hs, hr, n_iter, np, cKDTree, coo_matrix, connected_components):
    """Mean shift mode-finding in joint (x, y, L, a, b) space. Returns a (h, w)
    integer label map, one label per converged mode."""
    h, w, _ = small_rgb.shape
    lab = rgb255_to_lab(small_rgb, np)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.stack(
        [xx.ravel() / hs, yy.ravel() / hs, lab[..., 0].ravel() / hr, lab[..., 1].ravel() / hr, lab[..., 2].ravel() / hr],
        axis=1,
    ).astype(np.float32)
    Y = F.copy()
    for _ in range(n_iter):
        diff = Y[:, None, :] - F[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        K = np.exp(-0.5 * d2)
        Ynew = (K @ F) / K.sum(axis=1, keepdims=True)
        if np.abs(Ynew - Y).max() < 1e-3:
            Y = Ynew
            break
        Y = Ynew

    n_pts = Y.shape[0]
    pairs = cKDTree(Y).query_pairs(r=0.5, output_type="ndarray")
    if len(pairs) == 0:
        labels = np.arange(n_pts)
        n_comp = n_pts
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n_pts)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n_pts)])
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pts, n_pts))
        n_comp, labels = connected_components(adj, directed=False)
    return labels.reshape(h, w), n_comp


@app.cell
def _(mo):
    ms_image_dropdown = mo.ui.dropdown(
        options=["coffee", "astronaut", "chelsea", "raccoon", "coins"], value="coffee", label="image"
    )
    ms_hs_slider = mo.ui.slider(start=2.0, stop=15.0, value=6.0, step=1.0, label="spatial bandwidth h_s", debounce=True)
    ms_hr_slider = mo.ui.slider(start=4.0, stop=25.0, value=12.0, step=1.0, label="color bandwidth h_r", debounce=True)
    mo.vstack([ms_image_dropdown, mo.hstack([ms_hs_slider, ms_hr_slider], justify="start", gap=2)])
    return ms_hr_slider, ms_hs_slider, ms_image_dropdown


@app.cell
def _(
    IMAGES_DIR,
    cKDTree,
    connected_components,
    coo_matrix,
    mo,
    ms_hr_slider,
    ms_hs_slider,
    ms_image_dropdown,
    np,
    plt,
):
    _img = load_rgb255(IMAGES_DIR, ms_image_dropdown.value, plt, np)
    _small = block_downsample(_img, 45, np)
    _labels, _n_comp = mean_shift_segment(
        _small, ms_hs_slider.value, ms_hr_slider.value, n_iter=20,
        np=np, cKDTree=cKDTree, coo_matrix=coo_matrix, connected_components=connected_components,
    )

    _posterized = _small.copy()
    _flat = _posterized.reshape(-1, 3)
    _lab_flat = _labels.ravel()
    for _c in range(_n_comp):
        _m = _lab_flat == _c
        _flat[_m] = _small.reshape(-1, 3)[_m].mean(axis=0)

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.8))
    _axes[0].imshow(_small.astype(np.uint8))
    _axes[0].set_title("downsampled input")
    _axes[1].imshow(_labels, cmap="tab20")
    _axes[1].set_title(f"mean-shift modes (n={_n_comp})")
    _axes[2].imshow(_posterized.astype(np.uint8))
    _axes[2].set_title("posterized (mean color/mode)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Graph-based segmentation (§7.5.1)

    Treat the image as a graph: one node per pixel, edges between neighbors
    weighted by dissimilarity (e.g., color difference). Felzenszwalb and
    Huttenlocher (2004) merge two regions only when the dissimilarity *between*
    them is small relative to the dissimilarity *within* each one already — a
    relative, adaptive criterion, rather than one fixed global threshold. This
    lets it correctly keep a highly-textured small region intact even when its
    internal variation exceeds a neighboring region's boundary contrast (exactly
    the example in the book's Figure 7.52).

    Rather than re-deriving the union-find merging procedure here, the demo
    below calls the widely-used reference implementation directly (available in
    `skimage.segmentation`) — this is one of the "still very much in active use"
    algorithms, mainly today as a fast superpixel generator (see §5 for why
    generating superpixels is useful in the first place — the same reasoning
    applies here). Increase `scale` for fewer, larger regions; `min_size`
    enforces a floor on region size after merging. As before, the third panel
    replaces every pixel with its superpixel's mean color, and the title
    reports how many numbers that took — a large reduction from one value per
    pixel to one per region.

    *(A related probabilistic variant, segmentation by weighted aggregation
    (SWA), merges regions hierarchically using both gray-level and texture
    similarity — see §7.5.1 for details.)*
    """)
    return


@app.cell
def _(mo):
    fh_image_dropdown = mo.ui.dropdown(
        options=["coffee", "astronaut", "chelsea", "raccoon", "coins"], value="coffee", label="image"
    )
    fh_scale_slider = mo.ui.slider(start=20, stop=500, value=150, step=10, label="scale", debounce=True)
    fh_sigma_slider = mo.ui.slider(start=0.1, stop=2.0, value=1.0, step=0.1, label="pre-smoothing sigma", debounce=True)
    fh_minsize_slider = mo.ui.slider(start=10, stop=300, value=80, step=10, label="min_size", debounce=True)
    mo.vstack([
        fh_image_dropdown,
        mo.hstack([fh_scale_slider, fh_sigma_slider, fh_minsize_slider], justify="start", gap=2),
    ])
    return (
        fh_image_dropdown,
        fh_minsize_slider,
        fh_scale_slider,
        fh_sigma_slider,
    )


@app.cell
def _(
    IMAGES_DIR,
    felzenszwalb,
    fh_image_dropdown,
    fh_minsize_slider,
    fh_scale_slider,
    fh_sigma_slider,
    mark_boundaries,
    mo,
    np,
    plt,
):
    _img = load_rgb255(IMAGES_DIR, fh_image_dropdown.value, plt, np) / 255.0
    _segs = felzenszwalb(_img, scale=fh_scale_slider.value, sigma=fh_sigma_slider.value, min_size=fh_minsize_slider.value)
    _n_segs = len(np.unique(_segs))

    _compressed = _img.copy()
    _flat = _compressed.reshape(-1, 3)
    _segs_flat = _segs.ravel()
    for _s in np.unique(_segs):
        _m = _segs_flat == _s
        _flat[_m] = _img.reshape(-1, 3)[_m].mean(axis=0)

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _axes[0].imshow(_img)
    _axes[0].set_title("input")
    _axes[1].imshow(mark_boundaries(_img, _segs))
    _axes[1].set_title(f"{_n_segs} regions")
    _axes[2].imshow(_compressed)
    _n_pixels = _img.shape[0] * _img.shape[1]
    _axes[2].set_title(f"superpixel-compressed\n{_n_pixels} pixels -> {_n_segs} values")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Snakes** (active contours) minimize an energy combining internal
      smoothness (Eqs. 7.26–7.27) with an external image term (Eq. 7.28) —
      here, notebook 15's gradient magnitude — using a greedy local search.
    - **Level sets** replace the explicit curve with an implicit one (the
      zero-crossing of an evolving embedding function), handling topology
      changes for free at the cost of solving a PDE (Eq. 7.38).
    - **Segmentation** partitions the whole image rather than tracking one
      boundary, via four different ideas of what "go together" means:
      **k-means** clusters plain color vectors (Lloyd's algorithm); **watershed**
      floods a distance/gradient landscape from markers, built on the same
      `distance_transform_edt` as the level-set solver; **mean shift** finds
      modes in a *joint* spatial+color feature space, fixing k-means's
      spatial-incoherence problem; **graph-based merging** grows regions using a
      relative dissimilarity criterion.
    - Mean shift and graph-based merging are the two still in wide practical
      use today, mainly as fast superpixel generators.

    **Next up:** fitting simple geometric primitives — lines and circles — to
    edge points, first by voting (the Hough transform) and then by robust
    random sampling (RANSAC).
    """)
    return


if __name__ == "__main__":
    app.run()
