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
    from scipy.signal import correlate2d
    from scipy.ndimage import gaussian_filter, maximum_filter

    return correlate2d, gaussian_filter, maximum_filter, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return IMAGES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Feature Detection

    Every notebook so far worked with a *whole* image at once. But many of
    the most useful things you can do with an image — stitch two photos
    into a panorama, track a point across a video, recognize an object
    seen from a different angle, or recover 3D structure from multiple
    views — all reduce to the same underlying question: **which small
    patches of this image can I reliably find again in another image?**

    A **feature** is exactly that: a small, distinctive, relocatable patch
    (and, later, a compact numerical description of it) that lets two
    images be related to each other without comparing them pixel by pixel.
    Szeliski's Chapter 7 builds a three-stage pipeline around this idea —
    **detect** stable points → **describe** each one as a vector →
    **match** descriptors between images — that essentially every later
    chapter in the book (stitching, motion estimation, structure from
    motion, 3D reconstruction) leans on. This notebook is stage one:
    finding the *where*, before a future notebook covers *what* (§7.1.2,
    descriptors) and *correspondence* (§7.1.3, matching). A detector that
    isn't stable and repeatable undermines everything built on top of it,
    which is why the chapter — and this notebook — spends real effort on
    it.

    This notebook covers Szeliski **§7.1.1, "Feature detectors,"**
    jumping ahead from notebook 12's Chapter 3 material — **Chapters 4–6**
    (model fitting/optimization, deep learning, recognition) are outside
    this course's scope. Selectively, it covers:

    1. Image gradients as a vector field (a lead-in not explicitly in the
       book's own structure, but the direct prerequisite for everything
       below)
    2. A short eigenvalue/eigenvector primer (not assumed knowledge)
    3. The auto-correlation matrix (Eqs. 7.1–7.8) and what its eigenvalues mean
    4. The Harris corner detector (Eq. 7.9)
    5. Scale-invariant detection via the Difference-of-Gaussians scale-space
    6. Orientation estimation

    A few real parts of §7.1.1 are skipped as not essential to anything
    downstream: alternative corner-response formulas (Triggs' variant, the
    harmonic mean — one sentence, not a section), adaptive non-maximal
    suppression, repeatability measurement methodology, affine invariance
    / MSER, and the two literature-survey lists of more recent hand-crafted
    and learned detectors. No Predict/Investigate/Modify activities this
    time — just demos.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-12)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(np):
    # A normalized Sobel: dividing by 8 makes the response equal the true
    # intensity-change-per-pixel (verify: on a ramp of slope s, this recovers
    # exactly s), unlike notebook 8's raw kernel (which is 8x too large) — needed
    # here since we're about to do real quantitative eigenvalue arithmetic on it.
    GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    GY = GX.T

    def compute_gradients(img_gray, sigma, gaussian_filter, correlate2d):
        smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect") if sigma > 0 else img_gray
        Ix = correlate2d(smoothed, GX, mode="same", boundary="symm")
        Iy = correlate2d(smoothed, GY, mode="same", boundary="symm")
        return Ix, Iy

    return (compute_gradients,)


@app.cell
def _(np):
    def draw_uncertainty_ellipse(ax, A, target_radius=None, **plot_kwargs):
        """Draw the level set {du : du^T A du = k} of a 2x2 symmetric matrix A —
        an ellipse whose semi-axis along each eigenvector has length sqrt(k/eigenvalue),
        i.e. proportional to (eigenvalue)^-1/2 (Figure 7.6). k is chosen so the
        larger semi-axis equals target_radius, purely for display purposes."""
        evals, evecs = np.linalg.eigh(A)
        evals = np.maximum(evals, 1e-12)
        k = 1.0 if target_radius is None else target_radius**2 * evals[0]
        t = np.linspace(0, 2 * np.pi, 200)
        circle = np.stack([np.cos(t), np.sin(t)])
        semi_axes = np.sqrt(k / evals)
        ellipse = evecs @ (semi_axes[:, None] * circle)
        ax.plot(ellipse[0], ellipse[1], **plot_kwargs)
        return evals, evecs

    return (draw_uncertainty_ellipse,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Image gradients as a vector field

    Notebook 8 §7 introduced $G_x$ and $G_y$ as two separate derivative
    *filters*. Put their outputs together at a single pixel and you get a
    **gradient vector**,

    $$ \nabla I(x,y) = \big(I_x(x,y),\, I_y(x,y)\big), $$

    whose **magnitude** $\|\nabla I\|$ measures how fast intensity is
    changing there (edge strength) and whose **direction** points toward
    the fastest increase in intensity — i.e., *perpendicular* to whatever
    edge passes through that pixel. Viewed this way, every image has an
    associated vector field, one arrow per pixel. That field is exactly
    what the rest of this notebook operates on.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_4_aperture_problem.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.4 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 422, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    A patch with gradients pointing in only *one* direction (an edge) can
    only be localized along that direction — slide it sideways along the
    edge and nothing changes (the classic **aperture problem**, panel (b)
    above). A patch with gradients pointing in *at least two* different
    directions (panel (a)) is fully localizable — exactly the property
    §3 below turns into a precise test.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_grad = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_grad_slider = mo.ui.slider(start=0.0, stop=4.0, value=1.0, step=0.5, label="pre-smoothing σ", debounce=True)
    row_slider_grad = mo.ui.slider(start=60, stop=440, value=150, step=10, label="window center row", debounce=True)
    col_slider_grad = mo.ui.slider(start=60, stop=440, value=200, step=10, label="window center col", debounce=True)
    mo.vstack([image_dropdown_grad, mo.hstack([sigma_grad_slider, row_slider_grad, col_slider_grad], justify="start", gap=2)])
    return (
        col_slider_grad,
        image_dropdown_grad,
        row_slider_grad,
        sigma_grad_slider,
    )


@app.cell
def _(
    IMAGES_DIR,
    col_slider_grad,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_grad,
    mo,
    np,
    plt,
    row_slider_grad,
    sigma_grad_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_grad.value, plt, np))
    _sigma = sigma_grad_slider.value
    _img_smoothed = gaussian_filter(_img_gray, sigma=_sigma, mode="reflect") if _sigma > 0 else _img_gray
    _Ix, _Iy = compute_gradients(_img_gray, _sigma, gaussian_filter, correlate2d)
    _mag = np.sqrt(_Ix**2 + _Iy**2)

    _half = 50
    _H, _W = _img_gray.shape
    _cy = int(np.clip(row_slider_grad.value, _half, _H - _half - 1))
    _cx = int(np.clip(col_slider_grad.value, _half, _W - _half - 1))
    _rows = slice(_cy - _half, _cy + _half)
    _cols = slice(_cx - _half, _cx + _half)

    _step = 5
    _yy, _xx = np.mgrid[_step // 2 : 2 * _half : _step, _step // 2 : 2 * _half : _step]
    _yy_full, _xx_full = _yy + (_cy - _half), _xx + (_cx - _half)

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1]})
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].add_patch(plt.Rectangle((_cx - _half, _cy - _half), 2 * _half, 2 * _half, edgecolor="lime", facecolor="none", lw=1.5))
    _axes[0].set_title("full image (green = window)")

    _axes[1].imshow(_img_smoothed[_rows, _cols], cmap="gray", vmin=0, vmax=255, extent=[0, 2 * _half, 2 * _half, 0])
    _axes[1].quiver(
        _xx, _yy, _Ix[_yy_full, _xx_full], _Iy[_yy_full, _xx_full], _mag[_yy_full, _xx_full],
        cmap="autumn", scale=None, angles="xy", scale_units="xy",
    )
    _axes[1].set_title(f"gradient vector field, {2*_half}×{2*_half} window\n(pre-smoothed, σ={_sigma:.1f})")

    _axes[2].imshow(_mag[_rows, _cols], cmap="inferno")
    _axes[2].set_title("gradient magnitude ‖∇I‖ (same window)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Arrows point toward increasing brightness, scaled and colored by magnitude. Notice they're dense and multi-directional around corners/texture, sparse and one-directional along a straight edge, and near-absent in flat regions."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Eigenvalues, briefly

    The next section needs one piece of linear algebra: what do the
    eigenvalues of a $2\times2$ symmetric matrix mean, geometrically?

    **The problem.** For a symmetric matrix $A$, ask: is there some
    direction $v$ that $A$ doesn't rotate at all — only stretches or
    shrinks? Formally, find a nonzero vector $v$ and a scalar $\lambda$
    such that

    $$ Av = \lambda v \quad\Longleftrightarrow\quad (A-\lambda I)v = 0. $$

    **The solution.** A nonzero $v$ can only satisfy this if
    $(A-\lambda I)$ is singular, i.e. $\det(A-\lambda I)=0$ — a quadratic
    equation in $\lambda$ (the *characteristic equation*). For any real
    symmetric $2\times2$ matrix, this always has two real roots
    $\lambda_0\le\lambda_1$, the **eigenvalues**, each with its own
    **eigenvector** ($v_0,v_1$) — and the two eigenvectors are always
    orthogonal to each other.

    **Example.** Take $A=\begin{bmatrix}3&1\\1&3\end{bmatrix}$. It has
    $\lambda_0=2$ with $v_0=\begin{bmatrix}1\\-1\end{bmatrix}$, and
    $\lambda_1=4$ with $v_1=\begin{bmatrix}1\\1\end{bmatrix}$ (how to find
    these isn't the point here — just check them by direct substitution):
    $Av_1=(3{+}1,\,1{+}3)=(4,4)=4v_1$, and
    $Av_0=(3{-}1,\,1{-}3)=(2,-2)=2v_0$. Along each $v_i$, applying $A$ is
    nothing but multiplying by the number $\lambda_i$ — no rotation, no
    mixing with the other direction.

    **From algebra to geometry.** Write an arbitrary vector in the
    eigenvector basis, $v=c_0v_0+c_1v_1$. Because the eigenvectors are
    orthogonal and $Av_i=\lambda_iv_i$, the quadratic form $v^TAv$ loses
    its cross term entirely:

    $$ v^TAv = c_0^2\lambda_0 + c_1^2\lambda_1. $$

    This is exactly why the level set $\{v : v^TAv = k\}$ — all the
    vectors giving the same fixed value $k$ — is an **ellipse aligned
    with the eigenvectors**: moving along $v_i$ alone ($c_{1-i}=0$),
    reaching that fixed value $k$ takes $|c_i|=\sqrt{k/\lambda_i}$ — a
    *short* distance when $\lambda_i$ is large (a squeezed axis) and a
    *long* distance when $\lambda_i$ is small (an elongated axis). That's
    precisely the ellipse the demo below draws, and precisely the
    $\lambda^{-1/2}$ semi-axis labeling in Figure 7.6.

    In §3, this generic $v$ becomes something concrete: the patch
    displacement $\Delta u$.
    """)
    return


@app.cell
def _(mo):
    randomize_ellipse_button = mo.ui.button(
        label="Randomize matrix A",
        value=0,
        on_click=lambda count: count + 1,
    )
    randomize_ellipse_button
    return (randomize_ellipse_button,)


@app.cell
def _(draw_uncertainty_ellipse, mo, np, plt, randomize_ellipse_button):
    _rng = np.random.default_rng(seed=randomize_ellipse_button.value)
    _lam0 = _rng.uniform(0.3, 1.2)
    _ratio = _rng.uniform(4.0, 10.0)
    _lam1 = _lam0 * _ratio
    _theta = _rng.uniform(0, np.pi)
    _v0 = np.array([np.cos(_theta), np.sin(_theta)])
    _v1 = np.array([-np.sin(_theta), np.cos(_theta)])
    _V = np.stack([_v0, _v1], axis=1)
    _A = _V @ np.diag([_lam0, _lam1]) @ _V.T

    _lim = 2.5
    _grid = np.linspace(-_lim, _lim, 200)
    _V1, _V2 = np.meshgrid(_grid, _grid)
    _Q = _A[0, 0] * _V1**2 + 2 * _A[0, 1] * _V1 * _V2 + _A[1, 1] * _V2**2

    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _im = _ax.imshow(_Q, extent=[-_lim, _lim, -_lim, _lim], origin="lower", cmap="viridis")
    _evals, _evecs = draw_uncertainty_ellipse(_ax, _A, target_radius=None, color="white", lw=2)
    for _i, _lab in [(0, "λ0⁻¹ᐟ²"), (1, "λ1⁻¹ᐟ²")]:
        _len = 1.0 / np.sqrt(_evals[_i])
        _vec = _evecs[:, _i] * _len
        _ax.annotate("", xy=tuple(_vec), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
        _ax.text(*(_vec * 1.1), _lab, color="white", fontsize=10)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.set_xlabel("v₁")
    _ax.set_ylabel("v₂")
    _cbar = _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.set_label("vᵀAv")

    _matrix_str = (
        f"A = ⎡{_A[0,0]:6.2f} {_A[0,1]:6.2f}⎤    λ0 = {_lam0:.2f}\n"
        f"    ⎣{_A[1,0]:6.2f} {_A[1,1]:6.2f}⎦    λ1 = {_lam1:.2f}"
    )
    _ax.set_title(_matrix_str, fontfamily="monospace", fontsize=11, linespacing=1.6)

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md("""
    Click the button a few times. When the two eigenvalues land close
    together, the heatmap's bowl is round and the ellipse is close to a
    circle — equally uncertain in every direction. When they're very
    different, the bowl is a narrow valley and the ellipse stretches into
    a long, thin shape — well localized along one eigenvector, ambiguous
    along the other. §3 makes the connection concrete: this same overlay
    reappears directly on a *real* auto-correlation surface.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The auto-correlation matrix

    Formalizing §1's intuition: compare a patch against itself, shifted by
    a small displacement $\Delta u$,

    $$ E_{AC}(\Delta u) = \sum_i w(x_i)\,[I_0(x_i+\Delta u) - I_0(x_i)]^2 \qquad \text{(Eq. 7.2)}. $$

    Here $w(x_i)$ is a **weighting (window) function**: it decides which
    nearby pixels $x_i$ count toward the sum, and how much. The simplest
    choice — what the demo below uses — is a flat box: $w=1$ inside a
    small patch around the point of interest, $0$ outside. Förstner and
    Harris found a **Gaussian** window (full weight at the center, fading
    toward the edges) works better in practice, since it makes the
    response insensitive to exactly where the patch boundary falls and to
    in-plane rotation — this is the window §4's Harris detector actually
    uses.

    To see exactly where the quadratic form comes from, Taylor-expand
    $I_0(x_i+\Delta u)$ to first order,
    $I_0(x_i+\Delta u) \approx I_0(x_i) + \nabla I_0(x_i)\cdot \Delta u$,
    and substitute it into Eq. 7.2 (Eqs. 7.3–7.6):

    $$
    \begin{aligned}
    E_{AC}(\Delta u) &= \sum_i w(x_i)\,[I_0(x_i+\Delta u) - I_0(x_i)]^2 &\text{(7.3)} \\
    &\approx \sum_i w(x_i)\,[I_0(x_i) + \nabla I_0(x_i)\cdot\Delta u - I_0(x_i)]^2 &\text{(7.4)} \\
    &= \sum_i w(x_i)\,[\nabla I_0(x_i)\cdot\Delta u]^2 &\text{(7.5)} \\
    &= \Delta u^T A\, \Delta u, &\text{(7.6)}
    \end{aligned}
    $$

    The $I_0(x_i)$ terms cancel in going from (7.3) to (7.4)–(7.5), leaving
    only the *linear* part of the change — exactly why this is only an
    approximation, valid for small $\Delta u$. Expanding the squared dot
    product in (7.5), $[\nabla I_0\cdot\Delta u]^2 = \Delta u^T(\nabla
    I_0\nabla I_0^T)\Delta u$, and summing the resulting outer products
    with their weights gives the matrix $A$ from (7.6) explicitly,

    $$ A = w * \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix} \qquad \text{(Eq. 7.8)}, $$

    built from exactly the gradient products §1's vector field gives us —
    $w*(\cdot)$ here means the weighted sum $\sum_i w(x_i)(\cdot)$, i.e.
    correlating each gradient-product image with the window.
    Figure 7.5 shows what $E_{AC}(\Delta u)$ actually looks like for three
    real patches — notice how (b) the flower bed has a sharp, well-defined
    minimum (a **corner**-like patch: both eigenvalues of $A$ large), (c)
    the roof edge has a trough (an **edge**: one eigenvalue much larger
    than the other), and (d) the cloud is nearly flat (**neither**
    eigenvalue is large — nothing to localize against).
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 8.5))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_5_autocorrelation_surfaces.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.5 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 423, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_6_uncertainty_ellipse.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.6 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 425, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The demo below computes the *exact* $E_{AC}(\Delta u)$ for a real
    patch (Eq. 7.2, literally shifting and differencing — no
    approximation) and overlays §2's ellipse, computed from that same
    patch's actual matrix $A$, to check that the quadratic approximation
    really does match the true surface near its minimum. On the
    astronaut image, try a corner (row 280, col 400 — the shuttle
    model's wing/tail junction), an edge (row 360, col 440 — along the
    shuttle's body), and a flat region (row 320, col 460 — the dark
    background) to see all three cases from Figure 7.5.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_ac = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    row_slider_ac = mo.ui.slider(start=40, stop=460, value=280, step=5, label="patch row", debounce=True)
    col_slider_ac = mo.ui.slider(start=40, stop=460, value=400, step=5, label="patch col", debounce=True)
    mo.vstack([image_dropdown_ac, mo.hstack([row_slider_ac, col_slider_ac], justify="start", gap=2)])
    return col_slider_ac, image_dropdown_ac, row_slider_ac


@app.cell
def _(
    IMAGES_DIR,
    col_slider_ac,
    compute_gradients,
    correlate2d,
    draw_uncertainty_ellipse,
    gaussian_filter,
    image_dropdown_ac,
    mo,
    np,
    plt,
    row_slider_ac,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_ac.value, plt, np))
    _H, _W = _img_gray.shape
    _half = 15
    _shift_lim = 8
    _cy = int(np.clip(row_slider_ac.value, _half + _shift_lim, _H - _half - _shift_lim - 1))
    _cx = int(np.clip(col_slider_ac.value, _half + _shift_lim, _W - _half - _shift_lim - 1))

    _patch = _img_gray[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1]
    _du_range = np.arange(-_shift_lim, _shift_lim + 1)
    _EAC = np.zeros((len(_du_range), len(_du_range)))
    for _i, _dv in enumerate(_du_range):
        for _j, _du in enumerate(_du_range):
            _shifted = _img_gray[_cy - _half + _dv : _cy + _half + 1 + _dv, _cx - _half + _du : _cx + _half + 1 + _du]
            _EAC[_i, _j] = np.sum((_shifted - _patch) ** 2)

    _Ix, _Iy = compute_gradients(_img_gray, 0.5, gaussian_filter, correlate2d)
    _ix_patch = _Ix[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1]
    _iy_patch = _Iy[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1]
    _A = np.array([
        [np.sum(_ix_patch**2), np.sum(_ix_patch * _iy_patch)],
        [np.sum(_ix_patch * _iy_patch), np.sum(_iy_patch**2)],
    ])
    _evals = np.linalg.eigvalsh(_A)
    _kind = "corner (both eigenvalues large)" if _evals[0] > 0.15 * _evals[1] else "edge (one eigenvalue much larger)" if _evals[1] > 1e-6 else "flat (both eigenvalues small)"

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].add_patch(plt.Rectangle((_cx - _half, _cy - _half), 2 * _half, 2 * _half, edgecolor="lime", facecolor="none", lw=1.5))
    _axes[0].set_title("patch location")
    _axes[0].axis("off")

    _extent = [-_shift_lim, _shift_lim, _shift_lim, -_shift_lim]
    _axes[1].imshow(_EAC, cmap="viridis", extent=_extent)
    draw_uncertainty_ellipse(_axes[1], _A, target_radius=_shift_lim * 0.7, color="white", lw=2, ls="--")
    _axes[1].set_xlim(-_shift_lim, _shift_lim)
    _axes[1].set_ylim(_shift_lim, -_shift_lim)
    _axes[1].set_xlabel("Δu (x)")
    _axes[1].set_ylabel("Δu (y)")
    _axes[1].set_title(f"E_AC(Δu), true surface\n(dashed: quadratic-form ellipse)")

    mo.vstack([
        mo.md(f"**λ0={_evals[0]:.1f}, λ1={_evals[1]:.1f} → looks like a {_kind}.** The dashed ellipse (from $A$, via §2's exact drawing code) should hug the true surface's minimum closely."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The Harris corner detector

    Computing full eigenvalues everywhere is more work than necessary.
    Förstner and Harris (1988) proposed a cheaper, still rotationally
    invariant, quantity built from just the determinant and trace of $A$:

    $$ \det(A) - \alpha\,\text{trace}(A)^2 = \lambda_0\lambda_1 - \alpha(\lambda_0+\lambda_1)^2 \qquad \text{(Eq. 7.9)}, $$

    with $\alpha=0.06$. This is large and positive only when *both*
    eigenvalues are large (a corner); it's negative along edges (one
    eigenvalue dominates) and small near zero in flat regions. (Other
    combinations of the two eigenvalues have been proposed — Triggs'
    $\lambda_0-\alpha\lambda_1$, a harmonic-mean variant — but they're
    minor variations on the same idea.)

    The demo below computes exactly this quantity at every pixel,
    $R = \det(A) - \alpha\,\text{trace}(A)^2$, using $I_x^2, I_y^2, I_xI_y$
    smoothed by a Gaussian window (the $w$ from §3) to build $A$, keeps
    only local maxima of $R$ above a threshold, and marks those as
    detected corners. $R$'s raw values span many orders of magnitude — a
    handful of very strong corners otherwise wash out everything else on
    a linear color scale — so the response panel displays a **signed
    log**, $\text{sign}(R)\log(1+|R|)$, purely for visibility; detection
    itself still thresholds the real, untransformed $R$.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_harris = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_harris_slider = mo.ui.slider(start=1.0, stop=5.0, value=2.0, step=0.5, label="integration scale σ", debounce=True)
    thresh_harris_slider = mo.ui.slider(start=0.001, stop=0.1, value=0.01, step=0.001, label="response threshold (fraction of max)", debounce=True)
    mo.vstack([image_dropdown_harris, mo.hstack([sigma_harris_slider, thresh_harris_slider], justify="start", gap=2)])
    return image_dropdown_harris, sigma_harris_slider, thresh_harris_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_harris,
    maximum_filter,
    mo,
    np,
    plt,
    sigma_harris_slider,
    thresh_harris_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_harris.value, plt, np))
    _Ix, _Iy = compute_gradients(_img_gray, 0.5, gaussian_filter, correlate2d)

    _sigma = sigma_harris_slider.value
    _Ixx = gaussian_filter(_Ix**2, sigma=_sigma, mode="reflect")
    _Iyy = gaussian_filter(_Iy**2, sigma=_sigma, mode="reflect")
    _Ixy = gaussian_filter(_Ix * _Iy, sigma=_sigma, mode="reflect")

    _detA = _Ixx * _Iyy - _Ixy**2
    _traceA = _Ixx + _Iyy
    _R = _detA - 0.06 * _traceA**2

    _thresh = thresh_harris_slider.value * _R.max()
    _local_max = (_R == maximum_filter(_R, size=9)) & (_R > _thresh)
    _ys, _xs = np.nonzero(_local_max)

    _R_log = np.sign(_R) * np.log1p(np.abs(_R))

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.8))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].plot(_xs, _ys, "r+", ms=7, mew=1.5)
    _axes[0].set_title(f"{len(_xs)} corners detected")
    _axes[1].imshow(_R_log, cmap="RdBu_r", vmin=-np.abs(_R_log).max(), vmax=np.abs(_R_log).max())
    _axes[1].set_title("Harris response R = det(A) − α·trace(A)²\n(signed log scale)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Detected corners (red crosses) should sit on real corner-like structure. Raise the threshold to keep only the strongest corners; raise σ to require corner structure over a larger neighborhood (fewer, more stable detections)."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Scale-invariant detection: the DoG scale-space

    A detector that only looks at one fixed patch size assumes the
    "interesting" scale of image structure is known in advance. It isn't —
    the same kind of blob might be 5 pixels across in one image and 50 in
    another. Notebook 11's difference-of-Gaussians idea (Eq. 3.70) solves
    this directly: build a **stack** of DoG bands at increasing $\sigma$
    (all at the *same* resolution this time, so locations stay directly
    comparable across the stack) and look for extrema across space *and*
    scale — a pixel that's a local max/min compared to all its neighbors
    *and* to the corresponding pixel one level up and down (Figure 7.11).
    A blob of a given radius produces its strongest response at the
    matching scale.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 5.5))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_11_dog_scale_space.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.11 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 430, © 2004 Springer (Lowe 2004), reproduced for educational use.*"),
    ])
    return


@app.cell
def _(np):
    def make_blob_image(size=220):
        yy, xx = np.mgrid[0:size, 0:size]
        img = np.full((size, size), 200.0)
        blobs = [(55, 55, 8), (160, 55, 14), (55, 160, 22), (160, 160, 32)]
        for cy, cx, r in blobs:
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = 50.0
        return img, blobs

    return (make_blob_image,)


@app.cell
def _(mo):
    sigma_base_slider = mo.ui.slider(start=2, stop=28, value=4, step=2, label="base σ of the stack", debounce=True)
    sigma_base_slider
    return (sigma_base_slider,)


@app.cell
def _(gaussian_filter, make_blob_image, mo, np, plt, sigma_base_slider):
    _img, _blobs = make_blob_image()
    _sigmas = [sigma_base_slider.value * m for m in [1, 2, 3, 4]]

    _fig, _axes = plt.subplots(1, len(_sigmas), figsize=(3.0 * len(_sigmas), 3.4))
    for _ax, _s in zip(_axes, _sigmas):
        _blur1 = gaussian_filter(_img, sigma=_s)
        _blur2 = gaussian_filter(_img, sigma=_s * 1.6)
        _dog = _blur1 - _blur2
        _vmax = max(np.abs(_dog).max(), 1e-6)
        _ax.imshow(_dog, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        for _cy, _cx, _r in _blobs:
            _ax.add_patch(plt.Circle((_cx, _cy), _r, edgecolor="lime", facecolor="none", lw=1, ls=":"))
        _ax.set_title(f"σ={_s:.0f}", fontsize=10)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Four fixed blobs of increasing radius (dotted outlines), same image, four DoG bands at increasing σ. Watch which blob lights up strongest (darkest ring) at each σ — smaller blobs peak at small σ, larger blobs need a larger σ to \"fit.\""),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Orientation estimation

    A detected corner has a location and (from §5) a scale. For a
    descriptor built from the patch around it to survive image rotation,
    it also needs a canonical **orientation** — Lowe (2004) computes a
    36-bin histogram of gradient orientations in the patch, weighted by
    gradient magnitude and by a Gaussian falloff from the patch center,
    then takes the strongest peak(s) as the dominant orientation(s)
    (Figure 7.12).
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_12_orientation_histogram.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.12 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 431, © 2004 Springer (Lowe 2004), reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    image_dropdown_orient = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    row_slider_orient = mo.ui.slider(start=40, stop=460, value=280, step=5, label="patch row", debounce=True)
    col_slider_orient = mo.ui.slider(start=40, stop=460, value=400, step=5, label="patch col", debounce=True)
    mo.vstack([image_dropdown_orient, mo.hstack([row_slider_orient, col_slider_orient], justify="start", gap=2)])
    return col_slider_orient, image_dropdown_orient, row_slider_orient


@app.cell
def _(
    IMAGES_DIR,
    col_slider_orient,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_orient,
    mo,
    np,
    plt,
    row_slider_orient,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_orient.value, plt, np))
    _H, _W = _img_gray.shape
    _half = 20
    _cy = int(np.clip(row_slider_orient.value, _half, _H - _half - 1))
    _cx = int(np.clip(col_slider_orient.value, _half, _W - _half - 1))

    _Ix, _Iy = compute_gradients(_img_gray, 0.5, gaussian_filter, correlate2d)
    _ix_patch = _Ix[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1]
    _iy_patch = _Iy[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1]
    _mag_patch = np.sqrt(_ix_patch**2 + _iy_patch**2)
    _angle_patch = np.arctan2(_iy_patch, _ix_patch) % (2 * np.pi)

    _yy, _xx = np.mgrid[-_half : _half + 1, -_half : _half + 1]
    _dist = np.sqrt(_yy**2 + _xx**2)
    _gauss_w = np.exp(-(_dist**2) / (2 * (_half / 2) ** 2))
    _weight = _mag_patch * _gauss_w

    _n_bins = 36
    _bin_edges = np.linspace(0, 2 * np.pi, _n_bins + 1)
    _hist, _ = np.histogram(_angle_patch, bins=_bin_edges, weights=_weight)
    _bin_centers = 0.5 * (_bin_edges[:-1] + _bin_edges[1:])
    _peak_bin = np.argmax(_hist)
    _peak_angle = _bin_centers[_peak_bin]

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].add_patch(plt.Circle((_cx, _cy), _half, edgecolor="lime", facecolor="none", lw=1.5))
    _axes[0].set_title("patch location")
    _axes[0].axis("off")

    _axes[1].imshow(_img_gray[_cy - _half : _cy + _half + 1, _cx - _half : _cx + _half + 1], cmap="gray")
    _step = 3
    _axes[1].quiver(
        _xx[::_step, ::_step] + _half, _yy[::_step, ::_step] + _half,
        _ix_patch[::_step, ::_step], _iy_patch[::_step, ::_step],
        color="tab:red", angles="xy", scale_units="xy",
    )
    _axes[1].set_title("gradients within patch")
    _axes[1].axis("off")

    _axes[2].bar(_bin_centers, _hist, width=2 * np.pi / _n_bins, color="tab:orange", edgecolor="black", lw=0.3)
    _axes[2].axvline(_peak_angle, color="tab:red", ls="--", lw=1.5, label=f"peak = {np.degrees(_peak_angle):.0f}°")
    _axes[2].set_xlim(0, 2 * np.pi)
    _axes[2].set_xlabel("gradient orientation (rad)")
    _axes[2].set_title("36-bin orientation histogram")
    _axes[2].legend(fontsize=9)
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Image gradients** form a vector field: magnitude = edge strength,
      direction = steepest ascent. Together, $I_x,I_y$ feed everything below.
    - **Eigenvalues/eigenvectors** of a symmetric matrix describe an
      ellipse — the directions and amounts a quadratic form grows fastest
      and slowest.
    - The **auto-correlation matrix** $A$ (Eqs. 7.1–7.8) turns "can this
      patch be localized?" into an eigenvalue question: both eigenvalues
      large → corner, one large → edge, both small → flat.
    - The **Harris detector** (Eq. 7.9) approximates that eigenvalue test
      cheaply via $\det(A)-\alpha\,\text{trace}(A)^2$.
    - **Scale-space DoG detection** finds extrema across space *and*
      scale, so detection doesn't depend on guessing the right patch size
      in advance.
    - **Orientation estimation** assigns each keypoint a canonical
      direction from its local gradient histogram, the last piece needed
      before a descriptor can be built.

    **Next up:** feature descriptors (§7.1.2) — turning each detected,
    scaled, and oriented keypoint into a numerical vector that can
    actually be matched between images.
    """)
    return


if __name__ == "__main__":
    app.run()
