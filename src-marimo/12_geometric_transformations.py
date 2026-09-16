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
    from scipy.ndimage import map_coordinates

    return map_coordinates, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return IMAGES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Geometric Transformations

    Every operator so far (notebooks 8–11) left a pixel's *location* alone
    and only changed its *value*:

    $$ g(\mathbf{x}) = h(f(\mathbf{x})) \qquad \text{(Eq. 3.73)}. $$

    This notebook covers Szeliski **§3.6, "Geometric transformations,"**
    where instead the *domain* is transformed,

    $$ g(\mathbf{x}) = f(h(\mathbf{x})) \qquad \text{(Eq. 3.74)}, $$

    i.e., where a pixel's value *comes from* moves. Two topics:

    - **§3.6.1 Parametric transformations** — notebook 2 already covered
      translation, Euclidean, similarity, and affine transforms; this
      notebook adds the one skipped there, **projective** (the full
      8-degree-of-freedom homography), and then asks the more basic
      question of *how do you actually paint the transformed image*
      (Algorithms 3.1–3.2).
    - **§3.6.2 Mesh-based warping** — local, spatially-varying
      deformations, applied here via two pre-defined deformation fields.

    Skipped, as requested: MIP-mapping, Elliptical Weighted Average,
    anisotropic filtering, and multi-pass transforms — all resampling-speed
    optimizations for the case of shrinking/magnifying textures in
    real-time graphics, not central to the geometric-transformation idea
    itself. No Predict/Investigate/Modify activities this time — just demos.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-11)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.function
def resample_bilinear(img_rgb, src_x, src_y, map_coordinates, np):
    """Sample img_rgb at the (possibly non-integer) coordinates (src_x, src_y),
    one output pixel per entry, per channel. This is Eq. 3.75's FIR resampling,
    with order=1 (bilinear) as the interpolating kernel h."""
    out = np.zeros((*src_x.shape, 3))
    for ch in range(3):
        out[:, :, ch] = map_coordinates(img_rgb[:, :, ch], [src_y, src_x], order=1, mode="constant", cval=0.0)
    return out


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Parametric transformations

    Table 3.3's hierarchy — translation, rigid/Euclidean, similarity,
    affine — was built up in notebook 2. The row skipped there is
    **projective**: an $8$-degree-of-freedom transform represented by a
    full $3\times 3$ homogeneous matrix $\tilde{\mathbf{H}}$ (defined only
    up to scale — $\tilde{\mathbf{H}}$ and $c\tilde{\mathbf{H}}$ represent
    the same transform for any $c \ne 0$, which is why 9 matrix entries
    give only 8 degrees of freedom),

    $$
    \tilde{\mathbf{x}}' = \tilde{\mathbf{H}}\tilde{\mathbf{x}}, \qquad
    x' = \frac{h_{00}x + h_{01}y + h_{02}}{h_{20}x + h_{21}y + h_{22}}, \qquad
    y' = \frac{h_{10}x + h_{11}y + h_{12}}{h_{20}x + h_{21}y + h_{22}}.
    $$

    Unlike affine, the division by a term that depends on $x$ and $y$ makes
    this transform *non-linear* in image coordinates — which is exactly
    what lets it capture the "looking at a flat surface from an angle"
    effect (perspective foreshortening) that affine cannot. Affine
    transforms always keep parallel lines parallel (a parallelogram stays
    a parallelogram); projective transforms only guarantee that *straight*
    lines stay straight — parallel lines can converge, as they do in a real
    photograph of railroad tracks.

    For reference, here is Table 3.3 in full. Each transformation also
    preserves every property listed below it — e.g. similarity preserves
    not only angles but also parallelism and straight lines:

    | Transformation | Matrix | # DoF | Preserves |
    |---|:---:|:---:|---|
    | translation | $\begin{bmatrix}\mathbf{I} & \mathbf{t}\end{bmatrix}_{2\times3}$ | 2 | orientation |
    | rigid (Euclidean) | $\begin{bmatrix}\mathbf{R} & \mathbf{t}\end{bmatrix}_{2\times3}$ | 3 | lengths |
    | similarity | $\begin{bmatrix}s\mathbf{R} & \mathbf{t}\end{bmatrix}_{2\times3}$ | 4 | angles |
    | affine | $\begin{bmatrix}\mathbf{A}\end{bmatrix}_{2\times3}$ | 6 | parallelism |
    | **projective** | $\begin{bmatrix}\tilde{\mathbf{H}}\end{bmatrix}_{3\times3}$ | 8 | straight lines |

    (The $2\times3$ matrices are extended with a third row $[\mathbf{0}^T\;1]$ to
    form a full $3\times3$ matrix for homogeneous coordinates, exactly like
    $\tilde{\mathbf{H}}$ already is.)
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 2.4))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_44_transformation_hierarchy.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.44 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 169, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(np):
    def compute_homography(src_pts, dst_pts):
        """Solve for the 8 unknowns of H (h22 fixed to 1) from 4 exact point
        correspondences — a direct linear solve, no least squares needed."""
        A, b = [], []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
            b.append(xp)
            A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
            b.append(yp)
        h = np.linalg.solve(np.array(A), np.array(b))
        return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

    return (compute_homography,)


@app.cell
def _(mo):
    randomize_button = mo.ui.button(
        label="Randomize affine & projective transforms",
        value=0,
        on_click=lambda count: count + 1,
    )
    randomize_button
    return (randomize_button,)


@app.cell
def _(compute_homography, mo, np, plt, randomize_button):
    _rng = np.random.default_rng(seed=randomize_button.value)
    _square = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=float)
    _square_h = np.vstack([_square.T, np.ones(4)])

    _M = _rng.uniform(-1.0, 1.0, size=(2, 3))
    _T_affine = np.vstack([_M + [[1, 0, 0], [0, 1, 0]], [0, 0, 1]])
    _affine_h = _T_affine @ _square_h
    _affine_pts = (_affine_h[:2] / _affine_h[2]).T

    _dst_corners = _square + _rng.uniform(-3.5, 3.5, size=(4, 2))
    _Hmat = compute_homography(_square, _dst_corners)
    _proj_h = _Hmat @ _square_h
    _proj_pts = (_proj_h[:2] / _proj_h[2]).T

    _orig_x = np.append(_square[:, 0], _square[0, 0])
    _orig_y = np.append(_square[:, 1], _square[0, 1])
    _affine_closed = np.vstack([_affine_pts, _affine_pts[0]])
    _proj_closed = np.vstack([_proj_pts, _proj_pts[0]])

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.plot(_orig_x, _orig_y, "--", color="gray", label="original square")
    _ax.fill(_affine_closed[:, 0], _affine_closed[:, 1], color="tab:blue", alpha=0.5, edgecolor="black", label="affine (parallelogram)")
    _ax.fill(_proj_closed[:, 0], _proj_closed[:, 1], color="tab:red", alpha=0.5, edgecolor="black", label="projective (general quadrilateral)")
    _lim = 14
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left", fontsize=9)
    _ax.set_title(f"click count: {randomize_button.value}")

    mo.vstack([
        _fig,
        mo.md("The affine result is always a parallelogram — opposite sides stay parallel. The projective result generally isn't: its sides stay *straight*, but not parallel."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Forward vs. inverse warping

    Given a transform $\mathbf{x}'=\mathbf{h}(\mathbf{x})$ and a source
    image $f$, how do you actually paint the output image $g$? The
    obvious approach — loop over every source pixel, compute where it
    goes, copy it there — is **forward warping** (Algorithm 3.1,
    Figure 3.45). It has two real problems: $\mathbf{x}'$ is generally not
    an integer location, and — worse — many destination pixels end up with
    **no** source pixel landing on them at all, leaving cracks and holes
    (worst when the transform magnifies part of the image).
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4.8))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_45_forward_warping.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.45 and Algorithm 3.1 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 170, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The fix is to flip the loop: for every *destination* pixel $\mathbf{x}'$,
    compute where it came from, $\mathbf{x}=\hat{\mathbf{h}}(\mathbf{x}')$
    (usually just $\mathbf{h}^{-1}$), and **resample** $f$ there — this is
    **inverse warping** (Algorithm 3.2, Figure 3.46). Every destination
    pixel gets a value, and since $\mathbf{x}$ is non-integer, resampling
    uses the same FIR interpolation machinery as decimation (notebook 11,
    Eq. 3.75) — bilinear here, for speed, matching the book's own
    recommendation for interactive use.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4.8))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_46_inverse_warping.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.46 and Algorithm 3.2 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 171, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(np):
    def forward_warp_rgb(img_rgb, H):
        """Algorithm 3.1: for every source pixel, round its destination
        location and copy it there. Later writes silently overwrite earlier
        ones; unreached destination pixels stay at 0 (black) — the holes."""
        H_img, W_img = img_rgb.shape[:2]
        yy, xx = np.mgrid[0:H_img, 0:W_img]
        ones = np.ones(H_img * W_img)
        pts = np.stack([xx.ravel().astype(float), yy.ravel().astype(float), ones], axis=0)
        dst = H @ pts
        dst_x = np.round(dst[0] / dst[2]).astype(int)
        dst_y = np.round(dst[1] / dst[2]).astype(int)
        valid = (dst_x >= 0) & (dst_x < W_img) & (dst_y >= 0) & (dst_y < H_img)
        out = np.zeros_like(img_rgb)
        out[dst_y[valid], dst_x[valid]] = img_rgb.reshape(-1, 3)[valid]
        return out

    def inverse_warp_rgb(img_rgb, H, map_coordinates):
        """Algorithm 3.2: for every destination pixel, look up its source
        location via H^-1 and bilinearly resample."""
        H_img, W_img = img_rgb.shape[:2]
        yy, xx = np.mgrid[0:H_img, 0:W_img]
        ones = np.ones(H_img * W_img)
        pts = np.stack([xx.ravel().astype(float), yy.ravel().astype(float), ones], axis=0)
        src = np.linalg.inv(H) @ pts
        src_x = (src[0] / src[2]).reshape(H_img, W_img)
        src_y = (src[1] / src[2]).reshape(H_img, W_img)
        return resample_bilinear(img_rgb, src_x, src_y, map_coordinates, np)

    return forward_warp_rgb, inverse_warp_rgb


@app.cell
def _(mo):
    image_dropdown_warp = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    perspective_strength_slider = mo.ui.slider(start=0.05, stop=0.35, value=0.2, step=0.05, label="perspective (\"keystone\") strength", debounce=True)
    mo.vstack([image_dropdown_warp, perspective_strength_slider])
    return image_dropdown_warp, perspective_strength_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_homography,
    forward_warp_rgb,
    image_dropdown_warp,
    inverse_warp_rgb,
    map_coordinates,
    mo,
    np,
    perspective_strength_slider,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_warp.value, plt, np)
    _H_img, _W_img = _img_rgb.shape[:2]
    _s = perspective_strength_slider.value

    _src_corners = np.array([[0, 0], [_W_img, 0], [_W_img, _H_img], [0, _H_img]], dtype=float)
    _dst_corners = _src_corners.copy()
    _dst_corners[0, 0] += _s * _W_img
    _dst_corners[1, 0] -= _s * _W_img
    _Hmat = compute_homography(_src_corners, _dst_corners)

    _fwd = forward_warp_rgb(_img_rgb, _Hmat)
    _inv = inverse_warp_rgb(_img_rgb, _Hmat, map_coordinates)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    _axes[0].set_title("original")
    _axes[1].imshow(np.clip(_fwd, 0, 255).astype(np.uint8))
    _axes[1].set_title("forward warp\n(Algorithm 3.1)")
    _axes[2].imshow(np.clip(_inv, 0, 255).astype(np.uint8))
    _axes[2].set_title("inverse warp\n(Algorithm 3.2)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Look for the streaky black gaps in the forward-warped middle panel, especially where the transform compresses the image — those are exactly the \"cracks and holes\" the book describes. The inverse-warped result on the right has none."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Mesh-based warping

    A single parametric transform moves *every* pixel the same
    algebraic way. Local edits — Figure 3.49's classic example is turning a
    frown into a smile — need different displacements in different parts
    of the image. The book surveys several ways to *specify* such a
    spatially-varying displacement field (sparse control points, denser
    correspondences, oriented line segments, or a hand-fitted mesh); once
    a field is specified, it's rendered with exactly the same **inverse
    warping** machinery from §1 — the per-pixel source location $\hat{\mathbf{h}}(\mathbf{x}')$
    just varies across the image now, instead of being one global matrix.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 5.2))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_49_mesh_warping_alternatives.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.49 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 175, © 1999 Morgan Kaufmann (Gomes, Darsa et al. 1999), reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Specifying a mesh by hand (as in the figure above) takes real
    interactive tooling, which is beyond what a notebook demo can offer.
    Instead, the two deformation fields below are **pre-defined, closed-form**
    displacement fields — no control points to place — applied via inverse
    warping exactly as in §1, with the underlying deformation made visible
    by warping a regular grid alongside the image:

    - **Bulge:** a radial power-law warp centered on the image — pixels
      near the center get pulled outward, magnifying it like a lens.
    - **Wave:** a horizontal sinusoidal displacement that depends on
      $y$ — every row shifts sideways by a different amount, rippling the
      image.
    """)
    return


@app.cell
def _(np):
    def bulge_inverse(xp, yp, cx, cy, R, power):
        dx, dy = xp - cx, yp - cy
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r == 0, 1e-6, r)
        t = np.clip(r / R, 0, 1)
        r_src = R * t ** (1.0 / power)
        scale = np.where(r > R, 1.0, r_src / r_safe)
        return cx + dx * scale, cy + dy * scale

    def bulge_forward(x, y, cx, cy, R, power):
        dx, dy = x - cx, y - cy
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r == 0, 1e-6, r)
        t = np.clip(r / R, 0, 1)
        r_dst = R * t**power
        scale = np.where(r > R, 1.0, r_dst / r_safe)
        return cx + dx * scale, cy + dy * scale

    def wave_inverse(xp, yp, amplitude, wavelength):
        return xp - amplitude * np.sin(2 * np.pi * yp / wavelength), yp

    def wave_forward(x, y, amplitude, wavelength):
        return x + amplitude * np.sin(2 * np.pi * y / wavelength), y

    return bulge_forward, bulge_inverse, wave_forward, wave_inverse


@app.cell
def _(mo):
    image_dropdown_mesh = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    deformation_dropdown = mo.ui.dropdown(options=["bulge", "wave"], value="bulge", label="deformation")
    mo.vstack([image_dropdown_mesh, deformation_dropdown])
    return deformation_dropdown, image_dropdown_mesh


@app.cell
def _(
    IMAGES_DIR,
    bulge_forward,
    bulge_inverse,
    deformation_dropdown,
    image_dropdown_mesh,
    map_coordinates,
    mo,
    np,
    plt,
    wave_forward,
    wave_inverse,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_mesh.value, plt, np)
    _H_img, _W_img = _img_rgb.shape[:2]
    _cx, _cy = _W_img / 2, _H_img / 2
    _R = 0.9 * min(_H_img, _W_img) / 2
    _power = 0.5
    _amplitude = 0.04 * _W_img
    _wavelength = 0.5 * _H_img

    _yy, _xx = np.mgrid[0:_H_img, 0:_W_img]
    if deformation_dropdown.value == "bulge":
        _src_x, _src_y = bulge_inverse(_xx.astype(float), _yy.astype(float), _cx, _cy, _R, _power)
        _fwd_map = lambda x, y: bulge_forward(x, y, _cx, _cy, _R, _power)
    else:
        _src_x, _src_y = wave_inverse(_xx.astype(float), _yy.astype(float), _amplitude, _wavelength)
        _fwd_map = lambda x, y: wave_forward(x, y, _amplitude, _wavelength)

    _warped = resample_bilinear(_img_rgb, _src_x, _src_y, map_coordinates, np)

    _n_lines = 12
    _grid_coords = np.linspace(0, _W_img, _n_lines)
    _grid_coords_y = np.linspace(0, _H_img, _n_lines)
    _samples_x = np.linspace(0, _W_img, 200)
    _samples_y = np.linspace(0, _H_img, 200)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    for _xv in _grid_coords:
        _axes[0].plot([_xv] * len(_samples_y), _samples_y, color="cyan", lw=0.7)
    for _yv in _grid_coords_y:
        _axes[0].plot(_samples_x, [_yv] * len(_samples_x), color="cyan", lw=0.7)
    _axes[0].set_title("original + regular grid")
    _axes[0].axis("off")

    _axes[1].imshow(np.clip(_warped, 0, 255).astype(np.uint8))
    for _xv in _grid_coords:
        _gx, _gy = _fwd_map(np.full_like(_samples_y, _xv), _samples_y)
        _axes[1].plot(_gx, _gy, color="cyan", lw=0.7)
    for _yv in _grid_coords_y:
        _gx, _gy = _fwd_map(_samples_x, np.full_like(_samples_x, _yv))
        _axes[1].plot(_gx, _gy, color="cyan", lw=0.7)
    _axes[1].set_xlim(0, _W_img)
    _axes[1].set_ylim(_H_img, 0)
    _axes[1].set_title(f"{deformation_dropdown.value} + deformed grid")
    _axes[1].axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Parametric transforms** move every pixel by the same rule.
      **Projective** transforms (8 DOF, a full $3\times3$ homogeneous
      matrix) are the most general: they preserve straight lines but not
      parallelism, unlike affine.
    - **Forward warping** (Algorithm 3.1) loops over source pixels and can
      leave holes; **inverse warping** (Algorithm 3.2) loops over
      destination pixels and resamples the source, avoiding holes entirely
      — the standard approach in practice.
    - **Mesh-based warping** applies a *spatially-varying* displacement
      field instead of one global matrix, still rendered via inverse
      warping — demonstrated here with two closed-form fields (bulge,
      wave) instead of hand-placed control points.

    This wraps up this course's tour of Szeliski Chapter 3 (image
    processing). Later notebooks will move into new territory.
    """)
    return


if __name__ == "__main__":
    app.run()
