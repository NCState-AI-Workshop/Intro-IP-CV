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
    from scipy.ndimage import gaussian_filter, map_coordinates, label

    return correlate2d, gaussian_filter, label, map_coordinates, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return IMAGES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Edge Detection

    Interest points (notebooks 13–14) are sparse and built for matching. **Edges**
    are a different, far denser kind of feature: every boundary between regions of
    different intensity produces one, and they carry real semantic meaning —
    object silhouettes, shadow boundaries, creases. This notebook covers Szeliski
    **§7.2.1, "Edge detection"** (grayscale only, per your instruction — the
    book's separate color-edge-detection material is skipped), plus one piece
    pulled forward from §7.2.2: **hysteresis thresholding**, the step that
    completes the classic Canny pipeline. The rest of §7.2.2 (chain codes,
    arc-length curve parameterization, Fourier shape descriptors) is skipped as
    genuinely dated — mostly curve-*encoding* tricks, not detection itself, and
    the book's own later citations show the field has moved past them.

    No Predict/Investigate/Modify activities this time — just demos. The edge map
    built here is exactly what a future notebook on active contours (snakes) will
    need for its edge-attraction term.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-14)."""
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Gradients as edge strength (Eqs. 7.19–7.21)

    Exactly the same object notebook 13 built for a completely different purpose:

    $$ \mathbf{J}(\mathbf{x}) = \nabla I(\mathbf{x}) = \left(\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right)(\mathbf{x}) \qquad \text{(Eq. 7.19)}. $$

    Here $\|\mathbf{J}\|$ is **edge strength** and $\angle\mathbf{J}$ points
    *perpendicular* to the edge (toward brighter intensity). Since differentiation
    amplifies high-frequency noise, we pre-smooth with a Gaussian first — and
    because differentiation commutes with convolution, this is the same as
    convolving directly with the derivative of a Gaussian (Eqs. 7.20–7.21).
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_32_human_boundary_detection.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.32 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 456, © 2004 IEEE (Martin, Fowlkes, and Malik 2004), reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    image_dropdown_grad = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_grad_slider = mo.ui.slider(start=0.5, stop=4.0, value=1.0, step=0.5, label="pre-smoothing σ", debounce=True)
    mo.vstack([image_dropdown_grad, sigma_grad_slider])
    return image_dropdown_grad, sigma_grad_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_grad,
    mo,
    np,
    plt,
    sigma_grad_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_grad.value, plt, np))
    _Ix, _Iy = compute_gradients(_img_gray, sigma_grad_slider.value, gaussian_filter, correlate2d)
    _mag = np.sqrt(_Ix**2 + _Iy**2)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original (grayscale)")
    _axes[1].imshow(_mag, cmap="inferno")
    _axes[1].set_title(f"gradient magnitude ‖∇I‖, σ={sigma_grad_slider.value:.1f}\n(thick — not yet an \"edge\")")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("This is edge *strength* everywhere, not yet a clean edge — real edges are several pixels wide here. §2 thins this down to single-pixel curves."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Non-maximum suppression

    To turn a thick band of high gradient magnitude into a single-pixel-wide
    curve, keep a pixel only if its magnitude is a **local maximum along the
    gradient direction** — i.e., compare it to its neighbors one step forward and
    one step backward along $\hat{\mathbf{J}}$ (bilinearly interpolated, since
    that direction generally isn't aligned with the pixel grid) and discard it if
    either neighbor is stronger. This is exactly "finding the ridge line" of the
    edge-strength surface.
    """)
    return


@app.cell
def _(map_coordinates, mo, np, plt):
    # A small synthetic soft edge, for a fully controlled, zoomed-in illustration
    # of the local-max test (a real image's gradient is too noisy to read off by eye).
    _size = 21
    _yy, _xx = np.mgrid[0:_size, 0:_size].astype(float)
    _theta = np.deg2rad(30)
    _signed_dist = (_xx - _size / 2) * np.cos(_theta) + (_yy - _size / 2) * np.sin(_theta)
    _patch = 255.0 / (1.0 + np.exp(-_signed_dist / 1.5))

    _gy, _gx = np.gradient(_patch)
    _mag = np.hypot(_gx, _gy)

    # locate the actual ridge pixel near the patch center (ties to the theory:
    # this is the pixel non-max suppression should keep).
    _cy0, _cx0 = _size // 2, _size // 2
    _sub = _mag[_cy0 - 2 : _cy0 + 3, _cx0 - 2 : _cx0 + 3]
    _iy, _ix = np.unravel_index(np.argmax(_sub), _sub.shape)
    _cy, _cx = _cy0 - 2 + _iy, _cx0 - 2 + _ix

    _norm = max(np.hypot(_gx[_cy, _cx], _gy[_cy, _cx]), 1e-9)
    _ux, _uy = _gx[_cy, _cx] / _norm, _gy[_cy, _cx] / _norm
    _fwd_pt = (_cy + _uy, _cx + _ux)
    _bwd_pt = (_cy - _uy, _cx - _ux)
    _fwd_val = map_coordinates(_mag, [[_fwd_pt[0]], [_fwd_pt[1]]], order=1)[0]
    _bwd_val = map_coordinates(_mag, [[_bwd_pt[0]], [_bwd_pt[1]]], order=1)[0]
    _center_val = _mag[_cy, _cx]

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    _axes[0].imshow(_mag, cmap="inferno")
    _axes[0].quiver(_cx, _cy, _ux, _uy, color="cyan", scale=6, width=0.025, label="gradient direction")
    _axes[0].plot(
        [_bwd_pt[1], _cx, _fwd_pt[1]], [_bwd_pt[0], _cy, _fwd_pt[0]], "o-", color="white", markersize=6
    )
    _axes[0].plot([_cx], [_cy], marker="*", color="gold", markersize=18, markeredgecolor="black", zorder=5)
    _axes[0].set_xlim(_cx - 5, _cx + 5)
    _axes[0].set_ylim(_cy + 5, _cy - 5)
    _axes[0].set_title("zoomed-in gradient magnitude\n(cyan = gradient direction)")
    _axes[0].axis("off")

    _labels = ["one step\nbackward", "pixel under\ntest (★)", "one step\nforward"]
    _vals = [_bwd_val, _center_val, _fwd_val]
    _axes[1].bar(_labels, _vals, color=["0.65", "gold", "0.65"], edgecolor="black")
    _axes[1].set_ylabel("gradient magnitude")
    _axes[1].set_title("local max along the gradient direction\n→ kept by non-max suppression")

    _fig.tight_layout()

    mo.vstack([
        mo.md(
            "**How the test works, zoomed in on one pixel of a soft synthetic "
            "edge.** At the starred pixel, sample the gradient magnitude one "
            "step forward and one step backward along the gradient direction "
            "(the cyan arrow) — bilinearly interpolated, since those two points "
            "generally fall between pixel centers, not on them. Here the "
            "starred pixel's magnitude is larger than both neighbors, so it "
            "passes the test and is kept; walking along this same line, every "
            "*other* pixel would find a stronger neighbor on at least one side "
            "and get suppressed to zero, leaving only this single ridge pixel."
        ),
        _fig,
    ])
    return


@app.function
def non_max_suppress(mag, Ix, Iy, map_coordinates, np):
    H, W = mag.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    norm = np.maximum(np.hypot(Ix, Iy), 1e-9)
    ux, uy = Ix / norm, Iy / norm
    fwd = map_coordinates(mag, [yy + uy, xx + ux], order=1, mode="constant", cval=0.0)
    bwd = map_coordinates(mag, [yy - uy, xx - ux], order=1, mode="constant", cval=0.0)
    keep = (mag >= fwd) & (mag >= bwd)
    return np.where(keep, mag, 0.0)


@app.cell
def _(mo):
    image_dropdown_nms = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_nms_slider = mo.ui.slider(start=0.5, stop=4.0, value=1.0, step=0.5, label="pre-smoothing σ", debounce=True)
    mo.vstack([image_dropdown_nms, sigma_nms_slider])
    return image_dropdown_nms, sigma_nms_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_nms,
    map_coordinates,
    mo,
    np,
    plt,
    sigma_nms_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_nms.value, plt, np))
    _Ix, _Iy = compute_gradients(_img_gray, sigma_nms_slider.value, gaussian_filter, correlate2d)
    _mag = np.sqrt(_Ix**2 + _Iy**2)
    _thin = non_max_suppress(_mag, _Ix, _Iy, map_coordinates, np)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _axes[0].imshow(_mag, cmap="inferno")
    _axes[0].set_title("gradient magnitude (thick)")
    _axes[1].imshow(_thin > 0, cmap="gray_r")
    _axes[1].set_title("after non-max suppression (thin)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Zoom in mentally on any edge: the left panel is a blurry band several pixels wide; the right panel is a single-pixel-wide curve tracing its ridge."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Hysteresis thresholding → the full Canny pipeline

    A single threshold on the thinned magnitude forces an awkward compromise: too
    low and noise gets through as broken, speckled edges; too high and real but
    faint edges (or faint *segments* of a real edge) get dropped. **Hysteresis**
    (Canny 1986) uses two thresholds instead: a pixel above the **high** threshold
    is a definite edge; a pixel above only the **low** threshold is kept *only if*
    it's connected to a definite edge, and discarded otherwise. This rescues weak
    but genuine edge segments while still rejecting isolated weak noise.

    Gradient magnitude → non-max suppression → hysteresis is, in order, exactly
    the classic **Canny edge detector**.
    """)
    return


@app.function
def hysteresis(mag_thin, low, high, label, np):
    strong = mag_thin >= high
    weak = mag_thin >= low
    labeled, _ = label(weak, structure=np.ones((3, 3)))
    keep_labels = set(np.unique(labeled[strong]))
    keep_labels.discard(0)
    return np.isin(labeled, list(keep_labels))


@app.cell
def _(mo):
    image_dropdown_canny = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_canny_slider = mo.ui.slider(start=0.5, stop=4.0, value=1.5, step=0.5, label="pre-smoothing σ", debounce=True)
    low_thresh_slider = mo.ui.slider(start=1, stop=15, value=3, step=1, label="low threshold", debounce=True)
    high_thresh_slider = mo.ui.slider(start=2, stop=25, value=8, step=1, label="high threshold", debounce=True)
    mo.vstack([
        image_dropdown_canny,
        mo.hstack([sigma_canny_slider, low_thresh_slider, high_thresh_slider], justify="start", gap=2),
    ])
    return (
        high_thresh_slider,
        image_dropdown_canny,
        low_thresh_slider,
        sigma_canny_slider,
    )


@app.function
def make_ridge(shape, p0, p1, profile_fn, width, np):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    seg_len = np.hypot(*(p1 - p0))
    dirv = (p1 - p0) / seg_len
    vx, vy = xx - p0[0], yy - p0[1]
    t = (vx * dirv[0] + vy * dirv[1]) / seg_len
    perp = vx * (-dirv[1]) + vy * dirv[0]
    mag = profile_fn(np.clip(t, 0, 1)) * np.exp(-(perp**2) / (2 * width**2))
    return np.where((t >= -0.02) & (t <= 1.02), mag, 0.0)


@app.cell
def _(high_thresh_slider, label, low_thresh_slider, mo, np, plt):
    # A synthetic edge with a weak dip in the middle (connected to strong on
    # both sides) plus a separate, entirely weak, isolated segment — the two
    # canonical cases hysteresis is designed to tell apart. Uses the same
    # low/high sliders as the real-image demo below.
    _main = make_ridge(
        (100, 220), (10, 25), (200, 80),
        lambda t: np.clip(25 - 20 * np.exp(-((t - 0.5) ** 2) / (2 * 0.12**2)), 0, None), 1.6, np,
    )
    _iso = make_ridge((100, 220), (150, 18), (195, 18), lambda t: 6.0 * np.sin(np.pi * t), 1.6, np)
    _mag = np.maximum(_main, _iso)

    _low = min(low_thresh_slider.value, high_thresh_slider.value)
    _high = max(low_thresh_slider.value, high_thresh_slider.value)
    _strong = _mag >= _high
    _weak = _mag >= _low
    _kept = hysteresis(_mag, _low, _high, label, np)

    _classify = np.zeros(_mag.shape, dtype=int)
    _classify[_weak] = 1
    _classify[_strong] = 2

    _result = np.zeros(_mag.shape, dtype=int)
    _result[_weak & ~_kept] = 1
    _result[_kept] = 2

    from matplotlib.colors import ListedColormap as _LCmap

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.6))
    _axes[0].imshow(_mag, cmap="inferno", vmin=0, vmax=max(_mag.max(), 1e-6))
    _axes[0].set_title("gradient magnitude")

    _axes[1].imshow(_classify, cmap=_LCmap(["black", "dimgray", "white"]), vmin=0, vmax=2)
    _axes[1].set_title(f"raw thresholds\n(white=≥high={_high}, gray=≥low={_low})")

    _axes[2].imshow(_result, cmap=_LCmap(["black", "red", "limegreen"]), vmin=0, vmax=2)
    _axes[2].set_title("after hysteresis\n(green=kept, red=removed)")
    _axes[2].annotate(
        "rescued weak segment\n(connected to strong)", xy=(102, 51), xytext=(102, 85),
        color="white", fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="white"),
    )
    _axes[2].annotate(
        "removed\n(isolated, never strong)", xy=(172, 18), xytext=(172, 60),
        color="white", fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="white"),
    )

    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md(
            "This synthetic edge has a weak middle segment connected to strong "
            "endpoints, plus a separate weak blob that never touches a strong "
            "pixel. Try dragging the low/high threshold sliders above: the weak "
            "segment stays green as long as it's still connected to a strong "
            "pixel and its peak clears the low threshold; the isolated blob "
            "is never rescued no matter how low you set the low threshold, "
            "since it never touches a strong pixel."
        ),
        _fig,
    ])
    return


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    high_thresh_slider,
    image_dropdown_canny,
    label,
    low_thresh_slider,
    map_coordinates,
    mo,
    np,
    plt,
    sigma_canny_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_canny.value, plt, np))
    _Ix, _Iy = compute_gradients(_img_gray, sigma_canny_slider.value, gaussian_filter, correlate2d)
    _mag = np.sqrt(_Ix**2 + _Iy**2)
    _thin = non_max_suppress(_mag, _Ix, _Iy, map_coordinates, np)
    _low = min(low_thresh_slider.value, high_thresh_slider.value)
    _high = max(low_thresh_slider.value, high_thresh_slider.value)
    _edges = hysteresis(_thin, _low, _high, label, np)

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original")
    _axes[1].imshow(_mag, cmap="inferno")
    _axes[1].set_title("gradient magnitude")
    _axes[2].imshow(_thin > 0, cmap="gray_r")
    _axes[2].set_title("+ non-max suppression")
    _axes[3].imshow(_edges, cmap="gray_r")
    _axes[3].set_title(f"+ hysteresis (Canny)\nlow={_low}, high={_high}")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. A second paradigm: Laplacian-of-Gaussian / DoG zero-crossings (Eqs. 7.22–7.25)

    Instead of thresholding the gradient magnitude, take the **second**
    derivative — the Laplacian $\nabla^2$ (notebook 8 §8) of a Gaussian-smoothed
    image — and look for **zero crossings**: locations where the sign flips from
    positive (locally darker than average) to negative (locally lighter than
    average). This is a completely different edge-finding principle (curvature
    sign change, not gradient magnitude), yet it typically lands on the same
    edges. In practice, a difference-of-Gaussians (notebook 11) is often
    substituted for the Laplacian-of-Gaussian, since the two kernel shapes are
    nearly identical and DoG is usually already available if a pyramid has been
    built.
    """)
    return


@app.function
def log_zero_crossings(img_gray, sigma, gaussian_filter, correlate2d, np):
    smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect")
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    lap = correlate2d(smoothed, lap_kernel, mode="same", boundary="symm")
    sign = np.sign(lap)
    zero_cross = np.zeros_like(lap, dtype=bool)
    zero_cross[:, :-1] |= sign[:, :-1] * sign[:, 1:] < 0
    zero_cross[:-1, :] |= sign[:-1, :] * sign[1:, :] < 0
    return lap, zero_cross


@app.cell
def _(mo):
    image_dropdown_log = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_log_slider = mo.ui.slider(start=0.5, stop=4.0, value=1.5, step=0.5, label="pre-smoothing σ", debounce=True)
    mo.vstack([image_dropdown_log, sigma_log_slider])
    return image_dropdown_log, sigma_log_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    high_thresh_slider,
    image_dropdown_log,
    label,
    low_thresh_slider,
    map_coordinates,
    mo,
    np,
    plt,
    sigma_log_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_log.value, plt, np))
    _lap, _zero_cross = log_zero_crossings(_img_gray, sigma_log_slider.value, gaussian_filter, correlate2d, np)

    _Ix, _Iy = compute_gradients(_img_gray, sigma_log_slider.value, gaussian_filter, correlate2d)
    _mag = np.sqrt(_Ix**2 + _Iy**2)
    _thin = non_max_suppress(_mag, _Ix, _Iy, map_coordinates, np)
    _low = min(low_thresh_slider.value, high_thresh_slider.value)
    _high = max(low_thresh_slider.value, high_thresh_slider.value)
    _canny_edges = hysteresis(_thin, _low, _high, label, np)

    _fig, _axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    _vmax = max(np.abs(_lap).max(), 1e-6)
    _axes[0].imshow(_lap, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[0].set_title("Laplacian-of-Gaussian response")
    _axes[1].imshow(_zero_cross, cmap="gray_r")
    _axes[1].set_title(f"LoG zero-crossing edges, σ={sigma_log_slider.value:.1f}")
    _axes[2].imshow(_canny_edges, cmap="gray_r")
    _axes[2].set_title("Canny edges from §3\n(same σ/thresholds)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Two entirely different principles — gradient-magnitude ridges vs. second-derivative sign changes — landing on largely the same object boundaries."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Edge strength** is the same gradient vector field from notebook 13,
      reused for a new purpose (Eqs. 7.19–7.21).
    - **Non-maximum suppression** thins the thick raw gradient-magnitude band down
      to single-pixel-wide curves by keeping only ridge points along the gradient
      direction.
    - **Hysteresis thresholding** (two thresholds instead of one) completes the
      classic **Canny** pipeline, rescuing weak-but-connected edge segments while
      rejecting isolated weak noise.
    - **Laplacian-of-Gaussian / DoG zero-crossings** offer a second, independent
      edge-finding principle — second-derivative sign changes instead of
      gradient-magnitude maxima — that typically agrees with Canny on real object
      boundaries.

    **Next up:** active contours (snakes) will use exactly this edge information
    to pull an initial curve estimate onto real object boundaries.
    """)
    return


if __name__ == "__main__":
    app.run()
