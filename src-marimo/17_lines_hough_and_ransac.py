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
    from scipy.ndimage import gaussian_filter, map_coordinates, label, maximum_filter

    return (
        correlate2d,
        gaussian_filter,
        label,
        map_coordinates,
        maximum_filter,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lines, Circles, Hough Transform, and RANSAC

    The human-made world is full of straight lines and circles, and edges
    (notebook 15) are the raw material for finding them: this notebook covers
    Szeliski **§7.4, "Lines and vanishing points"** and the **RANSAC** algorithm
    (§8.1.4), applying both to edge points detected exactly as in notebook 15.

    Two very different strategies show up repeatedly for fitting a geometric
    model to noisy, incomplete data: **voting** (every data point casts a vote
    for the models it's consistent with; the model with the most votes wins —
    the Hough transform), and **random sampling** (repeatedly guess a model from
    a minimal random subset, keep whichever guess has the most support —
    RANSAC). Both show up across computer vision far beyond lines and circles.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-16)."""
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
def non_max_suppress(mag, Ix, Iy, map_coordinates, np):
    """Same non-maximum suppression as notebook 15 §2."""
    H, W = mag.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    norm = np.maximum(np.hypot(Ix, Iy), 1e-9)
    ux, uy = Ix / norm, Iy / norm
    fwd = map_coordinates(mag, [yy + uy, xx + ux], order=1, mode="constant", cval=0.0)
    bwd = map_coordinates(mag, [yy - uy, xx - ux], order=1, mode="constant", cval=0.0)
    keep = (mag >= fwd) & (mag >= bwd)
    return np.where(keep, mag, 0.0)


@app.function
def hysteresis(mag_thin, low, high, label, np):
    """Same hysteresis thresholding as notebook 15 §3."""
    strong = mag_thin >= high
    weak = mag_thin >= low
    labeled, _ = label(weak, structure=np.ones((3, 3)))
    keep_labels = set(np.unique(labeled[strong]))
    keep_labels.discard(0)
    return np.isin(labeled, list(keep_labels))


@app.function
def detect_edges(img_gray, sigma, low, high, compute_gradients, non_max_suppress, hysteresis, gaussian_filter, correlate2d, map_coordinates, label, np):
    """The full notebook-15 Canny pipeline, packaged as one call: returns
    (edge mask, Ix, Iy) so gradient orientation is available at each edge
    point for the oriented Hough transform below."""
    Ix, Iy = compute_gradients(img_gray, sigma, gaussian_filter, correlate2d)
    mag = np.hypot(Ix, Iy)
    thin = non_max_suppress(mag, Ix, Iy, map_coordinates, np)
    edges = hysteresis(thin, min(low, high), max(low, high), label, np)
    return edges, Ix, Iy


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Hough transform for lines (§7.4.2)

    A line can be written in normal form: a unit normal $\hat{n} = (\cos\theta,
    \sin\theta)$ and a distance to the origin $d$, so that every point $(x,y)$
    on the line satisfies

    $$ d = x\cos\theta + y\sin\theta. \qquad (7.39) $$

    The classic Hough transform has every edge point vote for *every* $(\theta,
    d)$ pair it's consistent with — an entire sinusoidal curve in $(\theta, d)$
    space per point — and lines correspond to accumulator bins where many
    curves intersect. A **better** approach, since we already have gradient
    orientation at every edge point (notebook 15), is the **oriented** Hough
    transform (Figure 7.47): a line's normal $\hat{n}$ points in the same
    direction as the image gradient there, so each edgel can vote for a
    *single* $(\theta, d)$ cell directly, using its own gradient direction for
    $\theta$ and Eq. 7.39 for $d$. Peaks in this 2D accumulator are the detected
    lines.

    The demo defaults to a synthetic scene (a rotated rectangle plus a free
    diagonal line) since our natural test photos mostly lack long, unambiguous
    straight edges — but try switching to a real image to see how much noisier
    real accumulator peaks are.
    """)
    return


@app.function
def hough_lines_oriented(edges, Ix, Iy, n_theta, n_d, np):
    """Oriented Hough transform (Figure 7.47): each edgel votes once, using its
    own gradient direction for theta and Eq. 7.39 for d."""
    ys, xs = np.nonzero(edges)
    ex, ey = Ix[ys, xs], Iy[ys, xs]
    theta = np.degrees(np.arctan2(ey, ex)) % 180.0
    d = xs * np.cos(np.radians(theta)) + ys * np.sin(np.radians(theta))
    d_max = np.hypot(*edges.shape)
    t_idx = np.clip((theta / 180.0 * n_theta).astype(int), 0, n_theta - 1)
    d_idx = np.clip(((d + d_max) / (2 * d_max) * n_d).astype(int), 0, n_d - 1)
    acc = np.zeros((n_theta, n_d), dtype=np.int32)
    np.add.at(acc, (t_idx, d_idx), 1)
    return acc, d_max


@app.function
def hough_peaks(acc, min_votes, maximum_filter, np, size=9):
    mx = maximum_filter(acc, size=size)
    peaks = (acc == mx) & (acc >= min_votes)
    return np.argwhere(peaks)


@app.function
def make_synthetic_shapes(H, W, np):
    """A rotated rectangle plus a free diagonal line -- clean, unambiguous
    straight edges to make the accumulator peaks easy to read, unlike the
    natural test photos (which mostly lack long straight edges)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cx, cy, theta = W * 0.42, H * 0.5, np.radians(12)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - cx) * ct + (yy - cy) * st
    yr = -(xx - cx) * st + (yy - cy) * ct
    rect = (np.abs(xr) < W * 0.22) & (np.abs(yr) < H * 0.28)
    img = np.where(rect, 200.0, 30.0)
    line_mask = np.abs((xx - W * 0.75) - 0.6 * (yy - H * 0.1)) < 1.5
    img = np.where(line_mask, 220.0, img)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255)


@app.cell
def _(mo):
    hl_image_dropdown = mo.ui.dropdown(
        options=["synthetic shapes", "coins", "astronaut", "coffee", "chelsea", "raccoon"],
        value="synthetic shapes", label="image",
    )
    hl_sigma_slider = mo.ui.slider(start=0.5, stop=3.0, value=1.0, step=0.5, label="pre-smoothing sigma", debounce=True)
    hl_votes_slider = mo.ui.slider(start=3, stop=100, value=10, step=1, label="min votes for a peak", debounce=True)
    mo.vstack([hl_image_dropdown, mo.hstack([hl_sigma_slider, hl_votes_slider], justify="start", gap=2)])
    return hl_image_dropdown, hl_sigma_slider, hl_votes_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    hl_image_dropdown,
    hl_sigma_slider,
    hl_votes_slider,
    label,
    map_coordinates,
    maximum_filter,
    mo,
    np,
    plt,
):
    if hl_image_dropdown.value == "synthetic shapes":
        _img_gray = make_synthetic_shapes(150, 200, np)
    else:
        _img_gray = to_luma(load_rgb255(IMAGES_DIR, hl_image_dropdown.value, plt, np))
    _edges, _Ix, _Iy = detect_edges(
        _img_gray, hl_sigma_slider.value, 3, 8, compute_gradients, non_max_suppress, hysteresis,
        gaussian_filter, correlate2d, map_coordinates, label, np,
    )
    _n_theta, _n_d = 180, 200
    _acc, _d_max = hough_lines_oriented(_edges, _Ix, _Iy, _n_theta, _n_d, np)
    _peaks = hough_peaks(_acc, hl_votes_slider.value, maximum_filter, np)

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 9))
    _H, _W = _img_gray.shape

    _axes[0, 0].imshow(_img_gray, cmap="gray")
    _axes[0, 0].set_title("original")
    _axes[0, 0].axis("off")

    _axes[0, 1].imshow(_edges, cmap="gray_r")
    _axes[0, 1].set_title("edges (notebook 15)")
    _axes[0, 1].axis("off")

    _axes[1, 0].imshow(
        _acc, cmap="inferno", aspect="auto", origin="upper", extent=[-_d_max, _d_max, 180, 0],
    )
    if len(_peaks):
        _peak_theta = _peaks[:, 0] / _n_theta * 180.0
        _peak_d = _peaks[:, 1] / _n_d * 2 * _d_max - _d_max
        _axes[1, 0].scatter(_peak_d, _peak_theta, c="lime", s=25, edgecolors="black", linewidths=0.5)
    _axes[1, 0].set_xlabel("d")
    _axes[1, 0].set_ylabel(r"$\theta$ (degrees)")
    _axes[1, 0].set_title("(θ, d) accumulator, votes")

    _axes[1, 1].imshow(_img_gray, cmap="gray")
    for _ti, _di in _peaks:
        _theta = _ti / _n_theta * 180.0
        _d = _di / _n_d * 2 * _d_max - _d_max
        _ct, _st = np.cos(np.radians(_theta)), np.sin(np.radians(_theta))
        if abs(_st) > abs(_ct):
            _x0, _x1 = 0, _W
            _y0, _y1 = (_d - _x0 * _ct) / _st, (_d - _x1 * _ct) / _st
        else:
            _y0, _y1 = 0, _H
            _x0, _x1 = (_d - _y0 * _st) / _ct, (_d - _y1 * _st) / _ct
        _axes[1, 1].plot([_x0, _x1], [_y0, _y1], "lime", lw=1.5)
    _axes[1, 1].set_xlim(0, _W)
    _axes[1, 1].set_ylim(_H, 0)
    _axes[1, 1].set_title(f"{len(_peaks)} detected lines")
    _axes[1, 1].axis("off")

    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Hough transform for circles

    The book notes in passing (§7.4.2, footnote) that the Hough transform
    generalizes to other shapes, "such as circles" — a circle needs 3
    parameters, $(c_x, c_y, r)$, instead of a line's 2. Voting the naive way (every
    edgel votes for *every* possible circle through it, for every candidate
    radius) is expensive. The same trick used above helps again: since the
    gradient at an edgel on a circle points radially, straight toward or away
    from the center, an edgel only needs to vote for the (at most two) candidate
    centers $\,c = (x, y) \pm r\,\hat{n}\,$ for each candidate radius $r$ — a 1D
    sweep per edgel per radius, not a full circle of votes.

    The demo below sweeps a small range of radii, accumulating votes into a 2D
    $(c_x, c_y)$ image per radius, then keeps the strongest, non-overlapping
    peaks across all radii.
    """)
    return


@app.function
def hough_circles_oriented(edges, Ix, Iy, r_values, min_votes, maximum_filter, np):
    """Gradient-oriented circle Hough transform: each edgel votes for the (at
    most two) centers r pixels away along its own gradient direction."""
    ys, xs = np.nonzero(edges)
    H, W = edges.shape
    norm = np.maximum(np.hypot(Ix[ys, xs], Iy[ys, xs]), 1e-9)
    nx, ny = Ix[ys, xs] / norm, Iy[ys, xs] / norm

    candidates = []
    for r in r_values:
        acc = np.zeros((H, W), dtype=np.int32)
        for sign in (+1, -1):
            cxs = np.round(xs + sign * r * nx).astype(int)
            cys = np.round(ys + sign * r * ny).astype(int)
            valid = (cxs >= 0) & (cxs < W) & (cys >= 0) & (cys < H)
            np.add.at(acc, (cys[valid], cxs[valid]), 1)
        mx = maximum_filter(acc, size=9)
        peaks = (acc == mx) & (acc >= min_votes)
        cy_idx, cx_idx = np.nonzero(peaks)
        for cy, cx in zip(cy_idx, cx_idx):
            candidates.append((int(acc[cy, cx]), int(cx), int(cy), int(r)))

    candidates.sort(reverse=True)
    chosen = []
    for v, cx, cy, r in candidates:
        if all(np.hypot(cx - ccx, cy - ccy) >= 0.5 * (r + cr) for _, ccx, ccy, cr in chosen):
            chosen.append((v, cx, cy, r))
    return chosen


@app.cell
def _(mo):
    hc_sigma_slider = mo.ui.slider(start=0.5, stop=3.0, value=1.5, step=0.5, label="pre-smoothing sigma", debounce=True)
    hc_rmin_slider = mo.ui.slider(start=5, stop=40, value=15, step=1, label="min radius", debounce=True)
    hc_rmax_slider = mo.ui.slider(start=10, stop=60, value=42, step=1, label="max radius", debounce=True)
    hc_votes_slider = mo.ui.slider(start=10, stop=40, value=18, step=1, label="min votes for a peak", debounce=True)
    mo.hstack([hc_sigma_slider, hc_rmin_slider, hc_rmax_slider, hc_votes_slider], justify="start", gap=2)
    return hc_rmax_slider, hc_rmin_slider, hc_sigma_slider, hc_votes_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    hc_rmax_slider,
    hc_rmin_slider,
    hc_sigma_slider,
    hc_votes_slider,
    label,
    map_coordinates,
    maximum_filter,
    mo,
    np,
    plt,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, "coins", plt, np))
    _edges, _Ix, _Iy = detect_edges(
        _img_gray, hc_sigma_slider.value, 5, 12, compute_gradients, non_max_suppress, hysteresis,
        gaussian_filter, correlate2d, map_coordinates, label, np,
    )
    _r_values = np.arange(hc_rmin_slider.value, hc_rmax_slider.value + 1, 1)
    _chosen = hough_circles_oriented(_edges, _Ix, _Iy, _r_values, hc_votes_slider.value, maximum_filter, np)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _axes[0].imshow(_edges, cmap="gray_r")
    _axes[0].set_title("edges")
    _axes[1].imshow(_img_gray, cmap="gray")
    for _v, _cx, _cy, _r in _chosen:
        _axes[1].add_patch(plt.Circle((_cx, _cy), _r, color="lime", fill=False, lw=1.5))
    _axes[1].set_title(f"{len(_chosen)} detected circles")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. RANSAC (§8.1.4)

    Before getting to RANSAC, it's worth being explicit about the naive
    approach it improves on, since we haven't covered it yet: ordinary
    **least-squares line fitting**. Writing a line as $y = mx + b$, choose
    $(m, b)$ to minimize the sum of squared vertical residuals,

    $$ E_{LS}(m, b) = \sum_i \big(y_i - (mx_i + b)\big)^2. $$

    Stack every point into a row of $A = [x_i \ \ 1]$ and let $p = (m, b)^T$;
    this is a linear least-squares problem $Ap \approx y$, solved directly by
    the normal equations $p = (A^TA)^{-1}A^Ty$ — exactly what `np.linalg.lstsq`
    computes below. This is a good fit when the noise is small and
    well-behaved, but every point pulls on the line with a force proportional
    to its *squared* residual, so a single wild outlier can drag the whole fit
    arbitrarily far from the truth — far more than any single well-behaved
    point could ever pull it back. That fragility is exactly what RANSAC below
    is built to fix.

    Both Hough transforms above are a form of **voting**: robust to noise, but
    needing a discretized parameter space that grows fast with the number of
    model parameters (fine for a line's 2, already awkward for a circle's 3).
    **RANSAC** (RANdom SAmple Consensus, Fischler and Bolles 1981) takes a
    completely different approach:

    1. Randomly pick the *minimum* number of points needed to define a model
       (2 for a line).
    2. Fit the model to just those points.
    3. Count **inliers**: points whose residual $\|r_i\| \le \epsilon$ (Eq. 8.28).
    4. Repeat $S$ times; keep the model with the most inliers, then refit using
       *all* of its inliers for a final, better estimate.

    How many trials $S$ are enough? If $p$ is the probability that a random
    point is an inlier and $k$ points are needed per model, the chance that one
    trial is *all* inliers is $p^k$, so the chance that $S$ trials all fail is
    $(1-p^k)^S$. Requiring this to be below $1 - P$ for a target success
    probability $P$ gives

    $$ S = \frac{\log(1-P)}{\log(1-p^k)}. \qquad (8.30) $$

    Even at $p = 0.5$, $k=2$ needs only $S=17$ trials for $P=0.99$ — this is
    why RANSAC uses the *smallest possible* $k$ in practice (Table 8.2).

    Below: a line's worth of points buried in heavy outlier noise. A single
    outlier can drag an ordinary least-squares fit far from the truth; RANSAC
    finds the line anyway by explicitly identifying (and ignoring) the
    outliers.
    """)
    return


@app.function
def ransac_line(xs, ys, epsilon, n_trials, rng, np):
    """Fit a line y = m*x + b to noisy, outlier-contaminated points via RANSAC
    (Eqs. 8.28-8.30). Returns (m, b, inlier_mask)."""
    N = len(xs)
    best_inliers, best_count = None, -1
    for _ in range(n_trials):
        i, j = rng.choice(N, 2, replace=False)
        if abs(xs[j] - xs[i]) < 1e-9:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - m * xs[i]
        resid = np.abs(ys - (m * xs + b)) / np.sqrt(1 + m**2)
        inliers = resid < epsilon
        if inliers.sum() > best_count:
            best_count, best_inliers = inliers.sum(), inliers
    A = np.stack([xs[best_inliers], np.ones(best_inliers.sum())], axis=1)
    m_final, b_final = np.linalg.lstsq(A, ys[best_inliers], rcond=None)[0]
    return m_final, b_final, best_inliers


@app.cell
def _(mo):
    ransac_outlier_slider = mo.ui.slider(start=0, stop=200, value=90, step=10, label="number of outliers", debounce=True)
    ransac_eps_slider = mo.ui.slider(start=1.0, stop=10.0, value=4.0, step=0.5, label="inlier threshold epsilon", debounce=True)
    ransac_trials_slider = mo.ui.slider(start=5, stop=500, value=100, step=5, label="number of trials S", debounce=True)
    mo.hstack([ransac_outlier_slider, ransac_eps_slider, ransac_trials_slider], justify="start", gap=2)
    return ransac_eps_slider, ransac_outlier_slider, ransac_trials_slider


@app.cell
def _(
    mo,
    np,
    plt,
    ransac_eps_slider,
    ransac_outlier_slider,
    ransac_trials_slider,
):
    _rng = np.random.default_rng(3)
    _n_inlier = 60
    _true_m, _true_b = 0.6, 15.0
    _xs_in = _rng.uniform(0, 200, _n_inlier)
    _ys_in = _true_m * _xs_in + _true_b + _rng.normal(0, 3, _n_inlier)
    _xs_out = _rng.uniform(0, 200, ransac_outlier_slider.value)
    _ys_out = _rng.uniform(0, 150, ransac_outlier_slider.value)
    _xs = np.concatenate([_xs_in, _xs_out])
    _ys = np.concatenate([_ys_in, _ys_out])

    _A = np.stack([_xs, np.ones_like(_xs)], axis=1)
    _m_ls, _b_ls = np.linalg.lstsq(_A, _ys, rcond=None)[0]

    _m_ransac, _b_ransac, _inliers = ransac_line(
        _xs, _ys, ransac_eps_slider.value, ransac_trials_slider.value, np.random.default_rng(0), np
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    _xx = np.linspace(0, 200, 10)

    _axes[0].scatter(_xs, _ys, s=14, c="0.4")
    _axes[0].set_title("raw data (no fit yet)")

    _axes[1].scatter(_xs, _ys, s=14, c=np.where(_inliers, "tab:green", "tab:red"))
    _axes[1].plot(_xx, _true_m * _xx + _true_b, "k--", label="true line")
    _axes[1].plot(_xx, _m_ls * _xx + _b_ls, "b-", lw=2, label="naive least squares (all points)")
    _axes[1].plot(_xx, _m_ransac * _xx + _b_ransac, "g-", lw=2, label="RANSAC (green = inliers)")
    _axes[1].legend(loc="upper left", fontsize=8)
    _axes[1].set_title(f"{_inliers.sum()} of {len(_xs)} points kept as inliers")

    for _ax in _axes:
        _ax.set_xlim(_xs.min() - 5, _xs.max() + 5)
        _ax.set_ylim(min(_ys.min(), 0) - 5, _ys.max() + 5)
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Vanishing points (§7.4.3)

    One application of line detection worth knowing about, if not implementing
    here: in real scenes, sets of 3D-parallel lines (building edges, railway
    tracks, tiled floors) project to 2D lines that all pass through a common
    **vanishing point**. Finding these points — typically by having pairs of
    detected lines vote for a candidate intersection, then refitting robustly —
    can help calibrate a camera or recover its orientation relative to a scene
    (Section 11.1.1), and is a building block for detecting 3D rectangular
    structure in architectural photos (Figure 7.51). RANSAC-based line fitting
    and vanishing-point voting are the same "random sample + consensus"
    strategy applied one level up.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - The **Hough transform** finds lines (and, with a 3rd parameter, circles)
      by voting: using gradient orientation lets each edgel cast one vote
      instead of a whole family of them (Eq. 7.39, Figure 7.47).
    - **RANSAC** finds a model a completely different way — random minimal-sample
      hypotheses plus inlier counting (Eqs. 8.28–8.30) — and is dramatically more
      robust to outliers than ordinary least squares.
    - Both strategies (vote, or randomly sample-and-verify) reappear throughout
      computer vision well beyond lines and circles, including in image
      alignment and 3D geometry estimation (Chapters 8 and 11).

    This closes out the Chapter 7 sequence: features and matching (13–14),
    edges and active contours (15–16), and geometric primitive fitting (this
    notebook).
    """)
    return


if __name__ == "__main__":
    app.run()
