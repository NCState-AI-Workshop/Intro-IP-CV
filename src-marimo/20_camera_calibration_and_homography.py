# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "plotly",
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import correlate2d
    from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

    return (
        correlate2d,
        gaussian_filter,
        go,
        make_subplots,
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
    # Camera Calibration: Intrinsics, Extrinsics, and Homography Estimation

    Notebook 4 built the camera matrix $P=K[R\mid t]$ and projected *known* 3D
    points with a *given* $K,R,t$ -- but closed with its own inverse problem
    unsolved: "a single image can't tell intrinsics and extrinsics apart...
    real camera calibration needs multiple views of a known target." Notebooks
    12 and 14 defined homographies and even solved for one from 4 point
    correspondences -- but only ever for a *known* transform used to synthesize
    a test image, never estimated from real, noisy, outlier-contaminated
    matches. Notebook 17 built a complete RANSAC engine and explicitly
    foreshadowed its own reuse "in image alignment and geometry estimation."

    This notebook closes all three loops: estimate a homography from actual
    correspondences, robustly; and estimate a camera's intrinsics *and*
    extrinsics from observations, two different ways. Doing this honestly
    needs one genuinely new piece of linear algebra -- the **singular value
    decomposition (SVD)** -- plus a useful decomposition trick (**RQ
    decomposition**) for pulling $K$ and $R$ back out of an estimated camera
    matrix. Both are introduced from scratch below.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting (notebooks 8-19's convention)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Recap: the camera matrix

    From notebook 4: the **intrinsic** calibration matrix

    $$ K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, $$

    the **extrinsic** rotation and translation $R,t$ taking world points into
    the camera frame ($X_{cam}=RX_{world}+t$), and the full **camera matrix**

    $$ P = K[R\mid t], $$

    an 11-DOF (up to scale) $3\times4$ matrix sending homogeneous 3D world
    points directly to homogeneous pixel coordinates, $\tilde x \sim P\tilde X$.
    Everything below is about *estimating* $P$ (and then $K,R,t$) -- and the
    closely related problem of estimating a homography -- from observed point
    correspondences, instead of assuming they're given.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. A new tool: the singular value decomposition (SVD)

    Notebook 13 asked: for a symmetric matrix $A$, is there a direction $v$ it
    doesn't rotate, only stretches ($Av=\lambda v$)? The SVD answers the
    analogous question for an **arbitrary** (not necessarily square or
    symmetric) matrix $A$ ($m\times n$): is there an orthogonal input
    direction that $A$ maps to an orthogonal output direction, with no
    rotation mixed in? The answer is always yes, and it factors $A$ as

    $$ A = U\Sigma V^T, $$

    where $U$ ($m\times m$) and $V$ ($n\times n$) are orthogonal, and $\Sigma$
    is diagonal with non-negative entries $\sigma_1\ge\sigma_2\ge\cdots\ge0$,
    the **singular values**. Column $v_i$ of $V$ maps to $\sigma_i u_i$ ($u_i$
    the matching column of $U$): $Av_i=\sigma_i u_i$. Geometrically, $A$ sends
    the unit sphere to an ellipsoid; $V$'s columns are the orthogonal
    directions *on the sphere* that land on the ellipsoid's orthogonal axes,
    and $\sigma_i$ is that axis's length.

    **This *is* notebook 13's eigenvalue idea, one level up.** $A^TA$ is
    always symmetric, so it has eigenvalues and orthogonal eigenvectors
    exactly like notebook 13's $2\times2$ structure tensor — and substituting
    $A=U\Sigma V^T$ shows $A^TA=V\Sigma^2V^T$: $V$'s columns are $A^TA$'s
    eigenvectors, with eigenvalues $\sigma_i^2$ (notebook 13 never wrote this
    general matrix factorization explicitly, only worked with $v^TAv$
    directly, but it's the same $Av=\lambda v$ underneath). The SVD is what
    happens when a matrix isn't symmetric (or even square) and "eigenvectors
    of $A$ itself" stops making sense, but "eigenvectors of $A^TA$" still
    does.
    """)
    return


@app.cell
def _(mo, np, plt):
    _A = np.array([[2.0, 1.0], [0.5, 1.5]])
    _U, _S, _Vt = np.linalg.svd(_A)
    _V = _Vt.T
    _theta = np.linspace(0, 2 * np.pi, 200)
    _circle = np.stack([np.cos(_theta), np.sin(_theta)])
    _ellipse = _A @ _circle

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _axes[0].plot(_circle[0], _circle[1], "b-")
    _axes[0].set_title("unit circle (domain)")
    for _i in range(2):
        _axes[0].annotate("", xy=tuple(_V[:, _i]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:red"))
        _axes[0].text(*(_V[:, _i] * 1.15), f"$v_{_i+1}$", color="tab:red")
    _axes[1].plot(_ellipse[0], _ellipse[1], "g-")
    _axes[1].set_title("image under A (ellipse)")
    for _i in range(2):
        _vec = _U[:, _i] * _S[_i]
        _axes[1].annotate("", xy=tuple(_vec), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:purple"))
        _axes[1].text(*(_vec * 1.1), f"$\\sigma_{_i+1}u_{_i+1}$", color="tab:purple")
    for _ax in _axes:
        _ax.set_aspect("equal")
        _ax.axhline(0, color="gray", lw=0.5)
        _ax.axvline(0, color="gray", lw=0.5)
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"$A=\\begin{{bmatrix}}2&1\\\\0.5&1.5\\end{{bmatrix}}$: the orthogonal "
            f"pair $v_1,v_2$ on the circle maps to the orthogonal pair "
            f"$\\sigma_1u_1,\\sigma_2u_2$ on the ellipse -- $\\sigma_1={_S[0]:.2f}$, "
            f"$\\sigma_2={_S[1]:.2f}$ are exactly that ellipse's semi-axis lengths."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The one fact this notebook actually needs.** Every estimation problem
    below reduces to: given a matrix $A$ built from noisy measurements, find
    the unit vector $h$ minimizing $\|Ah\|^2$ (an exact solution $Ah=0$ usually
    doesn't exist once there's noise or more equations than unknowns; forcing
    $\|h\|=1$ rules out the useless $h=0$). Expand the objective,

    $$ \|Ah\|^2 = h^TA^TAh, $$

    a **Rayleigh quotient** of the symmetric matrix $A^TA$ — precisely
    notebook 13 §2's quadratic form $v^TAv$, with $A^TA$ playing the role of
    $A$ there. Notebook 13 showed a quadratic form's level sets are governed
    by its eigenvalues; the same argument shows its minimum over unit vectors
    equals the *smallest* eigenvalue, attained at the matching eigenvector.
    Since $A^TA$'s eigenvectors are $V$'s columns with eigenvalues
    $\sigma_i^2$, the minimizer is $V$'s **last column** — the right singular
    vector for the smallest singular value. In code, `np.linalg.svd(A)`
    returns `Vt` with rows sorted by *decreasing* singular value, so the
    answer is `Vt[-1]`.

    **Why not just build $A^TA$ and call `np.linalg.eigh` on it, then?**
    Mathematically it's the same answer; numerically it's a worse way to get
    it. For any matrix, the **condition number** $\kappa(A)=\sigma_1/\sigma_n$
    (largest singular value over smallest) is literally how stretched the
    §2 demo's ellipse is -- its long axis over its short one. That ratio is
    also the standard measure of numerical sensitivity: solving a linear
    system or least-squares problem built from $A$ can amplify any small
    relative error already in the input (sensor noise, floating-point
    rounding) by up to a factor of $\kappa(A)$ in the worst case. $\kappa(A)=1$
    (a perfect circle, no stretching) means no amplification at all;
    $\kappa(A)\gg1$ means a tiny wobble in the input can produce a wildly
    different answer -- exactly the risk every DLT solve below runs, since
    its $A$ is always built from noisy pixel measurements. If $A$ has
    condition number $\kappa(A)$, then $A^TA$ has condition number
    $\kappa(A)^2$ (its eigenvalues are $\sigma_i^2$) — squaring an
    already-large number costs real floating-point digits, and it
    costs them precisely where this problem needs them most, in the
    *smallest* singular value's direction, which is the whole answer we're
    after. `np.linalg.svd` is built to never actually form $A^TA$ internally
    for exactly this reason. §4 computes a real $\mathrm{cond}(A)$ for this
    notebook's own DLT matrix — square that number in your head and you'll
    see how close to the edge of double precision $A^TA$ would already be
    sitting, before even solving anything.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Homography estimation: the general DLT

    A homography relates two images of the same plane, $\tilde x'\sim\tilde
    H\tilde x$ (Eq. 2.20), where $\tilde x=(x,y,1)^T$, $\tilde
    x'=(x',y',1)^T$, and

    $$ H = \begin{bmatrix} h_{00} & h_{01} & h_{02} \\ h_{10} & h_{11} & h_{12} \\ h_{20} & h_{21} & h_{22} \end{bmatrix}. $$

    The $\sim$ hides an unknown nonzero scale $w$ -- homogeneous coordinates
    only mean something after dividing by the third entry, so this is really
    saying $H\tilde x = (wx',\,wy',\,w)^T$ for *some* $w$. Writing the matrix
    product out and then dividing the first two rows by the third to recover
    actual pixel coordinates:

    $$
    x' = \frac{h_{00}x+h_{01}y+h_{02}}{h_{20}x+h_{21}y+h_{22}}, \qquad
    y' = \frac{h_{10}x+h_{11}y+h_{12}}{h_{20}x+h_{21}y+h_{22}}.
    $$

    This is *nonlinear* in $H$'s entries (they appear in a denominator too) --
    unusable directly in a least-squares solve. **Clearing denominators**
    (multiply both sides by the shared denominator) removes the division
    entirely:

    $$
    h_{00}x+h_{01}y+h_{02} - x'(h_{20}x+h_{21}y+h_{22}) = 0, \qquad
    h_{10}x+h_{11}y+h_{12} - y'(h_{20}x+h_{21}y+h_{22}) = 0,
    $$

    and expanding the products makes every term exactly **linear** in $H$'s 9
    entries (the book's Eq. 8.19/8.21, a strategy Hartley and Zisserman (2004)
    call the **direct linear transform, DLT**):

    $$
    h_{00}x+h_{01}y+h_{02} - h_{20}xx'-h_{21}yx'-h_{22}x' = 0, \qquad
    h_{10}x+h_{11}y+h_{12} - h_{20}xy'-h_{21}yy'-h_{22}y' = 0.
    $$

    **Flatten $H$ row-major into a 9-vector** $h=(h_{00},h_{01},h_{02},\,
    h_{10},h_{11},h_{12},\,h_{20},h_{21},h_{22})^T$, so each equation above is
    literally the dot product of $h$ with one row of coefficients read
    straight off it:

    $$
    \begin{bmatrix} x & y & 1 & 0 & 0 & 0 & -xx' & -yx' & -x' \\ 0 & 0 & 0 & x & y & 1 & -xy' & -yy' & -y' \end{bmatrix} h = 0.
    $$

    Stack 2 rows per correspondence into one matrix $A$ ($2N\times9$) across
    all $N$ points: $Ah=0$. For exactly
    $N=4$ points in general position, $A$ has rank 8 and a 1-dimensional null
    space — the unique (up to scale) exact solution, which is precisely what
    notebooks 12 and 14's `compute_homography` already solves (fixing
    $h_{22}=1$ and calling `np.linalg.solve`). For $N>4$ noisy correspondences,
    there's usually no exact null vector at all — so instead, apply §2's fact
    directly: the best-fit $h$ is the smallest right singular vector of $A$,
    found by SVD. The two methods **agree exactly** when $N=4$ (verified
    below), and the SVD route is the one that still works for real,
    overdetermined, noisy data.
    """)
    return


@app.function
def homography_dlt(src_pts, dst_pts, np):
    """General N-point (N>=4) direct linear transform for a homography, via
    the SVD fact from §2: stack 2 equations/correspondence into A, take the
    smallest right singular vector of A as h, reshape to 3x3."""
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp, -yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


@app.function
def apply_homography(H, pts, np):
    pts = np.asarray(pts, dtype=float)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ pts_h.T).T
    return out[:, :2] / out[:, 2:3]


@app.cell
def _(mo):
    mo.md(r"""
    **Reprojection error.** For a point $x$ and an estimated homography
    $\hat H$, the *predicted* location of its correspondence is
    $\hat x'=\hat H x$; the **reprojection error** is simply how far that
    prediction lands from where the point actually belongs,

    $$ e(x) = \|\hat x' - x'\|_2 = \sqrt{(\hat x'-x')^2+(\hat y'-y')^2}\ \ \text{pixels}. $$

    Normally $x'$ is the *observed* correspondence, and the "mean
    reprojection error" is this averaged over the correspondences used to fit
    $\hat H$ -- a measure of fit quality. In the synthetic demos below the
    true, noiseless $H$ is known, so instead $e$ is measured on a dense grid
    of points against *that* ground truth, not just at the handful of
    correspondences used to estimate $\hat H$ -- reporting how well the whole
    transform was recovered, not just how well $\hat H$ fits the specific
    points it saw.

    ### Predict

    A homography has 8 degrees of freedom. If 4 noiseless correspondences
    pin it down exactly, what do you expect to happen to this reprojection
    error as you add a 5th, 10th, 20th *noisy* correspondence and re-run the
    SVD solve — does more (noisy) data make the estimate better, worse, or is
    4 points always enough?
    """)
    return


@app.cell
def _(mo):
    dlt_npts_slider = mo.ui.slider(start=4, stop=60, value=4, step=2, label="number of correspondences N", debounce=True)
    dlt_noise_slider = mo.ui.slider(start=0.0, stop=3.0, value=1.0, step=0.25, label="correspondence noise σ (px)", debounce=True)
    mo.hstack([dlt_npts_slider, dlt_noise_slider], justify="start", gap=2)
    return dlt_noise_slider, dlt_npts_slider


@app.function
def mat_to_latex_bmatrix(M, sig=4):
    rows = [" & ".join(f"{v:.{sig}g}" for v in row) for row in M]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


@app.cell
def _(dlt_noise_slider, dlt_npts_slider, mo, np, plt):
    _H_true = np.array([[1.2, 0.15, 30.0], [-0.1, 1.1, 20.0], [0.0007, 0.0004, 1.0]])
    _rng = np.random.default_rng(0)
    _src = _rng.uniform(0, 600, (dlt_npts_slider.value, 2))
    _dst = apply_homography(_H_true, _src, np)
    _dst_noisy = _dst + _rng.normal(0, dlt_noise_slider.value, _dst.shape)
    _H_est = homography_dlt(_src, _dst_noisy, np)

    _grid = np.stack(np.meshgrid(np.linspace(0, 600, 15), np.linspace(0, 400, 10)), axis=-1).reshape(-1, 2)
    _pred = apply_homography(_H_est, _grid, np)
    _true = apply_homography(_H_true, _grid, np)
    _err = np.sqrt(((_pred - _true) ** 2).sum(axis=1))

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _axes[0].scatter(_src[:, 0], _src[:, 1], s=14, c="0.5")
    _axes[0].set_title("source points (domain, before H)")
    _axes[1].scatter(_dst_noisy[:, 0], _dst_noisy[:, 1], s=14, c="0.5")
    _axes[1].set_title(f"destination points (after H + noise)\nN={dlt_npts_slider.value}, mean grid reprojection error = {_err.mean():.3f} px")
    for _ax in _axes:
        _ax.set_aspect("equal")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Left: the $N$ source points as sampled. Right: where $H$ actually "
            f"sent them, plus the injected noise -- $\\hat H$ below is fit from "
            f"*only* this right-hand scatter and its known correspondence to "
            f"the left-hand one.\n\n"
            f"$$ H_{{true}} = {mat_to_latex_bmatrix(_H_true)}, \\qquad "
            f"\\hat H = {mat_to_latex_bmatrix(_H_est)} $$"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        With **zero noise**, 4 points are exactly enough — adding more exact
        points changes nothing (the null space is already 1-dimensional).
        With **noise**, 4 points give one noisy estimate with no way to tell
        it apart from a better one; adding more noisy correspondences lets
        the SVD effectively *average out* the noise (minimizing total squared
        residual over all of them at once), so the reprojection error
        generally shrinks as $N$ grows — more noisy data is still better than
        less, exactly like least-squares fitting anywhere else in this
        course. Try $\sigma=0$ above and confirm $N$ stops mattering.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Why normalization matters

    Raw pixel coordinates run into the hundreds; a term like $xx'$ in the DLT
    matrix $A$ can be $10^4$-$10^5$ while the $1$'s and $x$'s sit at $10^0$-$10^2$
    — wildly different scales in the same matrix. SVD is numerically exact in
    theory, but any floating-point implementation is only as good as $A$'s
    **conditioning**, and that scale mismatch inflates $A$'s condition number
    badly. Hartley and Zisserman's fix (not in this textbook, but standard in
    the wider literature): before building $A$, translate each point set so
    its centroid is the origin and rescale so the average point distance from
    the origin is $\sqrt2$ (a similarity transform $T$ per image), solve for
    $\hat H$ in these **normalized** coordinates, then undo the normalization,
    $H=T_2^{-1}\hat HT_1$.
    """)
    return


@app.function
def normalize_points(pts, np):
    """Hartley normalization: centroid to the origin, mean distance sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    mean_dist = np.sqrt((centered**2).sum(axis=1)).mean()
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    normed = (T @ pts_h.T).T
    return normed[:, :2] / normed[:, 2:3], T


@app.function
def homography_dlt_normalized(src_pts, dst_pts, np):
    src_n, T1 = normalize_points(src_pts, np)
    dst_n, T2 = normalize_points(dst_pts, np)
    H_n = homography_dlt(src_n, dst_n, np)
    H = np.linalg.inv(T2) @ H_n @ T1
    return H / H[2, 2]


@app.function
def build_dlt_matrix(src_pts, dst_pts, np):
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp, -yp])
    return np.array(A)


@app.cell
def _(mo):
    norm_noise_slider = mo.ui.slider(start=0.25, stop=3.0, value=1.0, step=0.25, label="correspondence noise σ (px)", debounce=True)
    norm_noise_slider
    return (norm_noise_slider,)


@app.cell
def _(mo, norm_noise_slider, np):
    _H_true = np.array([[1.2, 0.15, 30.0], [-0.1, 1.1, 20.0], [0.0007, 0.0004, 1.0]])
    _rng = np.random.default_rng(0)
    _src = _rng.uniform(0, 600, (30, 2))
    _dst = apply_homography(_H_true, _src, np)
    _dst_noisy = _dst + _rng.normal(0, norm_noise_slider.value, _dst.shape)

    _A_unnorm = build_dlt_matrix(_src, _dst_noisy, np)
    _src_n, _T1 = normalize_points(_src, np)
    _dst_n, _T2 = normalize_points(_dst_noisy, np)
    _A_norm = build_dlt_matrix(_src_n, _dst_n, np)

    _H_unnorm = homography_dlt(_src, _dst_noisy, np)
    _H_norm = homography_dlt_normalized(_src, _dst_noisy, np)

    _grid = np.stack(np.meshgrid(np.linspace(0, 600, 15), np.linspace(0, 400, 10)), axis=-1).reshape(-1, 2)
    _true = apply_homography(_H_true, _grid, np)
    _err_unnorm = np.sqrt(((apply_homography(_H_unnorm, _grid, np) - _true) ** 2).sum(axis=1)).mean()
    _err_norm = np.sqrt(((apply_homography(_H_norm, _grid, np) - _true) ** 2).sum(axis=1)).mean()

    mo.md(f"""
    | | condition number of $A$ | mean reprojection error |
    |---|---|---|
    | unnormalized | {np.linalg.cond(_A_unnorm):,.0f} | {_err_unnorm:.3f} px |
    | normalized | {np.linalg.cond(_A_norm):,.1f} | {_err_norm:.3f} px |

    Same points, same noise, same SVD solve -- only the coordinate scaling
    changed, yet the condition number drops by **{np.linalg.cond(_A_unnorm)/np.linalg.cond(_A_norm):,.0f}x**
    and the reprojection error improves too. Every homography estimate from
    here on uses the normalized version.

    As a concrete check on §2's warning: $\\mathrm{{cond}}(A)^2\\approx
    {np.linalg.cond(_A_unnorm)**2:.1e}$ for the *unnormalized* matrix above --
    comparable to double precision's own $\\sim10^{{16}}$ ceiling, which is
    exactly why solving via $A^TA$'s eigenvectors would be a much worse idea
    here than it already is to use unnormalized coordinates with the SVD.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Robust estimation with RANSAC, on a real image pair

    Real feature matches (notebooks 13-14) are never contamination-free: NNDR
    rejects *most* wrong matches, but not all, and even a handful of bad
    correspondences can wreck an unweighted least-squares DLT fit -- exactly
    notebook 17's line-fitting lesson, one dimension up. The fix is the same:
    **RANSAC** (Fischler and Bolles 1981, Eq. 8.30), with the *homography's*
    own minimal sample size of 4 points standing in for a line's 2.
    """)
    return


@app.function
def ransac_homography(src_pts, dst_pts, epsilon, n_trials, rng, np):
    """Fit a homography to noisy, outlier-contaminated correspondences via
    RANSAC (Eq. 8.30), minimal sample 4. Returns (H, inlier_mask)."""
    N = len(src_pts)
    best_inliers, best_count = None, -1
    for _ in range(n_trials):
        idx = rng.choice(N, 4, replace=False)
        try:
            H = homography_dlt(src_pts[idx], dst_pts[idx], np)
        except np.linalg.LinAlgError:
            continue
        pred = apply_homography(H, src_pts, np)
        resid = np.sqrt(((pred - dst_pts) ** 2).sum(axis=1))
        inliers = resid < epsilon
        if inliers.sum() > best_count:
            best_count, best_inliers = inliers.sum(), inliers
    H_final = homography_dlt_normalized(src_pts[best_inliers], dst_pts[best_inliers], np)
    return H_final, best_inliers


@app.function
def harris_corners(img_gray, compute_gradients, correlate2d, gaussian_filter, maximum_filter, np, sigma=2.0, thresh_frac=0.01, max_pts=200):
    """Same Harris detector as notebooks 13-14, packaged as a reusable function."""
    Ix, Iy = compute_gradients(img_gray, 0.5, gaussian_filter, correlate2d)
    Ixx = gaussian_filter(Ix**2, sigma=sigma)
    Iyy = gaussian_filter(Iy**2, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)
    detA = Ixx * Iyy - Ixy**2
    traceA = Ixx + Iyy
    R = detA - 0.06 * traceA**2
    thresh = thresh_frac * R.max()
    local_max = (R == maximum_filter(R, size=9)) & (R > thresh)
    ys, xs = np.nonzero(local_max)
    order = np.argsort(-R[ys, xs])[:max_pts]
    return xs[order], ys[order]


@app.function
def patch_descriptor(img_gray, xs, ys, half, np):
    """Bias/gain-normalized (MOPS-style) patch descriptor, as in notebook 14."""
    H, W = img_gray.shape
    descs, valid = [], []
    for x, y in zip(xs, ys):
        if x - half < 0 or x + half >= W or y - half < 0 or y + half >= H:
            valid.append(False)
            continue
        patch = img_gray[y - half : y + half, x - half : x + half]
        descs.append((patch - patch.mean()) / (patch.std() + 1e-6))
        valid.append(True)
    return descs, np.array(valid)


@app.cell
def _(np):
    def compute_gradients(img_gray, sigma, gaussian_filter, correlate2d):
        GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        GY = GX.T
        smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect") if sigma > 0 else img_gray
        Ix = correlate2d(smoothed, GX, mode="same", boundary="symm")
        Iy = correlate2d(smoothed, GY, mode="same", boundary="symm")
        return Ix, Iy

    return (compute_gradients,)


@app.function
def warp_image_by_homography(img, H, np, map_coordinates):
    """Warp img by H: out(x) = img(H^-1 x), the standard inverse-warp construction."""
    Hh, Ww = img.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    pts = np.stack([xx.ravel().astype(float), yy.ravel().astype(float), np.ones(Hh * Ww)])
    src = np.linalg.inv(H) @ pts
    src_x = (src[0] / src[2]).reshape(Hh, Ww)
    src_y = (src[1] / src[2]).reshape(Hh, Ww)
    return map_coordinates(img, [src_y, src_x], order=1, mode="constant", cval=0.0)


@app.cell
def _(mo):
    mo.md(r"""
    Below: `coffee.png` and a second view synthesized by a *known* homography
    -- a pronounced perspective tilt about the image center, plus a bit of
    rotation and translation (a real projective warp, not just the rigid
    motion notebook 14 used, and strong enough that the keystone effect is
    plainly visible, not just a disguised rotation). Harris
    corners + NNDR matching (notebooks 13-14) give real correspondences, with
    the true transform known so "correct" is checkable, not eyeballed. A
    slider then injects additional synthetic bad matches on top of these real
    ones, exactly as notebook 17 injected outlier points into its line-fitting
    demo.

    ### Predict

    Naive (normalized) DLT uses *every* proposed match, good or bad, with
    equal weight. As injected bad matches increase, what happens to its
    reprojection error -- does it degrade gracefully, or can a *handful* of
    outliers already ruin it? What do you expect for RANSAC's error as the
    same outliers are added?
    """)
    return


@app.cell
def _(mo):
    ransac_outlier_slider = mo.ui.slider(start=0, stop=120, value=40, step=10, label="injected bad matches", debounce=True)
    ransac_eps_slider = mo.ui.slider(start=1.0, stop=8.0, value=3.0, step=0.5, label="inlier threshold ε (px)", debounce=True)
    mo.hstack([ransac_outlier_slider, ransac_eps_slider], justify="start", gap=2)
    return ransac_eps_slider, ransac_outlier_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    map_coordinates,
    maximum_filter,
    mo,
    np,
    plt,
    ransac_eps_slider,
    ransac_outlier_slider,
):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _H_img, _W_img = _gray.shape
    # A visibly perspective (not just rotation-like) warp: apply the tilt
    # about the image *center* rather than the (0,0) corner, so the keystone
    # effect is symmetric and doesn't just push everything off-canvas.
    _cx, _cy = _W_img / 2, _H_img / 2
    _T = np.array([[1, 0, _cx], [0, 1, _cy], [0, 0, 1]])
    _Tinv = np.array([[1, 0, -_cx], [0, 1, -_cy], [0, 0, 1]])
    _H_local = np.array([[1.08, 0.06, 8.0], [-0.04, 0.96, 5.0], [0.0010, 0.0007, 1.0]])
    _H_true = _T @ _H_local @ _Tinv
    _H_true = _H_true / _H_true[2, 2]
    _grayB = warp_image_by_homography(_gray, _H_true, np, map_coordinates)

    _xsA, _ysA = harris_corners(_gray, compute_gradients, correlate2d, gaussian_filter, maximum_filter, np)
    _xsB, _ysB = harris_corners(_grayB, compute_gradients, correlate2d, gaussian_filter, maximum_filter, np)
    _descA, _validA = patch_descriptor(_gray, _xsA, _ysA, 8, np)
    _descB, _validB = patch_descriptor(_grayB, _xsB, _ysB, 8, np)
    _idxA = np.nonzero(_validA)[0]
    _idxB = np.nonzero(_validB)[0]
    _DA = np.stack(_descA).reshape(len(_descA), -1)
    _DB = np.stack(_descB).reshape(len(_descB), -1)
    _dists = np.linalg.norm(_DA[:, None, :] - _DB[None, :, :], axis=2)
    _nndr_matches = []
    for _i in range(_dists.shape[0]):
        _order = np.argsort(_dists[_i])
        _d1, _d2 = _dists[_i, _order[0]], _dists[_i, _order[1]]
        if _d1 / max(_d2, 1e-9) < 0.8:
            _nndr_matches.append((_i, int(_order[0])))

    _src_real = np.array([[_xsA[_idxA[_i]], _ysA[_idxA[_i]]] for _i, _j in _nndr_matches], dtype=float)
    _dst_real = np.array([[_xsB[_idxB[_j]], _ysB[_idxB[_j]]] for _i, _j in _nndr_matches], dtype=float)

    _rng = np.random.default_rng(42)
    _n_out = ransac_outlier_slider.value
    _out_src = np.stack([_rng.uniform(0, _W_img, _n_out), _rng.uniform(0, _H_img, _n_out)], axis=1)
    _out_dst = np.stack([_rng.uniform(0, _W_img, _n_out), _rng.uniform(0, _H_img, _n_out)], axis=1)
    _all_src = np.vstack([_src_real, _out_src])
    _all_dst = np.vstack([_dst_real, _out_dst])

    _H_naive = homography_dlt_normalized(_all_src, _all_dst, np)
    _H_ransac, _inliers = ransac_homography(_all_src, _all_dst, ransac_eps_slider.value, 500, np.random.default_rng(0), np)

    _grid = np.stack(np.meshgrid(np.linspace(0, _W_img, 15), np.linspace(0, _H_img, 10)), axis=-1).reshape(-1, 2)
    _true_grid = apply_homography(_H_true, _grid, np)
    _err_naive = np.sqrt(((apply_homography(_H_naive, _grid, np) - _true_grid) ** 2).sum(axis=1)).mean()
    _err_ransac = np.sqrt(((apply_homography(_H_ransac, _grid, np) - _true_grid) ** 2).sum(axis=1)).mean()

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5.2))
    _combined = np.hstack([_gray, _grayB])
    _gap = _W_img
    for _ax, _mask, _title, _err in [
        (_axes[0], np.ones(len(_all_src), dtype=bool), "every proposed match (naive DLT)", _err_naive),
        (_axes[1], _inliers, "RANSAC inliers only", _err_ransac),
    ]:
        _ax.imshow(_combined, cmap="gray", vmin=0, vmax=255)
        for _k in np.nonzero(_mask)[0]:
            _color = "lime" if _k < len(_src_real) else "red"
            _ax.plot([_all_src[_k, 0], _all_dst[_k, 0] + _gap], [_all_src[_k, 1], _all_dst[_k, 1]], color=_color, lw=0.6, alpha=0.7)
        _ax.set_title(f"{_title}\nmean reprojection error: {_err:.2f} px", fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Green = genuine NNDR matches ({len(_src_real)} of them); red = "
            f"injected bad matches ({_n_out}). RANSAC recovers the same "
            f"{_inliers.sum()} inliers regardless of how many red matches are "
            f"added, because a random 4-point sample is overwhelmingly likely "
            f"to include at least one outlier once they're this numerous -- "
            f"but *some* 4-point samples are still all-green, and those are "
            f"the ones with the most consensus."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        Naive DLT does **not** degrade gracefully: because every match counts
        equally in a least-squares fit, even a modest number of outliers
        among dozens of good matches can pull the solution arbitrarily far
        from correct (reprojection error can reach the hundreds or thousands
        of pixels). RANSAC's error, in contrast, stays essentially flat at
        its clean-data baseline regardless of how many outliers are added —
        it explicitly searches for (and refits only on) a self-consistent
        subset, rather than trying to average everything into one answer.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. From 2D-3D correspondences to the camera matrix

    A homography's DLT generalizes directly to estimating the *full camera
    matrix* $P$ (Sutherland 1974, the book's own Eq. 11.11-11.13) from known
    3D points $\tilde X=(X,Y,Z,1)^T$ and their observed pixels
    $\tilde x=(x,y,1)^T$ -- the same symbols $x,y$ (2D) and $X,Y,Z$ (3D) as
    everywhere else in this notebook, never $p$, to keep "a point" and "an
    entry of $P$" visually distinct. Write $P$'s 12 entries explicitly,

    $$ P = \begin{bmatrix} p_{00}&p_{01}&p_{02}&p_{03}\\ p_{10}&p_{11}&p_{12}&p_{13}\\ p_{20}&p_{21}&p_{22}&p_{23} \end{bmatrix}. $$

    Exactly as in §3: $\tilde x\sim P\tilde X$ hides an unknown scale $w$, so
    $P\tilde X=(wx,wy,w)^T$; dividing the first two rows by the third gives
    pixel coordinates as a ratio,

    $$
    x=\frac{p_{00}X+p_{01}Y+p_{02}Z+p_{03}}{p_{20}X+p_{21}Y+p_{22}Z+p_{23}}, \qquad
    y=\frac{p_{10}X+p_{11}Y+p_{12}Z+p_{13}}{p_{20}X+p_{21}Y+p_{22}Z+p_{23}};
    $$

    clearing denominators and expanding turns each correspondence into 2
    equations linear in $P$'s entries,

    $$
    p_{00}X+p_{01}Y+p_{02}Z+p_{03} - p_{20}xX-p_{21}xY-p_{22}xZ-p_{23}x = 0, \qquad
    p_{10}X+p_{11}Y+p_{12}Z+p_{13} - p_{20}yX-p_{21}yY-p_{22}yZ-p_{23}y = 0;
    $$

    and flattening $P$ row-major into the 12-vector
    $\mathrm{vec}(P)=(p_{00},p_{01},p_{02},p_{03},\,p_{10},p_{11},p_{12},p_{13},\,p_{20},p_{21},p_{22},p_{23})^T$
    turns each equation into a dot product read straight off it:

    $$
    \begin{bmatrix} X & Y & Z & 1 & 0&0&0&0 & -xX & -xY & -xZ & -x \\ 0&0&0&0 & X & Y & Z & 1 & -yX & -yY & -yZ & -y \end{bmatrix} \mathrm{vec}(P) = 0.
    $$

    Stack 2 rows per correspondence into $A$ ($2N\times12$): $A\,\mathrm{vec}(P)=0$,
    solved exactly as before by the smallest right singular vector of $A$.
    $P$ has 11 DOF, so $N\ge6$ points (in general position, not all on one
    plane) pin it down.

    **Recovering $K,R,t$ from $P$.** Write $P=[M\mid P_4]$ with $M$ the left
    $3\times3$ block; since $M=KR$, we need to factor an upper-triangular $K$
    *times* an orthogonal $R$ — note the order: `np.linalg.qr` only gives
    orthogonal-times-upper-triangular, the reverse. This is an **RQ
    decomposition** (named for its output order), computed from an ordinary
    QR via a standard reversal trick (flip $M$'s rows, transpose, QR, flip
    back), then sign-corrected so $K$'s diagonal is positive (real focal
    lengths) and $\det R=+1$ (a proper rotation, not a reflection). Finally
    $t=K^{-1}P_4$ -- using the *same* (unnormalized) $K$ the RQ step produced
    before rescaling it to $K_{33}=1$, since $P_4$ was estimated at that same
    unknown overall scale.
    """)
    return


@app.function
def camera_matrix_dlt(X3d, x2d, np):
    """DLT for the 3x4 camera matrix P from >=6 2D-3D correspondences."""
    A = []
    for (X, Y, Z), (x, y) in zip(X3d, x2d):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


@app.function
def rq_decomposition(M, np):
    """RQ decomposition M = K @ R (K upper-triangular, R orthogonal), via the
    standard flip-QR-flip trick -- numpy only provides QR directly. Fixes
    signs so K's diagonal is positive and det(R)=+1."""
    Q_qr, R_qr = np.linalg.qr(np.flipud(M).T)
    K = np.fliplr(np.flipud(R_qr.T))
    R = np.flipud(Q_qr.T)
    d = np.sign(np.diag(K))
    d[d == 0] = 1
    D = np.diag(d)
    return K @ D, D @ R


@app.function
def decompose_camera_matrix(P, np):
    """P = K[R|t] -> (K, R, t), K normalized so K[2,2]=1."""
    M = P[:, :3]
    K_raw, R = rq_decomposition(M, np)
    t = np.linalg.inv(K_raw) @ P[:, 3]
    return K_raw / K_raw[2, 2], R, t


@app.cell
def _(mo):
    mo.md(r"""
    Validate the whole pipeline on a synthetic 3D calibration object: points
    scattered in front of a camera with known ground-truth $K,R,t$, observed
    through it with added pixel noise. Recover $\hat P$ via the DLT above,
    decompose it into $\hat K,\hat R,\hat t$, and compare each to the ground
    truth -- the sliders control how many correspondences are used and how
    noisy they are.
    """)
    return


@app.cell
def _(mo):
    pose_npts_slider = mo.ui.slider(start=6, stop=150, value=20, step=2, label="number of 2D-3D correspondences N", debounce=True)
    pose_noise_slider = mo.ui.slider(start=0.0, stop=2.0, value=0.5, step=0.1, label="pixel noise σ", debounce=True)
    mo.hstack([pose_npts_slider, pose_noise_slider], justify="start", gap=2)
    return pose_noise_slider, pose_npts_slider


@app.cell
def _(mo, np, pose_noise_slider, pose_npts_slider):
    def _rot_from_euler(rx, ry, rz, np):
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    _K_true = np.array([[600.0, 0, 320.0], [0, 580.0, 240.0], [0, 0, 1.0]])
    _R_true = _rot_from_euler(0.15, -0.25, 0.1, np)
    _t_true = np.array([0.5, -0.3, 8.0])
    _P_true = _K_true @ np.hstack([_R_true, _t_true.reshape(3, 1)])

    _rng = np.random.default_rng(0)
    _N = pose_npts_slider.value
    _X3d = _rng.uniform(-2, 2, (_N, 3))
    _X3d[:, 2] += 5

    def _project(P, X3d, np):
        Xh = np.hstack([X3d, np.ones((len(X3d), 1))])
        x = (P @ Xh.T).T
        return x[:, :2] / x[:, 2:3]

    _x2d = _project(_P_true, _X3d, np)
    _x2d_noisy = _x2d + _rng.normal(0, pose_noise_slider.value, _x2d.shape)

    _P_est = camera_matrix_dlt(_X3d, _x2d_noisy, np)
    if np.dot(_P_est[2, :3], _P_true[2, :3]) < 0:
        _P_est = -_P_est
    _K_est, _R_est, _t_est = decompose_camera_matrix(_P_est, np)

    mo.md(f"""
    | | $f_x$ | $f_y$ | $c_x$ | $c_y$ |
    |---|---|---|---|---|
    | true | {_K_true[0,0]:.1f} | {_K_true[1,1]:.1f} | {_K_true[0,2]:.1f} | {_K_true[1,2]:.1f} |
    | estimated | {_K_est[0,0]:.1f} | {_K_est[1,1]:.1f} | {_K_est[0,2]:.1f} | {_K_est[1,2]:.1f} |

    $\\det(R_{{est}})={np.linalg.det(_R_est):.4f}$ (should be $+1$), $\\;\\|R_{{est}}-R_{{true}}\\|_\\infty={np.abs(_R_est-_R_true).max():.4f}$,
    $\\;\\|t_{{est}}-t_{{true}}\\|_\\infty={np.abs(_t_est-_t_true).max():.4f}$ (true $t=({_t_true[0]:.1f},{_t_true[1]:.1f},{_t_true[2]:.1f})$).
    Push $N$ up or $\\sigma$ down and watch every column converge -- exactly
    notebook 4's "a single image can't disentangle intrinsics from
    extrinsics" caveat resolved by using *enough* known 3D points at once.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Planar calibration: Zhang's method

    §6's recipe needs precisely-measured 3D points — a real calibration rig
    (Tsai 1987) that's awkward to build and measure accurately. Zhang (2000)
    showed a much more practical alternative: photograph an ordinary **flat**
    checkerboard, whose points all have a known, trivial 3D layout ($Z=0$ in
    the board's own frame), from several *unknown* poses, and let the
    overlapping constraints across views pin down $K$.

    The two panels below show the same five poses two equivalent ways: moving
    the checkerboard in front of a fixed camera (left) is geometrically
    identical to holding the checkerboard still and moving the camera around
    it (right) -- only *relative* pose between camera and board ever matters,
    which is exactly what each view's homography $H_j$ captures.
    """)
    return


@app.function
def rot_from_euler_xyz(rx, ry, rz, np):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@app.function
def camera_pyramid(C, R_cam_to_world, depth, hw, hh, np):
    """Apex at the camera center C; base is the image plane, `depth` along
    the camera's forward (+z) axis, spanned by its right/up axes."""
    right, up, fwd = R_cam_to_world[:, 0], R_cam_to_world[:, 1], R_cam_to_world[:, 2]
    center = C + fwd * depth
    corners = [center + s1 * hw * right + s2 * hh * up for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return C, corners


@app.function
def pyramid_lines(apex, corners):
    """(x,y,z) coordinate lists for one Scatter3d 'lines' trace drawing the
    pyramid's 4 edges from the apex plus its 4 base edges, None-separated."""
    xs, ys, zs = [], [], []
    for c in corners:
        xs += [apex[0], c[0], None]
        ys += [apex[1], c[1], None]
        zs += [apex[2], c[2], None]
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return xs, ys, zs


@app.function
def checkerboard_mesh(R, t, color, np, nx=6, ny=4, sx=180.0, sy=120.0):
    """Vertices + triangle indices + per-face colors for an nx*ny checker
    grid (alternating 'white' and `color`), centered at the origin in
    board-local coordinates and placed in the world by (R, t)."""
    xs = np.linspace(-sx / 2, sx / 2, nx + 1)
    ys = np.linspace(-sy / 2, sy / 2, ny + 1)
    verts_local, vid = [], {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            vid[(ix, iy)] = len(verts_local)
            verts_local.append([xs[ix], ys[iy], 0.0])
    verts_world = (R @ np.array(verts_local).T).T + t

    I, J, K, facecolor = [], [], [], []
    for iy in range(ny):
        for ix in range(nx):
            a, b, c, d = vid[(ix, iy)], vid[(ix + 1, iy)], vid[(ix + 1, iy + 1)], vid[(ix, iy + 1)]
            I += [a, a]
            J += [b, c]
            K += [c, d]
            cc = color if (ix + iy) % 2 == 0 else "white"
            facecolor += [cc, cc]
    return verts_world, I, J, K, facecolor


@app.function
def board_outline(R, t, np, sx=180.0, sy=120.0):
    """(x,y,z) coordinate lists for one closed Scatter3d 'lines' trace
    tracing a checkerboard's outer rectangle, for a visible contour."""
    corners_local = np.array([[-sx / 2, -sy / 2, 0], [sx / 2, -sy / 2, 0], [sx / 2, sy / 2, 0], [-sx / 2, sy / 2, 0], [-sx / 2, -sy / 2, 0]])
    corners_world = (R @ corners_local.T).T + t
    return corners_world[:, 0], corners_world[:, 1], corners_world[:, 2]


@app.cell
def _(go, make_subplots, np):
    _deg = np.radians
    _tilt = _deg(24)
    _viz_poses = [(0.0, 0.0, 0.0, np.array([0.0, 0.0, 380.0]))]
    _offsets = [(-170, -130), (170, -130), (-170, 150), (170, 150)]
    for _k, _phi_deg in enumerate([45, 135, 225, 315]):
        _phi = _deg(_phi_deg)
        _rx, _ry = _tilt * np.sin(_phi), _tilt * np.cos(_phi)
        _ox, _oy = _offsets[_k]
        _viz_poses.append((_rx, _ry, _deg(5) * (-1) ** _k, np.array([_ox, _oy, 360.0])))

    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # plotly's own qualitative palette

    _fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("camera fixed, checkerboard at 5 poses", "checkerboard fixed, camera at 5 poses"),
    )

    _apex0, _corners0 = camera_pyramid(np.zeros(3), np.eye(3), depth=55, hw=38, hh=27, np=np)
    _xs, _ys, _zs = pyramid_lines(_apex0, _corners0)
    _fig.add_trace(go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines", line=dict(color="black", width=6), showlegend=False), row=1, col=1)
    for _i, (_rx, _ry, _rz, _t) in enumerate(_viz_poses):
        _R = rot_from_euler_xyz(_rx, _ry, _rz, np)
        _V, _I, _J, _K, _fc = checkerboard_mesh(_R, _t, _colors[_i], np)
        _fig.add_trace(
            go.Mesh3d(x=_V[:, 0], y=_V[:, 1], z=_V[:, 2], i=_I, j=_J, k=_K, facecolor=_fc,
                       flatshading=True, lighting=dict(ambient=1, diffuse=0, specular=0),
                       name=f"pose {_i+1}", showlegend=False, hoverinfo="name"),
            row=1, col=1,
        )
        _ox, _oy, _oz = board_outline(_R, _t, np)
        _fig.add_trace(go.Scatter3d(x=_ox, y=_oy, z=_oz, mode="lines", line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"), row=1, col=1)

    _Vb, _Ib, _Jb, _Kb, _fcb = checkerboard_mesh(np.eye(3), np.zeros(3), "lightgray", np)
    _fig.add_trace(
        go.Mesh3d(x=_Vb[:, 0], y=_Vb[:, 1], z=_Vb[:, 2], i=_Ib, j=_Jb, k=_Kb, facecolor=_fcb,
                   flatshading=True, lighting=dict(ambient=1, diffuse=0, specular=0), showlegend=False),
        row=1, col=2,
    )
    _obx, _oby, _obz = board_outline(np.eye(3), np.zeros(3), np)
    _fig.add_trace(go.Scatter3d(x=_obx, y=_oby, z=_obz, mode="lines", line=dict(color="black", width=3), showlegend=False, hoverinfo="skip"), row=1, col=2)
    for _i, (_rx, _ry, _rz, _t) in enumerate(_viz_poses):
        _R = rot_from_euler_xyz(_rx, _ry, _rz, np)
        _C = -_R.T @ _t
        _apex, _corners = camera_pyramid(_C, _R.T, depth=55, hw=38, hh=27, np=np)
        _xs, _ys, _zs = pyramid_lines(_apex, _corners)
        _fig.add_trace(
            go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines", line=dict(color=_colors[_i], width=6),
                         name=f"pose {_i+1}", showlegend=False, hoverinfo="name"),
            row=1, col=2,
        )

    _fig.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    _fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why a flat target collapses to a homography.** With $Z=0$ for every
    board point, the third column of $R$ never multiplies anything nonzero,
    so the projection $P=K[r_1\ r_2\ r_3\mid t]$ simplifies to

    $$ \tilde x \sim K[r_1\ r_2\ t]\,(X,Y,1)^T, $$

    i.e. board-to-image is *exactly* a homography $H=K[r_1\ r_2\ t]$ (up to
    scale) — estimate one per view with §3-5's own machinery, no new
    algorithm needed.

    **From homographies to $K$.** Write $H=[h_1\ h_2\ h_3]$ (columns), so
    $h_1=\lambda Kr_1,\ h_2=\lambda Kr_2$ for some scale $\lambda$. Since
    $r_1,r_2$ are orthonormal columns of a rotation ($r_1\cdot r_2=0$,
    $\|r_1\|=\|r_2\|$), substituting $r_i=K^{-1}h_i/\lambda$ turns these into
    two *linear* constraints on $B=K^{-T}K^{-1}$ (symmetric, 6 unknowns;
    Zhang's "image of the absolute conic," Eq. 11.1):

    $$ h_1^TBh_2=0, \qquad h_1^TBh_1=h_2^TBh_2. $$

    Each view contributes 2 such equations; stack $\ge3$ views into one
    $V\,b=0$ ($b$ the 6 entries of $B$) and — a **third** use of §2's exact
    trick — solve for $b$ as $V$'s smallest right singular vector. $K$ then
    falls out of $B$ via a **Cholesky factorization** ($B=LL^T$, $L$
    lower-triangular; since $B=K^{-T}K^{-1}=(K^{-1})^TK^{-1}$ and $K^{-1}$ is
    itself upper-triangular, $L=K^{-T}$ up to scale and sign, fixed up the same
    way as §6). Each view's $r_1,r_2,t$ then come directly from $H$ and the
    now-known $K$; $r_3=r_1\times r_2$ completes a rotation that's projected
    onto the *nearest true* rotation via one more SVD (an orthogonal
    Procrustes fit) to absorb estimation noise.
    """)
    return


@app.function
def homography_v_vector(H, i, j, np):
    """Zhang's v_ij: the vector with h_i^T B h_j = v_ij . b (Eq. 11.1)."""
    h = H
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j],
    ])


@app.function
def zhang_calibrate(homographies, np):
    """Estimate K from >=3 homographies of one planar pattern (Zhang 2000)."""
    V = []
    for H in homographies:
        V.append(homography_v_vector(H, 0, 1, np))
        V.append(homography_v_vector(H, 0, 0, np) - homography_v_vector(H, 1, 1, np))
    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if B[0, 0] < 0 or B[1, 1] < 0 or B[2, 2] < 0:
        B = -B
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L.T)
    return K / K[2, 2]


@app.function
def zhang_recover_extrinsics(H, K, np):
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = (1.0 / np.linalg.norm(K_inv @ h1) + 1.0 / np.linalg.norm(K_inv @ h2)) / 2
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    t = lam * (K_inv @ h3)
    return R, t


@app.cell
def _(mo):
    mo.md(r"""
    ### Predict

    $B$ has 6 unknowns, and each view contributes 2 linear equations. How
    many views should be the bare minimum for the homogeneous system to pin
    down $b$ at all? Try the slider below at that number and one fewer.
    """)
    return


@app.cell
def _(mo):
    zhang_views_slider = mo.ui.slider(start=2, stop=7, value=5, step=1, label="number of calibration views", debounce=True)
    zhang_noise_slider = mo.ui.slider(start=0.0, stop=1.0, value=0.3, step=0.1, label="corner-detection noise σ (px)", debounce=True)
    mo.hstack([zhang_views_slider, zhang_noise_slider], justify="start", gap=2)
    return zhang_noise_slider, zhang_views_slider


@app.cell
def _(mo, np, zhang_noise_slider, zhang_views_slider):
    def _rot_from_euler(rx, ry, rz, np):
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    _K_true = np.array([[700.0, 0, 320.0], [0, 680.0, 240.0], [0, 0, 1.0]])
    _gx, _gy = np.meshgrid(np.arange(7) * 30.0, np.arange(5) * 30.0)
    _board_pts = np.stack([_gx.ravel(), _gy.ravel()], axis=1)

    _all_poses = [
        (0.2, 0.1, 0.0, np.array([-50, -40, 400.0])),
        (-0.3, 0.25, 0.1, np.array([30, -20, 420.0])),
        (0.1, -0.35, 0.2, np.array([-20, 30, 380.0])),
        (0.35, 0.15, -0.2, np.array([10, 10, 450.0])),
        (-0.15, -0.2, 0.3, np.array([0, 0, 400.0])),
        (0.25, -0.1, 0.15, np.array([20, -10, 410.0])),
        (-0.2, 0.3, -0.1, np.array([-10, 20, 390.0])),
    ]

    _rng = np.random.default_rng(0)
    _Hs, _Rs_true, _ts_true = [], [], []
    for _rx, _ry, _rz, _t in _all_poses[: zhang_views_slider.value]:
        _R = _rot_from_euler(_rx, _ry, _rz, np)
        _H_true = _K_true @ np.stack([_R[:, 0], _R[:, 1], _t], axis=1)
        _H_true = _H_true / _H_true[2, 2]
        _pts_h = np.hstack([_board_pts, np.ones((len(_board_pts), 1))])
        _img_pts = (_H_true @ _pts_h.T).T
        _img_pts = _img_pts[:, :2] / _img_pts[:, 2:3]
        _img_pts_noisy = _img_pts + _rng.normal(0, zhang_noise_slider.value, _img_pts.shape)
        _Hs.append(homography_dlt(_board_pts, _img_pts_noisy, np))
        _Rs_true.append(_R)
        _ts_true.append(_t)

    try:
        _K_est = zhang_calibrate(_Hs, np)
        _r_errs, _t_errs = [], []
        for _H_est, _R_true, _t_true in zip(_Hs, _Rs_true, _ts_true):
            _R_est, _t_est = zhang_recover_extrinsics(_H_est, _K_est, np)
            _r_errs.append(np.abs(_R_est - _R_true).max())
            _t_errs.append(np.abs(_t_est - _t_true).max())
        _status = mo.md(f"""
        | | $f_x$ | $f_y$ | $c_x$ | $c_y$ | skew |
        |---|---|---|---|---|---|
        | true | {_K_true[0,0]:.1f} | {_K_true[1,1]:.1f} | {_K_true[0,2]:.1f} | {_K_true[1,2]:.1f} | 0.0 |
        | estimated (shared across all views) | {_K_est[0,0]:.1f} | {_K_est[1,1]:.1f} | {_K_est[0,2]:.1f} | {_K_est[1,2]:.1f} | {_K_est[0,1]:.2f} |

        Per-view extrinsics recovered with max $R$ error {max(_r_errs):.4f} and
        max $t$ error {max(_t_errs):.3f} across all
        {zhang_views_slider.value} views -- **one** shared $K$ explains every
        view at once, exactly what a real checkerboard calibration routine
        (e.g. OpenCV's `calibrateCamera`) delivers.
        """)
    except np.linalg.LinAlgError:
        _status = mo.md(
            f"**Failed at {zhang_views_slider.value} view(s):** $B$ isn't "
            f"positive-definite, so Cholesky factorization fails -- too few "
            f"independent constraints (6 unknowns need at least 3 views' "
            f"worth of equations)."
        )
    _status
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        $6$ unknowns need at least $6$ equations, and each view gives $2$, so
        **3 views is the theoretical minimum** — confirmed above: 2 views
        either throws a Cholesky failure or returns a nonsense $K$ (the
        homogeneous system is rank-deficient), while 3 or more views recovers
        $K$ accurately, improving further as views are added (more
        overdetermined equations for the same SVD to average over, exactly
        like §3's "more noisy data helps" lesson).
        """)
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Lens distortion

    One real-world wrinkle every method above has quietly ignored: real
    lenses aren't perfect pinholes, and straight lines in the world often
    don't stay straight in the image. The standard fix is a low-order
    polynomial correction in normalized coordinates $(x,y)$ (radius
    $r^2=x^2+y^2$), the **Brown-Conrady model** (Brown 1966, 1971; the book's
    Eq. 2.78-2.79, revisited for calibration in Eq. 11.10),

    $$ x_d = x(1+k_1r^2+k_2r^4), \qquad y_d = y(1+k_1r^2+k_2r^4). $$

    A real calibration pipeline estimates $k_1,k_2$ *jointly* with $K,R,t$ via
    nonlinear refinement (Zhang 2000; Hartley and Kang 2007) rather than as a
    separate linear step — genuinely more machinery than fits here, so this
    stays at the visual, citation-only level.
    """)
    return


@app.cell
def _(mo):
    k1_slider = mo.ui.slider(start=-0.6, stop=0.6, value=-0.3, step=0.1, label="k1", debounce=True)
    k2_slider = mo.ui.slider(start=-0.3, stop=0.3, value=0.1, step=0.05, label="k2", debounce=True)
    mo.hstack([k1_slider, k2_slider], justify="start", gap=2)
    return k1_slider, k2_slider


@app.cell
def _(k1_slider, k2_slider, mo, np, plt):
    _lines = []
    for _v in np.linspace(-0.8, 0.8, 9):
        _xs = np.linspace(-0.8, 0.8, 50)
        _lines.append(np.stack([_xs, np.full_like(_xs, _v)], axis=1))
        _lines.append(np.stack([np.full_like(_xs, _v), _xs], axis=1))

    def _distort(pts, k1, k2, np):
        r2 = (pts**2).sum(axis=1)
        return pts * (1 + k1 * r2 + k2 * r2**2)[:, None]

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    for _line in _lines:
        _axes[0].plot(_line[:, 0], _line[:, 1], "b-", lw=1)
        _d = _distort(_line, k1_slider.value, k2_slider.value, np)
        _axes[1].plot(_d[:, 0], _d[:, 1], "r-", lw=1)
    _axes[0].set_title("ideal grid")
    _axes[1].set_title(f"distorted (k1={k1_slider.value}, k2={k2_slider.value})")
    for _ax in _axes:
        _ax.set_aspect("equal")
        _ax.set_xlim(-1.2, 1.2)
        _ax.set_ylim(-1.2, 1.2)
    _fig.tight_layout()
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Beyond today

    - **Vanishing-point calibration** (Eq. 11.2-11.3) -- already foreshadowed
      in notebook 17 §4: parallel 3D lines' image intersections constrain $K$
      directly, no checkerboard needed.
    - **Single-view metrology** (Criminisi, Reid, and Zisserman 2000) --
      measuring real-world distances from one photo using known vanishing
      points and a single reference length.
    - **Self-calibration from motion** (§11.1.3, Eq. 11.4-11.9) -- recovering
      $K$ purely from a moving camera's own image sequence, no calibration
      target at all.
    - **Nonlinear refinement.** Every DLT above is a *linear* approximation;
      real pipelines follow it with nonlinear least-squares (Eq. 8.22-8.23's
      iteratively reweighted refinement) minimizing true reprojection error
      directly -- what this notebook calls "good enough," a real system calls
      "a good initial guess for Levenberg-Marquardt."
    - **Two-view epipolar geometry** -- the essential and fundamental
      matrices, the normalized 8-point algorithm (Hartley 1997), and
      recovering camera pose *without* a known 3D target or planar pattern at
      all (§11.3). A distinct, meatier topic -- structure from motion -- left
      for its own future notebook.

    ## Summary

    - The **SVD** factors any matrix $A=U\Sigma V^T$; the one fact used
      everywhere here is that the unit vector minimizing $\|Ah\|$ is $A$'s
      smallest right singular vector -- notebook 13's eigenvalue/Rayleigh-quotient
      argument, applied to $A^TA$.
    - **Homography DLT** (Eq. 8.19/8.21) generalizes notebooks 12/14's exact
      4-point solve to noisy, overdetermined data via that same SVD fact;
      **normalization** (Hartley and Zisserman 2004) keeps the linear system
      well-conditioned; **RANSAC** (Eq. 8.30) makes it robust to real
      mismatches.
    - The same DLT trick, scaled up to 12 unknowns, estimates the full
      **camera matrix** from 2D-3D correspondences (Eq. 11.11-11.13); **RQ
      decomposition** pulls $K$ and $R$ back out.
    - **Zhang's method** (Eq. 11.1) replaces a hard-to-build 3D rig with
      several views of one flat pattern, using the *same* homography and SVD
      machinery a third time to solve for a shared $K$ across all views.
    - **Lens distortion** (Eq. 2.78-2.79/11.10) and nonlinear refinement are
      the real-world finishing touches left for citation only.
    """)
    return


if __name__ == "__main__":
    app.run()
