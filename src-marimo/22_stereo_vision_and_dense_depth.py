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
    from scipy.ndimage import uniform_filter, map_coordinates

    return go, map_coordinates, mo, np, plt, uniform_filter


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    EXTERNAL_FIGURES_DIR = IMAGES_DIR / "external_figures"
    return (EXTERNAL_FIGURES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Stereo Vision and Dense Depth Estimation

    Notebook 21 recovers *sparse* point correspondences and an *unknown*
    relative pose from two views. This notebook assumes the opposite is
    already true -- the stereo rig's relative pose is known (from notebook 20
    or 21, or a physical calibration rig) -- and asks a different question:
    for *every* pixel, how far away is it? The result is a dense depth map
    tied to one camera's viewpoint, sometimes informally called a "2.5D"
    representation in the wider literature to distinguish it from a full 3D
    model -- though it's worth being upfront that **this isn't actually a term
    Szeliski's book uses anywhere**; the book just calls this Chapter 12
    "Depth estimation," and that's the name used below.

    This covers Szeliski **§12.1 (epipolar geometry and rectification), §12.3-12.4
    (dense correspondence, local methods), and §12.5 (global optimization)**,
    following the same matching-cost / aggregation / optimization / refinement
    taxonomy as Scharstein and Szeliski (2002) -- the same paper whose standard
    test images (Tsukuba, Map, Venus) this notebook uses directly.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting (notebooks 8-21's convention)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Recap: epipolar geometry

    From notebook 21: two cameras observing the same scene relate corresponding
    points by the epipolar constraint, and knowing the relative pose $(R,t)$
    tells you, for any point in one image, exactly which *line* its match must
    lie on in the other image (the epipolar line) -- never the full 2D image.
    §2 below pushes this one step further: with the right coordinate change,
    that line can always be made *horizontal*, at the *same row* in both
    images.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Epipolar rectification

    Rotate both cameras (conceptually) to a new, shared orientation: its
    $x$-axis along the baseline $C_1-C_0$, its $z$-axis close to the average
    of the two original viewing directions, and $y$ completing a right-handed
    frame. With both cameras sharing this orientation (and, for convenience,
    the same intrinsics $K$), the epipolar lines become identical horizontal
    scanlines in both images, and corresponding points satisfy the
    **post-rectification correspondence relation**

    $$ x' = x + d(x,y), \qquad y' = y. \qquad (\text{Eq. 12.2}) $$

    All the 2D search notebook 21 needed collapses to a **1D search along one
    row** -- the entire reason dense stereo matching is tractable at pixel
    scale. The procedure itself is described only in prose in the book, not
    as one boxed equation (Loop and Zhang 1999; Fusiello, Trucco, and Verri
    2000; Hartley and Zisserman 2004); the demo below builds the rectifying
    homographies $H_0,H_1$ explicitly and checks Eq. 12.2 holds on a handful
    of 3D points, using the same toe-in two-camera setup as notebook 21 §2.
    """)
    return


@app.function
def rectifying_homographies(K0, R0, t0, K1, R1, t1, K_new, np):
    """Build H0, H1 mapping each camera's pixels into a shared rectified
    frame: x-axis along the baseline, z-axis near the average viewing
    direction, so post-rectification correspondences satisfy Eq. 12.2."""
    C0 = -R0.T @ t0
    C1 = -R1.T @ t1
    right_new = (C1 - C0) / np.linalg.norm(C1 - C0)
    avg_fwd = R0.T[:, 2] + R1.T[:, 2]
    avg_fwd = avg_fwd - (avg_fwd @ right_new) * right_new
    fwd_new = avg_fwd / np.linalg.norm(avg_fwd)
    up_new = np.cross(fwd_new, right_new)
    up_new /= np.linalg.norm(up_new)
    right_new = np.cross(up_new, fwd_new)
    R_new_cam_to_world = np.stack([right_new, up_new, fwd_new], axis=1)
    R_new = R_new_cam_to_world.T
    H0 = K_new @ R_new @ R0.T @ np.linalg.inv(K0)
    H1 = K_new @ R_new @ R1.T @ np.linalg.inv(K1)
    return H0, H1


@app.function
def apply_homography(H, pts, np):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ pts_h.T).T
    return out[:, :2] / out[:, 2:3]


@app.function
def fundamental_matrix(K0, R0, t0, K1, R1, t1, np):
    """F relating image0 and image1 (same X_cam = R X_world + t convention
    used throughout): l1 = F@x0 is x0's epipolar line in image1, l0 = F.T@x1
    is x1's epipolar line in image0 -- tilted in general, exactly horizontal
    (and at a matching row) only once both images are rectified (Eq. 12.2)."""
    R_rel = R1 @ R0.T
    t_rel = t1 - R_rel @ t0
    t_x = np.array([[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]])
    E = t_x @ R_rel
    return np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)


@app.function
def project_points(K, R, t, X, np):
    Xc = (R @ X.T).T + t
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


@app.function
def rot_from_euler_xyz(rx, ry, rz, np):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@app.function
def rotation_homography(K, R, np):
    """Homography for a pure rotation about the camera's own optical center:
    x' = K R K^-1 x, valid at ANY scene depth (no planarity assumption) --
    unlike a general planar-capture homography, this is the right tool for
    re-posing a real, non-planar photo (e.g. Tsukuba) as if a different,
    nearby-but-differently-tilted camera had captured it."""
    return K @ R @ np.linalg.inv(K)


@app.function
def inscribed_crop(H, Wt, Ht, np, margin=2.0):
    """The largest centered rectangle (matching the original aspect ratio) in
    H's output canvas whose H^-1 preimage stays fully inside the original
    [0,Wt]x[0,Ht] image -- a homography maps convex sets to convex sets, so
    checking just the 4 rectangle corners is enough to guarantee the whole
    rectangle's preimage (and hence every pixel in it) is valid, un-warped
    content, with no black border."""
    H_inv = np.linalg.inv(H)
    corners_out = apply_homography(H, np.array([[0, 0], [Wt, 0], [Wt, Ht], [0, Ht]]), np)
    cx, cy = corners_out[:, 0].mean(), corners_out[:, 1].mean()
    aspect = Wt / Ht

    def valid(s):
        hh, hw = s, s * aspect
        rect = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
        pre = apply_homography(H_inv, rect, np)
        return np.all((pre[:, 0] >= margin) & (pre[:, 0] <= Wt - margin) & (pre[:, 1] >= margin) & (pre[:, 1] <= Ht - margin))

    lo, hi = 0.0, 1.0
    while valid(hi) and hi < 1e5:
        hi *= 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if valid(mid) else (lo, mid)
    hh, hw = lo, lo * aspect
    return cx - hw, cy - hh, cx + hw, cy + hh


@app.function
def warp_image_by_homography(img, H, np, map_coordinates, out_shape=None):
    """Inverse-warp img by H onto a canvas of out_shape (defaults to img's own
    shape), sampling with bilinear interpolation and zero-fill outside img."""
    Hh, Ww = out_shape if out_shape is not None else img.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    pts = np.stack([xx.ravel().astype(float), yy.ravel().astype(float), np.ones(Hh * Ww)])
    src = np.linalg.inv(H) @ pts
    src_x = (src[0] / src[2]).reshape(Hh, Ww)
    src_y = (src[1] / src[2]).reshape(Hh, Ww)
    return map_coordinates(img, [src_y, src_x], order=1, mode="constant", cval=0.0)


@app.cell
def _(mo, np):
    _K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    _C0, _C1 = np.zeros(3), np.array([1.8, 0.0, 0.0])
    _X = np.array([[0.3, -0.15, 2.0], [0.5, 0.1, 2.5], [-0.2, 0.2, 3.0], [0.0, -0.3, 2.2], [0.4, 0.3, 2.8]])

    _fwd1 = (_X.mean(axis=0) - _C1)
    _fwd1 /= np.linalg.norm(_fwd1)
    _right1 = np.cross(_fwd1, np.array([0.0, 1.0, 0.0]))
    _right1 /= np.linalg.norm(_right1)
    _up1 = np.cross(_fwd1, _right1)
    _R1 = np.stack([_right1, _up1, _fwd1], axis=1).T
    _t1 = -_R1 @ _C1
    _R0, _t0 = np.eye(3), np.zeros(3)

    _x0 = project_points(_K, _R0, _t0, _X, np)
    _x1 = project_points(_K, _R1, _t1, _X, np)

    _H0, _H1 = rectifying_homographies(_K, _R0, _t0, _K, _R1, _t1, _K, np)
    _x0r = apply_homography(_H0, _x0, np)
    _x1r = apply_homography(_H1, _x1, np)

    mo.md(f"""
    | point | $y_0$ (before) | $y_1$ (before) | $y_0'$ (after) | $y_1'$ (after) |
    |---|---|---|---|---|
    {chr(10).join(f"| {i+1} | {_x0[i,1]:.1f} | {_x1[i,1]:.1f} | {_x0r[i,1]:.2f} | {_x1r[i,1]:.2f} |" for i in range(len(_X)))}

    Before rectification, each point's row differs between images (general
    epipolar lines). After applying $H_0,H_1$, $y_0'=y_1'$ for *every* point
    (max discrepancy {np.abs(_x0r[:,1]-_x1r[:,1]).max():.1e}, floating-point
    noise) -- exactly Eq. 12.2. Every demo from here on assumes this step has
    already happened (both the real datasets below are already rectified, as
    published).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The table above checks Eq. 12.2 numerically on isolated 3D points, but
    it's worth *seeing* what rectification does to actual images and to
    **epipolar lines** themselves. Below, Tsukuba's real (already-rectified)
    left/right pair stands in for two cameras' raw, *unrectified* captures:
    each image is re-warped by its own synthetic rotation-only homography
    ($x'=KRK^{-1}x$, valid at any scene depth, not just a flat plane -- a
    different tilt for left and right), then cropped to the largest region
    with no black border, so each still looks like an ordinary photo. Because
    the *original* pair's rectified geometry and the two applied rotations
    are both exactly known, so is the resulting fundamental matrix $F$ --
    and, via Tsukuba's own ground-truth disparity, so is the *exact* match
    for any pixel, with no estimation involved. The colored line through each
    point is its real epipolar line (from $F$): tilted here, before
    rectification. Re-deriving $H_0,H_1$ for these two tilted cameras and
    applying them below recovers a shared rectified frame -- now with real
    black regions (neither original photo quite covers it) but perfectly
    horizontal, shared-row epipolar lines.
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, map_coordinates, mo, np, plt):
    _left, _right, _gt = load_middlebury_pair("tsukuba", EXTERNAL_FIGURES_DIR, plt, np)
    _Ht, _Wt = _left.shape
    _f = 500.0
    _cx, _cy = _Wt / 2, _Ht / 2
    _K = np.array([[_f, 0, _cx], [0, _f, _cy], [0, 0, 1.0]])
    _K_new = np.array([[0.88 * _f, 0, _cx], [0, 0.88 * _f, _cy], [0, 0, 1.0]])

    # Tsukuba's own pair is already rectified: R_orig=I for both, baseline
    # along world x. Each camera then gets its own extra synthetic tilt.
    _B = 1.0
    _R0_extra = rot_from_euler_xyz(0.0, 0.18, 0.12, np)
    _R1_extra = rot_from_euler_xyz(0.0, -0.14, -0.16, np)
    _R0, _t0 = _R0_extra, np.zeros(3)
    _R1, _t1 = _R1_extra, _R1_extra @ np.array([-_B, 0.0, 0.0])
    _H0 = rotation_homography(_K, _R0_extra, np)
    _H1 = rotation_homography(_K, _R1_extra, np)
    _F = fundamental_matrix(_K, _R0, _t0, _K, _R1, _t1, np)

    _cand = [(150, 90), (300, 130)]
    _feat0 = np.array(_cand, dtype=float)
    _feat1 = np.array([(x - _gt[y, x], y) for x, y in _cand])

    _p0_pre = apply_homography(_H0, _feat0, np)
    _p1_pre = apply_homography(_H1, _feat1, np)
    _p0_pre_h = np.hstack([_p0_pre, np.ones((2, 1))])
    _p1_pre_h = np.hstack([_p1_pre, np.ones((2, 1))])
    _lines0_pre = (_F.T @ _p1_pre_h.T).T  # each point's line in the OTHER image
    _lines1_pre = (_F @ _p0_pre_h.T).T

    _x0c, _y0c, _x1c, _y1c = inscribed_crop(_H0, _Wt, _Ht, np)
    _x0c1, _y0c1, _x1c1, _y1c1 = inscribed_crop(_H1, _Wt, _Ht, np)
    _cw0, _ch0 = int(round(_x1c - _x0c)), int(round(_y1c - _y0c))
    _cw1, _ch1 = int(round(_x1c1 - _x0c1)), int(round(_y1c1 - _y0c1))
    _Tcrop0 = np.array([[1, 0, -_x0c], [0, 1, -_y0c], [0, 0, 1]])
    _Tcrop1 = np.array([[1, 0, -_x0c1], [0, 1, -_y0c1], [0, 0, 1]])
    _before0 = warp_image_by_homography(_left, _Tcrop0 @ _H0, np, map_coordinates, out_shape=(_ch0, _cw0))
    _before1 = warp_image_by_homography(_right, _Tcrop1 @ _H1, np, map_coordinates, out_shape=(_ch1, _cw1))
    _b0 = apply_homography(_Tcrop0, _p0_pre, np)
    _b1 = apply_homography(_Tcrop1, _p1_pre, np)
    _lines0 = (np.linalg.inv(_Tcrop0).T @ _lines0_pre.T).T
    _lines1 = (np.linalg.inv(_Tcrop1).T @ _lines1_pre.T).T

    _Hrect0, _Hrect1 = rectifying_homographies(_K, _R0, _t0, _K, _R1, _t1, _K_new, np)
    _Hfull0, _Hfull1 = _Hrect0 @ _H0, _Hrect1 @ _H1
    _center = np.array([[_cx, _cy]])
    _c0, _c1 = apply_homography(_Hfull0, _center, np)[0], apply_homography(_Hfull1, _center, np)[0]
    _shift = np.array([_cx, _cy]) - (_c0 + _c1) / 2
    _T = np.array([[1, 0, _shift[0]], [0, 1, _shift[1]], [0, 0, 1]])
    _Hfull0r, _Hfull1r = _T @ _Hfull0, _T @ _Hfull1
    _after0 = warp_image_by_homography(_left, _Hfull0r, np, map_coordinates, out_shape=(_Ht, _Wt))
    _after1 = warp_image_by_homography(_right, _Hfull1r, np, map_coordinates, out_shape=(_Ht, _Wt))
    _a0 = apply_homography(_Hfull0r, _feat0, np)
    _a1 = apply_homography(_Hfull1r, _feat1, np)
    # Carry each line into the rectified frame via the homography relating
    # its OWN native (pre-crop, post-tilt) image to the rectified canvas.
    _lines0r = (np.linalg.inv(_T @ _Hrect0).T @ _lines0_pre.T).T
    _lines1r = (np.linalg.inv(_T @ _Hrect1).T @ _lines1_pre.T).T

    _colors = ["red", "limegreen"]
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for _ax, _img, _pts, _lines, _w, _h, _title in [
        (_axes[0, 0], _before0, _b0, _lines0, _cw0, _ch0, "before rectification: left"),
        (_axes[0, 1], _before1, _b1, _lines1, _cw1, _ch1, "before rectification: right"),
        (_axes[1, 0], _after0, _a0, _lines0r, _Wt, _Ht, "after rectification: left"),
        (_axes[1, 1], _after1, _a1, _lines1r, _Wt, _Ht, "after rectification: right"),
    ]:
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        for _k in range(2):
            _a, _b, _c = _lines[_k]
            _xs = np.array([0.0, _w])
            _ys = -(_a * _xs + _c) / _b
            _ax.plot(_xs, _ys, color=_colors[_k], lw=1.3, alpha=0.9)
            _ax.plot(_pts[_k, 0], _pts[_k, 1], "o", color=_colors[_k], ms=7, mec="k")
        _ax.set_xlim(0, _w)
        _ax.set_ylim(_h, 0)
        _ax.set_title(_title, fontsize=10)
        _ax.axis("off")
    _fig.tight_layout()

    _slope_before = np.concatenate([-_lines0[:, 0] / _lines0[:, 1], -_lines1[:, 0] / _lines1[:, 1]])
    _slope_after = np.concatenate([-_lines0r[:, 0] / _lines0r[:, 1], -_lines1r[:, 0] / _lines1r[:, 1]])
    mo.vstack([
        _fig,
        mo.md(
            f"Epipolar-line slope for the 2 marked features: "
            f"{np.abs(_slope_before).min():.3f}-{np.abs(_slope_before).max():.3f} "
            f"before rectification, {np.abs(_slope_after).max():.1e} after "
            f"(flat, floating-point noise); the matching feature's row now "
            f"agrees exactly between left and right "
            f"({np.abs(_a0[:,1]-_a1[:,1]).max():.1e}px) -- and rectification's "
            f"real cost is visible too: {np.mean(_after0==0)*100:.0f}% of each "
            f"rectified canvas is black, content neither original photo covered."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Disparity and depth

    Once rectified, a point's **disparity** $d=x_0-x_1$ (its horizontal pixel
    shift between the two images) relates directly to its depth $Z$ via similar
    triangles with the baseline $B$ and focal length $f$ (in pixels):

    $$ d = \frac{fB}{Z}. \qquad (\text{Eq. 12.1}) $$

    Two consequences worth internalizing before writing any matching code:
    disparity is *inversely* proportional to depth, so nearby objects produce
    large, easy-to-measure disparities while distant objects shrink toward
    $d\to0$; and because the relationship is nonlinear, a fixed pixel-level
    disparity error $\Delta d$ translates to a depth error
    $\Delta Z\approx(Z^2/fB)\,\Delta d$ that grows with the *square* of
    distance -- stereo is precise up close and rapidly imprecise far away.
    """)
    return


@app.cell
def _(mo, np, plt):
    _f, _B = 600.0, 0.1
    _Z = np.linspace(0.3, 8, 200)
    _d = _f * _B / _Z
    _dZ_per_dd = (_Z**2) / (_f * _B)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(_Z, _d)
    _axes[0].set_xlabel("depth Z (m)")
    _axes[0].set_ylabel("disparity d (px)")
    _axes[0].set_title("d = fB/Z")
    _axes[1].plot(_Z, _dZ_per_dd)
    _axes[1].set_xlabel("depth Z (m)")
    _axes[1].set_ylabel(r"$\Delta Z$ per 1px of $\Delta d$ (m)")
    _axes[1].set_title("depth error grows with Z²")
    _fig.tight_layout()
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    That error-growth curve is already the full story in one dimension, but
    it's worth seeing it as *geometry*, not just a formula: a small
    localization error on the matched feature in **each** image (at most
    $\pm$ one slider's worth of pixels, independently in each camera) turns a
    single true 3D point into an entire **uncertainty region** -- computed
    below by re-triangulating all four combinations of $\pm$ pixel error on
    $x_0$ and $x_1$ and taking the bounding box of the results, exactly Eq.
    12.1 run in reverse ($Z=fB/d$, then $X=x_0Z/f$). Try shrinking the
    baseline or pushing the depth out: the region stretches almost entirely
    *along* the depth axis, barely at all sideways -- the geometric reason
    the analytic $\Delta Z\propto Z^2/(fB)$ scaling above exists at all.
    """)
    return


@app.cell
def _(mo):
    uncertainty_B_slider = mo.ui.slider(start=0.1, stop=0.5, value=0.3, step=0.05, label="baseline B (m)", debounce=True)
    uncertainty_Z_slider = mo.ui.slider(start=0.5, stop=2.0, value=1.0, step=0.1, label="depth Z (m)", debounce=True)
    uncertainty_err_slider = mo.ui.slider(start=1, stop=5, value=1, step=1, label="pixel error (px)", debounce=True)
    mo.hstack([uncertainty_B_slider, uncertainty_Z_slider, uncertainty_err_slider], justify="start", gap=2)
    return uncertainty_B_slider, uncertainty_Z_slider, uncertainty_err_slider


@app.function
def triangulate_1d(x0, x1, f, B, np):
    """Recover (X,Z) from a pair of 1D rectified image coordinates (Eq. 12.1
    run in reverse: Z=fB/d, then back-project x0 through Z)."""
    Z = f * B / (x0 - x1)
    X = x0 * Z / f
    return X, Z


@app.cell
def _(
    mo,
    np,
    plt,
    uncertainty_B_slider,
    uncertainty_Z_slider,
    uncertainty_err_slider,
):
    _f = 600.0
    _B, _Zt, _pix_err = uncertainty_B_slider.value, uncertainty_Z_slider.value, uncertainty_err_slider.value
    _Xt = _B / 2.0
    _x0t, _x1t = _f * _Xt / _Zt, _f * (_Xt - _B) / _Zt

    # Vertex order (-1,-1) -> (-1,1) -> (1,1) -> (1,-1) traces the diamond's
    # boundary with each edge a literal segment of one camera's error-cone
    # ray (holding s0 or s1 fixed while the other sweeps from -1 to 1): the
    # true intersection-of-two-cones region, not just its bounding box.
    _order = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    _diamond = np.array([triangulate_1d(_x0t + _s0 * _pix_err, _x1t + _s1 * _pix_err, _f, _B, np) for _s0, _s1 in _order])
    _Xmin, _Xmax = _diamond[:, 0].min(), _diamond[:, 0].max()
    _Zmin, _Zmax = _diamond[:, 1].min(), _diamond[:, 1].max()
    _dX, _dZ = _Xmax - _Xmin, _Zmax - _Zmin

    _Zfar = _Zt * 1.15
    _cone0_X = [(_x0t - _pix_err) / _f * _Zfar, (_x0t + _pix_err) / _f * _Zfar]
    _cone1_X = [_B + (_x1t - _pix_err) / _f * _Zfar, _B + (_x1t + _pix_err) / _f * _Zfar]

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    _ax = _axes[0]
    _ax.plot([0, _B], [0, 0], "ks", ms=9, zorder=5)
    _ax.annotate("cam 0", (0, 0), textcoords="offset points", xytext=(-10, -12), ha="right", va="top")
    _ax.annotate("cam 1", (_B, 0), textcoords="offset points", xytext=(10, -12), ha="left", va="top")
    for _x_end in _cone0_X:
        _ax.plot([0, _x_end], [0, _Zfar], color="tab:orange", lw=0.9, ls=":")
    for _x_end in _cone1_X:
        _ax.plot([_B, _x_end], [0, _Zfar], color="tab:purple", lw=0.9, ls=":")
    _ax.add_patch(plt.Polygon(_diamond, closed=True, facecolor="tab:red", alpha=0.5, edgecolor="tab:red", lw=1.0))
    _xlim = _ax.get_xlim()
    _ax.set_xlim(_xlim[0] - 0.08 * _B, _xlim[1] + 0.08 * _B)
    _ax.set_xlabel("X (m)")
    _ax.set_ylabel("Z (m, depth from baseline)")
    _ax.set_ylim(-0.3, _Zfar)
    _ax.set_aspect("equal")
    _ax.set_title("cone of error from each camera, intersecting in a diamond")

    _ax2 = _axes[1]
    _ax2.add_patch(plt.Polygon(_diamond, closed=True, facecolor="tab:red", alpha=0.5, edgecolor="tab:red", lw=1.5))
    _ax2.plot(_Xt, _Zt, "o", color="tab:blue", ms=9, zorder=5)
    _m = max(_dX, _dZ, 1e-3) * 0.7
    _ax2.set_xlim(_Xt - _m, _Xt + _m)
    _ax2.set_ylim(_Zt - _m, _Zt + _m)
    _ax2.set_xlabel("X (m)")
    _ax2.set_ylabel("Z (m)")
    _ax2.set_title(f"uncertainty region (zoom): {_dX*1000:.1f} mm wide x {_dZ*1000:.1f} mm deep")
    _ax2.set_aspect("equal")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Each camera's $\\pm${_pix_err:.0f}px localization error sweeps out a thin "
            f"**cone** of possible rays (orange for cam 0, purple for cam 1); the true "
            f"point can only lie where *both* cones overlap, which is exactly the "
            f"**diamond** shaded above (its edges are literal segments of those four "
            f"rays). At $B$={_B:.2f}m, $Z$={_Zt:.1f}m: depth extent ({_dZ*1000:.1f}mm) is "
            f"about {_dZ/max(_dX,1e-9):.0f}x the lateral extent ({_dX*1000:.1f}mm) -- a "
            f"longer baseline or closer depth shrinks and re-rounds the diamond; a "
            f"shorter baseline or farther depth stretches it further along $Z$, matching "
            f"the $\\Delta Z\\propto Z^2/(fB)$ scaling above."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Dense correspondence: similarity measures

    Matching a small window around each pixel against candidate windows along
    the opposite scanline needs a cost function. Three standard choices
    (§12.3.1; SSD is the same cost notebook 18 used for optical flow, Eq. 9.1):

    $$ \text{SSD} = \sum_{(i,j)\in W}\big[I_0(x{+}i,y{+}j)-I_1(x{+}i{-}d,y{+}j)\big]^2, \qquad \text{SAD} = \sum_{(i,j)\in W}\big|I_0-I_1\big| $$

    $$ \text{NCC} = \frac{\sum_{(i,j)\in W}\big(I_0-\bar I_0\big)\big(I_1-\bar I_1\big)}{\sqrt{\sum(I_0-\bar I_0)^2\sum(I_1-\bar I_1)^2}} \qquad (\text{Eq. 9.11}) $$

    SAD is cheaper than SSD (no squaring) but similarly sensitive to
    brightness differences between the two cameras; NCC normalizes out
    per-window brightness/contrast (bias and gain), at some extra cost --
    the exact same bias/gain-robustness argument notebook 14's MOPS descriptor
    made for sparse matching, now applied densely.
    """)
    return


@app.function
def shift_cols(img, d, np):
    """shifted(x,y) = img(x-d,y), edge-replicated at the boundary -- the
    convention matching d = x_left - x_right (Eq. 12.1)."""
    if d == 0:
        return img.copy()
    H, W = img.shape
    padded = np.pad(img, ((0, 0), (d, 0)), mode="edge")
    return padded[:, :W]


@app.function
def cost_volume(left, right, max_disp, win_size, metric, uniform_filter, np):
    """Stack of per-disparity matching-cost maps, vectorized via the same
    shift-and-box-filter trick used for non-local means in notebook 19:
    for each candidate disparity, one box filter gives every pixel's
    windowed cost at once. Lower is always a better match (NCC is negated)."""
    H, W = left.shape
    cv = np.zeros((max_disp + 1, H, W))
    mean_l = uniform_filter(left, size=win_size, mode="reflect")
    var_l = uniform_filter(left**2, size=win_size, mode="reflect") - mean_l**2
    for d in range(max_disp + 1):
        r = shift_cols(right, d, np)
        if metric == "SSD":
            cv[d] = uniform_filter((left - r) ** 2, size=win_size, mode="reflect")
        elif metric == "SAD":
            cv[d] = uniform_filter(np.abs(left - r), size=win_size, mode="reflect")
        else:  # NCC
            mean_r = uniform_filter(r, size=win_size, mode="reflect")
            var_r = uniform_filter(r**2, size=win_size, mode="reflect") - mean_r**2
            cov = uniform_filter(left * r, size=win_size, mode="reflect") - mean_l * mean_r
            cv[d] = -cov / np.sqrt(np.clip(var_l * var_r, 1e-6, None))
    return cv


@app.function
def subpixel_refine(disp, cv, np):
    """Parabola fit through the cost volume around the discrete minimum, for
    sub-pixel disparity precision (§12.4.1, Eq. 12.6)."""
    D, H, W = cv.shape
    d = disp.astype(int)
    d_c = np.clip(d, 1, D - 2)
    yy, xx = np.mgrid[0:H, 0:W]
    c0, c1, c2 = cv[d_c - 1, yy, xx], cv[d_c, yy, xx], cv[d_c + 1, yy, xx]
    denom = c0 - 2 * c1 + c2
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    delta = np.clip(0.5 * (c0 - c2) / denom_safe, -1, 1)
    refined = d_c + delta
    return np.where((d == 0) | (d == D - 1), d.astype(np.float64), refined)


@app.function
def load_middlebury_pair(name, external_figures_dir, plt, np):
    """Load a classic Middlebury 2001 rectified pair (Scharstein and Szeliski
    2002) with its ground-truth disparity, already de-scaled to true pixel
    disparity (see images/SOURCES.md for the per-dataset scale factors)."""
    left = to_luma(plt.imread(str(external_figures_dir / f"middlebury_{name}_left.png"))[:, :, :3] * 255.0)
    right = to_luma(plt.imread(str(external_figures_dir / f"middlebury_{name}_right.png"))[:, :, :3] * 255.0)
    gt = np.load(str(external_figures_dir / f"middlebury_{name}_gt.npy")).astype(np.float64)
    return left, right, gt


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Local block matching

    The classic algorithm (§12.4) is a per-pixel **winner-take-all** 1D
    search: for each pixel, try every candidate disparity in
    $[0,d_{max}]$ along its scanline and keep whichever minimizes the cost
    above. This is the *same* windowed local-correspondence idea as notebook
    18's Lucas-Kanade optical flow -- but restricted to a **1D, discrete**
    search instead of LK's 2D, continuous one, precisely because rectification
    (§2) already pins down the one direction worth searching.

    Pick a scene below -- three real, classic Middlebury pairs (Tsukuba, Map,
    and Venus, Scharstein and Szeliski 2002), each with their own measured
    ground-truth disparity map -- and a window size.
    """)
    return


@app.cell
def _(mo):
    stereo_dataset_dropdown = mo.ui.dropdown(
        options=["Tsukuba", "Map", "Venus"],
        value="Tsukuba", label="scene",
    )
    stereo_win_slider = mo.ui.slider(start=3, stop=15, value=9, step=2, label="window size", debounce=True)
    stereo_metric_dropdown = mo.ui.dropdown(options=["SSD", "SAD", "NCC"], value="SSD", label="cost")
    mo.vstack([stereo_dataset_dropdown, mo.hstack([stereo_win_slider, stereo_metric_dropdown], justify="start", gap=2)])
    return stereo_dataset_dropdown, stereo_metric_dropdown, stereo_win_slider


@app.cell
def _(EXTERNAL_FIGURES_DIR, np, plt, stereo_dataset_dropdown):
    _dataset_params = {
        "Tsukuba": dict(max_disp=16, smoothness=30.0),
        "Map": dict(max_disp=32, smoothness=60.0),
        "Venus": dict(max_disp=24, smoothness=45.0),
    }
    _name = stereo_dataset_dropdown.value
    _left, _right, _gt = load_middlebury_pair(_name.lower(), EXTERNAL_FIGURES_DIR, plt, np)
    stereo_left, stereo_right, stereo_gt = _left, _right, _gt
    stereo_max_disp = _dataset_params[_name]["max_disp"]
    stereo_smoothness = _dataset_params[_name]["smoothness"]
    return stereo_gt, stereo_left, stereo_max_disp, stereo_right


@app.cell
def _(
    mo,
    np,
    plt,
    stereo_dataset_dropdown,
    stereo_gt,
    stereo_left,
    stereo_max_disp,
    stereo_metric_dropdown,
    stereo_right,
    stereo_win_slider,
    uniform_filter,
):
    _cv = cost_volume(stereo_left, stereo_right, stereo_max_disp, stereo_win_slider.value, stereo_metric_dropdown.value, uniform_filter, np)
    _disp = np.argmin(_cv, axis=0).astype(np.float64)
    _disp_sub = subpixel_refine(_disp, _cv, np)
    _err = np.abs(_disp - stereo_gt)
    _err_sub = np.abs(_disp_sub - stereo_gt)

    _fig, _axes = plt.subplots(1, 4, figsize=(15, 4.2))
    _axes[0].imshow(stereo_left, cmap="gray")
    _axes[0].set_title("left")
    _im1 = _axes[1].imshow(stereo_gt, cmap="viridis", vmin=0, vmax=stereo_max_disp)
    _axes[1].set_title("ground truth")
    _axes[2].imshow(_disp_sub, cmap="viridis", vmin=0, vmax=stereo_max_disp)
    _axes[2].set_title(f"block match ({stereo_metric_dropdown.value})\nmean |err| = {_err_sub.mean():.2f}px")
    _im3 = _axes[3].imshow(_err, cmap="inferno", vmin=0, vmax=max(stereo_max_disp / 4, 2))
    _axes[3].set_title("|error| (integer disparity)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.colorbar(_im1, ax=_axes[1], fraction=0.046)
    _fig.colorbar(_im3, ax=_axes[3], fraction=0.046)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"**{stereo_dataset_dropdown.value}**: integer-disparity mean error "
            f"{_err.mean():.3f}px, sub-pixel refinement (Eq. 12.6) brings it to "
            f"{_err_sub.mean():.3f}px. The error map tends to light up at depth "
            f"discontinuities and low-texture regions -- exactly where a single "
            f"independent per-pixel decision has the least evidence to work with."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Why local matching fails, and global optimization

    Winner-take-all treats every pixel's disparity as an *independent*
    decision. In a textureless or repetitive region, many candidate
    disparities fit the local window almost equally well, so the arg-min is
    effectively noise -- scattered, isolated wrong answers with no relationship
    to their neighbors, visible above as speckle in the error map.

    The fix (§12.5) is to stop treating pixels independently: minimize one
    **global energy** over the *entire* scanline (or image) at once,

    $$ E(d) = \underbrace{\sum_x C(x,d_x)}_{\text{data term}} + \underbrace{\sum_x V(d_x,d_{x+1})}_{\text{smoothness term}}, \qquad (\text{Eqs. 12.7-12.10}) $$

    the exact same "data term + smoothness term" energy shape as the level-set
    evolution in notebook 16 and TV/ROF denoising in notebook 19 -- but solved
    with a genuinely different tool this time. Those were *continuous* fields
    minimized by gradient descent; disparity here is a small set of *discrete*
    labels per pixel, so the natural algorithm is **dynamic programming**
    (§12.5.1): for each scanline independently, find the disparity sequence
    minimizing total cost in one exact pass, in roughly the time a naive search
    over every possible sequence would need to try just one candidate.
    The real practical version used today, **semi-global matching** (SGM;
    Hirschmuller 2008), aggregates this same 1D idea along *several* directions
    through the image at once -- cited here, not implemented.
    """)
    return


@app.function
def dp_stereo_scanline(cv, smoothness, np):
    """Per-scanline dynamic programming with a Potts smoothness prior: along
    each row independently, the optimal disparity at column x is either the
    same as its predecessor (cost carried over unchanged) or a jump to
    whichever disparity was cheapest so far plus one fixed penalty -- letting
    every column be resolved in O(D) instead of O(D^2)."""
    D, H, W = cv.shape
    disp_out = np.zeros((H, W))
    for y in range(H):
        cost = cv[:, y, :]
        dp = np.zeros((D, W))
        back = np.zeros((D, W), dtype=np.int32)
        dp[:, 0] = cost[:, 0]
        for x in range(1, W):
            prev = dp[:, x - 1]
            best_prev, best_prev_idx = prev.min(), prev.argmin()
            switch_cost = best_prev + smoothness
            choose_switch = switch_cost < prev
            dp[:, x] = cost[:, x] + np.where(choose_switch, switch_cost, prev)
            back[:, x] = np.where(choose_switch, best_prev_idx, np.arange(D))
        d = int(dp[:, -1].argmin())
        path = np.zeros(W, dtype=np.int32)
        path[-1] = d
        for x in range(W - 1, 0, -1):
            d = back[d, x]
            path[x - 1] = d
        disp_out[y] = path
    return disp_out


@app.cell
def _(mo):
    mo.md(r"""
    Comparing block matching against this scanline-DP refinement, across all
    three scenes at once (same window size and cost as §5's defaults, SSD):
    """)
    return


@app.cell
def _(EXTERNAL_FIGURES_DIR, mo, np, plt, uniform_filter):
    _rows = []
    _params = [
        ("Tsukuba", "tsukuba", 16, 9, 30.0),
        ("Map", "map", 32, 9, 60.0),
        ("Venus", "venus", 24, 9, 45.0),
    ]
    for _label, _key, _max_disp, _win, _smooth in _params:
        _left, _right, _gt = load_middlebury_pair(_key, EXTERNAL_FIGURES_DIR, plt, np)
        _cv = cost_volume(_left, _right, _max_disp, _win, "SSD", uniform_filter, np)
        _disp_bm = np.argmin(_cv, axis=0).astype(np.float64)
        _disp_dp = dp_stereo_scanline(_cv, _smooth, np)
        _err_bm = np.abs(_disp_bm - _gt).mean()
        _err_dp = np.abs(_disp_dp - _gt).mean()
        _rows.append(f"| {_label} | {_err_bm:.3f} | {_err_dp:.3f} |")

    mo.md(f"""
    | scene | block match mean \\|err\\| | scanline DP mean \\|err\\| |
    |---|---|---|
    {chr(10).join(_rows)}

    DP's smoothness prior helps most on Tsukuba and Venus (textured scenes
    with real low-contrast/ambiguous patches it can smooth over); on Map it's
    roughly a wash (Map's disparity genuinely varies a lot pixel-to-pixel, so
    the smoothness assumption fights the real signal as much as it fixes
    noise) -- a direct, honest illustration that global regularization is a
    real but not unconditional improvement, not a free lunch.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. From disparity to a dense depth map

    Applying Eq. 12.1 to a disparity map gives a dense depth estimate at every
    pixel -- the "2.5D" result in informal usage: a height field tied to one
    camera's viewpoint, not a full 3D model (there's no information here about
    what's behind the visible surface). Pick any scene below (ground-truth
    disparity rendered as a textured height field, $5\times$ vertically
    exaggerated so the relief is easy to read) -- for the three real
    Middlebury scenes, brighter/nearer regions clearly separate from the
    farther background.
    """)
    return


@app.cell
def _(mo):
    depth_dataset_dropdown = mo.ui.dropdown(
        options=["Tsukuba", "Map", "Venus"],
        value="Tsukuba", label="scene",
    )
    depth_dataset_dropdown
    return (depth_dataset_dropdown,)


@app.cell
def _(EXTERNAL_FIGURES_DIR, depth_dataset_dropdown, go, np, plt):
    _name = depth_dataset_dropdown.value
    _left, _right, _gt = load_middlebury_pair(_name.lower(), EXTERNAL_FIGURES_DIR, plt, np)

    _exaggeration = 5.0
    _step = 3
    _zz = _gt[::_step, ::_step] * _exaggeration
    _cc = _left[::_step, ::_step]

    _fig = go.Figure(data=[go.Surface(z=_zz, surfacecolor=_cc, colorscale="gray", showscale=False)])
    _fig.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_title=f"disparity x{_exaggeration:.0f} (inverse depth)")
    _fig.update_layout(height=520, margin=dict(l=0, r=0, t=20, b=0))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Beyond today

    - **Deep stereo networks** (§12.6) -- learned feature extraction and cost
      aggregation now dominate the Middlebury/KITTI leaderboards, the same
      shift to learning this course has flagged in every classical topic so
      far (denoising, optical flow).
    - **Multi-view stereo** (§12.7) -- combining more than two calibrated
      views (plus notebook 21's triangulation) into a full volumetric or mesh
      3D reconstruction, rather than one viewpoint's depth map.
    - **Monocular depth estimation** (§12.8) -- inferring depth from a single
      image using learned priors, with no second view or baseline at all.
    - **Occlusion-aware and segmentation-based methods** (§12.5.2) -- real
      scenes have points visible in only one camera; the simple DP model here
      doesn't explicitly label occlusions.
    - **Benchmarks**: Middlebury (now with much larger, higher-resolution
      scenes than the 2001 set used here) and KITTI (Geiger, Lenz, and
      Urtasun 2012) remain the standard evaluation suites for new algorithms.

    ## Summary

    - **Rectification** (Eq. 12.2) warps both images so epipolar lines become
      identical horizontal scanlines, reducing notebook 21's general 2D
      correspondence search to a 1D one.
    - **Disparity and depth** are related by $d=fB/Z$ (Eq. 12.1) -- nearby
      objects are measured precisely, distant ones imprecisely, since depth
      error grows with $Z^2$.
    - **Local block matching** (§12.4) is winner-take-all per pixel, using a
      similarity measure (SSD/SAD/NCC) over a small window -- the same local
      windowed-correspondence idea as notebook 18's optical flow, restricted
      to a 1D discrete search.
    - Treating pixels independently fails in low-texture/repetitive regions;
      **scanline dynamic programming** (§12.5.1) fixes this by solving one
      global energy (the same data+smoothness shape as notebooks 16 and 19)
      exactly per row, with real (not universal) measured gains on Tsukuba
      and Venus.
    - The resulting dense disparity map converts directly to a depth map via
      Eq. 12.1 -- a per-viewpoint "2.5D" surface, validated here on three
      classic Middlebury pairs (Scharstein and Szeliski 2002) with their own
      measured ground truth.
    """)
    return


if __name__ == "__main__":
    app.run()
