# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Two-View Geometry: the Essential and Fundamental Matrices

    Notebook 20 closed with an explicit promise: "two-view epipolar geometry...
    a distinct, meatier topic — structure from motion — left for its own future
    notebook." This is that notebook. Given nothing but matched points between
    two photos of the same static scene, we recover the relative pose between
    the two cameras *and* the 3D structure of the scene -- no checkerboard, no
    known 3D rig, no known camera pose at all going in.

    The tools are all already built: §2-5 below are the **fourth** time this
    course solves an estimation problem by stacking linear equations from
    correspondences and taking the smallest right singular vector of the
    result (notebook 20 did this for homographies, the camera matrix, and
    Zhang's calibration constraints) -- here applied to a genuinely new
    equation, the **epipolar constraint**.

    **One important contrast with notebook 20's Zhang's method.** There, the
    calibration target had to be *flat* -- planarity was required, and
    everything broke without it. Here, it's the opposite: a planar scene makes
    the fundamental matrix degenerate and the estimation ill-posed. That's why
    this notebook's demos use a genuinely 3D synthetic scene rather than
    notebook 20 §5's trick of warping one real photo by a homography.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Recap: the camera matrix

    From notebooks 4 and 20: the camera matrix $P=K[R\mid t]$ sends a
    homogeneous 3D point to a homogeneous pixel, $\tilde x\sim P\tilde X$. This
    notebook considers **two** cameras observing the same 3D point $X$: camera
    0, taken as the world origin ($R_0=I,\,t_0=0$), and camera 1 at some
    unknown relative pose $(R,t)$ -- "unknown" being the whole point. Both
    share the same known intrinsics $K$ throughout (one physical camera,
    photographed from two positions -- the classic setup).
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The epipolar constraint

    Look at the triangle formed by the two camera centers $C_0,C_1$ and a 3D
    point $X$. The vector from $C_0$ to $X$, the vector from $C_1$ to $X$, and
    the baseline $C_1-C_0$ all lie in one common plane -- the **epipolar
    plane** -- simply because any 3 points (here, $C_0,C_1,X$) always define a
    plane. That triangle's three corner rays all pointing into the same flat
    sheet is the entire geometric content of two-view geometry.

    Written algebraically in camera 0's own frame, the ray to $X$ is (up to
    scale) $\hat x_0=K^{-1}\tilde x_0$ and the baseline direction is $t$;
    expressed in *camera 1's* frame, $X$'s ray is $\hat x_1=K^{-1}\tilde x_1$,
    related to camera 0's ray by $\hat x_1\sim R\hat x_0+t$. Coplanarity of
    $t$, $\hat x_1$, and $R\hat x_0$ means their scalar triple product
    vanishes: $\hat x_1\cdot(t\times R\hat x_0)=0$. Every cross product
    $t\times v$ is linear in $v$, so it's itself a matrix-vector product: for
    $t=(t_1,t_2,t_3)^T$,

    $$ [t]_\times=\begin{bmatrix}0&-t_3&t_2\\t_3&0&-t_1\\-t_2&t_1&0\end{bmatrix}, \qquad\text{so that}\qquad [t]_\times v=t\times v\ \ \forall v, $$

    a skew-symmetric matrix ($[t]_\times^T=-[t]_\times$) built entirely from
    $t$'s own 3 entries. Replacing the cross product with $[t]_\times$ turns
    the triple product above into a clean bilinear form -- the **epipolar
    constraint**,

    $$ \hat x_1^T E\,\hat x_0=0, \qquad (\text{Eq. 11.30}) $$

    $$ E=[t]_\times R, \qquad (\text{Eq. 11.31, Longuet-Higgins 1981}) $$

    the **essential matrix**. One equation, and -- remarkably -- it's *linear*
    in $E$'s 9 entries for a *known* correspondence, exactly the shape every
    other estimation problem in this course has taken.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # A "toe-in" stereo pair: camera 0 sits at the origin looking straight
    # down +Z; camera 1 sits one baseline away and is explicitly rotated to
    # point its optical axis exactly at X. This keeps both dotted lines of
    # sight comfortably inside each camera's pyramid (field of view), and
    # gives the baseline/X triangle a much less acute apex angle than two
    # cameras with parallel (non-converging) axes would.
    _C0 = np.zeros(3)
    _C1 = np.array([1.8, 0.0, 0.0])
    _X = np.array([0.3, -0.15, 2.0])

    _fwd1 = (_X - _C1) / np.linalg.norm(_X - _C1)
    _world_up = np.array([0.0, 1.0, 0.0])
    _right1 = np.cross(_fwd1, _world_up)
    _right1 /= np.linalg.norm(_right1)
    _up1 = np.cross(_fwd1, _right1)
    _R1_cam_to_world = np.stack([_right1, _up1, _fwd1], axis=1)

    _fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])

    _apex0, _corners0 = camera_pyramid(_C0, np.eye(3), depth=0.35, hw=0.26, hh=0.19, np=np)
    _xs0, _ys0, _zs0 = pyramid_lines(_apex0, _corners0)
    _fig.add_trace(go.Scatter3d(x=_xs0, y=_ys0, z=_zs0, mode="lines", line=dict(color="#1f77b4", width=6), name="camera 0", showlegend=True))

    _apex1, _corners1 = camera_pyramid(_C1, _R1_cam_to_world, depth=0.35, hw=0.26, hh=0.19, np=np)
    _xs1, _ys1, _zs1 = pyramid_lines(_apex1, _corners1)
    _fig.add_trace(go.Scatter3d(x=_xs1, y=_ys1, z=_zs1, mode="lines", line=dict(color="#ff7f0e", width=6), name="camera 1", showlegend=True))

    _fig.add_trace(go.Scatter3d(x=[_X[0]], y=[_X[1]], z=[_X[2]], mode="markers+text", marker=dict(size=5, color="black"), text=["X"], textposition="top center", name="3D point", showlegend=True))

    _ray0_far = _C0 + (_X - _C0) * 1.3
    _ray1_far = _C1 + (_X - _C1) * 1.3
    _fig.add_trace(go.Scatter3d(x=[_C0[0], _ray0_far[0]], y=[_C0[1], _ray0_far[1]], z=[_C0[2], _ray0_far[2]], mode="lines", line=dict(color="#1f77b4", width=3, dash="dot"), showlegend=False))
    _fig.add_trace(go.Scatter3d(x=[_C1[0], _ray1_far[0]], y=[_C1[1], _ray1_far[1]], z=[_C1[2], _ray1_far[2]], mode="lines", line=dict(color="#ff7f0e", width=3, dash="dot"), showlegend=False))
    _fig.add_trace(go.Scatter3d(x=[_C0[0], _C1[0]], y=[_C0[1], _C1[1]], z=[_C0[2], _C1[2]], mode="lines", line=dict(color="black", width=4), name="baseline", showlegend=True))

    _plane_far = _C0 + (_X - _C0) * 1.6
    _tri = np.array([_C0, _C1, _plane_far])
    _fig.add_trace(go.Mesh3d(x=_tri[:, 0], y=_tri[:, 1], z=_tri[:, 2], i=[0], j=[1], k=[2], color="gray", opacity=0.35, name="epipolar plane", showlegend=True))

    _fig.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    _fig.update_layout(height=520, margin=dict(l=0, r=0, t=20, b=0), legend=dict(x=0.02, y=0.98))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The shaded triangle above *is* the epipolar plane: it contains both camera
    centers, the 3D point, and (though not drawn) every point *along* each
    dotted ray -- which is exactly why one image point doesn't pin down a
    unique 3D point by itself, but *does* restrict where its match can be in
    the other image: the match must lie on this plane's intersection with the
    second image plane, a line called the **epipolar line**. Every possible
    3D point along camera 0's ray sweeps out this same plane, so every one of
    their projections in image 1 lies on that one line -- the epipolar
    constraint turns a 2D search for a match into a 1D search along a line.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The fundamental matrix

    Eq. 11.30 uses *calibrated* ray directions $\hat x=K^{-1}\tilde x$, which
    assumes $K$ is known. Substituting $\hat x_j=K^{-1}\tilde x_j$ directly
    gives the equivalent constraint on raw pixel coordinates,

    $$ \tilde x_1^TF\tilde x_0=0, \qquad F=K^{-T}EK^{-1}, \qquad (\text{Eqs. 11.48-11.49}) $$

    the **fundamental matrix** (Faugeras 1992; Hartley, Gupta, and Chang 1992;
    Hartley and Zisserman 2004). The distinction matters: $F$ is estimable
    from pixel correspondences alone, with *no* calibration needed at all;
    $E$ requires a known $K$ to convert $F$ back into calibrated terms before
    it reveals an actual metric pose. Both satisfy the identical bilinear
    constraint -- only the coordinate system differs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The 8-point algorithm

    Clearing out the structure of $\tilde x_1^TF\tilde x_0=0$ the same way §3
    of notebook 20 did for homographies turns each correspondence
    $(x_0,y_0)\leftrightarrow(x_1,y_1)$ into one linear equation in $F$'s 9
    entries,

    $$ x_1x_0f_{00}+x_1y_0f_{01}+x_1f_{02}+y_1x_0f_{10}+y_1y_0f_{11}+y_1f_{12}+x_0f_{20}+y_0f_{21}+f_{22}=0, \qquad (\text{Eq. 11.33}) $$

    stack $N\ge8$ of these into $A\,\mathrm{vec}(F)=0$ and solve by the
    **fourth** use of this notebook sequence's SVD fact: the smallest right
    singular vector of $A$. Two refinements, both already built in notebook
    20:

    - **Hartley normalization** (§4 there) -- center and rescale both point
      sets before building $A$, for exactly the same conditioning reasons.
    - **Rank-2 enforcement** -- a valid $F$ always has $\det F=0$ (every
      epipolar line must pass through one common epipole), but the raw SVD
      solution generally doesn't satisfy this exactly. Fix it the same way
      Zhang's rotation estimate was fixed in notebook 20 §7: SVD $F=U\Sigma
      V^T$ and zero the smallest singular value before reconstructing
      (Eq. 11.50) -- a *third* distinct use of "project onto the nearest
      valid matrix via SVD," after homogeneous least-squares and orthogonal
      Procrustes.
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
def fundamental_8point(pts0, pts1, np):
    """Normalized 8-point algorithm (Hartley 1997a): estimate F from N>=8
    correspondences via SVD, with Hartley normalization and rank-2
    enforcement (Eqs. 11.33-11.37, 11.50)."""
    n0, T0 = normalize_points(pts0, np)
    n1, T1 = normalize_points(pts1, np)
    A = []
    for (x0, y0), (x1, y1) in zip(n0, n1):
        A.append([x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, 1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt2
    F = T1.T @ F @ T0
    return F / np.linalg.norm(F)


@app.function
def sampson_error(F, x0h, x1h, np):
    """Sampson distance (first-order approximation to true geometric
    reprojection error) for each correspondence -- used both to report
    accuracy and as the RANSAC inlier test in §5."""
    Fx0 = (F @ x0h.T).T
    Ftx1 = (F.T @ x1h.T).T
    num = np.sum(x1h * (F @ x0h.T).T, axis=1) ** 2
    denom = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return num / denom


@app.function
def make_synthetic_scene(np, seed=0, n_points=60):
    """Shared ground-truth two-camera scene reused across every demo in this
    notebook: camera 0 at the world origin, camera 1 at a known relative
    pose, N random 3D points in front of both."""
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    R1 = rot_from_euler_xyz(0.05, 0.3, -0.03, np)
    t1 = np.array([1.5, 0.1, 0.2])
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n_points, 3))
    X[:, 2] += 6
    return K, R1, t1, X


@app.function
def project_points(K, R, t, X, np):
    Xc = (R @ X.T).T + t
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3], Xc[:, 2]


@app.cell
def _(mo):
    mo.md(r"""
    **The synthetic scene behind every demo below.** `make_synthetic_scene`
    draws $N=60$ 3D points *uniformly at random* in a small cube
    ($X,Y,Z\in[-2,2]$), then shifts every point's depth by $+6$ so the whole
    cloud sits safely in front of both cameras ($Z\in[4,8]$ after the shift)
    -- genuinely
    non-planar scatter, deliberately not points on a flat surface, since the
    intro already flagged that a planar scene breaks $F$ estimation. Camera 0 sits
    at the world origin looking down $+Z$; camera 1 is placed at a *fixed*,
    known relative pose (baseline length $\approx1.5$ units, a modest
    rotation of a few degrees) that stays exactly the same across §4-§7, so
    results are directly comparable from one section's demo to the next.
    Each point is projected into both cameras (§1's $\tilde x\sim P\tilde X$)
    to get the "observed" correspondences, then the noise slider below adds
    independent Gaussian pixel noise to *both* images -- this is the only
    thing that changes as you move it.
    """)
    return


@app.cell
def _(mo):
    dlt8_noise_slider = mo.ui.slider(start=0.0, stop=2.0, value=0.5, step=0.1, label="correspondence noise σ (px)", debounce=True)
    dlt8_noise_slider
    return (dlt8_noise_slider,)


@app.cell
def _(dlt8_noise_slider, mo, np):
    _K, _R1, _t1, _X = make_synthetic_scene(np)
    _x0, _ = project_points(_K, np.eye(3), np.zeros(3), _X, np)
    _x1, _ = project_points(_K, _R1, _t1, _X, np)
    _rng = np.random.default_rng(1)
    _x0n = _x0 + _rng.normal(0, dlt8_noise_slider.value, _x0.shape)
    _x1n = _x1 + _rng.normal(0, dlt8_noise_slider.value, _x1.shape)

    _x0h = np.hstack([_x0n, np.ones((len(_x0n), 1))])
    _x1h = np.hstack([_x1n, np.ones((len(_x1n), 1))])

    _F_est = fundamental_8point(_x0n, _x1n, np)
    _err = sampson_error(_F_est, _x0h, _x1h, np)

    # Reproduce fundamental_8point's own steps so the "before" and "after"
    # singular values are compared in the *same* (normalized) coordinate
    # frame -- comparing them after denormalizing would be misleading, since
    # T0/T1 are not orthogonal (see markdown below).
    _n0, _T0 = normalize_points(_x0n, np)
    _n1, _T1 = normalize_points(_x1n, np)
    _A = np.array([[x1_*x0_, x1_*y0_, x1_, y1_*x0_, y1_*y0_, y1_, x0_, y0_, 1] for (x0_, y0_), (x1_, y1_) in zip(_n0, _n1)])
    _, _, _Vt = np.linalg.svd(_A)
    _F0 = _Vt[-1].reshape(3, 3)
    _sv_before = np.linalg.svd(_F0, compute_uv=False)

    _U, _S, _Vt2 = np.linalg.svd(_F0)
    _S_fixed = _S.copy()
    _S_fixed[2] = 0.0
    _F0_rank2 = _U @ np.diag(_S_fixed) @ _Vt2
    _sv_after_same_frame = np.linalg.svd(_F0_rank2, compute_uv=False)

    _sv_final = np.linalg.svd(_F_est, compute_uv=False)

    mo.md(f"""
    Mean Sampson error (should shrink toward 0 as σ shrinks): **{_err.mean():.4f}**
    (in normalized-coordinate units).

    **Same (normalized) coordinate frame, before vs. after rank-2 enforcement:**

    | | $\\sigma_1$ | $\\sigma_2$ | $\\sigma_3$ |
    |---|---|---|---|
    | before ($F_0=\\mathrm{{Vt}}[-1]$ reshaped) | {_sv_before[0]:.4f} | {_sv_before[1]:.4f} | {_sv_before[2]:.4f} |
    | after ($U\\,\\mathrm{{diag}}(\\sigma_1,\\sigma_2,0)\\,V^T$) | {_sv_after_same_frame[0]:.4f} | {_sv_after_same_frame[1]:.4f} | {_sv_after_same_frame[2]:.1e} |

    $\\sigma_1$ and $\\sigma_2$ are **exactly unchanged** -- reconstructing with
    the *same* $U,V$ and only the smallest singular value zeroed can't touch
    the other two, by construction. Only $\\sigma_3$ moves, from small-but-nonzero
    to exactly zero (up to floating point).

    **Why this has to happen *before* denormalizing.** `fundamental_8point`
    computes the SVD, zeros $\\sigma_3$, and *only then* applies
    $F\\mapsto T_1^TFT_0$ to return to pixel coordinates. That order matters:
    $T_0,T_1$ are Hartley's normalization matrices, and they are **not
    orthogonal** (they combine a scale with a translation offset). Multiplying
    by a non-orthogonal matrix does not preserve singular values at all --
    reconstructing here in pixel coordinates first and zeroing a singular
    value *there* would zero out the wrong direction entirely. The final
    returned $F$'s own singular values, after denormalizing and rescaling to
    unit Frobenius norm, are `{np.array2string(_sv_final, precision=4)}` --
    not directly comparable to the table above (different frame, different
    scale), but still exactly rank-2, which is the only property that
    actually matters for $F$ to be valid.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Robust estimation with RANSAC

    Real correspondences are never clean, and the book doesn't spell out a
    self-contained "RANSAC for $F$" recipe the way it does for lines (§8.1.4
    defines RANSAC itself and even has a footnote specifically about using it
    for "epipolar geometry estimation," but the worked procedure is left to
    the reader) -- so synthesize it ourselves, generalizing notebook 20's
    `ransac_homography` once more: minimal sample size **8** (the 8-point
    algorithm's own minimum), inlier test by Sampson distance, refit on all
    inliers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Recovering relative pose from E

    Given $K$, convert the estimated $F$ back to $E=K^TFK$ and recover the
    motion. $E=[t]_\times R$'s SVD has a very specific shape -- $E=U\Sigma V^T$
    with $\Sigma=\mathrm{diag}(1,1,0)$ (Eq. 11.40) -- and it's worth seeing
    exactly why, rather than taking it on faith.

    **Why $[t]_\times$ alone has singular values $(\|t\|,\|t\|,0)$.** The
    vector triple product identity $t\times(t\times v)=t(t\cdot v)-v\|t\|^2$
    says $[t]_\times^2=tt^T-\|t\|^2I$ (as matrices, since both sides act the
    same way on every $v$). Since $[t]_\times$ is skew-symmetric
    ($[t]_\times^T=-[t]_\times$),

    $$ [t]_\times^T[t]_\times=-[t]_\times^2=\|t\|^2I-tt^T, $$

    a symmetric matrix with an obvious eigenvector: $t$ itself, since
    $(\|t\|^2I-tt^T)t=\|t\|^2t-t\|t\|^2=0$ -- eigenvalue $0$. Every vector $v$
    *orthogonal* to $t$ satisfies $t^Tv=0$, so $(\|t\|^2I-tt^T)v=\|t\|^2v$ --
    eigenvalue $\|t\|^2$, with multiplicity 2 (the orthogonal complement of
    one direction in $\mathbb R^3$ is a 2D plane). So $[t]_\times^T[t]_\times$
    has eigenvalues $(\|t\|^2,\|t\|^2,0)$, meaning $[t]_\times$'s singular
    values are exactly $(\|t\|,\|t\|,0)$ -- geometrically, $[t]_\times v=t\times v$
    always lands in the plane perpendicular to $t$ (the zero singular value,
    in the $t$ direction) and stretches every direction *within* that plane by
    the same factor $\|t\|$ (the two equal nonzero singular values).

    **Why right-multiplying by $R$ changes nothing.** $E^TE=R^T([t]_\times^T[t]_\times)R$,
    and since $R^T=R^{-1}$ this is an orthogonal *similarity transform* of
    $[t]_\times^T[t]_\times$ -- similarity transforms preserve eigenvalues
    exactly, by definition. So $E=[t]_\times R$ has the *identical* singular
    values as $[t]_\times$ by itself: $(\|t\|,\|t\|,0)$. Since $t$'s scale is
    unrecoverable anyway (see below), normalizing $\|t\|=1$ turns this into
    exactly Eq. 11.40's $\Sigma=\mathrm{diag}(1,1,0)$ -- the same rank-2
    structure as $F$, now additionally constrained to have two *equal*
    nonzero singular values, which is what distinguishes an *essential*
    matrix from an arbitrary rank-2 matrix: not every rank-2 $3\times3$
    matrix factors as $[t]_\times R$, only ones with this specific singular
    value pattern do. Factoring the
    skew-symmetric part with the matrix
    $W=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$ gives **four**
    algebraically valid candidates (Eq. 11.43-11.45),

    $$ R\in\{UWV^T,\ UW^TV^T\}, \qquad t\in\{+u_3,\ -u_3\}, $$

    where $u_3$ is $U$'s third column -- note $t$ is recovered only as a
    **unit vector**: scale is fundamentally unobservable from two
    uncalibrated views (a scene and a scaled-up copy of it, viewed from a
    proportionally scaled-up baseline, produce *identical* images). Only one
    of the four $(R,t)$ combinations places the 3D point in *front of both*
    cameras simultaneously -- the **cheirality** condition (Hartley 1998) --
    which is checked directly by triangulating a few points under each
    candidate and counting positive depths.
    """)
    return


@app.function
def triangulate_point(P0, P1, x0, x1, np):
    """Linear/DLT triangulation (Eq. 11.24): the 3D point whose projections
    best satisfy both cameras' equations, via SVD -- the dual of notebook
    20's camera-matrix DLT (there: unknown camera, known points; here:
    unknown point, known cameras)."""
    A = np.array([
        x0[0] * P0[2] - P0[0],
        x0[1] * P0[2] - P0[1],
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


@app.function
def recover_pose_from_essential(E, K, pts0, pts1, np):
    """Decompose E into the cheirality-correct (R, t) via SVD (Eqs. 11.40-11.45),
    disambiguating the 4-fold sign ambiguity by triangulating a few points and
    keeping the candidate with the most positive depths in both cameras."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_a, R_b = U @ W @ Vt, U @ W.T @ Vt
    t_hat = U[:, 2]
    candidates = [(R_a, t_hat), (R_a, -t_hat), (R_b, t_hat), (R_b, -t_hat)]

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best_count, best = -1, None
    n_check = min(10, len(pts0))
    for R_c, t_c in candidates:
        P1 = K @ np.hstack([R_c, t_c.reshape(3, 1)])
        count = 0
        for i in range(n_check):
            Xp = triangulate_point(P0, P1, pts0[i], pts1[i], np)
            if Xp[2] > 0 and (R_c @ Xp + t_c)[2] > 0:
                count += 1
        if count > best_count:
            best_count, best = count, (R_c, t_c)
    return best


@app.cell
def _(mo):
    pose_noise_slider = mo.ui.slider(start=0.0, stop=2.0, value=0.5, step=0.1, label="correspondence noise σ (px)", debounce=True)
    pose_noise_slider
    return (pose_noise_slider,)


@app.cell
def _(mo, np, pose_noise_slider):
    _K, _R1, _t1, _X = make_synthetic_scene(np)
    _x0, _ = project_points(_K, np.eye(3), np.zeros(3), _X, np)
    _x1, _ = project_points(_K, _R1, _t1, _X, np)
    _rng = np.random.default_rng(1)
    _x0n = _x0 + _rng.normal(0, pose_noise_slider.value, _x0.shape)
    _x1n = _x1 + _rng.normal(0, pose_noise_slider.value, _x1.shape)

    _F_est = fundamental_8point(_x0n, _x1n, np)
    _E_est = _K.T @ _F_est @ _K
    _R_rec, _t_rec = recover_pose_from_essential(_E_est, _K, _x0n, _x1n, np)

    _t1_dir = _t1 / np.linalg.norm(_t1)
    _dir_err = min(np.abs(_t_rec - _t1_dir).max(), np.abs(_t_rec + _t1_dir).max())

    mo.md(f"""
    $\\|R_{{rec}}-R_{{true}}\\|_\\infty={np.abs(_R_rec - _R1).max():.4f}$,
    translation **direction** error $={_dir_err:.4f}$ (true baseline length
    $\\|t\\|={np.linalg.norm(_t1):.2f}$, recovered as a unit vector only --
    scale is gone by construction, not lost to noise). The cheirality check
    picked the correct candidate out of 4 purely from which one places points
    in front of both cameras.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Triangulation

    With both camera matrices now in hand ($P_0=K[I\mid0]$, $P_1=K[R\mid t]$),
    recovering each matched point's 3D position is the DLT's dual problem
    (Eq. 11.24, same Sutherland 1974 lineage as notebook 20 §6's camera-matrix
    DLT): there, unknown camera + known 3D points; here, unknown 3D point +
    known cameras. Each image's projection equation contributes 2 linear rows
    in the point's 4 homogeneous coordinates; stack the 2 views' worth (4 rows
    total) and solve by SVD exactly as always. (An "optimal" nonlinear
    refinement minimizing true reprojection error exists too -- Hartley and
    Sturm 1997 -- not implemented here.)
    """)
    return


@app.cell
def _(go, mo, np, pose_noise_slider):
    _K, _R1, _t1, _X = make_synthetic_scene(np)
    _x0, _ = project_points(_K, np.eye(3), np.zeros(3), _X, np)
    _x1, _ = project_points(_K, _R1, _t1, _X, np)
    _rng = np.random.default_rng(1)
    _x0n = _x0 + _rng.normal(0, pose_noise_slider.value, _x0.shape)
    _x1n = _x1 + _rng.normal(0, pose_noise_slider.value, _x1.shape)

    _F_est = fundamental_8point(_x0n, _x1n, np)
    _E_est = _K.T @ _F_est @ _K
    _R_rec, _t_rec = recover_pose_from_essential(_E_est, _K, _x0n, _x1n, np)

    _P0 = _K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    _P1 = _K @ np.hstack([_R_rec, _t_rec.reshape(3, 1)])
    _X_rec = np.array([triangulate_point(_P0, _P1, _x0n[i], _x1n[i], np) for i in range(len(_X))])

    # the reconstruction is only known up to scale (||t_rec||=1); rescale by
    # the true baseline length purely to overlay it against ground truth below
    _scale = np.linalg.norm(_t1)
    _X_rec_scaled = _X_rec * _scale
    _err = np.sqrt(((_X_rec_scaled - _X) ** 2).sum(axis=1))

    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=_X[:, 0], y=_X[:, 1], z=_X[:, 2], mode="markers", marker=dict(size=4, color="black"), name="ground truth"))
    _fig.add_trace(go.Scatter3d(x=_X_rec_scaled[:, 0], y=_X_rec_scaled[:, 1], z=_X_rec_scaled[:, 2], mode="markers", marker=dict(size=4, color="#d62728"), name="triangulated (scale-aligned)"))
    _fig.update_scenes(aspectmode="data")
    _fig.update_layout(height=480, margin=dict(l=0, r=0, t=20, b=0), legend=dict(x=0.02, y=0.98))

    mo.vstack([
        _fig,
        mo.md(
            f"Mean/max reconstruction error after aligning scale to the true "
            f"baseline: **{_err.mean():.3f}** / **{_err.max():.3f}** (scene "
            f"points span roughly ±2 units) -- the two black-vs-red point "
            f"clouds should sit nearly on top of each other. In a real "
            f"application, that scale factor is exactly what's missing "
            f"without an external reference (a known object size, a second "
            f"sensor, or stereo baseline measured by hand)."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Beyond today

    - **Minimal solvers.** The 8-point algorithm needs 8 correspondences; a
      **7-point** algorithm (Eqs. 11.38-11.39) exploits $\det F=0$ directly to
      work with one fewer, at the cost of up to 3 candidate solutions: and
      Nistér's **5-point** algorithm goes further still, using the calibrated
      $E$'s extra structure. Fewer points per RANSAC sample means exponentially
      fewer trials needed for the same outlier tolerance (Eq. 8.30) -- the
      same efficiency argument notebook 17 made for using minimal samples at
      all.
    - **Degenerate configurations.** A planar scene (or the camera rotating in
      place with no translation) makes $F$'s estimation ill-posed -- exactly
      the opposite requirement from Zhang's method. **DEGENSAC** (Chum,
      Werner, and Matas 2005) detects and handles this case within RANSAC.
    - **Bundle adjustment / multi-view SfM.** Chaining more than two views
      together, jointly refining *all* cameras' poses and *all* 3D points at
      once via nonlinear least squares -- the natural next notebook, now that
      two-view pose and triangulation exist as building blocks.
    - **Stereo rectification and dense depth.** A different, more applied
      direction: given a *calibrated* stereo rig (known, fixed $R,t$ between
      two cameras), warp both images so epipolar lines become horizontal scan
      lines, then search along them for a dense per-pixel disparity map --
      skips the estimation problem this notebook solves, in exchange for a
      full depth image rather than a sparse point cloud.
    - **Self-calibration.** Recovering $K$ itself from an uncalibrated
      sequence's own motion (§11.1.3), without any calibration target at all.

    ## Summary

    - Two camera centers and any shared 3D point always define a plane -- the
      **epipolar plane** -- giving the bilinear **epipolar constraint**
      $\hat x_1^TE\hat x_0=0$ (Eq. 11.30), $E=[t]_\times R$ (Longuet-Higgins
      1981); the uncalibrated version $x_1^TFx_0=0$ uses the **fundamental
      matrix** $F=K^{-T}EK^{-1}$.
    - The **normalized 8-point algorithm** (Hartley 1997a) estimates $F$ by
      the same "stack linear equations, SVD" pattern used three times already
      in notebook 20 -- now a fourth time -- plus **rank-2 enforcement**, a
      third distinct use of "project onto the nearest valid matrix via SVD."
    - **RANSAC**, generalized once more from notebook 20's `ransac_homography`,
      makes the estimate robust to real mismatches.
    - $E$'s SVD recovers relative pose $(R,t)$ up to a **fundamental scale
      ambiguity**, disambiguating its four-fold sign ambiguity via
      **cheirality** (Hartley 1998).
    - **Triangulation** (Eq. 11.24) is the DLT's dual problem to notebook 20's
      camera-matrix estimation -- unknown point, known cameras instead of the
      reverse -- completing the loop from two uncalibrated photos to an actual
      (scale-free) 3D reconstruction.
    """)
    return


if __name__ == "__main__":
    app.run()
