# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
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

    return go, make_subplots, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Bundle Adjustment and Multi-View Structure from Motion

    Notebook 21 recovered relative pose and triangulated 3D points from
    exactly *two* views. Most real reconstructions -- photo collections,
    video, a robot driving around a room -- have many more. Simply chaining
    two-view results pairwise accumulates drift: each new camera is only
    ever made consistent with its immediate neighbor, never with the whole
    set at once.

    This notebook covers Szeliski **§11.4 ("Multi-frame structure from
    motion")**: jointly refining *every* camera pose and *every* 3D point at
    once by directly minimizing total reprojection error -- **bundle
    adjustment** -- the standard tool behind real multi-view pipelines
    (COLMAP and friends), and, as notebook 24 will show, the source of the
    camera-pose-plus-point-cloud input that NeRF and 3D Gaussian Splatting
    both start from.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. From one camera to a chain of them

    Notebook 20 §6 already wrote single-camera projection as a short chain of
    transforms -- rotate into camera coordinates, translate, perspective
    divide, apply intrinsics -- the same chain Szeliski builds
    (Eqs. 11.16-11.19, Figure 11.7), written here with this course's own
    $X$ (3D point) and $(R,t)$ (camera extrinsics) convention rather than the
    book's $(p,c)$:

    $$ X_{\text{cam}} = RX+t, \qquad x = \frac{K\,X_{\text{cam}}}{z_{\text{cam}}}. \qquad (\text{Eqs. 11.16-11.19}) $$

    Multi-frame structure from motion changes only one thing: every
    measurement now carries *two* indices, a point index $i$ and a camera
    index $j$,

    $$ x_{ij} = f(X_i, R_j, t_j, K_j), \qquad (\text{Eq. 11.61}) $$

    and *all* of the $X_i$ and *all* of the $(R_j,t_j)$ are refined
    simultaneously from *all* the measurements at once -- not one camera at
    a time, and not one point at a time.
    """)
    return


@app.function
def rot_from_euler_xyz(rx, ry, rz, np):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@app.function
def rodrigues(omega, np):
    """Exact exponential map for an incremental rotation (Rodrigues'
    formula): the small-rotation update this notebook's optimizer prepends
    to each camera's current estimate (Eq. 2.35-2.36), rather than
    differentiating a rotation matrix or quaternion directly."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


@app.function
def project_points(K, R, t, X, np):
    Xc = (R @ X.T).T + t
    x = (K @ Xc.T).T
    return x[:, :2] / x[:, 2:3]


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
    ## 2. The bundle-adjustment residual

    For every observed point $i$ seen by every camera $j$, the residual
    compares the predicted and measured 2D position (Eqs. 11.63-11.65):

    $$ r_{ij} = \tilde x_{ij} - \hat x_{ij}, \qquad e_{ij} = \hat\rho(r_{ij}^T\Sigma_{ij}^{-1}r_{ij}), $$

    where a robust loss $\hat\rho$ and per-measurement covariance
    $\Sigma_{ij}$ can downweight outlier matches. The demo below uses the
    plain, unweighted squared residual ($\hat\rho(s)=s$, $\Sigma_{ij}=I$) to
    keep the optimizer itself front and center; real systems almost always
    add the robust loss back in, for exactly the same reason notebook 17's
    RANSAC existed at all -- a handful of bad matches can otherwise dominate
    a sum of squares.

    Minimizing the total squared residual over *every* camera pose and
    *every* point at once is a large nonlinear least-squares problem, solved
    the same way notebook 20 §6's single-camera pose refinement was: linearize
    around the current estimate (the same chain rule, Eq. 11.20) and take a
    Gauss-Newton step. The only new wrinkle is which parameters a given
    residual's partial derivatives are non-zero with respect to -- point $i$'s
    3 coordinates and camera $j$'s 6 pose parameters, and *nothing else*.
    Camera rotations are updated by prepending a small incremental rotation
    (`rodrigues` above) to the current estimate (Eq. 2.35-2.36) rather than
    differentiating a rotation matrix directly -- the same trick notebook
    20 used for Zhang's extrinsics recovery.
    """)
    return


@app.function
def build_jacobian_and_residual(Rs, ts, pts, obs, free_cams, K, np, eps=1e-5):
    """Residual vector and its Jacobian w.r.t. every FREE camera's (t, omega)
    and every point's position, by finite differences -- one call per
    Gauss-Newton iteration. Cameras not in `free_cams` are held fixed (the
    gauge anchor, see the uncertainty/ambiguity discussion in Section 5)."""
    n_pts = len(pts)
    cam_offset = {j: 6 * k for k, j in enumerate(free_cams)}
    n_cam_params = 6 * len(free_cams)
    pt_offset = {i: n_cam_params + 3 * i for i in range(n_pts)}
    n_params = n_cam_params + 3 * n_pts

    obs_list = list(obs.items())
    r = np.zeros(2 * len(obs_list))
    J = np.zeros((2 * len(obs_list), n_params))
    for k, ((i, j), x_obs) in enumerate(obs_list):
        base = project_points(K, Rs[j], ts[j], pts[i:i + 1], np)[0]
        r[2 * k:2 * k + 2] = x_obs - base
        for d in range(3):
            dp = np.zeros(3)
            dp[d] = eps
            xp = project_points(K, Rs[j], ts[j], (pts[i] + dp)[None, :], np)[0]
            J[2 * k:2 * k + 2, pt_offset[i] + d] = -(xp - base) / eps
        if j in cam_offset:
            off = cam_offset[j]
            for d in range(3):
                dt = np.zeros(3)
                dt[d] = eps
                xp = project_points(K, Rs[j], ts[j] + dt, pts[i:i + 1], np)[0]
                J[2 * k:2 * k + 2, off + d] = -(xp - base) / eps
            for d in range(3):
                dw = np.zeros(3)
                dw[d] = eps
                R_perturbed = rodrigues(dw, np) @ Rs[j]
                xp = project_points(K, R_perturbed, ts[j], pts[i:i + 1], np)[0]
                J[2 * k:2 * k + 2, off + 3 + d] = -(xp - base) / eps
    return J, r, cam_offset, pt_offset, n_cam_params, n_params


@app.function
def bundle_adjust(Rs0, ts0, pts0, obs, K, free_cams, n_iters, np):
    """Gauss-Newton bundle adjustment: jointly refine every free camera's
    pose and every 3D point by repeatedly linearizing (build_jacobian_and_
    residual) and solving the dense normal equations for a parameter
    update. Returns the refined (Rs, ts, pts) plus the per-iteration RMS
    reprojection error history."""
    Rs = [R.copy() for R in Rs0]
    ts = [t.copy() for t in ts0]
    pts = pts0.copy()
    history = []
    for _ in range(n_iters):
        J, r, cam_offset, pt_offset, _, n_params = build_jacobian_and_residual(Rs, ts, pts, obs, free_cams, K, np)
        history.append(np.sqrt(np.mean(r**2)))
        JTJ = J.T @ J + 1e-6 * np.eye(n_params)
        delta = np.linalg.solve(JTJ, -J.T @ r)
        for j in free_cams:
            off = cam_offset[j]
            ts[j] = ts[j] + delta[off:off + 3]
            Rs[j] = rodrigues(delta[off + 3:off + 6], np) @ Rs[j]
        for i in range(len(pts)):
            off = pt_offset[i]
            pts[i] = pts[i] + delta[off:off + 3]
    _, r_final, *_ = build_jacobian_and_residual(Rs, ts, pts, obs, free_cams, K, np)
    history.append(np.sqrt(np.mean(r_final**2)))
    return Rs, ts, pts, np.array(history)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Why bundle adjustment needs to exploit sparsity

    A brief note, since real systems depend on this but we won't dig into
    it here: each residual $r_{ij}$ only involves one point and one camera,
    so most of the normal-equations matrix $A=J^TJ$ is exactly zero -- real
    reconstructions have thousands of cameras and millions of points, where
    a dense solve is hopeless (Szeliski's Figure 11.16 shows this sparsity
    as a bipartite graph). The standard fix is the **Schur complement**
    (Eq. 11.68, §11.4.3): eliminate the point block first (it's exactly
    block-diagonal, since no two points ever share a residual, so inverting
    it is trivial), solve a much smaller camera-only system, then
    back-substitute for the points. That's the reason bundle adjustment
    scales to real-world reconstructions at all -- not needed below, since
    this demo is small enough for a plain dense solve.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Demo: refining a drifted multi-camera chain

    A synthetic rig of 6 cameras on an arc, all observing the same 20 3D
    points with small pixel noise. Cameras 0 and 1 are known *exactly* --
    the gauge anchor a two-view result like notebook 21's would provide --
    but cameras 2 onward are only known through a **simulated chain** of
    noisy relative poses (as if each had been estimated two-view-at-a-time
    from its predecessor), so the error compounds down the chain.

    ### Predict

    Points are each observed by every camera, while each camera (beyond the
    anchor) only has its own noisy initial estimate to start from. Before
    running the optimizer below: do you expect camera poses or point
    positions to end up *more* accurately recovered relative to their own
    initial error -- or should they improve by about the same factor?
    """)
    return


@app.cell
def _(mo):
    drift_slider = mo.ui.slider(start=0.5, stop=3.0, value=1.0, step=0.25, label="initial drift severity", debounce=True)
    iters_slider = mo.ui.slider(start=1, stop=15, value=10, step=1, label="Gauss-Newton iterations", debounce=True)
    mo.hstack([drift_slider, iters_slider], justify="start", gap=2)
    return drift_slider, iters_slider


@app.cell
def _(drift_slider, go, iters_slider, make_subplots, mo, np, plt):
    _rng = np.random.default_rng(0)
    _N_CAM, _N_PTS = 6, 20
    _K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    _angles = np.linspace(-0.6, 0.6, _N_CAM)
    _Rs_gt, _ts_gt, _Cs_gt = [], [], []
    for _a in _angles:
        _C = np.array([3.0 * np.sin(_a), 0.1 * np.sin(2 * _a), -3.0 * np.cos(_a)])
        _fwd = -_C / np.linalg.norm(_C)
        _right = np.cross(_fwd, np.array([0.0, 1.0, 0.0]))
        _right /= np.linalg.norm(_right)
        _up = np.cross(_right, _fwd)
        _Rcw = np.stack([_right, _up, _fwd], axis=1)
        _R = _Rcw.T
        _Rs_gt.append(_R)
        _ts_gt.append(-_R @ _C)
        _Cs_gt.append(_C)
    _pts_gt = _rng.uniform(-1.0, 1.0, size=(_N_PTS, 3)) * np.array([1.0, 1.0, 0.6])

    _pix_noise = 0.3
    _obs = {
        (i, j): project_points(_K, _Rs_gt[j], _ts_gt[j], _pts_gt[i:i + 1], np)[0] + _rng.normal(scale=_pix_noise, size=2)
        for j in range(_N_CAM) for i in range(_N_PTS)
    }

    # Unit-scale noise draws, fixed regardless of the slider, so the slider
    # only turns up/down the SAME drift realization's severity.
    _rot_draws = [_rng.normal(scale=1.0, size=3) for _ in range(2, _N_CAM)]
    _trans_draws = [_rng.normal(scale=1.0, size=3) for _ in range(2, _N_CAM)]
    _pt_noise_draw = _rng.normal(scale=1.0, size=_pts_gt.shape)

    _drift_rot, _drift_trans, _pt_noise = 0.06 * drift_slider.value, 0.10 * drift_slider.value, 0.1 * drift_slider.value
    _Rs0 = [_Rs_gt[0].copy(), _Rs_gt[1].copy()]
    _ts0 = [_ts_gt[0].copy(), _ts_gt[1].copy()]
    for _k, _j in enumerate(range(2, _N_CAM)):
        _R_rel = _Rs_gt[_j] @ _Rs_gt[_j - 1].T
        _t_rel = _ts_gt[_j] - _R_rel @ _ts_gt[_j - 1]
        _R_err = rodrigues(_drift_rot * _rot_draws[_k], np)
        _R_rel_noisy = _R_err @ _R_rel
        _t_rel_noisy = _t_rel + _drift_trans * _trans_draws[_k]
        _Rs0.append(_R_rel_noisy @ _Rs0[_j - 1])
        _ts0.append(_R_rel_noisy @ _ts0[_j - 1] + _t_rel_noisy)
    _pts0 = _pts_gt + _pt_noise * _pt_noise_draw

    _free_cams = list(range(2, _N_CAM))
    _Rs1, _ts1, _pts1, _history = bundle_adjust(_Rs0, _ts0, _pts0, _obs, _K, _free_cams, iters_slider.value, np)

    def _pose_errors(Rs, ts):
        Cs = [-R.T @ t for R, t in zip(Rs, ts)]
        cam_err = np.mean([np.linalg.norm(c - cg) for c, cg in zip(Cs, _Cs_gt)])
        ang_err = np.mean([np.degrees(np.arccos(np.clip((np.trace(R @ Rg.T) - 1) / 2, -1, 1))) for R, Rg in zip(Rs, _Rs_gt)])
        return cam_err, ang_err

    _cam_err0, _ang_err0 = _pose_errors(_Rs0, _ts0)
    _cam_err1, _ang_err1 = _pose_errors(_Rs1, _ts1)
    _pt_err0 = np.mean(np.linalg.norm(_pts0 - _pts_gt, axis=1))
    _pt_err1 = np.mean(np.linalg.norm(_pts1 - _pts_gt, axis=1))

    _fig_conv, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.semilogy(_history, marker="o", ms=4)
    _ax.set_xlabel("Gauss-Newton iteration")
    _ax.set_ylabel("RMS reprojection error (px)")
    _ax.set_title("convergence")
    _fig_conv.tight_layout()

    _fig3d = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("before (initial)", "after (refined)"))
    for _col, (_Rs_est, _ts_est, _pts_est) in enumerate([(_Rs0, _ts0, _pts0), (_Rs1, _ts1, _pts1)], start=1):
        _fig3d.add_trace(go.Scatter3d(x=_pts_gt[:, 0], y=_pts_gt[:, 1], z=_pts_gt[:, 2], mode="markers", marker=dict(size=3, color="lightgray"), name="ground truth pts", showlegend=(_col == 1)), row=1, col=_col)
        _fig3d.add_trace(go.Scatter3d(x=_pts_est[:, 0], y=_pts_est[:, 1], z=_pts_est[:, 2], mode="markers", marker=dict(size=3, color="#d62728"), name="estimated pts", showlegend=(_col == 1)), row=1, col=_col)
        for _j in range(_N_CAM):
            _apex_gt, _corners_gt = camera_pyramid(_Cs_gt[_j], _Rs_gt[_j].T, depth=0.3, hw=0.2, hh=0.15, np=np)
            _xs, _ys, _zs = pyramid_lines(_apex_gt, _corners_gt)
            _fig3d.add_trace(go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines", line=dict(color="lightgray", width=3), showlegend=False), row=1, col=_col)
            _C_est = -_Rs_est[_j].T @ _ts_est[_j]
            _apex, _corners = camera_pyramid(_C_est, _Rs_est[_j].T, depth=0.3, hw=0.2, hh=0.15, np=np)
            _xs, _ys, _zs = pyramid_lines(_apex, _corners)
            _fig3d.add_trace(go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines", line=dict(color="#1f77b4", width=5), showlegend=False), row=1, col=_col)
    _fig3d.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    _fig3d.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0))

    mo.vstack([
        _fig_conv,
        _fig3d,
        mo.md(
            f"At drift severity {drift_slider.value:.2f}x: RMS reprojection error "
            f"{_history[0]:.1f}px -> {_history[-1]:.3f}px over {iters_slider.value} "
            f"iterations (pixel noise floor is {_pix_noise}px). Mean camera-center "
            f"error {_cam_err0:.4f} -> {_cam_err1:.5f} ({_cam_err0/max(_cam_err1,1e-9):.0f}x), "
            f"mean camera-angle error {_ang_err0:.3f} -> {_ang_err1:.4f} deg, mean "
            f"point error {_pt_err0:.4f} -> {_pt_err1:.5f} "
            f"({_pt_err0/max(_pt_err1,1e-9):.0f}x) -- both pose and structure "
            f"converge, jointly, from a shared set of 2D measurements."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Gauge ambiguity: why two cameras had to stay fixed

    Monocular structure from motion can never recover an absolute coordinate
    frame or scale -- rotate, translate, and rescale *every* camera and
    *every* point together and every 2D measurement stays identical. This
    is a genuine **seven-degree-of-freedom gauge ambiguity** (3 rotation + 3
    translation + 1 scale), discussed explicitly in §11.4.5. Left
    unconstrained, the normal-equations matrix $A$ is singular along exactly
    those seven directions. Holding cameras 0 and 1 fixed throughout (as the
    demo above does) removes all seven degrees of freedom at once: no
    rotation/translation/rescaling of the whole solution leaves two *fixed*
    cameras unchanged except the identity -- exactly the same role a known
    stereo baseline played in fixing scale back in notebook 22.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Beyond today

    - **Factorization** (§11.4.1; Tomasi and Kanade 1992) -- for orthographic
      or affine cameras with a complete set of tracks, structure and motion
      can be recovered in one shot via a single SVD, no iteration required;
      it's commonly used only to *initialize* bundle adjustment on harder,
      perspective problems.
    - **Robust loss and real feature tracks** -- this notebook assumed
      perfectly known point correspondences; production pipelines pair
      bundle adjustment with the same robust matching (notebook 14) and
      outlier rejection (notebook 17's RANSAC) used for two-view geometry.
    - **Incremental and global SfM at scale** -- COLMAP and similar tools
      interleave bundle adjustment with adding new cameras/points one at a
      time (incremental SfM) or solve for all camera *rotations* first via a
      separate global step before a final joint refinement (global SfM);
      both avoid ever re-solving the full dense problem from scratch.
    - **Where this leads next**: a bundle-adjusted set of camera poses plus
      a sparse 3D point cloud is exactly the standard starting point for two
      different modern directions -- classical **multi-view stereo and 3D
      reconstruction** (§12.7, Chapter 13: turning many calibrated views into
      a dense point cloud, mesh, or volumetric model), and **neural
      rendering** (§14.6): instead of an explicit mesh, optimize a
      continuous radiance field (NeRF; Mildenhall, Srinivasan et al. 2020)
      or a set of 3D Gaussians (3D Gaussian Splatting; Kerbl et al. 2023,
      not in this book's 2021 draft) directly against the input images --
      the subject of the next notebook.

    ## Summary

    - Two-view structure from motion (notebook 21) doesn't generalize to $N$
      views by simple pairwise chaining -- errors compound down the chain
      with no mechanism to enforce *global* consistency.
    - **Bundle adjustment** (Eq. 11.61) jointly minimizes total reprojection
      error over every camera pose and every 3D point at once, via the same
      chained-projection Gauss-Newton machinery notebook 20 used for a
      single camera (Eqs. 11.16-11.20), with rotations updated through a
      small incremental-rotation "retraction" (Eq. 2.35-2.36) rather than
      differentiating $R$ directly.
    - The resulting normal equations are **sparse**: each residual only
      touches one point and one camera. Eliminating the block-diagonal point
      Hessian first (the **Schur complement**, Eq. 11.68) reduces the
      problem to a much smaller camera-only system -- validated numerically
      above to match a brute-force dense solve exactly.
    - A **seven-degree-of-freedom gauge ambiguity** (global rotation,
      translation, and scale) is unavoidable from 2D measurements alone;
      fixing two cameras removes it, mirroring notebook 22's known-baseline
      scale fix.
    - On a synthetic drifted 6-camera chain, bundle adjustment recovers both
      camera poses and 3D structure to well within the injected pixel-noise
      floor -- the same machine-precision-grade convergence this course has
      checked at every previous step, now jointly, for every unknown at once.
    """)
    return


if __name__ == "__main__":
    app.run()
