# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return MplPolygon, Poly3DCollection, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Camera Calibration: Intrinsics, Extrinsics, and P = K[R|t]

    Notebook 3 projected a cube with a deliberately simplified pinhole
    model — camera fixed at the world origin, looking down $+z$, with
    $P_{\text{persp}} = [f, 0, 0, 0;\ 0, f, 0, 0;\ 0, 0, 1, 0]$. That model
    hid two things a real camera can't avoid: the camera can sit *anywhere*
    in the world at *any* orientation, and even after the perspective
    divide, normalized coordinates still need to be mapped onto actual
    sensor pixels.

    This notebook fills in both pieces:

    1. Normalized vs. pixel image coordinates
    2. **Intrinsics** — the calibration matrix $K$ (focal length, principal
       point, skew)
    3. **Extrinsics** — the camera's pose in the world, $R$ and $t$
    4. The full camera matrix $P = K[R \mid t]$

    Each interactive section follows a **Predict → Run → Investigate**
    cycle: write down what you expect *before* touching the sliders, then
    play, then push further with a follow-up question.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Normalized vs. pixel image coordinates

    Notebook 3's perspective divide,

    $$
    (x, y, z) \;\longmapsto\; \left(\frac{x}{z}, \frac{y}{z}\right),
    $$

    gives **normalized image coordinates**: the image plane sits one unit
    in front of the camera center, centered on the optical axis, with no
    notion of pixels at all. A real sensor differs in three ways:

    - its pixels are counted in **pixels**, not world units, so there's an
      overall scale factor — the **focal length**, expressed in pixels;
    - horizontal and vertical pixel pitch can differ (non-square pixels),
      so the horizontal and vertical scale factors, $f_x$ and $f_y$, can
      differ too;
    - pixel row/column indices start from a **corner** of the sensor, not
      its center, so there's an offset — the **principal point** $(c_x,
      c_y)$ — between the optical axis and pixel $(0,0)$.

    The matrix that captures all of this (plus one more subtlety) is the
    **calibration matrix** $K$, built up next.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Camera intrinsics — the calibration matrix K

    Collecting focal length, principal point, and pixel skew into a single
    $3\times3$ matrix:

    $$
    K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
    $$

    Applied to normalized coordinates $(x/z,\ y/z,\ 1)$, this gives pixel
    coordinates $(u, v, 1)$:

    $$
    \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} x/z \\ y/z \\ 1 \end{bmatrix}
    \;\Longrightarrow\;
    u = f_x \frac{x}{z} + s\frac{y}{z} + c_x, \qquad v = f_y \frac{y}{z} + c_y
    $$

    | Parameter | Meaning |
    |---|---|
    | $f_x, f_y$ | focal length in pixels along each axis (unequal ⇒ non-square pixels / aspect ratio) |
    | $c_x, c_y$ | principal point — pixel location of the optical axis |
    | $s$ | skew — non-perpendicular pixel axes (essentially always ≈0 on real sensors, included for completeness) |

    $K$ is upper-triangular with 5 free parameters — the **intrinsic
    degrees of freedom**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Predict

    A $5\times5$ grid of points sits in front of the camera. With
    $f_x=f_y=300,\ c_x=c_y=0,\ s=0$, its image is centered and square.

    Before touching the sliders below: if you raise $c_x$ from 0 to 150,
    what happens to the grid image — does it stretch, shear, or shift? If
    you instead raise the skew $s$ from 0 while keeping $f_x=f_y$, what
    happens to the (still square) outline of the grid?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        - **Raising $c_x$** shifts every image point right by the same
          amount — $c_x$ only appears additively in $u$, so it's a pure
          **translation** of the image, not a stretch or shear.
        - **Raising skew $s$** adds a term $s \cdot (y/z)$ into $u$ only:
          points with the same $x$ but different $y$ get pushed apart
          horizontally by an amount proportional to $y$. A square grid
          turns into a **parallelogram** (shear), while $v$ is untouched.
        - Neither changes the *size* of the grid — that's controlled by
          $f_x, f_y$ instead.
        """)
    })
    return


@app.cell
def _(mo):
    fx_slider = mo.ui.slider(start=100, stop=500, value=300, step=20, label="fx")
    fy_slider = mo.ui.slider(start=100, stop=500, value=300, step=20, label="fy")
    skew_slider = mo.ui.slider(start=-200, stop=200, value=0.0, step=20, label="skew s")
    cx_slider = mo.ui.slider(start=-150, stop=150, value=0, step=10, label="cx")
    cy_slider = mo.ui.slider(start=-150, stop=150, value=0, step=10, label="cy")
    mo.hstack([fx_slider, fy_slider, skew_slider, cx_slider, cy_slider], justify="start", gap=2)
    return cx_slider, cy_slider, fx_slider, fy_slider, skew_slider


@app.cell
def _(cx_slider, cy_slider, fx_slider, fy_slider, mo, np, plt, skew_slider):
    _fx, _fy, _s = fx_slider.value, fy_slider.value, skew_slider.value
    _cx, _cy = cx_slider.value, cy_slider.value
    _K = np.array([[_fx, _s, _cx], [0, _fy, _cy], [0, 0, 1]])

    _grid = np.linspace(-0.4, 0.4, 5)
    _gx, _gy = np.meshgrid(_grid, _grid)
    _norm_pts = np.stack([_gx.ravel(), _gy.ravel(), np.ones(_gx.size)])  # (3, 25)
    _pix_pts = _K @ _norm_pts
    _pix_pts = _pix_pts[:2] / _pix_pts[2]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5))
    _ax1.scatter(_norm_pts[0], _norm_pts[1], color="tab:blue", s=25)
    _ax1.axhline(0, color="lightgray", linewidth=0.7)
    _ax1.axvline(0, color="lightgray", linewidth=0.7)
    _ax1.set_xlim(-0.6, 0.6)
    _ax1.set_ylim(-0.6, 0.6)
    _ax1.set_aspect("equal")
    _ax1.set_title("normalized coordinates (x/z, y/z)")
    _ax1.set_xlabel("x/z")
    _ax1.set_ylabel("y/z")

    _ax2.scatter(_pix_pts[0], _pix_pts[1], color="tab:orange", s=25)
    _ax2.scatter([_cx], [_cy], color="black", marker="+", s=100, label="principal point")
    _ax2.set_xlim(-400, 400)
    _ax2.set_ylim(-400, 400)
    _ax2.invert_yaxis()
    _ax2.set_aspect("equal")
    _ax2.grid(True, linewidth=0.3)
    _ax2.legend(loc="upper right")
    _ax2.set_title("pixel coordinates (u, v)")
    _ax2.set_xlabel("u")
    _ax2.set_ylabel("v")
    _fig.tight_layout()

    _readout = mo.md(f"""
    K =
    ```
    [{_fx:6.0f}  {_s:6.2f}  {_cx:6.0f}]
    [{0:6.0f}  {_fy:6.0f}  {_cy:6.0f}]
    [{0:6.0f}  {0:6.0f}  {1:6.0f}]
    ```
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Set $f_x \neq f_y$ (e.g. $f_x=500,\ f_y=160$) with skew and principal
      point at 0. What real camera property does this correspond to
      physically? What happens to a circle drawn in normalized coordinates?
    - Can *skew alone* turn the square grid into a parallelogram, without
      touching $f_x, f_y, c_x, c_y$? Try it.
    - Reset $f_x = f_y = 300$. Is there any combination of intrinsic
      parameters that could make the grid appear **rotated**? (Look
      carefully at where each parameter enters $K$ — this will matter when
      we get to extrinsics.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Camera extrinsics — placing the camera in the world (R, t)

    $K$ only describes the camera's *internal* geometry — it says nothing
    about where the camera is or which way it's pointing. That's the job of
    the **extrinsic** parameters: a rotation $R$ and translation $t$ that
    convert a world point into the camera's own coordinate frame,

    $$
    X_{\text{cam}} = R\, X_{\text{world}} + t,
    $$

    exactly the rigid-body (Euclidean) transformation from Notebook 3's
    §4, now interpreted as *"where is the camera relative to the world."*
    The camera's position in world coordinates (its **center**) is
    recovered as $C = -R^\top t$.

    Rather than raw roll/pitch/yaw sliders (already explored in Notebook
    3), the camera below **orbits** a cube sitting at the world origin —
    controlled by distance, azimuth, elevation, and roll (rotation about
    the viewing axis) — which is how calibration rigs are usually
    described in practice (a set of camera poses looking at a fixed
    target). $R$ and $t$ are computed from this orbit and shown below the
    plot.
    """)
    return


@app.cell
def _(np):
    # Unit cube centered at the origin, side length 2 (vertices at (+-1, +-1, +-1)).
    cube_vertices_local = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ]).T  # shape (3, 8)

    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    cube_faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (1, 2, 6, 5), (0, 3, 7, 4),
    ]
    cube_face_colors = ["#dddddd", "#444444", "#CC0000", "#FFB300", "#006747", "#4A90D9"]
    return cube_edges, cube_face_colors, cube_faces, cube_vertices_local


@app.cell
def _(np):
    def look_at(dist, az_deg, el_deg, roll_deg, target=None):
        """Orbit camera: returns (R, t, C) with X_cam = R @ X_world + t."""
        if target is None:
            target = np.zeros(3)
        _az, _el, _roll = np.radians([az_deg, el_deg, roll_deg])
        _C = target + dist * np.array(
            [np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)]
        )
        _fwd = target - _C
        _fwd = _fwd / np.linalg.norm(_fwd)
        _world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(_fwd, _world_up)) > 0.999:
            _world_up = np.array([0.0, 1.0, 0.0])
        _right = np.cross(_fwd, _world_up)
        _right = _right / np.linalg.norm(_right)
        _up = np.cross(_right, _fwd)
        _cr, _sr = np.cos(_roll), np.sin(_roll)
        _right_r = _cr * _right + _sr * _up
        _up_r = -_sr * _right + _cr * _up
        # camera axes expressed in world coords (rows): x_cam=right, y_cam=-up (image y grows downward), z_cam=fwd
        _R = np.stack([_right_r, -_up_r, _fwd], axis=0)
        _t = -_R @ _C
        return _R, _t, _C

    return (look_at,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Predict

    The cube sits fixed at the world origin. Suppose the camera orbits
    *farther away* (distance ↑) while its focal length stays fixed — does
    the cube's image get bigger, smaller, or stay the same size? Separately,
    if you crank the **roll** slider all the way to $180°$, what happens to
    the image — does the cube translate, rotate about its own image
    center, or vanish?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        - **Distance ↑ ⇒ image shrinks.** Farther objects subtend a smaller
          angle at the camera — this is perspective foreshortening again,
          now caused by moving the *camera* instead of the object.
        - **Roll = 180°** doesn't move the camera at all (distance,
          azimuth, elevation are unchanged) — it only rotates the camera
          about its own viewing axis. The cube's image rotates $180°$
          about the image center (appears **upside-down**), but stays the
          same size and position.
        """)
    })
    return


@app.cell
def _(mo):
    orbit_dist_slider = mo.ui.slider(start=4, stop=10, value=6, step=0.5, label="orbit distance")
    orbit_az_slider = mo.ui.slider(start=-180, stop=180, value=0, step=5, label="orbit azimuth")
    orbit_el_slider = mo.ui.slider(start=-80, stop=80, value=0, step=5, label="orbit elevation")
    orbit_roll_slider = mo.ui.slider(start=-180, stop=180, value=0, step=5, label="camera roll")
    mo.hstack(
        [orbit_dist_slider, orbit_az_slider, orbit_el_slider, orbit_roll_slider],
        justify="start", gap=2,
    )
    return (
        orbit_az_slider,
        orbit_dist_slider,
        orbit_el_slider,
        orbit_roll_slider,
    )


@app.cell
def _(
    Poly3DCollection,
    cube_edges,
    cube_face_colors,
    cube_faces,
    cube_vertices_local,
    look_at,
    mo,
    np,
    orbit_az_slider,
    orbit_dist_slider,
    orbit_el_slider,
    orbit_roll_slider,
    plt,
):
    _R, _t, _C = look_at(
        orbit_dist_slider.value, orbit_az_slider.value, orbit_el_slider.value, orbit_roll_slider.value
    )
    _f = 2.5  # fixed normalized focal length, just to draw an image-plane patch and a 2D preview
    _Xcam = _R @ cube_vertices_local + _t[:, None]  # (3, 8)
    _u = _f * _Xcam[0] / _Xcam[2]
    _v = _f * _Xcam[1] / _Xcam[2]

    _fig = plt.figure(figsize=(9.5, 5))
    _ax3d = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax2d = _fig.add_subplot(1, 2, 2)

    _face_polys = [cube_vertices_local.T[list(_face)] for _face in cube_faces]
    _ax3d.add_collection3d(
        Poly3DCollection(_face_polys, facecolor=cube_face_colors, edgecolor="black", linewidths=0.6, alpha=0.95)
    )
    _ax3d.scatter(*_C, color="black", s=40, label="camera center")
    # camera frustum: 4 corners of the image plane in world coords, plus lines of sight to cube vertices
    _corners_cam = _f * np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]).T  # (3,4) in normalized cam coords
    _corners_world = _R.T @ (_corners_cam - _t[:, None])
    for _k in range(4):
        _ax3d.plot(*zip(_C, _corners_world[:, _k]), color="gray", linewidth=0.7)
    _frustum = np.hstack([_corners_world, _corners_world[:, :1]])
    _ax3d.plot(_frustum[0], _frustum[1], _frustum[2], color="tab:blue", linewidth=1)
    for _k in range(8):
        _ax3d.plot(
            [_C[0], cube_vertices_local[0, _k]], [_C[1], cube_vertices_local[1, _k]], [_C[2], cube_vertices_local[2, _k]],
            color="tab:red", linewidth=0.5, linestyle=":",
        )
    _lim = 8
    _ax3d.set_xlim(-_lim, _lim)
    _ax3d.set_ylim(-_lim, _lim)
    _ax3d.set_zlim(-_lim, _lim)
    _ax3d.set_box_aspect((1, 1, 1))
    _ax3d.view_init(elev=20, azim=-50)
    _ax3d.set_xlabel("x")
    _ax3d.set_ylabel("y")
    _ax3d.set_zlabel("z")
    _ax3d.legend(loc="upper left")
    _ax3d.set_title("camera orbiting a fixed cube, lines of sight (dotted)")

    for _i, _j in cube_edges:
        _ax2d.plot([_u[_i], _u[_j]], [_v[_i], _v[_j]], color="tab:blue")
    _ax2d.set_xlim(-2, 2)
    _ax2d.set_ylim(-2, 2)
    _ax2d.invert_yaxis()
    _ax2d.set_aspect("equal")
    _ax2d.grid(True, linewidth=0.3)
    _ax2d.set_title("image seen by the camera")

    _readout = mo.md(f"""
    R (world → camera) =
    ```
    [{_R[0,0]:6.2f}  {_R[0,1]:6.2f}  {_R[0,2]:6.2f}]
    [{_R[1,0]:6.2f}  {_R[1,1]:6.2f}  {_R[1,2]:6.2f}]
    [{_R[2,0]:6.2f}  {_R[2,1]:6.2f}  {_R[2,2]:6.2f}]
    ```
    t = **({_t[0]:.2f}, {_t[1]:.2f}, {_t[2]:.2f})**, camera center C = **({_C[0]:.2f}, {_C[1]:.2f}, {_C[2]:.2f})**
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Slowly increase the orbit **elevation**. Find an angle where one of
      the cube's colored faces that was visible from below disappears from
      view. Does the projected wireframe on the right *know* that face is
      now hidden, or does it just keep drawing every edge regardless of
      occlusion? What does that tell you about the difference between
      *projection* (the math in this notebook) and *rendering* (what a
      graphics engine or renderer also has to solve)?
    - At what **roll** value does the image appear upside-down? Does
      changing roll ever move where the cube sits in the image, or only
      how it's oriented?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Putting it together — P = K[R|t]

    Composing extrinsics (world → camera) and intrinsics (camera → pixels)
    gives the full $3\times4$ **camera matrix**:

    $$
    P = K\,[R \mid t\,], \qquad
    \begin{bmatrix} u \\ v \\ w \end{bmatrix} = P \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix},
    \qquad (u_{\text{px}}, v_{\text{px}}) = \left(\frac{u}{w}, \frac{v}{w}\right)
    $$

    $P$ is defined only up to overall scale (multiplying every entry by a
    nonzero constant doesn't change the pixel it maps to), so it has:

    | Component | Symbol | DOF | Parameters |
    |---|---|---|---|
    | Intrinsics | $K$ | **5** | $f_x, f_y, s, c_x, c_y$ |
    | Extrinsics | $R, t$ | **6** | 3 rotation + $t_x, t_y, t_z$ |
    | **Total (up to scale)** | $P$ | **11** | — |

    This single matrix is what real camera calibration solves for — given
    known 3D points and their observed pixel locations (e.g. a
    checkerboard), recover $K$, $R$, and $t$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Predict

    Both an intrinsic slider group and an extrinsic (orbit) slider group
    are about to appear together, both acting on the same cube. Which
    **single** slider — one intrinsic, one extrinsic — could you move to
    shift the cube sideways in the image *without* changing its apparent
    size or shape at all? Is your answer an intrinsic or an extrinsic
    parameter?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        The **principal point** $c_x$ (intrinsic) shifts the image purely
        sideways with no change in size, since it enters $u$ only
        additively. Moving the **camera** sideways (an extrinsic change to
        $t$, e.g. orbiting in azimuth) *also* shifts the image, but — for
        a perspective camera — it generally changes the cube's apparent
        size and shape too, because it changes the distance and viewing
        angle to different parts of the cube. So a pure, shape-preserving
        sideways shift is an **intrinsic** effect, not an extrinsic one —
        this is exactly the ambiguity real calibration has to untangle
        using several different views of a known target.
        """)
    })
    return


@app.cell
def _(mo):
    p_fx_slider = mo.ui.slider(start=100, stop=500, value=300, step=20, label="fx")
    p_fy_slider = mo.ui.slider(start=100, stop=500, value=300, step=20, label="fy")
    p_cx_slider = mo.ui.slider(start=-150, stop=150, value=0, step=10, label="cx")
    p_cy_slider = mo.ui.slider(start=-150, stop=150, value=0, step=10, label="cy")
    p_dist_slider = mo.ui.slider(start=4, stop=10, value=6, step=0.5, label="orbit distance")
    p_az_slider = mo.ui.slider(start=-180, stop=180, value=35, step=5, label="orbit azimuth")
    p_el_slider = mo.ui.slider(start=-80, stop=80, value=20, step=5, label="orbit elevation")
    p_roll_slider = mo.ui.slider(start=-180, stop=180, value=10, step=5, label="camera roll")
    mo.vstack([
        mo.hstack([p_fx_slider, p_fy_slider, p_cx_slider, p_cy_slider], justify="start", gap=2),
        mo.hstack([p_dist_slider, p_az_slider, p_el_slider, p_roll_slider], justify="start", gap=2),
    ])
    return (
        p_az_slider,
        p_cx_slider,
        p_cy_slider,
        p_dist_slider,
        p_el_slider,
        p_fx_slider,
        p_fy_slider,
        p_roll_slider,
    )


@app.cell
def _(
    MplPolygon,
    Poly3DCollection,
    cube_face_colors,
    cube_faces,
    cube_vertices_local,
    look_at,
    mo,
    np,
    p_az_slider,
    p_cx_slider,
    p_cy_slider,
    p_dist_slider,
    p_el_slider,
    p_fx_slider,
    p_fy_slider,
    p_roll_slider,
    plt,
):
    _R, _t, _C = look_at(p_dist_slider.value, p_az_slider.value, p_el_slider.value, p_roll_slider.value)
    _K = np.array([
        [p_fx_slider.value, 0, p_cx_slider.value],
        [0, p_fy_slider.value, p_cy_slider.value],
        [0, 0, 1],
    ])
    _P = _K @ np.hstack([_R, _t[:, None]])  # (3, 4)

    _Xw_h = np.vstack([cube_vertices_local, np.ones(8)])  # (4, 8)
    _uvw = _P @ _Xw_h
    _uv = _uvw[:2] / _uvw[2]
    _Xcam = _R @ cube_vertices_local + _t[:, None]

    _fig = plt.figure(figsize=(9.5, 5))
    _ax3d = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax2d = _fig.add_subplot(1, 2, 2)

    _face_polys = [cube_vertices_local.T[list(_face)] for _face in cube_faces]
    _ax3d.add_collection3d(
        Poly3DCollection(_face_polys, facecolor=cube_face_colors, edgecolor="black", linewidths=0.6, alpha=0.95)
    )
    _ax3d.scatter(*_C, color="black", s=40, label="camera center")
    for _k in range(8):
        _ax3d.plot(
            [_C[0], cube_vertices_local[0, _k]], [_C[1], cube_vertices_local[1, _k]], [_C[2], cube_vertices_local[2, _k]],
            color="tab:red", linewidth=0.5, linestyle=":",
        )
    _lim = 8
    _ax3d.set_xlim(-_lim, _lim)
    _ax3d.set_ylim(-_lim, _lim)
    _ax3d.set_zlim(-_lim, _lim)
    _ax3d.set_box_aspect((1, 1, 1))
    _ax3d.view_init(elev=20, azim=-50)
    _ax3d.set_xlabel("x")
    _ax3d.set_ylabel("y")
    _ax3d.set_zlabel("z")
    _ax3d.legend(loc="upper left")
    _ax3d.set_title("world scene: camera pose + lines of sight")

    _face_depth = [_Xcam[2, list(_face)].mean() for _face in cube_faces]
    _order = np.argsort(_face_depth)[::-1]  # back to front (painter's algorithm)
    for _idx in _order:
        _face = cube_faces[_idx]
        _pts = _uv[:, list(_face)].T
        _ax2d.add_patch(
            MplPolygon(_pts, closed=True, facecolor=cube_face_colors[_idx], edgecolor="black", linewidth=0.8)
        )
    _ax2d.set_xlim(-400, 400)
    _ax2d.set_ylim(-400, 400)
    _ax2d.invert_yaxis()
    _ax2d.set_aspect("equal")
    _ax2d.grid(True, linewidth=0.3)
    _ax2d.set_title("rendered photo (pixel coordinates)")
    _ax2d.set_xlabel("u")
    _ax2d.set_ylabel("v")

    _readout = mo.md(f"""
    P = K[R|t] =
    ```
    [{_P[0,0]:7.1f}  {_P[0,1]:7.1f}  {_P[0,2]:7.1f}  {_P[0,3]:7.1f}]
    [{_P[1,0]:7.1f}  {_P[1,1]:7.1f}  {_P[1,2]:7.1f}  {_P[1,3]:7.1f}]
    [{_P[2,0]:7.2f}  {_P[2,1]:7.2f}  {_P[2,2]:7.2f}  {_P[2,3]:7.2f}]
    ```
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - With the camera pose fixed, sweep $f_x$ and $f_y$ together. Is a
      change in "how big the cube looks" caused by moving the camera
      (extrinsic) distinguishable from the same change caused by
      increasing focal length (intrinsic), from a *single* photo alone?
      This is exactly why calibration needs **multiple views** of a known
      target — a single image can't tell intrinsics and extrinsics apart.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - Normalized image coordinates $(x/z, y/z)$ become **pixel**
      coordinates through the intrinsic calibration matrix
      $K = [f_x, s, c_x;\ 0, f_y, c_y;\ 0, 0, 1]$ — 5 DOF: focal length
      ($f_x, f_y$), principal point ($c_x, c_y$), and skew ($s$).
    - **Extrinsics** $R, t$ place the camera in the world:
      $X_{\text{cam}} = R X_{\text{world}} + t$, with camera center
      $C = -R^\top t$ — the same rigid-body transformation from
      Notebook 3, now describing camera pose instead of object pose.
    - The full **camera matrix** $P = K[R\mid t]$ is a $3\times4$ matrix
      with 11 DOF (up to scale) that sends homogeneous 3D world points
      directly to homogeneous pixel coordinates.
    - Intrinsic and extrinsic changes can look similar in a single image
      (e.g. principal-point shift vs. camera translation) — resolving that
      ambiguity is why real camera calibration uses **multiple views** of
      a known target rather than a single photo.

    **Next up:** given our camera geometry, we explain where the discrete color or intensity values come from in an image, and how they relate to lighting, and material and optical properties.
    """)
    return


if __name__ == "__main__":
    app.run()
