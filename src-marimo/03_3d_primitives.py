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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return Poly3DCollection, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 3D Geometry, Rotations, and Projections

    Notebook 2 built up 2D points, lines, and transformations in homogeneous
    coordinates. This notebook lifts the same ideas into **3D**, and then
    asks the question a camera has to answer every time it takes a picture:
    how do we turn a 3D scene into a 2D image?

    This notebook covers:

    1. **3D lines** — point + direction parametric form
    2. **3D planes** — homogeneous $(a, b, c, d)$ form and the normal +
       distance form
    3. **Rotation using Euler angles** — building a 3D rotation from three
       elementary rotations, applied to a cube
    4. **3D transformations** as $4 \times 4$ matrices: translation,
       Euclidean, similarity, and affine, and their degrees of freedom
    5. **3D-to-2D projection** — orthographic and perspective (pinhole
       camera) projection of a cube, including the lines of sight
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 3D lines

    A line in 3D is naturally described in **parametric form**: pick a
    point $p_0$ on the line and a direction vector $d$, then every point on
    the line is

    $$
    p(t) = p_0 + t\, d, \qquad t \in \mathbb{R}
    $$

    Unlike in 2D, a single linear equation $ax+by+cz+d=0$ describes a
    **plane**, not a line — a 3D line needs two such equations (the
    intersection of two planes), so the point + direction form is usually
    the more convenient representation. Homogeneous points in 3D are
    4-vectors, $(x, y, z) \to (x, y, z, 1)$, again defined only up to
    scale.

    Drag the view sliders to rotate the camera and get a feel for the line
    sitting in 3D space.
    """)
    return


@app.cell
def _(mo):
    az_line_slider = mo.ui.slider(
        start=0, stop=360, value=40, step=10, label="Direction azimuth θ (degrees)"
    )
    el_line_slider = mo.ui.slider(
        start=-80, stop=80, value=25, step=10, label="Direction elevation φ (degrees)"
    )
    view_elev_line_slider = mo.ui.slider(
        start=-90, stop=90, value=20, step=5, label="View elevation"
    )
    view_azim_line_slider = mo.ui.slider(
        start=-180, stop=180, value=-60, step=5, label="View azimuth"
    )
    mo.hstack(
        [az_line_slider, el_line_slider, view_elev_line_slider, view_azim_line_slider],
        justify="start", gap=2,
    )
    return (
        az_line_slider,
        el_line_slider,
        view_azim_line_slider,
        view_elev_line_slider,
    )


@app.cell
def _(
    az_line_slider,
    el_line_slider,
    mo,
    np,
    plt,
    view_azim_line_slider,
    view_elev_line_slider,
):
    _theta = np.radians(az_line_slider.value)
    _phi = np.radians(el_line_slider.value)
    _d = np.array([np.cos(_phi) * np.cos(_theta), np.cos(_phi) * np.sin(_theta), np.sin(_phi)])
    _p0 = np.array([1.0, -1.0, 0.5])

    _t = np.linspace(-4, 4, 2)
    _line_pts = _p0[:, None] + _t[None, :] * _d[:, None]

    _fig = plt.figure(figsize=(5.5, 5.5))
    _ax = _fig.add_subplot(111, projection="3d")
    _ax.plot([-4, 4], [0, 0], [0, 0], color="lightgray", linewidth=0.7)
    _ax.plot([0, 0], [-4, 4], [0, 0], color="lightgray", linewidth=0.7)
    _ax.plot([0, 0], [0, 0], [-4, 4], color="lightgray", linewidth=0.7)
    _ax.plot(_line_pts[0], _line_pts[1], _line_pts[2], color="tab:blue", linewidth=2, label="line")
    _ax.scatter(*_p0, color="tab:red", s=40, label="point p0")
    _ax.quiver(*_p0, *_d, length=1.5, color="tab:red", normalize=True)
    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
    _ax.set_zlim(-4, 4)
    _ax.set_box_aspect((1, 1, 1))
    _ax.view_init(elev=view_elev_line_slider.value, azim=view_azim_line_slider.value)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_zlabel("z")
    _ax.legend(loc="upper left")
    _ax.set_title(f"θ = {az_line_slider.value}°, φ = {el_line_slider.value}°")

    _readout = mo.md(f"""
    - point on line p0 = **({_p0[0]:.1f}, {_p0[1]:.1f}, {_p0[2]:.1f})**
    - direction d = **({_d[0]:.2f}, {_d[1]:.2f}, {_d[2]:.2f})**
    - parametric form: p(t) = p0 + t·d
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 3D planes

    A plane in 3D satisfies a single linear equation

    $$
    a x + b y + c z + d = 0
    $$

    Collecting the coefficients into a 4-vector $\pi = (a, b, c, d)$ gives
    the plane's homogeneous coordinates — directly generalizing the 2D line
    from Notebook 2. A homogeneous point $\tilde{x} = (x, y, z, 1)$ lies
    **on** the plane exactly when

    $$
    \pi \cdot \tilde{x} = 0
    $$

    As with 2D lines, dividing through by $\|(a, b, c)\|$ gives the
    **normal form**

    $$
    n \cdot x = \delta, \qquad n = (a, b, c) / \|(a, b, c)\|,\ \ \|n\| = 1
    $$

    where $n$ is the plane's unit normal and $\delta \geq 0$ is the
    perpendicular distance from the origin to the plane.
    """)
    return


@app.cell
def _(mo):
    theta_n_slider = mo.ui.slider(
        start=0, stop=360, value=40, step=10, label="Normal azimuth θ (degrees)"
    )
    phi_n_slider = mo.ui.slider(
        start=-80, stop=80, value=30, step=10, label="Normal elevation φ (degrees)"
    )
    d_plane_slider = mo.ui.slider(
        start=0, stop=6, value=3, step=0.5, label="Distance from origin δ"
    )
    view_elev_plane_slider = mo.ui.slider(
        start=-90, stop=90, value=20, step=5, label="View elevation"
    )
    view_azim_plane_slider = mo.ui.slider(
        start=-180, stop=180, value=-60, step=5, label="View azimuth"
    )
    mo.hstack(
        [theta_n_slider, phi_n_slider, d_plane_slider, view_elev_plane_slider, view_azim_plane_slider],
        justify="start", gap=2,
    )
    return (
        d_plane_slider,
        phi_n_slider,
        theta_n_slider,
        view_azim_plane_slider,
        view_elev_plane_slider,
    )


@app.cell
def _(
    Poly3DCollection,
    d_plane_slider,
    mo,
    np,
    phi_n_slider,
    plt,
    theta_n_slider,
    view_azim_plane_slider,
    view_elev_plane_slider,
):
    _theta = np.radians(theta_n_slider.value)
    _phi = np.radians(phi_n_slider.value)
    _n = np.array([np.cos(_phi) * np.cos(_theta), np.cos(_phi) * np.sin(_theta), np.sin(_phi)])
    _delta = d_plane_slider.value
    _foot = _delta * _n

    _ref = np.array([0.0, 0.0, 1.0]) if abs(_n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    _u = np.cross(_ref, _n)
    _u = _u / np.linalg.norm(_u)
    _v = np.cross(_n, _u)
    _size = 3.5
    _corners = np.array([
        _foot + _size * _u + _size * _v,
        _foot + _size * _u - _size * _v,
        _foot - _size * _u - _size * _v,
        _foot - _size * _u + _size * _v,
    ])

    _fig = plt.figure(figsize=(5.5, 5.5))
    _ax = _fig.add_subplot(111, projection="3d")
    _ax.plot([-5, 5], [0, 0], [0, 0], color="lightgray", linewidth=0.7)
    _ax.plot([0, 0], [-5, 5], [0, 0], color="lightgray", linewidth=0.7)
    _ax.plot([0, 0], [0, 0], [-5, 5], color="lightgray", linewidth=0.7)
    _poly = Poly3DCollection([_corners], facecolor="tab:blue", edgecolor="black", alpha=0.4)
    _ax.add_collection3d(_poly)
    _ax.plot([0, _foot[0]], [0, _foot[1]], [0, _foot[2]], "--", color="tab:red", linewidth=1.5)
    _ax.scatter(*_foot, color="tab:red", s=40, label="foot point (distance δ)")
    _ax.scatter(0, 0, 0, color="black", s=30, label="origin")
    _ax.quiver(*_foot, *_n, length=1.2, color="tab:red", normalize=True)
    _ax.set_xlim(-5, 5)
    _ax.set_ylim(-5, 5)
    _ax.set_zlim(-5, 5)
    _ax.set_box_aspect((1, 1, 1))
    _ax.view_init(elev=view_elev_plane_slider.value, azim=view_azim_plane_slider.value)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_zlabel("z")
    _ax.legend(loc="upper left")
    _ax.set_title(f"θ = {theta_n_slider.value}°, φ = {phi_n_slider.value}°, δ = {_delta:.1f}")

    _readout = mo.md(f"""
    - unit normal n = **({_n[0]:.2f}, {_n[1]:.2f}, {_n[2]:.2f})**
    - distance δ = **{_delta:.1f}**
    - plane coefficients (a, b, c, d) = **({_n[0]:.2f}, {_n[1]:.2f}, {_n[2]:.2f}, {-_delta:.2f})**
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Rotation using Euler angles

    Any 3D rotation can be built by composing three elementary rotations
    about the coordinate axes:

    $$
    R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}
    \quad
    R_y(\beta) = \begin{bmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{bmatrix}
    \quad
    R_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}
    $$

    $\alpha, \beta, \gamma$ are the **Euler angles** — commonly called
    **roll**, **pitch**, and **yaw**. A full rotation is their product, e.g.

    $$
    R(\alpha, \beta, \gamma) = R_z(\gamma)\, R_y(\beta)\, R_x(\alpha)
    $$

    (the "yaw-pitch-roll" convention: roll first, then pitch, then yaw).
    Matrix multiplication does **not** commute, so **order matters**:
    $R_z R_y R_x \neq R_x R_y R_z$ in general, and swapping the order gives
    a different final orientation. This convention also has a well-known
    failure mode called **gimbal lock**: when the pitch angle reaches
    $\pm 90°$, the roll and yaw axes become aligned and one degree of
    rotational freedom is lost.

    Each face of the cube below is colored differently so you can see the
    rotation happen — try pushing pitch toward $\pm90°$ to see roll and yaw
    start to do the same thing (gimbal lock).
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
def _(mo):
    roll_slider = mo.ui.slider(start=-180, stop=180, value=20, step=5, label="Roll α (degrees, about x)")
    pitch_slider = mo.ui.slider(start=-180, stop=180, value=25, step=5, label="Pitch β (degrees, about y)")
    yaw_slider = mo.ui.slider(start=-180, stop=180, value=30, step=5, label="Yaw γ (degrees, about z)")
    mo.hstack([roll_slider, pitch_slider, yaw_slider], justify="start", gap=2)
    return pitch_slider, roll_slider, yaw_slider


@app.cell
def _(
    Poly3DCollection,
    cube_face_colors,
    cube_faces,
    cube_vertices_local,
    mo,
    np,
    pitch_slider,
    plt,
    roll_slider,
    yaw_slider,
):
    _alpha = np.radians(roll_slider.value)
    _beta = np.radians(pitch_slider.value)
    _gamma = np.radians(yaw_slider.value)

    _Rx = np.array([[1, 0, 0], [0, np.cos(_alpha), -np.sin(_alpha)], [0, np.sin(_alpha), np.cos(_alpha)]])
    _Ry = np.array([[np.cos(_beta), 0, np.sin(_beta)], [0, 1, 0], [-np.sin(_beta), 0, np.cos(_beta)]])
    _Rz = np.array([[np.cos(_gamma), -np.sin(_gamma), 0], [np.sin(_gamma), np.cos(_gamma), 0], [0, 0, 1]])
    _R = _Rz @ _Ry @ _Rx

    _rot_verts = (_R @ cube_vertices_local).T  # (8, 3)
    _face_polys = [_rot_verts[list(_face)] for _face in cube_faces]

    _fig = plt.figure(figsize=(5.5, 5.5))
    _ax = _fig.add_subplot(111, projection="3d")
    _poly = Poly3DCollection(_face_polys, facecolor=cube_face_colors, edgecolor="black", linewidths=0.8, alpha=0.95)
    _ax.add_collection3d(_poly)
    _lim = 2.2
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_zlim(-_lim, _lim)
    _ax.set_box_aspect((1, 1, 1))
    _ax.view_init(elev=20, azim=-60)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_zlabel("z")
    _ax.set_title(f"α (roll) = {roll_slider.value}°, β (pitch) = {pitch_slider.value}°, γ (yaw) = {yaw_slider.value}°")

    mo.vstack([
        mo.md(f"R = Rz(**{yaw_slider.value}°**) · Ry(**{pitch_slider.value}°**) · Rx(**{roll_slider.value}°**)"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 3D transformations as $4\times 4$ matrices

    Exactly as in 2D, every 3D transformation below has the same template —
    a $4\times4$ matrix applied to a homogeneous point — with the bottom
    row fixed at $(0, 0, 0, 1)$:

    $$
    T = \begin{bmatrix} A & t \\ 0\ \ 0\ \ 0 & 1 \end{bmatrix}, \qquad
    \begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = T \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
    $$

    where $A$ is a $3\times3$ linear part and $t = (t_x, t_y, t_z)$ is a
    translation. As before, the fewer constraints on $A$, the more degrees
    of freedom (DOF) and the more the object is allowed to distort:

    | Transformation | Linear part $A$ | DOF | Free parameters | Preserves |
    |---|---|---|---|---|
    | **Translation** | identity | **3** | $t_x, t_y, t_z$ | lengths, angles, volume, orientation |
    | **Euclidean (rigid body)** | rotation $R(\alpha,\beta,\gamma) \in SO(3)$ | **6** | 3 Euler angles, $t_x, t_y, t_z$ | lengths, angles, volume |
    | **Similarity** | scaled rotation $s\,R(\alpha,\beta,\gamma)$ | **7** | 3 Euler angles, $s, t_x, t_y, t_z$ | angles, shape (length ratios) |
    | **Affine** | any invertible $3\times3$ matrix | **12** | 9 entries of $A$, $t_x, t_y, t_z$ | parallelism, length ratios along a line |

    Same nesting pattern as in 2D: translation ⊂ Euclidean ⊂ similarity ⊂
    affine, each relaxing one more constraint on $A$. The rotation demo
    above is a pure Euclidean transformation with $t = 0$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 3D-to-2D projection

    A camera maps 3D world points to a 2D image. We'll place the camera at
    the origin, looking down the $+z$ axis ("depth" increases as $z$
    increases), and compare the two projections most relevant to computer
    vision. In both cases, each 3D point sends out a **line of sight**
    (also called a projection ray) toward the camera, and where that ray
    crosses the **image plane** is the projected 2D point.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.1 Orthographic projection

    Orthographic projection simply **drops the depth coordinate**:

    $$
    (x, y, z) \;\longmapsto\; (x, y)
    $$

    In homogeneous form, this is the $3\times4$ matrix

    $$
    P_{\text{ortho}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix},
    \qquad
    \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = P_{\text{ortho}} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
    $$

    All lines of sight are **parallel** to the viewing axis — there is no
    single center of projection. A direct consequence: moving the cube
    closer or farther from the camera (the depth slider) changes *where*
    the cube sits, but not the *size* of its projected image. Orthographic
    projection is what you get from architectural/engineering drawings and
    from a camera with a very long lens far from the scene.
    """)
    return


@app.cell
def _(mo):
    roll_o_slider = mo.ui.slider(start=-180, stop=180, value=20, step=10, label="Roll (degrees)")
    pitch_o_slider = mo.ui.slider(start=-180, stop=180, value=25, step=10, label="Pitch (degrees)")
    yaw_o_slider = mo.ui.slider(start=-180, stop=180, value=30, step=10, label="Yaw (degrees)")
    zc_o_slider = mo.ui.slider(start=3, stop=12, value=6, step=0.5, label="Cube depth (distance from camera)")
    mo.hstack([roll_o_slider, pitch_o_slider, yaw_o_slider, zc_o_slider], justify="start", gap=2)
    return pitch_o_slider, roll_o_slider, yaw_o_slider, zc_o_slider


@app.cell
def _(
    cube_edges,
    cube_vertices_local,
    mo,
    np,
    pitch_o_slider,
    plt,
    roll_o_slider,
    yaw_o_slider,
    zc_o_slider,
):
    _alpha = np.radians(roll_o_slider.value)
    _beta = np.radians(pitch_o_slider.value)
    _gamma = np.radians(yaw_o_slider.value)
    _Rx = np.array([[1, 0, 0], [0, np.cos(_alpha), -np.sin(_alpha)], [0, np.sin(_alpha), np.cos(_alpha)]])
    _Ry = np.array([[np.cos(_beta), 0, np.sin(_beta)], [0, 1, 0], [-np.sin(_beta), 0, np.cos(_beta)]])
    _Rz = np.array([[np.cos(_gamma), -np.sin(_gamma), 0], [np.sin(_gamma), np.cos(_gamma), 0], [0, 0, 1]])
    _R = _Rz @ _Ry @ _Rx
    _zc = zc_o_slider.value

    _world = _R @ cube_vertices_local + np.array([[0.0], [0.0], [_zc]])  # (3, 8)
    _proj = _world[:2]  # (2, 8), orthographic image coordinates

    _fig = plt.figure(figsize=(9.5, 5))
    _ax3d = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax2d = _fig.add_subplot(1, 2, 2)

    _plane_size = 4
    _xx, _yy = np.meshgrid([-_plane_size, _plane_size], [-_plane_size, _plane_size])
    _ax3d.plot_surface(_xx, _yy, np.zeros_like(_xx), color="tab:blue", alpha=0.15)
    _ax3d.scatter(0, 0, 0, color="black", s=30, label="camera")

    for _i, _j in cube_edges:
        _ax3d.plot(
            [_world[0, _i], _world[0, _j]], [_world[1, _i], _world[1, _j]], [_world[2, _i], _world[2, _j]],
            color="tab:red", linewidth=1.5,
        )
        _ax3d.plot(
            [_proj[0, _i], _proj[0, _j]], [_proj[1, _i], _proj[1, _j]], [0, 0],
            color="tab:blue", linewidth=1.5,
        )
    for _k in range(8):
        _x, _y, _z = _world[:, _k]
        _ax3d.plot([_x, _x], [_y, _y], [_z, 0], color="gray", linewidth=0.6, linestyle=":")

    _ax3d.set_xlim(-6, 6)
    _ax3d.set_ylim(-6, 6)
    _ax3d.set_zlim(0, 14)
    _ax3d.set_box_aspect((1, 1, 1))
    _ax3d.view_init(elev=15, azim=-70)
    _ax3d.set_xlabel("x")
    _ax3d.set_ylabel("y")
    _ax3d.set_zlabel("z (depth)")
    _ax3d.legend(loc="upper left")
    _ax3d.set_title("cube, lines of sight (dotted), image plane")

    for _i, _j in cube_edges:
        _ax2d.plot([_proj[0, _i], _proj[0, _j]], [_proj[1, _i], _proj[1, _j]], color="tab:blue")
    _ax2d.set_xlim(-4, 4)
    _ax2d.set_ylim(-4, 4)
    _ax2d.set_aspect("equal")
    _ax2d.grid(True, linewidth=0.3)
    _ax2d.set_title("orthographic image (u, v)")

    mo.vstack([
        mo.md(f"depth = **{_zc:.1f}** — the image on the right does not change size as you move this slider"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 Perspective projection

    Perspective projection models a **pinhole camera**: all lines of sight
    pass through a single point, the **center of projection** (the pinhole,
    here at the origin), and are captured where they cross an image plane a
    focal length $f$ in front of the camera:

    $$
    (x, y, z) \;\longmapsto\; \left(f\frac{x}{z},\ f\frac{y}{z}\right)
    $$

    In homogeneous form,

    $$
    P_{\text{persp}} = \begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix},
    \qquad
    \begin{bmatrix} f x \\ f y \\ z \end{bmatrix} = P_{\text{persp}} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
    \;\longrightarrow\;
    (u, v) = \left(\frac{fx}{z}, \frac{fy}{z}\right)
    $$

    Dividing by $z$ — the point's depth — is exactly the dehomogenization
    step from Notebook 2, now dividing by depth instead of an arbitrary
    scale factor. Unlike orthographic projection, moving the cube farther
    away (larger depth) now visibly **shrinks** its image: this is
    perspective foreshortening, and it's why far-away objects look smaller.
    A shorter focal length $f$ widens the field of view and exaggerates the
    effect.
    """)
    return


@app.cell
def _(mo):
    roll_p_slider = mo.ui.slider(start=-180, stop=180, value=20, step=10, label="Roll (degrees)")
    pitch_p_slider = mo.ui.slider(start=-180, stop=180, value=25, step=10, label="Pitch (degrees)")
    yaw_p_slider = mo.ui.slider(start=-180, stop=180, value=30, step=10, label="Yaw (degrees)")
    zc_p_slider = mo.ui.slider(start=5, stop=12, value=6, step=0.5, label="Cube depth (distance from camera)")
    focal_p_slider = mo.ui.slider(start=1, stop=3, value=2, step=0.25, label="Focal length f")
    mo.hstack(
        [roll_p_slider, pitch_p_slider, yaw_p_slider, zc_p_slider, focal_p_slider],
        justify="start", gap=2,
    )
    return (
        focal_p_slider,
        pitch_p_slider,
        roll_p_slider,
        yaw_p_slider,
        zc_p_slider,
    )


@app.cell
def _(
    cube_edges,
    cube_vertices_local,
    focal_p_slider,
    mo,
    np,
    pitch_p_slider,
    plt,
    roll_p_slider,
    yaw_p_slider,
    zc_p_slider,
):
    _alpha = np.radians(roll_p_slider.value)
    _beta = np.radians(pitch_p_slider.value)
    _gamma = np.radians(yaw_p_slider.value)
    _Rx = np.array([[1, 0, 0], [0, np.cos(_alpha), -np.sin(_alpha)], [0, np.sin(_alpha), np.cos(_alpha)]])
    _Ry = np.array([[np.cos(_beta), 0, np.sin(_beta)], [0, 1, 0], [-np.sin(_beta), 0, np.cos(_beta)]])
    _Rz = np.array([[np.cos(_gamma), -np.sin(_gamma), 0], [np.sin(_gamma), np.cos(_gamma), 0], [0, 0, 1]])
    _R = _Rz @ _Ry @ _Rx
    _zc = zc_p_slider.value
    _f = focal_p_slider.value

    _world = _R @ cube_vertices_local + np.array([[0.0], [0.0], [_zc]])  # (3, 8)
    _u = _f * _world[0] / _world[2]
    _v = _f * _world[1] / _world[2]
    _plane_pts = np.vstack([_u, _v, np.full(8, _f)])  # (3, 8), points on the image plane

    _fig = plt.figure(figsize=(9.5, 5))
    _ax3d = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax2d = _fig.add_subplot(1, 2, 2)

    _plane_size = 4
    _xx, _yy = np.meshgrid([-_plane_size, _plane_size], [-_plane_size, _plane_size])
    _ax3d.plot_surface(_xx, _yy, np.full_like(_xx, _f), color="tab:blue", alpha=0.15)
    _ax3d.scatter(0, 0, 0, color="black", s=40, label="camera (center of projection)")

    for _i, _j in cube_edges:
        _ax3d.plot(
            [_world[0, _i], _world[0, _j]], [_world[1, _i], _world[1, _j]], [_world[2, _i], _world[2, _j]],
            color="tab:red", linewidth=1.5,
        )
        _ax3d.plot(
            [_plane_pts[0, _i], _plane_pts[0, _j]], [_plane_pts[1, _i], _plane_pts[1, _j]], [_plane_pts[2, _i], _plane_pts[2, _j]],
            color="tab:blue", linewidth=1.5,
        )
    for _k in range(8):
        _x, _y, _z = _world[:, _k]
        _ax3d.plot([0, _x], [0, _y], [0, _z], color="gray", linewidth=0.6, linestyle=":")

    _ax3d.set_xlim(-6, 6)
    _ax3d.set_ylim(-6, 6)
    _ax3d.set_zlim(0, 14)
    _ax3d.set_box_aspect((1, 1, 1))
    _ax3d.view_init(elev=15, azim=-70)
    _ax3d.set_xlabel("x")
    _ax3d.set_ylabel("y")
    _ax3d.set_zlabel("z (depth)")
    _ax3d.legend(loc="upper left")
    _ax3d.set_title("cube, lines of sight through the camera center")

    for _i, _j in cube_edges:
        _ax2d.plot([_plane_pts[0, _i], _plane_pts[0, _j]], [_plane_pts[1, _i], _plane_pts[1, _j]], color="tab:blue")
    _lim2d = _plane_size
    _ax2d.set_xlim(-_lim2d, _lim2d)
    _ax2d.set_ylim(-_lim2d, _lim2d)
    _ax2d.set_aspect("equal")
    _ax2d.grid(True, linewidth=0.3)
    _ax2d.set_title(f"perspective image (u, v), f = {_f:.2f}")

    mo.vstack([
        mo.md(f"depth = **{_zc:.1f}**, focal length = **{_f:.2f}** — increasing depth now shrinks the image on the right"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A 3D line is naturally a **point + direction** pair,
      $p(t) = p_0 + t d$; a 3D **plane** is the direct analogue of a 2D
      line, a 4-vector $\pi = (a, b, c, d)$ with $\pi \cdot \tilde{x} = 0$
      for points on the plane, and a normal form $n \cdot x = \delta$.
    - Any 3D rotation can be built from three **Euler angles** (roll,
      pitch, yaw) via $R = R_z(\gamma) R_y(\beta) R_x(\alpha)$ — order
      matters, and this convention can hit **gimbal lock** at
      $\beta = \pm90°$.
    - 3D transformations are $4\times4$ matrices with a fixed bottom row
      $(0,0,0,1)$: translation (3 DOF) ⊂ Euclidean (6 DOF) ⊂ similarity
      (7 DOF) ⊂ affine (12 DOF) — the same nesting pattern as in 2D, one
      dimension up.
    - A camera turns 3D into 2D by projection: **orthographic** projection
      drops depth entirely (parallel lines of sight, size is
      depth-independent), while **perspective** projection divides by
      depth through a single center of projection (converging lines of
      sight, giving foreshortening — objects farther away look smaller).

    Next up: putting all of this to work on actual images — pixels,
    intensity, and how images are represented as arrays of numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
