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

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 2D Geometry and Transformations in Homogeneous Coordinates

    In Notebook 1 we rotated and translated points using two separate
    operations: a matrix multiplication for rotation, and a vector addition
    for translation. **Homogeneous coordinates** are a small trick that lets
    us fold *both* of those into a single matrix multiplication — and, as a
    bonus, let us represent points and lines with the same kind of object.

    This notebook covers:

    1. **2D points** in homogeneous coordinates
    2. **2D lines** in homogeneous coordinates — both the standard
       $(a, b, c)$ form and a normalized form using a unit normal and a
       distance from the origin
    3. **2D transformations** as $3 \times 3$ matrices: translation,
       Euclidean (rigid body), similarity, and affine — applied to the same
       block "S" shape from Notebook 1
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 2D points in homogeneous coordinates

    An ordinary 2D point $(x, y)$ is written in homogeneous coordinates by
    appending a $1$:

    $$
    x = \begin{bmatrix} x \\ y \end{bmatrix}
    \quad\longrightarrow\quad
    \tilde{x} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
    $$

    More generally, *any* triple $(x, y, w)$ with $w \neq 0$ represents the
    Euclidean point $(x/w,\ y/w)$. Points are only defined **up to scale**:
    $(x, y, w)$ and $(kx, ky, kw)$ describe the same point for any $k \neq 0$.
    """)
    return


@app.cell
def _(np):
    pt_2d = np.array([3.0, 5.0])
    pt_h = np.array([pt_2d[0], pt_2d[1], 1.0])   # append a 1
    pt_h
    return (pt_h,)


@app.cell
def _(mo, pt_h):
    _pt_h_scaled = 2.5 * pt_h              # same point, different scale
    _pt_recovered = _pt_h_scaled[:2] / _pt_h_scaled[2]
    mo.md(f"""
    `pt_h` is `{pt_h}`. Scaling it by `2.5` gives `{_pt_h_scaled}` — dividing
    the first two entries by the third recovers the same Euclidean point:
    `{_pt_recovered}`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 2D lines in homogeneous coordinates

    ### 2.1 The standard representation

    A line in the plane can always be written implicitly as

    $$
    a x + b y + c = 0
    $$

    Collecting the coefficients into a 3-vector $l = (a, b, c)$ gives the
    line's homogeneous coordinates. A homogeneous point
    $\tilde{x} = (x, y, 1)$ lies **on** the line exactly when

    $$
    l \cdot \tilde{x} = 0
    $$

    Points and lines are both just 3-vectors, and "on the line" is a single
    dot product — a nice duality. Like points, lines are only defined up to
    scale: $l$ and $k l$ describe the same line for any $k \neq 0$.

    A convenient consequence: the line through two homogeneous points
    $\tilde{x}_1$ and $\tilde{x}_2$ is their **cross product**,
    $l = \tilde{x}_1 \times \tilde{x}_2$ (and, dually, the intersection point
    of two lines is the cross product of the lines). Let's check that the
    line built this way really does pass through both points:
    """)
    return


@app.cell
def _(mo, np):
    _pt1_h = np.array([1.0, 1.0, 1.0])
    _pt2_h = np.array([4.0, 3.0, 1.0])
    _line_l = np.cross(_pt1_h, _pt2_h)
    mo.md(f"""
    `l = pt1 × pt2 = {_line_l}`, so the line is
    `{_line_l[0]:.1f}x + {_line_l[1]:.1f}y + {_line_l[2]:.1f} = 0`.

    Checking `l · pt1 = {_line_l @ _pt1_h:.1f}` and
    `l · pt2 = {_line_l @ _pt2_h:.1f}` — both zero, as expected, so the line
    passes through both points.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 The normalized representation: normal vector + distance

    Divide $l = (a, b, c)$ through by $\|(a, b)\|$ and the line equation
    becomes

    $$
    n_x x + n_y y = d,
    \qquad n = (n_x, n_y),\ \ \|n\| = 1
    $$

    Geometrically, $n$ is the **unit normal** to the line, and $d$ is the
    **perpendicular distance from the origin to the line** (choosing the sign
    of the normalization so $d \geq 0$, i.e. $n$ points from the origin
    *toward* the line).

    Writing the normal by its angle, $n = (\cos\theta, \sin\theta)$, gives

    $$
    x\cos\theta + y\sin\theta = d
    $$

    This $(\theta, d)$ parameterization is exactly the one used by the
    **Hough transform** for line detection later in the course. Try it below
    — the point on the line closest to the origin sits at distance $d$ along
    direction $\theta$, i.e. at $d \cdot (\cos\theta, \sin\theta)$.
    """)
    return


@app.cell
def _(mo):
    line_theta_slider = mo.ui.slider(
        start=0, stop=360, value=40, step=5, label="Normal angle θ (degrees)"
    )
    line_d_slider = mo.ui.slider(
        start=0, stop=8, value=3, step=0.5, label="Distance from origin d"
    )
    mo.hstack([line_theta_slider, line_d_slider], justify="start", gap=2)
    return line_d_slider, line_theta_slider


@app.cell
def _(line_d_slider, line_theta_slider, mo, np, plt):
    _theta_rad = np.radians(line_theta_slider.value)
    _n = np.array([np.cos(_theta_rad), np.sin(_theta_rad)])
    _d = line_d_slider.value
    _a, _b, _c = _n[0], _n[1], -_d

    _foot = _d * _n                              # point on the line closest to the origin
    _dir = np.array([-np.sin(_theta_rad), np.cos(_theta_rad)])  # direction along the line
    _p_start = _foot - 10 * _dir
    _p_end = _foot + 10 * _dir

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.plot([_p_start[0], _p_end[0]], [_p_start[1], _p_end[1]], color="tab:blue", label="line")
    _ax.plot([0, _foot[0]], [0, _foot[1]], "--", color="tab:red", linewidth=1.5)
    _ax.scatter(*_foot, color="tab:red", zorder=3, label="closest point (distance d)")
    _ax.scatter(0, 0, color="black", zorder=3)
    _ax.annotate(
        f"d = {_d:.1f}", xy=_foot / 2, xytext=(5, 5), textcoords="offset points",
        color="tab:red",
    )
    _lim = 10
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left")
    _ax.set_title(f"x cos θ + y sin θ = d,   θ = {line_theta_slider.value}°, d = {_d:.1f}")

    _readout = mo.md(f"""
    - θ = **{line_theta_slider.value}°**
    - d = **{_d:.1f}**
    - unit normal n = **({_n[0]:.2f}, {_n[1]:.2f})**
    - line coefficients (a, b, c) = **({_a:.2f}, {_b:.2f}, {_c:.2f})**
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 2D transformations as $3\times 3$ matrices

    Every transformation below has the same template: a $3\times3$ matrix
    applied to a homogeneous point.

    $$
    T = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad
    \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = T \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
    $$

    Because the bottom row is fixed at $(0, 0, 1)$, the third coordinate of
    the result is always $1$ again. That's the trick: it lets translation
    "ride along" inside an ordinary matrix multiplication, instead of needing
    a separate vector addition like in Notebook 1 ($p' = R(\theta) p + t$).

    Different transformations correspond to different *constraints* on the
    matrix — the fewer constraints, the more degrees of freedom (DOF), and
    the more the shape is allowed to distort:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Transformation | Linear part ($a_{11}, a_{12}, a_{21}, a_{22}$) | DOF | Free parameters | Preserves |
    |---|---|---|---|---|
    | **Translation** | identity | **2** | $t_x, t_y$ | lengths, angles, area, orientation |
    | **Euclidean (rigid body)** | rotation $R(\theta)$ | **3** | $\theta, t_x, t_y$ | lengths, angles, area |
    | **Similarity** | scaled rotation $s\,R(\theta)$ | **4** | $\theta, s, t_x, t_y$ | angles, shape (length ratios) |
    | **Affine** | any invertible $2\times2$ matrix | **6** | $a_{11}, a_{12}, a_{21}, a_{22}, t_x, t_y$ | parallelism, length ratios along a line |

    Each row only adds constraints on top of the previous one: translation is
    a special case of Euclidean (with $\theta = 0$), Euclidean is a special
    case of similarity (with $s = 1$), and similarity is a special case of
    affine (with the linear part restricted to a scaled rotation).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We'll apply each transformation to the same block "S" shape from
    Notebook 1, now expressed in homogeneous coordinates as a
    $3 \times N$ matrix (one homogeneous point per column).
    """)
    return


@app.cell
def _(np):
    # Same block "S" outline as Notebook 1, as a 2 x N matrix of (x, y) points.
    ncsu_shape_local = np.array([
        (-2, 3.5), (2, 3.5), (2, 2.5), (-1, 2.5), (-1, 0.5), (2, 0.5),
        (2, -3.5), (-2, -3.5), (-2, -2.5), (1, -2.5), (1, -0.5), (-2, -0.5),
    ]).T
    # Homogeneous version: append a row of 1's -> shape (3, N).
    ncsu_shape_h = np.vstack([ncsu_shape_local, np.ones(ncsu_shape_local.shape[1])])
    return ncsu_shape_h, ncsu_shape_local


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.1 Translation — 2 degrees of freedom

    $$
    T_{\text{translation}}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}
    $$

    Every point moves by the same $(t_x, t_y)$ offset — the shape doesn't
    rotate, scale, or deform.
    """)
    return


@app.cell
def _(mo):
    tx_t_slider = mo.ui.slider(start=-5, stop=5, value=3, step=0.5, label="tx")
    ty_t_slider = mo.ui.slider(start=-5, stop=5, value=-2, step=0.5, label="ty")
    mo.hstack([tx_t_slider, ty_t_slider], justify="start", gap=2)
    return tx_t_slider, ty_t_slider


@app.cell
def _(mo, ncsu_shape_h, ncsu_shape_local, np, plt, tx_t_slider, ty_t_slider):
    _T = np.array([
        [1, 0, tx_t_slider.value],
        [0, 1, ty_t_slider.value],
        [0, 0, 1],
    ])
    _shape_h = _T @ ncsu_shape_h
    _shape_transformed = _shape_h[:2] / _shape_h[2]   # dehomogenize (w stays 1 here)

    _orig_x = np.append(ncsu_shape_local[0], ncsu_shape_local[0, 0])
    _orig_y = np.append(ncsu_shape_local[1], ncsu_shape_local[1, 0])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.plot(_orig_x, _orig_y, "--", color="gray", label="original")
    _ax.fill(_shape_transformed[0], _shape_transformed[1],
             color="#CC0000", alpha=0.85, edgecolor="black", linewidth=1, label="translated")
    _lim = 10
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left")
    _ax.set_title(f"tx = {tx_t_slider.value}, ty = {ty_t_slider.value}")
    mo.vstack([mo.md(f"tx = **{tx_t_slider.value}**, ty = **{ty_t_slider.value}**"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.2 Euclidean (rigid body) — 3 degrees of freedom

    $$
    T_{\text{Euclidean}}(\theta, t_x, t_y) = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}
    $$

    Adds a rotation angle $\theta$ on top of translation. Distances and
    angles between points are preserved — this is a *rigid* motion, just
    like Section 5 of Notebook 1, now with the rotation and translation
    combined into one matrix.
    """)
    return


@app.cell
def _(mo):
    theta_e_slider = mo.ui.slider(start=-180, stop=180, value=25, step=5, label="θ (degrees)")
    tx_e_slider = mo.ui.slider(start=-5, stop=5, value=-2, step=0.5, label="tx")
    ty_e_slider = mo.ui.slider(start=-5, stop=5, value=3, step=0.5, label="ty")
    mo.hstack([theta_e_slider, tx_e_slider, ty_e_slider], justify="start", gap=2)
    return theta_e_slider, tx_e_slider, ty_e_slider


@app.cell
def _(
    mo,
    ncsu_shape_h,
    ncsu_shape_local,
    np,
    plt,
    theta_e_slider,
    tx_e_slider,
    ty_e_slider,
):
    _theta_rad = np.radians(theta_e_slider.value)
    _T = np.array([
        [np.cos(_theta_rad), -np.sin(_theta_rad), tx_e_slider.value],
        [np.sin(_theta_rad),  np.cos(_theta_rad), ty_e_slider.value],
        [0, 0, 1],
    ])
    _shape_h = _T @ ncsu_shape_h
    _shape_transformed = _shape_h[:2] / _shape_h[2]

    _orig_x = np.append(ncsu_shape_local[0], ncsu_shape_local[0, 0])
    _orig_y = np.append(ncsu_shape_local[1], ncsu_shape_local[1, 0])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.plot(_orig_x, _orig_y, "--", color="gray", label="original")
    _ax.fill(_shape_transformed[0], _shape_transformed[1],
             color="#CC0000", alpha=0.85, edgecolor="black", linewidth=1, label="rotated + translated")
    _lim = 10
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left")
    _ax.set_title(f"θ = {theta_e_slider.value}°, tx = {tx_e_slider.value}, ty = {ty_e_slider.value}")
    mo.vstack([
        mo.md(f"θ = **{theta_e_slider.value}°**, tx = **{tx_e_slider.value}**, ty = **{ty_e_slider.value}**"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.3 Similarity — 4 degrees of freedom

    $$
    T_{\text{similarity}}(\theta, s, t_x, t_y) = \begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}
    $$

    Adds one more parameter — a uniform scale $s > 0$ — on top of the rigid
    motion. Angles are still preserved, but lengths are all scaled by the
    same factor $s$ (so area scales by $s^2$); the shape keeps its
    proportions.
    """)
    return


@app.cell
def _(mo):
    theta_s_slider = mo.ui.slider(start=-180, stop=180, value=-30, step=5, label="θ (degrees)")
    scale_s_slider = mo.ui.slider(start=0.3, stop=2.0, value=1.4, step=0.1, label="scale s")
    tx_s_slider = mo.ui.slider(start=-5, stop=5, value=3, step=0.5, label="tx")
    ty_s_slider = mo.ui.slider(start=-5, stop=5, value=2, step=0.5, label="ty")
    mo.hstack([theta_s_slider, scale_s_slider, tx_s_slider, ty_s_slider], justify="start", gap=2)
    return scale_s_slider, theta_s_slider, tx_s_slider, ty_s_slider


@app.cell
def _(
    mo,
    ncsu_shape_h,
    ncsu_shape_local,
    np,
    plt,
    scale_s_slider,
    theta_s_slider,
    tx_s_slider,
    ty_s_slider,
):
    _theta_rad = np.radians(theta_s_slider.value)
    _s = scale_s_slider.value
    _T = np.array([
        [_s * np.cos(_theta_rad), -_s * np.sin(_theta_rad), tx_s_slider.value],
        [_s * np.sin(_theta_rad),  _s * np.cos(_theta_rad), ty_s_slider.value],
        [0, 0, 1],
    ])
    _shape_h = _T @ ncsu_shape_h
    _shape_transformed = _shape_h[:2] / _shape_h[2]

    _orig_x = np.append(ncsu_shape_local[0], ncsu_shape_local[0, 0])
    _orig_y = np.append(ncsu_shape_local[1], ncsu_shape_local[1, 0])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.plot(_orig_x, _orig_y, "--", color="gray", label="original")
    _ax.fill(_shape_transformed[0], _shape_transformed[1],
             color="#CC0000", alpha=0.85, edgecolor="black", linewidth=1, label="rotated + scaled + translated")
    _lim = 16
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left")
    _ax.set_title(
        f"θ = {theta_s_slider.value}°, s = {_s:.1f}, "
        f"tx = {tx_s_slider.value}, ty = {ty_s_slider.value}"
    )
    mo.vstack([
        mo.md(
            f"θ = **{theta_s_slider.value}°**, s = **{_s:.1f}**, "
            f"tx = **{tx_s_slider.value}**, ty = **{ty_s_slider.value}**"
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.4 Affine — 6 degrees of freedom

    $$
    T_{\text{affine}} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix}
    $$

    Now all four entries of the linear part are free — not just a rotation
    and a scale — so the shape can also **shear** and be **stretched
    differently along different directions**. Parallel lines stay parallel,
    but angles and lengths are no longer preserved in general.

    Six free numbers is a lot for sliders, so instead click the button below
    to draw all six matrix entries at random from $[-2, 2]$.
    """)
    return


@app.cell
def _(mo):
    affine_button = mo.ui.button(
        label="Randomize affine matrix (entries in [-2, 2])",
        value=0,
        on_click=lambda count: count + 1,
    )
    affine_button
    return (affine_button,)


@app.cell
def _(affine_button, mo, ncsu_shape_h, ncsu_shape_local, np, plt):
    _rng = np.random.default_rng(seed=affine_button.value)
    _M = _rng.uniform(-2, 2, size=(2, 3))          # [[a11, a12, tx], [a21, a22, ty]]
    _T = np.vstack([_M, [0, 0, 1]])
    _shape_h = _T @ ncsu_shape_h
    _shape_transformed = _shape_h[:2] / _shape_h[2]

    _orig_x = np.append(ncsu_shape_local[0], ncsu_shape_local[0, 0])
    _orig_y = np.append(ncsu_shape_local[1], ncsu_shape_local[1, 0])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.plot(_orig_x, _orig_y, "--", color="gray", label="original")
    _ax.fill(_shape_transformed[0], _shape_transformed[1],
             color="#CC0000", alpha=0.85, edgecolor="black", linewidth=1, label="affine")
    _lim = 15
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.grid(True, linewidth=0.3)
    _ax.legend(loc="upper left")
    _ax.set_title(f"random affine matrix (click count: {affine_button.value})")

    _matrix_text = "\n".join(
        "  ".join(f"{value:6.2f}" for value in row) for row in _T
    )
    _readout = mo.md(f"""
    ```
    {_matrix_text}
    ```
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A 2D point becomes a homogeneous point by appending a $1$:
      $(x, y) \to (x, y, 1)$, and both points and homogeneous coordinates in
      general are only defined up to scale.
    - A 2D line is also a 3-vector $l = (a, b, c)$, with $l \cdot \tilde{x} = 0$
      for every point $\tilde{x}$ on the line — points and lines share the
      same representation. Normalizing $l$ gives a unit normal $n$ and a
      distance $d$ from the origin, the $(\theta, d)$ form used by the Hough
      transform.
    - Every 2D transformation we care about is a $3\times3$ matrix applied to
      a homogeneous point, with the bottom row fixed at $(0, 0, 1)$:
      translation (2 DOF) ⊂ Euclidean (3 DOF) ⊂ similarity (4 DOF) ⊂ affine
      (6 DOF) — each one relaxing a constraint on the matrix and preserving
      less geometric structure than the last.

    Next up: putting these ideas to work on actual images.
    """)
    return


if __name__ == "__main__":
    app.run()
