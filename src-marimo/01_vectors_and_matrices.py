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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vectors and Matrices with NumPy

    Before we can talk about images, filters, or transformations, we need a
    common language for describing collections of numbers: **vectors** and
    **matrices**. In image processing and computer vision, this is the
    language everything else is built on — an image is a matrix, a pixel's
    color is a vector, and most operations we apply (blurring, rotating,
    warping) are just matrix operations in disguise.

    This notebook is a quick, hands-on review of:

    1. How vectors and matrices are represented in NumPy
    2. How matrix-vector multiplication works, step by step
    3. A common pitfall: element-wise `*` vs. matrix multiplication `@`
    4. A geometric picture of what a matrix "does" to a vector
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Vectors as NumPy arrays

    A vector is just an ordered list of numbers. In NumPy, we represent a
    vector with a 1-D `array`. For example, the vector

    $$
    v = \begin{bmatrix} 3 \\ 5 \end{bmatrix}
    $$

    is written in code as `np.array([3, 5])`. NumPy doesn't distinguish
    between "row" and "column" vectors for a 1-D array — it's just a list
    of numbers with a shape `(2,)`.
    """)
    return


@app.cell
def _(np):
    v = np.array([3, 5])
    v
    return (v,)


@app.cell
def _(mo, v):
    mo.md(f"""
    `v` has shape `{v.shape}`, meaning it's a 1-D array with `{v.shape[0]}`
    entries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Matrices as NumPy arrays

    A matrix is a rectangular grid of numbers — in NumPy, a 2-D array. The
    matrix

    $$
    A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}
    $$

    is written as a list of *rows*: `np.array([[2, 1], [0, 3]])`. The first
    axis (`axis=0`) indexes rows, the second (`axis=1`) indexes columns.
    """)
    return


@app.cell
def _(np):
    A = np.array([[2, 1],
                  [0, 3]])
    A
    return (A,)


@app.cell
def _(A, mo):
    mo.md(f"""
    `A` has shape `{A.shape}`: `{A.shape[0]}` rows and `{A.shape[1]}`
    columns.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Matrix-vector multiplication

    To multiply a matrix $A$ ($m \times n$) by a vector $v$ (length $n$),
    each entry of the result is the **dot product of a row of $A$ with
    $v$**. The output is a new vector of length $m$.

    For our $2\times2$ example:

    $$
    Av =
    \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}
    \begin{bmatrix} 3 \\ 5 \end{bmatrix}
    =
    \begin{bmatrix} 2\times 3 + 1\times 5 \\ 0\times 3 + 3\times 5 \end{bmatrix}
    =
    \begin{bmatrix} 11 \\ 15 \end{bmatrix}
    $$

    Each **row** of $A$ contributes one number to the output: row $i$ says
    "take this weighted combination of the entries of $v$".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Doing it by hand first

    Before letting NumPy do the work, it's worth writing the loop out
    explicitly once, so the "row dot vector" pattern is clear:
    """)
    return


@app.cell
def _(A, np, v):
    # Manual matrix-vector multiplication: one dot product per row of A.
    result_manual = np.zeros(A.shape[0])
    for i in range(A.shape[0]):          # for each row of A...
        row_sum = 0
        for j in range(A.shape[1]):      # ...combine it with every entry of v
            row_sum += A[i, j] * v[j]
        result_manual[i] = row_sum
    result_manual
    return (result_manual,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The NumPy way

    In practice we never write that loop by hand — NumPy gives us three
    equivalent ways to do it, all much faster than a Python loop:

    - `A @ v` — the `@` operator is Python's dedicated matrix-multiplication
      operator (recommended, most readable)
    - `np.dot(A, v)` — the general-purpose dot-product function
    - `np.matmul(A, v)` — the explicit matrix-multiply function (`@` calls
      this under the hood)
    """)
    return


@app.cell
def _(A, np, v):
    result_at = A @ v
    result_dot = np.dot(A, v)
    result_matmul = np.matmul(A, v)
    result_at, result_dot, result_matmul
    return (result_at,)


@app.cell
def _(mo, result_at, result_manual):
    mo.md(f"""
    Manual loop gives `{result_manual}`, NumPy's `@` gives `{result_at}` —
    they match, as expected: **{bool((result_manual == result_at).all())}**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. A common pitfall: `*` is *not* matrix multiplication

    On plain NumPy arrays, the `*` operator performs **element-wise**
    multiplication, not matrix multiplication. This trips up almost
    everyone the first time. Compare:
    """)
    return


@app.cell
def _(A):
    elementwise = A * A   # multiplies each entry by itself: A[i, j] * A[i, j]
    matrix_product = A @ A  # true matrix product: row·column dot products
    elementwise, matrix_product
    return


@app.cell
def _(mo):
    mo.md(r"""
    `A * A` squares every entry in place, while `A @ A` recombines rows and
    columns via dot products — two very different operations that happen
    to look similar. **Rule of thumb:** use `@` (or `np.matmul`/`np.dot`)
    whenever you mean "matrix multiplication" in the linear-algebra sense.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. What a matrix *does* to a vector

    Geometrically, multiplying a vector by a matrix moves it: it can
    rotate, stretch, shear, or flip the vector. This is exactly the idea
    behind image transformations you'll see later in the course (rotating
    an image, resizing it, correcting perspective, ...) — all of those are
    matrices applied to the coordinates of every pixel.

    Try the slider below: it builds a **rotation matrix**

    $$
    R(\theta) =
    \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
    $$

    and applies it to our vector $v$. Because this is a Marimo notebook,
    moving the slider automatically re-runs the cells that depend on it —
    no need to manually re-run anything.
    """)
    return


@app.cell
def _(mo, np, v):
    lim = float(np.abs(v).max()) + 2

    angle_slider = mo.ui.slider(
        start=-180, stop=180, value=30, step=5, label="Rotation angle θ (degrees)"
    )
    angle_slider
    return angle_slider, lim


@app.cell
def _(angle_slider, lim, np, plt, v):
    theta = np.radians(angle_slider.value)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    v_rotated = R @ v  # apply the rotation matrix to v, same operation as section 3

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
              color="tab:blue", label="v (original)")
    ax.quiver(0, 0, v_rotated[0], v_rotated[1], angles="xy", scale_units="xy", scale=1,
              color="tab:red", label="R @ v (rotated)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper left")
    ax.set_title(f"Rotating v by {angle_slider.value}°")
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Combining rotation and translation: moving a shape

    A single point was a useful example, but in practice we usually transform
    a whole **shape** made of many points at once. Below, a simple block "S"
    (evoking NC State's Block S mark) is stored as a $2 \times N$ matrix
    `ncsu_shape_local`, where **each column is one $(x, y)$ point** of the
    outline.

    - **Rotation** by an angle $\theta$ uses the exact same rotation matrix
      $R(\theta)$ from Section 5. Computing `R_shape @ ncsu_shape_local`
      rotates *every column (point) at once* — the same "row · column"
      pattern from Section 3, just repeated across many vectors instead of
      one. Rotation always happens **about the origin** $(0, 0)$, and it
      preserves the shape's size and angles (a *rigid* transformation).
    - **Translation** by a vector $t = (t_x, t_y)$ simply adds a constant
      offset to every point — no matrix multiplication involved, just vector
      addition. It slides the shape around without rotating, scaling, or
      distorting it.

    Putting the two together, every point $p$ of the shape maps to:

    $$
    p' = R(\theta)\, p + t
    $$

    **Order matters:** we rotate first (about the origin) and translate
    second. Rotating *after* translating would spin the shape around the
    *origin*, not around wherever it had just been moved to — swapping the
    order changes the result.

    Use the sliders below to rotate and translate the shape and see this in
    action.
    """)
    return


@app.cell
def _(np):
    # A simplified block "S" outline (evoking NC State's Block S mark),
    # stored as a 2 x N matrix: each column is one (x, y) point, centered on
    # the origin so rotation happens "in place" before any translation.

    ncsu_shape_local = np.array([
        [   0,  336], [   0,  796], [ 578,  796], [ 579,  481],
        [1246,  482], [1245,  891], [ 294,  891], [   0, 1174],
        [   0, 1853], [ 323, 2189], [1479, 2189], [1802, 1853],
        [1802, 1413], [1239, 1413], [1238, 1697], [ 586, 1696],
        [ 587, 1332], [1238, 1332], [1239, 1352], [1530, 1352],
        [1813, 1008], [1813,  324], [1489,    0], [ 323,    0],
    ]).T/500
    return (ncsu_shape_local,)


@app.cell
def _(mo):
    rotation_slider = mo.ui.slider(
        start=-180, stop=180, value=25, step=5, label="Rotation angle θ (degrees)"
    )
    tx_slider = mo.ui.slider(start=-5, stop=5, value=3, step=0.5, label="Translation tx")
    ty_slider = mo.ui.slider(start=-5, stop=5, value=-2, step=0.5, label="Translation ty")
    mo.hstack([rotation_slider, tx_slider, ty_slider], justify="start", gap=2)
    return rotation_slider, tx_slider, ty_slider


@app.cell
def _(ncsu_shape_local, np, plt, rotation_slider, tx_slider, ty_slider):
    shape_theta = np.radians(rotation_slider.value)
    R_shape = np.array([
        [np.cos(shape_theta), -np.sin(shape_theta)],
        [np.sin(shape_theta),  np.cos(shape_theta)],
    ])
    t_vec = np.array([[tx_slider.value], [ty_slider.value]])

    # Rotate every point (column) at once, then add the translation to every point.
    ncsu_shape_transformed = R_shape @ ncsu_shape_local + t_vec

    # Repeat the first point at the end so the outline draws as a closed loop.
    original_x = np.append(ncsu_shape_local[0], ncsu_shape_local[0, 0])
    original_y = np.append(ncsu_shape_local[1], ncsu_shape_local[1, 0])

    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.axvline(0, color="gray", linewidth=0.5)
    ax3.plot(original_x, original_y, "--", color="gray", label="original (local coords)")
    ax3.fill(
        ncsu_shape_transformed[0], ncsu_shape_transformed[1],
        color="#CC0000", alpha=0.85, edgecolor="black", linewidth=1,
        label="rotated + translated",
    )
    _lim = 10
    ax3.set_xlim(-_lim, _lim)
    ax3.set_ylim(-_lim, _lim)
    ax3.set_aspect("equal")
    ax3.grid(True, linewidth=0.3)
    ax3.legend(loc="upper left")
    ax3.set_title(
        f"θ = {rotation_slider.value}°, "
        f"t = ({tx_slider.value}, {ty_slider.value})"
    )
    fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A **vector** is a 1-D NumPy array; a **matrix** is a 2-D NumPy array.
    - Matrix-vector multiplication `A @ v` combines each row of `A` with
      `v` via a dot product, producing one output number per row.
    - Use `@` / `np.matmul` / `np.dot` for matrix multiplication —
      plain `*` is element-wise and means something different.
    - Geometrically, matrices transform vectors (rotate, scale, shear).
      This same idea will come back when we transform whole images.

    Next up: representing an **image** as a matrix of pixel values.
    """)
    return


if __name__ == "__main__":
    app.run()
