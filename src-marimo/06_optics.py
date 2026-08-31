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
    from matplotlib.patches import Arc

    return Arc, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Optics: How a Lens Focuses Light onto a Sensor

    Notebook 5 followed the physics of light from a source, off a surface,
    and out into the world as radiance $L(\hat v;\lambda)$. But a camera
    doesn't just catch that radiance in mid-air — it has to **focus** it
    onto a sensor, or every pixel would just see a blur of light arriving
    from every direction at once. That's the job of a **lens**, and lenses
    work because light bends when it crosses between materials.

    This notebook builds geometric optics from scratch — no prior optics
    background assumed — following the chain:

    **refraction (Snell's law) → total internal reflection → dispersion
    (why prisms make rainbows) → how a curved piece of glass focuses light
    → the thin-lens model, focal ratio, and numerical aperture.**

    Two sections below use the **Predict → Run → Investigate** format:
    make a prediction, check it against a hidden explanation, then play
    with the interactive demo and dig further.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. What is light, and the index of refraction $n$

    Light is an electromagnetic wave that travels at a fixed speed in
    vacuum, $c \approx 3\times10^8\,\text{m/s}$. Inside a transparent
    material, it travels *slower*, at some speed $v \le c$. The
    **index of refraction** of that material is defined as the ratio

    $$ n = \frac{c}{v} \;\ge\; 1 $$

    A bigger $n$ means light moves through that material more slowly.
    Some typical values:

    | Medium | $n$ |
    |---|---|
    | Vacuum | 1.0000 |
    | Air (at sea level) | 1.0003 |
    | Water | 1.33 |
    | Crown glass | 1.52 |
    | Diamond | 2.42 |

    **A subtlety that matters later:** a light wave's *frequency* $f$ is
    set by its source and never changes as it crosses into a new medium —
    what changes is its speed, and therefore its **wavelength**
    $\lambda = v/f$. Since $v=c/n$, the wavelength inside a medium shrinks
    to $\lambda_{\text{medium}} = \lambda_{\text{vacuum}}/n$. We'll always
    quote wavelength as its *vacuum* value (e.g. "550 nm" for green light),
    which is also what determines a photon's color and energy.

    One more honest caveat: whenever light hits an interface between two
    media, it doesn't *all* transmit through — some always reflects too
    (governed by the Fresnel equations, which we won't derive here). For
    most of this notebook we'll track only the transmitted (refracted)
    ray, and bring the reflected ray back explicitly once it becomes the
    *only* possible outcome — total internal reflection, in §4.
    """)
    return


@app.cell
def _(np):
    def refract2d(d, n_hat, n1, n2):
        """Vector form of Snell's law in 2D. `d` is the unit incident
        propagation direction, `n_hat` the interface normal (either
        orientation — it's auto-flipped to oppose `d`). `n1` is the index
        of the medium `d` is leaving, `n2` the index it's entering.
        Returns the unit refracted direction, or None on total internal
        reflection."""
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        n_hat = np.asarray(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
        if np.dot(d, n_hat) > 0:
            n_hat = -n_hat
        cos_i = -np.dot(d, n_hat)
        r = n1 / n2
        sin2_t = r ** 2 * (1 - cos_i ** 2)
        if sin2_t > 1.0:
            return None
        cos_t = np.sqrt(1 - sin2_t)
        return r * d + (r * cos_i - cos_t) * n_hat

    def reflect2d(d, n_hat):
        """Law of reflection in 2D, same conventions as refract2d."""
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        n_hat = np.asarray(n_hat, dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
        if np.dot(d, n_hat) > 0:
            n_hat = -n_hat
        return d - 2 * np.dot(d, n_hat) * n_hat

    def snell_refracted_angle(n1, n2, theta1_deg):
        """Scalar Snell's law: n1 sin(theta1) = n2 sin(theta2). Returns
        theta2 in degrees, or None on total internal reflection."""
        s2 = n1 / n2 * np.sin(np.radians(theta1_deg))
        if abs(s2) > 1.0:
            return None
        return np.degrees(np.arcsin(s2))

    def ray_circle_hit(p, d, R, skip_t=1e-9):
        """Smallest t > skip_t such that |p + t*d| = R, or None if the ray
        (from point p, unit direction d) misses the circle of radius R
        centered at the origin."""
        p = np.asarray(p, dtype=float)
        d = np.asarray(d, dtype=float)
        a = np.dot(d, d)
        b = 2 * np.dot(p, d)
        c = np.dot(p, p) - R ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        candidates = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t > skip_t]
        return min(candidates) if candidates else None

    return ray_circle_hit, reflect2d, refract2d, snell_refracted_angle


@app.cell
def _(Arc, np):
    def draw_arrow(ax, start, end, color, lw=2.2, zorder=3):
        """Draw a single straight arrow from `start` to `end`. Clipped to
        the axes box rather than skipped outright when `end` (or the whole
        segment) falls outside the current view — matplotlib's default
        annotation clipping hides the *entire* arrow once its target point
        is off-screen, so a ray that exits the plotted region would
        otherwise just vanish instead of showing as a partial line."""
        ann = ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16),
            zorder=zorder, annotation_clip=False,
        )
        ann.arrow_patch.set_clip_path(ax.patch)
        ann.arrow_patch.set_clip_box(ax.bbox)

    def draw_angle_arc(ax, vertex, dir1, dir2, radius, color, label, label_scale=1.4, fontsize=11):
        """Draw a small arc between two directions (unit vectors) anchored
        at `vertex`, with a text label placed along the bisector — the 2D
        analogue of the theta-arcs used in notebook 5's 3D BRDF diagram."""
        a1 = np.degrees(np.arctan2(dir1[1], dir1[0])) % 360
        a2 = np.degrees(np.arctan2(dir2[1], dir2[0])) % 360
        diff = (a2 - a1) % 360
        if diff <= 180:
            t1, t2 = a1, a1 + diff
        else:
            diff = (a1 - a2) % 360
            t1, t2 = a2, a2 + diff
        arc = Arc(vertex, 2 * radius, 2 * radius, angle=0, theta1=t1, theta2=t2, color=color, lw=1.5, zorder=2)
        ax.add_patch(arc)
        mid = np.radians(t1 + diff / 2)
        label_pos = np.asarray(vertex, dtype=float) + radius * label_scale * np.array([np.cos(mid), np.sin(mid)])
        ax.text(*label_pos, label, color=color, ha="center", va="center", fontsize=fontsize)

    return draw_angle_arc, draw_arrow


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Refraction and Snell's law

    When a ray crosses from a medium of index $n_1$ into a medium of
    index $n_2$, it bends. The rule governing the bend is **Snell's law**:

    $$ n_1 \sin\theta_1 = n_2 \sin\theta_2 $$

    where $\theta_1,\theta_2$ are the angles the incident and refracted
    rays make with the **normal** (the line perpendicular to the
    interface) — *not* with the interface itself.

    **Why does it bend?** Think of the wave as a series of wavefronts
    (like ripples) marching toward the interface. If the wave hits the
    interface at an angle, one edge of the wavefront touches down and
    starts slowing to speed $v_2=c/n_2$ *before* the rest of the
    wavefront (still in medium 1, moving at $v_1=c/n_1$) gets there. If
    $v_2<v_1$ (entering a denser medium), the leading edge is held back
    while the trailing edge catches up — this pivots the whole wavefront,
    and with it the direction of travel, **toward the normal**. Run it in
    reverse (leaving a denser medium) and the ray bends **away from the
    normal**. This geometric argument, made precise, is exactly Snell's
    law — $n_1\sin\theta_1=n_2\sin\theta_2$ is just a restatement of
    "$\sin\theta/v$ is the same on both sides."

    Alongside refraction, the interface always partially **reflects** too,
    obeying the much older law of reflection: $\theta_i=\theta_r$,
    measured from the same normal.

    The figure below fixes $n_1=1.0$ (air), $n_2=1.5$ (glass),
    $\theta_1=40°$ as a worked example.
    """)
    return


@app.cell
def _(draw_angle_arc, draw_arrow, mo, np, plt, snell_refracted_angle):
    _n1, _n2, _theta1 = 1.0, 1.5, 40.0
    _theta2 = snell_refracted_angle(_n1, _n2, _theta1)

    _t1 = np.radians(_theta1)
    _t2 = np.radians(_theta2)
    _d_in = np.array([np.sin(_t1), -np.cos(_t1)])
    _d_out = np.array([np.sin(_t2), -np.cos(_t2)])
    _O = np.array([0.0, 0.0])
    _L = 1.3

    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _ax.axhspan(0, 1.5, color="#eaf4ff", zorder=0)
    _ax.axhspan(-1.5, 0, color="#fff3e0", zorder=0)
    _ax.plot([-1.6, 1.6], [0, 0], color="black", lw=1.5)
    _ax.plot([0, 0], [-1.3, 1.3], color="gray", lw=1, ls="--")
    draw_arrow(_ax, _O - _L * _d_in, _O, "tab:blue")
    draw_arrow(_ax, _O, _O + _L * _d_out, "tab:red")
    draw_angle_arc(_ax, _O, np.array([0.0, 1.0]), -_d_in, 0.35, "tab:blue", "θ₁")
    draw_angle_arc(_ax, _O, np.array([0.0, -1.0]), _d_out, 0.3, "tab:red", "θ₂")
    _ax.text(-1.55, 1.15, f"medium 1  (n₁ = {_n1:.2f})", fontsize=10)
    _ax.text(-1.55, -1.2, f"medium 2  (n₂ = {_n2:.2f})", fontsize=10)
    _ax.set_xlim(-1.6, 1.6)
    _ax.set_ylim(-1.4, 1.4)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(f"θ₁ = {_theta1:.0f}°  →  θ₂ = {_theta2:.1f}°  (bends toward the normal, n₂ > n₁)")

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Interactive demo — a glass slab (air → glass → air)

    Now let's trace a ray through a slab with parallel faces — like a
    window pane or a sheet of glass — going from air, through glass, and
    back into air.

    ### Predict

    Before touching the sliders: once the ray exits the *bottom* of the
    slab, back into the same medium it started in — does it come out
    traveling in a **different direction** than it went in, or does it
    just come out **shifted sideways**, still parallel to the original
    ray? And does raising $n_2$ (a denser slab) bend the ray *more* or
    *less* inside the glass?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        - Because the top and bottom faces are **parallel**, the ray that
          exits the bottom is always **parallel to the original ray** —
          apply Snell's law at the top interface ($n_1\to n_2$) and then
          at the bottom ($n_2\to n_1$) and the two bends exactly undo each
          other. The only lasting effect is a **lateral (sideways) shift**
          $d = \dfrac{t\,\sin(\theta_1-\theta_2)}{\cos\theta_2}$, where $t$
          is the slab thickness.
        - **Raising $n_2$** bends the ray *more* toward the normal inside
          the glass (smaller $\theta_2$), which *increases* the sideways
          shift $d$ for a fixed thickness.
        """)
    })
    return


@app.cell
def _(mo):
    n1_slab_slider = mo.ui.slider(start=1.0, stop=2.0, value=1.0, step=0.05, label="n₁ (medium above)")
    n2_slab_slider = mo.ui.slider(start=1.0, stop=2.5, value=1.5, step=0.05, label="n₂ (slab)")
    theta1_slab_slider = mo.ui.slider(start=0, stop=85, value=40, step=1, label="incidence angle θ₁ (°)")
    t_slab_slider = mo.ui.slider(start=0.3, stop=2.0, value=1.0, step=0.1, label="slab thickness t")
    mo.hstack([n1_slab_slider, n2_slab_slider, theta1_slab_slider, t_slab_slider], justify="start", gap=2)
    return n1_slab_slider, n2_slab_slider, t_slab_slider, theta1_slab_slider


@app.cell
def _(
    draw_angle_arc,
    draw_arrow,
    mo,
    n1_slab_slider,
    n2_slab_slider,
    np,
    plt,
    reflect2d,
    refract2d,
    snell_refracted_angle,
    t_slab_slider,
    theta1_slab_slider,
):
    _n1, _n2 = n1_slab_slider.value, n2_slab_slider.value
    _theta1 = theta1_slab_slider.value
    _t = t_slab_slider.value

    _t1 = np.radians(_theta1)
    _d_in = np.array([np.sin(_t1), -np.cos(_t1)])
    _n_top = np.array([0.0, 1.0])
    _O1 = np.array([0.0, 0.0])
    _L = 1.4

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.axhspan(0, 1.6, color="#eaf4ff", zorder=0)
    _ax.axhspan(-_t, 0, color="#fff3e0", zorder=0)
    _ax.axhspan(-_t - 1.6, -_t, color="#eaf4ff", zorder=0)
    _ax.plot([-1.8, 1.8], [0, 0], color="black", lw=1.5)
    _ax.plot([-1.8, 1.8], [-_t, -_t], color="black", lw=1.5)
    _ax.plot([0, 0], [-_t - 1.0, 1.0], color="gray", lw=0.8, ls="--")

    draw_arrow(_ax, _O1 - _L * _d_in, _O1, "tab:blue")
    draw_angle_arc(_ax, _O1, _n_top, -_d_in, 0.35, "tab:blue", "θ₁")

    _d_glass = refract2d(_d_in, _n_top, _n1, _n2)
    _theta2 = None
    _lateral = None

    if _d_glass is None:
        _d_refl = reflect2d(_d_in, _n_top)
        draw_arrow(_ax, _O1, _O1 + _L * _d_refl, "tab:purple")
        _note = "**Total internal reflection** at the top interface (θ₁ ≥ critical angle) — no ray enters the slab; all the light reflects."
    else:
        _theta2 = snell_refracted_angle(_n1, _n2, _theta1)
        draw_angle_arc(_ax, _O1, -_n_top, _d_glass, 0.3, "tab:red", "θ₂")
        _t_hit = (-_t - _O1[1]) / _d_glass[1]
        _O2 = _O1 + _t_hit * _d_glass
        draw_arrow(_ax, _O1, _O2, "tab:red")
        _ax.plot([_O2[0], _O2[0]], [-_t - 1.0, -_t + 1.0], color="gray", lw=0.8, ls="--")

        _n_bottom = np.array([0.0, 1.0])
        # Mathematically, theta2 <= arcsin(n1/n2) always, which is exactly
        # the critical angle for the reverse (glass -> air) transition, so
        # this branch can't trigger for a parallel-sided slab — kept as a
        # defensive fallback for floating-point edge cases near grazing
        # incidence.
        _d_out = refract2d(_d_glass, _n_bottom, _n2, _n1)
        if _d_out is None:
            _d_refl2 = reflect2d(_d_glass, _n_bottom)
            draw_arrow(_ax, _O2, _O2 + _L * _d_refl2, "tab:purple")
            _note = "**Total internal reflection** at the bottom interface — the ray never re-exits into medium 1."
        else:
            _theta3 = snell_refracted_angle(_n2, _n1, _theta2)
            draw_angle_arc(_ax, _O2, -_n_bottom, _d_out, 0.3, "tab:green", "θ₃")
            draw_arrow(_ax, _O2, _O2 + _L * _d_out, "tab:green")
            _lateral = np.linalg.norm(_O2 - _O1) * np.sin(np.radians(_theta1 - _theta2))
            _note = f"θ₃ = {_theta3:.1f}°  (always equals θ₁ for parallel interfaces — the exit ray is parallel to the entry ray)."

    _ax.set_xlim(-1.8, 1.8)
    _ax.set_ylim(-_t - 1.6, 1.6)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(f"n₁={_n1:.2f}, n₂={_n2:.2f}, θ₁={_theta1}°, t={_t:.1f}")

    _lines = [f"**θ₂ (inside slab):** {'—' if _theta2 is None else f'{_theta2:.1f}°'}"]
    if _lateral is not None:
        _lines.append(f"**lateral shift d:** {_lateral:.3f}  (sign indicates the direction of the shift)")
    _lines.append(_note)

    mo.vstack([mo.md("  \n".join(_lines)), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Set $n_1=n_2$ (e.g. both 1.5). What happens to the bend, and to the
      lateral shift $d$?
    - Keep $n_1,n_2,\theta_1$ fixed and grow the thickness $t$. Does the
      shift $d$ grow linearly with $t$, or some other way?
    - Try to find slider settings where the ray gets permanently **trapped**
      inside the slab (i.e. it never re-exits the bottom face). Can you?
      What does that imply about a window or a windshield — can light that
      makes it in ever get "stuck" bouncing around inside?
    - Push $n_1$ *above* $n_2$ (make the slab the *rarer* medium) and raise
      $\theta_1$. What happens at the *top* interface once $\theta_1$ gets
      large enough? We'll look at this in detail next.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Total internal reflection and the critical angle

    ### Predict

    For light going from water ($n_1=1.33$) into air ($n_2=1.0$): at
    $\theta_1=30°$ incidence, does a refracted ray exist? What about at
    $\theta_1=50°$?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        Set $\theta_2=90°$ (a refracted ray that grazes exactly along the
        interface) in Snell's law and solve for the incidence angle that
        produces it:

        $$
        n_1\sin\theta_1 = n_2\sin 90° = n_2
        \quad\Longrightarrow\quad
        \theta_c = \arcsin\!\left(\frac{n_2}{n_1}\right)
        $$

        (defined only when $n_1>n_2$ — going from a *denser* medium into a
        *rarer* one). For $\theta_1>\theta_c$, Snell's law would demand
        $\sin\theta_2>1$, which is impossible — **no refracted ray can
        exist**, and by conservation of energy *all* the light reflects:
        **total internal reflection (TIR)**.

        For water → air: $\theta_c=\arcsin(1/1.33)\approx 48.8°$. So at
        $30°$ a refracted ray exists (you're below $\theta_c$); at $50°$
        you're past it — total internal reflection, no refracted ray at
        all. This is exactly why, looking straight up from underwater,
        you only see a bright circular "window" to the sky directly above
        you (within $\theta_c$ of straight up) — beyond that cone
        everything is a mirror-like reflection of the water below. It's
        also how optical fibers trap light for kilometers, and how the
        prisms in binoculars redirect light without a mirror coating.
        """)
    })
    return


@app.cell
def _(mo):
    n1_tir_slider = mo.ui.slider(start=1.1, stop=2.5, value=1.33, step=0.01, label="n₁ (denser medium)")
    n2_tir_slider = mo.ui.slider(start=1.0, stop=2.0, value=1.0, step=0.01, label="n₂ (rarer medium)")
    theta1_tir_slider = mo.ui.slider(start=0, stop=89, value=30, step=1, label="incidence angle θ₁ (°)")
    mo.hstack([n1_tir_slider, n2_tir_slider, theta1_tir_slider], justify="start", gap=2)
    return n1_tir_slider, n2_tir_slider, theta1_tir_slider


@app.cell
def _(
    draw_angle_arc,
    draw_arrow,
    mo,
    n1_tir_slider,
    n2_tir_slider,
    np,
    plt,
    reflect2d,
    refract2d,
    snell_refracted_angle,
    theta1_tir_slider,
):
    _n1, _n2, _theta1 = n1_tir_slider.value, n2_tir_slider.value, theta1_tir_slider.value
    _n_hat = np.array([0.0, 1.0])
    _O = np.array([0.0, 0.0])
    _t1 = np.radians(_theta1)
    _d_in = np.array([np.sin(_t1), -np.cos(_t1)])
    _L = 1.35

    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _ax.axhspan(0, 1.6, color="#eaf4ff", zorder=0)
    _ax.axhspan(-1.6, 0, color="#fff3e0", zorder=0)
    _ax.plot([-1.8, 1.8], [0, 0], color="black", lw=1.5)
    _ax.plot([0, 0], [-1.4, 1.4], color="gray", lw=0.8, ls="--")

    draw_arrow(_ax, _O - _L * _d_in, _O, "tab:blue")
    draw_angle_arc(_ax, _O, _n_hat, -_d_in, 0.35, "tab:blue", "θ₁")

    _theta_c = np.degrees(np.arcsin(_n2 / _n1)) if _n1 > _n2 else None
    _d_t = refract2d(_d_in, _n_hat, _n1, _n2)
    _d_refl = reflect2d(_d_in, _n_hat)

    if _d_t is None:
        draw_arrow(_ax, _O, _O + _L * _d_refl, "tab:purple", lw=2.6)
        draw_angle_arc(_ax, _O, _n_hat, _d_refl, 0.3, "tab:purple", "θ₁")
        _tc_text = f"{_theta_c:.1f}°" if _theta_c is not None else "—"
        _status = f"**TOTAL INTERNAL REFLECTION** — θ₁ = {_theta1}° ≥ θ_c = {_tc_text}. No refracted ray exists; all the light reflects."
    else:
        draw_arrow(_ax, _O, _O + _L * _d_refl, "tab:purple", lw=1.0)
        _theta2 = snell_refracted_angle(_n1, _n2, _theta1)
        draw_arrow(_ax, _O, _O + _L * _d_t, "tab:red")
        draw_angle_arc(_ax, _O, -_n_hat, _d_t, 0.3, "tab:red", "θ₂")
        _tc_text = f"{_theta_c:.1f}°" if _theta_c is not None else "— (n₁ ≤ n₂: no critical angle exists for this pair)"
        _status = f"θ₂ = {_theta2:.1f}°.  Critical angle θ_c = {_tc_text}."

    _ax.text(-1.75, 1.2, f"medium 1  (n₁ = {_n1:.2f}, denser)", fontsize=10)
    _ax.text(-1.75, -1.3, f"medium 2  (n₂ = {_n2:.2f}, rarer)", fontsize=10)
    _ax.set_xlim(-1.9, 1.9)
    _ax.set_ylim(-1.6, 1.6)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(f"θ₁ = {_theta1}°  (faint purple ray = the partial reflection always present, even below θ_c)")

    mo.vstack([mo.md(_status), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Find $\theta_c$ for glass→air ($n_1=1.5$) and for diamond→air
      ($n_1=2.42$). Which material traps light inside more easily (i.e.
      has the *smaller* critical angle)? This is directly related to why
      cut diamonds sparkle — light entering the top has a hard time
      escaping out the bottom facets and bounces around instead.
    - Set $n_1<n_2$ (e.g. air → water) and increase $\theta_1$ all the
      way to $89°$. Does TIR ever happen? Why not — look at how $\theta_c$
      is defined.
    - At exactly $\theta_1=\theta_c$, what does the refracted ray's
      direction look like?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Dispersion — refraction depends on wavelength

    We've been treating $n$ as a single number per material, but it's
    actually a mild function of wavelength, $n(\lambda)$ — this is
    **dispersion**. A common empirical model (accurate over the visible
    range) is **Cauchy's equation**:

    $$ n(\lambda) = A + \frac{B}{\lambda^2} \qquad (\lambda \text{ in µm}) $$

    For ordinary crown glass, $A\approx1.5046$ and $B\approx0.0042\,\mu
    m^2$. Since $B>0$, $n$ is *larger* at short wavelengths (blue,
    ~450 nm) than at long ones (red, ~650 nm) — blue light bends more.
    Sending white light (a mix of all visible wavelengths) through a
    non-parallel-faced piece of glass — a **prism** — therefore splits it
    into a fan of colors, because each wavelength takes a slightly
    different path through Snell's law.

    (Real camera lenses are made of dispersive glass too — this is
    exactly what causes **chromatic aberration**, colored fringing around
    high-contrast edges in photos, since the lens focuses different
    colors to slightly different points.)
    """)
    return


@app.function
def wavelength_to_rgb(wavelength, gamma=0.8):
    """Approximate visible RGB for a wavelength in nm (Bruton's algorithm). Display only — not colorimetrically exact."""
    w = float(wavelength)
    if 380 <= w <= 440:
        attenuation = 0.3 + 0.7 * (w - 380) / (440 - 380)
        r = ((-(w - 440) / (440 - 380)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma
    elif 440 <= w <= 490:
        r = 0.0
        g = ((w - 440) / (490 - 440)) ** gamma
        b = 1.0
    elif 490 <= w <= 510:
        r = 0.0
        g = 1.0
        b = (-(w - 510) / (510 - 490)) ** gamma
    elif 510 <= w <= 580:
        r = ((w - 510) / (580 - 510)) ** gamma
        g = 1.0
        b = 0.0
    elif 580 <= w <= 645:
        r = 1.0
        g = (-(w - 645) / (645 - 580)) ** gamma
        b = 0.0
    elif 645 <= w <= 750:
        attenuation = 0.3 + 0.7 * (750 - w) / (750 - 645)
        r = (1.0 * attenuation) ** gamma
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0
    return (r, g, b)


@app.cell
def _(mo):
    prism_apex_slider = mo.ui.slider(start=20, stop=80, value=25, step=5, label="apex angle A (°)")
    prism_incidence_slider = mo.ui.slider(start=-60, stop=60, value=0, step=2, label="incidence angle on face 1 (°)")
    dispersion_B_slider = mo.ui.slider(start=0.0, stop=0.02, value=0.0042, step=0.0005, label="dispersion strength B (µm²)")
    mo.hstack([prism_apex_slider, prism_incidence_slider, dispersion_B_slider], justify="start", gap=2)
    return dispersion_B_slider, prism_apex_slider, prism_incidence_slider


@app.cell
def _(
    dispersion_B_slider,
    draw_angle_arc,
    draw_arrow,
    mo,
    np,
    plt,
    prism_apex_slider,
    prism_incidence_slider,
    refract2d,
):
    _A_deg = prism_apex_slider.value
    _theta_i = np.radians(prism_incidence_slider.value)
    _B = dispersion_B_slider.value
    _A_cauchy = 1.5046

    _H = 1.3
    _half_base = _H * np.tan(np.radians(_A_deg) / 2)
    _P0 = np.array([0.0, _H])
    _P1 = np.array([-_half_base, 0.0])
    _P2 = np.array([_half_base, 0.0])
    _centroid = (_P0 + _P1 + _P2) / 3

    def _outward_normal(a, b, interior_pt):
        d = b - a
        d = d / np.linalg.norm(d)
        n_c = np.array([d[1], -d[0]])
        mid = (a + b) / 2
        if np.dot(n_c, mid - interior_pt) < 0:
            n_c = -n_c
        return n_c

    def _cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _ray_segment_hit(p, d, a, b):
        """Ray (point p, unit direction d) intersected with the bounded
        segment a->b. Returns the ray parameter t, or None if the ray
        misses the segment (wrong direction, parallel, or beyond an
        endpoint)."""
        s_vec = b - a
        denom = _cross(d, s_vec)
        if abs(denom) < 1e-9:
            return None
        t = _cross(a - p, s_vec) / denom
        if t <= 1e-9:
            return None
        point = p + t * d
        s = np.dot(point - a, s_vec) / np.dot(s_vec, s_vec)
        if s < -1e-6 or s > 1 + 1e-6:
            return None
        return t

    _normal1 = _outward_normal(_P0, _P1, _centroid)
    _normal2 = _outward_normal(_P0, _P2, _centroid)
    _normal_base = np.array([0.0, -1.0])
    _tangent1 = np.array([-_normal1[1], _normal1[0]])
    # entry point near the apex (not the midpoint) — for a wide range of
    # apex/incidence angles this keeps the refracted ray inside the glass
    # long enough to reach face 2 rather than exiting through the base
    _M1 = _P0 + 0.2 * (_P1 - _P0)

    _d_in = -_normal1 * np.cos(_theta_i) + _tangent1 * np.sin(_theta_i)

    _fig, _ax = plt.subplots(figsize=(7, 5.5))
    _ax.fill([_P0[0], _P1[0], _P2[0]], [_P0[1], _P1[1], _P2[1]], color="#eef3ff", zorder=0)
    _ax.plot([_P0[0], _P1[0], _P2[0], _P0[0]], [_P0[1], _P1[1], _P2[1], _P0[1]], color="black", lw=1.5)

    draw_arrow(_ax, _M1 - 1.5 * _d_in, _M1, "black", lw=1.6)
    draw_angle_arc(_ax, _M1, -_normal1, _d_in, 0.28, "black", "θᵢ")
    _ax.plot([_M1[0], _M1[0] - 0.6 * _normal1[0]], [_M1[1], _M1[1] - 0.6 * _normal1[1]], color="gray", lw=0.8, ls="--")

    _wavelengths = np.linspace(390, 730, 9)
    _exit_angles = []
    _n_tir = 0
    _n_missed = 0
    for _wl in _wavelengths:
        _n_wl = _A_cauchy + _B / (_wl / 1000.0) ** 2
        _d_glass = refract2d(_d_in, _normal1, 1.0, _n_wl)
        if _d_glass is None:
            _n_tir += 1
            continue

        # the refracted ray may exit through face 2 (the usual case) or,
        # for steep apex/incidence combinations, through the base instead
        # — check both bounded edges and take whichever is actually hit.
        _t_face2 = _ray_segment_hit(_M1, _d_glass, _P0, _P2)
        _t_base = _ray_segment_hit(_M1, _d_glass, _P1, _P2)
        _candidates = [(t, n) for t, n in [(_t_face2, _normal2), (_t_base, _normal_base)] if t is not None]
        if not _candidates:
            _n_missed += 1
            continue
        _t_hit, _exit_normal = min(_candidates, key=lambda c: c[0])

        _exit_pt = _M1 + _t_hit * _d_glass
        _color = wavelength_to_rgb(_wl)
        _ax.plot([_M1[0], _exit_pt[0]], [_M1[1], _exit_pt[1]], color=_color, lw=1.3)
        _d_out = refract2d(_d_glass, _exit_normal, _n_wl, 1.0)
        if _d_out is None:
            _n_tir += 1
            continue
        draw_arrow(_ax, _exit_pt, _exit_pt + 1.6 * _d_out, _color, lw=1.6)
        _exit_angle = np.degrees(np.arccos(np.clip(np.dot(_d_out, _exit_normal), -1, 1)))
        _exit_angles.append(_exit_angle)

    _ax.set_xlim(-2.6, 3.0)
    _ax.set_ylim(-0.3, 2.0)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(f"apex angle A = {_A_deg}°,  dispersion B = {_B:.4f} µm²")

    if len(_exit_angles) >= 2:
        _spread = max(_exit_angles) - min(_exit_angles)
        _note = f"Exit angles range over {min(_exit_angles):.2f}°–{max(_exit_angles):.2f}° across the sampled wavelengths — an angular spread of **{_spread:.2f}°**."
    elif len(_exit_angles) == 1:
        _note = "Only one sampled wavelength makes it through these settings — try a smaller apex or incidence angle."
    else:
        _note = "No sampled wavelength makes it through these settings (all hit total internal reflection) — try a smaller apex or incidence angle."
    if _n_tir > 0:
        _note += f"  ({_n_tir} of {len(_wavelengths)} sampled wavelengths were blocked by total internal reflection.)"
    if _n_missed > 0:
        _note += f"  ({_n_missed} sampled wavelengths didn't hit either exit face — a very oblique setting.)"

    mo.vstack([mo.md(_note), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Drag the dispersion slider all the way to $B=0$. What happens to
      the fan of colored rays? (This is what a hypothetical *non-dispersive*
      glass would look like — no chromatic aberration, ever.)
    - For fixed $B$, does a bigger apex angle $A$ spread the colors more,
      or does the *incidence angle* matter more? Try to isolate each
      effect.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Lenses — full sphere vs. half sphere

    A single curved glass surface can already focus parallel light to a
    point — that's the essence of a lens. Here we exactly ray-trace
    parallel rays (using the same `refract2d` from above, no small-angle
    approximation) through two simple shapes: a **full glass sphere** and
    a **plano-convex hemisphere**, oriented **round side first** — the
    curved face meets the incoming parallel light, and the flat face is
    on the far (image) side.

    - **Hemisphere:** *both* surfaces now bend the light — the curved
      face does the first bending on entry, then the ray travels through
      the glass and bends again at the flat exit face (except exactly on
      the optical axis, where it hits the flat face straight-on).
    - **Full sphere:** light bends at *two* curved surfaces (entry and
      exit), which focuses it even more strongly than the hemisphere does.

    Look closely at where the rays actually cross the optical axis: rays
    entering farther from the axis converge to a *different* point than
    rays near the axis. This is **spherical aberration** — one of the
    reasons camera lenses are almost never simple spheres.
    """)
    return


@app.cell
def _(mo):
    R_lens_slider = mo.ui.slider(start=0.5, stop=2.0, value=1.0, step=0.1, label="radius R")
    n_lens_slider = mo.ui.slider(start=1.2, stop=2.4, value=1.5, step=0.05, label="index n")
    n_rays_slider = mo.ui.slider(start=3, stop=15, value=7, step=2, label="number of rays")
    lens_mode_radio = mo.ui.radio(options=["Full sphere", "Hemisphere"], value="Full sphere", label="lens shape")
    mo.vstack([mo.hstack([R_lens_slider, n_lens_slider, n_rays_slider], justify="start", gap=2), lens_mode_radio])
    return R_lens_slider, lens_mode_radio, n_lens_slider, n_rays_slider


@app.cell
def _(
    R_lens_slider,
    lens_mode_radio,
    mo,
    n_lens_slider,
    n_rays_slider,
    np,
    plt,
    ray_circle_hit,
    refract2d,
):
    _R = R_lens_slider.value
    _n = n_lens_slider.value
    _mode = lens_mode_radio.value
    _n_rays = n_rays_slider.value

    _x_start = -2.6 * _R  # both shapes' leftmost (curved) surface sits at x=-R

    _heights = np.linspace(-0.85 * _R, 0.85 * _R, _n_rays)
    _d0 = np.array([1.0, 0.0])

    _fig, _ax = plt.subplots(figsize=(7.5, 5))
    _theta_full = np.linspace(0, 2 * np.pi, 200)
    if _mode == "Full sphere":
        _ax.plot(_R * np.cos(_theta_full), _R * np.sin(_theta_full), color="black", lw=1.5)
        _ax.fill(_R * np.cos(_theta_full), _R * np.sin(_theta_full), color="#dceeff", alpha=0.5, zorder=0)
    else:
        # curved cap facing -x (toward the incoming light), flat face at x=0
        _theta_half = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
        _xs = _R * np.cos(_theta_half)
        _ys = _R * np.sin(_theta_half)
        _ax.plot(np.append(_xs, _xs[0]), np.append(_ys, _ys[0]), color="black", lw=1.5)
        _ax.fill(_xs, _ys, color="#dceeff", alpha=0.5, zorder=0)

    _crossings = []
    for _y0 in _heights:
        if abs(_y0) < 1e-9:
            _ax.plot([_x_start, _x_start + 8 * _R], [0, 0], color="tab:red", lw=1.2)
            continue

        _p0 = np.array([_x_start, _y0])
        if _mode == "Full sphere":
            _t_in = ray_circle_hit(_p0, _d0, _R)
            if _t_in is None:
                continue
            _p_in = _p0 + _t_in * _d0
            _ax.plot([_p0[0], _p_in[0]], [_p0[1], _p_in[1]], color="tab:red", lw=1.2)
            _d_glass = refract2d(_d0, _p_in / _R, 1.0, _n)
            if _d_glass is None:
                continue
            _t_out = ray_circle_hit(_p_in, _d_glass, _R)
            if _t_out is None:
                continue
            _p_exit = _p_in + _t_out * _d_glass
            _ax.plot([_p_in[0], _p_exit[0]], [_p_in[1], _p_exit[1]], color="tab:red", lw=1.2)
            _d_final = refract2d(_d_glass, _p_exit / _R, _n, 1.0)
        else:
            # curved face first (entry), flat face second (exit at x=0)
            _x_c = -np.sqrt(max(_R ** 2 - _y0 ** 2, 0.0))
            _p_curved = np.array([_x_c, _y0])
            _ax.plot([_p0[0], _p_curved[0]], [_p0[1], _p_curved[1]], color="tab:red", lw=1.2)
            _d_glass = refract2d(_d0, _p_curved / _R, 1.0, _n)
            if _d_glass is None:
                continue
            _t_flat = (0.0 - _p_curved[0]) / _d_glass[0]
            _p_exit = _p_curved + _t_flat * _d_glass
            _ax.plot([_p_curved[0], _p_exit[0]], [_p_curved[1], _p_exit[1]], color="tab:red", lw=1.2)
            _n_flat = np.array([1.0, 0.0])
            _d_final = refract2d(_d_glass, _n_flat, _n, 1.0)

        if _d_final is None:
            continue
        _p_far = _p_exit + 2.6 * _R * _d_final
        _ax.plot([_p_exit[0], _p_far[0]], [_p_exit[1], _p_far[1]], color="tab:red", lw=1.2)
        if abs(_d_final[1]) > 1e-9:
            _t_cross = -_p_exit[1] / _d_final[1]
            if _t_cross > 0:
                _crossings.append(_p_exit[0] + _t_cross * _d_final[0])

    _ax.axhline(0, color="gray", lw=0.7, ls="--")

    _x_right = max(_crossings) + 1.2 * _R if _crossings else _x_start + 6 * _R

    _ax.set_aspect("equal")
    _ax.set_xlim(_x_start - 0.2, _x_right)
    _ax.set_yticks([])
    _ax.set_xlabel("optical axis")
    _ax.set_title(f"{_mode}:  R={_R:.2f}, n={_n:.2f}")

    if len(_crossings) >= 2:
        _spread = max(_crossings) - min(_crossings)
        _note = f"Marginal-ray crossings span x ∈ [{min(_crossings):.2f}, {max(_crossings):.2f}] — a **{_spread:.3f}**-wide spread due to spherical aberration (rays farther from the axis focus closer to the lens)."
    else:
        _note = ""

    mo.vstack([mo.md(_note), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The thin-lens model, focal ratio (F/#), and numerical aperture

    Real camera and eyeglass lenses are (much) thinner than their radius
    of curvature, so instead of tracing every ray through the exact glass
    shape, we usually use the **thin-lens approximation**: all the
    bending is treated as happening at a single plane, characterized by
    one number, the **focal length** $f$.

    $$ \frac{1}{s_o} + \frac{1}{s_i} = \frac{1}{f} $$

    where $s_o$ is the object's distance in front of the lens and $s_i$
    is the resulting image distance (behind the lens, if positive). The
    image's **magnification** is $m=-s_i/s_o$ — negative means inverted.

    Any point on the object can be located in the image by tracing just
    two of these three easy rays from it (the third automatically agrees):

    1. A ray **parallel** to the axis, which bends to pass through the
       *far* focal point.
    2. A ray through the *near* focal point, which exits **parallel** to
       the axis.
    3. A ray through the lens **center**, which goes straight through
       undeviated.

    Where does the single number $f$ actually come from? A real lens is
    two curved surfaces, each with its own radius of curvature, ground
    into a piece of glass of index $n$. The **lensmaker's equation**
    combines them:

    $$ \frac{1}{f} = (n-1)\left(\frac{1}{R_1}-\frac{1}{R_2}\right) $$

    $R_1$ is the front surface's radius, $R_2$ the back surface's — with
    the sign convention that **a radius is positive if its center of
    curvature lies on the far (transmission) side of that surface**, and
    negative if the center of curvature is on the near (incoming) side.
    So a front surface bulging *toward* the incoming light has $R_1>0$,
    and a back surface bulging *away* (out the far side, like the right
    half of a ball) has $R_2<0$. A flat surface has $R=\infty$
    (equivalently, zero curvature $1/R=0$) and drops out of the equation
    entirely — that's exactly the plano-convex case from §6.

    The upshot: **many different shapes can share the same $f$.** A
    symmetric biconvex lens, a plano-convex lens, and a meniscus can all
    be built to the same focal length by trading $R_1$ against $R_2$ —
    the demo below lets you see that trade directly.
    """)
    return


@app.cell
def _(mo):
    so_slider = mo.ui.slider(start=0.3, stop=4.0, value=2.0, step=0.1, label="object distance sₒ")
    C1_slider = mo.ui.slider(start=-2.0, stop=2.0, value=1.0, step=0.1, label="front curvature 1/R₁")
    C2_slider = mo.ui.slider(start=-2.0, stop=2.0, value=-1.0, step=0.1, label="back curvature 1/R₂")
    D_slider = mo.ui.slider(start=0.2, stop=2.0, value=1.0, step=0.1, label="aperture diameter D")
    mo.vstack([mo.hstack([so_slider, D_slider], justify="start", gap=2), mo.hstack([C1_slider, C2_slider], justify="start", gap=2)])
    return C1_slider, C2_slider, D_slider, so_slider


@app.cell
def _(C1_slider, C2_slider, D_slider, draw_arrow, mo, np, plt, so_slider):
    _so, _D = so_slider.value, D_slider.value
    _C1, _C2 = C1_slider.value, C2_slider.value
    _n_glass = 1.5
    _ho = 0.6

    def _R_of(C):
        return None if abs(C) < 1e-9 else 1.0 / C

    def _sag(C, y):
        # x-offset of a surface with curvature C (0 = flat) at height y,
        # vertex at x=0, using the sign convention that a positive radius
        # bulges toward -x (i.e. toward the incoming light on the front
        # surface). Height is clipped to the surface's own radius, since a
        # spherical cap can't extend past it.
        if abs(C) < 1e-9:
            return np.zeros_like(y)
        R = 1.0 / C
        y_clipped = np.clip(y, -abs(R), abs(R))
        return R - np.sign(R) * np.sqrt(np.maximum(R ** 2 - y_clipped ** 2, 0.0))

    _R1, _R2 = _R_of(_C1), _R_of(_C2)
    _inv_f = (_n_glass - 1) * (_C1 - _C2)
    _R1_text = "∞ (flat)" if _R1 is None else f"{_R1:.2f}"
    _R2_text = "∞ (flat)" if _R2 is None else f"{_R2:.2f}"

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.axvline(0, color="black", lw=0.3)

    # lens-shape icon built from the two surfaces' actual curvatures
    _t_icon = 0.12
    _y_icon = np.linspace(-_D / 2, _D / 2, 80)
    _front_x = -_t_icon / 2 + _sag(_C1, _y_icon)
    _back_x = _t_icon / 2 + _sag(_C2, _y_icon)
    _ax.fill(np.concatenate([_front_x, _back_x[::-1]]), np.concatenate([_y_icon, _y_icon[::-1]]), color="#dceeff", alpha=0.7, zorder=1)
    _ax.plot(_front_x, _y_icon, color="black", lw=1.8, zorder=2)
    _ax.plot(_back_x, _y_icon, color="black", lw=1.8, zorder=2)

    if abs(_inv_f) < 1e-6:
        _ax.set_aspect("equal")
        _ax.set_xlim(-_so - 0.6, 2.5)
        _ax.set_ylim(-1.3, 1.3)
        _ax.axis("off")
        _ax.set_title(f"R₁={_R1_text}, R₂={_R2_text}, n={_n_glass}")
        _readout = mo.md(
            f"1/f = (n-1)(1/R₁ − 1/R₂) ≈ 0  →  **this shape has (essentially) zero optical power** — "
            f"parallel rays pass through with no net focusing. Try making the front surface more convex "
            f"(raise C₁) or the back surface more convex the other way (lower C₂)."
        )
    elif 1.0 / _inv_f <= 0:
        _f = 1.0 / _inv_f
        _ax.set_aspect("equal")
        _ax.set_xlim(-_so - 0.6, 2.5)
        _ax.set_ylim(-1.3, 1.3)
        _ax.axis("off")
        _ax.set_title(f"R₁={_R1_text}, R₂={_R2_text}, n={_n_glass}  →  f = {_f:.2f} (diverging)")
        _readout = mo.md(
            f"f = {_f:.2f} < 0: **this shape is diverging**, not converging — parallel rays spread apart "
            f"instead of meeting at a real focus (think of a concave lens). The ray-diagram below only "
            f"covers converging shapes; try raising C₁ or lowering C₂ to flip it back to converging."
        )
    else:
        _f = 1.0 / _inv_f

        _ax.axhline(0, color="gray", lw=0.6, ls="--")
        _ax.plot([-_f, _f], [0, 0], marker="o", color="gray", ms=4, ls="none")
        _ax.text(-_f, -0.12, "F", ha="center", color="gray", fontsize=10)
        _ax.text(_f, -0.12, "F'", ha="center", color="gray", fontsize=10)

        draw_arrow(_ax, [-_so, 0], [-_so, _ho], "black", lw=2)
        _ax.text(-_so, -0.12, "object", ha="center", fontsize=9)

        _tip = np.array([-_so, _ho])

        if abs(_so - _f) < 1e-6:
            _note = "s₀ = f exactly: refracted rays emerge parallel — the image forms at infinity (this is how a lens makes a collimated beam, e.g. a flashlight or laser collimator)."
            _si = None
        else:
            _si = 1.0 / (1.0 / _f - 1.0 / _so)
            _m = -_si / _so
            _hi = _m * _ho

            # ray 1: parallel to axis, then through far focal point (f, 0)
            _draw_x = max(abs(_si) + 0.8, 2.2 * _f)
            draw_arrow(_ax, _tip, [0, _ho], "tab:blue", lw=1.4)

            # ray 2: through near focal point (-f, 0), then parallel to axis
            _y2 = _ho * _f / (_f - _so)
            draw_arrow(_ax, _tip, [0, _y2], "tab:orange", lw=1.4)

            # ray 3: straight through the center
            _slope3 = -_ho / _so

            if _si > 0:
                draw_arrow(_ax, [0, _ho], [_si, _hi], "tab:blue", lw=1.4)
                draw_arrow(_ax, [0, _y2], [_si, _y2], "tab:orange", lw=1.4)
                draw_arrow(_ax, _tip, [_si, _slope3 * _si], "tab:green", lw=1.4)
                draw_arrow(_ax, [_si, 0], [_si, _hi], "black", lw=2)
                _ax.text(_si, (0.12 if _hi >= 0 else -0.12), "image", ha="center", fontsize=9)
                _kind = "real, " + ("inverted" if _m < 0 else "upright") + f", magnification m = {_m:.2f}"
            else:
                _far = _draw_x
                draw_arrow(_ax, [0, _ho], [_far, _ho + (_far / _f) * (0 - _ho)], "tab:blue", lw=1.4)
                _ax.plot([0, _si], [_ho, _hi], color="tab:blue", lw=1.0, ls=":")
                draw_arrow(_ax, [0, _y2], [_far, _y2], "tab:orange", lw=1.4)
                draw_arrow(_ax, _tip, [_far, _slope3 * _far], "tab:green", lw=1.4)
                _ax.plot([0, _si], [0, _hi], color="tab:green", lw=1.0, ls=":")
                _ax.annotate("", xy=(_si, _hi), xytext=(_si, 0), arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.6, ls=(0, (3, 2))))
                _ax.text(_si, _hi + 0.12, "virtual image", ha="center", fontsize=9, color="dimgray")
                _kind = "virtual, " + ("inverted" if _m < 0 else "upright") + f", magnification m = {_m:.2f}"

            _note = f"sᵢ = {_si:.2f}  →  {_kind}"

        _ax.set_aspect("equal")
        _ax.set_xlim(-_so - 0.6, max((_si if _si is not None else 0) + 1.2, _f + 1.2, 2.5))
        _ax.set_ylim(-1.3, 1.3)
        _ax.axis("off")
        _ax.set_title(f"R₁={_R1_text}, R₂={_R2_text}, n={_n_glass}  →  f = {_f:.2f}")

        _F_number = _f / _D
        _NA = 1.0 / (2 * _F_number)
        _readout = mo.md(f"{_note}  \n**F-number:** F/# = f/D = {_F_number:.2f}    **Numerical aperture:** NA ≈ {_NA:.3f}")

    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The aperture diameter $D$ (how much of the lens actually lets light
    through — in a real camera this is set by the adjustable iris/diaphragm)
    matters for two more reasons beyond just image position:

    **Focal ratio (F-number):** $\quad F/\# = \dfrac{f}{D}$

    A smaller $F/\#$ means a *wider* aperture relative to the focal
    length — more light-gathering area, so a brighter image (image
    irradiance scales as $1/(F/\#)^2$, since it depends on the
    *area* of the aperture). A smaller aperture (bigger $F/\#$) also
    increases **depth of field** — the range of object distances that
    appear acceptably in focus at once — at the cost of a dimmer image,
    the classic trade-off in photography.

    **Numerical aperture (NA):** describes the actual cone angle of light
    converging to (or diverging from) the focal point,
    $$ NA = n\sin\theta_{\text{marginal}} $$
    where $\theta_{\text{marginal}}$ is the half-angle subtended by the
    aperture as seen from the focal point, and $n$ is the index of the
    medium the light is traveling through (n=1 in air). For an object at
    infinity, $\theta_{\text{marginal}}\approx\arctan(D/2f)$, giving the
    handy approximation
    $$ NA \approx \frac{D}{2f} = \frac{1}{2\,(F/\#)} $$
    NA shows up again once you start asking how *small* a detail a lens
    can resolve — the diffraction limit scales roughly as
    $\lambda/(2\,NA)$ — a topic for another day, but the aperture you set
    here is exactly what controls it.

    **How does this compare to a pinhole camera?** Notebooks 3–4's camera
    model — a single projection center, every ray drawn as a straight
    line from a 3D point through that one center to the image — is
    exactly a **pinhole camera**: an infinitesimally small aperture
    ($D\to 0$) with no lens at all. Only one ray from each scene point
    can get through such a tiny opening, so a pinhole image is in
    perfect focus at *every* depth simultaneously (infinite depth of
    field, no focal length to set) — but almost no light makes it
    through, so a real pinhole needs very long exposures and gets dimmer
    still as you shrink the hole further to sharpen the image (and, past
    a certain point, actually blurs again due to diffraction — a wave
    effect this notebook doesn't cover). A lens is the fix: opening the
    aperture up to a real, finite $D$ admits far more light
    ($\propto D^2$), but now the diverging rays from a scene point need
    to be bent back together — which is only exact at one object
    distance $s_o$ (the rest fall inside/outside the depth of field), and
    only approximately even then, since real lenses add the aberrations
    we've been seeing (spherical aberration in §6, chromatic aberration
    in §5). In short: the pinhole trades away light for perfect, focus-free
    geometry; a lens buys back the light at the cost of a focus distance
    and a handful of aberrations to manage.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - The **index of refraction** $n=c/v$ sets how much a material slows
      light down, and controls how strongly it bends at an interface.
    - **Snell's law**, $n_1\sin\theta_1=n_2\sin\theta_2$, governs
      refraction; the law of reflection, $\theta_i=\theta_r$, governs the
      part that always reflects too.
    - Going from a denser to a rarer medium past the **critical angle**
      $\theta_c=\arcsin(n_2/n_1)$ produces **total internal reflection** —
      no transmitted ray at all.
    - $n$ depends weakly on wavelength (**dispersion**), which is why
      prisms make rainbows and why real lenses show **chromatic
      aberration**.
    - A curved glass surface focuses light by refraction; comparing a
      full sphere to a hemisphere shows both the basic single/double
      surface focusing math *and* **spherical aberration** — real optics
      are more than a single focal number.
    - The **thin-lens equation** $1/s_o+1/s_i=1/f$ locates images;
      aperture diameter $D$ sets the **F-number** $f/D$ (brightness,
      depth of field) and the **numerical aperture** $\approx 1/(2\,F/\#)$
      (light cone angle, resolution).

    **Next up:** all of this has assumed a *continuous* image forming on
    an idealized flat plane. The next notebook covers **digital image
    formation and color spaces** — how that continuous, focused image
    gets sampled into a discrete grid of pixels, quantized to a finite
    number of levels, and represented as color.
    """)
    return


if __name__ == "__main__":
    app.run()
