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

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    return go, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Lighting and Reflectance: How Surfaces Turn Light into Pixels

    Notebooks 3–4 built the *geometric* half of image formation: how a 3D
    point ends up at a particular pixel. But a pixel isn't just a location
    — it's a **value**. Where does that value come from?

    This notebook follows Szeliski §2.2, *Photometric image formation*,
    and works through its generative chain: a **light source** emits
    radiance $L(\hat v;\lambda)$, a surface **scatters** it according to
    its **BRDF**, and the two most important special cases of that
    scattering — **diffuse (Lambertian)** and **specular** reflection —
    are what give objects their matte or shiny appearance.

    1. Light, wavelength, and color
    2. Light sources — point and environment, and $L(\hat v;\lambda)$
    3. The bidirectional reflectance distribution function (BRDF)
    4. Diffuse (Lambertian) reflection
    5. Specular reflection

    This one is mostly **demonstrations** — play with the sliders, watch
    how the equation and the picture move together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Light, wavelength, and color

    Light is electromagnetic radiation; a light source's color comes from
    *how much* energy it emits at each wavelength $\lambda$ — its spectral
    power distribution. The human eye is sensitive to roughly
    **380–750 nm**, the **visible spectrum**; wavelength within that band
    is what we perceive as hue, from violet (short) through blue, green,
    yellow, to red (long).

    A light source's full color is really a *distribution* over
    wavelength, $L(\lambda)$ — exactly the quantity Szeliski §2.2.1
    formalizes next.
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
    return (min(1.0, max(0.0, r)), min(1.0, max(0.0, g)), min(1.0, max(0.0, b)))


@app.cell
def _(mo):
    lambda_slider = mo.ui.slider(start=380, stop=750, value=550, step=5, label="wavelength λ (nm)")
    lambda_slider
    return (lambda_slider,)


@app.cell
def _(lambda_slider, mo, np, plt):
    _lam = lambda_slider.value
    _sample_lams = np.linspace(380, 750, 300)
    _bar = np.array([wavelength_to_rgb(_l) for _l in _sample_lams])[None, :, :]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 2.6), gridspec_kw={"width_ratios": [4, 1]})
    _ax1.imshow(_bar, extent=[380, 750, 0, 1], aspect="auto")
    _ax1.axvline(_lam, color="white", linewidth=2)
    _ax1.axvline(_lam, color="black", linewidth=1, linestyle="--")
    _ax1.set_yticks([])
    _ax1.set_xlabel("wavelength (nm)")
    _ax1.set_title("visible spectrum")

    _rgb = wavelength_to_rgb(_lam)
    _ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=_rgb))
    _ax2.set_xlim(0, 1)
    _ax2.set_ylim(0, 1)
    _ax2.set_xticks([])
    _ax2.set_yticks([])
    _ax2.set_title("perceived color")
    _fig.tight_layout()

    _readout = mo.md(f"λ = **{_lam} nm** → approximate display RGB = **({_rgb[0]:.2f}, {_rgb[1]:.2f}, {_rgb[2]:.2f})**")
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Light sources — point, environment, and $L(\hat v;\lambda)$

    **Point light source.** Originates at a single location in space (a
    bulb, or the Sun, potentially at infinity). Besides its position, a
    point source has an intensity and a color spectrum — a distribution
    over wavelength, $L(\lambda)$. Its intensity falls off with the
    **square of the distance** to the object being lit, because the same
    total light spreads over a larger spherical area (Szeliski §2.2.1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Variables.** $L_0$ is the point source's radiant intensity — how
    much power it emits, independent of distance. $r$ is the distance
    from the source to the surface being lit. $E(r)$ is the resulting
    **irradiance**: the power *per unit area* actually arriving at the
    surface, which is what determines how bright that surface appears.

    **Intuition.** The source radiates the same total power $L_0$ in every
    direction, spreading it over the surface of an ever-larger imaginary
    sphere of radius $r$ centered on the source. A sphere's surface area
    is $4\pi r^2$, so the power *per unit area* landing on it is

    $$
    E(r) = \frac{L_0}{4\pi r^2} \;\propto\; \frac{L_0}{r^2}.
    $$

    Double the distance and the same total power is smeared over $4\times$
    the area, so each unit of surface receives $1/4$ the light — this is
    the inverse-square law, and it's why a lamp looks so much dimmer from
    across a room than up close.
    """)
    return


@app.cell
def _(mo):
    dist_slider = mo.ui.slider(start=1, stop=10, value=3, step=0.25, label="distance r")
    L0_slider = mo.ui.slider(start=20, stop=200, value=100, step=10, label="source intensity L₀")
    mo.hstack([dist_slider, L0_slider], justify="start", gap=2)
    return L0_slider, dist_slider


@app.cell
def _(L0_slider, dist_slider, mo, np, plt):
    _r = dist_slider.value
    _L0 = L0_slider.value
    _E = _L0 / _r**2

    _rs = np.linspace(1, 10, 200)
    _Es = _L0 / _rs**2

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    _ax1.plot(_rs, _Es, color="tab:orange")
    _ax1.scatter([_r], [_E], color="black", zorder=3)
    _ax1.set_xlabel("distance r")
    _ax1.set_ylabel("irradiance ∝ L₀ / r²")
    _ax1.set_title("inverse-square falloff")
    _ax1.grid(True, linewidth=0.3)

    _brightness = min(1.0, _E / _L0)
    _ax2.add_patch(plt.Rectangle((0, 0), 1, 1, color=(_brightness, _brightness, _brightness)))
    _ax2.set_xlim(0, 1)
    _ax2.set_ylim(0, 1)
    _ax2.set_xticks([])
    _ax2.set_yticks([])
    _ax2.set_title("brightness at r")
    _fig.tight_layout()

    _readout = mo.md(f"E(r) = L₀ / r² = {_L0:.0f} / {_r:.2f}² = **{_E:.2f}**")
    mo.vstack([_readout, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Environment (area) light source.** Real illumination often arrives
    from every direction at once (a cloudy sky, a room's walls). This is
    captured by an **environment map** — a function that maps each
    incident direction $\hat v$ to a color/radiance value,

    $$
    L(\hat v;\lambda) \qquad \text{(Szeliski Eq. 2.81)}.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The bidirectional reflectance distribution function (BRDF)

    When light hits a surface it's scattered (Szeliski §2.2.2, Fig. 2.15).
    The most general description of this scattering is the **BRDF**.
    Relative to a local frame on the surface, $(\hat d_x, \hat d_y,
    \hat n)$, it's a 4D function of the incident direction $\hat v_i$ and
    the reflected (viewing) direction $\hat v_r$, written in terms of the
    angles each makes with that frame:

    $$
    f_r(\theta_i, \phi_i, \theta_r, \phi_r;\ \lambda) \qquad \text{(Eq. 2.82)}
    $$

    The BRDF is **reciprocal**: swapping the roles of $\hat v_i$ and
    $\hat v_r$ gives the same value (Helmholtz reciprocity).

    Most surfaces are **isotropic** — there's no preferred direction on
    the surface itself, so only the *difference* $\phi_r - \phi_i$
    matters, not the absolute azimuths:

    $$
    f_r(\theta_i, \theta_r, |\phi_r - \phi_i|;\ \lambda) \qquad \text{(Eq. 2.83)}
    $$

    (Anisotropic surfaces, like brushed aluminum, break this — reflectance
    also depends on orientation relative to the scratches.)

    The diagram below is a **live 3D plot** — click and drag to rotate it,
    scroll to zoom. Dotted arcs mark the polar angle $\theta$ (from
    $\hat n$) and azimuth $\phi$ (from $\hat d_x$, in the surface plane)
    for each of $\hat v_i$ (red) and $\hat v_r$ (blue).
    """)
    return


@app.cell
def _(np):
    def spherical_dir(theta_deg, phi_deg):
        """Unit vector at polar angle theta (from n_hat = +z) and azimuth phi (from d_x = +x)."""
        t, p = np.radians(theta_deg), np.radians(phi_deg)
        return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    return (spherical_dir,)


@app.cell
def _(go, np):
    def brdf_arrow(direction, color, label, length=1.0, cone_size=0.09):
        """A line + cone-tip arrow from the origin, with a text label past the tip."""
        _tip = direction * length
        _base = _tip * (1 - cone_size)
        _line = go.Scatter3d(
            x=[0, _base[0]], y=[0, _base[1]], z=[0, _base[2]],
            mode="lines", line=dict(color=color, width=6),
            showlegend=False, hoverinfo="skip",
        )
        _cone = go.Cone(
            x=[_base[0]], y=[_base[1]], z=[_base[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            colorscale=[[0, color], [1, color]], showscale=False,
            sizemode="absolute", sizeref=cone_size * 1.4, anchor="tail", hoverinfo="skip",
        )
        _text = go.Scatter3d(
            x=[_tip[0] * 1.12], y=[_tip[1] * 1.12], z=[_tip[2] * 1.12],
            mode="text", text=[label], textfont=dict(color=color, size=16),
            showlegend=False, hoverinfo="skip",
        )
        return [_line, _cone, _text]

    def brdf_theta_arc(theta_deg, phi_deg, color, label, radius=0.35):
        """Dotted arc from n_hat to the vector at (theta_deg, phi_deg), in the vertical plane containing both."""
        _p = np.radians(phi_deg)
        _e_phi = np.array([np.cos(_p), np.sin(_p), 0.0])
        _n_hat = np.array([0.0, 0.0, 1.0])
        _ts = np.linspace(0, np.radians(theta_deg), 30)
        _pts = radius * (np.outer(np.cos(_ts), _n_hat) + np.outer(np.sin(_ts), _e_phi))
        _mid = _pts[len(_pts) // 2] * 1.15
        _line = go.Scatter3d(
            x=_pts[:, 0], y=_pts[:, 1], z=_pts[:, 2], mode="lines",
            line=dict(color=color, width=4, dash="dot"), showlegend=False, hoverinfo="skip",
        )
        _text = go.Scatter3d(
            x=[_mid[0]], y=[_mid[1]], z=[_mid[2]], mode="text",
            text=[label], textfont=dict(color=color, size=13), showlegend=False, hoverinfo="skip",
        )
        return [_line, _text]

    def brdf_phi_arc(phi_deg, color, label, radius=0.55):
        """Dotted arc from d_x to azimuth phi_deg, in the surface (z=0) plane."""
        _ts = np.linspace(0, np.radians(phi_deg), 30)
        _pts = radius * np.stack([np.cos(_ts), np.sin(_ts), np.zeros_like(_ts)], axis=-1)
        _mid = _pts[len(_pts) // 2] * 1.15
        _line = go.Scatter3d(
            x=_pts[:, 0], y=_pts[:, 1], z=_pts[:, 2], mode="lines",
            line=dict(color=color, width=4, dash="dot"), showlegend=False, hoverinfo="skip",
        )
        _text = go.Scatter3d(
            x=[_mid[0]], y=[_mid[1]], z=[_mid[2]], mode="text",
            text=[label], textfont=dict(color=color, size=13), showlegend=False, hoverinfo="skip",
        )
        return [_line, _text]

    return brdf_arrow, brdf_phi_arc, brdf_theta_arc


@app.cell
def _(mo):
    theta_i_slider = mo.ui.slider(start=0, stop=85, value=30, step=5, label="θi (incident)")
    phi_i_slider = mo.ui.slider(start=0, stop=355, value=20, step=5, label="φi (incident)")
    theta_r_slider = mo.ui.slider(start=0, stop=85, value=40, step=5, label="θr (reflected)")
    phi_r_slider = mo.ui.slider(start=0, stop=355, value=100, step=5, label="φr (reflected)")
    mo.hstack([theta_i_slider, phi_i_slider, theta_r_slider, phi_r_slider], justify="start", gap=2)
    return phi_i_slider, phi_r_slider, theta_i_slider, theta_r_slider


@app.cell
def _(
    brdf_arrow,
    brdf_phi_arc,
    brdf_theta_arc,
    go,
    mo,
    np,
    phi_i_slider,
    phi_r_slider,
    spherical_dir,
    theta_i_slider,
    theta_r_slider,
):
    _vi = spherical_dir(theta_i_slider.value, phi_i_slider.value)
    _vr = spherical_dir(theta_r_slider.value, phi_r_slider.value)
    _dphi = ((phi_r_slider.value - phi_i_slider.value + 180) % 360) - 180

    _xx, _yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
    _traces = [
        go.Surface(
            x=_xx, y=_yy, z=np.zeros_like(_xx), showscale=False,
            colorscale=[[0, "tan"], [1, "tan"]], opacity=0.25, hoverinfo="skip",
        )
    ]
    _traces += brdf_arrow(np.array([1.0, 0.0, 0.0]), "gray", "dx")
    _traces += brdf_arrow(np.array([0.0, 1.0, 0.0]), "gray", "dy")
    _traces += brdf_arrow(np.array([0.0, 0.0, 1.0]), "black", "n")
    _traces += brdf_arrow(_vi, "red", "vi")
    _traces += brdf_arrow(_vr, "blue", "vr")
    _traces += brdf_theta_arc(theta_i_slider.value, phi_i_slider.value, "red", "θi", radius=0.35)
    _traces += brdf_theta_arc(theta_r_slider.value, phi_r_slider.value, "blue", "θr", radius=0.30)
    _traces += brdf_phi_arc(phi_i_slider.value, "red", "φi", radius=0.55)
    _traces += brdf_phi_arc(phi_r_slider.value, "blue", "φr", radius=0.65)

    _fig = go.Figure(data=_traces)
    _fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1], title="x"),
            yaxis=dict(range=[-1.1, 1.1], title="y"),
            zaxis=dict(range=[0, 1.3], title="z"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        title="local surface frame + incident/reflected directions (drag to rotate)",
    )

    _readout = mo.md(f"""
    θi = **{theta_i_slider.value}°**, φi = **{phi_i_slider.value}°**  ·
    θr = **{theta_r_slider.value}°**, φr = **{phi_r_slider.value}°**  ·
    φr − φi = **{_dphi}°**
    """)
    mo.vstack([_readout, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrating the incoming light against the BRDF gives the light
    leaving a surface point in direction $\hat v_r$:

    $$
    L_r(\hat v_r;\lambda) = \int L_i(\hat v_i;\lambda)\, f_r(\hat v_i,\hat v_r,\hat n;\lambda)\, \cos^+\theta_i\, d\hat v_i \qquad \text{(Eq. 2.84–2.85)}
    $$

    or, for a finite set of point sources,

    $$
    L_r(\hat v_r;\lambda) = \sum_i L_i(\lambda)\, f_r(\hat v_i,\hat v_r,\hat n;\lambda)\, \cos^+\theta_i \qquad \text{(Eq. 2.86)}
    $$

    **Reading the equation.** $L_i(\hat v_i;\lambda)$ is the radiance
    arriving from direction $\hat v_i$; the BRDF $f_r(\hat v_i,\hat
    v_r,\hat n;\lambda)$ says what *fraction* of that light, arriving from
    $\hat v_i$, gets redirected toward $\hat v_r$. Multiplying the two and
    integrating (or summing, for a handful of point sources) over every
    incoming direction $\hat v_i$ totals up every light source's
    contribution to what leaves the surface toward $\hat v_r$. The
    remaining factor, $\cos^+\theta_i = \max(0, \cos\theta_i)$ with
    $\theta_i$ the angle between $\hat v_i$ and the normal $\hat n$, is
    the piece that's easy to skip past — and it's doing two jobs at once:

    1. **Foreshortening.** The BRDF is defined *per unit area of the
       surface*, but a beam of light of fixed cross-section spreads over
       *more* surface area when it lands at a grazing angle than head-on
       — the same beam illuminates a bigger patch. The irradiance actually
       received per unit *surface* area therefore scales with $\cos\theta_i$,
       exactly the same $1/r^2$-style geometric dilution you saw with the
       point-light source in §2, just from tilting the surface instead of
       moving it farther away. (Photographers know this instinctively:
       tilt a reflector edge-on to a lamp and it goes dim, even though it
       hasn't moved.)
    2. **The "+" (clamping to zero).** $\cos\theta_i$ alone would go
       *negative* once $\theta_i > 90°$ — i.e., once $\hat v_i \cdot \hat n
       < 0$, meaning the light source is *behind* the surface at that
       point (on the side away from the outward normal) and physically
       cannot illuminate it at all. A bare cosine would subtract light
       that was never there; clamping with $\max(0,\cdot)$ correctly zeroes
       out any light source the surface is facing away from, instead of
       producing a nonsensical negative radiance.

    So $\cos^+\theta_i$ is exactly $\cos\theta_i$ while the surface faces
    the light ($0° \le \theta_i < 90°$), and exactly $0$ once the light
    has swung around behind it — both effects the diffuse demo below lets
    you see directly, since its "foreshortening" curve is this same
    $\cos^+\theta_i$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Diffuse (Lambertian) reflection

    **Diffuse (Lambertian) reflection** is the special case where the BRDF
    is *constant* — light scatters equally in every direction:

    $$
    f_d(\hat v_i, \hat v_r, \hat n;\lambda) = f_d(\lambda) \qquad \text{(Eq. 2.87)}
    $$

    so the only thing that still depends on geometry is the incidence
    angle:

    $$
    L_d(\hat v_r;\lambda) = \sum_i L_i(\lambda)\, f_d(\lambda)\, [\hat v_i \cdot \hat n]^+ \qquad \text{(Eq. 2.88–2.89)}
    $$

    Physically, the surface area exposed to a given amount of light grows
    at oblique angles, becoming fully self-shadowed once $\hat v_i \cdot
    \hat n \le 0$ — this is why $L_d$ is *view-independent*:
    it doesn't depend on $\hat v_r$ at all, only on the light direction and
    the surface normal.
    """)
    return


@app.cell
def _(np):
    def light_dir(az_deg, el_deg):
        """Unit direction toward the light, from horizon-style azimuth/elevation (same convention as the orbit camera in Notebook 4)."""
        az, el = np.radians(az_deg), np.radians(el_deg)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def sphere_normals(n_px=220):
        """Per-pixel outward normal for an orthographic view of a unit sphere; NaN outside the silhouette."""
        lin = np.linspace(-1, 1, n_px)
        px, py = np.meshgrid(lin, lin)
        r2 = px**2 + py**2
        mask = r2 <= 1.0
        pz = np.sqrt(np.clip(1 - r2, 0, None))
        return px, py, pz, mask

    return light_dir, sphere_normals


@app.cell
def _(mo):
    diff_az_slider = mo.ui.slider(start=-180, stop=180, value=35, step=5, label="light azimuth")
    diff_el_slider = mo.ui.slider(start=-80, stop=80, value=40, step=5, label="light elevation")
    kd_slider = mo.ui.slider(start=0.0, stop=1.0, value=0.8, step=0.05, label="albedo fd")
    Li_slider = mo.ui.slider(start=0.2, stop=1.5, value=1.0, step=0.1, label="light intensity Li")
    mo.hstack([diff_az_slider, diff_el_slider, kd_slider, Li_slider], justify="start", gap=2)
    return Li_slider, diff_az_slider, diff_el_slider, kd_slider


@app.cell
def _(
    Li_slider,
    diff_az_slider,
    diff_el_slider,
    kd_slider,
    light_dir,
    mo,
    np,
    plt,
    sphere_normals,
):
    _px, _py, _pz, _mask = sphere_normals()
    _l = light_dir(diff_az_slider.value, diff_el_slider.value)
    _ndotl = _px * _l[0] + _py * _l[1] + _pz * _l[2]
    _Ld = kd_slider.value * Li_slider.value * np.clip(_ndotl, 0, None)
    _img = np.where(_mask, _Ld, np.nan)

    _theta_marker = np.degrees(np.arccos(np.clip(_l[2], -1, 1)))
    _thetas = np.linspace(-90, 90, 200)
    _curve = np.clip(np.cos(np.radians(_thetas)), 0, None)
    _marker_val = np.clip(np.cos(np.radians(_theta_marker)), 0, None)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _im = _ax1.imshow(_img, cmap="gray", vmin=0, vmax=1, origin="lower")
    _ax1.set_xticks([])
    _ax1.set_yticks([])
    _ax1.set_title("Lambertian-shaded sphere")

    _ax2.plot(_thetas, _curve, color="tab:orange")
    _ax2.scatter([_theta_marker], [_marker_val], color="black", zorder=3)
    _ax2.set_xlabel("θi (deg)")
    _ax2.set_ylabel("cos⁺ θi")
    _ax2.set_title("foreshortening term")
    _ax2.grid(True, linewidth=0.3)
    _fig.tight_layout()

    _readout = mo.md(f"At the sphere's front-facing point: θi ≈ **{_theta_marker:.1f}°**, brightness = **{_marker_val * kd_slider.value * Li_slider.value:.2f}** — matches the render's center pixel.")
    mo.vstack([_readout, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the right-hand plot.** The orange **curve** is the generic
    shape of $\cos^+\theta_i$ itself, plotted for every possible incidence
    angle from $-90°$ to $90°$ — it doesn't refer to any one point on the
    sphere. It's $1$ at $\theta_i=0$ (light straight on), falls off
    smoothly as the light direction tilts away from the normal, and is
    clamped to exactly $0$ once $|\theta_i|\ge 90°$ (light behind the
    surface — see the discussion above).

    The **black dot** marks one specific sample on that curve: the
    incidence angle $\theta_i$ *at the sphere's front-facing point* — the
    point whose outward normal $\hat n=(0,0,1)$ points straight at the
    viewer, i.e. the **center pixel** of the render on the left — for
    whatever light azimuth/elevation the sliders are currently set to.
    Its height on the curve, $\cos^+\theta_i$, is exactly the diffuse
    reflectance at that point, and (after scaling by $f_d$ and $L_i$) is
    the number reported in the readout, which always matches the center
    pixel's brightness.

    Every *other* point on the sphere has a different surface normal, so
    it sits at a *different* $\theta_i$ on this same curve — that's the
    whole reason the render is shaded rather than a flat disk of gray. The
    dot is just one concrete example letting you connect an abstract
    angle on the curve to an actual pixel in the picture; try dragging the
    light sliders and watching the dot slide along the curve while the
    center of the sphere brightens and dims to match.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Specular reflection

    The second major BRDF component depends strongly on the *outgoing*
    direction. For a perfect mirror, the incident ray is reflected about
    the surface normal into the direction

    $$
    \hat s_i = \hat v_{i,\parallel} - \hat v_{i,\perp} = (2\hat n \hat n^\top - I)\, \hat v_i \qquad \text{(Eq. 2.90)}
    $$

    **Where this comes from.** Any unit vector can be split into two
    pieces relative to the surface normal $\hat n$:

    - $\hat v_\parallel = (\hat v \cdot \hat n)\,\hat n$ — the component
      of $\hat v_i$ **along** the normal;
    - $\hat v_\perp = \hat v - \hat v_\parallel$ — the remaining
      component, lying **in the tangent plane** of the surface
      (perpendicular to $\hat n$),

    so that $\hat v = \hat v_\parallel + \hat v_\perp$. A perfect mirror
    reflects a ray the way a ball bounces off a flat floor: the part of
    its direction sticking *out* of the surface ($\hat v_{i,\parallel}$) is
    unchanged, while the part running *along* the surface ($\hat
    v_{i,\perp}$) is flipped $180°$ to the opposite side. That's exactly

    $$
    \hat s_i = \hat v_{i,\parallel} - \hat v_{i,\perp},
    $$

    and substituting $\hat v_{i,\parallel} = (\hat v_i\cdot\hat n)\hat n$ and
    $\hat v_{i,\perp} = \hat v_i - (\hat v_i\cdot\hat n)\hat n$ gives
    $\hat s_i = 2(\hat v_i\cdot\hat n)\hat n - \hat v_i = (2\hat n\hat
    n^\top - I)\hat v_i$ — the matrix form above.

    The live diagram below decomposes $\hat v_i$ into $\hat v_{i,\parallel}$ and $\hat v_{i,\perp}$ (orange),
    and shows the mirrored $-\hat v_{i,\perp}$ (light blue) that combines with
    the unchanged $\hat v_{i,\parallel}$ to build $\hat s_i$ (blue). Drag to
    rotate; try an elevation below $0°$ to see $\hat v_\parallel$ point
    *into* the surface — exactly the "light is behind the surface" case
    from §4 where $\cos^+\theta_i$ clamped to zero.
    """)
    return


@app.cell
def _(mo):
    v_az_slider = mo.ui.slider(start=-180, stop=180, value=35, step=5, label="light azimuth")
    v_el_slider = mo.ui.slider(start=-80, stop=80, value=40, step=5, label="light elevation")

    mo.hstack([v_az_slider, v_el_slider], justify="start", gap=2)
    return v_az_slider, v_el_slider


@app.cell
def _(brdf_arrow, go, light_dir, mo, np, v_az_slider, v_el_slider):
    _vi = light_dir(v_az_slider.value, v_el_slider.value)
    _n = np.array([0.0, 0.0, 1.0])
    _v_par = (_vi @ _n) * _n
    _v_perp = _vi - _v_par
    _si = _v_par - _v_perp

    _xx, _yy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
    _traces = [
        go.Surface(
            x=_xx, y=_yy, z=np.zeros_like(_xx), showscale=False,
            colorscale=[[0, "tan"], [1, "tan"]], opacity=0.25, hoverinfo="skip",
        )
    ]
    _traces += brdf_arrow(_n, "black", "n")
    _traces += brdf_arrow(_vi, "red", "vi")
    _traces += brdf_arrow(_si, "blue", "si")

    _par_norm = np.linalg.norm(_v_par)
    _perp_norm = np.linalg.norm(_v_perp)
    if _par_norm > 1e-6:
        _traces += brdf_arrow(_v_par / _par_norm, "darkorange", "vi∥", length=_par_norm, cone_size=0.18)
    if _perp_norm > 1e-6:
        _perp_hat = _v_perp / _perp_norm
        _traces += brdf_arrow(_perp_hat, "darkorange", "vi⊥", length=_perp_norm, cone_size=0.18)
        _traces += brdf_arrow(-_perp_hat, "steelblue", "−vi⊥", length=_perp_norm, cone_size=0.18)

    # dotted parallelogram edges: v_i = v_par + v_perp, s_i = v_par + (-v_perp)
    _traces.append(go.Scatter3d(
        x=[_v_par[0], _vi[0]], y=[_v_par[1], _vi[1]], z=[_v_par[2], _vi[2]],
        mode="lines", line=dict(color="gray", width=2, dash="dot"), showlegend=False, hoverinfo="skip",
    ))
    _traces.append(go.Scatter3d(
        x=[_v_par[0], _si[0]], y=[_v_par[1], _si[1]], z=[_v_par[2], _si[2]],
        mode="lines", line=dict(color="gray", width=2, dash="dot"), showlegend=False, hoverinfo="skip",
    ))

    _fig = go.Figure(data=_traces)
    _fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], title="x"),
            yaxis=dict(range=[-1.2, 1.2], title="y"),
            zaxis=dict(range=[-1.2, 1.2], title="z"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        title="mirror reflection: vi = v∥ + v⊥,  si = v∥ − v⊥ (drag to rotate)",
    )

    _readout = mo.md(f"vi = ({_vi[0]:.2f}, {_vi[1]:.2f}, {_vi[2]:.2f})  ·  si = ({_si[0]:.2f}, {_si[1]:.2f}, {_si[2]:.2f})")
    mo.vstack([_readout, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The amount of light reflected toward the viewer $\hat v_r$ then
    depends on the angle $\theta_s = \cos^{-1}(\hat v_r \cdot \hat s_i)$
    between the view direction and this mirror direction. The **Phong**
    model uses a power of the cosine of that angle,

    $$
    f_s(\theta_s;\lambda) = k_s(\lambda)\, \cos^{k_e}\theta_s \qquad \text{(Eq. 2.91)}
    $$

    Larger exponents $k_e$ produce tighter, shinier highlights; smaller
    exponents spread the highlight into a softer gloss.
    """)
    return


@app.cell
def _(mo):
    spec_az_slider = mo.ui.slider(start=-180, stop=180, value=35, step=5, label="light azimuth")
    spec_el_slider = mo.ui.slider(start=-80, stop=80, value=40, step=5, label="light elevation")
    ks_slider = mo.ui.slider(start=0.0, stop=1.0, value=1.0, step=0.05, label="ks")
    ke_slider = mo.ui.slider(start=2, stop=400, value=60, step=2, label="Phong exponent ke")
    mo.hstack([spec_az_slider, spec_el_slider, ks_slider, ke_slider], justify="start", gap=2)
    return ke_slider, ks_slider, spec_az_slider, spec_el_slider


@app.cell
def _(
    ke_slider,
    ks_slider,
    light_dir,
    mo,
    np,
    plt,
    spec_az_slider,
    spec_el_slider,
    sphere_normals,
):
    _px, _py, _pz, _mask = sphere_normals()
    _l = light_dir(spec_az_slider.value, spec_el_slider.value)
    _ndotl = _px * _l[0] + _py * _l[1] + _pz * _l[2]
    # s_i = (2 n n^T - I) v_i, evaluated per-pixel (n varies, v_i = l is constant)
    _sx = 2 * _ndotl * _px - _l[0]
    _sy = 2 * _ndotl * _py - _l[1]
    _sz = 2 * _ndotl * _pz - _l[2]
    # viewer looks straight down +z (orthographic), so v_r . s_i = s_z = cos(theta_s)
    _cos_theta_s = _sz
    _spec = ks_slider.value * np.clip(_cos_theta_s, 0, None) ** ke_slider.value
    _img = np.where(_mask, _spec, np.nan)

    _fig = plt.figure(figsize=(9.5, 4.2))
    _ax1 = _fig.add_subplot(1, 2, 1)
    _ax1.imshow(_img, cmap="gray", vmin=0, vmax=1, origin="lower")
    _ax1.set_xticks([])
    _ax1.set_yticks([])
    _ax1.set_title("specular-only sphere (highlight)")

    _ax2 = _fig.add_subplot(1, 2, 2, projection="polar")
    _theta_deg = np.linspace(-90, 90, 181)
    _theta_rad = np.radians(_theta_deg)
    for _ke_ref in (10, 100, 1000):
        _lobe_ref = np.clip(np.cos(_theta_rad), 0, None) ** _ke_ref
        _ax2.plot(_theta_rad, _lobe_ref, "--", color="gray", linewidth=1, label=f"ke={_ke_ref}")
    _lobe_cur = np.clip(np.cos(_theta_rad), 0, None) ** ke_slider.value
    _ax2.plot(_theta_rad, _lobe_cur, color="tab:red", linewidth=2.5, label=f"ke={ke_slider.value} (current)")
    _ax2.set_theta_zero_location("N")
    _ax2.set_theta_direction(-1)
    _ax2.set_thetamin(-90)
    _ax2.set_thetamax(90)
    _ax2.set_title("specular lobe fs(θs) (cf. Fig. 2.18b)", pad=20)
    _ax2.legend(loc="lower left", fontsize=7)
    _fig.tight_layout()

    _readout = mo.md(f"ks = **{ks_slider.value:.2f}**, ke = **{ke_slider.value}** — larger ke ⇒ tighter, shinier highlight.")
    mo.vstack([_readout, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the specular lobe plot.** This is a polar plot of $f_s(\theta_s) = \cos^{k_e}\theta_s$
    itself (Eq. 2.91, with $k_s{=}1$ so only the *shape* is shown) — the
    angular profile of specular brightness, independent of any particular
    point on the sphere. The angle around the plot is $\theta_s$, the
    angle between the viewer and the mirror direction $\hat s_i$, with
    $\theta_s = 0$ pointing straight up (the exact mirror direction) and
    sweeping to $\pm 90°$ on either side; the radius at each angle is the
    value of $f_s(\theta_s)$, from $0$ at the center to $1$ at the rim.

    The three dashed gray curves are **fixed references** at $k_e = 10,
    100, 1000$, always drawn for comparison. The solid red curve is the
    **current** $k_e$ from the slider above. A small exponent traces a
    wide, blunt petal — light stays bright over a broad range of viewing
    angles around $\hat s_i$, i.e. a soft, spread-out gloss. A large
    exponent traces a narrow spike hugging $\theta_s=0$ — brightness
    collapses to almost nothing just a few degrees off the mirror
    direction, i.e. a sharp, mirror-like highlight. This lobe *is* the
    angular falloff that produces the bright spot in the render on the
    left: dragging $k_e$ up narrows both the lobe here and the highlight
    there, for the same reason.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A light source's color is a distribution over wavelength; the
      visible spectrum spans roughly **380–750 nm**.
    - A **point light source** has a position, intensity, and spectrum
      $L(\lambda)$, with intensity falling off as $1/r^2$. An
      **environment (area) light source** instead assigns a radiance to
      every incident direction, $L(\hat v;\lambda)$.
    - The **BRDF** $f_r(\theta_i,\phi_i,\theta_r,\phi_r;\lambda)$ is the
      most general description of how a surface scatters light; for
      **isotropic** materials it collapses to depend only on
      $\theta_i,\theta_r,|\phi_r-\phi_i|$.
    - **Diffuse (Lambertian) reflection**: constant BRDF $f_d(\lambda)$,
      view-independent radiance governed entirely by the foreshortening
      term $[\hat v_i\cdot\hat n]^+$.
    - **Specular reflection**: radiance concentrated around the mirror
      direction $\hat s_i = (2\hat n\hat n^\top - I)\hat v_i$, with the
      Phong exponent $k_e$ controlling highlight tightness.

    **Next up:** this radiance still has to pass through a lens before it
    reaches the sensor. The next notebook will cover optics, the last piece of the physical path
    from a light source to a raw pixel value.
    """)
    return


if __name__ == "__main__":
    app.run()
