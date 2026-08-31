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
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Sensors and Color: From Photons to RGB (and Beyond)

    Notebook 6 ended with light focused onto the sensor plane by a lens. This
    notebook picks up right there and asks: how does a sensor turn that
    continuous, focused irradiance into the discrete $(R,G,B)$ numbers that
    make up a digital image — and once we have those numbers, what other
    ways can we represent color?

    This follows Szeliski §2.3, "The digital camera," but selectively:

    - A **brief overview** of the sensing pipeline and sensor hardware
      (§2.3, plus a peek at what §2.3.1 covers).
    - We're **skipping the sampling-and-aliasing math** of §2.3.1 — that's
      really a signal-processing topic about filters and resampling, and it
      fits naturally once we build spatial filters in the *next* notebook.
    - Most of our time goes to §2.3.2, **Color**: how RGB relates to human
      color perception through CIE XYZ, and the perceptually-motivated
      **CIELAB** space — with interactive demos throughout.

    No Predict/Run/Investigate activities in this notebook — just exploration.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. From photons to a JPEG — the sensing pipeline

    Once light lands on the sensor, a camera runs it through a whole
    processing pipeline before you ever see a pixel value:

    | Stage | What happens |
    |---|---|
    | Optics → Aperture → Shutter | The lens (notebook 6) focuses light; the aperture and shutter control *how much* light and for *how long*. |
    | Sensor (CCD/CMOS) → Gain (ISO) → ADC | Photons become an analog electrical charge, get amplified, and are digitized — the result is a **RAW** image. |
    | Demosaic → Denoise & sharpen | Missing color samples (see the Bayer pattern below) are interpolated; noise is suppressed. |
    | White balance → Gamma/curve → Compress | Colors are corrected for the scene's lighting, remapped through a non-linear curve, and compressed — the result is the **JPEG** you actually open. |

    Figure 2.23 below lays this out as a block diagram.
    """)
    return


@app.cell
def _(IMAGES_DIR, mo):
    mo.vstack([
        mo.image(
            src=str(IMAGES_DIR / "textbook_figures" / "szeliski_fig2_23_sensing_pipeline.png"),
            width=560,
        ),
        mo.md(
            "**Figure 2.23**, reproduced from Richard Szeliski, *Computer Vision: "
            "Algorithms and Applications*, 2nd ed. (2022), for educational use. "
            "See `images/SOURCES.md` for the full citation."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sensor hardware: CCD vs. CMOS

    The two dominant sensor technologies work differently at the pixel
    level, even though both start by converting photons into electrons:

    - **CCD** (charge-coupled device): each pixel accumulates charge during
      exposure, and then charges are shuffled from pixel to pixel in a kind
      of "bucket brigade" until they reach a single set of amplifiers at the
      edge of the chip, which convert charge to voltage.
    - **CMOS**: each pixel converts its own charge to voltage *locally*,
      right where it's collected, and the results are read out through a
      multiplexing scheme — no bucket brigade needed. This is what almost
      every phone and modern camera uses today.

    Figure 2.24(a) below shows this side by side; Figure 2.24(b) is a
    cutaway of a single CMOS pixel, showing the **microlens** that
    concentrates incoming light onto a much smaller **photodiode**, and the
    **color filter** (here, red) that only lets one color of light through
    to that particular pixel.
    """)
    return


@app.cell
def _(IMAGES_DIR, mo):
    mo.vstack([
        mo.hstack([
            mo.image(src=str(IMAGES_DIR / "textbook_figures" / "szeliski_fig2_24a_ccd_cmos.png"), width=360),
            mo.image(src=str(IMAGES_DIR / "textbook_figures" / "szeliski_fig2_24b_cmos_cutaway.png"), width=280),
        ], justify="start", gap=2),
        mo.md(
            "**Figure 2.24(a, b)**, reproduced from Szeliski (2022) Figure 2.24. "
            "Panel (a) is itself reproduced there from Litwiller (2005) © 2005 "
            "Photonics Spectra; panel (b) is from "
            "[micro.magnet.fsu.edu](https://micro.magnet.fsu.edu/primer/digitalimaging/cmosimagesensors.html). "
            "Full attribution chain in `images/SOURCES.md`."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Global vs. rolling shutter.** The CCD/CMOS distinction also decides
    *when* each pixel gets exposed, which matters a lot for anything that
    moves:

    - **Global shutter:** every pixel starts and stops integrating light at
      exactly the same instant, and the whole frame is read out afterward.
      This is what a CCD's charge-transfer design naturally gives you (the
      "bucket brigade" only starts moving *after* the entire chip has
      finished exposing), and standard digital-still capture on any sensor
      normally works this way too.
    - **Rolling shutter:** rows are exposed and read out one at a time, in
      sequence (top to bottom), rather than all at once. Because each
      pixel converts charge to voltage locally, most CMOS sensors are
      built this way by default — it's cheaper and needs far less
      per-pixel circuitry than storing every pixel's charge until a
      simultaneous global readout. "Global shutter CMOS" sensors do exist,
      but need extra transistors at every pixel to hold the charge, which
      costs money and chip area.

    The practical consequence: with a rolling shutter, different rows of a
    single frame were actually captured at *slightly different times*. For
    a static scene this is invisible, but a fast-moving subject, a quick
    camera pan, or a propeller/fan blade can end up **skewed, wobbled, or
    smeared** ("jello effect"), since the top of the frame reflects an
    earlier moment than the bottom. This is exactly why most video is shot
    with a rolling-shutter CMOS sensor but professional cinema/broadcast
    cameras and machine-vision systems often pay extra for global shutter.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **So how does one sensor, with one photodiode per pixel, produce a
    *color* image?** It doesn't measure color directly — each pixel is
    covered by a single color filter (red, green, or blue), arranged in a
    checkerboard-like mosaic called a **color filter array (CFA)**. The
    most common layout is the **Bayer pattern**, shown below: twice as many
    green filters as red or blue, since the human visual system is far more
    sensitive to detail in luminance (mostly carried by green) than in
    color.
    """)
    return


@app.cell
def _(mo, np, plt):
    _pattern = np.array([
        [[1.0, 0.55, 0.55], [0.55, 1.0, 0.55], [1.0, 0.55, 0.55], [0.55, 1.0, 0.55]],
        [[0.55, 1.0, 0.55], [0.55, 0.55, 1.0], [0.55, 1.0, 0.55], [0.55, 0.55, 1.0]],
        [[1.0, 0.55, 0.55], [0.55, 1.0, 0.55], [1.0, 0.55, 0.55], [0.55, 1.0, 0.55]],
        [[0.55, 1.0, 0.55], [0.55, 0.55, 1.0], [0.55, 1.0, 0.55], [0.55, 0.55, 1.0]],
    ])
    _labels = [["R", "G", "R", "G"], ["G", "B", "G", "B"], ["R", "G", "R", "G"], ["G", "B", "G", "B"]]

    _fig, _ax = plt.subplots(figsize=(3.2, 3.2))
    _ax.imshow(_pattern, extent=(0, 4, 0, 4))
    for _i in range(4):
        for _j in range(4):
            _ax.text(_j + 0.5, 4 - _i - 0.5, _labels[_i][_j], ha="center", va="center", fontsize=14, fontweight="bold")
    _ax.set_xticks(range(5))
    _ax.set_yticks(range(5))
    _ax.grid(True, color="white", linewidth=2)
    _ax.set_xticklabels([])
    _ax.set_yticklabels([])
    _ax.set_title("Bayer color filter array (RGGB)")

    mo.vstack([_fig, mo.md(
        "Every pixel only ever measures **one** color; the other two are "
        "filled in by interpolating neighboring pixels — a process called "
        "**demosaicing** (one of the ISP steps in Figure 2.23 above)."
    )])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. RGB and human color vision

    Why do three numbers — red, green, blue — suffice to describe almost
    any color we can perceive? Not because light itself is made of three
    wavelengths, but because of *us*: the human retina has three types of
    color-sensitive cone cells, each responding most strongly to a
    different part of the visible spectrum. This **trichromatic** nature of
    human vision is the entire reason additive color (mixing red, green,
    and blue light) and subtractive color (mixing cyan, magenta, and
    yellow pigments) both work, and why cameras and monitors standardized
    on three color channels.

    In the 1930s, the CIE (Commission Internationale d'Éclairage) formalized
    this by having human observers match every visible color using just
    three fixed primaries (700.0 nm red, 546.1 nm green, 435.8 nm blue),
    producing the historical **CIE RGB** color-matching functions. Because
    matching some colors required a *negative* amount of one primary
    (impossible to actually display), the CIE also defined a purely
    mathematical space called **XYZ**, built as a linear transform of CIE
    RGB, that contains every visible color with non-negative coordinates.

    Modern cameras and monitors don't use the 1930s CIE RGB primaries —
    they use standardized primaries closer to real phosphors/filters,
    codified as **ITU-R BT.709** (the same primaries as sRGB), with a fixed
    RGB→XYZ matrix. That's the transform the interactive demo below
    actually uses.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. From RGB to CIE XYZ to CIELAB

    Real image files store **gamma-compressed** RGB (denoted $R'G'B'$) —
    values already passed through a non-linear curve, roughly
    $R' \approx R^{1/2.2}$, for reasons covered later (§2.3.2's gamma
    discussion — perceptual efficiency and backward-compatibility with old
    CRTs). To get physically meaningful XYZ values, we first need to
    **undo** that curve (the standard sRGB curve, a close cousin of the
    simple power law above), then apply the BT.709 primaries matrix:

    $$
    \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
    =
    \begin{bmatrix}
    0.412453 & 0.357580 & 0.180423 \\
    0.212671 & 0.715160 & 0.072169 \\
    0.019334 & 0.119193 & 0.950227
    \end{bmatrix}
    \begin{bmatrix} R \\ G \\ B \end{bmatrix}
    $$

    ($R,G,B$ here are the *linear*, gamma-decoded values in $[0,1]$.)

    XYZ separates brightness ($Y$) from color, but it still isn't
    **perceptually uniform** — equal steps in XYZ don't look like equal
    steps in color to a human eye. **CIELAB** ($L^*a^*b^*$) fixes this with
    a non-linear remapping:

    $$
    L^* = 116\,f\!\left(\frac{Y}{Y_n}\right) - 16, \qquad
    a^* = 500\left[f\!\left(\frac{X}{X_n}\right) - f\!\left(\frac{Y}{Y_n}\right)\right], \qquad
    b^* = 200\left[f\!\left(\frac{Y}{Y_n}\right) - f\!\left(\frac{Z}{Z_n}\right)\right]
    $$

    where $(X_n,Y_n,Z_n)$ is the white point and $f(t)=t^{1/3}$ for
    $t>\delta^3$ (a finite-slope linear approximation otherwise, with
    $\delta=6/29$). $L^*$ (0–100) is lightness; $a^*$ is a **green↔red**
    axis and $b^*$ is a **blue↔yellow** axis, both roughly signed and
    unbounded (commonly around ±100 for real colors).

    Try it below: the sliders set an $R,G,B$ color (0–255, the numbers
    you'd see in any image editor); the readout shows every stage of the
    pipeline above for that exact color.
    """)
    return


@app.cell
def _(np):
    _M_RGB_TO_XYZ = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ])
    _WHITE = np.array([0.950456, 1.0, 1.088754])  # D65, Y normalized to 1
    _DELTA = 6 / 29

    def srgb_to_linear(c):
        """Undo the sRGB gamma curve. c in [0,1], any shape."""
        c = np.asarray(c, dtype=float)
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    def rgb_to_xyz(rgb_linear):
        """Linear RGB (..., 3) -> XYZ (..., 3), via the BT.709 matrix (Szeliski Eq. 2.109)."""
        return rgb_linear @ _M_RGB_TO_XYZ.T

    def _lab_f(t):
        return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4 / 29)

    def xyz_to_lab(xyz):
        """XYZ (..., 3) -> CIELAB (..., 3), via Szeliski Eq. 2.106-2.108."""
        fx, fy, fz = _lab_f(xyz[..., 0] / _WHITE[0]), _lab_f(xyz[..., 1] / _WHITE[1]), _lab_f(xyz[..., 2] / _WHITE[2])
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    def rgb255_to_lab(rgb255):
        """Convenience: gamma-encoded RGB in [0,255] (..., 3) all the way to Lab."""
        lin = srgb_to_linear(np.asarray(rgb255, dtype=float) / 255.0)
        return xyz_to_lab(rgb_to_xyz(lin))

    return rgb_to_xyz, srgb_to_linear, xyz_to_lab


@app.cell
def _(mo):
    R_slider = mo.ui.slider(start=0, stop=255, value=200, step=1, label="R")
    G_slider = mo.ui.slider(start=0, stop=255, value=90, step=1, label="G")
    B_slider = mo.ui.slider(start=0, stop=255, value=40, step=1, label="B")
    mo.hstack([R_slider, G_slider, B_slider], justify="start", gap=2)
    return B_slider, G_slider, R_slider


@app.cell
def _(
    B_slider,
    G_slider,
    R_slider,
    mo,
    np,
    plt,
    rgb_to_xyz,
    srgb_to_linear,
    xyz_to_lab,
):
    _rgb255 = np.array([R_slider.value, G_slider.value, B_slider.value])
    _rgb01 = _rgb255 / 255.0
    _lin = srgb_to_linear(_rgb01)
    _xyz = rgb_to_xyz(_lin)
    _lab = xyz_to_lab(_xyz)

    _fig, _ax = plt.subplots(figsize=(2, 2))
    _ax.imshow([[_rgb01]])
    _ax.axis("off")
    _ax.set_title("swatch")

    _readout = mo.md(f"""
    | | R | G | B |
    |---|---|---|---|
    | gamma-encoded (0–255) | {_rgb255[0]} | {_rgb255[1]} | {_rgb255[2]} |
    | linear (0–1) | {_lin[0]:.3f} | {_lin[1]:.3f} | {_lin[2]:.3f} |

    **XYZ:** X={_xyz[0]:.3f}, Y={_xyz[1]:.3f}, Z={_xyz[2]:.3f}

    **CIELAB:** L\\*={_lab[0]:.1f}, a\\*={_lab[1]:.1f}, b\\*={_lab[2]:.1f}
    """)

    mo.hstack([_fig, _readout], justify="start", align="center", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A few things worth trying

    - Set R=G=B (any gray). What happens to $a^*$ and $b^*$? Why does that
      make sense given they're defined as *differences* between the
      $f(\cdot)$-remapped channels?
    - Find a color where $a^*$ is strongly negative (green) versus
      strongly positive (red) — does $b^*$ change much? Are $a^*$ and
      $b^*$ behaving like independent axes?
    - Keep the *ratio* of R:G:B fixed but scale all three down together
      (dimmer, same hue). Does $L^*$ change roughly linearly, or does the
      cube-root in $f(t)$ show up?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Color-channel decomposition of a real image

    The same pipeline applies pixel-by-pixel to a whole photograph. Below,
    pick one of four standard test images and see it broken into its R,
    G, B channels (top row) and its L\*, a\*, b\* channels (bottom row).
    The $a^*$ and $b^*$ panels use **diverging colormaps** — green↔red and
    blue↔yellow — centered at zero, so the sign of each pixel's value is
    immediately visible (something a grayscale rendering can't show).
    """)
    return


@app.cell
def _(mo):
    image_dropdown = mo.ui.dropdown(options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="test image")
    image_dropdown
    return (image_dropdown,)


@app.cell
def _(
    IMAGES_DIR,
    LinearSegmentedColormap,
    image_dropdown,
    mo,
    plt,
    rgb_to_xyz,
    srgb_to_linear,
    xyz_to_lab,
):
    _img = plt.imread(str(IMAGES_DIR / f"{image_dropdown.value}.png"))[:, :, :3]
    _lab_img = xyz_to_lab(rgb_to_xyz(srgb_to_linear(_img)))
    _L, _a, _b = _lab_img[..., 0], _lab_img[..., 1], _lab_img[..., 2]
    _a_max = max(abs(_a.min()), abs(_a.max()))
    _b_max = max(abs(_b.min()), abs(_b.max()))

    _green_red = LinearSegmentedColormap.from_list("green_red", ["#1a7a1a", "white", "#c0392b"])
    _blue_yellow = LinearSegmentedColormap.from_list("blue_yellow", ["#1a4fa0", "white", "#d4a017"])

    _fig1, _axes1 = plt.subplots(1, 4, figsize=(13, 3.6))
    _axes1[0].imshow(_img)
    _axes1[0].set_title("Color (RGB)")
    _axes1[1].imshow(_img[:, :, 0], cmap="gray", vmin=0, vmax=1)
    _axes1[1].set_title("R")
    _axes1[2].imshow(_img[:, :, 1], cmap="gray", vmin=0, vmax=1)
    _axes1[2].set_title("G")
    _axes1[3].imshow(_img[:, :, 2], cmap="gray", vmin=0, vmax=1)
    _axes1[3].set_title("B")
    for _ax in _axes1:
        _ax.axis("off")
    _fig1.tight_layout()

    _fig2, _axes2 = plt.subplots(1, 3, figsize=(10, 3.6))
    _axes2[0].imshow(_L, cmap="gray", vmin=0, vmax=100)
    _axes2[0].set_title("L* (lightness)")
    _axes2[1].imshow(_a, cmap=_green_red, vmin=-_a_max, vmax=_a_max)
    _axes2[1].set_title("a* (green ↔ red)")
    _axes2[2].imshow(_b, cmap=_blue_yellow, vmin=-_b_max, vmax=_b_max)
    _axes2[2].set_title("b* (blue ↔ yellow)")
    for _ax in _axes2:
        _ax.axis("off")
    _fig2.tight_layout()

    mo.vstack([_fig1, _fig2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A camera's **sensing pipeline** runs light through optics, an
      aperture and shutter, a **CCD or CMOS sensor**, gain, and an ADC to
      produce a RAW image, then an **image signal processor** (demosaic,
      denoise, white balance, gamma, compress) to produce the JPEG you see.
    - Color sensing relies on a **color filter array** (usually the Bayer
      pattern) since each pixel can only measure one color; the rest is
      **demosaiced** (interpolated).
    - RGB works because human vision is **trichromatic** (three cone
      types); the CIE formalized this into the **XYZ** color space, a
      linear transform of RGB that contains every visible color with
      non-negative coordinates.
    - **CIELAB** ($L^*a^*b^*$) is a non-linear remapping of XYZ designed to
      be **perceptually uniform** — $L^*$ is lightness, $a^*$ is
      green↔red, $b^*$ is blue↔yellow.
    - We set aside §2.3.1 (**sampling and aliasing**) on purpose — it's a
      filtering question.

    **Next up:** point and spatial filtering — the operations (blurring,
    sharpening, edge detection, and yes, resampling and aliasing) that work
    directly on the pixel grids we've been building and decomposing this
    notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
