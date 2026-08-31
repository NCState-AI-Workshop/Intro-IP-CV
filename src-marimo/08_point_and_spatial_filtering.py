# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
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
    from scipy.signal import correlate2d
    from scipy.ndimage import gaussian_filter

    return correlate2d, gaussian_filter, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Point Operators and Linear Filtering

    Notebook 7 got us to a real RGB image — a grid of numbers. Now we start
    Chapter 3 of Szeliski, *Image processing*: the operators that turn a raw
    pixel grid into something more useful, either for a viewer or for a
    later algorithm.

    This notebook covers, selectively:

    - **§3.1** Point operators, and specifically **§3.1.1** pixel transforms
      and **§3.1.4** histogram equalization (including its locally-adaptive
      variant).
    - The **start of §3.2**, Linear filtering — correlation, convolution,
      and four workhorse kernels: the **box filter**, the **Gaussian
      filter**, **directional derivatives**, and the **Laplacian**.

    Two **Predict → Run → Investigate** activities, plus a **Modify**
    activity where you'll hand-edit an actual filter kernel and watch the
    result update live.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the common, simple convention for basic image
    processing — not the full colorimetric Y from notebook 7)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Point operators

    The simplest image-processing operators are **point operators** (also
    called point processes): each output pixel depends *only* on the
    corresponding input pixel,

    $$ g(i,j) = h(f(i,j)), $$

    not on any of its neighbors. Brightness and contrast adjustments,
    gamma correction, and histogram equalization are all point operators —
    even though, as we'll see with histogram equalization, the function
    $h$ itself can depend on statistics gathered from the *whole* image.
    (Contrast this with the **neighborhood operators** in §3.2 below, where
    a pixel's neighbors matter too.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Pixel transforms

    Two of the most common point operators are multiplication and addition
    by constants,

    $$ g(x) = a\,f(x) + b, $$

    where $a$ is called the **gain** (controls contrast) and $b$ the
    **bias** (controls brightness). A related operator, the **linear
    blend**, cross-dissolves between two images, $g=(1-\alpha)f_0+\alpha
    f_1$ — the basis of slideshow transitions and morphing.

    A third, *non-linear* pixel transform is **gamma correction**,

    $$ g(x) = f(x)^{1/\gamma}, $$

    used to undo (or apply) the non-linear response of sensors and
    displays (notebook 7's sRGB curve is a real-world example, with
    $\gamma\approx 2.2$).

    ### Predict

    Suppose you only raise the bias $b$ (brighten the image) without
    touching the gain. Can an already-bright pixel keep getting brighter
    forever as you keep raising $b$, or does something eventually give?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        Pixel values have to live in a fixed range (0–255 for an 8-bit
        image). Once $a f(x) + b$ would exceed 255, the result gets
        **clipped** back down to 255 — it can't go any higher. So a bright
        region stops changing at all once it saturates, while a *dark*
        region can still keep climbing. The practical effect: raising $b$
        too far doesn't make the image "brighter" so much as it **crushes
        the highlights** — everything that was already fairly bright
        merges into a single flat white region, losing detail.

        More generally: **gain** $a$ stretches or compresses the spread of
        the histogram (contrast), while **bias** $b$ shifts the whole
        histogram left or right (brightness) — and both can push values
        outside $[0,255]$, where they get clipped.
        """)
    })
    return


@app.cell
def _(mo):
    image_dropdown_transform = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    gain_slider = mo.ui.slider(start=0.2, stop=3.0, value=1.0, step=0.05, label="gain a", debounce=True)
    bias_slider = mo.ui.slider(start=-150, stop=150, value=0, step=5, label="bias b", debounce=True)
    gamma_slider = mo.ui.slider(start=0.3, stop=3.0, value=1.0, step=0.05, label="gamma γ", debounce=True)
    mo.vstack([image_dropdown_transform, mo.hstack([gain_slider, bias_slider, gamma_slider], justify="start", gap=2)])
    return bias_slider, gain_slider, gamma_slider, image_dropdown_transform


@app.cell
def _(
    IMAGES_DIR,
    bias_slider,
    gain_slider,
    gamma_slider,
    image_dropdown_transform,
    mo,
    np,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_transform.value, plt, np)
    _linear = np.clip(gain_slider.value * _img_rgb + bias_slider.value, 0, 255)
    _out = 255.0 * (_linear / 255.0) ** (1.0 / gamma_slider.value)
    _out = np.clip(_out, 0, 255)

    _fig, _axes = plt.subplots(2, 2, figsize=(9, 7))
    _axes[0, 0].imshow(_img_rgb.astype(np.uint8))
    _axes[0, 0].set_title("original")
    _axes[0, 0].axis("off")
    _axes[0, 1].imshow(_out.astype(np.uint8))
    _axes[0, 1].set_title(f"a={gain_slider.value:.2f}, b={bias_slider.value:.0f}, γ={gamma_slider.value:.2f}")
    _axes[0, 1].axis("off")

    for _ch, _color, _label in [(0, "tab:red", "R"), (1, "tab:green", "G"), (2, "tab:blue", "B")]:
        _h_before, _edges = np.histogram(_img_rgb[:, :, _ch], bins=64, range=(0, 255))
        _axes[1, 0].plot(_edges[:-1], _h_before, color=_color, label=_label, lw=1.2)
        _h_after, _ = np.histogram(_out[:, :, _ch], bins=64, range=(0, 255))
        _axes[1, 1].plot(_edges[:-1], _h_after, color=_color, label=_label, lw=1.2)
    for _ax, _title in [(_axes[1, 0], "original histogram"), (_axes[1, 1], "transformed histogram")]:
        _ax.set_title(_title)
        _ax.set_xlim(0, 255)
        _ax.legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Can you find gain/bias values that make the image look roughly
      **inverted** (dark ↔ light)? (Hint: what does a negative gain do?)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Histogram equalization

    Given an image whose values are unevenly distributed — too many dark
    pixels, too many light pixels, not enough in the middle — how do we
    automatically pick a mapping that uses the *full* dynamic range well?

    The idea: choose a mapping $f(I)$ so that the *output* histogram is as
    flat as possible. The trick is the same one used to sample from any
    probability distribution — integrate the histogram $h(I)$ to get the
    **cumulative distribution function**,

    $$ c(I) = \frac{1}{N}\sum_{i=0}^{I} h(i), $$

    where $N$ is the total pixel count. Mapping each pixel through
    $f(I)=c(I)$ (rescaled to $[0,255]$) is **histogram equalization**.

    Fully equalizing often looks "muddy" — flat in the statistical sense,
    but visually washed out. A common fix is to only **partially**
    equalize, blending the equalized mapping with the identity (do
    nothing) mapping:

    $$ f(I) = \alpha\, c(I) + (1-\alpha)\, I. $$

    One more wrinkle for *color* images: equalizing R, G, and B
    independently can shift hue and saturation (each channel gets
    stretched differently), the same artifact notebook 7 and §3.1.2
    warn about for plain brightening. The standard fix — what the demo
    below actually does — is to equalize only the **luminance** $Y$ and
    then rescale each RGB pixel by the ratio of new to old luma,
    $\text{RGB}_{\text{new}} = \text{RGB}_{\text{old}}\cdot(Y_{\text{new}}/Y_{\text{old}})$,
    which adjusts brightness while leaving color untouched.

    ### Predict

    Picture an image that's mostly dark, with just a few bright pixels
    (imagine a night photo with a lamp in it). What shape do you expect
    the cumulative-distribution mapping curve $c(I)$ to have? And will the
    *output* histogram end up perfectly flat?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Click to check your prediction": mo.md(r"""
        Since $c(I)$ accumulates the histogram from $0$ up to $I$, and
        most of the pixels are dark, $c(I)$ shoots up steeply over the
        dark range (where almost all the "mass" is) and stays nearly flat
        over the bright range (where there's hardly any mass left to add).
        That steep-then-flat shape is exactly what *stretches* the crowded
        dark values out over a wider output range — which is the whole
        point.

        The output histogram will be **flatter, but not perfectly flat**.
        Pixel values are discrete (only 256 possible levels), and many
        different input pixels can share the exact same intensity — they
        all get mapped to the *same* output value by $f$, so an image with
        large flat regions still produces spikes in its equalized
        histogram. Equalization can spread mass around; it can't
        manufacture new distinct values that weren't there.
        """)
    })
    return


@app.cell
def _(mo):
    mo.md("""
    *(For this demo, the loaded image is artificially darkened — real photos straight off the camera are rarely this washed out — so equalization has an obvious problem to fix.)*
    """)
    return


@app.cell
def _(mo):
    image_dropdown_histeq = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    alpha_slider = mo.ui.slider(start=0.0, stop=1.0, value=1.0, step=0.05, label="equalization strength α", debounce=True)
    mo.vstack([image_dropdown_histeq, alpha_slider])
    return alpha_slider, image_dropdown_histeq


@app.cell
def _(IMAGES_DIR, alpha_slider, image_dropdown_histeq, mo, np, plt):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_histeq.value, plt, np)
    _img_rgb = _img_rgb * 0.35  # artificially underexposed, so equalization has something obvious to fix
    _img_gray = to_luma(_img_rgb)

    _hist, _ = np.histogram(_img_gray, bins=256, range=(0, 256))
    _cdf = np.cumsum(_hist).astype(np.float64)
    _cdf = _cdf / _cdf[-1] * 255.0
    _identity = np.arange(256, dtype=np.float64)
    _mapping = alpha_slider.value * _cdf + (1 - alpha_slider.value) * _identity

    _idx = np.clip(_img_gray, 0, 255).astype(np.int32)
    _Y_new = _mapping[_idx]
    _ratio = _Y_new / np.maximum(_img_gray, 1e-3)
    _out = np.clip(_img_rgb * _ratio[:, :, None], 0, 255)

    _fig, _axes = plt.subplots(2, 3, figsize=(12, 7))
    _colors = ["tab:red", "tab:green", "tab:blue", "black"]
    _labels = ["R", "G", "B", "Y"]
    _before = [_img_rgb[:, :, 0], _img_rgb[:, :, 1], _img_rgb[:, :, 2], _img_gray]
    _after = [_out[:, :, 0], _out[:, :, 1], _out[:, :, 2], _Y_new]
    for _chan_before, _chan_after, _color, _label in zip(_before, _after, _colors, _labels):
        _h_before, _ = np.histogram(_chan_before, bins=256, range=(0, 256))
        _axes[0, 0].plot(np.arange(256), _h_before, color=_color, label=_label, lw=1.0)
        _h_after, _ = np.histogram(_chan_after, bins=256, range=(0, 256))
        _axes[0, 2].plot(np.arange(256), _h_after, color=_color, label=_label, lw=1.0)

    _axes[0, 1].plot(np.arange(256), _mapping, color="black", lw=1.5, label="Y transfer function")
    _axes[0, 1].plot([0, 255], [0, 255], color="gray", ls=":", lw=1, label="identity")

    _axes[0, 0].set_title("original histograms (R,G,B,Y)")
    _axes[0, 1].set_title(f"Y transfer function (α={alpha_slider.value:.2f})")
    _axes[0, 2].set_title("result histograms (R,G,B,Y)")
    for _ax in _axes[0]:
        _ax.set_xlim(0, 255)
        _ax.legend(fontsize=7)

    _axes[1, 0].imshow(_img_rgb.astype(np.uint8))
    _axes[1, 0].set_title("original")
    _axes[1, 0].axis("off")
    _axes[1, 1].axis("off")
    _axes[1, 2].imshow(_out.astype(np.uint8))
    _axes[1, 2].set_title(f"equalized (α={alpha_slider.value:.2f}, luma-preserving)")
    _axes[1, 2].axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Investigate

    - Compare $\alpha=1$ (full equalization) against a partial value like
      $\alpha=0.5$ — which looks more natural? Try this on more than one
      of the four images.
    - Does an already well-exposed image (try the astronaut) benefit from
      equalization as much as a lower-contrast one? What does the transfer
      function look like when the image barely needs correction?
    - Look at the transfer function plot: is it always monotonically
      non-decreasing? Why must that always be true, for *any* image,
      regardless of α?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Locally adaptive histogram equalization

    Global equalization computes *one* transfer function for the whole
    image — which can't do right by an image that has, say, a deeply
    shadowed foreground *and* a bright sky in the same frame: whatever
    curve helps one region hurts the other.

    The fix is to compute **separate** histograms in local tiles across
    the image, so each region gets its own transfer function. Naively
    equalizing each non-overlapping tile independently creates visible
    **blocking artifacts** (sharp jumps at tile boundaries). The standard
    solution — the one below — instead **bilinearly blends** the four
    nearest tiles' transfer functions for every pixel, based on how close
    that pixel is to each tile's center:

    $$
    f_{s,t}(I) = (1-s)(1-t) f_{00}(I) + s(1-t) f_{10}(I) + (1-s)t\, f_{01}(I) + st\, f_{11}(I)
    $$

    where $(s,t)$ is the pixel's fractional position between its four
    surrounding tile centers. This is **adaptive histogram equalization
    (AHE)**; its contrast-limited variant (which clips each tile's
    histogram before accumulating, to avoid over-amplifying noise in flat
    regions) is **CLAHE** — we don't implement the clipping step here, just
    the tiling + blending. As in §3, we equalize luminance only and
    rescale RGB by the ratio — with *tile-sized* neighborhoods now
    getting independently stretched, equalizing R, G, and B separately
    produces much more visible (and much uglier) color fringing than the
    global version did.
    """)
    return


@app.cell
def _(np):
    def build_ahe_channel(chan255, tile_size):
        """Locally-adaptive histogram equalization for one channel, via
        bilinear blending between per-tile CDFs (see markdown above)."""
        H, W = chan255.shape
        n_ty = max(1, int(np.ceil(H / tile_size)))
        n_tx = max(1, int(np.ceil(W / tile_size)))
        img_idx = np.clip(chan255, 0, 255).astype(np.int32)

        cdfs = np.zeros((n_ty, n_tx, 256))
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
                x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
                block = img_idx[y0:y1, x0:x1]
                hist = np.bincount(block.ravel(), minlength=256).astype(np.float64)
                cdf = np.cumsum(hist)
                cdfs[ty, tx] = cdf / max(cdf[-1], 1.0) * 255.0

        centers_y = (np.arange(n_ty) + 0.5) * tile_size
        centers_x = (np.arange(n_tx) + 0.5) * tile_size
        ty_f = np.interp(np.arange(H, dtype=np.float64), centers_y, np.arange(n_ty))
        tx_f = np.interp(np.arange(W, dtype=np.float64), centers_x, np.arange(n_tx))

        ty0 = np.clip(np.floor(ty_f).astype(int), 0, n_ty - 1)
        ty1 = np.clip(ty0 + 1, 0, n_ty - 1)
        tx0 = np.clip(np.floor(tx_f).astype(int), 0, n_tx - 1)
        tx1 = np.clip(tx0 + 1, 0, n_tx - 1)
        s = np.clip(ty_f - ty0, 0, 1)[:, None]
        t = np.clip(tx_f - tx0, 0, 1)[None, :]
        R0, R1 = ty0[:, None], ty1[:, None]
        C0, C1 = tx0[None, :], tx1[None, :]

        f00 = cdfs[R0, C0, img_idx]
        f10 = cdfs[R1, C0, img_idx]
        f01 = cdfs[R0, C1, img_idx]
        f11 = cdfs[R1, C1, img_idx]
        return (1 - s) * (1 - t) * f00 + s * (1 - t) * f10 + (1 - s) * t * f01 + s * t * f11

    return (build_ahe_channel,)


@app.cell
def _(mo):
    mo.md("""
    *(As in §3, the loaded image is artificially darkened here so both equalization methods have an obvious problem to fix.)*
    """)
    return


@app.cell
def _(mo):
    image_dropdown_ahe = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    tile_size_slider = mo.ui.slider(start=16, stop=256, value=64, step=8, label="tile size (pixels)", debounce=True)
    mo.vstack([image_dropdown_ahe, tile_size_slider])
    return image_dropdown_ahe, tile_size_slider


@app.cell
def _(
    IMAGES_DIR,
    build_ahe_channel,
    image_dropdown_ahe,
    mo,
    np,
    plt,
    tile_size_slider,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_ahe.value, plt, np)
    _img_rgb = _img_rgb * 0.35  # artificially underexposed, as in §3, so equalization has something obvious to fix
    _img_gray = to_luma(_img_rgb)

    _hist, _ = np.histogram(_img_gray, bins=256, range=(0, 256))
    _cdf = np.cumsum(_hist).astype(np.float64)
    _cdf = _cdf / _cdf[-1] * 255.0
    _Y_global = _cdf[np.clip(_img_gray, 0, 255).astype(np.int32)]
    _Y_local = build_ahe_channel(_img_gray, tile_size_slider.value)

    _ratio_global = _Y_global / np.maximum(_img_gray, 1e-3)
    _ratio_local = _Y_local / np.maximum(_img_gray, 1e-3)
    _global_out = np.clip(_img_rgb * _ratio_global[:, :, None], 0, 255)
    _local_out = np.clip(_img_rgb * _ratio_local[:, :, None], 0, 255)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    _axes[0].set_title("original")
    _axes[1].imshow(np.clip(_global_out, 0, 255).astype(np.uint8))
    _axes[1].set_title("global equalization")
    _axes[2].imshow(np.clip(_local_out, 0, 255).astype(np.uint8))
    _axes[2].set_title(f"locally-adaptive (tile={tile_size_slider.value}px)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Compare the middle and right panels: the locally-adaptive version should reveal detail in regions the single global curve either over- or under-corrects."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Linear filtering

    Point operators only ever look at one pixel at a time. **Neighborhood
    operators** look at a small patch of pixels around each location — the
    most common kind being a **linear filter**, where the output is a
    fixed weighted sum of nearby input pixels:

    $$ g(i,j) = \sum_{k,l} f(i+k, j+l)\, h(k,l) $$

    This is called **correlation**, written $g = f\otimes h$. A closely
    related operator flips the kernel's offsets,

    $$ g(i,j) = \sum_{k,l} f(i-k, j-l)\, h(k,l), $$

    called **convolution**, $g = f * h$. For a kernel that's symmetric
    under 180° rotation (true of every box, Gaussian, and Laplacian kernel
    below), correlation and convolution give the *same* result — so the
    two names get used almost interchangeably. They differ for an
    **asymmetric** kernel, like the derivative filter in §7: convolving
    with it instead of correlating with it flips the sign of the result.
    Every filter in this notebook is applied via **correlation** —
    literally sliding the kernel over the image as written, with no flip.

    A small worked example, correlating a $6\times 6$ image with a
    $3\times3$ box-averaging kernel (only the interior, where the kernel
    fits without running off the edge):
    """)
    return


@app.cell
def _(correlate2d, np):
    f_example = np.array([
        [10, 10, 20, 30, 30, 30],
        [10, 15, 20, 30, 35, 30],
        [15, 15, 25, 35, 35, 40],
        [20, 20, 30, 40, 45, 45],
        [25, 25, 35, 45, 50, 50],
        [25, 30, 35, 45, 55, 55],
    ], dtype=float)
    h_example = np.ones((3, 3)) / 9.0
    g_example = correlate2d(f_example, h_example, mode="valid")
    return f_example, g_example, h_example


@app.cell
def _(f_example, g_example, h_example, mo, plt):
    _r0, _c0 = 1, 1  # top-left corner of the highlighted 3x3 neighborhood in f

    def _draw_grid(ax, a, fmt="{:.0f}"):
        _n_rows, _n_cols = a.shape
        for _i in range(_n_rows):
            for _j in range(_n_cols):
                ax.add_patch(plt.Rectangle((_j, _i), 1, 1, facecolor="white", edgecolor="black", lw=0.6))
                ax.text(_j + 0.5, _i + 0.5, fmt.format(a[_i, _j]), ha="center", va="center", fontsize=9)
        ax.set_xlim(0, _n_cols)
        ax.set_ylim(0, _n_rows)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [6, 3, 4]})

    _draw_grid(_axes[0], f_example)
    _axes[0].add_patch(plt.Rectangle((_c0, _r0), 3, 3, facecolor="tab:blue", alpha=0.25, edgecolor="tab:blue", lw=2.5))
    _axes[0].set_title("f — source neighborhood\n(highlighted)", fontsize=10)

    _draw_grid(_axes[1], h_example, fmt="1/9")
    for _patch in _axes[1].patches:
        _patch.set_edgecolor("tab:orange")
        _patch.set_linewidth(1.2)
    _axes[1].set_title("h — kernel\n(box average)", fontsize=10)

    _draw_grid(_axes[2], g_example, fmt="{:.1f}")
    _axes[2].add_patch(plt.Rectangle((_c0, _r0), 1, 1, facecolor="tab:green", alpha=0.35, edgecolor="tab:green", lw=2.5))
    _axes[2].set_title("g = f ⊗ h — destination pixel\n(highlighted)", fontsize=10)

    _fig.suptitle("Correlation: the highlighted 3×3 neighborhood in f, weighted by h, produces the single highlighted pixel in g", fontsize=10, y=1.04)
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Box and Gaussian filters

    The two most basic linear filters both **blur** (low-pass) an image —
    they differ only in *how* they weight nearby pixels:

    - **Box filter**: every pixel in a $K\times K$ window gets equal
      weight $1/K^2$ — a simple moving average.
    - **Gaussian filter**: pixels are weighted by a bell curve,
      $G(x,y;\sigma)=\frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}$ —
      nearby pixels count more than far-away ones, which avoids the sharp
      "ringing" the box filter's hard cutoff can introduce.

    The demo below lets you explore the **Gaussian** filter interactively
    (applied per-channel, so color is preserved), plotting its 1D
    cross-section alongside so the bell-curve shape — and how it widens as
    σ grows — is visible directly, not just implied.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_blur = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    gaussian_sigma_slider = mo.ui.slider(start=0.5, stop=15.0, value=3.0, step=0.5, label="Gaussian σ", debounce=True)
    mo.vstack([image_dropdown_blur, gaussian_sigma_slider])
    return gaussian_sigma_slider, image_dropdown_blur


@app.cell
def _(
    IMAGES_DIR,
    gaussian_filter,
    gaussian_sigma_slider,
    image_dropdown_blur,
    mo,
    np,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_blur.value, plt, np)
    _out = np.zeros_like(_img_rgb)
    _sigma = gaussian_sigma_slider.value
    for _ch in range(3):
        _out[:, :, _ch] = gaussian_filter(_img_rgb[:, :, _ch], sigma=_sigma, mode="reflect")
    _xs = np.linspace(-4 * _sigma, 4 * _sigma, 200)
    _profile = np.exp(-_xs ** 2 / (2 * _sigma ** 2))
    _kernel_label = f"Gaussian, σ={_sigma:.1f}"

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    _axes[0].set_title("original")
    _axes[1].imshow(np.clip(_out, 0, 255).astype(np.uint8))
    _axes[1].set_title(f"filtered ({_kernel_label})")
    _axes[0].axis("off")
    _axes[1].axis("off")

    _axes[2].plot(_xs, _profile, color="tab:purple")
    _axes[2].set_title("1D kernel cross-section")
    _axes[2].set_xlabel("offset from center (px)")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Directional derivatives

    Instead of blurring, a filter can **differentiate**. The simplest
    horizontal/vertical derivative filters are $G_x$ and $G_y$ (the
    3×3 **Sobel** kernels below combine a central difference in one
    direction with a smoothing tent in the other). More generally, the
    derivative in *any* direction $\hat u=(\cos\theta,\sin\theta)$ is a
    linear combination of the two:

    $$ G_{\hat u} = \cos\theta\, G_x + \sin\theta\, G_y $$

    — no new convolution needed per angle, just a weighted sum of two
    fixed filter responses (this is the core idea behind *steerable*
    filters, though we stop here rather than building the full theory).

    The demo below shows just $G_x$ — the **x-direction** derivative —
    pre-smoothed with a Gaussian first (σ below), matching the "smoothed
    directional derivative" $\hat u\cdot\nabla(G*f)$: plain derivatives of
    a noisy image are themselves very noisy. You'll steer this filter to
    other directions yourself in the Modify activity right after.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_deriv = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    sigma_deriv_slider = mo.ui.slider(start=0.0, stop=5.0, value=1.0, step=0.25, label="pre-smoothing σ", debounce=True)
    mo.vstack([image_dropdown_deriv, sigma_deriv_slider])
    return image_dropdown_deriv, sigma_deriv_slider


@app.cell
def _(
    IMAGES_DIR,
    correlate2d,
    gaussian_filter,
    image_dropdown_deriv,
    mo,
    np,
    plt,
    sigma_deriv_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_deriv.value, plt, np))
    _smoothed = gaussian_filter(_img_gray, sigma=sigma_deriv_slider.value, mode="reflect") if sigma_deriv_slider.value > 0 else _img_gray

    _Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    _resp = correlate2d(_smoothed, _Gx, mode="same", boundary="symm")
    _vmax = max(np.abs(_resp).max(), 1e-6)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original (grayscale)")
    _axes[1].imshow(_resp, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1].set_title(f"Gx (x-direction derivative), σ={sigma_deriv_slider.value:.2f}")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Modify: steer the kernel yourself

    Below is a plain, literal $G_x$ kernel — the same one used above — as
    an editable NumPy array, convolved with the image and displayed
    immediately underneath. Since this is a live reactive cell, editing
    the array and re-running it updates the result right away.

    **Try it:**

    1. Edit `Gx_modify` below so it detects **vertical** edges ($G_y$)
       instead of horizontal ones.
    2. Edit it again so it detects **diagonal** edges instead.

    (Reminder: this notebook applies kernels via *correlation* — no
    flipping — so whatever you write is exactly what gets applied.)
    """)
    return


@app.cell
def _(mo):
    image_dropdown_modify = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    image_dropdown_modify
    return (image_dropdown_modify,)


@app.cell
def _(IMAGES_DIR, correlate2d, image_dropdown_modify, mo, np, plt):
    Gx_modify = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)

    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_modify.value, plt, np))
    _resp = correlate2d(_img_gray, Gx_modify, mode="same", boundary="symm")
    _vmax = max(np.abs(_resp).max(), 1e-6)

    _fig, _ax = plt.subplots(figsize=(5, 4.5))
    _ax.imshow(_resp, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _ax.set_title("your kernel's response")
    _ax.axis("off")

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. The Laplacian operator

    The (undirected) second derivative of an image,

    $$ \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}, $$

    is the **Laplacian**. Unlike $G_x$/$G_y$, it has no preferred
    direction — it responds to edges and blobs equally regardless of
    orientation. The standard discrete approximation is the 5-point
    kernel

    $$
    \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}
    $$

    (notice its weights sum to zero — a constant region always produces a
    response of exactly zero). Blurring with a Gaussian first and then
    taking the Laplacian — equivalent to convolving directly with the
    **Laplacian of Gaussian (LoG)** kernel — suppresses noise before
    differentiating, at the cost of localization; try raising σ below to
    see the trade-off.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_laplacian = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    sigma_lap_slider = mo.ui.slider(start=0.0, stop=6.0, value=0.5, step=0.25, label="pre-smoothing σ", debounce=True)
    mo.vstack([image_dropdown_laplacian, sigma_lap_slider])
    return image_dropdown_laplacian, sigma_lap_slider


@app.cell
def _(
    IMAGES_DIR,
    correlate2d,
    gaussian_filter,
    image_dropdown_laplacian,
    mo,
    np,
    plt,
    sigma_lap_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_laplacian.value, plt, np))
    _smoothed = gaussian_filter(_img_gray, sigma=sigma_lap_slider.value, mode="reflect") if sigma_lap_slider.value > 0 else _img_gray

    _lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    _resp = correlate2d(_smoothed, _lap_kernel, mode="same", boundary="symm")
    _vmax = max(np.abs(_resp).max(), 1e-6)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original (grayscale)")
    _axes[1].imshow(_resp, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _axes[1].set_title(f"Laplacian response (σ={sigma_lap_slider.value:.2f})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Point operators** transform each pixel independently:
      gain/bias ($g=af+b$), gamma ($g=f^{1/\gamma}$), and — using
      whole-image statistics — **histogram equalization**, $f(I)=c(I)$
      via the cumulative distribution, with a partial-blend parameter α
      to avoid the "muddy" look of full equalization.
    - **Locally-adaptive histogram equalization** computes separate
      transfer functions per tile and bilinearly blends them, fixing
      global equalization's blindness to local contrast variation.
    - **Linear filters** are neighborhood operators: **correlation**
      slides a kernel as written; **convolution** flips it first — the
      same for symmetric kernels, different (a sign flip) for asymmetric
      ones like a derivative filter.
    - **Box** and **Gaussian** filters blur (low-pass); **directional
      derivatives** ($G_x$, $G_y$, and any $G_\theta$ via their linear
      combination) and the **Laplacian** ($\nabla^2f$) differentiate,
      picking out edges and fine structure instead of smoothing it away.

    **Next up:** non-linear neighborhood operators (median and bilateral
    filtering — better-behaved than a Gaussian blur when noise is
    non-Gaussian) and frequency-domain operators (the Fourier transform
    view of everything we just did with kernels).
    """)
    return


if __name__ == "__main__":
    app.run()
