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
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return IMAGES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Multi-resolution Representations

    Notebooks 8–10 always produced an output the same size as the input.
    This notebook covers Szeliski **§3.5, "Pyramids and wavelets,"**
    selectively:

    - **§3.5.2** Decimation — downsampling an image *correctly*.
    - **§3.5.1** Interpolation — upsampling an image, needed below to
      build the Laplacian/DoG pyramid the way Burt and Adelson actually
      did it.
    - **§3.5.3** Multi-resolution representations — the traditional
      (Gaussian) image pyramid, and the difference-of-Gaussians (DoG)
      pyramid.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-10)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.function
def apply_kernel_rgb(img_rgb, kernel, correlate2d, np):
    """Apply a 2D kernel to each of the 3 channels via correlation, mode='same',
    mirror boundary — the same convention used throughout notebooks 8-10."""
    out = np.zeros_like(img_rgb)
    for ch in range(3):
        out[:, :, ch] = correlate2d(img_rgb[:, :, ch], kernel, mode="same", boundary="symm")
    return out


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Decimation: downsampling correctly

    To reduce an image's resolution by a factor $r$, it's tempting to just
    keep every $r$-th pixel. That's *wrong*: it skips the low-pass step
    that has to happen first,

    $$ g(i,j) = \sum_{k,l} f(k,l)\, h(ri-k, rj-l) \qquad \text{(Eq. 3.66)}, $$

    i.e., **convolve with a low-pass filter, then keep every $r$-th
    sample** (Figure 3.29, below). Skipping the filter doesn't just lose
    detail — high frequencies that can't be represented at the new, lower
    sampling rate don't disappear, they **alias**, folding back down and
    masquerading as spurious low-frequency content (moiré patterns, jagged
    edges) that was never in the original image. This is the same
    ringing/aliasing lesson from notebook 10 §5, viewed from the opposite
    direction: there, a sharp *frequency-domain* cutoff caused ringing in
    space; here, a sharp *spatial-domain* cutoff (i.e., none at all) causes
    aliasing in frequency.

    **What filter is needed?** After downsampling by $r$, the new Nyquist
    frequency is $1/(2r)$ (in cycles per *original* pixel) — anything above
    that must be removed first. A Gaussian with $\sigma \approx r/2$ is a
    simple, effective choice; §2 below uses the specific **binomial**
    filter (Eq. 3.69) that Burt and Adelson standardized for $r=2$.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_decim = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    downsample_factor_slider = mo.ui.slider(start=2, stop=16, value=4, step=2, label="downsampling factor r", debounce=True)
    mo.vstack([image_dropdown_decim, downsample_factor_slider])
    return downsample_factor_slider, image_dropdown_decim


@app.cell
def _(
    IMAGES_DIR,
    downsample_factor_slider,
    gaussian_filter,
    image_dropdown_decim,
    mo,
    np,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_decim.value, plt, np)
    _r = downsample_factor_slider.value

    _naive = _img_rgb[::_r, ::_r]

    _sigma = _r / 2.0
    _blurred = np.zeros_like(_img_rgb)
    for _ch in range(3):
        _blurred[:, :, _ch] = gaussian_filter(_img_rgb[:, :, _ch], sigma=_sigma, mode="reflect")
    _filtered = _blurred[::_r, ::_r]

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    _axes[0].set_title(f"original ({_img_rgb.shape[1]}×{_img_rgb.shape[0]})")
    _axes[1].imshow(np.clip(_naive, 0, 255).astype(np.uint8), interpolation="nearest")
    _axes[1].set_title(f"naive: every {_r}th pixel\n({_naive.shape[1]}×{_naive.shape[0]}, no filter)")
    _axes[2].imshow(np.clip(_filtered, 0, 255).astype(np.uint8), interpolation="nearest")
    _axes[2].set_title(f"filtered first (σ={_sigma:.1f}), then\nevery {_r}th pixel ({_filtered.shape[1]}×{_filtered.shape[0]})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Look for moiré patterns and jagged, sparkling edges in the middle panel, especially in fine texture (fur, weave, granular surfaces) — that's aliasing. The right panel should look softer but clean, with no spurious patterning."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The traditional (Gaussian) image pyramid

    Stacking repeatedly-decimated versions of an image builds a **pyramid**
    (Figure 3.31): each level has half the width and height — a quarter of
    the pixels — of the level below it. Because adjacent levels differ by a
    sampling rate $r=2$, this is called an **octave pyramid**.

    Burt and Adelson's (1983) construction blurs with a small, fixed 5-tap
    kernel before each decimation step,

    $$ [\,c\;\; b\;\; a\;\; b\;\; c\,], \qquad b=\tfrac14,\;\; c=\tfrac14-\tfrac{a}{2} \qquad \text{(Eq. 3.68)}, $$

    with $a=3/8$ in practice, giving the **binomial** kernel

    $$ \frac{1}{16}[\,1\;\; 4\;\; 6\;\; 4\;\; 1\,] \qquad \text{(Eq. 3.69)}. $$

    It's called a *Gaussian* pyramid because repeated convolutions of this
    kernel with itself converge toward a Gaussian shape (the same
    central-limit-theorem behavior explored via 1D self-convolution in the
    code-walkthrough notebook) — even though only a *single* application of
    the binomial kernel is used between each pair of levels.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_31_traditional_image_pyramid.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.31 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 156, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(np):
    def binomial_kernel_2d():
        b1d = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        return np.outer(b1d, b1d)

    def build_gaussian_pyramid(img_rgb, n_levels, kernel, correlate2d):
        levels = [img_rgb]
        current = img_rgb
        for _ in range(n_levels):
            blurred = apply_kernel_rgb(current, kernel, correlate2d, np)
            current = blurred[::2, ::2]
            levels.append(current)
        return levels

    return binomial_kernel_2d, build_gaussian_pyramid


@app.cell
def _(mo):
    image_dropdown_pyramid = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    n_levels_slider = mo.ui.slider(start=2, stop=6, value=4, step=1, label="pyramid levels", debounce=True)
    mo.vstack([image_dropdown_pyramid, n_levels_slider])
    return image_dropdown_pyramid, n_levels_slider


@app.cell
def _(
    IMAGES_DIR,
    binomial_kernel_2d,
    build_gaussian_pyramid,
    correlate2d,
    image_dropdown_pyramid,
    mo,
    n_levels_slider,
    np,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_pyramid.value, plt, np)
    _kernel = binomial_kernel_2d()
    _pyramid = build_gaussian_pyramid(_img_rgb, n_levels_slider.value, _kernel, correlate2d)

    _widths = [lvl.shape[1] for lvl in _pyramid]
    _fig, _axes = plt.subplots(1, len(_pyramid), figsize=(2.4 * len(_pyramid), 3.2), gridspec_kw={"width_ratios": _widths})
    for _l, (_ax, _lvl) in enumerate(zip(_axes, _pyramid)):
        _ax.imshow(np.clip(_lvl, 0, 255).astype(np.uint8))
        _tag = "fine" if _l == 0 else ("coarse" if _l == len(_pyramid) - 1 else "medium")
        _ax.set_title(f"l={_l} ({_tag})\n{_lvl.shape[1]}×{_lvl.shape[0]}", fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Interpolation: upsampling

    Going the other direction — increasing resolution — needs a matching
    trick: stretch the image onto a larger, mostly-empty grid, then
    convolve with an interpolation kernel to fill in the gaps,

    $$ g(i,j) = \sum_{k,l} f(k,l)\, h(i-rk, j-rl) \qquad \text{(Eq. 3.64)}. $$

    Concretely, for $r=2$: insert a zero between every pixel in both
    directions (doubling the width and height), then convolve. Since only
    1 in every 4 samples on that stretched grid is non-zero, a plain
    convolution would come out 4× too dim — so the **reconstruction**
    kernel is the same binomial kernel as decimation, just scaled up (by 2
    per dimension) to compensate. This "reconstruction filter coefficients
    are twice the analysis coefficients" rule is exactly what Figure 3.32
    shows below.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_32_gaussian_pyramid_signal_diagram.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.32 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 156, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(np):
    def upsample_2x(img_rgb, kernel, correlate2d):
        """Zero-stuff to double each dimension, then convolve with the
        reconstruction kernel (4x the analysis kernel, Eq. 3.64/Figure 3.32)."""
        H, W = img_rgb.shape[:2]
        stuffed = np.zeros((H * 2, W * 2, 3))
        stuffed[::2, ::2] = img_rgb
        return apply_kernel_rgb(stuffed, kernel * 4.0, correlate2d, np)

    return (upsample_2x,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The difference-of-Gaussians (DoG) pyramid

    Subtracting two different amounts of blur produces a **band-pass**
    filter — it keeps a range of "medium" frequencies and discards both the
    very low and very high ones (Figure 3.34):

    $$ \text{DoG}\{I;\sigma_1,\sigma_2\} = G_{\sigma_1} * I - G_{\sigma_2} * I = (G_{\sigma_1} - G_{\sigma_2}) * I \qquad \text{(Eq. 3.70)}. $$

    This is what Burt and Adelson's **Laplacian pyramid** actually stores
    at each level (a slight misnomer — the continuous Laplacian-of-Gaussian,
    $\text{LoG}\{I;\sigma\}=\nabla^2(G_\sigma * I)$ from notebook 8 §8, is
    closely approximated by a DoG with two nearby σ's). Their construction
    (Figure 3.33, not reproduced here) builds each band by taking the
    *next coarser* Gaussian-pyramid level, **upsampling** it back up with
    §3's interpolation, and subtracting:

    $$ L_l = G_l - \text{upsample}(G_{l+1}). $$

    $\text{upsample}(G_{l+1})$ is a blurrier version of the image than
    $G_l$ (it's been through one extra round of decimation blur), so $L_l$
    is still a genuine difference-of-Gaussians band — just computed
    *across* adjacent pyramid levels instead of within a single one. This
    construction is also **perfectly invertible**: adding every band back
    onto the coarsest Gaussian level, from the top down, reconstructs the
    original image exactly — which is why the book calls the Laplacian
    images "sufficient to exactly reconstruct the original image."
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig3_34_dog_bandpass.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 3.34 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 158, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    image_dropdown_dog = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    n_levels_dog_slider = mo.ui.slider(start=2, stop=6, value=4, step=1, label="pyramid levels", debounce=True)
    mo.vstack([image_dropdown_dog, n_levels_dog_slider])
    return image_dropdown_dog, n_levels_dog_slider


@app.cell
def _(
    IMAGES_DIR,
    binomial_kernel_2d,
    build_gaussian_pyramid,
    correlate2d,
    image_dropdown_dog,
    mo,
    n_levels_dog_slider,
    np,
    plt,
    upsample_2x,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_dog.value, plt, np)
    _kernel = binomial_kernel_2d()
    _pyramid = build_gaussian_pyramid(_img_rgb, n_levels_dog_slider.value, _kernel, correlate2d)

    _bands = []
    for _l in range(len(_pyramid) - 1):
        _up = upsample_2x(_pyramid[_l + 1], _kernel, correlate2d)
        _Hc, _Wc = _pyramid[_l].shape[:2]
        _bands.append(_pyramid[_l] - _up[:_Hc, :_Wc])
    _bands.append(_pyramid[-1])  # coarsest level, kept as-is for reconstruction

    _widths = [b.shape[1] for b in _bands]
    _fig, _axes = plt.subplots(1, len(_bands), figsize=(2.4 * len(_bands), 3.2), gridspec_kw={"width_ratios": _widths})
    for _l, (_ax, _band) in enumerate(zip(_axes, _bands)):
        if _l == len(_bands) - 1:
            _ax.imshow(np.clip(_band, 0, 255).astype(np.uint8))
            _tag = "base"
        else:
            _gray_band = to_luma(_band)
            _vmax = max(np.abs(_gray_band).max(), 1e-6)
            _ax.imshow(_gray_band, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            _tag = "fine" if _l == 0 else "medium"
        _ax.set_title(f"l={_l} ({_tag})\n{_band.shape[1]}×{_band.shape[0]}", fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()

    _recon = _bands[-1]
    for _l in range(len(_bands) - 2, -1, -1):
        _up = upsample_2x(_recon, _kernel, correlate2d)
        _Hc, _Wc = _bands[_l].shape[:2]
        _recon = _up[:_Hc, :_Wc] + _bands[_l]
    _max_diff = np.abs(_recon - _img_rgb).max()

    mo.vstack([
        mo.md(f"Each band $L_l = G_l - \\text{{upsample}}(G_{{l+1}})$ (luma shown, diverging colormap); the last panel is the coarsest Gaussian level itself. Finer bands pick out small, sharp detail; coarser bands pick out broader structure. Summing every band back onto the coarsest level reconstructs the original to within **{_max_diff:.1e}** intensity levels — essentially exact, up to floating-point roundoff."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Decimation** (downsampling) requires a low-pass filter *before*
      subsampling (Eq. 3.66) — skip it and high frequencies alias into
      spurious low-frequency patterns instead of disappearing.
    - The **binomial filter** $\frac{1}{16}[1,4,6,4,1]$ (Eq. 3.69) is the
      standard cheap, effective choice for $r=2$ decimation, converging to
      a Gaussian shape under repeated self-convolution.
    - **Interpolation** (upsampling) is decimation run in reverse: zero-stuff,
      then convolve with the same kernel scaled up to compensate (Eq. 3.64).
    - The **traditional (Gaussian) image pyramid** stacks repeatedly
      decimated copies of an image — half the resolution, a quarter of the
      pixels, per level (Figure 3.31).
    - The **difference-of-Gaussians (DoG) pyramid** (Eq. 3.70) extracts
      band-pass detail at each pyramid level by upsampling and subtracting
      adjacent Gaussian-pyramid levels — Burt and Adelson's actual
      Laplacian pyramid construction, and perfectly invertible.
    """)
    return


if __name__ == "__main__":
    app.run()
