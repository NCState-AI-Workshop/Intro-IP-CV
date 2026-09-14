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
    from scipy.ndimage import (
        median_filter,
        gaussian_filter,
        zoom,
        binary_dilation,
        binary_erosion,
        binary_opening,
        binary_closing,
    )

    return (
        binary_closing,
        binary_dilation,
        binary_erosion,
        binary_opening,
        gaussian_filter,
        median_filter,
        mo,
        np,
        plt,
        zoom,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Non-linear Filtering and Binary Morphology

    Notebook 8 covered **linear** neighborhood operators — correlation with a
    fixed kernel, always a weighted *sum* of nearby pixels. That's powerful,
    but it has a blind spot: a linear filter treats every pixel in its
    neighborhood the same way regardless of what value it actually holds, so
    a single wild outlier (a dead sensor pixel, a spike of "shot" noise)
    drags the whole weighted average with it.

    This notebook covers Szeliski **§3.3, "More neighborhood operators"**,
    selectively:

    - **§3.3.1** Non-linear filtering — the **median filter**.
    - **§3.3.2** **Bilateral filtering** — edge-preserving smoothing.
    - **§3.3.3** Binary image processing — thresholding and **morphology**
      (dilation, erosion, opening, closing), pushed a bit further than the
      book's brief treatment to make the mechanics concrete.

    No Predict/Investigate/Modify activities this time — just interactive
    demos, meant to be explored together in class.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same simple convention used in notebook 8)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Beyond linear filters

    Recall a linear filter's output is always a fixed weighted sum,
    $g(i,j)=\sum_{k,l}f(i+k,j+l)h(k,l)$ — every neighboring pixel
    contributes, weighted only by its *position* in the kernel, never by its
    *value*. That's exactly what makes a Gaussian blur unable to remove
    **shot noise** (occasional pixels with wildly wrong values, e.g. a
    dead/stuck sensor pixel or a transmission error): averaging a bad pixel
    in with its neighbors doesn't erase it, it just smears it into a soft
    but still-visible blob.

    The operators below fix this by using the pixel *values* themselves,
    not just their positions: the **median filter** picks a single
    order statistic instead of averaging, and the **bilateral filter**
    down-weights neighbors whose value is too different from the center
    pixel's. Both are non-linear: doubling the input does not double the
    output.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Median filtering

    The **median filter** replaces every pixel with the median value found
    in its neighborhood,

    $$ g(i,j) = \operatorname{median}\{\, f(i+k,j+l) : (k,l) \in \mathcal{N} \,\}. $$

    The median is a much more *robust* statistic than the mean: a
    single extreme outlier barely moves a median, but it can move a mean
    arbitrarily far.

    Because it only ever outputs a value that was *actually present* in the
    neighborhood (never a blend), the median filter is very good at
    deleting shot/impulse noise (a "salt-and-pepper" pixel is an extreme
    outlier — median filtering with a big enough window ignores it
    completely) while still leaving sharp edges mostly intact, unlike a
    Gaussian blur strong enough to do the same job.
    """)
    return


@app.cell
def _(np):
    def add_salt_pepper_noise(img_rgb, density, seed=0):
        """Randomly replace a fraction `density` of pixels (all channels
        together, so noise looks like a true dead/saturated pixel) with
        either 0 or 255."""
        rng = np.random.default_rng(seed)
        out = img_rgb.copy()
        H, W, _ = img_rgb.shape
        mask = rng.random((H, W)) < density
        vals = rng.choice([0.0, 255.0], size=mask.sum())
        out[mask] = vals[:, None]
        return out

    return (add_salt_pepper_noise,)


@app.cell
def _(mo):
    image_dropdown_median = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    noise_density_slider = mo.ui.slider(start=0.0, stop=0.3, value=0.1, step=0.01, label="salt-and-pepper density", debounce=True)
    median_size_slider = mo.ui.slider(start=3, stop=11, value=5, step=2, label="window size K (K×K)", debounce=True)
    mo.vstack([image_dropdown_median, mo.hstack([noise_density_slider, median_size_slider], justify="start", gap=2)])
    return image_dropdown_median, median_size_slider, noise_density_slider


@app.cell
def _(
    IMAGES_DIR,
    add_salt_pepper_noise,
    gaussian_filter,
    image_dropdown_median,
    median_filter,
    median_size_slider,
    mo,
    noise_density_slider,
    np,
    plt,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_median.value, plt, np)
    _noisy = add_salt_pepper_noise(_img_rgb, noise_density_slider.value)
    _k = median_size_slider.value

    _median_out = np.zeros_like(_noisy)
    _gauss_out = np.zeros_like(_noisy)
    for _ch in range(3):
        _median_out[:, :, _ch] = median_filter(_noisy[:, :, _ch], size=_k)
        _gauss_out[:, :, _ch] = gaussian_filter(_noisy[:, :, _ch], sigma=_k / 4.0, mode="reflect")

    _fig, _axes = plt.subplots(1, 4, figsize=(15, 4.2))
    _axes[0].imshow(_img_rgb.astype(np.uint8))
    _axes[0].set_title("original")
    _axes[1].imshow(np.clip(_noisy, 0, 255).astype(np.uint8))
    _axes[1].set_title(f"salt & pepper (p={noise_density_slider.value:.2f})")
    _axes[2].imshow(np.clip(_median_out, 0, 255).astype(np.uint8))
    _axes[2].set_title(f"median filtered (K={_k})")
    _axes[3].imshow(np.clip(_gauss_out, 0, 255).astype(np.uint8))
    _axes[3].set_title("Gaussian filtered\n(for comparison, notebook 8)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Compare the last two panels: the median filter should remove most speckle while keeping edges much sharper than a Gaussian blur strong enough to do the same job."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bilateral filtering

    A median filter is robust, but it's an all-or-nothing rank statistic —
    it can't smoothly trade off "how much to trust a neighbor." The
    **bilateral filter** instead keeps a weighted-average structure like a
    Gaussian blur, but multiplies in a second weight that rejects (softly)
    any neighbor whose *value* is too different from the center pixel:

    $$ g(i,j) = \frac{\sum_{k,l} f(k,l)\, w(i,j,k,l)}{\sum_{k,l} w(i,j,k,l)} \qquad \text{(Eq. 3.34)} $$

    where $w$ is the product of a **domain kernel** (plain spatial Gaussian,
    same as notebook 8's blur, Eq. 3.35) and a **range kernel** (a Gaussian
    over *intensity difference*, Eq. 3.36):

    $$
    d(i,j,k,l) = \exp\!\left(-\frac{(i-k)^2+(j-l)^2}{2\sigma_d^2}\right), \qquad
    r(i,j,k,l) = \exp\!\left(-\frac{\lVert f(i,j)-f(k,l)\rVert^2}{2\sigma_r^2}\right)
    $$

    $$ w(i,j,k,l) = d(i,j,k,l)\cdot r(i,j,k,l) \qquad \text{(Eq. 3.37)} $$

    $\sigma_d$ controls the *spatial* extent, exactly like the Gaussian
    filter's σ in notebook 8. $\sigma_r$ controls how similar two intensities
    must be to count as "the same surface": as $\sigma_r\to\infty$ the range
    kernel stops discriminating at all ($r\to 1$ everywhere) and the
    bilateral filter reduces to a plain Gaussian blur — so notebook 8's
    filter is really the $\sigma_r=\infty$ special case of this one. A
    *small* $\sigma_r$, on the other hand, means only near-identical pixels
    get averaged together, which is exactly what preserves edges: across a
    strong edge, the far side's pixels are too different in value to
    contribute, so the filter never blurs across it.

    (As the book notes, this brute-force formula is much slower than a
    separable linear filter — there's no shortcut analogous to notebook 8's
    fast `scipy.ndimage` calls. The demo below therefore runs on a
    downsampled copy of the image so it stays interactive.)
    """)
    return


@app.cell
def _(np):
    def bilateral_channel(chan, sigma_d, sigma_r):
        """Brute-force bilateral filter for one channel (Eq. 3.34-3.37),
        vectorized over the neighborhood offsets rather than over pixels."""
        radius = max(1, int(np.ceil(3 * sigma_d)))
        H, W = chan.shape
        padded = np.pad(chan, radius, mode="reflect")
        acc = np.zeros_like(chan)
        wsum = np.zeros_like(chan)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                neighbor = padded[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
                domain_w = np.exp(-(dy**2 + dx**2) / (2 * sigma_d**2))
                range_w = np.exp(-((neighbor - chan) ** 2) / (2 * sigma_r**2))
                w = domain_w * range_w
                acc += w * neighbor
                wsum += w
        return acc / wsum

    return (bilateral_channel,)


@app.cell
def _(mo):
    image_dropdown_bilateral = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    sigma_d_slider = mo.ui.slider(start=0.5, stop=6.0, value=2.0, step=0.5, label="domain σ_d (spatial)", debounce=True)
    sigma_r_slider = mo.ui.slider(start=5, stop=100, value=30, step=5, label="range σ_r (intensity)", debounce=True)
    mo.vstack([image_dropdown_bilateral, mo.hstack([sigma_d_slider, sigma_r_slider], justify="start", gap=2)])
    return image_dropdown_bilateral, sigma_d_slider, sigma_r_slider


@app.cell
def _(
    IMAGES_DIR,
    bilateral_channel,
    gaussian_filter,
    image_dropdown_bilateral,
    mo,
    np,
    plt,
    sigma_d_slider,
    sigma_r_slider,
    zoom,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_bilateral.value, plt, np)
    _H, _W, _ = _img_rgb.shape
    _target = 260
    _scale = _target / max(_H, _W)
    _small = zoom(_img_rgb, (_scale, _scale, 1), order=1)
    _small = np.clip(_small, 0, 255)

    _bilateral_out = np.zeros_like(_small)
    _gauss_out = np.zeros_like(_small)
    for _ch in range(3):
        _bilateral_out[:, :, _ch] = bilateral_channel(_small[:, :, _ch], sigma_d_slider.value, sigma_r_slider.value)
        _gauss_out[:, :, _ch] = gaussian_filter(_small[:, :, _ch], sigma=sigma_d_slider.value, mode="reflect")

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4.5))
    _axes[0].imshow(_small.astype(np.uint8))
    _axes[0].set_title(f"original (downsampled to {_small.shape[1]}×{_small.shape[0]})")
    _axes[1].imshow(np.clip(_bilateral_out, 0, 255).astype(np.uint8))
    _axes[1].set_title(f"bilateral (σ_d={sigma_d_slider.value:.1f}, σ_r={sigma_r_slider.value:.0f})")
    _axes[2].imshow(np.clip(_gauss_out, 0, 255).astype(np.uint8))
    _axes[2].set_title(f"Gaussian only (σ={sigma_d_slider.value:.1f})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Compare the last two panels at the same spatial σ: the bilateral filter should keep edges noticeably crisper — try lowering σ_r to see edges preserved even more aggressively, or raising it toward 100 to see the result approach the plain Gaussian."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Binary image processing and morphology

    Many vision pipelines first reduce a grayscale image to a **binary**
    image via thresholding (Eq. 3.44),

    $$ \theta(f,t) = \begin{cases} 1 & \text{if } f \ge t \\ 0 & \text{otherwise} \end{cases} $$

    — e.g., separating dark foreground objects from a light background.
    **Morphological operators** then reshape the resulting blobs. Szeliski
    defines all of them through one unifying trick: convolve the binary
    image $f$ with a binary **structuring element** $s$ (any small
    shape — a 3×3 square, a cross, a disk, ...) to get an integer count,

    $$ c = f \otimes s \qquad \text{(Eq. 3.45)} $$

    — literally: slide $s$ over $f$ and, at each position, count how many
    of the structuring element's active cells land on a foreground (1)
    pixel — then threshold that count $c$ at different levels using
    $\theta$ from above. With $S$ = the number of active pixels in $s$:

    - **dilation:** $\operatorname{dilate}(f,s) = \theta(c, 1)$ — foreground
      if *any* overlap at all.
    - **erosion:** $\operatorname{erode}(f,s) = \theta(c, S)$ — foreground
      only if *every* active cell of $s$ overlaps foreground.
    - **opening:** $\operatorname{open}(f,s) = \operatorname{dilate}(\operatorname{erode}(f,s), s)$
      — erode, then dilate.
    - **closing:** $\operatorname{close}(f,s) = \operatorname{erode}(\operatorname{dilate}(f,s), s)$
      — dilate, then erode.

    That convolution recipe is exact and compact, but it's worth also
    seeing the more standard *set* description most other textbooks use,
    since it maps more directly onto "what actually happens to the shape,"
    which the book doesn't spell out in as much depth:

    - **Erosion "fits":** a pixel survives only if the structuring element,
      centered there, **fits entirely inside** the foreground. Any pixel
      close enough to a boundary or inside a hole/gap narrower than $s$
      gets deleted — so erosion **shrinks** foreground regions and
      completely removes anything **smaller than** $s$.
    - **Dilation "hits":** a pixel becomes foreground if the structuring
      element, centered there, **touches ("hits") any** foreground pixel at
      all. Dilation therefore **grows** foreground regions outward by
      roughly the radius of $s$, and can bridge together two blobs that
      were separated by a gap narrower than $s$.
    - **Opening = erode, then dilate.** The erosion step deletes anything
      thinner than $s$ (small dots, thin protrusions) *and shrinks
      everything else*; the dilation step grows the *survivors* back to
      roughly their original size. Net effect: small isolated bits and
      thin spikes disappear, but large regions end up close to their
      original size and shape, just with outward-facing corners rounded
      off.
    - **Closing = dilate, then erode** — the mirror image of opening. The
      dilation step bridges/fills anything narrower than $s$ (small holes,
      thin gaps between blobs); the erosion step shrinks everything back
      down. Net effect: small holes and gaps get filled in, large regions
      are largely unaffected, and inward-facing corners get rounded off.

    A useful mnemonic: **opening removes small foreground stuff, closing
    fills small background stuff** — they're each other's complement.
    Crucially, *neither* one is "erosion and dilation cancel out" — a
    feature has to be **big enough to survive the first step** to come back
    at all, which is exactly what makes them useful for cleaning up
    thresholding noise without destroying the real objects.
    """)
    return


@app.cell
def _(np):
    def build_structuring_element(shape, size):
        """A size×size boolean structuring element, centered."""
        radius = size // 2
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        if shape == "square":
            return np.ones((size, size), dtype=bool)
        elif shape == "cross":
            return (yy == 0) | (xx == 0)
        elif shape == "disk":
            return (yy**2 + xx**2) <= radius**2
        raise ValueError(shape)

    return (build_structuring_element,)


@app.cell
def _(mo):
    mo.md("""
    ### A small worked example

    Before applying these to a real photo, here's a synthetic 16×16 binary
    shape — a solid blob with a **1-pixel hole**, a **thin 1-pixel-wide
    protrusion**, and an **isolated single-pixel speck** — run through all
    four operators with a 3×3 square structuring element ($S=9$). Watch
    which small features survive which operation.
    """)
    return


@app.cell
def _(
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    build_structuring_element,
    np,
):
    A_example = np.zeros((16, 16), dtype=bool)
    A_example[2:9, 2:9] = True  # solid blob
    A_example[5, 5] = False  # 1-pixel hole
    A_example[2, 9:11] = True  # thin protrusion off the top edge
    A_example[13, 13] = True  # isolated noise speck

    s_example = build_structuring_element("square", 3)
    dilate_example = binary_dilation(A_example, structure=s_example)
    erode_example = binary_erosion(A_example, structure=s_example)
    open_example = binary_opening(A_example, structure=s_example)
    close_example = binary_closing(A_example, structure=s_example)
    return (
        A_example,
        close_example,
        dilate_example,
        erode_example,
        open_example,
        s_example,
    )


@app.cell
def _(
    A_example,
    close_example,
    dilate_example,
    erode_example,
    mo,
    open_example,
    plt,
    s_example,
):
    def _draw_binary(ax, a, title):
        ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    _fig, _axes = plt.subplots(2, 3, figsize=(9, 6.4))
    _draw_binary(_axes[0, 0], A_example, "binary input f\n(hole, protrusion, speck)")
    _draw_binary(_axes[0, 1], s_example, "structuring element s\n(3×3 square, S=9)")
    _axes[0, 2].axis("off")
    _draw_binary(_axes[1, 0], dilate_example, "dilation\n(grows, bridges gaps)")
    _draw_binary(_axes[1, 1], erode_example, "erosion\n(shrinks, deletes thin bits)")
    _draw_binary(_axes[1, 2], open_example, "opening\n(speck + protrusion gone,\nhole remains)")
    _fig.tight_layout()

    _fig2, _ax2 = plt.subplots(figsize=(3.2, 3.2))
    _draw_binary(_ax2, close_example, "closing\n(hole filled,\nspeck remains)")
    _fig2.tight_layout()

    mo.vstack([_fig, _fig2])
    return


@app.cell
def _(mo):
    mo.md("""
    Notice the asymmetry in the bottom row: **opening** deletes the
    protrusion and the isolated speck (both too thin to contain a 3×3
    square) but leaves the 1-pixel hole untouched (opening never *adds*
    pixels). **Closing** does the opposite — it fills the hole but leaves
    the speck exactly where it was (closing never *removes* foreground
    pixels, so an isolated dot just gets briefly dilated and then eroded
    right back to itself).
    """)
    return


@app.cell
def _(mo):
    image_dropdown_morph = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="coffee", label="image"
    )
    threshold_slider = mo.ui.slider(start=0, stop=255, value=110, step=5, label="threshold t", debounce=True)
    se_shape_dropdown = mo.ui.dropdown(options=["square", "cross", "disk"], value="disk", label="structuring element shape")
    se_size_slider = mo.ui.slider(start=3, stop=15, value=5, step=2, label="structuring element size", debounce=True)
    mo.vstack([
        image_dropdown_morph,
        mo.hstack([threshold_slider, se_shape_dropdown, se_size_slider], justify="start", gap=2),
    ])
    return (
        image_dropdown_morph,
        se_shape_dropdown,
        se_size_slider,
        threshold_slider,
    )


@app.cell
def _(mo):
    mo.md("""
    *(Foreground = pixels **darker** than the threshold — i.e. `luma < t`.)*
    """)
    return


@app.cell
def _(
    IMAGES_DIR,
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    build_structuring_element,
    image_dropdown_morph,
    mo,
    np,
    plt,
    se_shape_dropdown,
    se_size_slider,
    threshold_slider,
):
    _img_rgb = load_rgb255(IMAGES_DIR, image_dropdown_morph.value, plt, np)
    _img_gray = to_luma(_img_rgb)
    _binary = _img_gray < threshold_slider.value
    _se = build_structuring_element(se_shape_dropdown.value, se_size_slider.value)

    _dilated = binary_dilation(_binary, structure=_se)
    _eroded = binary_erosion(_binary, structure=_se)
    _opened = binary_opening(_binary, structure=_se)
    _closed = binary_closing(_binary, structure=_se)

    def _draw_binary(ax, a, title, cmap="gray"):
        ax.imshow(a, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _fig, _axes = plt.subplots(3, 2, figsize=(8, 11))
    _draw_binary(_axes[0, 0], _binary, f"binary (t={threshold_slider.value})")
    _se_panel = np.zeros((se_size_slider.value + 2, se_size_slider.value + 2))
    _se_panel[1:-1, 1:-1] = _se
    _draw_binary(_axes[0, 1], _se_panel, f"structuring element\n({se_shape_dropdown.value}, {se_size_slider.value}×{se_size_slider.value}, S={_se.sum()})")
    _draw_binary(_axes[1, 0], _dilated, "dilation")
    _draw_binary(_axes[1, 1], _eroded, "erosion")
    _draw_binary(_axes[2, 0], _opened, "opening")
    _draw_binary(_axes[2, 1], _closed, "closing")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Non-linear filters** use the neighborhood's *values*, not just
      positions, so a single outlier can't drag the result the way it
      would with a linear filter.
    - The **median filter** picks the middle value of the neighborhood —
      robust to shot/impulse noise, less good than a linear filter at
      averaging down ordinary Gaussian noise (Eq. 3.33).
    - The **bilateral filter** keeps a Gaussian-weighted-average structure
      but multiplies in a second, *intensity-based* weight (Eqs. 3.34–3.37)
      that rejects dissimilar neighbors — it reduces to a plain Gaussian
      blur as $\sigma_r\to\infty$, and preserves edges as $\sigma_r$ shrinks.
    - **Binary morphology** operates on thresholded (Eq. 3.44) images:
      **dilation** grows foreground regions, **erosion** shrinks them
      (Eq. 3.45), and their compositions **opening** (erode→dilate) and
      **closing** (dilate→erode) remove small foreground specks or fill
      small background holes respectively, while leaving large regions
      mostly intact.

    **Next up:** frequency-domain operators — the Fourier transform view of
    everything we've done with kernels so far in notebooks 8 and 9.
    """)
    return


if __name__ == "__main__":
    app.run()
