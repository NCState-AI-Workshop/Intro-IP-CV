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

    return correlate2d, mo, np, plt


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Fourier Transforms

    Notebooks 8 and 9 stayed entirely in the **spatial domain**: every filter
    was a kernel slid across the image, pixel by pixel. This notebook
    introduces the other way to look at an image or a filter — as a sum of
    sinusoids of different frequencies — following Szeliski **§3.4, "Fourier
    transforms."**
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8 and 9)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.function
def center_crop_to(img, shape):
    """Center-crop a 2D array down to `shape` (used to pair two differently-sized images)."""
    H, W = shape
    h0, w0 = img.shape[:2]
    y0, x0 = (h0 - H) // 2, (w0 - W) // 2
    return img[y0 : y0 + H, x0 : x0 + W]


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. A filter's response to a sinusoid

    Pass a sinusoid $s(x) = \sin(\omega x + \phi_i)$ (Eq. 3.50) through a
    filter $h(x)$, and out comes *another sinusoid at the same frequency* —
    only its amplitude and phase have changed (Eq. 3.51):

    $$ o(x) = h(x) * s(x) = A\sin(\omega x + \phi_o). $$

    Tabulating that gain $A$ and phase shift $\phi_o - \phi_i$ across every
    frequency $\omega$ *is* the Fourier transform of the filter,

    $$ H(\omega) = \mathcal{F}\{h(x)\} = Ae^{j\phi} \qquad \text{(Eq. 3.54)}. $$

    Pick a small 1D kernel and a test frequency below. The left panel shows
    the actual input/output sinusoids; the right panel shows the full gain
    curve $|H(\omega)|$ (computed via the discrete Fourier transform, Eq.
    3.57) with the chosen frequency marked — the red dot's height should
    match the amplitude ratio visible on the left.
    """)
    return


@app.cell
def _(np):
    KERNEL_OPTIONS_1D = [
        "box-3",
        "box-5",
        "binomial-3 [1,2,1]",
        "binomial-5 [1,4,6,4,1]",
        "Sobel [-1,0,1]",
        "corner (2nd diff.) [1,-2,1]",
    ]

    def get_kernel_1d(name):
        return {
            "box-3": np.ones(3) / 3,
            "box-5": np.ones(5) / 5,
            "binomial-3 [1,2,1]": np.array([1.0, 2.0, 1.0]) / 4,
            "binomial-5 [1,4,6,4,1]": np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16,
            "Sobel [-1,0,1]": np.array([-1.0, 0.0, 1.0]) / 2,
            "corner (2nd diff.) [1,-2,1]": np.array([1.0, -2.0, 1.0]),
        }[name]

    def freq_response_1d(h, omegas):
        """H(omega) = sum_k h(k) exp(-j k omega), k centered at 0 (Eq. 3.57-style sum)."""
        L = len(h)
        offsets = np.arange(L) - (L - 1) // 2
        H = np.zeros(len(omegas), dtype=complex)
        for k, hk in zip(offsets, h):
            H = H + hk * np.exp(-1j * k * omegas)
        return H

    return KERNEL_OPTIONS_1D, freq_response_1d, get_kernel_1d


@app.cell
def _(KERNEL_OPTIONS_1D, mo):
    kernel_dropdown_freq = mo.ui.dropdown(options=KERNEL_OPTIONS_1D, value="binomial-5 [1,4,6,4,1]", label="1D kernel")
    freq_slider = mo.ui.slider(start=0.02, stop=0.48, value=0.10, step=0.01, label="test frequency f (cycles/sample)", debounce=True)
    mo.vstack([kernel_dropdown_freq, freq_slider])
    return freq_slider, kernel_dropdown_freq


@app.cell
def _(
    freq_response_1d,
    freq_slider,
    get_kernel_1d,
    kernel_dropdown_freq,
    mo,
    np,
    plt,
):
    _h = get_kernel_1d(kernel_dropdown_freq.value)
    _omega0 = 2 * np.pi * freq_slider.value

    _n = 200
    _x = np.arange(_n)
    _s = np.sin(_omega0 * _x)
    _o = np.convolve(_s, _h, mode="same")

    _omegas = np.linspace(0, np.pi, 400)
    _mag = np.abs(freq_response_1d(_h, _omegas))
    _mag0 = np.abs(freq_response_1d(_h, np.array([_omega0]))[0])

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _window = slice(60, 140)
    _axes[0].plot(_x[_window], _s[_window], label="input s(x)", lw=1.5)
    _axes[0].plot(_x[_window], _o[_window], label="output o(x) = h*s", lw=1.5)
    _axes[0].set_title(f"time domain, f={freq_slider.value:.2f}")
    _axes[0].set_xlabel("x")
    _axes[0].legend(fontsize=8)

    _axes[1].plot(_omegas / (2 * np.pi), _mag, color="black")
    _axes[1].axvline(freq_slider.value, color="tab:red", ls="--", lw=1)
    _axes[1].plot([freq_slider.value], [_mag0], "o", color="tab:red")
    _axes[1].set_xlim(0, 0.5)
    _axes[1].set_ylim(bottom=0)
    _axes[1].set_xlabel("frequency f (cycles/sample)")
    _axes[1].set_title(f"gain |H(f)| — {kernel_dropdown_freq.value}")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Two-dimensional Fourier transforms

    Everything above extends directly to images: an oriented 2D sinusoid
    $s(x,y)=\sin(\omega_x x + \omega_y y)$ (Eq. 3.58) has a 2D transform

    $$ H(\omega_x,\omega_y) = \iint h(x,y)\, e^{-j(\omega_x x + \omega_y y)}\, dx\, dy \qquad \text{(Eq. 3.59)}, $$

    with the discrete version (Eq. 3.60) computed here via the FFT. The
    usual way to *look* at $|F(\omega_x,\omega_y)|$ is a **log-magnitude
    spectrum**, shifted so zero frequency sits at the center: bright pixels
    near the middle are coarse, low-frequency structure, and brightness
    farther out is finer, high-frequency detail (edges, texture).
    """)
    return


@app.cell
def _(mo):
    image_dropdown_fft = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    image_dropdown_fft
    return (image_dropdown_fft,)


@app.cell
def _(IMAGES_DIR, image_dropdown_fft, mo, np, plt):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_fft.value, plt, np))
    _F = np.fft.fftshift(np.fft.fft2(_img_gray))
    _log_mag = np.log1p(np.abs(_F))

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original (grayscale)")
    _axes[1].imshow(_log_mag, cmap="inferno")
    _axes[1].set_title("log-magnitude spectrum log(1+|F|)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Magnitude vs. phase

    A 2D Fourier transform is complex-valued: at every frequency it has
    both a **magnitude** (how much of that frequency is present) and a
    **phase** (where it's positioned). It's tempting to assume magnitude —
    the part usually visualized as the spectrum above — carries most of an
    image's information. It doesn't.

    Below, two images are cropped to a common size, Fourier transformed,
    and their magnitude and phase are **swapped** before inverting the
    transform back to an image. Watch which reconstruction looks like which
    source image.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_phase_a = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image A"
    )
    image_dropdown_phase_b = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="coffee", label="image B"
    )
    mo.hstack([image_dropdown_phase_a, image_dropdown_phase_b], justify="start", gap=2)
    return image_dropdown_phase_a, image_dropdown_phase_b


@app.cell
def _(IMAGES_DIR, image_dropdown_phase_a, image_dropdown_phase_b, mo, np, plt):
    _a = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_phase_a.value, plt, np))
    _b = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_phase_b.value, plt, np))
    _common_shape = (min(_a.shape[0], _b.shape[0]), min(_a.shape[1], _b.shape[1]))
    _a = center_crop_to(_a, _common_shape)
    _b = center_crop_to(_b, _common_shape)

    _Fa, _Fb = np.fft.fft2(_a), np.fft.fft2(_b)
    _mag_a, _phase_a = np.abs(_Fa), np.angle(_Fa)
    _mag_b, _phase_b = np.abs(_Fb), np.angle(_Fb)

    _out_phaseA_magB = np.real(np.fft.ifft2(_mag_b * np.exp(1j * _phase_a)))
    _out_phaseB_magA = np.real(np.fft.ifft2(_mag_a * np.exp(1j * _phase_b)))

    _fig, _axes = plt.subplots(2, 2, figsize=(9, 8.5))
    _axes[0, 0].imshow(_a, cmap="gray", vmin=0, vmax=255)
    _axes[0, 0].set_title(f"A: {image_dropdown_phase_a.value}")
    _axes[0, 1].imshow(_b, cmap="gray", vmin=0, vmax=255)
    _axes[0, 1].set_title(f"B: {image_dropdown_phase_b.value}")
    _axes[1, 0].imshow(_out_phaseA_magB, cmap="gray")
    _axes[1, 0].set_title("phase(A) + magnitude(B)")
    _axes[1, 1].imshow(_out_phaseB_magA, cmap="gray")
    _axes[1, 1].set_title("phase(B) + magnitude(A)")
    for _row in _axes:
        for _ax in _row:
            _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("The bottom-left image should look structurally like **A**, and the bottom-right like **B** — despite each having the *other* image's magnitude. Phase, not magnitude, carries most of the structural information."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The convolution theorem

    Convolving two signals in the spatial domain is the
    *same* as multiplying their transforms in the frequency domain,

    $$ f * h \;\; \overset{\mathcal{F}}{\longleftrightarrow} \;\; F(\omega)\cdot H(\omega). $$

    This is also *why* the FFT matters practically: multiplying two
    transformed images is $O(N)$, and even with the forward/inverse FFTs
    included, filtering a large image with a large kernel this way is
    $O(N\log N)$ — independent of the kernel's size — versus $O(N\cdot
    K^2)$ for direct spatial correlation.

    Below, the same Gaussian-blurred image is produced two ways: spatially,
    with `correlate2d(..., mode="same", boundary="symm")` exactly as in
    notebook 8 §5–6; and via the convolution theorem, multiplying in the
    frequency domain (zero-padded to avoid wraparound) and inverting. Since
    a Gaussian kernel is symmetric under 180° rotation, correlation and
    convolution coincide here (as in notebook 8), so this is an
    apples-to-apples comparison.
    """)
    return


@app.cell
def _(np):
    def gaussian_kernel_2d(sigma):
        size = int(np.ceil(6 * sigma)) | 1
        r = size // 2
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return k / k.sum()

    def fft_linear_correlate_same(img, kernel):
        """Linear convolution via the convolution theorem, zero-padded to
        avoid circular wraparound, cropped back to 'same' output size."""
        H, W = img.shape
        kh, kw = kernel.shape
        out_h, out_w = H + kh - 1, W + kw - 1
        F_img = np.fft.fft2(img, s=(out_h, out_w))
        F_ker = np.fft.fft2(kernel, s=(out_h, out_w))
        full = np.real(np.fft.ifft2(F_img * F_ker))
        sh, sw = kh // 2, kw // 2
        return full[sh : sh + H, sw : sw + W]

    return fft_linear_correlate_same, gaussian_kernel_2d


@app.cell
def _(mo):
    image_dropdown_convthm = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    sigma_convthm_slider = mo.ui.slider(start=1.0, stop=8.0, value=3.0, step=0.5, label="Gaussian σ", debounce=True)
    mo.vstack([image_dropdown_convthm, sigma_convthm_slider])
    return image_dropdown_convthm, sigma_convthm_slider


@app.cell
def _(
    IMAGES_DIR,
    correlate2d,
    fft_linear_correlate_same,
    gaussian_kernel_2d,
    image_dropdown_convthm,
    mo,
    np,
    plt,
    sigma_convthm_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_convthm.value, plt, np))
    _kernel = gaussian_kernel_2d(sigma_convthm_slider.value)

    _spatial = correlate2d(_img_gray, _kernel, mode="same", boundary="symm")
    _freq = fft_linear_correlate_same(_img_gray, _kernel)

    _r = _kernel.shape[0] // 2
    _diff = np.abs(_spatial - _freq)
    _interior_max = _diff[_r:-_r, _r:-_r].max()

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_spatial, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("spatial: correlate2d\n(notebook 8 convention)")
    _axes[1].imshow(_freq, cmap="gray", vmin=0, vmax=255)
    _axes[1].set_title("frequency: FFT multiply\n(convolution theorem)")
    _axes[2].imshow(_diff, cmap="inferno")
    _axes[2].set_title(f"|difference|\n(interior max: {_interior_max:.1e})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("The two blurred images should be visually identical, and the interior max difference above should be at machine precision. The difference map lights up only in a thin border strip: `correlate2d`'s `boundary=\"symm\"` mirrors pixels at the edge, while the FFT version implicitly zero-pads — different boundary conventions, not a bug."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Filtering directly in the frequency domain

    Since filtering is just multiplying by $H(\omega)$, we can design a
    filter *as* a frequency-domain mask instead of a spatial kernel. The
    simplest low-pass filter you could imagine is an **ideal** filter: keep
    every frequency inside some cutoff radius, discard everything outside.
    Table 3.1's lesson generalizes here — a hard cutoff doesn't uniformly
    damp high frequencies, it truncates them sharply, which shows up as
    **ringing** (Gibbs-phenomenon halos) around sharp edges. A **Gaussian**
    cutoff falls off smoothly instead, with no ringing — at the cost of
    also softening more of the low-frequency content near the cutoff.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_freqfilt = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    cutoff_slider = mo.ui.slider(start=0.02, stop=0.40, value=0.08, step=0.01, label="cutoff (fraction of Nyquist)", debounce=True)
    filter_type_dropdown = mo.ui.dropdown(
        options=["ideal (hard cutoff)", "Gaussian (soft cutoff)"], value="ideal (hard cutoff)", label="low-pass type"
    )
    mo.vstack([image_dropdown_freqfilt, mo.hstack([cutoff_slider, filter_type_dropdown], justify="start", gap=2)])
    return cutoff_slider, filter_type_dropdown, image_dropdown_freqfilt


@app.cell
def _(
    IMAGES_DIR,
    cutoff_slider,
    filter_type_dropdown,
    image_dropdown_freqfilt,
    mo,
    np,
    plt,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_freqfilt.value, plt, np))
    _H, _W = _img_gray.shape
    _fy = np.fft.fftshift(np.fft.fftfreq(_H))
    _fx = np.fft.fftshift(np.fft.fftfreq(_W))
    _FX, _FY = np.meshgrid(_fx, _fy)
    _R = np.sqrt(_FX**2 + _FY**2)
    _cutoff = cutoff_slider.value

    if filter_type_dropdown.value.startswith("ideal"):
        _mask = (_R <= _cutoff).astype(float)
    else:
        _mask = np.exp(-(_R**2) / (2 * _cutoff**2))

    _F = np.fft.fftshift(np.fft.fft2(_img_gray))
    _filtered = np.real(np.fft.ifft2(np.fft.ifftshift(_F * _mask)))

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
    _axes[0].imshow(_img_gray, cmap="gray", vmin=0, vmax=255)
    _axes[0].set_title("original")
    _axes[1].imshow(_filtered, cmap="gray", vmin=0, vmax=255)
    _axes[1].set_title(f"low-pass filtered\n({filter_type_dropdown.value}, cutoff={_cutoff:.2f})")
    _axes[2].imshow(np.log1p(np.abs(_F)), cmap="inferno")
    _axes[2].contour(_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    _axes[2].set_title("log-magnitude spectrum\n(cyan = mask boundary)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        mo.md("Look closely near sharp edges with the ideal filter at a small cutoff: faint repeated halos are ringing. Switch to the Gaussian cutoff at the same radius and they should disappear, at the cost of a slightly softer result."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A filter's **Fourier transform** $H(\omega)$ tabulates the
      gain/phase it applies to every input frequency (Eqs. 3.50–3.54);
      the **DFT** (Eq. 3.57) and its fast implementation, the **FFT**,
      compute this directly from a kernel's coefficients.
    - **2D Fourier transforms** (Eqs. 3.58–3.60) extend this to images;
      the log-magnitude spectrum shows how much of each spatial frequency
      is present, but **phase**, not magnitude, carries most of an
      image's structural information.
    - The **convolution theorem** — convolution in space equals
      multiplication in frequency — connects everything back to notebooks
      8–9's spatial filters, and is why FFT-based filtering scales as
      $O(N\log N)$ instead of $O(N\cdot K^2)$ for large kernels.
    - Filters can be designed directly as frequency-domain masks: an
      **ideal** (hard-cutoff) low-pass filter causes **ringing**; a
      **Gaussian** (soft-cutoff) filter avoids it — the same lesson as
      Table 3.1's box-vs-binomial comparison, now visible in 2D.

    **Next up:** pyramids and multi-resolution representations (§3.5) —
    Gaussian and Laplacian pyramids, and why that same ringing-avoidance
    problem matters again when repeatedly downsampling an image.
    """)
    return


if __name__ == "__main__":
    app.run()
