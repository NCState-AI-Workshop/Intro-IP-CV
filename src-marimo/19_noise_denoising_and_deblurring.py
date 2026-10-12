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
    from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

    return gaussian_filter, median_filter, mo, np, plt, uniform_filter


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Noise, Denoising, and Deblurring

    Notebooks 8 and 9 touched noise only in passing — a quick Gaussian blur, a
    median filter for salt-and-pepper speckle. This notebook covers Szeliski
    **§10.1 ("Photometric calibration," noise sections), §10.3 ("Super-resolution,
    denoising, and blur removal")**, and the related application §3.4.2
    ("Sharpening, blur, and noise removal"): what sensor noise actually looks
    like, two genuinely new *non-deep-learning* denoising algorithms (non-local
    means, total variation), and deblurring as deconvolution. We stay entirely
    pre-deep-learning on purpose — per the book's own framing, every one of
    today's best denoisers and deblurrers is now a trained network, and that's
    a different (and different-looking) notebook. HDR/tone-mapping (§10.2) and
    demosaicing (§10.3.1) are set aside for a possible future notebook.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting (notebooks 8-18's convention)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.function
def psnr(a, b, np, max_val=255.0):
    """Peak signal-to-noise ratio (Eq. 2.120) -- the book's own standard metric
    for denoising results against a known-clean reference."""
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(max_val**2 / mse) if mse > 0 else np.inf


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Noise models: what sensor noise actually looks like

    Real sensor noise is not the uniform "salt-and-pepper" speckle notebook 9
    used to motivate the median filter -- it comes from two physically distinct
    sources (§2.3, revisited in §10.1):

    - **Read noise.** Amplifier and quantization noise in the sensor's readout
      electronics. It's essentially independent of the signal, and is well
      modeled as **additive Gaussian noise**, $n_{read}\sim\mathcal N(0,\sigma_r^2)$.
    - **Shot noise.** Photon arrival is itself a random (Poisson) process: even
      a perfectly noiseless sensor imaging a perfectly constant radiance would
      still see photon counts fluctuate frame to frame. This noise is
      **signal-dependent** -- its standard deviation grows with $\sqrt{\text{signal}}$,
      so dark regions of a photo are *relatively* far noisier than bright ones,
      even though a bright region's *absolute* noise is larger.

    A real photograph has both at once -- a **mixed Poisson-Gaussian noise
    model**. Rather than track a single noise standard deviation, Szeliski
    §10.1.2 defines the **noise level function (NLF)**: the noise standard
    deviation as a function of the (noise-free) pixel value, estimated in
    practice either from repeated exposures (temporal variance; Healey and
    Kondepudy 1994) or, more practically, from the spatial variance within
    flat/slowly-varying regions of a *single* photo (Liu, Szeliski et al.
    2008) -- the method notebook uses next, in §2.

    The demo below simulates both noise sources separately, then mixed
    together, on the same clean image -- watch how the Poisson (shot) panel's
    noise is visibly *weaker in dark regions* (the table) and *stronger in
    bright regions* (the cup's rim and saucer), the opposite of what a single
    fixed-sigma Gaussian model would predict.
    """)
    return


@app.function
def add_gaussian_noise(img, sigma, rng):
    """Read noise: additive, signal-independent."""
    return img + rng.normal(0, sigma, img.shape)


@app.function
def add_poisson_noise(img, peak, rng, np):
    """Shot noise: simulate a photon count by scaling intensity to a peak
    photon budget, drawing Poisson samples, and rescaling back -- std grows
    with sqrt(signal), and a smaller peak means a noisier (lower-light) shot."""
    scaled = np.clip(img, 0, None) / 255.0 * peak
    return rng.poisson(scaled).astype(float) / peak * 255.0


@app.function
def add_mixed_noise(img, sigma_read, peak, rng, np):
    """The realistic case: shot noise from the imaging process plus read
    noise from the sensor electronics, both at once."""
    return add_gaussian_noise(add_poisson_noise(img, peak, rng, np), sigma_read, rng)


@app.cell
def _(mo):
    noise_sigma_slider = mo.ui.slider(start=5, stop=40, value=20, step=5, label="read noise σ", debounce=True)
    noise_peak_slider = mo.ui.slider(start=10, stop=300, value=40, step=10, label="photon peak (light level)", debounce=True)
    mo.hstack([noise_sigma_slider, noise_peak_slider], justify="start", gap=2)
    return noise_peak_slider, noise_sigma_slider


@app.cell
def _(IMAGES_DIR, mo, noise_peak_slider, noise_sigma_slider, np, plt):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _rng = np.random.default_rng(0)

    _dark_box = (slice(0, 40), slice(0, 40))
    _bright_box = (slice(10, 40), slice(160, 200))

    _g_noisy = add_gaussian_noise(_gray, noise_sigma_slider.value, _rng)
    _p_noisy = add_poisson_noise(_gray, noise_peak_slider.value, _rng, np)
    _mix_noisy = add_mixed_noise(_gray, noise_sigma_slider.value / 2, noise_peak_slider.value, _rng, np)

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for _ax, _img, _title in zip(
        _axes,
        [_gray, _g_noisy, _p_noisy, _mix_noisy],
        ["clean", f"+ read noise (σ={noise_sigma_slider.value})", f"+ shot noise (peak={noise_peak_slider.value})", "+ mixed (both)"],
    ):
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        for _box, _color in [(_dark_box, "cyan"), (_bright_box, "yellow")]:
            _ax.add_patch(plt.Rectangle(
                (_box[1].start, _box[0].start), _box[1].stop - _box[1].start, _box[0].stop - _box[0].start,
                fill=False, edgecolor=_color, linewidth=1.5,
            ))
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()

    _dark_std = (_p_noisy - _gray)[_dark_box].std()
    _bright_std = (_p_noisy - _gray)[_bright_box].std()

    mo.vstack([
        _fig,
        mo.md(
            f"In the shot-noise panel, the injected noise's std-dev in the **dark** "
            f"cyan box (mean level {_gray[_dark_box].mean():.0f}) is **{_dark_std:.1f}**, "
            f"versus **{_bright_std:.1f}** in the **bright** yellow box (mean level "
            f"{_gray[_bright_box].mean():.0f}) -- the same noise *source*, but roughly "
            f"${np.sqrt(_gray[_bright_box].mean()/_gray[_dark_box].mean()):.1f}\\times$ "
            f"stronger where the signal is brighter, exactly the "
            f"$\\sqrt{{\\text{{signal}}}}$ scaling shot noise predicts. Read noise (second "
            f"panel) has no such pattern -- it's the same everywhere."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Estimating noise level from an image

    Given only a noisy photo (no access to the clean original), how do you
    even know how much noise is present -- information every denoiser below
    needs as a parameter? Szeliski's practical answer (§10.1.2, Liu, Szeliski
    et al. 2008): find a region that *should* be flat or slowly varying, and
    treat its pixel-to-pixel deviation as a direct estimate of the local noise
    standard deviation. Below, the cyan box is a real flat region of
    `coffee.png` (the dark table background) with a *known* injected noise
    level -- compare the estimate recovered from nothing but that patch's
    spatial standard deviation against the ground truth.
    """)
    return


@app.cell
def _(mo):
    nle_sigma_slider = mo.ui.slider(start=5, stop=35, value=15, step=5, label="true injected σ", debounce=True)
    nle_sigma_slider
    return (nle_sigma_slider,)


@app.cell
def _(IMAGES_DIR, mo, nle_sigma_slider, np, plt):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _rng = np.random.default_rng(1)
    _noisy = add_gaussian_noise(_gray, nle_sigma_slider.value, _rng)

    _flat_box = (slice(0, 40), slice(0, 40))
    _patch = _noisy[_flat_box]
    # Remove the (small) intrinsic slope within the patch with a linear fit,
    # per the book's "fit a constant or linear function, then measure the
    # residual spread" recipe -- a flat patch isn't perfectly flat in reality.
    _yy, _xx = np.mgrid[0 : _patch.shape[0], 0 : _patch.shape[1]]
    _A = np.stack([_xx.ravel(), _yy.ravel(), np.ones(_xx.size)], axis=1)
    _coeffs, *_ = np.linalg.lstsq(_A, _patch.ravel(), rcond=None)
    _fitted = (_A @ _coeffs).reshape(_patch.shape)
    _estimated_sigma = (_patch - _fitted).std()

    _fig, _axes = plt.subplots(1, 2, figsize=(8.5, 4.3))
    _axes[0].imshow(np.clip(_noisy, 0, 255), cmap="gray", vmin=0, vmax=255)
    _axes[0].add_patch(plt.Rectangle((0, 0), 40, 40, fill=False, edgecolor="cyan", linewidth=1.5))
    _axes[0].set_title(f"noisy photo (true σ={nle_sigma_slider.value})")
    _im = _axes[1].imshow(_patch - _fitted, cmap="gray", vmin=-30, vmax=30)
    _axes[1].set_title(f"flat-region residual\n(estimated σ={_estimated_sigma:.1f})")
    for _ax in _axes:
        _ax.axis("off")
    _fig.colorbar(_im, ax=_axes[1], fraction=0.046)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "The estimate from one small patch already lands close to the true, "
            "known value -- this is exactly the number every denoiser below "
            "needs (NLM's $h$, TV's $\\lambda$) when there's no synthetic ground "
            "truth to tune against."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Why the old tools fall short

    Notebook 8's Gaussian blur and notebook 9's median filter can both reduce
    Gaussian noise, but neither was designed for it: Gaussian blur smooths
    noise and edges alike (it has no way to tell them apart), and the median
    filter's strength is rejecting *impulsive* outliers, not averaging down
    smoothly-distributed Gaussian noise. Below, both are tuned to their own
    best setting against the same noisy image -- even at their best, compare
    the zoomed crop: Gaussian blur visibly softens the real edge, while the
    median filter keeps the edge crisp but leaves the noise *within* each flat
    region still speckled, since there's no single "impulsive" outlier for it
    to reject. Non-local means (§4) and total variation (§5) are specifically
    designed to avoid both failure modes at once.
    """)
    return


@app.cell
def _(IMAGES_DIR, gaussian_filter, median_filter, mo, np, plt):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _rng = np.random.default_rng(0)
    _noisy = add_gaussian_noise(_gray, 20, _rng)
    _blurred = gaussian_filter(_noisy, sigma=1.0)
    _medianed = median_filter(_noisy, size=3)
    _crop = (slice(140, 230), slice(230, 340))

    _fig, _axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for _ax, _img, _title in zip(
        _axes[0],
        [_noisy, _blurred, _medianed],
        [f"noisy (PSNR {psnr(_noisy, _gray, np):.1f})", f"Gaussian blur σ=1 (PSNR {psnr(_blurred, _gray, np):.1f})", f"median k=3 (PSNR {psnr(_medianed, _gray, np):.1f})"],
    ):
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    for _ax, _img in zip(_axes[1], [_noisy, _blurred, _medianed]):
        _ax.imshow(np.clip(_img[_crop], 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.axis("off")
    _axes[1, 0].set_title("zoom: noisy", fontsize=9)
    _axes[1, 1].set_title("zoom: edges softened", fontsize=9)
    _axes[1, 2].set_title("zoom: noise still speckled", fontsize=9)
    _fig.tight_layout()
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Non-local means: weight by whole-patch similarity

    The bilateral filter (notebook 9) already improved on plain blurring by
    weighting each neighbor by how similar its *single pixel value* is.
    **Non-local means** (Buades, Coll, and Morel 2005a,b) pushes the same idea
    further in two ways: it compares whole *patches*, not single pixel values
    (far more robust to noise, since noise mostly cancels out when averaged
    over a patch), and it searches for similar patches over a *large* window,
    not just immediate neighbors -- repeated texture far away in the image is
    just as useful a match as a nearby pixel.

    $$ \mathrm{NLM}(i) = \frac{\sum_{j\in\mathcal{S}} w(i,j)\, I(j)}{\sum_{j\in\mathcal S} w(i,j)}, \qquad w(i,j) = \exp\!\left(-\frac{d(P_i,P_j)}{h^2}\right), $$

    where $\mathcal S$ is a search window around $i$, $P_i,P_j$ are small
    patches centered at $i,j$, $d(P_i,P_j)$ is their mean squared difference,
    and $h$ controls how strict the similarity match has to be (larger $h$
    blends more, including less-similar patches). The **search radius** $r$
    below sets how large $\mathcal S$ is: every pixel within $r$ pixels
    horizontally *and* vertically of $i$ -- a $(2r+1)\times(2r+1)$ window of
    *candidate match locations* (not to be confused with the much smaller
    patch itself, which stays a fixed $7\times7$ here) -- is checked as a
    possible match for $i$'s patch. A literal implementation
    compares every pixel against every candidate patch -- $O(n^2)$ overall,
    the same "brute-force but acceptable at this resolution" tradeoff
    notebooks 13-16 already made. A much faster trick avoids ever extracting
    patches explicitly: for each fixed candidate offset $(dy,dx)$, the whole
    image's patch-distance map is just a box-filtered squared-difference image
    (`scipy.ndimage.uniform_filter`), so the search loops over *offsets*, not
    pixels.
    """)
    return


@app.function
def non_local_means(img, h, patch_radius, search_radius, uniform_filter, np):
    """Buades-Coll-Morel non-local means via shift-and-box-filter: for each
    candidate offset, a box filter over the squared difference gives every
    pixel's patch-distance to its shifted candidate at once."""
    patch_size = 2 * patch_radius + 1
    acc = np.zeros_like(img)
    wsum = np.zeros_like(img)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
            patch_dist = uniform_filter((img - shifted) ** 2, size=patch_size)
            w = np.exp(-patch_dist / h**2)
            acc += w * shifted
            wsum += w
    return acc / wsum


@app.cell
def _(mo):
    nlm_h_slider = mo.ui.slider(start=5, stop=40, value=20, step=5, label="h (similarity strictness)", debounce=True)
    nlm_search_slider = mo.ui.slider(start=3, stop=10, value=7, step=1, label="search radius r (window (2r+1)×(2r+1))", debounce=True)
    mo.hstack([nlm_h_slider, nlm_search_slider], justify="start", gap=2)
    return nlm_h_slider, nlm_search_slider


@app.cell
def _(
    IMAGES_DIR,
    mo,
    nlm_h_slider,
    nlm_search_slider,
    np,
    plt,
    uniform_filter,
):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _rng = np.random.default_rng(0)
    _noisy = add_gaussian_noise(_gray, 20, _rng)
    _denoised = non_local_means(_noisy, nlm_h_slider.value, 3, nlm_search_slider.value, uniform_filter, np)
    _crop = (slice(140, 230), slice(230, 340))

    _fig, _axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for _ax, _img, _title in zip(
        _axes[0],
        [_gray, _noisy, _denoised],
        ["clean (reference)", f"noisy (PSNR {psnr(_noisy, _gray, np):.1f})", f"non-local means (PSNR {psnr(_denoised, _gray, np):.1f})"],
    ):
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    for _ax, _img in zip(_axes[1], [_gray, _noisy, _denoised]):
        _ax.imshow(np.clip(_img[_crop], 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Unlike §3's blur/median comparison, the zoomed crop here should keep "
            f"the cup's engraved detail crisp while still removing most of the "
            f"noise -- raise $h$ to blend more aggressively (and watch fine "
            f"detail start to wash out as it over-blends with less-similar "
            f"patches), or lower it to stay closer to the noisy original. The "
            f"search radius $r={nlm_search_slider.value}$ means every pixel is "
            f"compared against "
            f"${(2 * nlm_search_slider.value + 1)**2}$ candidate shifted "
            f"patches drawn from a "
            f"${2 * nlm_search_slider.value + 1}\\times{2 * nlm_search_slider.value + 1}$ "
            f"neighborhood around it -- raise $r$ to search farther afield for "
            f"a match (slower, and occasionally better if a good match lies "
            f"just outside the current window), or lower it to search a purely "
            f"local area."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Total variation (ROF) denoising

    Rudin, Osher, and Fatemi (1992) take a completely different approach:
    instead of an explicit weighted-average filter, pose denoising as an
    optimization problem. Find the image $u$ that is both close to the noisy
    input $f$ *and* has small total variation -- the sum of gradient
    magnitudes, which penalizes noisy, jagged gradients far more than it
    penalizes a single clean step edge of the same height:

    $$ u^* = \arg\min_u\; \mathrm{TV}(u) + \frac{1}{2\lambda}\|u-f\|_2^2, \qquad \mathrm{TV}(u) = \sum_{i,j}\|\nabla u_{i,j}\|_2, $$

    where $\lambda$ trades off fidelity to the noisy image (small $\lambda$)
    against smoothness (large $\lambda$). This is solved here by direct
    gradient descent on a smoothed (differentiable) version of $\mathrm{TV}$
    -- the *same* "discretize an energy, iterate" pattern already used for
    Horn-Schunck-style optical flow (notebook 18) and level-set contour
    evolution (notebook 16), applied here a third time. (Chambolle (2004)
    gives a much faster exact dual algorithm for this same energy, not
    implemented here.)
    """)
    return


@app.function
def tv_denoise(noisy, lam, n_iter, dt, np, eps=1e-3):
    """Gradient descent on TV(u) + ||u-f||^2/(2*lambda), forward-difference
    gradient / backward-difference divergence, with eps-smoothed TV so the
    gradient of |.|_2 stays well-defined at flat (zero-gradient) regions."""
    u = noisy.copy()
    for _ in range(n_iter):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        gmag = np.sqrt(ux**2 + uy**2 + eps**2)
        px, py = ux / gmag, uy / gmag
        div = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = u + dt * (div - (u - noisy) / lam)
    return u


@app.cell
def _(mo):
    tv_lambda_slider = mo.ui.slider(start=2, stop=30, value=12, step=2, label="λ (smoothness)", debounce=True)
    tv_niter_slider = mo.ui.slider(start=50, stop=400, value=250, step=50, label="iterations", debounce=True)
    mo.hstack([tv_lambda_slider, tv_niter_slider], justify="start", gap=2)
    return tv_lambda_slider, tv_niter_slider


@app.cell
def _(IMAGES_DIR, mo, np, plt, tv_lambda_slider, tv_niter_slider):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _rng = np.random.default_rng(0)
    _noisy = add_gaussian_noise(_gray, 20, _rng)
    _denoised = tv_denoise(_noisy, tv_lambda_slider.value, tv_niter_slider.value, dt=0.15, np=np)
    _crop = (slice(140, 230), slice(230, 340))

    _fig, _axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for _ax, _img, _title in zip(
        _axes[0],
        [_gray, _noisy, _denoised],
        ["clean (reference)", f"noisy (PSNR {psnr(_noisy, _gray, np):.1f})", f"TV / ROF (PSNR {psnr(_denoised, _gray, np):.1f})"],
    ):
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    for _ax, _img in zip(_axes[1], [_gray, _noisy, _denoised]):
        _ax.imshow(np.clip(_img[_crop], 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            "Push $\\lambda$ high and watch TV's signature look emerge: flat, "
            "piecewise-constant regions separated by crisp edges (**staircasing**) "
            "-- a direct consequence of penalizing total gradient *magnitude* "
            "rather than smoothness, unlike NLM's more texture-preserving result "
            "above."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The classical-methods roster

    Non-local means and TV are two real, implementable examples, but a recent
    survey (Gu and Timofte 2019) that Szeliski's own §10.3 cites lists many
    more "best non-learned" denoisers -- named and cited here, not
    implemented, since doing any of them justice is well beyond this
    notebook's scope:

    - **BM3D** (Dabov, Foi et al. 2007) -- block-matching, grouping similar
      patches into a 3D stack, then collaboratively filtering the whole stack
      in a transform domain. Long considered the strongest non-learned
      denoiser.
    - **K-SVD** (Aharon, Elad, and Bruckstein 2006) -- learns a sparse
      dictionary of patch "atoms" *from the noisy image itself*, then
      represents each patch as a sparse combination of them.
    - **EPLL** (Zoran and Weiss 2011) -- fits a Gaussian mixture prior over
      natural-image patches, then denoises by maximizing each patch's
      likelihood under that prior.
    - **WNNM** (Gu, Zhang et al. 2014) -- weighted nuclear-norm minimization
      over groups of similar patches, a low-rank-matrix take on the same
      patch-grouping idea as BM3D.
    - **Gaussian scale mixtures** (Lyu and Simoncelli 2009) and **Field of
      Experts** (Roth and Black 2009) -- statistical models of natural-image
      gradient/wavelet statistics, used as a denoising prior.

    Per Szeliski's own note, every one of these has since been overtaken by
    trained denoising networks, starting with **DnCNN** (Zhang, Zuo et al.
    2017) -- squarely the "different notebook" this one is deliberately not.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Blur as convolution with a point-spread function (PSF)

    A blurry photo -- out of focus, camera shake, a moving subject -- can
    almost always be modeled as the sharp image convolved with some kernel,
    the **point spread function (PSF)**: $B = I * K$. Calibrating a camera's
    real PSF (motion blur, defocus, lens aberrations, the anti-aliasing
    filter in front of the sensor) is itself a measurement problem (§10.1.4):
    photograph a known sharp pattern, then solve a least-squares fit for the
    unknown kernel $K$,

    $$ K = \arg\min_K \|B - D(I * K)\|^2, \qquad (10.1) $$

    where $D$ is an optional downsampling operator (Joshi, Szeliski, and
    Kriegman 2008). Two common PSF shapes are shown below: an (isotropic)
    Gaussian, typical of simple defocus blur, and a directional line kernel,
    typical of motion blur along the camera's shake direction -- both are
    just small 2D kernels, exactly notebook 10's convolution-theorem
    machinery, reused here for a new purpose: given a *known* (or calibrated)
    $K$, can the blur be reversed?
    """)
    return


@app.function
def gaussian_psf(size, sigma, np):
    r = size // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


@app.function
def motion_psf(length, angle_deg, np):
    """A thin line kernel at the given angle -- the classic motion-blur PSF."""
    size = length | 1
    k = np.zeros((size, size))
    r = size // 2
    theta = np.radians(angle_deg)
    for t in np.linspace(-r, r, length * 4):
        x, y = int(round(r + t * np.cos(theta))), int(round(r + t * np.sin(theta)))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    return k / k.sum()


@app.cell
def _(mo, np, plt):
    _gpsf = gaussian_psf(21, 3.0, np)
    _mpsf = motion_psf(21, 30, np)
    _fig, _axes = plt.subplots(1, 2, figsize=(6, 3.2))
    _axes[0].imshow(_gpsf, cmap="inferno")
    _axes[0].set_title("defocus (Gaussian PSF)", fontsize=9)
    _axes[1].imshow(_mpsf, cmap="inferno")
    _axes[1].set_title("motion blur (line PSF)", fontsize=9)
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Wiener deconvolution

    Given a known PSF, the convolution theorem (notebook 10) suggests an
    obvious fix: $B=I*K \;\Leftrightarrow\; \hat B=\hat I\hat K$ in the
    frequency domain, so just divide, $\hat I = \hat B / \hat K$, and invert.
    This **naive inverse filter** is catastrophic in practice: wherever
    $\hat K(\omega)$ is small (nearly every real PSF suppresses high
    frequencies), dividing by it massively *amplifies* whatever noise is
    present at that frequency -- turning a mild deconvolution into pure
    noise.

    **Wiener deconvolution** fixes this with a noise-aware regularization
    term,

    $$ \hat I(\omega) = \frac{\overline{\hat K(\omega)}}{|\hat K(\omega)|^2 + K_{reg}}\,\hat B(\omega), $$

    where $\overline{\hat K}$ is the complex conjugate. **What is $K_{reg}$?**
    The classical derivation picks $\hat I$ to minimize expected squared
    error given the noise and (unknown) signal's power spectra, which works
    out to a *per-frequency* regularizer $K_{reg}(\omega)=S_n(\omega)/S_f(\omega)$
    -- the ratio of noise power to signal power at that frequency. Where the
    signal is strong relative to the noise, $K_{reg}(\omega)$ is small and
    the formula is nearly the naive inverse; where noise dominates (as it
    typically does at high frequencies, exactly where $\hat K$ is already
    small), $K_{reg}(\omega)$ is large and the denominator stays safely
    bounded instead of blowing up. In practice neither power spectrum is
    known exactly, so $K_{reg}$ is usually simplified to one **scalar**
    tuned by hand (or from a noise estimate, §2) rather than a full function
    of $\omega$ -- exactly what the slider below does, trading a little of
    the formula's theoretical optimality for something estimable from a
    single photo.

    **How is $\hat K$ itself obtained here?** Everything above assumes the
    PSF is *known*. In this demo that's true by construction -- the exact
    kernel used to synthetically create the blur is handed straight to
    the deconvolution as $\hat K$, with zero estimation error. This
    isolates the regularization question above from the separate, harder
    problem of *calibrating* $K$ from a real photo in the first place
    (§7's Eq. 10.1); a real pipeline would deconvolve with a necessarily
    imperfect *estimated* $\hat K$, compounding calibration error on top of
    the noise-amplification trade-off shown below. Compare all three below
    on a known PSF -- switch between the Gaussian and motion kernels from §7
    to see how the *shape* of the blur changes what "well-regularized"
    looks like.
    """)
    return


@app.function
def psf_to_otf(psf, shape, np):
    """Zero-pad the kernel to image size and circularly shift it so its
    center lands at index (0,0) -- the standard alignment trick for FFT-based
    convolution to match the spatial-domain kernel's centering."""
    H, W = shape
    kh, kw = psf.shape
    padded = np.zeros((H, W))
    padded[:kh, :kw] = psf
    padded = np.roll(np.roll(padded, -(kh // 2), axis=0), -(kw // 2), axis=1)
    return np.fft.fft2(padded)


@app.function
def wiener_deconvolve(blurred, otf, k_reg, np):
    G = np.fft.fft2(blurred)
    F_hat = (np.conj(otf) / (np.abs(otf) ** 2 + k_reg)) * G
    return np.real(np.fft.ifft2(F_hat))


@app.function
def naive_inverse_filter(blurred, otf, eps, np):
    G = np.fft.fft2(blurred)
    return np.real(np.fft.ifft2(G / (otf + eps)))


@app.cell
def _(mo):
    wiener_blur_dropdown = mo.ui.dropdown(
        options=["Gaussian (defocus)", "motion (shake)"], value="Gaussian (defocus)", label="blur type"
    )
    wiener_blur_dropdown
    return (wiener_blur_dropdown,)


@app.cell
def _(mo, wiener_blur_dropdown):
    # The best K_reg genuinely differs by PSF shape (see the markdown above),
    # so the slider's starting point is re-picked per blur type rather than
    # sharing one default that would look broken for one of the two.
    _k_default = 0.01 if wiener_blur_dropdown.value == "Gaussian (defocus)" else 0.05
    wiener_k_slider = mo.ui.slider(start=0.001, stop=0.1, value=_k_default, step=0.001, label="K_reg (regularization strength)", debounce=True)
    wiener_noise_slider = mo.ui.slider(start=1, stop=15, value=5, step=1, label="noise σ (added to the blurred image)", debounce=True)
    mo.hstack([wiener_k_slider, wiener_noise_slider], justify="start", gap=2)
    return wiener_k_slider, wiener_noise_slider


@app.cell
def _(
    IMAGES_DIR,
    mo,
    np,
    plt,
    wiener_blur_dropdown,
    wiener_k_slider,
    wiener_noise_slider,
):
    _gray = to_luma(load_rgb255(IMAGES_DIR, "coffee", plt, np))
    _H, _W = _gray.shape
    _psf = gaussian_psf(21, 3.0, np) if wiener_blur_dropdown.value == "Gaussian (defocus)" else motion_psf(21, 30, np)
    _otf = psf_to_otf(_psf, (_H, _W), np)
    _blurred = np.real(np.fft.ifft2(np.fft.fft2(_gray) * _otf))
    _rng = np.random.default_rng(0)
    _blurred_noisy = _blurred + _rng.normal(0, wiener_noise_slider.value, _gray.shape)

    _naive = naive_inverse_filter(_blurred_noisy, _otf, eps=1e-3, np=np)
    _wiener = wiener_deconvolve(_blurred_noisy, _otf, wiener_k_slider.value, np)

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for _ax, _img, _title in zip(
        _axes,
        [_gray, _blurred_noisy, _naive, _wiener],
        [
            "original (sharp)",
            f"blurred + noisy (PSNR {psnr(_blurred_noisy, _gray, np):.1f})",
            f"naive inverse (PSNR {psnr(np.clip(_naive, 0, 255), _gray, np):.1f})",
            f"Wiener (PSNR {psnr(np.clip(_wiener, 0, 255), _gray, np):.1f})",
        ],
    ):
        _ax.imshow(np.clip(_img, 0, 255), cmap="gray", vmin=0, vmax=255)
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"**Blur: {wiener_blur_dropdown.value}.** `K_reg` is the scalar "
            f"regularization constant $K_{{reg}}$ in the formula above; "
            f"`noise σ` is the standard deviation of the Gaussian noise added "
            f"to the blurred image *before* any deconvolution is attempted "
            f"(a harder noise σ makes the naive inverse's amplification even "
            f"worse, and pushes the best `K_reg` higher too). The naive "
            f"inverse filter is essentially pure noise here -- exactly the "
            f"catastrophic amplification the regularization term exists to "
            f"prevent. Raise `K_reg` and the Wiener result drifts back toward "
            f"the blurred-but-safe input; lower it and it drifts toward the "
            f"naive filter's noise amplification. Notice the motion kernel "
            f"needs a noticeably *larger* `K_reg` than the Gaussian one to "
            f"look as good -- its directional PSF has deeper, narrower nulls "
            f"in frequency space, so naive inversion blows up more easily "
            f"there, and the regularizer has to work harder to compensate."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Beyond today

    - **Richardson-Lucy deconvolution** -- an iterative, non-negativity-respecting
      alternative to one-shot Wiener filtering, well known in the wider
      literature (not named in this particular textbook edition) and
      standard in astronomical and microscopy image restoration.
    - **Blind deconvolution** -- everything in §8 assumed a *known* PSF; when
      it's unknown too, the blur kernel and the sharp image must be estimated
      jointly, a genuinely harder chicken-and-egg problem (Kundur and
      Hatzinakos 1996; Levin 2006; Levin, Weiss et al. 2011; Campisi and
      Egiazarian 2017).
    - **Multi-frame / burst denoising.** Rather than denoise one noisy photo,
      production smartphone cameras capture and align a *burst* of frames at
      low ISO and combine them -- Google's HDR+ (Hasinoff, Sharlet et al.
      2016) and Night Sight (Liba, Murthy et al. 2019) systems are real,
      shipped examples of exactly this.
    - **Deep learning.** DnCNN (Zhang, Zuo et al. 2017) onward for denoising,
      and a parallel wave of learned deblurring networks, per §10.3 -- the
      current state of the art in both problems, and out of scope here.
    - Also set aside from this chapter: **HDR imaging and tone mapping**
      (§10.2) and **color image demosaicing** (§10.3.1) -- different
      problems, left for a possible future notebook.

    ## Summary

    - Real sensor noise is a **mixed Poisson-Gaussian** process -- signal-dependent
      shot noise plus signal-independent read noise -- characterized in
      practice by the **noise level function** (§10.1.2, Liu, Szeliski et al.
      2008), estimable from a single photo's flat regions.
    - Gaussian blur and median filtering (notebooks 8-9) were never designed
      for this noise and visibly cost real detail even at their best setting.
    - **Non-local means** (Buades, Coll, and Morel 2005a,b) and **total
      variation / ROF** (Rudin, Osher, and Fatemi 1992) are two real,
      implementable non-deep-learning denoisers, each with a distinct visual
      signature (texture-preserving vs. piecewise-constant staircasing); BM3D,
      K-SVD, EPLL, and WNNM round out the classical state of the art
      (cited, not implemented).
    - Blur is convolution with a **PSF** (Eq. 10.1); naive frequency-domain
      inversion catastrophically amplifies noise, which **Wiener
      deconvolution**'s regularization term is specifically designed to
      avoid -- directly reusing notebook 10's convolution-theorem machinery
      for a new purpose.
    """)
    return


if __name__ == "__main__":
    app.run()
