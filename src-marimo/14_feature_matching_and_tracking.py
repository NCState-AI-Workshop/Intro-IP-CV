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
    from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates

    return (
        correlate2d,
        gaussian_filter,
        map_coordinates,
        maximum_filter,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    TEXTBOOK_FIGURES_DIR = IMAGES_DIR / "textbook_figures"
    return IMAGES_DIR, TEXTBOOK_FIGURES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Feature Matching and Tracking

    Notebook 13 found stable keypoints; the guest lecture on SIFT and HOG covered
    turning each one into a descriptor vector. This notebook picks up right after
    that: given two sets of descriptors, how do we decide which ones correspond to
    each other (**matching**, §7.1.3), and how do we follow a feature across a
    video sequence instead of re-detecting it from scratch every frame
    (**tracking**, §7.1.5)?

    To keep the focus on matching *mechanics* rather than repeating the SIFT
    descriptor already covered, this notebook uses the simplest legitimate
    descriptor the book itself describes — a bias/gain-normalized patch (the MOPS
    idea, §7.1.2) — not SIFT/HOG.

    Skipped, as citation-only asides (engineering/scaling concerns, not core CV
    concepts — matching this course's established pattern): efficient indexing
    (k-d trees, hashing, best-bin-first search, §7.1.3) and large-scale
    retrieval/CBIR (§7.1.4). No Predict/Investigate/Modify activities this time —
    just demos.
    """)
    return


@app.function
def to_luma(img255):
    """Standard NTSC/ITU-R BT.601 luma weighting, applied directly to
    gamma-encoded RGB (the same convention used in notebooks 8-13)."""
    return 0.299 * img255[:, :, 0] + 0.587 * img255[:, :, 1] + 0.114 * img255[:, :, 2]


@app.function
def load_rgb255(images_dir, name, plt, np):
    """Load one of the saved test images as a (H,W,3) float array in [0,255]."""
    return plt.imread(str(images_dir / f"{name}.png"))[:, :, :3].astype(np.float64) * 255.0


@app.cell
def _(np):
    # Normalized Sobel (notebook 13): dividing by 8 makes the response equal the
    # true intensity-change-per-pixel, needed for quantitative eigenvalue work.
    GX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    GY = GX.T

    def compute_gradients(img_gray, sigma, gaussian_filter, correlate2d):
        smoothed = gaussian_filter(img_gray, sigma=sigma, mode="reflect") if sigma > 0 else img_gray
        Ix = correlate2d(smoothed, GX, mode="same", boundary="symm")
        Iy = correlate2d(smoothed, GY, mode="same", boundary="symm")
        return Ix, Iy

    return (compute_gradients,)


@app.cell
def _(np):
    def compute_homography(src_pts, dst_pts):
        """Same DLT-style solve as notebook 12: 4 point correspondences -> 3x3 H."""
        A, b = [], []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
            b.append(xp)
            A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
            b.append(yp)
        h = np.linalg.solve(np.array(A), np.array(b))
        return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

    def make_second_frame(img_gray, np, map_coordinates, theta_deg, tx, ty):
        """Warp img_gray by a small known rotation+translation to synthesize a
        'next frame' with a *known* ground-truth transform, so match/track
        correctness can be checked exactly instead of just eyeballed."""
        H_img, W_img = img_gray.shape
        cx, cy = W_img / 2, H_img / 2
        theta = np.deg2rad(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        src_corners = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=float)

        def rot(p):
            x, y = p[0] - cx, p[1] - cy
            return [c * x - s * y + cx + tx, s * x + c * y + cy + ty]

        dst_corners = np.array([rot(p) for p in src_corners])
        H = compute_homography(src_corners, dst_corners)

        yy, xx = np.mgrid[0:H_img, 0:W_img]
        Hinv = np.linalg.inv(H)
        pts = np.stack([xx.ravel().astype(float), yy.ravel().astype(float), np.ones(H_img * W_img)])
        src = Hinv @ pts
        src_x = (src[0] / src[2]).reshape(H_img, W_img)
        src_y = (src[1] / src[2]).reshape(H_img, W_img)
        frame_b = map_coordinates(img_gray, [src_y, src_x], order=1, mode="constant", cval=0.0)
        return frame_b, H

    def project_point(H, x, y):
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    return make_second_frame, project_point


@app.cell
def _(gaussian_filter, np):
    def harris_corners(img_gray, compute_gradients, correlate2d, sigma=2.0, thresh_frac=0.01, max_pts=150):
        """Same Harris detector as notebook 13 §4, packaged as a reusable function."""
        Ix, Iy = compute_gradients(img_gray, 0.5, gaussian_filter, correlate2d)
        Ixx = gaussian_filter(Ix**2, sigma=sigma)
        Iyy = gaussian_filter(Iy**2, sigma=sigma)
        Ixy = gaussian_filter(Ix * Iy, sigma=sigma)
        detA = Ixx * Iyy - Ixy**2
        traceA = Ixx + Iyy
        R = detA - 0.06 * traceA**2
        from scipy.ndimage import maximum_filter as _mf

        thresh = thresh_frac * R.max()
        local_max = (R == _mf(R, size=9)) & (R > thresh)
        ys, xs = np.nonzero(local_max)
        order = np.argsort(-R[ys, xs])[:max_pts]
        return xs[order], ys[order]

    return (harris_corners,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. A simple descriptor: bias/gain-normalized patches

    Rather than re-deriving SIFT, use the simplest patch descriptor the book
    describes (§7.1.2, the MOPS idea): take a small patch around each keypoint and
    normalize it to zero mean, unit variance,

    $$ D = \frac{P - \text{mean}(P)}{\text{std}(P)}, $$

    which makes the descriptor robust to simple brightness/contrast changes
    (Eq. 3.3's gain and bias) between the two images being matched.
    """)
    return


@app.cell
def _(np):
    def patch_descriptor(img_gray, xs, ys, half=8):
        """Bias/gain-normalized (MOPS-style) patch descriptor at each (x,y)."""
        H, W = img_gray.shape
        descs, valid = [], []
        for x, y in zip(xs, ys):
            if x - half < 0 or x + half >= W or y - half < 0 or y + half >= H:
                valid.append(False)
                continue
            patch = img_gray[y - half : y + half, x - half : x + half]
            descs.append((patch - patch.mean()) / (patch.std() + 1e-6))
            valid.append(True)
        return descs, np.array(valid)

    return (patch_descriptor,)


@app.cell
def _(mo):
    mo.md(r"""
    **A couple of notes on this descriptor.** Unlike SIFT or HOG, which turn a
    patch into a gradient-orientation histogram, MOPS' descriptor *is* the patch:
    the vector $D$ above is literally the (normalized) pixel intensities
    themselves, flattened — no further encoding step.

    That normalization also quietly determines how matching works. §2 below
    ranks candidate matches by plain **Euclidean distance** between descriptor
    vectors — the same distance metric the book uses regardless of descriptor
    type (Eq. 7.18's $d_1,d_2$ are Euclidean distances too). But because each
    patch here has already been rescaled to zero mean and unit variance, that
    Euclidean distance is secretly equivalent to **normalized cross-correlation
    (NCC)**: for two length-$n$ vectors $x,y$ with $\|x\|^2=\|y\|^2=n$,

    $$ \|x-y\|^2 = \|x\|^2+\|y\|^2-2\,x\cdot y = 2n\big(1-\text{NCC}(x,y)\big), $$

    since $x\cdot y/n$ *is* the correlation coefficient between the two patches.
    So minimizing Euclidean distance and maximizing correlation give the exact
    same ranking here — the bias/gain normalization is what makes plain
    Euclidean matching behave like correlation-based matching, without ever
    computing a correlation explicitly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Matching strategies (§7.1.3)

    Given two sets of descriptors, how should we decide which pairs correspond?

    - **Fixed threshold**: accept any pair within a fixed distance. Too high a
      threshold lets in false positives; too low a threshold misses true matches
      (Figure 7.21, Table 7.1 — the TPR/FPR/PPV/ACC vocabulary for scoring a match
      set, Eqs. 7.14–7.17).
    - **Nearest neighbor**: match each descriptor to its single closest neighbor.
      Better, but still accepts a bad match whenever the true correspondence
      simply isn't the closest thing in feature space.
    - **Nearest-neighbor distance ratio (NNDR)** (Eq. 7.18, Figure 7.23) improves
      on both:

    $$ \text{NNDR} = \frac{d_1}{d_2} $$

    where $d_1,d_2$ are the nearest and second-nearest distances. A **small**
    ratio means the best match is decisively better than the runner-up
    (trustworthy); a **large** ratio means the two best candidates are nearly
    tied (ambiguous — reject it), even if $d_1$ itself looks small.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_21_false_positives_negatives.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.21 and Table 7.1 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 442, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Table 7.1's four counts, applied to matching (Figure 7.21):

    - **TP** (true positives): correct matches
    - **FN** (false negatives): true correspondences the algorithm failed to match
    - **FP** (false positives): proposed matches that are actually wrong
    - **TN** (true negatives): non-matches correctly rejected

    These combine into four rates (Eqs. 7.14–7.17):

    $$
    \text{TPR} = \frac{TP}{TP+FN} = \frac{TP}{P}, \qquad
    \text{FPR} = \frac{FP}{FP+TN} = \frac{FP}{N},
    $$

    $$
    \text{PPV} = \frac{TP}{TP+FP} = \frac{TP}{P'}, \qquad
    \text{ACC} = \frac{TP+TN}{P+N},
    $$

    where $P=TP+FN$ and $N=FP+TN$ are the actual number of positives/negatives,
    and $P'=TP+FP$, $N'=FN+TN$ are the *predicted* number of positives/negatives.
    In Table 7.1's worked example ($TP=18$, $FP=4$, $FN=2$, $TN=76$): $\text{TPR}=0.90$
    (90% of true matches were found), $\text{FPR}=0.05$ (5% of true non-matches were
    incorrectly matched), $\text{PPV}=0.82$ (82% of the *proposed* matches were
    actually correct), $\text{ACC}=0.94$. Varying the matching threshold traces out
    a **receiver operating characteristic (ROC) curve** of TPR vs. FPR — the demo
    below effectively compares three single points on such a curve.
    """)
    return


@app.cell
def _(TEXTBOOK_FIGURES_DIR, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.imshow(plt.imread(TEXTBOOK_FIGURES_DIR / "szeliski_fig7_23_nndr_matching.png"))
    _ax.axis("off")
    mo.vstack([
        _fig,
        mo.md("*Figure 7.23 from Szeliski, **Computer Vision: Algorithms and Applications**, 2nd ed. (final draft, Sept. 2021), p. 445, reproduced for educational use.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    The demo below detects Harris corners (notebook 13) in an image and in a
    second, synthetically rotated+translated copy of it — since the exact
    transform is known, a proposed match's correctness can be checked directly
    (project the point through the known transform and see how close it lands to
    its match), so "correct" below isn't a guess.
    """)
    return


@app.cell
def _(mo):
    image_dropdown_match = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="chelsea", label="image"
    )
    theta_match_slider = mo.ui.slider(start=0, stop=8, value=4, step=1, label="rotation θ (°)", debounce=True)
    ratio_thresh_slider = mo.ui.slider(start=0.5, stop=1.0, value=0.8, step=0.05, label="NNDR threshold", debounce=True)
    mo.vstack([image_dropdown_match, mo.hstack([theta_match_slider, ratio_thresh_slider], justify="start", gap=2)])
    return image_dropdown_match, ratio_thresh_slider, theta_match_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    harris_corners,
    image_dropdown_match,
    make_second_frame,
    map_coordinates,
    mo,
    np,
    patch_descriptor,
    plt,
    project_point,
    ratio_thresh_slider,
    theta_match_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_match.value, plt, np))
    _grayB, _Hmat = make_second_frame(_img_gray, np, map_coordinates, theta_match_slider.value, 5.0, 3.0)

    _xsA, _ysA = harris_corners(_img_gray, compute_gradients, correlate2d)
    _xsB, _ysB = harris_corners(_grayB, compute_gradients, correlate2d)
    _descA, _validA = patch_descriptor(_img_gray, _xsA, _ysA)
    _descB, _validB = patch_descriptor(_grayB, _xsB, _ysB)
    _idxA = np.nonzero(_validA)[0]
    _idxB = np.nonzero(_validB)[0]
    _DA = np.stack(_descA).reshape(len(_descA), -1)
    _DB = np.stack(_descB).reshape(len(_descB), -1)
    _dists = np.linalg.norm(_DA[:, None, :] - _DB[None, :, :], axis=2)

    def _is_correct(i, j, tol=3.0):
        px, py = project_point(_Hmat, _xsA[_idxA[i]], _ysA[_idxA[i]])
        qx, qy = _xsB[_idxB[j]], _ysB[_idxB[j]]
        return np.hypot(px - qx, py - qy) < tol

    _fixed_thresh = np.percentile(_dists, 5)
    _fixed_matches = list(zip(*np.nonzero(_dists < _fixed_thresh)))
    _nn_matches = [(i, int(np.argmin(_dists[i]))) for i in range(_dists.shape[0])]
    _nndr_matches = []
    for _i in range(_dists.shape[0]):
        _order = np.argsort(_dists[_i])
        _d1, _d2 = _dists[_i, _order[0]], _dists[_i, _order[1]]
        if _d1 / max(_d2, 1e-9) < ratio_thresh_slider.value:
            _nndr_matches.append((_i, int(_order[0])))

    _fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    _gs = _fig.add_gridspec(2, 2, height_ratios=[1, 2])
    _ax_fixed = _fig.add_subplot(_gs[0, 0])
    _ax_nn = _fig.add_subplot(_gs[0, 1])
    _ax_nndr = _fig.add_subplot(_gs[1, :])
    _gap = _img_gray.shape[1]
    for _ax, _matches, _title in [
        (_ax_fixed, _fixed_matches, "fixed threshold"),
        (_ax_nn, _nn_matches, "nearest neighbor"),
        (_ax_nndr, _nndr_matches, "NNDR"),
    ]:
        _combined = np.hstack([_img_gray, _grayB])
        _ax.imshow(_combined, cmap="gray", vmin=0, vmax=255)
        _n_correct = 0
        for _i, _j in _matches:
            _correct = _is_correct(_i, _j)
            _n_correct += _correct
            _color = "lime" if _correct else "red"
            _ax.plot(
                [_xsA[_idxA[_i]], _xsB[_idxB[_j]] + _gap],
                [_ysA[_idxA[_i]], _ysB[_idxB[_j]]],
                color=_color, lw=0.7, alpha=0.8,
            )
        _pct = 100 * _n_correct / max(len(_matches), 1)
        _ax.set_title(f"{_title}\n{len(_matches)} matches, {_n_correct} correct ({_pct:.0f}%)", fontsize=10)
        _ax.axis("off")

    mo.vstack([
        mo.md("Green lines = geometrically correct (consistent with the known transform); red = incorrect. NNDR should show far fewer red lines than fixed threshold or plain nearest-neighbor, at a comparable or better match count."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Feature tracking (§7.1.5): the KLT idea

    An alternative to detecting and matching independently in every frame is to
    **detect once, then track**: find good features in frame 1, then search for
    each one's new location in frame 2. "Good" here means the same thing it did in
    notebook 13 — a large **minimum eigenvalue** of the auto-correlation matrix
    $A$ (Shi and Tomasi, 1994), since that's exactly the condition for a patch to
    be well localized in *every* direction.

    Tracking itself is simple for small motion: search a small window of
    displacements $\Delta u$ around the feature's old position for the one with
    lowest SSD (Eq. 7.1) — literally the same brute-force search used for
    notebook 13 §3's auto-correlation surface, just applied between two frames
    instead of one image and itself. For larger motion or longer sequences, the
    book describes refining with an **affine** motion model and using
    **hierarchical (coarse-to-fine) search** — not implemented here — and warns
    that naive re-matching against a drifting patch accumulates error over a long
    sequence (a random-walk effect).
    """)
    return


@app.cell
def _(mo):
    image_dropdown_track = mo.ui.dropdown(
        options=["astronaut", "coffee", "chelsea", "raccoon"], value="astronaut", label="image"
    )
    theta_track_slider = mo.ui.slider(start=0, stop=6, value=2, step=1, label="rotation θ (°)", debounce=True)
    n_features_slider = mo.ui.slider(start=10, stop=80, value=40, step=10, label="number of features to track", debounce=True)
    mo.vstack([image_dropdown_track, mo.hstack([theta_track_slider, n_features_slider], justify="start", gap=2)])
    return image_dropdown_track, n_features_slider, theta_track_slider


@app.cell
def _(
    IMAGES_DIR,
    compute_gradients,
    correlate2d,
    gaussian_filter,
    image_dropdown_track,
    make_second_frame,
    map_coordinates,
    maximum_filter,
    mo,
    n_features_slider,
    np,
    plt,
    project_point,
    theta_track_slider,
):
    _img_gray = to_luma(load_rgb255(IMAGES_DIR, image_dropdown_track.value, plt, np))
    _grayB, _Hmat = make_second_frame(_img_gray, np, map_coordinates, theta_track_slider.value, 3.5, -2.5)

    _Ix, _Iy = compute_gradients(_img_gray, 0.5, gaussian_filter, correlate2d)
    _sigma_int = 2.0
    _Ixx = gaussian_filter(_Ix**2, sigma=_sigma_int)
    _Iyy = gaussian_filter(_Iy**2, sigma=_sigma_int)
    _Ixy = gaussian_filter(_Ix * _Iy, sigma=_sigma_int)
    _trace = _Ixx + _Iyy
    _diff = _Ixx - _Iyy
    _min_eig = (_trace - np.sqrt(_diff**2 + 4 * _Ixy**2)) / 2

    _local_max = (_min_eig == maximum_filter(_min_eig, size=9)) & (_min_eig > 0.01 * _min_eig.max())
    _ys, _xs = np.nonzero(_local_max)
    _order = np.argsort(-_min_eig[_ys, _xs])[: n_features_slider.value]
    _xs, _ys = _xs[_order], _ys[_order]

    _half, _search = 7, 8
    _tracked_du, _tracked_dv, _true_du, _true_dv, _valid_x, _valid_y = [], [], [], [], [], []
    _H_img, _W_img = _img_gray.shape
    for _x, _y in zip(_xs, _ys):
        if _x - _half - _search < 0 or _x + _half + _search >= _W_img or _y - _half - _search < 0 or _y + _half + _search >= _H_img:
            continue
        _patchA = _img_gray[_y - _half : _y + _half + 1, _x - _half : _x + _half + 1]
        _best_ssd = np.inf
        _best_du, _best_dv = 0, 0
        for _dv in range(-_search, _search + 1):
            for _du in range(-_search, _search + 1):
                _patchB = _grayB[_y - _half + _dv : _y + _half + 1 + _dv, _x - _half + _du : _x + _half + 1 + _du]
                _ssd = np.sum((_patchA - _patchB) ** 2)
                if _ssd < _best_ssd:
                    _best_ssd, _best_du, _best_dv = _ssd, _du, _dv
        _tx_true, _ty_true = project_point(_Hmat, _x, _y)
        _tracked_du.append(_best_du)
        _tracked_dv.append(_best_dv)
        _true_du.append(_tx_true - _x)
        _true_dv.append(_ty_true - _y)
        _valid_x.append(_x)
        _valid_y.append(_y)

    _tracked_du, _tracked_dv = np.array(_tracked_du), np.array(_tracked_dv)
    _true_du, _true_dv = np.array(_true_du), np.array(_true_dv)
    _errors = np.hypot(_tracked_du - _true_du, _tracked_dv - _true_dv)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.imshow(_grayB, cmap="gray", vmin=0, vmax=255)
    _ax.quiver(
        _valid_x, _valid_y, _tracked_du, _tracked_dv, color="lime",
        angles="xy", scale_units="xy", scale=1, label="tracked displacement",
    )
    _ax.scatter(_valid_x, _valid_y, c="red", s=10, label="feature start (frame 1)")
    _ax.legend(fontsize=9, loc="upper right")
    _ax.set_title(f"{len(_errors)} features tracked — mean error {_errors.mean():.2f}px, max {_errors.max():.2f}px")
    _ax.axis("off")

    mo.vstack([
        mo.md("Green arrows show the tracked displacement into frame 2; the reported error compares this against the exact known ground-truth motion (since we constructed frame 2 ourselves). Most features should track to well under 1 pixel of error — a few larger errors are expected where rotation locally distorts the patch beyond what pure translation search can find."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - **Matching strategy** matters as much as the descriptor itself:
      **NNDR** (Eq. 7.18) — comparing the best match to the runner-up, not just
      thresholding the best match alone — rejects far more false matches than a
      fixed threshold or plain nearest-neighbor at a comparable true-positive rate.
    - **Feature tracking** reuses notebook 13's auto-correlation eigenvalues
      directly as the "good features to track" criterion, then follows each one
      via local patch search — the classic **KLT tracker** idea, still the basis
      of feature-based tracking in SLAM and augmented reality today.

    **Next up:** edge detection (§7.2.1) — a different, far denser kind of
    feature, and the groundwork a future notebook on active contours will need.
    """)
    return


if __name__ == "__main__":
    app.run()
