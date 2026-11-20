# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "onnxruntime",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle


@app.cell
def _():
    ASSETS = mo.notebook_dir() / "assets" / "lecture2"
    return (ASSETS,)


@app.function
def figure_or_placeholder(path, caption, todo):
    """Show a pre-rendered figure (PNG or GIF) if it exists; otherwise a placeholder."""
    for candidate in (path, path.with_suffix(".gif")):
        if candidate.exists():
            return mo.vstack([mo.image(src=candidate.read_bytes()), mo.md(f"*{caption}*")])
    return mo.callout(
        mo.md(
            f"**Figure not generated yet** (`{path.name}`).\n\n{todo}\n\n"
            "*Instructor: run `build_lecture2_assets.py` to create it.*"
        ),
        kind="warn",
    )


@app.function
def missing_asset_callout(files, what):
    return mo.callout(
        mo.md(
            f"**Demo data not found** ({', '.join(f'`{f}`' for f in files)} in `assets/lecture2/`). "
            f"*Instructor: run `build_lecture2_assets.py`.*\n\n{what}"
        ),
        kind="warn",
    )


@app.cell
def _():
    mo.md(r"""
    # Segmentation Lecture 2 (of 3): Segmenting Without Your Own Labels
    ## ECE 492-071 / CSC 491-001 Intro to Image Processing and Computer Vision (IPCV)
    ## Dr. Chau-Wai Wong, co-developed with Claude Code

    Last time, every model learned from **dense masks**: a human label for every pixel of every
    training image. Those masks are expensive — Cityscapes reports more than 1.5 hours of annotation
    per image. Today we look at two ways around that cost:

    - **DINO** learns visual features with *no labels at all*;
    - **SAM** is trained once on an enormous collection of masks (built largely by the model itself), so
      that *you* do not need masks for your own problem.

    | Question | Lecture 1: dense prediction | Lecture 2: foundation models | Lecture 3: optimization |
    |---|---|---|---|
    | What defines a segment? | A fixed label set chosen by the dataset | A **prompt**: a click, a box, or a phrase | *next time* |
    | Where does the knowledge come from? | Dense human annotation | **Self-supervision** (DINO) and **data engines** (SAM) | *next time* |
    | How is the answer computed? | One feed-forward pass | **Prompt-conditioned decoding** on top of a heavy, reusable encoder | *next time* |

    **By the end of today you should be able to:**

    1. explain how DINO learns features without labels, and why it does not collapse to a trivial answer;
    2. explain why DINO's patch features are useful for segmentation;
    3. describe SAM's promptable task, its architecture, and how it handles ambiguous prompts;
    4. explain what SAM 2 and SAM 3 add;
    5. state the open question that Lecture 3 answers.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.1 Self-supervised learning in one picture

    **Self-supervised learning** manufactures a training signal from the data itself. The main families,
    in one line each:

    - *Contrastive* (SimCLR, MoCo): two augmented views of the same image should have similar embeddings;
      views of different images should differ.
    - *Self-distillation* (BYOL, **DINO**): a student network predicts the output of a teacher network on a
      different view of the same image; no "different image" examples are needed.
    - *Masked image modeling* (MAE): hide most of the image's patches and reconstruct them. (SAM's image
      encoder was pretrained this way.)

    **The core danger is collapse.** "Make two views agree" is trivially satisfied by mapping *every*
    image to the same output. Every self-supervised method needs some mechanism that prevents this.
    Keep this question in mind as we look at DINO.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.2 DINO and its descendants

    **DINO** — self-**di**stillation with **no** labels (Caron et al., 2021).

    **Setup.** Two networks with the same architecture (a Vision Transformer followed by a small
    projection head): a *student* and a *teacher*. Each outputs a vector of $K$ scores that is turned into
    a probability distribution with a softmax ($K = 65{,}536$ in the paper; these $K$ "classes" are not
    predefined — the network invents them). From each image, generate several augmented crops: two large
    *global* crops and several small *local* crops. **The teacher sees only the global crops; the student additionally sees small crops.**

    **The training signal.** The student's output distribution on *any* crop must match the teacher's
    output distribution on a *different* global crop. Matching a small local crop to a global view forces
    the student to infer the whole object from a part. The mismatch is measured with cross-entropy, the
    same loss as in classification — except that the "label" is the teacher's distribution instead of a
    one-hot vector.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart LR
        X["Image"] --> V1["Global crop 1"]
        X --> V2["Global crop 2"]
        X --> VL["Local crops"]
        V1 --> T["Teacher<br/>(no gradient)"]
        V2 --> T
        V1 --> S["Student"]
        V2 --> S
        VL --> S
        T --> CS["center, then sharpen<br/>(low temperature)"] --> LOSS["cross-entropy<br/>between different crops"]
        S --> SM["softmax"] --> LOSS
        S -.->|"moving average<br/>of weights"| T
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    **The teacher is not trained by gradient descent.** Its weights are a slowly *exponential moving average* of the
    student's weights, updated after every training step:

    $$
    \theta_t[n+1] \leftarrow \lambda\, \theta_t[n] + (1-\lambda)\, \theta_s[n]
    $$

    where

    - $n$ is the time index;
    - $\theta_s$ are the student's weights (trained normally with gradients);
    - $\theta_t$ are the teacher's weights;
    - $\lambda$ is a number close to 1 (it starts at 0.996 and rises to 1 during training).

    In words: the teacher is a smoothed, more stable version of the student's recent history, so it
    provides steadier targets.

    ---

    If unrolled:

    $$
    \theta_t[n] = \lambda\,\theta_t[n-1] + (1-\lambda)\,\theta_s[n]
    = (1-\lambda)\sum_{k=0}^{n-1} \lambda^{k}\,\theta_s[n-k] \;+\; \lambda^{n}\,\theta_t[0].
    $$

    $\theta_t$ is a **weighted average** of all past student weights, with weights $O(\lambda^k)$ that shrink geometrically with "age" $k$: recent students count most, old ones fade. With $\lambda = 0.996$, roughly the last 250 steps matter.

    ---

    **The danger: collapse.** "Make the outputs agree" has two trivial solutions: (a) every image produces
    the same *peaked* output distribution (always "class 17"), or (b) every image produces the *flat*, uniform output distribution.
    Both make the loss small and the features useless. DINO applies two simple operations to the teacher's
    output, each blocking one kind of collapse:

    - **(a) Centering** subtracts a moving average of the teacher's outputs, so no single output dimension can
      dominate for every image. This blocks (a), but on its own pushes toward flat outputs.
    - **(b) Sharpening** uses a low softmax *temperature* for the teacher, making its distribution more peaked.
      This blocks (b), but on its own pushes toward (a).

    Used together, they balance. The temperature $\tau$ enters the softmax like this:

    $$
    \operatorname{softmax}(z/\tau)_k = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}
    $$

    where $z$ is a score vector, $z_k$ its $k$-th entry, and $\tau > 0$ the temperature. For
    $z = (2, 1, 0)$: with $\tau = 1$ the probabilities are about $(0.67, 0.24, 0.09)$; with $\tau = 0.5$
    they become about $(0.87, 0.12, 0.02)$. **Lower temperature = more peaked.** (DINO's teacher uses
    $\tau \approx 0.04$–$0.07$; the student uses $0.1$.)
    """)
    return


@app.cell
def _():
    col_tau = mo.ui.slider(start=0.05, stop=2.0, value=1.0, step=0.05, label="teacher temperature τ", debounce=False)
    col_center = mo.ui.checkbox(value=False, label="centering (subtract the batch average)")
    mo.vstack([mo.md("### Demo: centering and sharpening against collapse"), mo.hstack([col_tau, col_center], justify="start", gap=2)])
    return col_center, col_tau


@app.function
def teacher_logits():
    """A toy batch of 6 images x 8 output dimensions. Dimension 0 is large for every image (a shared
    bias that drives collapse (a)); each image also has its own preferred dimension."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.15, (6, 8))
    z[:, 0] += 2.5
    for i in range(6):
        z[i, i + 1] += 1.2
    return z


@app.cell
def _(col_center, col_tau):
    _z = teacher_logits()
    if col_center.value:
        _z = _z - _z.mean(axis=0, keepdims=True)
    _p = np.exp(_z / col_tau.value)
    _p /= _p.sum(axis=1, keepdims=True)
    _entropy = -(_p * np.log(_p + 1e-12)).sum(axis=1).mean() / np.log(_p.shape[1])
    _distinct = len(set(_p.argmax(axis=1)))

    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _im = _ax.imshow(_p, cmap="Blues", vmin=0, vmax=1)
    for _i in range(_p.shape[0]):
        _ax.text(_p[_i].argmax(), _i, "★", ha="center", va="center", color="orange", fontsize=12)
    _ax.set_xlabel("output dimension (an invented 'class')")
    _ax.set_ylabel("image in the batch")
    _ax.set_title("teacher's output distribution for each image (★ = most likely)")
    plt.colorbar(_im, ax=_ax, fraction=0.03)
    _fig.tight_layout()

    if _distinct == 1:
        _verdict = "**Collapse (a):** every image picks the same dimension."
    elif _entropy > 0.85:
        _verdict = "**Close to collapse (b):** the outputs are almost flat — they carry little information."
    else:
        _verdict = "**Healthy:** different images pick different dimensions, with confident outputs."
    mo.hstack(
        [
            _fig,
            mo.md(
                f"Different most-likely dimensions across the 6 images: **{_distinct}**\n\n"
                f"Average flatness (normalized entropy, 0 = one-hot, 1 = uniform): **{_entropy:.2f}**\n\n"
                f"{_verdict}\n\n*Try:* centering off with a low τ (collapse a); centering on with τ = 2 "
                "(nearly flat, collapse b); centering on with τ ≈ 0.1 (healthy)."
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### The emergent property: DINO attends to objects

    In a Vision Transformer, a special `[CLS]` token (token index = 0) summarizes the image, and its *attention map* shows
    which patches it looks at. In a DINO-trained ViT, the last layer's `[CLS]` attention concentrates on the
    main objects, with sharp outlines. Thresholding these maps gives object masks of surprising quality,
    even though the network never saw a mask or a label. A ViT trained with labels produces much noisier
    attention maps.
    """)
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig2_2a_dino_attention.png",
        "Last-layer [CLS] attention of a DINO ViT-S/16, one panel per attention head, and the mask obtained "
        "by keeping the patches that hold 60% of the attention.",
        "Shows each attention head of a DINO-trained ViT on a few images, plus the thresholded mask.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### DINOv2, registers, and DINOv3

    **DINOv2** (Oquab et al., 2023) scaled the recipe up with a curated dataset of 142 million images and
    added a second objective: predict the features of hidden (masked) patches. The result is strong
    **per-patch** features. A frozen DINOv2 backbone plus a single linear layer gives competitive semantic
    segmentation.

    **Registers** (Darcet et al., 2024). Large ViTs, including DINOv2, develop a few odd "artifact" patches
    on boring background areas; the network secretly uses them as scratch memory, which spoils the
    attention and feature maps. Adding a few extra learnable tokens ("registers"), which are thrown away at
    the output, gives the network dedicated scratch space, and the artifacts disappear.

    **DINOv3** (Siméoni et al., 2025) scales further: a 7-billion-parameter model trained on about 1.7
    billion images, then distilled into smaller models. Its main new ingredient, **Gram anchoring**, fixes a
    problem seen in long training runs: the whole-image features keep improving while the per-patch features
    slowly get worse. Gram anchoring adds a loss that keeps the *table of patch-to-patch similarities* close
    to that of an earlier checkpoint whose patch features were still good. Remember that table — it is the
    **affinity matrix** at the end of today's lecture.
    """)
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig2_2b_registers.png",
        "Patch features of DINOv2 without registers (top) and with registers (bottom). For each image: the "
        "first three principal components of the patch features shown as RGB, and the length (norm) of each "
        "patch feature. The artifact patches are the isolated very bright spots in the norm map.",
        "Compares DINOv2 patch features with and without register tokens: PCA colors and feature-norm maps.",
    )
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig2_2c_kmeans.png",
        "The same k-means algorithm (K = 4) run on pixel colors (left) and on DINO patch features (right). "
        "Only the features changed.",
        "Compares k-means clustering on raw pixel colors with k-means on DINO patch features.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    **What to notice in the last figure.** The clustering algorithm is the one from notebook 16 — nothing
    about it changed. With pixel colors, clusters follow *colors*; with DINO features, clusters follow
    *objects and parts*. In other words: without any labels, DINO-family features make patches of the
    same object look alike. They give us an excellent measure of **affinity** between patches. We will
    come back to that at the end of the lecture.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.3 SAM: promptable segmentation

    **Segment Anything** (Kirillov et al., 2023) changes the *task* before changing the model.

    **The task: promptable segmentation.** Given an image and a *prompt* — foreground/background clicks, a
    box, or a rough mask — return a *valid* mask. "Valid" means that even when the prompt is ambiguous, the
    output should be a reasonable mask for at least one of the things the prompt could mean. SAM outputs
    masks without class names: it knows *where* objects are, not *what* they are.

    **Why define the task this way?** A promptable model can be reused for many problems just by changing
    the prompt: click-based interactive segmentation, turning a detector's boxes into masks, or automatic
    "segment everything" by prompting with a grid of points.

    **Architecture: heavy once, light many times.**
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart LR
        I["Image<br/>1024×1024"] --> E["Image encoder<br/>(large ViT)<br/>run once per image"]
        E --> EMB["Image embedding<br/>64×64×256"]
        P["Prompts:<br/>points, box, mask"] --> PE["Prompt encoder"]
        EMB --> MD["Mask decoder<br/>(small transformer)<br/>runs per prompt, milliseconds"]
        PE --> MD
        MD --> M["3 candidate masks"]
        MD --> S["3 predicted IoU scores"]
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    - **Image encoder.** A large ViT (pretrained by masked image modeling) that turns the image into a
      $64\times64$ grid of 256-number embeddings. Expensive, but computed **once per image**.
    - **Prompt encoder.** Turns each click or box corner into a vector that encodes its position and its
      type ("foreground point," "background point," "box corner").
    - **Mask decoder.** A small transformer in which the prompt vectors and the image embedding attend to
      each other; it outputs masks. Fast enough to run interactively in a web browser.

    **Ambiguity.** A single click on a person's shirt could mean the shirt, the person, or a button. If the
    model had to output one mask, it would learn to average these possibilities and produce a blurry,
    invalid mask. SAM instead outputs **three** masks (roughly whole, part, and sub-part) plus a *predicted
    IoU* for each — the model's own estimate of how good each mask is. During training, only the best of
    the three candidates receives the loss.

    **Training, briefly.** The mask loss is a combination of focal loss and Dice loss — both from Lecture 1.
    Interactive use is simulated during training: after each prediction, a new click is sampled from the
    region the model got wrong.

    **The data engine.** No dataset of a billion masks existed, so the authors built one in three stages:

    1. *Assisted-manual:* annotators click with an early SAM in the loop; the model is retrained as data grows.
    2. *Semi-automatic:* SAM proposes the masks it is confident about; annotators add what is missing.
    3. *Fully automatic:* SAM is prompted with a $32\times32$ grid of points on every image, and confident,
       stable masks are kept.

    The result, **SA-1B**, has about 11 million images and 1.1 billion masks, almost all from the fully
    automatic stage. The model and the dataset improved each other.

    **Limitations.** No class names; thin structures (wires, hair) can be lost; it may choose the "wrong"
    level among whole / part / sub-part; and a specialized model trained on one domain can still beat it there.
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            "**Not a new interface.** Clicking points and drawing boxes to get a mask is much older than SAM. "
            "In the early 2000s, interactive segmentation used the *same interface* with a completely different "
            "engine: a color model of this one image plus a graph cut. You will build that engine in Lecture 3."
        ),
        kind="info",
    )
    return


@app.cell
def _(ASSETS):
    import onnxruntime as ort

    _onnx, _data = ASSETS / "sam_decoder.onnx", ASSETS / "sam_embeddings.npz"
    if _onnx.exists() and _data.exists():
        sam_session = ort.InferenceSession(str(_onnx))
        _d = np.load(_data)
        sam_names = [str(_n) for _n in _d["names"]]
        sam_images = {_n: _d[f"image_{_n}"] for _n in sam_names}
        sam_embeds = {_n: _d[f"embed_{_n}"].astype(np.float32) for _n in sam_names}
    else:
        sam_session, sam_names, sam_images, sam_embeds = None, [], {}, {}
    return sam_embeds, sam_images, sam_names, sam_session


@app.cell
def _(sam_names):
    _opts = sam_names if sam_names else ["(no images)"]
    sam_img = mo.ui.dropdown(options=_opts, value=_opts[0], label="image")
    sam_p1x = mo.ui.slider(0, 100, value=50, label="point 1: x (%)")
    sam_p1y = mo.ui.slider(0, 100, value=50, label="point 1: y (%)")
    sam_p1l = mo.ui.radio(options=["foreground", "background"], value="foreground", inline=True)
    sam_p2on = mo.ui.checkbox(value=False, label="add point 2")
    sam_p2x = mo.ui.slider(0, 100, value=30, label="point 2: x (%)")
    sam_p2y = mo.ui.slider(0, 100, value=30, label="point 2: y (%)")
    sam_p2l = mo.ui.radio(options=["foreground", "background"], value="background", inline=True)
    sam_boxon = mo.ui.checkbox(value=False, label="add a box")
    sam_bx = mo.ui.range_slider(0, 100, value=[25, 75], label="box: x range (%)")
    sam_by = mo.ui.range_slider(0, 100, value=[25, 75], label="box: y range (%)")
    sam_multi = mo.ui.checkbox(value=True, label="show all 3 candidate masks")
    mo.vstack(
        [
            mo.md("### Demo: prompting SAM"),
            mo.hstack([sam_img, sam_multi], justify="start", gap=2),
            mo.hstack([sam_p1x, sam_p1y, sam_p1l], justify="start", gap=1),
            mo.hstack([sam_p2on, sam_p2x, sam_p2y, sam_p2l], justify="start", gap=1),
            mo.hstack([sam_boxon, sam_bx, sam_by], justify="start", gap=1),
        ]
    )
    return (
        sam_boxon,
        sam_bx,
        sam_by,
        sam_img,
        sam_multi,
        sam_p1l,
        sam_p1x,
        sam_p1y,
        sam_p2l,
        sam_p2on,
        sam_p2x,
        sam_p2y,
    )


@app.function
def sam_decode(session, embed, image_hw, points, labels, box=None):
    """Run SAM's prompt encoder + mask decoder (exported to ONNX by the official script).
    points: (N, 2) array of (x, y) in image pixels; labels: 1 = foreground, 0 = background.
    Returns (masks: (4, H, W) logits, iou: (4,)). Index 0 is the single-mask output; 1..3 are the
    three candidates of the multi-mask output."""
    H, W = image_hw
    scale = 1024 / max(H, W)  # SAM resizes the long side to 1024
    new_h, new_w = int(H * scale + 0.5), int(W * scale + 0.5)
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    lab = np.asarray(labels, np.float32).reshape(-1)
    if box is not None:
        x0, y0, x1, y1 = box
        pts = np.concatenate([pts, [[x0, y0], [x1, y1]]]).astype(np.float32)
        lab = np.concatenate([lab, [2, 3]]).astype(np.float32)  # 2, 3 = box corners
    else:
        pts = np.concatenate([pts, [[0.0, 0.0]]]).astype(np.float32)  # padding point, as in the official example
        lab = np.concatenate([lab, [-1]]).astype(np.float32)
    pts = pts * np.array([new_w / W, new_h / H], np.float32)
    masks, iou, _ = session.run(
        None,
        {
            "image_embeddings": embed,
            "point_coords": pts[None],
            "point_labels": lab[None],
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([H, W], np.float32),
        },
    )
    return masks[0], iou[0]


@app.cell
def _(
    sam_boxon,
    sam_bx,
    sam_by,
    sam_embeds,
    sam_images,
    sam_img,
    sam_multi,
    sam_p1l,
    sam_p1x,
    sam_p1y,
    sam_p2l,
    sam_p2on,
    sam_p2x,
    sam_p2y,
    sam_session,
):
    if sam_session is None:
        _out = missing_asset_callout(
            ["sam_decoder.onnx", "sam_embeddings.npz"],
            "The demo runs only SAM's small prompt encoder and mask decoder; each image's embedding from the "
            "large image encoder was computed once in advance. That is exactly SAM's 'heavy once, light many "
            "times' design.",
        )
    else:
        _img = sam_images[sam_img.value]
        _H, _W = _img.shape[:2]
        _pts = [[sam_p1x.value / 100 * (_W - 1), sam_p1y.value / 100 * (_H - 1)]]
        _labs = [1 if sam_p1l.value == "foreground" else 0]
        if sam_p2on.value:
            _pts.append([sam_p2x.value / 100 * (_W - 1), sam_p2y.value / 100 * (_H - 1)])
            _labs.append(1 if sam_p2l.value == "foreground" else 0)
        _box = None
        if sam_boxon.value:
            _box = (sam_bx.value[0] / 100 * (_W - 1), sam_by.value[0] / 100 * (_H - 1),
                    sam_bx.value[1] / 100 * (_W - 1), sam_by.value[1] / 100 * (_H - 1))
        _masks, _iou = sam_decode(sam_session, sam_embeds[sam_img.value], (_H, _W), _pts, _labs, _box)
        _show = [1, 2, 3] if sam_multi.value else [0]

        _fig, _axes = plt.subplots(1, 1 + len(_show), figsize=(4.2 * (1 + len(_show)), 4.4), squeeze=False)
        _axes = _axes[0]
        _axes[0].imshow(_img)
        _axes[0].set_title("image and prompts")
        for _a in _axes:
            for (_x, _y), _l in zip(_pts, _labs):
                _a.plot(_x, _y, marker="*" if _l else "X", ms=16, mec="white", mew=1.5,
                        color="limegreen" if _l else "red", ls="none")
            if _box is not None:
                _a.add_patch(Rectangle(_box[:2], _box[2] - _box[0], _box[3] - _box[1], fill=False, ec="yellow", lw=2))
        for _a, _k in zip(_axes[1:], _show):
            _m = _masks[_k] > 0
            _ov = _img / 255.0 * 0.45
            _ov[_m] = 0.45 * _img[_m] / 255.0 + 0.55 * np.array([0.1, 0.6, 1.0])
            _a.imshow(_ov)
            _a.set_title(f"{'candidate ' + str(_k) if sam_multi.value else 'single mask'}: predicted IoU {_iou[_k]:.2f}")
        for _a in _axes:
            _a.axis("off")
        _fig.tight_layout()
        _out = mo.vstack([
            _fig,
            mo.md("Green star = foreground click, red X = background click, yellow = box. *Try:* one click on an "
                  "ambiguous spot (a part of a larger object) and compare the three candidates; then add a "
                  "background click, or a box, and watch the ambiguity disappear. Every update takes "
                  "milliseconds because the heavy image encoder already ran."),
        ])
    _out
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.4 SAM 2: memory for video

    **SAM 2** (Ravi et al., 2024) extends promptable segmentation to video. You click on an object in any
    frame; the model segments it in every frame, and you can add corrections on later frames.

    The key new component is a **memory**. When SAM 2 processes a frame, it also attends to a memory bank
    holding information about the object from recent frames and from the frames you clicked on. After each
    frame, the new prediction is written into the memory. A single image is treated as a one-frame video, so
    the same model handles both. Its training data came from another data engine, producing the SA-V
    video dataset.

    Following an object's outline through time is what the snake and level-set *trackers* of notebook 16
    tried to do with hand-designed curve dynamics; SAM 2 learns the dynamics from data.
    """)
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig2_4_sam2_strip.png",
        "SAM 2 on a short clip: one click on the first frame, and the mask propagated through the remaining "
        "frames.",
        "Shows several frames of a video with the object mask that SAM 2 propagated from a single click on "
        "the first frame.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.5 SAM 3 and open-vocabulary segmentation

    **The missing piece: names.** SAM returns masks but cannot answer "segment all the cups." The common
    workaround (2023–2025) chained two models: an **open-vocabulary detector** turns a text phrase into
    boxes, then SAM turns each box into a mask. The best-known detector for this was Grounding DINO, and the
    combination was known as "Grounded-SAM."
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            "**Name clash.** *Grounding DINO* is built on a DETR-style **detector** that is also called DINO "
            "(\"DETR with Improved deNoising anchOr boxes\"). It has nothing to do with the self-supervised "
            "**DINO** from §2.2 except the acronym."
        ),
        kind="warn",
    )
    return


@app.cell
def _():
    mo.md(r"""
    **SAM 3** (Meta, November 2025) folds detection into the model. Its task is **promptable concept
    segmentation**: given a short noun phrase (e.g. "striped red umbrella") or an example image region,
    return masks for **every** instance of that concept in an image, with consistent identities across
    video frames. The original click and box prompts still work. Detection, segmentation, and tracking are
    unified in one model, again trained with a data engine — this time with AI models serving as annotators
    and verifiers alongside humans.

    Across the SAM family, the *prompt* defines what a segment is: a click (SAM), a click on one frame of a
    video (SAM 2), a concept phrase (SAM 3).
    """)
    return


@app.cell
def _(ASSETS):
    _dir = ASSETS / "fig2_5_sam3"
    _files = sorted(_dir.glob("*.png")) if _dir.exists() else []
    sam3_pick = (
        mo.ui.dropdown(options={_f.stem.replace("_", " "): _f for _f in _files},
                       value=_files[0].stem.replace("_", " "), label="text prompt")
        if _files else None
    )
    sam3_pick
    return (sam3_pick,)


@app.cell
def _(ASSETS, sam3_pick):
    if sam3_pick is not None:
        _out = mo.vstack([
            mo.image(src=sam3_pick.value.read_bytes()),
            mo.md("*SAM 3's output for the selected text prompt: every instance of the concept, each with its own mask.*"),
        ])
    else:
        _out = figure_or_placeholder(
            ASSETS / "fig2_5_sam3" / "prompt.png",
            "",
            "Shows one busy image segmented by SAM 3 for several text prompts (e.g. 'person', 'cup', 'wheel'), "
            "selectable from a dropdown. The outputs are pre-rendered, so no model runs here.",
        )
    _out
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.6 The question for next time: an affinity graph

    Return to DINO. Split one image into $n$ patches and take each patch's DINO feature vector, scaled to
    length 1. Collect all pairwise similarities into an $n \times n$ matrix:

    $$
    W_{ij} = f_i^\top f_j
    $$

    where

    - $i, j = 1, \dots, n$ index the patches (here $n = 32 \times 32 = 1024$);
    - $f_i$ is the feature vector of patch $i$ (384 numbers for the small model used here), scaled so that
      $\|f_i\| = 1$;
    - $W_{ij}$ is their dot product, which for unit vectors is the cosine of the angle between them: close to
      1 when the two patches "look like the same thing," much smaller when they don't.

    $W$ is called an **affinity matrix**. One row of it — the similarity of one chosen patch to every other
    patch — can be drawn as a heatmap over the image.
    """)
    return


@app.cell
def _(ASSETS):
    _p = ASSETS / "dino_features.npz"
    if _p.exists():
        _d = np.load(_p)
        dino_names = [str(_n) for _n in _d["names"]]
        dino_images = dict(zip(dino_names, _d["images"]))
        _f = _d["feats"].astype(np.float32)
        _f /= np.linalg.norm(_f, axis=-1, keepdims=True) + 1e-8
        dino_feats = dict(zip(dino_names, _f))
    else:
        dino_names, dino_images, dino_feats = [], {}, {}
    return dino_feats, dino_images, dino_names


@app.cell
def _(dino_feats, dino_names):
    _opts = dino_names if dino_names else ["(no images)"]
    _g = next(iter(dino_feats.values())).shape[0] if dino_feats else 32
    aff_img = mo.ui.dropdown(options=_opts, value=_opts[0], label="image")
    aff_px = mo.ui.slider(0, _g - 1, value=_g // 2, label="patch column")
    aff_py = mo.ui.slider(0, _g - 1, value=_g // 2, label="patch row")
    mo.vstack([mo.md("### Demo: one row of the affinity matrix"), mo.hstack([aff_img, aff_px, aff_py], justify="start", gap=2)])
    return aff_img, aff_px, aff_py


@app.function
def kmeans(X, k, iters=30, seed=0):
    """Plain k-means (as in notebook 16), used here only to reorder the affinity matrix."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return lab


@app.cell
def _(aff_img, aff_px, aff_py, dino_feats, dino_images):
    if not dino_feats:
        _out = missing_asset_callout(
            ["dino_features.npz"],
            "The demo uses DINO patch features computed in advance for a few images; it only multiplies "
            "vectors, so it needs no model.",
        )
    else:
        _F = dino_feats[aff_img.value]
        _img = dino_images[aff_img.value]
        _g = _F.shape[0]
        _flat = _F.reshape(-1, _F.shape[-1])
        _sim = (_flat @ _F[aff_py.value, aff_px.value]).reshape(_g, _g)
        _cell = _img.shape[0] / _g

        _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.8))
        _axes[0].imshow(_img)
        _axes[0].add_patch(Rectangle((aff_px.value * _cell, aff_py.value * _cell), _cell, _cell, fill=False, ec="red", lw=2))
        _axes[0].set_title("image (red = chosen patch)")
        _axes[1].imshow(_img)
        _lo = float(np.percentile(_sim, 5))  # color scale from the 5th percentile up to 1
        _hm = _axes[1].imshow(np.kron(_sim, np.ones((int(_cell), int(_cell)))), cmap="inferno", alpha=0.7, vmin=_lo, vmax=1)
        _axes[1].set_title("cosine similarity to the chosen patch")
        plt.colorbar(_hm, ax=_axes[1], fraction=0.04)
        for _a in _axes:
            _a.axis("off")
        _fig.tight_layout()

        _W = _flat @ _flat.T
        _order = np.argsort(kmeans(_flat, 4), kind="stable")
        _fig2, _ax2 = plt.subplots(1, 2, figsize=(10, 4.6))
        _wlo = float(np.percentile(_W, 5))
        _ax2[0].imshow(_W, cmap="inferno", vmin=_wlo, vmax=1)
        _ax2[0].set_title(f"W in raster order ({_W.shape[0]}×{_W.shape[1]})")
        _ax2[1].imshow(_W[np.ix_(_order, _order)], cmap="inferno", vmin=_wlo, vmax=1)
        _ax2[1].set_title("W with patches grouped by k-means label")
        for _a in _ax2:
            _a.set_xticks([]); _a.set_yticks([])
        _fig2.tight_layout()

        _out = mo.ui.tabs({
            "one row of W as a heatmap": mo.vstack([_fig, mo.md(
                "*Try:* pick a patch on the main object — the whole object lights up; pick a background patch — "
                "the background lights up.")]),
            "the whole matrix W": mo.vstack([_fig2, mo.md(
                "After grouping patches that belong together, $W$ becomes nearly **block-diagonal**: strong "
                "similarity inside each group, weak between groups.")]),
        })
    _out
    return


@app.cell
def _():
    mo.md(r"""
    ## Summary, and the question for next time

    | Question | Lecture 1 | Lecture 2 |
    |---|---|---|
    | What defines a segment? | A fixed label set chosen by the dataset | A **prompt**: a click, a box, or a phrase |
    | Where does the knowledge come from? | Dense human annotation | **Self-supervision** (DINO) and **data engines** (SAM) |
    | How is the answer computed? | One feed-forward pass | **Prompt-conditioned decoding** on top of a heavy, reusable encoder |

    **The question to take away.** SAM decides "which object" from a prompt, and Mask2Former from labeled
    training data. Suppose we have neither — only the affinity matrix $W$ from the last demo. *How do we cut
    that graph into objects without training anything?* That is where Lecture 3 begins.
    """)
    return


if __name__ == "__main__":
    app.run()
