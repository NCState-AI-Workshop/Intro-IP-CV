# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
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
    from scipy.ndimage import convolve


@app.cell
def _():
    ASSETS = mo.notebook_dir() / "assets" / "lecture1"
    return (ASSETS,)


@app.function
def figure_or_placeholder(path, caption, todo):
    """Show a pre-rendered figure if it exists; otherwise a placeholder that
    says what the figure will show."""
    if path.exists():
        return mo.vstack([mo.image(src=path.read_bytes()), mo.md(f"*{caption}*")])
    return mo.callout(
        mo.md(
            f"**Figure not generated yet** (`{path.name}`).\n\n{todo}\n\n"
            "*Instructor: run `build_lecture1_figures.py` to create it.*"
        ),
        kind="warn",
    )


@app.cell
def _():
    mo.md(r"""
    # Segmentation Lecture 1 (of 3): Dense Prediction
    ## ECE 492-071 / CSC 491-001 Intro to Image Processing and Computer Vision (IPCV)
    ## Dr. Chau-Wai Wong, co-developed with Claude Code

    ---

    This is the first of three lectures on **segmentation**: deciding, for every pixel, what it belongs to.
    The previous Object Detection lectures ended with boxes and with detection transformer (DETR). Today we push classification all the way
    down to individual pixels.

    **The plan for the three lectures.** We will keep coming back to three questions.

    | Question | Lecture 1: dense prediction | Lecture 2: foundation models | Lecture 3: optimization |
    |---|---|---|---|
    | What defines a segment? | A fixed label set chosen by the dataset | *next time* | *later* |
    | Where does the knowledge come from? | Dense human annotation | *next time* | *later* |
    | How is the answer computed? | One feed-forward pass | *next time* | *later* |

    **By the end of today you should be able to:**

    1. distinguish semantic, instance, and panoptic segmentation;
    2. explain why a classification network must be modified to produce a dense output, and how
       transposed convolution, skip connections, and dilated convolution help;
    3. describe how Mask R-CNN and Mask2Former produce one mask per object;
    4. choose a sensible loss and evaluation metric for a segmentation problem.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart LR
        D["Detection lectures<br/>(boxes, DETR)"] --> L1["<b>Seg-Lecture 1</b><br/>Dense prediction"]
        L1 --> L2["Seg-Lecture 2<br/>Foundation models<br/>(DINO, SAM)"]
        L2 --> L3["Seg-Lecture 3<br/>Segmentation as<br/>optimization"]
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.1 From boxes to pixels

    A bounding box tells us roughly *where* an object is, but it always contains some background.
    Segmentation assigns **every pixel**. There are three standard versions of the task. They differ in
    whether they separate individual objects and whether they cover the whole image.

    | Task | Output | Separates objects? | Covers every pixel? |
    |---|---|---|---|
    | **Semantic** segmentation | One class label per pixel | No — two touching people form one "person" region | Yes |
    | **Instance** segmentation | One mask + class per detected object | Yes | No — only countable objects |
    | **Panoptic** (means "all-seeing") segmentation | A class label *and* an object ID per pixel | Yes, for countable objects | Yes |

    **Things vs. stuff.** *Things* are countable objects (person, car, cup). *Stuff* is a region with no
    natural notion of "one instance" (sky, road, grass). Panoptic segmentation covers both.
    """)
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig1_1_three_tasks.png",
        "One image, three outputs: semantic (color = class), instance (color = object; stuff left gray), "
        "panoptic (stuff colored by class, things colored by object).",
        "Shows the input image beside its semantic, instance, and panoptic segmentations, all derived "
        "from one Mask2Former panoptic prediction so the three panels are consistent.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.2 The per-pixel formulation

    A **semantic** segmentation network takes an image and outputs, for every pixel, one score per class.
    A softmax at each pixel $p$ turns the logits/scores into probabilities:

    $$
    s_{p,c} = \frac{e^{\,z_{p,c}}}{\sum_{c'=1}^{C} e^{\,z_{p,c'}}}
    $$

    where

    - $p$ is the current pixel of interest;
    - $z_{p,c}$ is the network's raw logit/score for class $c$ at pixel $p$;
    - $s_{p,c}$ is the resulting probability that pixel $p$ belongs to class $c$; at each pixel these
      sum to 1 over the classes;
    - $C$ is the number of classes, and $c$ (or $c'$) indexes a class.

    Training minimizes the cross-entropy averaged over all pixels:

    $$
    \mathcal{L}_{\text{CE}} = -\frac{1}{|\Omega|} \sum_{p\in\Omega} \log s_{p,\,g_p}
    $$

    where
    - $\Omega$ is the set of all $H \times W$ pixels in the image, $|\Omega| = H \times W$ is the number of pixels;
    - $g_p$ is the ground-truth class of pixel $p$.

    In words: at every pixel, push up the probability of the correct class. It is the classification
    loss you already know, applied $H \times W$ times.

    At test time, each pixel takes its most probable class,

    $$
    \hat{x}_p = \arg\max_{c \in \{1, \dots, C\}} s_{p,c}
    $$

    independently of all other pixels.
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            "**Keep this in mind for Lecture 3.** The loss is a sum over pixels, and the prediction is "
            "chosen at each pixel separately. Neighboring predictions tend to be similar only because the "
            "network looks at overlapping image regions for neighboring pixels. **Nothing** in the loss "
            "**requires neighbors to agree**."
        ),
        kind="info",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.3 Fully convolutional network (FCN) and transposed convolution

    **From classifier to dense predictor.** Long, Shelhamer, and Darrell (2015) observed that the fully connected layers at the end of a classification CNN can be replaced by convolutional layers—most notably 1-by-1 convolutions, which linearly combine features across channels at each spatial location. Instead of flattening features into a single score vector for the whole image, the network outputs a coarse 2D grid of class score vectors, which is then upsampled back to the original input image's dimensions. Because all layers perform convolutions and there are no fully connected layers, this architecture is called a **fully convolutional network (FCN)**.

    ---
    **A tiny example (updated).** Suppose the final feature map has spatial size 2×2 with **2 channels** (say, Channel 1 detects *whiskers* and Channel 2 detects *floppy ears*), and we want class scores for **"cat"** and **"dog."**

    ```text
    Channel 1 (whiskers)      Channel 2 (floppy ears)
        ┌───┬───┐                 ┌───┬───┐
        │ 3 │ 1 │                 │ 0 │ 2 │
        ├───┼───┤                 ├───┼───┤
        │ 0 │ 0 │                 │ 1 │ 4 │
        └───┴───┘                 └───┴───┘
    ```
    *(Note: In the original image, top left has a cat and bottom right has a dog.)*

    **1. Standard CNN (Flattening destroys spatial layout):**
    Flattening stacks all 2 × 2 × 2 = 8 numbers into a single vector. A fully connected layer combines everything at once into **one global score per class**, losing *where* the cat and dog features were located:

    ```text
    flatten (8 values)           global scores
     ──►  cat = 5.2,  dog = 8.1   ("Overall: dog")
    ```

    **2. FCN (1×1 convolution preserves spatial layout):**
    Instead of flattening, we apply two 1×1×2 filters—one for **cat** (weights `[2, -1]` across channels) and one for **dog** (weights `[-1, 2]` across channels)—at every (x, y) position independently:

    * At **top-left** `(3, 0)`: cat = 3(2) + 0(-1) = 6, dog = 3(-1) + 0(2) = -3
    * At **bottom-right** `(0, 4)`: cat = 0(2) + 4(-1) = -4, dog = 0(-1) + 4(2) = 8

    ```text
    Cat score map             Dog score map
      ┌────┬────┐               ┌────┬────┐
      │  6 │  0 │               │ -3 │  3 │
      ├────┼────┤               ├────┼────┤
      │ -1 │ -4 │               │  2 │  8 │
      └────┴────┘               └────┴────┘
    ```
    The output is now a **$2\times2$ grid of class scores** instead of a single pair of numbers: the network sees a **cat in the top-left** and a **dog in the bottom-right**.

    ---

    The catch is **retained** resolution/spatial info. After five stride-2 stages, the score map is $32\times$ smaller than the image
    in each direction; we say the **output stride** is 32. We need a way to go back up.

    **Upsampling options.** Fixed bilinear interpolation (as in image resizing) is one option. A
    *learnable* alternative is the **transposed convolution**. (It is sometimes called "deconvolution,"
    which is misleading — it does not undo a convolution.)

    ### Transposed convolution as a matrix transpose

    **Step 1 — a strided convolution is a matrix.** Work in 1-D. Let the input
    $x = (x_0, \dots, x_5)$ have 6 samples, the kernel $w = (w_0, w_1, w_2)$ have 3 weights, and the
    stride be 2 (the kernel jumps two samples at a time). Each output is a weighted sum of three
    neighboring inputs:

    $$
    y_0 = w_0x_0 + w_1x_1 + w_2x_2, \quad y_1 = w_0x_2 + w_1x_3 + w_2x_4, \quad y_2 = w_0x_4 + w_1x_5
    $$

    (ignoring the missing sample at the right edge). Since every output is a weighted sum of inputs,
    the whole operation is a matrix–vector product $y = Cx$:

    $$
    \boxed{\underbrace{\begin{pmatrix} y_0\\ y_1\\ y_2 \end{pmatrix}}_{y\ (3\times1)} =
    \underbrace{\begin{pmatrix}
    w_0 & w_1 & w_2 & 0 & 0 & 0\\
    0 & 0 & w_0 & w_1 & w_2 & 0\\
    0 & 0 & 0 & 0 & w_0 & w_1
    \end{pmatrix}}_{C\ (3\times6)}
    \underbrace{\begin{pmatrix} x_0\\ x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{pmatrix}}_{x\ (6\times1)}}
    $$

    Each row of $C$ is the kernel, shifted right by 2 columns (the stride). $C$ maps 6 numbers to 3:
    it **downsamples**.

    **Step 2 — transpose it.** $C^\top$ is $6\times3$, so it maps 3 numbers back to 6: it **upsamples**.
    Define $\hat{x} = C^\top y$:

    $$
    \boxed{\begin{pmatrix} \hat{x}_0\\ \hat{x}_1\\ \hat{x}_2\\ \hat{x}_3\\ \hat{x}_4\\ \hat{x}_5 \end{pmatrix} =
    \begin{pmatrix}
    w_0 & 0 & 0\\
    w_1 & 0 & 0\\
    w_2 & w_0 & 0\\
    0 & w_1 & 0\\
    0 & w_2 & w_0\\
    0 & 0 & w_1
    \end{pmatrix}
    \begin{pmatrix} y_0\\ y_1\\ y_2 \end{pmatrix}
    = \begin{pmatrix} w_0 y_0\\ w_1 y_0\\ w_2 y_0 + w_0 y_1\\ w_1 y_1\\ w_2 y_1 + w_0 y_2\\ w_1 y_2 \end{pmatrix}}
    $$

    **Step 3 — read it as "stamping."** Each input value $y_i$ stamps a copy of the kernel, scaled by
    $y_i$, onto the output; consecutive stamps are placed 2 apart (the stride); where stamps overlap,
    they add. In a network the kernel $w$ is learned, so the network learns *how* to upsample.

    **Remark.** $C^\top$ is not the inverse of $C$: in general $C^\top C \neq I$. It reverses only the
    *shape* (3 → 6), not the computation.

    **Checkerboard artifacts.** In Step 2, $\hat{x}_2$ and $\hat{x}_4$ each receive *two* stamps, while
    $\hat{x}_1$ and $\hat{x}_3$ receive only *one*. Even with a perfectly constant input, the output
    alternates between large and small — a periodic pattern that becomes a checkerboard in 2-D. The
    overlap is even when the kernel size is divisible by the stride. A common alternative avoids the
    problem altogether: resize with ordinary interpolation, then apply a normal convolution.
    """)
    return


@app.function
def transposed_conv_matrix(n_in, k, s):
    """C^T for a 1-D transposed convolution with an all-ones kernel:
    output length (n_in - 1) * s + k; input i stamps the kernel at offset i * s."""
    n_out = (n_in - 1) * s + k
    CT = np.zeros((n_out, n_in))
    for i in range(n_in):
        CT[i * s : i * s + k, i] = 1.0
    return CT


@app.function
def transposed_conv_2d(Y, k, s):
    """2-D transposed convolution with an all-ones k x k kernel (stamping)."""
    h, w = Y.shape
    out = np.zeros(((h - 1) * s + k, (w - 1) * s + k))
    for i in range(h):
        for j in range(w):
            out[i * s : i * s + k, j * s : j * s + k] += Y[i, j]
    return out


@app.function
def resize_then_conv_2d(Y, k, s):
    """Nearest-neighbor upsampling by s, then an ordinary k x k averaging convolution."""
    up = np.repeat(np.repeat(Y, s, axis=0), s, axis=1)
    return convolve(up, np.ones((k, k)) / (k * k), mode="nearest")


@app.cell
def _():
    tc_k = mo.ui.slider(start=2, stop=7, value=3, step=1, label="kernel size k", debounce=True)
    tc_s = mo.ui.slider(start=1, stop=4, value=2, step=1, label="stride s", debounce=True)
    mo.vstack(
        [
            mo.md("### Demo: transposed convolution and checkerboards"),
            mo.hstack([tc_k, tc_s], justify="start", gap=2),
        ]
    )
    return tc_k, tc_s


@app.cell
def _(tc_k, tc_s):
    _k, _s = tc_k.value, tc_s.value
    _CT = transposed_conv_matrix(6, _k, _s)
    _x1d = transposed_conv_matrix(12, _k, _s) @ np.ones(12)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 2.4]})
    _axes[0].imshow(_CT, cmap="Blues", vmin=0, vmax=1)
    _axes[0].set_title(f"$C^T$ (input 6 → output {_CT.shape[0]})")
    _axes[0].set_xlabel("input index $i$")
    _axes[0].set_ylabel("output index")
    _axes[0].set_xticks(range(6))
    _n = len(_x1d)
    _axes[1].bar(range(_n), _x1d, color="tab:blue")
    _axes[1].axvspan(-0.5, _k - 1.5, color="0.9", zorder=0)
    _axes[1].axvspan(_n - _k + 0.5, _n - 0.5, color="0.9", zorder=0)
    _axes[1].set_title("Output for a constant input (all ones), all-ones kernel — gray = edges")
    _axes[1].set_xlabel("output index")
    _axes[1].set_ylabel("number of stamps")
    _fig.tight_layout()

    _Yc = np.ones((8, 8))
    _yy, _xx = np.mgrid[0:8, 0:8]
    _Yb = np.exp(-((_yy - 3.5) ** 2 + (_xx - 3.5) ** 2) / (2 * 2.0**2))
    _fig2, _ax2 = plt.subplots(2, 2, figsize=(8, 7.2))
    for _r, (_Y, _name) in enumerate([(_Yc, "constant input"), (_Yb, "smooth bump input")]):
        _tc = transposed_conv_2d(_Y, _k, _s)[_k - 1 : -(_k - 1) or None, _k - 1 : -(_k - 1) or None]
        _rc = resize_then_conv_2d(_Y, _k, _s)
        _tc, _rc = _tc / _tc.mean(), _rc / _rc.mean()  # same brightness scale for both panels
        _vmax = max(_tc.max(), _rc.max())
        _ax2[_r, 0].imshow(_tc, cmap="viridis", vmin=0, vmax=_vmax)
        _ax2[_r, 0].set_title(f"transposed conv — {_name}", fontsize=10)
        _ax2[_r, 1].imshow(_rc, cmap="viridis", vmin=0, vmax=_vmax)
        _ax2[_r, 1].set_title(f"resize, then conv — {_name}", fontsize=10)
    for _a in _ax2.ravel():
        _a.axis("off")
    _fig2.tight_layout()

    _even = (_k % _s == 0)
    mo.vstack(
        [
            _fig,
            _fig2,
            mo.md(
                f"With $k = {_k}$ and $s = {_s}$, the kernel size **is{'' if _even else ' not'}** divisible by "
                f"the stride, so the interior of the output is **{'flat' if _even else 'periodic (checkerboard)'}**. "
                "*What to notice:* the pattern comes from counting overlapping stamps, not from bad training. "
                "\"Resize, then convolve\" never has this problem."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### Skip fusion

    Upsampling a map that is $32\times$ too small in one step gives blobby boundaries. FCN combines
    predictions from earlier layers, which have higher resolution: upsample the coarsest scores by
    $2\times$, add scores computed from a layer at $1/16$ resolution, upsample again, add scores from a
    layer at $1/8$ resolution, then upsample $8\times$ to full size. These variants are called FCN-32s,
    FCN-16s, and FCN-8s.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart LR
        I["Input"] --> B["Backbone"]
        B --> P3["pool3<br/>1/8 resolution"]
        B --> P4["pool4<br/>1/16 resolution"]
        B --> P5["conv7<br/>1/32 resolution"]
        P5 --> S5["1×1 conv<br/>scores"] --> U1["↑2"]
        P4 --> S4["1×1 conv<br/>scores"] --> A1(("+"))
        U1 --> A1 --> U2["↑2"]
        P3 --> S3["1×1 conv<br/>scores"] --> A2(("+"))
        U2 --> A2 --> U3["↑8"] --> O["Output<br/>H×W×C"]
    """
    )
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig1_3b_coarse_vs_full.png",
        "A pretrained FCN: the coarse prediction before upsampling (blocky; this model's coarse map is "
        "1/8 of the input size), the final prediction after bilinear upsampling of the scores, and the "
        "final probability of the main class (note the soft, blurry boundary).",
        "Shows a pretrained FCN's coarse prediction (before the final upsampling) beside its final "
        "full-resolution output and class probability on the `chelsea` (cat) image.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.4 U-Net

    Ronneberger, Fischer, and Brox (2015) made the skip idea symmetric. A **contracting path** (encoder)
    repeatedly convolves and downsamples, doubling the number of channels; an **expanding path**
    (decoder) repeatedly upsamples and convolves, halving the channels. At every scale, the encoder's
    feature map is **concatenated** onto the decoder's feature map before further convolution.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart TB
        subgraph Encoder
        E1["64 ch, H"] --> E2["128 ch, H/2"] --> E3["256 ch, H/4"] --> E4["512 ch, H/8"]
        end
        E4 --> BN["1024 ch, H/16<br/>bottleneck"]
        subgraph Decoder
        BN --> D4["512 ch, H/8"] --> D3["256 ch, H/4"] --> D2["128 ch, H/2"] --> D1["64 ch, H"]
        end
        E4 -.->|"concat"| D4
        E3 -.->|"concat"| D3
        E2 -.->|"concat"| D2
        E1 -.->|"concat"| D1
        D1 --> OUT["1×1 conv → C classes"]
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    **Why it works.** Deep in the network, the decoder knows *what* is in the image (semantics) but has
    lost precisely *where*; the encoder's early layers know *where* (edges, fine positions) but not
    *what*. Concatenation lets the decoder learn how to combine the two. Compare with FCN: FCN *adds*
    class scores at a few scales; U-Net *concatenates* full feature maps at every scale and keeps
    learning after the merge.

    **Where you will meet it.** U-Net was designed for biomedical images with very few training
    examples. Its descendant **nnU-Net**, which configures a U-Net pipeline automatically for a new
    dataset, is still the baseline to beat in medical image segmentation. U-Net is also the standard
    denoising network inside many diffusion-based image generators.

    **Data augmentation for segmentation.** Geometric augmentations (crop, flip, rotate, scale,
    elastic deformation) must be applied *identically* to the image and its mask, and the mask must be
    resampled with nearest-neighbor interpolation so that its labels stay valid (no "class 2.5").
    Photometric augmentations (color jitter, blur) are applied to the image only.
    """)
    return


@app.cell
def _(ASSETS):
    import onnxruntime as ort

    _paths = {
        "skips on": ASSETS / "unet_skip.onnx",
        "skips off": ASSETS / "unet_noskip.onnx",
    }
    _data = ASSETS / "unet_testset.npz"
    if all(_p.exists() for _p in _paths.values()) and _data.exists():
        unet_sessions = {_name: ort.InferenceSession(str(_p)) for _name, _p in _paths.items()}
        _d = np.load(_data)
        unet_images, unet_masks = _d["images"], _d["masks"]
    else:
        unet_sessions, unet_images, unet_masks = None, None, None
    return unet_images, unet_masks, unet_sessions


@app.cell
def _(unet_images):
    _n = 6 if unet_images is None else len(unet_images)
    unet_pick = mo.ui.dropdown(
        options={f"test image {i + 1}": i for i in range(_n)}, value="test image 1", label="image"
    )
    unet_mode = mo.ui.radio(options=["skips on", "skips off", "compare both"], value="compare both", inline=True)
    mo.vstack([mo.md("### Demo: U-Net with and without skip connections"), mo.hstack([unet_pick, unet_mode], justify="start", gap=2)])
    return unet_mode, unet_pick


@app.function
def mask_iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


@app.function
def error_overlay(pred, gt):
    """RGB image: correct foreground white, false positives red, misses blue."""
    rgb = np.zeros(pred.shape + (3,))
    rgb[pred & gt] = (1, 1, 1)
    rgb[pred & ~gt] = (0.9, 0.2, 0.2)
    rgb[~pred & gt] = (0.2, 0.4, 1.0)
    return rgb


@app.cell
def _(unet_images, unet_masks, unet_mode, unet_pick, unet_sessions):
    if unet_sessions is None:
        _out = mo.callout(
            mo.md(
                "**Demo models not found** (`unet_skip.onnx`, `unet_noskip.onnx`, `unet_testset.npz` in "
                "`assets/lecture1/`). *Instructor: run `train_tiny_unet.py`.*\n\n"
                "The demo runs two tiny U-Nets with identical depth and width — one with skip connections, "
                "one without — on synthetic images of blobs and thin bars."
            ),
            kind="warn",
        )
    else:
        _i = unet_pick.value
        _x = unet_images[_i][None, None].astype(np.float32)
        _gt = unet_masks[_i] > 0.5
        _names = ["skips on", "skips off"] if unet_mode.value == "compare both" else [unet_mode.value]
        _fig, _axes = plt.subplots(len(_names), 4, figsize=(12, 3.2 * len(_names)), squeeze=False)
        for _r, _name in enumerate(_names):
            _prob = unet_sessions[_name].run(None, {"input": _x})[0][0, 0]
            _pred = _prob > 0.5
            _axes[_r, 0].imshow(unet_images[_i], cmap="gray", vmin=0, vmax=1)
            _axes[_r, 0].set_title("input")
            _axes[_r, 1].imshow(_gt, cmap="gray")
            _axes[_r, 1].set_title("ground truth")
            _axes[_r, 2].imshow(_prob, cmap="magma", vmin=0, vmax=1)
            _axes[_r, 2].set_title(f"{_name}: probability")
            _axes[_r, 3].imshow(error_overlay(_pred, _gt))
            _axes[_r, 3].set_title(f"{_name}: IoU = {mask_iou(_pred, _gt):.2f}")
            for _a in _axes[_r]:
                _a.axis("off")
        _fig.tight_layout()
        _out = mo.vstack(
            [
                _fig,
                mo.md(
                    "Error map: white = correct foreground, **red** = false positive, **blue** = missed. "
                    "*What to notice:* without skips, the network still finds the blobs, but boundaries become "
                    "rounded and the thin bars are lost — it knows *what* but not precisely *where*."
                ),
            ]
        )
    _out
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.5 DeepLab: dilated convolution and ASPP

    **The tension.** Striding and pooling let each output "see" a large region of the image (context),
    but they shrink the output. Removing them keeps the resolution but makes each output see only a
    small region.

    **Atrous (dilated) convolution** spreads the kernel's weights apart without adding weights. A
    $3\times3$ kernel with rate $r = 2$ places its 9 weights 2 pixels apart, so it covers a $5\times5$
    area with only 9 weights. In general, the area covered is

    $$
    k_{\text{eff}} = k + (k-1)(r-1)
    $$

    where

    - $k$ is the kernel size (number of weights along one side, e.g. 3);
    - $r$ is the dilation rate (spacing between weights; $r = 1$ is an ordinary convolution);
    - $k_{\text{eff}}$ is the side length of the area the kernel covers.

    **Receptive field, by example.** The *receptive field* of an output pixel is the region of the input
    that can affect it. With stride 1, each layer grows the receptive field by $k_{\text{eff}} - 1$.
    Compare four $3\times3$ layers:

    | Rates | $k_{\text{eff}}$ per layer | Receptive field after each layer |
    |---|---|---|
    | 1, 1, 1, 1 (ordinary) | 3, 3, 3, 3 | 3, 5, 7, 9 |
    | 1, 2, 4, 8 (dilated) | 3, 5, 9, 17 | 3, 7, 15, 31 |

    Same number of weights, same output resolution, but 31 pixels of context instead of 9.

    **Gridding.** If every layer uses the *same* large rate (say 2, 2, 2), the stack only ever looks at
    every other pixel; the pixels in between are never used. Mixing rates fixes this.

    **ASPP (atrous spatial pyramid pooling).** Run several dilated convolutions with different rates in
    parallel on the same feature map (DeepLabv3 uses rates 6, 12, and 18, plus a $1\times1$ branch and
    an image-level average branch) and concatenate their outputs. Small rates capture small objects;
    large rates capture large ones — like looking at the image at several zoom levels at once.

    *Coming back later:* DeepLab v1 and v2 cleaned up their outputs with a "CRF." You will see exactly
    what that is in Lecture 3.
    """)
    return


@app.cell
def _():
    rf_layers = mo.ui.slider(start=1, stop=5, value=4, step=1, label="number of layers", debounce=True)
    rf_rates = mo.ui.array(
        [mo.ui.dropdown(options=["1", "2", "4", "8"], value=_v, label=f"rate, layer {_i + 1}")
         for _i, _v in enumerate(["1", "2", "4", "8", "1"])]
    )
    mo.vstack([mo.md("### Demo: receptive field of stacked dilated convolutions"), rf_layers, mo.hstack(list(rf_rates), justify="start", gap=1)])
    return rf_layers, rf_rates


@app.function
def dilated_ones_kernel(r):
    """3x3 all-ones kernel with dilation r, as a (2r+1) x (2r+1) array."""
    kern = np.zeros((2 * r + 1, 2 * r + 1))
    kern[::r, ::r] = 1.0
    return kern


@app.cell
def _(rf_layers, rf_rates):
    _rates = [int(_v) for _v in rf_rates.value[: rf_layers.value]]
    _N = 91
    _paths = np.zeros((_N, _N))
    _paths[_N // 2, _N // 2] = 1.0
    for _r in _rates:
        _paths = convolve(_paths, dilated_ones_kernel(_r), mode="constant")
    _rf = 1 + sum(2 * _r for _r in _rates)
    _c, _h = _N // 2, (_rf - 1) // 2
    _box = _paths[_c - _h : _c + _h + 1, _c - _h : _c + _h + 1]
    _coverage = (_box > 0).mean()

    _fig, _ax = plt.subplots(figsize=(5.2, 5.2))
    _ax.imshow(np.log1p(_paths), cmap="magma")
    _ax.add_patch(Rectangle((_c - _h - 0.5, _c - _h - 0.5), _rf, _rf, fill=False, ec="cyan", lw=1.2))
    _ax.set_title(f"rates {_rates}: receptive field {_rf}×{_rf}")
    _ax.axis("off")
    _rows = "\n".join(
        f"| {_i + 1} | {_r} | {3 + 2 * (_r - 1)} | {1 + sum(2 * _q for _q in _rates[: _i + 1])} |"
        for _i, _r in enumerate(_rates)
    )
    mo.hstack(
        [
            _fig,
            mo.md(
                "| Layer | Rate $r$ | $k_{\\text{eff}}$ | Receptive field |\n|---|---|---|---|\n" + _rows +
                f"\n\nPixels inside the receptive field that are actually used: **{100 * _coverage:.0f}%**.\n\n"
                "Brightness = number of paths from that input pixel to the center output (log scale); the cyan "
                "square is the receptive field.\n\n*Try:* rates (1, 2, 4, 8) for fast growth with full coverage; "
                "rates (2, 2, 2) for gridding (holes); then mix rates to fill them."
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.6 Mask R-CNN's mask head and RoI Align

    **Recap.** Faster R-CNN = backbone + region proposal network + a head that classifies each proposed
    box (RoI, "region of interest") and refines its coordinates.

    **Mask R-CNN** (He, Gkioxari, Dollár, and Girshick, 2017) adds a third branch to that head: a small
    FCN applied to each RoI's feature map (e.g., $14\times14$) that outputs a small mask (e.g.,
    $28\times28$), which is then resized and pasted into the box. Two design choices matter:

    1. **One mask per class, no competition.** The branch outputs one mask for each class, each passed
       through a sigmoid, so every mask pixel is an independent foreground probability. During training,
       only the mask of the box's *true* class receives a loss (per-pixel binary cross-entropy).
       Deciding *which* class the object is stays the job of the box branch. This works better than
       making the classes compete for each pixel with a softmax.
    2. **RoI Align** instead of RoI Pooling.

    The total training loss is the sum of the three branches' losses: classification + box regression + mask.

    ### RoI Align, precisely

    **RoI Pooling rounds twice.** First, the box corners are converted from image pixels to
    feature-map cells (divide by the stride, e.g. 16) and rounded to whole cells. Second, the box is split
    into an output grid of bins (e.g. $7\times7$) whose edges are again rounded to whole cells, and each
    bin is max-pooled. At stride 16, being off by half a cell means being off by 8 image pixels — fine
    for deciding "is this a cat," bad for drawing the cat's outline.

    **RoI Align never rounds.** The box keeps its exact (fractional) coordinates on the feature map, the
    bins keep their exact edges, and each bin is represented by a few evenly spaced sample points (e.g.
    $2\times2$). A sample point usually falls *between* feature-map cells, so its value is estimated by
    **bilinear interpolation**: a weighted average of the four surrounding cells, with closer cells
    weighted more — the same interpolation used when resizing an image. The samples in each bin are then
    averaged. As a result, the output changes smoothly as the box moves.
    """)
    return


@app.function
def roi_feature_map():
    yy, xx = np.mgrid[0:16, 0:16].astype(float)
    return np.exp(-((xx - 7.5) ** 2 + (yy - 7.0) ** 2) / (2 * 2.2**2)) + 0.6 * (xx >= 10)


@app.function
def roi_pool(F, box, P):
    """RoI Pooling as in Fast R-CNN: round the box to whole cells, then bins with rounded edges, max."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    rw, rh = max(x2 - x1 + 1, 1), max(y2 - y1 + 1, 1)
    out = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            ys, ye = y1 + int(np.floor(i * rh / P)), y1 + int(np.ceil((i + 1) * rh / P))
            xs, xe = x1 + int(np.floor(j * rw / P)), x1 + int(np.ceil((j + 1) * rw / P))
            out[i, j] = F[max(ys, 0):min(ye, F.shape[0]), max(xs, 0):min(xe, F.shape[1])].max()
    return out, (x1 - 0.5, y1 - 0.5, rw, rh)


@app.function
def bilinear(F, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    ax, ay = x - x0, y - y0
    x1, y1 = min(x0 + 1, F.shape[1] - 1), min(y0 + 1, F.shape[0] - 1)
    return ((1 - ax) * (1 - ay) * F[y0, x0] + ax * (1 - ay) * F[y0, x1]
            + (1 - ax) * ay * F[y1, x0] + ax * ay * F[y1, x1])


@app.function
def roi_align(F, box, P, samples=2):
    """RoI Align: exact bin edges, samples x samples points per bin, bilinear interpolation, average."""
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) / P, (y2 - y1) / P
    out = np.zeros((P, P))
    pts = []
    for i in range(P):
        for j in range(P):
            vals = []
            for a in range(samples):
                for b in range(samples):
                    y = y1 + i * bh + (a + 0.5) * bh / samples
                    x = x1 + j * bw + (b + 0.5) * bw / samples
                    vals.append(bilinear(F, x, y))
                    pts.append((x, y))
            out[i, j] = np.mean(vals)
    return out, np.array(pts)


@app.cell
def _():
    roi_shift = mo.ui.slider(start=0.0, stop=5.0, value=0.0, step=0.1, label="move the box right (feature cells)", debounce=False)
    mo.vstack([mo.md("### Demo: RoI Pooling vs. RoI Align"), roi_shift])
    return (roi_shift,)


@app.cell
def _(roi_shift):
    _F = roi_feature_map()
    _P, _W = 4, 6.6

    def _box(_d):
        return (1.8 + _d, 3.2, 1.8 + _d + _W, 3.2 + _W)

    _pool, _qbox = roi_pool(_F, _box(roi_shift.value), _P)
    _align, _pts = roi_align(_F, _box(roi_shift.value), _P)
    _ds = np.round(np.arange(0, 5.01, 0.05), 2)
    _cp = [roi_pool(_F, _box(_d), _P)[0][1, 1] for _d in _ds]
    _ca = [roi_align(_F, _box(_d), _P)[0][1, 1] for _d in _ds]

    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.6]})
    _axes[0].imshow(_F, cmap="viridis")
    _b = _box(roi_shift.value)
    _axes[0].add_patch(Rectangle((_b[0], _b[1]), _W, _W, fill=False, ec="white", lw=2, label="true box"))
    _axes[0].add_patch(Rectangle(_qbox[:2], _qbox[2], _qbox[3], fill=False, ec="red", lw=2, ls="--", label="RoI Pool (rounded)"))
    _axes[0].plot(_pts[:, 0], _pts[:, 1], ".", color="white", ms=3)
    _axes[0].set_title("feature map, box, sample points")
    _axes[0].legend(loc="lower left", fontsize=7)
    _vmax = max(_F.max(), 1e-6)
    _axes[1].imshow(_pool, cmap="viridis", vmin=0, vmax=_vmax)
    _axes[1].set_title("RoI Pool output (4×4)")
    _axes[2].imshow(_align, cmap="viridis", vmin=0, vmax=_vmax)
    _axes[2].set_title("RoI Align output (4×4)")
    for _a in _axes[1:3]:
        _a.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec="orange", lw=2))
        _a.set_xticks([]); _a.set_yticks([])
    _axes[3].plot(_ds, _cp, color="red", label="RoI Pool")
    _axes[3].plot(_ds, _ca, color="tab:green", label="RoI Align")
    _axes[3].axvline(roi_shift.value, color="0.4", ls=":")
    _axes[3].set_xlabel("box shift (feature cells)")
    _axes[3].set_title("value of the orange output cell")
    _axes[3].legend(fontsize=8)
    _axes[0].set_xticks([]); _axes[0].set_yticks([])
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md("Real networks use a $7\\times7$ output; $4\\times4$ is used here so the bins are visible. "
              "*What to notice:* RoI Pool's curve is a staircase — moving the box a fraction of a cell changes "
              "nothing, then the output jumps. RoI Align's curve is smooth."),
    ])
    return


@app.cell
def _(ASSETS):
    figure_or_placeholder(
        ASSETS / "fig1_6_maskrcnn.png",
        "Mask R-CNN instance masks, with a zoom on one detection: its 28×28 mask before and after it is "
        "resized and pasted into the box.",
        "Shows Mask R-CNN's instance masks on a test image, and for one detected object the small "
        "28×28 mask predicted by the mask branch next to the same mask pasted back into the image.",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.7 Mask2Former: classifying masks instead of pixels

    **A change of viewpoint.** Everything so far *classifies pixels*. MaskFormer (2021) and Mask2Former
    (2022) instead *classify masks*. The network outputs a fixed number $N$ (e.g., 100) of predictions,
    and each prediction is a pair: a class probability vector (including an extra "no object" class) and
    a soft mask over the whole image. This is DETR with masks in place of boxes.

    **Architecture in three parts.** A backbone extracts features. A **pixel decoder** upsamples them so
    that every pixel gets an embedding vector. A **transformer decoder** refines $N$ learned query vectors
    by cross-attending to the image features, exactly as in DETR. Each refined query then produces
    (a) a class prediction through a linear layer and (b) a mask, by comparing the query with every
    pixel's embedding:

    $$
    m_i(p) = \sigma\!\left(q_i^\top F(p)\right)
    $$

    where

    - $i = 1, \dots, N$ indexes the queries;
    - $q_i$ is query $i$'s embedding vector after the transformer decoder;
    - $F(p)$ is the embedding vector of pixel $p$ from the pixel decoder (same length as $q_i$, e.g. 256);
    - $q_i^\top F(p)$ is their dot product — large when the pixel "matches" the query;
    - $\sigma$ is the sigmoid function, which squashes the dot product into $(0, 1)$;
    - $m_i(p)$ is the probability that pixel $p$ belongs to query $i$'s mask.

    In words: each query is a template, and its mask lights up wherever the pixel embeddings look like
    the template. Mask2Former's main addition over MaskFormer is **masked attention**: each query attends
    only to the pixels inside its own mask from the previous decoder layer, which keeps each query
    focused on one object.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
    flowchart LR
        I["Image"] --> BB["Backbone"] --> PD["Pixel decoder"]
        PD --> F["Per-pixel embeddings F(p)"]
        PD --> MS["Multi-scale features"]
        Q["N learned queries"] --> TD["Transformer decoder<br/>(masked cross-attention)"]
        MS --> TD
        TD --> QE["Refined queries q_i"]
        QE --> CL["Linear layer → class"]
        QE --> DOT(("dot product"))
        F --> DOT --> M["Mask m_i"]
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    **Training** uses DETR's recipe: Hungarian matching between the $N$ predictions and the
    ground-truth segments, where the matching cost combines class correctness and mask overlap;
    unmatched predictions are trained to say "no object."

    **One model, three tasks.** At test time, the same set of (class, mask) pairs can be read three ways:

    - *Semantic:* at each pixel, add up every query's mask value weighted by how strongly that query
      predicts class $c$; pick the class with the largest total.
    - *Instance:* keep the queries that confidently predict a real class; each kept query is one object.
    - *Panoptic:* keep confident queries and give each pixel to the query that claims it most strongly.

    Per-pixel classification cannot produce instances without extra machinery; mask classification
    handles all three tasks with one architecture.
    """)
    return


@app.cell
def _(ASSETS):
    _qdir = ASSETS / "fig1_7_queries"
    _files = sorted(_qdir.glob("*.png")) if _qdir.exists() else []
    if _files:
        query_pick = mo.ui.dropdown(options={_f.stem: _f for _f in _files}, value=_files[0].stem, label="query")
    else:
        query_pick = None
    query_pick
    return (query_pick,)


@app.cell
def _(ASSETS, query_pick):
    if query_pick is not None:
        _out = mo.vstack([
            mo.image(src=query_pick.value.read_bytes()),
            mo.md("*One query's mask, with its predicted class and confidence. Each confident query has become "
                  "an \"object slot,\" as in DETR.*"),
        ])
    else:
        _out = figure_or_placeholder(
            ASSETS / "fig1_7_query_grid.png",
            "The 12 most confident Mask2Former queries for one image, each shown as its mask with its "
            "predicted class and confidence.",
            "Shows, for one image, the masks of the most confident Mask2Former queries — one panel per "
            "query, with the class and confidence in the title.",
        )
    _out
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.8 Losses and metrics

    **Overlap measures first.** For one class, let $P$ be the set of pixels the model predicts as that
    class, and $G$ the set of pixels that truly are that class. Two standard overlap scores, both between
    0 (no overlap) and 1 (perfect):

    $$
    \text{IoU} = \frac{|P \cap G|}{|P \cup G|}, \qquad \text{Dice} = \frac{2\,|P \cap G|}{|P| + |G|}
    $$

    where $|\cdot|$ counts pixels, $P \cap G$ is the set of correctly predicted pixels, and $P \cup G$ is
    every pixel that is in either set. IoU is the same quantity you used for boxes in the detection
    lectures, now computed on pixel sets. Dice and IoU always rank predictions in the same order; Dice
    is a little more forgiving numerically.

    **An example.** A $256 \times 256$ image (65,536 pixels) contains a 100-pixel object. The prediction
    finds 50 of its pixels and marks nothing else. Pixel accuracy is $1 - 50/65{,}536 \approx 99.92\%$.
    Dice is $2 \cdot 50 / (50 + 100) \approx 0.67$; IoU is $50/100 = 0.5$. Accuracy says "almost perfect";
    Dice and IoU say "you missed half of what matters."

    **Losses.** Per-pixel cross-entropy (§1.2) averages over all pixels, so large regions dominate and
    small objects barely count — the same problem as accuracy in the example. Two common fixes:

    - **Dice loss** $= 1 - \text{Dice}$, computed with the predicted *probabilities* in place of hard
      0/1 predictions so that it can be trained with gradients. Because Dice divides by the object's
      size, missing half of a small object costs as much as missing half of a large one.
    - **Focal loss** is cross-entropy with each pixel's term multiplied by a factor that shrinks toward 0
      when that pixel is already classified confidently, so training focuses on the hard pixels.

    In practice, cross-entropy + Dice is a robust default.
    """)
    return


@app.cell
def _():
    met_r = mo.ui.slider(start=2, stop=60, value=6, step=1, label="object radius (pixels)", debounce=True)
    met_f = mo.ui.slider(start=0.0, stop=1.0, value=0.5, step=0.05, label="fraction of the object missed", debounce=True)
    mo.vstack([mo.md("### Demo: accuracy vs. cross-entropy vs. Dice for small objects"), mo.hstack([met_r, met_f], justify="start", gap=2)])
    return met_f, met_r


@app.function
def circle_scene(radius, frac_missed, size=256):
    """Ground-truth circle, and a prediction that misses a pie slice of the given fraction.
    Returns (gt, pred, soft foreground probability)."""
    yy, xx = np.mgrid[0:size, 0:size] - size / 2 + 0.5
    gt = xx**2 + yy**2 < radius**2
    ang = (np.arctan2(yy, xx) + np.pi) / (2 * np.pi)  # in [0, 1)
    pred = gt & (ang >= frac_missed)
    prob = np.where(pred, 0.95, 0.05)
    return gt, pred, prob


@app.function
def seg_metrics(gt, pred, prob):
    acc = (gt == pred).mean()
    ce = -np.mean(np.where(gt, np.log(prob), np.log(1 - prob)))
    inter = (gt & pred).sum()
    dice = 2 * inter / max(gt.sum() + pred.sum(), 1)
    iou = inter / max((gt | pred).sum(), 1)
    return acc, ce, dice, iou


@app.cell
def _(met_f, met_r):
    _gt, _pred, _prob = circle_scene(met_r.value, met_f.value)
    _acc, _ce, _dice, _iou = seg_metrics(_gt, _pred, _prob)
    _rs = np.arange(2, 61)
    _curves = np.array([seg_metrics(*circle_scene(_r, met_f.value)) for _r in _rs])

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    _c = 128
    _z = max(int(met_r.value * 1.6), 12)
    _axes[0].imshow(error_overlay(_pred, _gt)[_c - _z : _c + _z, _c - _z : _c + _z])
    _axes[0].set_title(f"zoom on the object (image is 256×256)")
    _axes[0].axis("off")
    _axes[1].plot(_rs, _curves[:, 0], label="pixel accuracy")
    _axes[1].plot(_rs, _curves[:, 2], label="Dice")
    _axes[1].plot(_rs, _curves[:, 3], label="IoU")
    _axes[1].plot(_rs, _curves[:, 1], label="mean cross-entropy", ls="--")
    _axes[1].axvline(met_r.value, color="0.4", ls=":")
    _axes[1].set_xlabel("object radius (pixels)")
    _axes[1].set_title(f"same relative error ({100 * met_f.value:.0f}% missed) at every size")
    _axes[1].set_ylim(-0.02, 1.02)
    _axes[1].legend(fontsize=8, loc="center right")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(
            f"| Pixel accuracy | Mean cross-entropy | Dice | IoU |\n|---|---|---|---|\n"
            f"| {100 * _acc:.2f}% | {_ce:.4f} | {_dice:.3f} | {_iou:.3f} |\n\n"
            "White = found, blue = missed. *What to notice:* for a small object, accuracy and mean "
            "cross-entropy hardly change no matter how much of it we miss; Dice and IoU depend only on the "
            "*fraction* missed, whatever the object's size."
        ),
    ])
    return


@app.cell
def _():
    mo.md(r"""
    ### Metrics for each task

    - **mIoU** (semantic segmentation): compute IoU for each class over the whole test set, then average
      over classes. Averaging over classes means rare classes count as much as common ones.
    - **Mask AP** (instance segmentation): the same AP as for boxes in the detection lectures, except
      that "does this prediction match that object?" is decided with mask IoU instead of box IoU.
    - **Panoptic Quality** (panoptic segmentation). First match predicted segments to true segments: a
      pair matches if it has the same class and IoU $> 0.5$. Then

    $$
    \text{PQ} = \text{SQ} \times \text{RQ}, \qquad
    \text{SQ} = \frac{\text{sum of IoUs of matched pairs}}{TP}, \qquad
    \text{RQ} = \frac{TP}{TP + \tfrac12 FP + \tfrac12 FN}
    $$

    where

    - $TP$ is the number of matched pairs (segments, not pixels);
    - $FP$ is the number of predicted segments left unmatched (false alarms);
    - $FN$ is the number of true segments left unmatched (misses);
    - SQ ("segmentation quality") is how well the matched segments overlap;
    - RQ ("recognition quality") is how well we found the right segments at all.

    **Common benchmarks.** PASCAL VOC 2012 (20 object classes + background), Cityscapes (urban driving,
    19 evaluation classes), ADE20K (150 classes, indoor and outdoor scenes), COCO (80 object classes for
    instance segmentation; 133 classes including "stuff" for panoptic).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1.9 Summary, and the problem for next time

    | Question | Lecture 1 answer |
    |---|---|
    | What defines a segment? | A fixed label set chosen by whoever built the dataset |
    | Where does the knowledge come from? | Dense, pixel-accurate human annotation |
    | How is the answer computed? | One feed-forward pass of a network |

    **The catch.** Every model today learned from dense masks, and dense masks are expensive: the
    Cityscapes authors report that fine annotation and quality control took more than 1.5 hours per
    image on average.

    **Question for next time:** what if we have *no* masks for our problem at all? Lecture 2 looks at
    models that learn without labels (DINO) and models trained once on an enormous mask collection so
    that you do not need your own (SAM).
    """)
    return


if __name__ == "__main__":
    app.run()
