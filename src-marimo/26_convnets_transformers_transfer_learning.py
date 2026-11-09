# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "pyarrow",
#     "scikit-learn",
#     "scipy",
#     "torch",
#     "torchvision",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
# torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    import tempfile
    import time
    import urllib.request

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    from scipy.ndimage import convolve, correlate
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return (
        F,
        Image,
        LogisticRegression,
        convolve,
        correlate,
        io,
        json,
        load_digits,
        mo,
        nn,
        np,
        os,
        plt,
        tempfile,
        time,
        torch,
        torchvision,
        urllib,
    )


@app.cell
def _(mo):
    IMAGES_DIR = mo.notebook_dir() / "images"
    return (IMAGES_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lecture 22: Convolutional Networks, Transformers, and Transfer Learning

    Lecture 21 ended with a problem. An MLP treats an image as an unordered list
    of numbers: shuffle the pixels and it learns just as well, because it never
    used the fact that neighboring pixels belong together. On a
    $224\times224$ color image, one fully connected layer of 1,000 units needs
    about 150 million weights, and whatever it learns at one position it must
    relearn at every other.

    This notebook covers three answers to that problem, following Szeliski
    §5.4–5.5:

    1. **Convolutional networks** build the structure in. A convolution layer
       is the linear filtering of notebook 08 with *learned* kernels: each
       kernel is small (local) and is reused at every position (shared).
    2. **Transformers** take the structure back out. Self-attention lets every
       image patch look at every other patch, and the network learns which
       relationships matter from large amounts of data.
    3. **Transfer learning** avoids paying that data cost again. A network
       trained on 1.28 million ImageNet photos has already learned edges,
       textures and object parts; we reuse it for a new task with only a few
       labels.

    Every section runs on a laptop CPU. Section 5 downloads about 25 MB of
    data and takes about two minutes the first time, so it waits for a button
    click.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. A convolution layer is a learned filter bank

    A convolution layer with $C_\text{in}$ input channels, $C_\text{out}$
    output channels, and $K\times K$ kernels computes, at every output
    position,

    $$y_{c}(i,j) = b_c + \sum_{c'=1}^{C_\text{in}} \sum_{u,v} w_{c,c'}(u,v)\;x_{c'}(i\cdot S + u\cdot D,\; j\cdot S + v\cdot D),$$

    with stride $S$ and dilation $D$. Two properties make it cheap and suited
    to images:

    - **Local connectivity.** Each output looks only at a $K\times K$
      neighborhood, the same assumption behind every filter in notebooks 08–11.
    - **Weight sharing.** The same kernel slides over the whole image, so the
      layer has $C_\text{out}(C_\text{in}K^2 + 1)$ parameters *regardless of
      image size*, and a pattern learned at one position is detected at all of
      them.

    The output size follows from how many times the (dilated) kernel fits:

    $$H_\text{out} = \left\lfloor \frac{H_\text{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1.$$

    Change the knobs below; the last row checks the formula against what
    `torch.nn.Conv2d` actually produces.
    """)
    return


@app.cell
def _(mo):
    conv_h = mo.ui.slider(8, 256, value=224, step=8, label="Input size H")
    conv_cin = mo.ui.slider(1, 64, value=3, step=1, label="C_in")
    conv_cout = mo.ui.slider(1, 256, value=64, step=1, label="C_out")
    conv_k = mo.ui.slider(1, 11, value=3, step=2, label="Kernel K")
    conv_p = mo.ui.slider(0, 5, value=1, step=1, label="Padding P")
    conv_s = mo.ui.slider(1, 4, value=1, step=1, label="Stride S")
    conv_d = mo.ui.slider(1, 4, value=1, step=1, label="Dilation D")
    mo.vstack([
        mo.hstack([conv_h, conv_cin, conv_cout], justify="start"),
        mo.hstack([conv_k, conv_p, conv_s, conv_d], justify="start"),
    ])
    return conv_cin, conv_cout, conv_d, conv_h, conv_k, conv_p, conv_s


@app.cell
def _(conv_cin, conv_cout, conv_d, conv_h, conv_k, conv_p, conv_s, mo, nn, torch):
    _H, _K, _P, _S, _D = conv_h.value, conv_k.value, conv_p.value, conv_s.value, conv_d.value
    _cin, _cout = conv_cin.value, conv_cout.value
    _hout = (_H + 2 * _P - _D * (_K - 1) - 1) // _S + 1
    if _hout < 1:
        _out = mo.md("**The kernel does not fit:** increase padding or input size.")
    else:
        _layer = nn.Conv2d(_cin, _cout, _K, stride=_S, padding=_P, dilation=_D)
        with torch.no_grad():
            _shape = tuple(_layer(torch.zeros(1, _cin, _H, _H)).shape)
        _conv_params = sum(_p.numel() for _p in _layer.parameters())
        _fc_params = (_cin * _H * _H) * (_cout * _hout * _hout) + _cout * _hout * _hout
        _out = mo.md(f"""
        | | value |
        |---|---|
        | Output size from the formula | **{_cout} × {_hout} × {_hout}** |
        | Output size from `nn.Conv2d` | {_shape[1]} × {_shape[2]} × {_shape[3]} |
        | Receptive field of one output | {_D * (_K - 1) + 1} × {_D * (_K - 1) + 1} pixels |
        | Parameters, convolution layer | **{_conv_params:,}** |
        | Parameters, fully connected layer with the same input and output sizes | **{_fc_params:,}** ({_fc_params / _conv_params:,.0f}× more) |
        """)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Deep learning "convolution" is correlation

    One detail trips up anyone coming from signal processing.
    `torch.nn.Conv2d` computes **cross-correlation**: it does *not* flip the
    kernel. True convolution, as in notebook 08, flips it first. For a
    symmetric kernel such as a Gaussian the two agree; for an antisymmetric
    one such as Sobel they differ by a sign. In a network the kernels are
    learned, so the flip does not matter: the network simply learns the
    flipped weights. It does matter when you load a hand-designed filter into
    a layer.
    """)
    return


@app.cell
def _(F, IMAGES_DIR, convolve, correlate, mo, np, plt, torch):
    _img = plt.imread(IMAGES_DIR / "coins.png").astype(np.float32)
    _img = _img[..., :3].mean(-1) if _img.ndim == 3 else _img
    _sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    with torch.no_grad():
        _torch_out = F.conv2d(torch.from_numpy(_img)[None, None], torch.from_numpy(_sobel_x)[None, None],
                              padding=1)[0, 0].numpy()
    _corr = correlate(_img, _sobel_x, mode="constant")
    _conv = convolve(_img, _sobel_x, mode="constant")
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for _ax, (_title, _im) in zip(_axes, [("image", _img), ("torch conv2d with Sobel-x", _torch_out),
                                          ("scipy convolve (flipped kernel)", _conv)]):
        _ax.imshow(_im, cmap="gray" if _title == "image" else "RdBu_r")
        _ax.set_title(_title, fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"*Max difference between `torch.conv2d` and scipy **correlate**: "
              f"{np.abs(_torch_out - _corr).max():.1e}. Between `torch.conv2d` and the negated scipy "
              f"**convolve**: {np.abs(_torch_out + _conv).max():.1e}. The edges flip from red to blue: "
              "same filter, opposite sign.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. What weight sharing buys: the shift test

    Weight sharing makes a convolution **translation equivariant**: shift the
    input and the feature map shifts by the same amount. A **global pooling**
    layer at the end (max or average over all positions) then discards *where*
    a feature fired, so the prediction becomes **translation invariant**. An
    MLP has neither property.

    The test: place each 8x8 digit at the center of a 16x16 canvas, train on
    centered digits only, then test with the digits shifted left or right by
    up to 4 pixels. Three models:

    - **MLP**: 256 inputs → 128 hidden units → 10 classes;
    - **CNN with pooling**: conv, *2x2 max-pool with stride 2*, conv, global max-pool;
    - **CNN without pooling**: the same two convolutions at stride 1, then global max-pool.

    All three are trained the same way (Adam, 15 epochs); about ten seconds in total.
    """)
    return


@app.cell
def _(F, torch):
    def train_classifier(model, X, y, epochs, batch_size=64, lr=3e-3, seed=0):
        """Mini-batch Adam on cross-entropy; deterministic given the seed."""
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ep in range(epochs):
            _idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed * 1000 + _ep))
            for _i in range(0, len(y), batch_size):
                _b = _idx[_i:_i + batch_size]
                _loss = F.cross_entropy(model(X[_b]), y[_b])
                opt.zero_grad()
                _loss.backward()
                opt.step()
        model.eval()
        return model

    def accuracy(model, X, y):
        with torch.no_grad():
            return (model(X).argmax(1) == y).float().mean().item()

    return accuracy, train_classifier


@app.cell
def _(load_digits, np, torch):
    _d = load_digits()
    digits8 = torch.tensor(_d.data / 16.0, dtype=torch.float32).view(-1, 1, 8, 8)
    digits_labels = torch.tensor(_d.target)
    _order = np.random.default_rng(0).permutation(len(digits_labels))
    _n_train = int(0.8 * len(_order))
    digits_train, digits_test = _order[:_n_train], _order[_n_train:]

    def place_on_canvas(X, dx, canvas=16):
        """Put 8x8 digits on a blank canvas, centered, then shifted dx pixels horizontally."""
        out = torch.zeros(len(X), 1, canvas, canvas)
        o = (canvas - 8) // 2
        out[:, :, o:o + 8, o + dx:o + dx + 8] = X
        return out

    return digits8, digits_labels, digits_test, digits_train, place_on_canvas


@app.cell
def _(
    accuracy,
    digits8,
    digits_labels,
    digits_test,
    digits_train,
    nn,
    place_on_canvas,
    torch,
    train_classifier,
):
    _models = {
        "MLP": nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)),
        "CNN with stride-2 pooling": nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 10)),
        "CNN without pooling": nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 10)),
    }
    shift_values = list(range(-4, 5))
    shift_results = {}
    _ytr, _yte = digits_labels[digits_train], digits_labels[digits_test]
    _Xtr = place_on_canvas(digits8[digits_train], 0)
    for _name, _model in _models.items():
        torch.manual_seed(0)
        _model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        train_classifier(_model, _Xtr, _ytr, epochs=15)
        shift_results[_name] = {
            "params": sum(_p.numel() for _p in _model.parameters()),
            "acc": [accuracy(_model, place_on_canvas(digits8[digits_test], _s), _yte) for _s in shift_values],
        }
    return shift_results, shift_values


@app.cell
def _(digits8, digits_test, mo, place_on_canvas, plt, shift_results, shift_values):
    _fig, _axd = plt.subplot_mosaic([["curve"] * 5, ["s0", "s1", "s2", "s3", "s4"]], figsize=(8, 5.2),
                                    gridspec_kw={"height_ratios": [2.6, 1]})
    _a1 = _axd["curve"]
    for _name, _r in shift_results.items():
        _a1.plot(shift_values, _r["acc"], "o-", label=f"{_name} ({_r['params']:,} params)")
    _a1.set(xlabel="horizontal shift at test time (pixels)", ylabel="test accuracy", ylim=(0, 1.02),
            title="Trained on centered digits only")
    _a1.grid(alpha=0.3)
    _a1.legend(fontsize=8, loc="lower center")
    for _k, _s in enumerate([-4, -2, 0, 2, 4]):
        _ax = _axd[f"s{_k}"]
        _ax.imshow(place_on_canvas(digits8[digits_test[:1]], _s)[0, 0], cmap="gray_r")
        _ax.set_title(f"shift {_s:+d}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    _mlp, _pool, _nopool = (shift_results[k]["acc"] for k in shift_results)
    mo.vstack([
        _fig,
        mo.md(f"""
        - The **MLP** is near perfect on centered digits ({_mlp[4]:.0%}) and falls to
          {min(_mlp):.0%}–{max(_mlp[:4] + _mlp[5:]):.0%} once the digit moves. A shifted digit
          lands on input weights it was never trained on.
        - The **CNN without pooling** is unchanged at every shift ({min(_nopool):.0%}–{max(_nopool):.0%}),
          with {shift_results['CNN without pooling']['params']:,} parameters against the MLP's
          {shift_results['MLP']['params']:,}. Stride-1 convolutions are exactly equivariant, and
          the global max-pool removes position.
        - The **CNN with stride-2 pooling** is good at *even* shifts ({min(_pool[0::2]):.0%}–{max(_pool[0::2]):.0%})
          and worse at *odd* ones ({min(_pool[1::2]):.0%}–{max(_pool[1::2]):.0%}). Subsampling by 2 is
          only equivariant to shifts that are multiples of 2: this is **aliasing**, the same
          sampling effect as in notebooks 07 and 11. Classic ConvNets subsample this way and lose some of
          their shift invariance. Zhang (ICML 2019, "Making Convolutional Networks Shift-Invariant
          Again") restores it by low-pass filtering before subsampling, the anti-aliasing step
          of an image pyramid.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. A real ConvNet: ResNet-18 pretrained on ImageNet

    Stack many such layers, subsample every few layers while doubling the
    channels, and add **residual connections** ($\mathbf{y} = \mathbf{x} +
    F(\mathbf{x})$, so a block only has to learn a correction to its input) to
    keep a deep network trainable. That is **ResNet** (He et al., CVPR 2016).
    ResNet-18 has 18 weight layers and 11.7 million parameters, and was trained
    on ImageNet-1k: 1.28 million photos in 1,000 classes.

    `torchvision` downloads the trained weights (about 45 MB, once). Pick an
    image from the course's test set; the network has never seen these exact
    photos.
    """)
    return


@app.cell
def _(torchvision):
    resnet_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet18(weights=resnet_weights).eval()
    resnet_preprocess = resnet_weights.transforms()
    imagenet_classes = resnet_weights.meta["categories"]
    return imagenet_classes, resnet, resnet_preprocess


@app.cell
def _(mo):
    resnet_image = mo.ui.dropdown(["chelsea", "coffee", "astronaut", "raccoon", "coins"],
                                  value="chelsea", label="Image")
    resnet_image
    return (resnet_image,)


@app.cell
def _(
    IMAGES_DIR,
    Image,
    imagenet_classes,
    mo,
    np,
    plt,
    resnet,
    resnet_image,
    resnet_preprocess,
    torch,
):
    _pil = Image.open(IMAGES_DIR / f"{resnet_image.value}.png").convert("RGB")
    resnet_input = resnet_preprocess(_pil)[None]
    with torch.no_grad():
        _probs = resnet(resnet_input).softmax(1)[0]
    _top = _probs.topk(5)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [1, 1.4]})
    _a1.imshow(np.asarray(_pil))
    _a1.axis("off")
    _a2.barh(range(5)[::-1], _top.values.numpy(), color="tab:blue")
    _a2.set_yticks(range(5)[::-1], [imagenet_classes[i] for i in _top.indices])
    _a2.set(xlim=(0, 1), xlabel="softmax probability", title="ResNet-18 top-5 (ImageNet classes)")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md("*Two things to notice. Predictions are only over ImageNet's 1,000 classes: "
              "there is no \"astronaut\" class, so the network names the closest things it knows. "
              "And the `coins` image is grayscale and unlike typical ImageNet photos, so expect "
              "low confidence there.*"),
    ])
    return (resnet_input,)


@app.cell
def _(mo, plt, resnet, resnet_input, torch):
    _w = resnet.conv1.weight.detach()
    _w = (_w - _w.amin(dim=(1, 2, 3), keepdim=True)) / (
        _w.amax(dim=(1, 2, 3), keepdim=True) - _w.amin(dim=(1, 2, 3), keepdim=True))
    _fig1, _axes1 = plt.subplots(4, 16, figsize=(10, 2.7))
    for _k, _ax in enumerate(_axes1.ravel()):
        _ax.imshow(_w[_k].permute(1, 2, 0).numpy())
        _ax.axis("off")
    _fig1.suptitle("All 64 first-layer kernels (7×7×3), learned from ImageNet", fontsize=9)
    with torch.no_grad():
        _x = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(resnet_input))))
        _feat = resnet.layer1(_x)[0]
    _fig2, _axes2 = plt.subplots(1, 8, figsize=(10, 1.6))
    for _k, _ax in enumerate(_axes2):
        _ax.imshow(_feat[_k].numpy(), cmap="magma")
        _ax.set_title(f"ch {_k}", fontsize=8)
        _ax.axis("off")
    mo.vstack([
        _fig1,
        mo.md("Nobody designed these kernels. Training produced **oriented edge detectors, "
              "Gabor-like stripes, and color-opponent blobs**: close relatives of the "
              "derivative-of-Gaussian filters from notebooks 13 and 15. The first layer "
              "of almost every ConvNet trained on natural images looks like this, which is one "
              "reason pretrained features transfer to new tasks (Section 5)."),
        _fig2,
        mo.md("*Eight of the 64 feature maps after the first residual stage (56×56) for the "
              "selected image. Each is the response of one learned filter at every position: "
              "a map of **what** was detected and **where**.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Attention: let every patch look at every other patch

    A convolution mixes information only within a fixed $K\times K$
    neighborhood. **Self-attention** (Vaswani et al., NeurIPS 2017) mixes
    information between *all* positions, with weights computed from the
    content itself. Each token (an image patch, in a Vision Transformer) emits
    a query $\mathbf{q}_i$, a key $\mathbf{k}_i$ and a value $\mathbf{v}_i$,
    each a learned linear map of the token's features. Token $i$'s output is a
    weighted average of all values:

    $$\mathbf{z}_i = \sum_j a_{ij}\,\mathbf{v}_j, \qquad
      a_{ij} = \mathrm{softmax}_j\!\left(\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d}}\right).$$

    Weights are large where the query and key agree. In matrix form,
    $Z = \mathrm{softmax}(QK^\top/\sqrt{d})\,V$.

    The **Vision Transformer (ViT)** (Dosovitskiy et al., ICLR 2021) cuts a
    $224\times224$ image into $16\times16$ patches: a $14\times14$ grid of
    196 tokens. Its first step turns each patch, $16\times16\times3 = 768$
    raw pixel values, into a token vector with one learned linear map. Below
    we stop right there, so the demo runs without training. The tokens are the
    raw patches, mean-centered and scaled to unit length, and the query and
    key maps are the identity. Attention then compares patch *content*. A real
    ViT *learns* $W_Q, W_K, W_V$ for each of several heads and stacks a dozen
    such layers; the mechanism is the same. Move the query patch and see which
    patches it attends to.
    """)
    return


@app.cell
def _(mo):
    attn_image = mo.ui.dropdown(["astronaut", "coffee", "chelsea"], value="astronaut", label="Image")
    attn_row = mo.ui.slider(0, 13, value=12, step=1, label="Query patch row")
    attn_col = mo.ui.slider(0, 13, value=1, step=1, label="Query patch column")
    attn_temp = mo.ui.slider(0.02, 0.5, value=0.1, step=0.01, label="Temperature (softmax sharpness)")
    attn_self = mo.ui.checkbox(label="Include the query's weight on itself", value=False)
    mo.vstack([
        mo.hstack([attn_image, attn_row, attn_col], justify="start"),
        mo.hstack([attn_temp, attn_self], justify="start"),
    ])
    return attn_col, attn_image, attn_row, attn_self, attn_temp


@app.cell
def _(IMAGES_DIR, Image, attn_image, resnet_preprocess, torch):
    _pil = Image.open(IMAGES_DIR / f"{attn_image.value}.png").convert("RGB")
    attn_input = resnet_preprocess(_pil)[None]                     # 3 x 224 x 224, ImageNet-normalized
    _patches = attn_input[0].unfold(1, 16, 16).unfold(2, 16, 16)    # 3 x 14 x 14 x 16 x 16
    _raw = _patches.permute(1, 2, 0, 3, 4).reshape(196, -1)         # 196 tokens x 768 pixel values
    attn_tokens = torch.nn.functional.normalize(_raw - _raw.mean(0, keepdim=True), dim=1)
    return attn_input, attn_tokens


@app.cell
def _(
    attn_col,
    attn_input,
    attn_row,
    attn_self,
    attn_temp,
    attn_tokens,
    mo,
    np,
    plt,
    torch,
):
    _i = attn_row.value * 14 + attn_col.value
    _scores = attn_tokens @ attn_tokens[_i] / attn_temp.value
    _a_full = torch.softmax(_scores, dim=0)
    if attn_self.value:
        _a = _a_full
    else:
        _scores = _scores.clone()
        _scores[_i] = -float("inf")
        _a = torch.softmax(_scores, dim=0)
    _a = _a.reshape(14, 14).numpy()
    _mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    _std = np.array([0.229, 0.224, 0.225])[:, None, None]
    _rgb = np.clip(attn_input[0].numpy() * _std + _mean, 0, 1).transpose(1, 2, 0)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8.5, 4))
    _a1.imshow(_rgb)
    for _t in range(15):
        _a1.axhline(_t * 16, color="w", lw=0.4, alpha=0.7)
        _a1.axvline(_t * 16, color="w", lw=0.4, alpha=0.7)
    _a1.add_patch(plt.Rectangle((attn_col.value * 16, attn_row.value * 16), 16, 16, fill=False, ec="yellow", lw=2))
    _a1.set_title("196 patch tokens (16×16 px); query in yellow", fontsize=9)
    _a1.axis("off")
    _a2.imshow(_rgb)
    _a2.imshow(np.kron(_a, np.ones((16, 16))), cmap="inferno", alpha=0.65)
    _a2.add_patch(plt.Rectangle((attn_col.value * 16, attn_row.value * 16), 16, 16, fill=False, ec="yellow", lw=1))
    _a2.set_title("attention weights a_ij" + ("" if attn_self.value else " over the other 195 patches"), fontsize=9)
    _a2.axis("off")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"*The query keeps {_a_full[_i]:.0%} of its weight for itself, since a patch always "
              "matches itself best. The map shows where the rest goes (tick the box to include it). "
              "On the astronaut, a patch of orange suit (the default, row 12, column 1) attends to "
              "suit patches on both sides of the body, including ones far away. A background patch "
              "(try row 2, column 1) picks out background on both sides of her. No single "
              "convolution layer can relate two distant patches like this: attention does it in "
              "one step, with weights computed from the image itself. With raw pixels as tokens, "
              "attention matches color and texture, not objects; try the other images to see its "
              "limits. Learned projections and many stacked layers are what let a trained ViT "
              "attend by meaning. Lower temperature gives a sharper softmax.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Attention ignores order, so position must be added

    Attention compares every token with every other token and then averages,
    so it never uses *where* a token sits. Shuffle the 196 tokens and the
    outputs are shuffled the same way, with no change. In symbols, for any
    permutation matrix $P$:
    $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$. The check below confirms it
    numerically. This is why every Transformer adds a **positional encoding**
    to its tokens: the sinusoidal encoding of the original Transformer, or
    learned position vectors in ViT. NeRF (notebook 24) used the same
    sinusoids to encode coordinates. It is also the main trade-off with
    convolution. A ConvNet *assumes* locality and gets it for free. A ViT has
    to *learn* spatial structure, which is why ViTs need more training data
    (Dosovitskiy et al. pretrained on 14M to 300M images) before they beat
    ConvNets.
    """)
    return


@app.cell
def _(attn_tokens, mo, torch):
    def self_attention(X, temperature=0.08):
        return torch.softmax(X @ X.T / temperature, dim=1) @ X

    _perm = torch.randperm(len(attn_tokens), generator=torch.Generator().manual_seed(0))
    _out = self_attention(attn_tokens)
    _out_perm = self_attention(attn_tokens[_perm])
    _err = (_out_perm - _out[_perm]).abs().max().item()
    _rows = []
    for _size in (224, 384, 512, 1024):
        _n = (_size // 16) ** 2
        _rows.append(f"| {_size}×{_size} | {_n:,} | {_n * _n:,} | {_n * _n * 4 / 2**20:,.1f} MB |")
    mo.md(f"""
    **Permutation check:** max |Attn(PX) − P·Attn(X)| = {_err:.1e} (floating-point round-off).

    The price of all-pairs mixing is that the attention matrix grows with the
    **square** of the number of tokens (16×16 patches, one head, float32):

    | image | tokens N | attention entries N² | memory per head per layer |
    |---|---|---|---|
    {chr(10).join(_rows)}

    Doubling the image side multiplies the cost by 16. Hierarchical and windowed
    designs such as Swin restrict attention to local windows, which brings back a
    convolution-like locality assumption.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Transfer learning: reuse what ImageNet taught

    Training a network from scratch needs many labeled images per class. A
    course project rarely has them. **Transfer learning** starts from a network
    pretrained on a large dataset and adapts it:

    - **Feature extraction (linear probe):** freeze the pretrained network,
      remove its last layer, and train only a new linear classifier on its
      512-dimensional features. This is the cheapest form, and it is what we run
      here.
    - **Fine-tuning:** start from the pretrained weights and keep training some
      or all layers on the new task with a small learning rate. It usually
      beats the linear probe once there are enough labels, at a much higher
      compute cost.

    **The experiment.** CIFAR-10 (Krizhevsky, 2009) has 32×32 photos in 10
    classes. We use its 10,000-image test split (one 24 MB file), take a
    labeled pool of 300 images per class and a separate test set of 1,000
    images, and compare at 5, 20 and 100 labels per class:

    1. **Pretrained:** frozen ImageNet ResNet-18 features (images upsampled to
       96×96), plus logistic regression;
    2. **From scratch:** a small 3-layer CNN trained on the 32×32 images with
       Adam for 1,000 mini-batch steps.

    The first run downloads the data and takes about two minutes on a laptop
    CPU. Results are cached on disk, so later runs are instant.
    """)
    return


@app.cell
def _(mo):
    transfer_run = mo.ui.run_button(label="Run the transfer-learning experiment")
    transfer_run
    return (transfer_run,)


@app.cell
def _(Image, io, mo, np, os, tempfile, transfer_run, urllib):
    mo.stop(not transfer_run.value, mo.md("*Click the button above to run Section 5.*"))
    import pyarrow.parquet as _pq

    CACHE_DIR = os.path.join(tempfile.gettempdir(), "ece491_l22_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    _path = os.path.join(CACHE_DIR, "cifar10_test.parquet")
    if not os.path.exists(_path):
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
            _path)
    _table = _pq.read_table(_path)
    _labels_all = np.array(_table.column("label").to_pylist())
    _rng = np.random.default_rng(0)
    _pool = np.concatenate([_rng.choice(np.flatnonzero(_labels_all == c), 300, replace=False) for c in range(10)])
    _test = _rng.choice(np.setdiff1d(np.arange(len(_labels_all)), _pool), 1000, replace=False)
    _use = np.concatenate([_pool, _test])
    _bytes = _table.column("img").to_pylist()
    cifar_images = np.stack([np.asarray(Image.open(io.BytesIO(_bytes[i]["bytes"])).convert("RGB")) for i in _use])
    cifar_labels = _labels_all[_use]
    cifar_pool_idx = np.arange(len(_pool))
    cifar_test_idx = np.arange(len(_pool), len(_use))
    CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return (
        CACHE_DIR,
        CIFAR_CLASSES,
        cifar_images,
        cifar_labels,
        cifar_pool_idx,
        cifar_test_idx,
    )


@app.cell
def _(CIFAR_CLASSES, cifar_images, cifar_labels, cifar_pool_idx, mo, plt):
    _fig, _axes = plt.subplots(2, 10, figsize=(10, 2.3))
    for _c in range(10):
        _idx = cifar_pool_idx[cifar_labels[cifar_pool_idx] == _c][:2]
        for _r in range(2):
            _axes[_r, _c].imshow(cifar_images[_idx[_r]])
            _axes[_r, _c].axis("off")
        _axes[0, _c].set_title(CIFAR_CLASSES[_c], fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("*CIFAR-10 at its native 32×32 resolution: small, low-detail, and "
                           "quite unlike the 224×224 ImageNet photos ResNet-18 was trained on.*")])
    return


@app.cell
def _(CACHE_DIR, F, cifar_images, np, os, resnet, torch):
    _feat_path = os.path.join(CACHE_DIR, "resnet18_feats_96px.npy")
    if os.path.exists(_feat_path):
        cifar_feats = np.load(_feat_path)
    else:
        _backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).eval()   # drop the 1000-way fc layer
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        _X = torch.tensor(cifar_images).permute(0, 3, 1, 2).float() / 255.0
        _chunks = []
        with torch.no_grad():
            for _i in range(0, len(_X), 250):
                _xb = F.interpolate(_X[_i:_i + 250], size=96, mode="bilinear", align_corners=False)
                _chunks.append(_backbone((_xb - _mean) / _std).flatten(1))
        cifar_feats = torch.cat(_chunks).numpy()
        np.save(_feat_path, cifar_feats)
    return (cifar_feats,)


@app.cell
def _(
    CACHE_DIR,
    LogisticRegression,
    accuracy,
    cifar_feats,
    cifar_images,
    cifar_labels,
    cifar_pool_idx,
    cifar_test_idx,
    json,
    nn,
    np,
    os,
    torch,
    train_classifier,
):
    _res_path = os.path.join(CACHE_DIR, "transfer_results_v1.json")
    if os.path.exists(_res_path):
        transfer_results = json.load(open(_res_path))
    else:
        _X = torch.tensor(cifar_images).permute(0, 3, 1, 2).float() / 255.0
        _y = torch.tensor(cifar_labels)
        _rng = np.random.default_rng(1)
        transfer_results = {"k": [5, 20, 100], "probe": [], "scratch": []}
        for _k in transfer_results["k"]:
            _sel = np.concatenate([_rng.choice(cifar_pool_idx[cifar_labels[cifar_pool_idx] == c], _k, replace=False)
                                   for c in range(10)])
            _probe = LogisticRegression(max_iter=3000).fit(cifar_feats[_sel], cifar_labels[_sel])
            transfer_results["probe"].append(_probe.score(cifar_feats[cifar_test_idx], cifar_labels[cifar_test_idx]))
            torch.manual_seed(0)
            _cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))
            _epochs = max(1, round(1000 * 50 / len(_sel)))      # about 1,000 steps of batch 50
            train_classifier(_cnn, _X[_sel], _y[_sel], epochs=_epochs, batch_size=50, lr=1e-3)
            transfer_results["scratch"].append(accuracy(_cnn, _X[cifar_test_idx], _y[cifar_test_idx]))
        json.dump(transfer_results, open(_res_path, "w"))
    return (transfer_results,)


@app.cell
def _(mo, plt, transfer_results):
    _k = transfer_results["k"]
    _fig, _ax = plt.subplots(figsize=(5.5, 3.4))
    _ax.plot(_k, transfer_results["probe"], "o-", label="pretrained ResNet-18 + linear probe")
    _ax.plot(_k, transfer_results["scratch"], "s-", label="small CNN trained from scratch")
    _ax.axhline(0.1, color="0.6", ls=":", lw=1)
    _ax.text(_k[0], 0.11, "chance", fontsize=8, color="0.4")
    _ax.set(xscale="log", xticks=_k, xticklabels=[str(v) for v in _k], ylim=(0, 1),
            xlabel="labeled training images per class", ylabel="test accuracy (1,000 images)",
            title="CIFAR-10 with few labels")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    _rows = "\n".join(f"| {k} | {10 * k} | {p:.1%} | {s:.1%} |"
                      for k, p, s in zip(_k, transfer_results["probe"], transfer_results["scratch"]))
    mo.vstack([
        _fig,
        mo.md(f"""
        | labels per class | total labels | pretrained + linear probe | from scratch |
        |---|---|---|---|
        {_rows}

        With a few labels per class, the network that has seen 1.28 million other photos is far
        ahead of the one that has seen only these. The pretrained model was never shown a
        CIFAR image, and only its last linear layer was trained here. Two caveats keep the
        comparison honest. The from-scratch CNN is small and trained briefly, so a bigger
        model with data augmentation would narrow the gap at 100 labels per class. And
        fine-tuning the pretrained network instead of freezing it would widen the gap again.
        The direction does not change: **when labels are scarce, start from a pretrained
        network.** That is the default for your course project.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary and what comes next

    - A **convolution layer** is a bank of learned filters: local, shared across
      positions, and orders of magnitude cheaper than a fully connected layer.
      Weight sharing gives translation equivariance; global pooling turns it
      into invariance; stride-2 subsampling quietly breaks it (aliasing).
    - Trained on ImageNet, the first layer of a ConvNet rediscovers oriented edge
      and Gabor-like filters, and deeper layers respond to textures, parts and
      objects. **ResNet's** residual connections make such deep stacks trainable.
    - **Self-attention** relates every patch to every other patch with
      content-dependent weights. It ignores position (hence positional
      encodings), costs $O(N^2)$ in the number of tokens, and needs more data
      than a ConvNet because it does not assume locality.
    - **Transfer learning:** pretrained features plus a linear classifier beat a
      CNN trained from scratch by a wide margin when labels are scarce.

    **Lecture 23** turns this into practice for **object classification**:
    datasets and benchmarks, data augmentation, fine-tuning recipes, and how to
    evaluate a classifier honestly.

    **Reading:** Szeliski, *Computer Vision: Algorithms and Applications* (2nd
    ed.), §5.4–5.5.
    """)
    return


if __name__ == "__main__":
    app.run()
