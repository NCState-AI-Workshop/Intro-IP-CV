# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "pyarrow",
#     "scikit-learn",
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
    from sklearn.linear_model import LogisticRegression
    from torchvision.transforms import v2 as T

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return (
        F,
        Image,
        LogisticRegression,
        T,
        io,
        json,
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
def _(mo, os, tempfile):
    IMAGES_DIR = mo.notebook_dir() / "images"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "ece491_l23_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return CACHE_DIR, CIFAR_CLASSES, IMAGES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Lecture 23: Object Classification

    Lecture 22 built the parts: convolution layers, residual networks,
    attention, and pretrained backbones. This notebook puts them together
    into the workflow you will use for an image-classification project
    (Szeliski §6.2):

    1. **Benchmarks and backbones.** What the standard datasets are, and what
       today's off-the-shelf networks cost and score.
    2. **Data augmentation.** Label-preserving transformations that make a
       small training set look larger.
    3. **The training recipe.** Normalization and augmentation, measured on
       one small network: which one helps the network *train*, and which one
       helps it *generalize*.
    4. **Fine-tuning a pretrained network,** and what a too-large learning
       rate does to it.
    5. **Evaluating honestly.** Train/validation/test splits, the confusion
       matrix, and whether the network's confidence can be trusted
       (calibration).
    6. **What the network looks at** (Grad-CAM).

    Sections 1, 2 and 6 run immediately. Sections 3–5 download about 25 MB
    of CIFAR-10 and train networks on the CPU, so each waits for a button.
    Results are cached, so a second click is instant.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Benchmarks and backbones

    A classifier maps an image to one label out of a fixed set. Progress in
    the field has been measured on a few shared benchmarks:

    | Benchmark | Year | Training images | Classes | Image size |
    |---|---|---|---|---|
    | MNIST | 1998 | 60,000 | 10 digits | 28×28 gray |
    | CIFAR-10 / CIFAR-100 | 2009 | 50,000 | 10 / 100 | 32×32 color |
    | ImageNet-1k (ILSVRC) | 2012 | 1.28 million | 1,000 | full-size photos |

    On ImageNet, the winning top-5 error fell from 25.8% (2011, hand-crafted
    features) to 15.3% (2012, AlexNet) and 3.6% (2015, ResNet). **Top-5
    error** counts a prediction as correct if the true label is among the
    network's five highest scores. ImageNet has many near-duplicate classes
    (about 120 dog breeds), so top-5 forgives some reasonable confusions.
    Today **top-1 accuracy** is the number people report.

    A benchmark score is a measurement on one test set. Recht et al. (ICML
    2019) collected a new ImageNet test set following the original protocol,
    and accuracy dropped by 11–14 points for every model they tested. The
    ranking of models was preserved, but the absolute numbers did not
    transfer. Keep this in mind in Section 5.

    The table below is read from torchvision's own model metadata: the
    networks you can download with one line of code, their size, their cost
    per $224\times224$ image, and their ImageNet accuracy.
    """)
    return


@app.cell
def _(mo, torchvision):
    _M = torchvision.models
    _rows = [
        ("MobileNetV3-Large", _M.MobileNet_V3_Large_Weights.IMAGENET1K_V1, "ConvNet for phones"),
        ("EfficientNet-B0", _M.EfficientNet_B0_Weights.IMAGENET1K_V1, "ConvNet, scaled by search"),
        ("ResNet-18", _M.ResNet18_Weights.IMAGENET1K_V1, "the ResNet we used in L22"),
        ("ResNet-50, 2016 recipe", _M.ResNet50_Weights.IMAGENET1K_V1, "original training recipe"),
        ("ResNet-50, 2021 recipe", _M.ResNet50_Weights.IMAGENET1K_V2, "**same network**, modern recipe"),
        ("ConvNeXt-Tiny", _M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "ConvNet modernized after ViT"),
        ("Swin-T", _M.Swin_T_Weights.IMAGENET1K_V1, "Transformer with local windows"),
        ("ViT-B/16", _M.ViT_B_16_Weights.IMAGENET1K_V1, "plain Vision Transformer"),
    ]
    _lines = ["| Backbone | Parameters | GFLOPs / image | Top-1 | Top-5 | Note |", "|---|---|---|---|---|---|"]
    for _name, _w, _note in _rows:
        _m = _w.meta
        _acc = _m["_metrics"]["ImageNet-1K"]
        _lines.append(f"| {_name} | {_m['num_params'] / 1e6:.1f} M | {_m['_ops']:.1f} | "
                      f"{_acc['acc@1']:.1f}% | {_acc['acc@5']:.1f}% | {_note} |")
    mo.md("\n".join(_lines) + r"""

    *Source: `torchvision.models` weight metadata (ImageNet-1k validation set, single
    $224\times224$ center crop).* Two rows deserve a second look. The two
    ResNet-50 rows are the **same architecture with the same number of
    parameters**: training it longer, with stronger augmentation, label smoothing
    and a better schedule raises top-1 accuracy by almost five points. That is
    more than separates the modern-recipe ResNet-50 from any newer
    architecture in the table. And
    the Transformer (ViT-B/16) is not more accurate than the best ConvNet here
    while costing four times the compute. Architecture matters, but so does
    the recipe, which is what the rest of this notebook is about.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data augmentation: more training images for free

    A cat photo flipped left–right is still a cat photo. **Data augmentation**
    applies such label-preserving transformations to every training image,
    with fresh random parameters each time the image is used. The network
    never sees exactly the same input twice, so it is pushed to rely on what
    stays constant (the object), not on accidents of one photo (its exact
    position, the lighting).

    Which transformations preserve the label is a property of the *task*, not
    of the network:

    - Horizontal flips are safe for natural photos, but not for text or digits
      (a mirrored "2" is not a "2").
    - Vertical flips are safe for satellite and microscope images, not for
      street scenes.
    - Strong color changes are wrong if color is the cue (ripe vs. unripe
      fruit).

    Turn transformations on and off below, and move the seed to draw new
    random parameters. *TrivialAugment* (Müller and Hutter, ICCV 2021) picks
    one operation from a fixed list at random per image and is part of the
    modern ResNet-50 recipe in the table above.
    """)
    return


@app.cell
def _(mo):
    aug_image = mo.ui.dropdown(["chelsea", "coffee", "astronaut", "raccoon"], value="chelsea", label="Image")
    aug_crop = mo.ui.checkbox(value=True, label="Random resized crop")
    aug_flip = mo.ui.checkbox(value=True, label="Horizontal flip")
    aug_color = mo.ui.checkbox(value=False, label="Color jitter")
    aug_rotate = mo.ui.checkbox(value=False, label="Rotation (±20°)")
    aug_erase = mo.ui.checkbox(value=False, label="Random erasing")
    aug_trivial = mo.ui.checkbox(value=False, label="TrivialAugment")
    aug_seed = mo.ui.slider(0, 50, value=0, label="Random seed")
    mo.vstack([
        mo.hstack([aug_image, aug_seed], justify="start"),
        mo.hstack([aug_crop, aug_flip, aug_color, aug_rotate, aug_erase, aug_trivial], justify="start"),
    ])
    return (
        aug_color,
        aug_crop,
        aug_erase,
        aug_flip,
        aug_image,
        aug_rotate,
        aug_seed,
        aug_trivial,
    )


@app.cell
def _(
    IMAGES_DIR,
    Image,
    T,
    aug_color,
    aug_crop,
    aug_erase,
    aug_flip,
    aug_image,
    aug_rotate,
    aug_seed,
    aug_trivial,
    plt,
    torch,
):
    _pil = Image.open(IMAGES_DIR / f"{aug_image.value}.png").convert("RGB")
    _ops = [T.RandomResizedCrop(224, scale=(0.25, 1.0)) if aug_crop.value else T.Resize((224, 224))]
    if aug_flip.value:
        _ops.append(T.RandomHorizontalFlip())
    if aug_color.value:
        _ops.append(T.ColorJitter(0.4, 0.4, 0.4, 0.1))
    if aug_rotate.value:
        _ops.append(T.RandomRotation(20))
    if aug_trivial.value:
        _ops.append(T.TrivialAugmentWide())
    _ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if aug_erase.value:
        _ops.append(T.RandomErasing(p=1.0, scale=(0.05, 0.2)))
    _pipeline = T.Compose(_ops)
    torch.manual_seed(aug_seed.value)
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5.2))
    _axes[0, 0].imshow(_pil.resize((224, 224)))
    _axes[0, 0].set_title("original", fontsize=9)
    for _k, _ax in enumerate(_axes.flat[1:]):
        _ax.imshow(_pipeline(_pil).permute(1, 2, 0).clamp(0, 1).numpy())
        _ax.set_title(f"sample {_k + 1}", fontsize=9)
    for _ax in _axes.flat:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The training recipe, measured

    The rest of the notebook uses **CIFAR-10**. To keep the download small we
    take its official 10,000-image test release and split it ourselves into
    three disjoint parts, stratified by class:

    | Split | Per class | Total | Used for |
    |---|---|---|---|
    | train | 200 | 2,000 | fitting the weights |
    | validation | 100 | 1,000 | choosing settings (epochs, learning rate, temperature) |
    | test | 700 | 7,000 | the final number, computed once per model |

    The rule that keeps the final number honest: **nothing about the model
    is chosen by looking at the test set.** Every time you tune a setting on
    the test set, the test accuracy drifts upward and stops predicting how
    the model will do on new images.
    """)
    return


@app.cell
def _(mo):
    data_run = mo.ui.run_button(label="Download CIFAR-10 (25 MB) and build the splits")
    data_run
    return (data_run,)


@app.cell
def _(CACHE_DIR, Image, data_run, io, mo, np, os, urllib):
    mo.stop(not data_run.value, mo.md("*Click the button above to load the data for Sections 3–5.*"))
    import pyarrow.parquet as _pq

    _path = os.path.join(CACHE_DIR, "cifar10_test.parquet")
    if not os.path.exists(_path):
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
            _path)
    _table = _pq.read_table(_path)
    cifar_y = np.array(_table.column("label").to_pylist())
    cifar_x = np.stack([np.asarray(Image.open(io.BytesIO(_d["bytes"])).convert("RGB"))
                        for _d in _table.column("img").to_pylist()])
    _rng = np.random.default_rng(0)
    _per_class = np.stack([_rng.permutation(np.flatnonzero(cifar_y == _c)) for _c in range(10)])  # 10 x 1000
    split_idx = {
        "train": _per_class[:, :200].ravel(),
        "val": _per_class[:, 200:300].ravel(),
        "test": _per_class[:, 300:].ravel(),
    }
    return cifar_x, cifar_y, split_idx


@app.cell
def _(CIFAR_CLASSES, cifar_x, cifar_y, mo, plt, split_idx):
    _fig, _axes = plt.subplots(2, 10, figsize=(10, 2.3))
    for _c in range(10):
        _idx = split_idx["train"][cifar_y[split_idx["train"]] == _c][:2]
        for _r in range(2):
            _axes[_r, _c].imshow(cifar_x[_idx[_r]])
            _axes[_r, _c].axis("off")
        _axes[0, _c].set_title(CIFAR_CLASSES[_c], fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"*Loaded {len(cifar_y):,} images: {len(split_idx['train']):,} train, "
        f"{len(split_idx['val']):,} validation, {len(split_idx['test']):,} test.*")])
    return


@app.cell
def _(F, nn, torch):
    def small_cnn(width=16, batchnorm=True):
        """Five 3x3 conv layers, two 2x2 max-pools, global average pooling, one linear layer."""
        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not batchnorm)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            return layers + [nn.ReLU()]
        w = width
        return nn.Sequential(
            *block(3, w), *block(w, w), nn.MaxPool2d(2),
            *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2),
            *block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 10))

    def flip_and_crop(xb, pad=4):
        """CIFAR's standard augmentation: random horizontal flip, then a random crop of a padded image."""
        n, _, h, w = xb.shape
        flip = torch.rand(n) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
        xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
        ij = torch.randint(0, 2 * pad + 1, (n, 2))
        return torch.stack([xp[k, :, a:a + h, b:b + w] for k, (a, b) in enumerate(ij.tolist())])

    def predict_logits(model, X, prep=None, batch=500):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = X[i:i + batch]
                outs.append(model(prep(xb) if prep is not None else xb))
        return torch.cat(outs)
    return flip_and_crop, predict_logits, small_cnn


@app.cell
def _(mo):
    mo.md(r"""
    We train the same small ConvNet (5 convolution layers, about 36,000
    parameters) three times, for 30 epochs each, with the same optimizer
    (Adam, one-cycle learning-rate schedule) and the same random seed:

    1. **plain:** no normalization, no augmentation;
    2. **+ BatchNorm:** a BatchNorm layer after every convolution (ECE 763
       Lecture 9; Ioffe and Szegedy, 2015);
    3. **+ BatchNorm + augmentation:** also a random flip and a random crop
       of the zero-padded image (4 pixels) at every step.

    Watch two numbers for each: how well the network fits its training images,
    and how well it does on the validation images it never trained on.
    """)
    return


@app.cell
def _(mo):
    recipe_run = mo.ui.run_button(label="Train the three networks (about 3 minutes)")
    recipe_run
    return (recipe_run,)


@app.cell
def _(
    CACHE_DIR,
    F,
    cifar_x,
    cifar_y,
    flip_and_crop,
    json,
    mo,
    os,
    predict_logits,
    recipe_run,
    small_cnn,
    split_idx,
    time,
    torch,
):
    mo.stop(not recipe_run.value, mo.md("*Click the button above to run the experiment.*"))
    _path = os.path.join(CACHE_DIR, "recipe_results_v1.json")
    if os.path.exists(_path):
        recipe_results = json.load(open(_path))
    else:
        _X = torch.tensor(cifar_x).permute(0, 3, 1, 2).float() / 255.0
        _y = torch.tensor(cifar_y)
        _tr = torch.tensor(split_idx["train"])
        _mean = _X[_tr].mean((0, 2, 3), keepdim=True)          # statistics of the TRAINING split only
        _std = _X[_tr].std((0, 2, 3), keepdim=True)
        _X = (_X - _mean) / _std

        def _acc(model, idx):
            return (predict_logits(model, _X[idx]).argmax(1) == _y[idx]).float().mean().item()

        _epochs, _batch, _lr = 30, 64, 3e-3
        recipe_results = {"epochs": _epochs, "arms": {}}
        for _name, _bn, _aug in [("plain", False, False), ("+ BatchNorm", True, False),
                                 ("+ BatchNorm + augmentation", True, True)]:
            torch.manual_seed(0)
            _model = small_cnn(16, batchnorm=_bn)
            _opt = torch.optim.Adam(_model.parameters(), _lr)
            _sched = torch.optim.lr_scheduler.OneCycleLR(_opt, max_lr=_lr, total_steps=_epochs * (len(_tr) // _batch))
            _curve, _t0 = [], time.time()
            for _ep in range(_epochs):
                _model.train()
                _perm = _tr[torch.randperm(len(_tr))]
                for _k in range(0, len(_perm) - _batch + 1, _batch):
                    _b = _perm[_k:_k + _batch]
                    _xb = flip_and_crop(_X[_b]) if _aug else _X[_b]
                    _loss = F.cross_entropy(_model(_xb), _y[_b])
                    _opt.zero_grad()
                    _loss.backward()
                    _opt.step()
                    _sched.step()
                _curve.append([_acc(_model, split_idx["train"]), _acc(_model, split_idx["val"])])
            recipe_results["arms"][_name] = {
                "curve": _curve,
                "test": _acc(_model, split_idx["test"]),
                "params": sum(p.numel() for p in _model.parameters()),
                "seconds": time.time() - _t0,
            }
        json.dump(recipe_results, open(_path, "w"))
    return (recipe_results,)


@app.cell
def _(mo, np, plt, recipe_results):
    _arms = recipe_results["arms"]
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for _ax, (_name, _r) in zip(_axes, _arms.items()):
        _c = np.array(_r["curve"])
        _ep = np.arange(1, len(_c) + 1)
        _ax.plot(_ep, _c[:, 0], color="tab:blue", label="train")
        _ax.plot(_ep, _c[:, 1], color="tab:orange", label="validation")
        _ax.fill_between(_ep, _c[:, 1], _c[:, 0], where=_c[:, 0] > _c[:, 1], color="tab:red", alpha=0.12)
        _ax.set(title=_name, xlabel="epoch", ylim=(0, 1))
        _ax.grid(alpha=0.3)
    _axes[0].set_ylabel("accuracy")
    _axes[0].legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {_n} | {_r['curve'][-1][0]:.1%} | {_r['curve'][-1][1]:.1%} | "
        f"{_r['curve'][-1][0] - _r['curve'][-1][1]:+.1%} | {_r['test']:.1%} |"
        for _n, _r in _arms.items())
    mo.vstack([_fig, mo.md(f"""
    | Recipe | Train accuracy | Validation accuracy | Gap (shaded) | Test accuracy |
    |---|---|---|---|---|
    {_rows}

    *Same network, data, optimizer and seed in all three; {recipe_results['epochs']} epochs,
    2,000 training images.*

    - **BatchNorm helps the network train.** Without it, the plain network has not
      even fit its own training set after 30 epochs. With it, the same network fits
      the training set almost perfectly.
    - **Augmentation helps the network generalize.** Fitting 2,000 images almost
      perfectly is memorization: the gap between train and validation accuracy is
      the overfitting. With flips and crops the network cannot memorize individual
      images. The gap shrinks to about a third, and validation and test accuracy
      rise by about two points.
    - The training accuracy is *lower* with augmentation. That is expected: the
      training images are harder (and different every epoch). A falling training
      accuracy is not a problem if the validation accuracy rises.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Fine-tuning a pretrained network

    Lecture 22 used a pretrained ResNet-18 as a frozen feature extractor and
    trained only a new linear layer on top (a **linear probe**).
    **Fine-tuning** goes one step further: replace the 1,000-way ImageNet
    layer with a new 10-way layer, then keep training *all* the weights on
    the new data.

    The usual practice is to give the pretrained layers a learning rate about
    ten times smaller than the new layer. The pretrained weights are already
    good; large steps would overwrite them before the new layer has learned
    anything useful. Try it: the dropdown sets the backbone learning rate
    (the new layer always gets 10× that), with AdamW, a cosine schedule,
    random flips, and 4 epochs at $64\times64$ input.
    """)
    return


@app.cell
def _(mo):
    ft_lr = mo.ui.dropdown(
        {"1e-4 (recommended)": 1e-4, "1e-3": 1e-3, "1e-2 (a from-scratch learning rate)": 1e-2},
        value="1e-4 (recommended)", label="Backbone learning rate")
    transfer_run = mo.ui.run_button(label="Run linear probe and fine-tuning (about 3 minutes per setting)")
    mo.hstack([ft_lr, transfer_run], justify="start")
    return ft_lr, transfer_run


@app.cell
def _(
    CACHE_DIR,
    F,
    LogisticRegression,
    cifar_x,
    cifar_y,
    ft_lr,
    json,
    mo,
    nn,
    np,
    os,
    predict_logits,
    split_idx,
    time,
    torch,
    torchvision,
    transfer_run,
):
    mo.stop(not transfer_run.value, mo.md("*Click the button above to run Section 4 (Section 5 uses its result).*"))
    _res = 64
    _weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    _X = torch.tensor(cifar_x).permute(0, 3, 1, 2).float() / 255.0
    _y = torch.tensor(cifar_y)
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _prep(xb):
        return (F.interpolate(xb, size=_res, mode="bilinear", align_corners=False) - _mean) / _std

    # Linear probe on frozen features (computed once, then cached).
    _probe_path = os.path.join(CACHE_DIR, "probe_results_v1.json")
    if os.path.exists(_probe_path):
        probe_results = json.load(open(_probe_path))
    else:
        _backbone = torchvision.models.resnet18(weights=_weights)
        _backbone.fc = nn.Identity()
        _f = {s: predict_logits(_backbone, _X[split_idx[s]], prep=_prep).numpy() for s in ("train", "val", "test")}
        _clf = LogisticRegression(max_iter=3000).fit(_f["train"], cifar_y[split_idx["train"]])
        probe_results = {s: float(_clf.score(_f[s], cifar_y[split_idx[s]])) for s in ("val", "test")}
        json.dump(probe_results, open(_probe_path, "w"))

    # Fine-tuning at the chosen learning rate (cached per learning rate).
    _tag = f"{ft_lr.value:.0e}"
    _ft_path = os.path.join(CACHE_DIR, f"finetune_{_tag}_v1.npz")
    if os.path.exists(_ft_path):
        _z = np.load(_ft_path)
        ft_val_logits, ft_test_logits, _curve, _secs = _z["val"], _z["test"], _z["curve"].tolist(), float(_z["secs"])
    else:
        torch.manual_seed(0)
        _model = torchvision.models.resnet18(weights=_weights)
        _model.fc = nn.Linear(512, 10)
        _lr = ft_lr.value
        _opt = torch.optim.AdamW([
            {"params": [p for n, p in _model.named_parameters() if not n.startswith("fc.")], "lr": _lr},
            {"params": _model.fc.parameters(), "lr": 10 * _lr}], weight_decay=1e-4)
        _epochs, _batch = 4, 32
        _tr = torch.tensor(split_idx["train"])
        _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, _epochs * (len(_tr) // _batch))
        _curve, _t0 = [], time.time()
        for _ep in range(_epochs):
            _model.train()
            _perm = _tr[torch.randperm(len(_tr))]
            for _k in range(0, len(_perm) - _batch + 1, _batch):
                _b = _perm[_k:_k + _batch]
                _xb = _X[_b]
                _xb = torch.where((torch.rand(len(_b)) < 0.5).view(-1, 1, 1, 1), _xb.flip(3), _xb)
                _loss = F.cross_entropy(_model(_prep(_xb)), _y[_b])
                _opt.zero_grad()
                _loss.backward()
                _opt.step()
                _sched.step()
            _vl = predict_logits(_model, _X[split_idx["val"]], prep=_prep)
            _curve.append((_vl.argmax(1).numpy() == cifar_y[split_idx["val"]]).mean())
        _secs = time.time() - _t0
        ft_val_logits = _vl.numpy()
        ft_test_logits = predict_logits(_model, _X[split_idx["test"]], prep=_prep).numpy()
        np.savez(_ft_path, val=ft_val_logits, test=ft_test_logits, curve=np.array(_curve), secs=_secs)
    ft_results = {
        "lr": ft_lr.value,
        "val_curve": [float(v) for v in _curve],
        "test": float((ft_test_logits.argmax(1) == cifar_y[split_idx["test"]]).mean()),
        "seconds": _secs,
    }
    return ft_results, ft_test_logits, ft_val_logits, probe_results


@app.cell
def _(ft_results, mo, probe_results):
    _curve = " → ".join(f"{v:.1%}" for v in ft_results["val_curve"])
    mo.md(f"""
    | Method (ResNet-18 pretrained on ImageNet, 2,000 CIFAR-10 training images) | Validation | Test |
    |---|---|---|
    | Linear probe: frozen backbone, logistic regression on its features | {probe_results['val']:.1%} | {probe_results['test']:.1%} |
    | Fine-tuning, backbone learning rate {ft_results['lr']:.0e} | {ft_results['val_curve'][-1]:.1%} | {ft_results['test']:.1%} |

    Validation accuracy after each fine-tuning epoch: {_curve}.

    With the recommended learning rate (1e-4), fine-tuning beats the linear probe,
    which in turn beats the best network trained from scratch in Section 3: the
    pretrained features are a better starting point than random weights, and
    letting them adapt to the new images helps further. Now select **1e-2**, a
    typical learning rate for training from scratch, and run again. The large
    first steps destroy the pretrained features, and four epochs on 2,000 images
    are far too few to learn new ones: the result falls *below* the network
    trained from scratch. Fine-tuning needs a small learning rate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Evaluating a classifier honestly

    One accuracy number hides almost everything that matters about a
    classifier. Below, the fine-tuned network from Section 4 is evaluated on
    the 7,000 test images it has never seen. (If you selected a different
    learning rate in Section 4, this section follows it.)

    **The confusion matrix** counts, for each true class (row), how often each
    class was predicted (column). Accuracy is the diagonal; the off-diagonal
    entries show *which* mistakes the network makes, and whether they are
    reasonable ones.
    """)
    return


@app.cell
def _(CIFAR_CLASSES, cifar_y, ft_test_logits, mo, np, plt, split_idx):
    _y = cifar_y[split_idx["test"]]
    _pred = ft_test_logits.argmax(1)
    _cm = np.zeros((10, 10), int)
    np.add.at(_cm, (_y, _pred), 1)
    _rate = _cm / _cm.sum(1, keepdims=True)
    _top2 = (np.argsort(-ft_test_logits, 1)[:, :2] == _y[:, None]).any(1).mean()
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    _a1.imshow(_rate, cmap="Blues", vmin=0, vmax=1)
    for _i in range(10):
        for _j in range(10):
            if _cm[_i, _j]:
                _a1.text(_j, _i, _cm[_i, _j], ha="center", va="center", fontsize=6.5,
                         color="white" if _rate[_i, _j] > 0.5 else "black")
    _a1.set_xticks(range(10), CIFAR_CLASSES, rotation=60, fontsize=8)
    _a1.set_yticks(range(10), CIFAR_CLASSES, fontsize=8)
    _a1.set(xlabel="predicted class", ylabel="true class", title="confusion matrix (test, counts)")
    _per = np.diag(_rate)
    _order = np.argsort(_per)
    _a2.barh(range(10), _per[_order], color="tab:blue")
    _a2.set_yticks(range(10), [CIFAR_CLASSES[i] for i in _order], fontsize=8)
    _a2.axvline(_per.mean(), color="k", ls="--", lw=1)
    _a2.set(xlim=(0, 1), xlabel="per-class accuracy (recall)", title="hardest classes at the bottom")
    _fig.tight_layout()
    _off = _cm.copy()
    np.fill_diagonal(_off, 0)
    _pairs = []
    for _flat in np.argsort(-_off, axis=None)[:4]:
        _i, _j = divmod(int(_flat), 10)
        _pairs.append(f"**{CIFAR_CLASSES[_i]} → {CIFAR_CLASSES[_j]}** ({_off[_i, _j]})")
    mo.vstack([_fig, mo.md(f"""
    Top-1 accuracy {(_pred == _y).mean():.1%}; top-2 accuracy {_top2:.1%}.
    Per-class accuracy ranges from {_per.min():.0%} ({CIFAR_CLASSES[_order[0]]}) to
    {_per.max():.0%} ({CIFAR_CLASSES[_order[-1]]}). The most frequent mistakes (true → predicted):
    {", ".join(_pairs)}. Mistakes concentrate on visually similar classes (animals
    with animals, vehicles with vehicles), which the single accuracy number does not
    tell you. With imbalanced classes (rare diseases, rare defects), always report
    per-class numbers: a classifier can reach 99% accuracy by never predicting a class
    that makes up 1% of the data.
    """)])
    return


@app.cell
def _(CIFAR_CLASSES, cifar_x, cifar_y, ft_test_logits, mo, np, plt, split_idx):
    _idx = split_idx["test"]
    _p = np.exp(ft_test_logits - ft_test_logits.max(1, keepdims=True))
    _p /= _p.sum(1, keepdims=True)
    _pred = _p.argmax(1)
    _wrong = np.flatnonzero(_pred != cifar_y[_idx])
    _worst = _wrong[np.argsort(-_p[_wrong].max(1))][:8]
    _fig, _axes = plt.subplots(1, 8, figsize=(11, 1.9))
    for _ax, _k in zip(_axes, _worst):
        _ax.imshow(cifar_x[_idx[_k]])
        _ax.set_title(f"true: {CIFAR_CLASSES[cifar_y[_idx[_k]]]}\nsaid: {CIFAR_CLASSES[_pred[_k]]} "
                      f"({_p[_k].max():.1%})", fontsize=7)
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([mo.md("**The most confident mistakes.** Look at your errors, not only "
                     "their count: some are the network's fault, some are ambiguous "
                     "32×32 images, and some may be labeling errors in the dataset."), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Calibration: can we trust the confidence?** A classifier outputs a
    softmax probability with every prediction. It is **calibrated** if, among
    all predictions made with 80% confidence, about 80% are correct. This
    matters whenever a decision depends on the confidence (send uncertain
    cases to a human; combine the classifier with other evidence).

    The **reliability diagram** bins test predictions by confidence and plots
    the actual accuracy in each bin; a calibrated model lies on the diagonal.
    The **expected calibration error** (ECE) is the average gap, weighted by
    how many predictions fall in each bin. Modern deep networks tend to be
    over-confident (Guo et al., ICML 2017). Their simple fix is **temperature
    scaling**: divide all logits by one number $T$, chosen to minimize the
    cross-entropy on the *validation* set. Dividing by $T$ does not change
    which class has the largest logit, so accuracy is unchanged.
    """)
    return


@app.cell
def _(cifar_y, ft_test_logits, ft_val_logits, mo, np, plt, split_idx):
    def _softmax(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def _nll(z, y):
        return -np.log(_softmax(z)[np.arange(len(y)), y] + 1e-12).mean()

    def _reliability(p, y, bins=10):
        conf, correct = p.max(1), p.argmax(1) == y
        edges = np.linspace(0, 1, bins + 1)
        centers, accs, counts = [], [], []
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                centers.append(conf[m].mean())
                accs.append(correct[m].mean())
                counts.append(m.sum())
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return np.array(centers), np.array(accs), np.array(counts), ece

    _yv, _yt = cifar_y[split_idx["val"]], cifar_y[split_idx["test"]]
    _temps = np.linspace(0.25, 5, 96)
    _best_t = float(_temps[np.argmin([_nll(ft_val_logits / t, _yv) for t in _temps])])
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    _stats = []
    for _ax, (_title, _t) in zip(_axes, [("before: T = 1", 1.0), (f"after temperature scaling: T = {_best_t:.2f}", _best_t)]):
        _p = _softmax(ft_test_logits / _t)
        _c, _a, _n, _ece = _reliability(_p, _yt)
        _ax.plot([0, 1], [0, 1], "k:", lw=1)
        _ax.bar(_c, _a, width=0.08, color="tab:blue", alpha=0.8, label="accuracy in bin")
        _ax.plot(_c, _c, "_", color="tab:red", ms=16, mew=2, label="mean confidence")
        _ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence", title=f"{_title}\nECE = {_ece:.1%}")
        _stats.append((_t, _ece, _nll(ft_test_logits / _t, _yt), (_p.argmax(1) == _yt).mean(), _p.max(1).mean()))
    _axes[0].set_ylabel("accuracy")
    _axes[0].legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    _rows = "\n".join(f"| {t:.2f} | {acc:.1%} | {conf:.1%} | {e:.1%} | {nll:.3f} |" for t, e, nll, acc, conf in _stats)
    mo.vstack([_fig, mo.md(f"""
    | Temperature $T$ | Test accuracy | Mean confidence | ECE | Test cross-entropy |
    |---|---|---|---|---|
    {_rows}

    $T$ was chosen on the validation set; the table reports the test set. A
    temperature above 1 means the network was over-confident and its probabilities
    are softened; below 1, under-confident. Accuracy does not move.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. What is the network looking at?

    A correct label can come from the wrong evidence: a "boat" classifier
    that has learned to detect water, or a "husky" classifier that detects
    snow. **Grad-CAM** (Selvaraju et al., ICCV 2017) gives a quick check. Take
    the last convolutional feature maps $A_k$ ($7\times7$ for ResNet-18), weight
    each map by how much the chosen class score $y_c$ increases with it, and
    keep the positive part:

    $$\text{CAM}_c = \operatorname{ReLU}\Big(\sum_k \alpha_k A_k\Big), \qquad
    \alpha_k = \frac{1}{49}\sum_{i,j}\frac{\partial y_c}{\partial A_k(i,j)}.$$

    The map is coarse ($7\times7$ upsampled to the image), but it shows
    *where* the evidence for a class lies. Pick a class from the network's
    top 5 and compare the maps (on `coffee`, try "espresso" against "cup").
    Then try `astronaut`: ImageNet has no astronaut class, so the network's
    best guesses are weak, and Grad-CAM shows what each guess was based on.
    """)
    return


@app.cell
def _(torchvision):
    cam_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    cam_model = torchvision.models.resnet18(weights=cam_weights).eval()
    return cam_model, cam_weights


@app.cell
def _(mo):
    cam_image = mo.ui.dropdown(["coffee", "chelsea", "raccoon", "astronaut"], value="coffee", label="Image")
    cam_image
    return (cam_image,)


@app.cell
def _(IMAGES_DIR, Image, cam_image, cam_model, cam_weights, mo, torch):
    cam_input = cam_weights.transforms()(Image.open(IMAGES_DIR / f"{cam_image.value}.png").convert("RGB"))[None]
    with torch.no_grad():
        _top = cam_model(cam_input).softmax(1)[0].topk(5)
    _names = cam_weights.meta["categories"]
    _options = {f"{_names[i]} ({p:.0%})": int(i) for p, i in zip(_top.values.tolist(), _top.indices.tolist())}
    cam_class = mo.ui.dropdown(_options, value=next(iter(_options)), label="Explain the class")
    cam_class
    return cam_class, cam_input


@app.cell
def _(F, cam_class, cam_input, cam_model, plt, torch):
    _store = {}
    _hook = cam_model.layer4.register_forward_hook(lambda m, i, o: _store.__setitem__("A", o))
    _logits = cam_model(cam_input)
    _hook.remove()
    _A = _store["A"]
    _grad = torch.autograd.grad(_logits[0, cam_class.value], _A)[0]
    _cam = F.relu((_grad.mean((2, 3), keepdim=True) * _A).sum(1, keepdim=True)).detach()
    _cam = F.interpolate(_cam, size=cam_input.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    _cam = (_cam / (_cam.max() + 1e-8)).numpy()
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    _img = (cam_input[0] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    _axes[0].imshow(_img)
    _axes[0].set_title("network input (224×224 center crop)", fontsize=9)
    _axes[1].imshow(_img)
    _axes[1].imshow(_cam, cmap="jet", alpha=0.45)
    _axes[1].set_title(f"Grad-CAM for “{cam_class.selected_key}”", fontsize=9)
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary and what comes next

    - **Benchmarks** (MNIST, CIFAR, ImageNet) made progress measurable, but a
      score is a measurement on one test set; it moves when the test set does.
    - **The recipe matters as much as the architecture.** The same ResNet-50
      gains almost five ImageNet points from a better recipe. In our
      experiment, BatchNorm made the network *trainable*, and augmentation
      made it *generalize* (it shrank the train–validation gap).
    - **Start from a pretrained network.** A linear probe is a strong
      baseline; fine-tuning is better, *if* the learning rate is small.
    - **Evaluate honestly:** choose everything on the validation set, report
      the test set once, and look past top-1 accuracy at per-class accuracy,
      the confusion matrix, the actual mistakes, calibration, and where the
      evidence lies (Grad-CAM).

    **Lecture 24** moves from *what* is in the image to *what is where*:
    **object detection**, from sliding windows and R-CNN to one-stage
    detectors such as YOLO.

    **Reading:** Szeliski, *Computer Vision: Algorithms and Applications* (2nd
    ed.), §6.2.
    """)
    return


if __name__ == "__main__":
    app.run()
