# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pillow",
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
    import functools
    import os
    import tempfile
    import time
    import urllib.request

    import marimo as mo
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    from sklearn.datasets import load_digits

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return (
        F,
        Image,
        functools,
        load_digits,
        mo,
        nn,
        np,
        os,
        patches,
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
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "ece491_l24_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR, IMAGES_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Lecture 24: Object Detection

    A classifier answers *what* is in an image with one label. A detector
    answers *what is where*: its output is a list of boxes, each with a class
    and a confidence score, and the length of the list changes from image to
    image (Szeliski §6.3). Three ideas carry most of the field:

    1. **Intersection over union (IoU)** measures how well two boxes agree. It
       decides which predictions count as correct, and which duplicates to
       remove.
    2. **Dense prediction plus non-maximum suppression (NMS).** Every modern
       detector proposes far more boxes than there are objects, then keeps
       the best box in each cluster of overlapping ones.
    3. **Two-stage vs. one-stage.** R-CNN-style detectors first propose
       regions and then classify them; YOLO-style detectors predict boxes
       and classes in one pass over a grid. The trade-off is speed against
       accuracy.

    This notebook measures each on the CPU: pretrained torchvision detectors
    on real photos (Sections 2–3), then a small YOLO-style detector that we
    write, train in under a minute, and evaluate with mean average precision
    (Sections 4–5). Section 2 downloads a 74 MB detector; Sections 4–5 wait
    for a button.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Boxes and intersection over union

    A box is stored as its corners $(x_1, y_1, x_2, y_2)$. The agreement
    between a predicted box $B_p$ and a ground-truth box $B_g$ is

    $$\text{IoU}(B_p, B_g) = \frac{\text{area}(B_p \cap B_g)}{\text{area}(B_p \cup B_g)} \in [0, 1].$$

    A detection usually counts as correct when IoU $\ge 0.5$ (PASCAL VOC).
    COCO averages over thresholds from 0.5 to 0.95. Move the second box and
    see how quickly IoU falls: for two equal boxes shifted sideways by $s$,
    IoU $= (w-s)/(w+s)$, so IoU $\ge 0.5$ tolerates a shift of a third of the
    box width, and IoU $\ge 0.75$ only a seventh.
    """)
    return


@app.cell
def _(mo):
    iou_dx = mo.ui.slider(-60, 60, value=20, label="Shift x")
    iou_dy = mo.ui.slider(-60, 60, value=0, label="Shift y")
    iou_scale = mo.ui.slider(0.5, 2.0, value=1.0, step=0.05, label="Scale")
    mo.hstack([iou_dx, iou_dy, iou_scale], justify="start")
    return iou_dx, iou_dy, iou_scale


@app.cell
def _(np):
    def box_area(b):
        b = np.asarray(b, float)
        return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

    def box_iou(a, b):
        """IoU between every box in a (N,4) and every box in b (M,4); returns (N,M)."""
        a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
        return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter)

    def nms(boxes, scores, iou_threshold):
        """Greedy non-maximum suppression: keep the best box, drop boxes that overlap it, repeat."""
        order = np.argsort(-np.asarray(scores))
        keep = []
        while order.size:
            best = order[0]
            keep.append(best)
            rest = order[1:]
            order = rest[box_iou(boxes[best], boxes[rest])[0] <= iou_threshold]
        return np.array(keep, int)

    def batched_nms(boxes, scores, classes, iou_threshold):
        """NMS separately within each class: a cat box never suppresses a dog box."""
        keep = [np.flatnonzero(classes == c)[nms(boxes[classes == c], scores[classes == c], iou_threshold)]
                for c in np.unique(classes)]
        keep = np.concatenate(keep) if keep else np.array([], int)
        return keep[np.argsort(-scores[keep])]
    return batched_nms, box_iou


@app.cell
def _(box_iou, iou_dx, iou_dy, iou_scale, mo, np, patches, plt, torch, torchvision):
    _g = np.array([60, 50, 160, 130], float)
    _w, _h = (_g[2] - _g[0]) * iou_scale.value, (_g[3] - _g[1]) * iou_scale.value
    _cx, _cy = (_g[0] + _g[2]) / 2 + iou_dx.value, (_g[1] + _g[3]) / 2 + iou_dy.value
    _p = np.array([_cx - _w / 2, _cy - _h / 2, _cx + _w / 2, _cy + _h / 2])
    _iou = box_iou(_g, _p)[0, 0]
    _check = torchvision.ops.box_iou(torch.tensor(_g[None]), torch.tensor(_p[None]))[0, 0].item()
    _fig, _ax = plt.subplots(figsize=(4.6, 3.4))
    _lt, _rb = np.maximum(_g[:2], _p[:2]), np.minimum(_g[2:], _p[2:])
    if (_rb > _lt).all():
        _ax.add_patch(patches.Rectangle(_lt, *(_rb - _lt), color="tab:purple", alpha=0.3, label="intersection"))
    _ax.add_patch(patches.Rectangle(_g[:2], *(_g[2:] - _g[:2]), fill=False, ec="tab:green", lw=2, label="ground truth"))
    _ax.add_patch(patches.Rectangle(_p[:2], *(_p[2:] - _p[:2]), fill=False, ec="tab:red", lw=2, ls="--", label="prediction"))
    _ax.set(xlim=(-20, 260), ylim=(200, -20), aspect="equal",
            title=f"IoU = {_iou:.3f}  ({'correct' if _iou >= 0.5 else 'miss'} at 0.5, "
                  f"{'correct' if _iou >= 0.75 else 'miss'} at 0.75)")
    _ax.legend(fontsize=7, loc="lower right")
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(f"Our `box_iou`: **{_iou:.4f}**  \n`torchvision.ops.box_iou`: **{_check:.4f}**")],
              justify="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. A pretrained detector, before and after non-maximum suppression

    Below is **Faster R-CNN** (Ren et al., NeurIPS 2015) with a MobileNetV3
    backbone, trained on the 80 object classes of MS-COCO and downloaded from
    torchvision. We switched off its built-in NMS so that you can see what the
    network really outputs: several boxes for every object, one from each
    region proposal that overlaps it, each slightly different.

    **Non-maximum suppression** cleans this up greedily: sort the boxes by
    score; keep the best one; delete every remaining box of the same class
    whose IoU with it exceeds a threshold; repeat with what is left. Our
    20-line `nms` above is checked against `torchvision.ops.batched_nms`.

    Try the threshold. Too low (0.1–0.2) and NMS deletes real neighbors (two
    people standing close together become one). Too high (0.9) and duplicates
    survive. Detectors usually use 0.5–0.7.
    """)
    return


@app.cell
def _(CACHE_DIR, IMAGES_DIR, Image, os, urllib):
    COCO_IMAGES = {
        "living room (COCO 139)": "000000000139",
        "tennis team (COCO 1000)": "000000001000",
        "two cats (COCO 39769)": "000000039769",
        "baseball (COCO 2153)": "000000002153",
    }

    def load_photo(name):
        """COCO val2017 photos are downloaded on first use; the local test images always work."""
        if name in COCO_IMAGES:
            path = os.path.join(CACHE_DIR, COCO_IMAGES[name] + ".jpg")
            if not os.path.exists(path):
                try:
                    urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{COCO_IMAGES[name]}.jpg", path)
                except OSError:
                    return Image.open(IMAGES_DIR / "astronaut.png").convert("RGB")
            return Image.open(path).convert("RGB")
        return Image.open(IMAGES_DIR / f"{name}.png").convert("RGB")

    PHOTO_CHOICES = list(COCO_IMAGES) + ["astronaut", "coffee", "chelsea"]
    return PHOTO_CHOICES, load_photo


@app.cell
def _(torchvision):
    frcnn_weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    COCO_NAMES = frcnn_weights.meta["categories"]
    # box_nms_thresh=1.0 switches the built-in NMS off; we apply our own below.
    frcnn_raw = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=frcnn_weights, box_nms_thresh=1.0, box_detections_per_img=1000, box_score_thresh=0.05).eval()
    return COCO_NAMES, frcnn_raw, frcnn_weights


@app.cell
def _(PHOTO_CHOICES, mo):
    nms_photo = mo.ui.dropdown(PHOTO_CHOICES, value=PHOTO_CHOICES[0], label="Photo")
    nms_iou = mo.ui.slider(0.1, 0.95, value=0.5, step=0.05, label="NMS IoU threshold")
    nms_score = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="Score threshold")
    mo.hstack([nms_photo, nms_iou, nms_score], justify="start")
    return nms_iou, nms_photo, nms_score


@app.cell
def _(frcnn_raw, frcnn_weights, load_photo, nms_photo, torch):
    nms_pil = load_photo(nms_photo.value)
    with torch.no_grad():
        _out = frcnn_raw([frcnn_weights.transforms()(nms_pil)])[0]
    raw_boxes = _out["boxes"].numpy()
    raw_scores = _out["scores"].numpy()
    raw_labels = _out["labels"].numpy()
    return nms_pil, raw_boxes, raw_labels, raw_scores


@app.cell
def _(
    COCO_NAMES,
    batched_nms,
    mo,
    nms_iou,
    nms_pil,
    nms_score,
    np,
    patches,
    plt,
    raw_boxes,
    raw_labels,
    raw_scores,
    torch,
    torchvision,
):
    _m = raw_scores >= nms_score.value
    _b, _s, _l = raw_boxes[_m], raw_scores[_m], raw_labels[_m]
    _keep = batched_nms(_b, _s, _l, nms_iou.value)
    _ref = torchvision.ops.batched_nms(torch.tensor(_b), torch.tensor(_s), torch.tensor(_l), nms_iou.value).numpy()
    _same = set(_keep.tolist()) == set(_ref.tolist())
    _colors = plt.cm.tab20(np.linspace(0, 1, 20))
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for _ax, _idx, _title in [(_axes[0], np.arange(len(_b)), f"before NMS: {len(_b)} boxes"),
                              (_axes[1], _keep, f"after NMS: {len(_keep)} boxes")]:
        _ax.imshow(nms_pil)
        for _i in _idx:
            _x1, _y1, _x2, _y2 = _b[_i]
            _c = _colors[_l[_i] % 20]
            _ax.add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, fill=False, ec=_c, lw=1.2))
            if _ax is _axes[1]:
                _ax.text(_x1, _y1, f"{COCO_NAMES[_l[_i]]} {_s[_i]:.2f}", fontsize=7, color="white",
                         bbox=dict(facecolor=_c, alpha=0.8, pad=1, lw=0), va="bottom")
        _ax.set_title(_title, fontsize=10)
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"*Boxes with score ≥ {nms_score.value:.2f}. Our NMS keeps the same boxes as "
        f"`torchvision.ops.batched_nms`: **{_same}**. Photos: MS-COCO val2017 (Lin et al., "
        f"ECCV 2014), downloaded at run time from cocodataset.org; each keeps its original "
        f"Flickr license.*")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Two-stage vs. one-stage: the speed–accuracy trade-off

    - A **two-stage** detector (R-CNN → Fast R-CNN → Faster R-CNN) first
      proposes a few hundred regions that might contain an object (the
      region proposal network, RPN), then crops features for each region
      (RoIAlign) and classifies it and refines its box.
    - A **one-stage** detector (YOLO, Redmon et al., CVPR 2016; SSD; RetinaNet)
      skips the proposals. Every cell of a feature-map grid directly predicts
      boxes and class scores, in one forward pass, followed by NMS.

    torchvision ships detectors of both kinds. The table lists their
    documented COCO accuracy (box mAP, averaged over IoU 0.5–0.95; Section 5
    explains it) and size. Select one to download it, run it on a photo, and
    time it on this machine's CPU.
    """)
    return


@app.cell
def _(mo, torchvision):
    _D = torchvision.models.detection
    DETECTORS = {
        "SSDlite320 MobileNetV3 (one-stage)": (_D.ssdlite320_mobilenet_v3_large, _D.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
        "Faster R-CNN MobileNetV3-FPN (two-stage)": (_D.fasterrcnn_mobilenet_v3_large_fpn, _D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
        "RetinaNet ResNet-50-FPN v2 (one-stage)": (_D.retinanet_resnet50_fpn_v2, _D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
        "Faster R-CNN ResNet-50-FPN v2 (two-stage)": (_D.fasterrcnn_resnet50_fpn_v2, _D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    }
    _lines = ["| Detector | COCO box mAP | Parameters | GFLOPs / image | Download |", "|---|---|---|---|---|"]
    for _name, (_fn, _w) in DETECTORS.items():
        _m = _w.meta
        _lines.append(f"| {_name} | {_m['_metrics']['COCO-val2017']['box_map']:.1f} | "
                      f"{_m['num_params'] / 1e6:.1f} M | {_m['_ops']:.1f} | {_m['_file_size']:.0f} MB |")
    mo.md("\n".join(_lines) + "\n\n*Source: torchvision weight metadata (COCO val2017).*")
    return (DETECTORS,)


@app.cell
def _(DETECTORS, PHOTO_CHOICES, mo):
    det_choice = mo.ui.dropdown(list(DETECTORS), value="SSDlite320 MobileNetV3 (one-stage)", label="Detector")
    det_photo = mo.ui.dropdown(PHOTO_CHOICES, value=PHOTO_CHOICES[1], label="Photo")
    det_score = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="Score threshold")
    mo.hstack([det_choice, det_photo, det_score], justify="start")
    return det_choice, det_photo, det_score


@app.cell
def _(DETECTORS, functools):
    @functools.lru_cache(maxsize=4)
    def get_detector(name):
        fn, weights = DETECTORS[name]
        return fn(weights=weights).eval(), weights
    return (get_detector,)


@app.cell
def _(det_choice, det_photo, get_detector, load_photo, time, torch):
    _det_model, det_weights = get_detector(det_choice.value)
    det_pil = load_photo(det_photo.value)
    _x = det_weights.transforms()(det_pil)
    with torch.no_grad():
        _det_model([_x])                      # warm-up run
        _t0 = time.perf_counter()
        det_out = _det_model([_x])[0]
        det_ms = 1000 * (time.perf_counter() - _t0)
    return det_ms, det_out, det_pil, det_weights


@app.cell
def _(
    det_choice,
    det_ms,
    det_out,
    det_pil,
    det_score,
    det_weights,
    mo,
    np,
    patches,
    plt,
):
    _names = det_weights.meta["categories"]
    _m = det_out["scores"].numpy() >= det_score.value
    _colors = plt.cm.tab20(np.linspace(0, 1, 20))
    _fig, _ax = plt.subplots(figsize=(7.5, 5))
    _ax.imshow(det_pil)
    for (_x1, _y1, _x2, _y2), _s, _l in zip(det_out["boxes"].numpy()[_m], det_out["scores"].numpy()[_m],
                                            det_out["labels"].numpy()[_m]):
        _c = _colors[_l % 20]
        _ax.add_patch(patches.Rectangle((_x1, _y1), _x2 - _x1, _y2 - _y1, fill=False, ec=_c, lw=1.5))
        _ax.text(_x1, _y1, f"{_names[_l]} {_s:.2f}", fontsize=7, color="white",
                 bbox=dict(facecolor=_c, alpha=0.8, pad=1, lw=0), va="bottom")
    _ax.axis("off")
    _ax.set_title(f"{det_choice.value}: {int(_m.sum())} objects, {det_ms:.0f} ms on this CPU", fontsize=10)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "Compare the four on the *tennis team* and *living room* photos. The small "
        "one-stage SSDlite is several times faster but misses small and partly hidden "
        "objects; the large two-stage Faster R-CNN finds the most and is the slowest. "
        "Recent YOLO versions (not in torchvision) report COCO accuracy close to the "
        "best two-stage detectors at one-stage speed, using larger backbones, "
        "multi-scale features and long training recipes.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. A YOLO-style detector from scratch

    The core of YOLO fits in a page. To train one on a laptop in under a
    minute, we use a toy task: **find and read every handwritten digit** in a
    $64\times64$ image that holds one to four digits of different sizes
    (8×8 scikit-learn digits, enlarged to 12–28 pixels).

    **Grid encoding.** Divide the image into an $8\times8$ grid of cells,
    $8\times8$ pixels each. The cell that contains an object's center is
    *responsible* for it and must predict 15 numbers:

    - **objectness** $o$: is there an object centered in this cell?
    - **box** $(t_x, t_y, t_w, t_h)$: the center as an offset inside the cell,
      $(\sigma(t_x), \sigma(t_y)) \in (0,1)^2$, and the size relative to a
      16-pixel reference box, $w = 16\,e^{t_w}$, $h = 16\,e^{t_h}$
      (YOLOv2's parametrization);
    - **class** scores for the 10 digits.

    **Network.** A fully convolutional network with three $2\times2$
    max-pools turns the $64\times64$ image into an $8\times8$ map with 15
    channels: one prediction vector per cell, in one forward pass.

    **Loss.** Binary cross-entropy for objectness in *every* cell (most cells
    are empty, and must learn to say so), plus an L1 box loss and a
    cross-entropy class loss in the responsible cells only.

    **Inference.** Score = objectness × class probability; drop low scores;
    run NMS. These are the three steps of the original YOLO; later versions
    add anchor boxes, several scales and many training tricks, but keep this
    structure.
    """)
    return


@app.cell
def _(F, load_digits, np, torch):
    CANVAS, GRID, STRIDE, N_CLASSES = 64, 8, 8, 10
    _digits = load_digits()
    DIGIT_IMAGES, DIGIT_LABELS = _digits.images / 16.0, _digits.target
    _perm = np.random.default_rng(0).permutation(len(DIGIT_IMAGES))
    DIGIT_POOLS = {"train": _perm[:1400], "test": _perm[1400:]}   # test scenes use digits never seen in training

    def make_scene(rng, pool):
        """Paste 1-4 digits (12-28 px) at random positions on a noisy canvas; return image, boxes, labels."""
        img = rng.normal(0, 0.05, (CANVAS, CANVAS)).astype(np.float32)
        boxes, labels = [], []
        for _ in range(rng.integers(1, 5)):
            for _attempt in range(20):
                k, s = pool[rng.integers(len(pool))], int(rng.integers(12, 29))
                glyph = F.interpolate(torch.tensor(DIGIT_IMAGES[k])[None, None].float(), size=(s, s),
                                      mode="bilinear", align_corners=False)[0, 0].numpy()
                ys, xs = np.nonzero(glyph > 0.15)
                x0, y0 = rng.integers(0, CANVAS - s + 1, 2)
                box = [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]   # tight box
                if all(overlap(box, b) < 0.1 for b in boxes):
                    img[y0:y0 + s, x0:x0 + s] = np.maximum(img[y0:y0 + s, x0:x0 + s], glyph)
                    boxes.append(box)
                    labels.append(int(DIGIT_LABELS[k]))
                    break
        return img, np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, int)

    def overlap(a, b):
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih)

    def encode_targets(boxes, labels):
        """(GRID, GRID, 6) target: objectness, t_x, t_y, t_w, t_h, class, in the responsible cell."""
        t = torch.zeros(GRID, GRID, 6)
        area = np.zeros((GRID, GRID))
        for (x1, y1, x2, y2), c in zip(boxes.tolist(), labels.tolist()):
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            i, j = int(cy // STRIDE), int(cx // STRIDE)
            if w * h > area[i, j]:          # two centers in one cell: the larger object wins
                area[i, j] = w * h
                t[i, j] = torch.tensor([1.0, cx / STRIDE - j, cy / STRIDE - i,
                                        float(np.log(w / 16)), float(np.log(h / 16)), float(c)])
        return t

    def make_dataset(n, split, seed):
        rng = np.random.default_rng(seed)
        return [make_scene(rng, DIGIT_POOLS[split]) for _ in range(n)]
    return (
        CANVAS,
        GRID,
        N_CLASSES,
        STRIDE,
        encode_targets,
        make_dataset,
    )


@app.cell
def _(make_dataset):
    yolo_train_set = make_dataset(3000, "train", seed=1)
    yolo_test_set = make_dataset(500, "test", seed=2)
    return yolo_test_set, yolo_train_set


@app.cell
def _(CANVAS, GRID, STRIDE, mo, patches, plt, yolo_train_set):
    _img, _boxes, _labels = yolo_train_set[3]
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    for _ax in _axes:
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=1)
        _ax.set_xticks([])
        _ax.set_yticks([])
    for _k in range(1, GRID):
        _axes[1].axhline(_k * STRIDE - 0.5, color="w", lw=0.4, alpha=0.6)
        _axes[1].axvline(_k * STRIDE - 0.5, color="w", lw=0.4, alpha=0.6)
    for (_x1, _y1, _x2, _y2), _c in zip(_boxes, _labels):
        _cx, _cy = (_x1 + _x2) / 2, (_y1 + _y2) / 2
        _i, _j = int(_cy // STRIDE), int(_cx // STRIDE)
        _axes[1].add_patch(patches.Rectangle((_j * STRIDE - 0.5, _i * STRIDE - 0.5), STRIDE, STRIDE,
                                             color="tab:orange", alpha=0.55))
        for _ax in _axes:
            _ax.add_patch(patches.Rectangle((_x1 - 0.5, _y1 - 0.5), _x2 - _x1, _y2 - _y1, fill=False, ec="lime", lw=1.5))
        _axes[1].plot(_cx - 0.5, _cy - 0.5, "r.", ms=6)
        _axes[0].text(_x1, _y1 - 1, str(_c), color="lime", fontsize=9, va="bottom")
    _axes[0].set_title(f"a {CANVAS}×{CANVAS} training scene and its labels", fontsize=9)
    _axes[1].set_title(f"{GRID}×{GRID} grid: the cell holding each center (red) is responsible", fontsize=9)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"*{len(yolo_train_set):,} training scenes are generated on the fly; "
                           "the test scenes use digits that never appear in training.*")])
    return


@app.cell
def _(F, N_CLASSES, nn, torch):
    class TinyYOLO(nn.Module):
        """64x64 image -> (8, 8, 5 + classes): objectness, box, class scores per grid cell."""

        def __init__(self, width=16):
            super().__init__()

            def block(cin, cout):
                return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1)]
            w = width
            self.body = nn.Sequential(
                *block(1, w), nn.MaxPool2d(2), *block(w, 2 * w), nn.MaxPool2d(2),
                *block(2 * w, 4 * w), nn.MaxPool2d(2), *block(4 * w, 4 * w), *block(4 * w, 4 * w))
            self.head = nn.Conv2d(4 * w, 5 + N_CLASSES, 1)

        def forward(self, x):
            return self.head(self.body(x)).permute(0, 2, 3, 1)

    def yolo_loss(pred, target):
        pos = target[..., 0] > 0
        obj = F.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0])
        box = F.l1_loss(torch.cat([pred[..., 1:3].sigmoid(), pred[..., 3:5]], -1)[pos], target[..., 1:5][pos])
        cls = F.cross_entropy(pred[..., 5:][pos], target[..., 5][pos].long())
        return 5 * obj + 2 * box + cls
    return TinyYOLO, yolo_loss


@app.cell
def _(mo):
    yolo_run = mo.ui.run_button(label="Train the detector (about 1 minute)")
    yolo_run
    return (yolo_run,)


@app.cell
def _(
    CACHE_DIR,
    TinyYOLO,
    encode_targets,
    mo,
    np,
    os,
    time,
    torch,
    yolo_loss,
    yolo_run,
    yolo_train_set,
):
    mo.stop(not yolo_run.value, mo.md("*Click the button above to train the detector (Sections 4–5).*"))
    yolo_model = TinyYOLO(width=16)
    _path = os.path.join(CACHE_DIR, "tinyyolo_v1.pt")
    if os.path.exists(_path):
        _state = torch.load(_path)
        yolo_model.load_state_dict(_state["model"])
        yolo_log = _state["log"]
    else:
        torch.manual_seed(0)
        _X = torch.tensor(np.stack([s[0] for s in yolo_train_set]))[:, None]
        _T = torch.stack([encode_targets(s[1], s[2]) for s in yolo_train_set])
        _epochs, _batch = 10, 64
        _opt = torch.optim.AdamW(yolo_model.parameters(), 2e-3, weight_decay=1e-4)
        _sched = torch.optim.lr_scheduler.OneCycleLR(_opt, 2e-3, total_steps=_epochs * (len(_X) // _batch))
        yolo_log = {"loss": [], "seconds": 0.0}
        _t0 = time.time()
        for _ep in range(_epochs):
            yolo_model.train()
            _perm = torch.randperm(len(_X))
            _losses = []
            for _k in range(0, len(_X) - _batch + 1, _batch):
                _b = _perm[_k:_k + _batch]
                _loss = yolo_loss(yolo_model(_X[_b]), _T[_b])
                _opt.zero_grad()
                _loss.backward()
                _opt.step()
                _sched.step()
                _losses.append(_loss.item())
            yolo_log["loss"].append(float(np.mean(_losses)))
        yolo_log["seconds"] = time.time() - _t0
        torch.save({"model": yolo_model.state_dict(), "log": yolo_log}, _path)
    yolo_model.eval()
    return (yolo_model,)


@app.cell
def _(GRID, STRIDE, batched_nms, np, torch):
    def yolo_decode(pred, score_threshold=0.01, iou_threshold=0.5, use_nms=True):
        """Grid predictions (B,S,S,15) -> per image (boxes, scores, classes) in pixels."""
        jj, ii = np.meshgrid(np.arange(GRID), np.arange(GRID))
        results = []
        for q in pred.detach().numpy():
            obj = 1 / (1 + np.exp(-q[..., 0]))
            cls = np.exp(q[..., 5:] - q[..., 5:].max(-1, keepdims=True))
            cls /= cls.sum(-1, keepdims=True)
            scores, classes = (obj[..., None] * cls).max(-1), cls.argmax(-1)
            cx = (jj + 1 / (1 + np.exp(-q[..., 1]))) * STRIDE
            cy = (ii + 1 / (1 + np.exp(-q[..., 2]))) * STRIDE
            w, h = 16 * np.exp(q[..., 3]), 16 * np.exp(q[..., 4])
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).reshape(-1, 4)
            scores, classes = scores.ravel(), classes.ravel()
            m = scores >= score_threshold
            boxes, scores, classes = boxes[m], scores[m], classes[m]
            keep = batched_nms(boxes, scores, classes, iou_threshold) if use_nms else np.argsort(-scores)
            results.append((boxes[keep], scores[keep], classes[keep]))
        return results

    def yolo_predict(model, scenes):
        with torch.no_grad():
            return model(torch.tensor(np.stack([s[0] for s in scenes]))[:, None])
    return yolo_decode, yolo_predict


@app.cell
def _(mo):
    yolo_score = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="Score threshold")
    yolo_page = mo.ui.slider(0, 9, value=0, label="Test scenes (page)")
    mo.hstack([yolo_score, yolo_page], justify="start")
    return yolo_page, yolo_score


@app.cell
def _(
    mo,
    patches,
    plt,
    yolo_decode,
    yolo_model,
    yolo_page,
    yolo_predict,
    yolo_score,
    yolo_test_set,
):
    _scenes = yolo_test_set[6 * yolo_page.value: 6 * yolo_page.value + 6]
    _dets = yolo_decode(yolo_predict(yolo_model, _scenes), score_threshold=yolo_score.value)
    _fig, _axes = plt.subplots(1, 6, figsize=(12, 2.4))
    for _ax, (_img, _gb, _gl), (_b, _s, _c) in zip(_axes, _scenes, _dets):
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=1)
        for _x1, _y1, _x2, _y2 in _gb:
            _ax.add_patch(patches.Rectangle((_x1 - 0.5, _y1 - 0.5), _x2 - _x1, _y2 - _y1, fill=False, ec="lime", lw=1, ls="--"))
        for (_x1, _y1, _x2, _y2), _si, _ci in zip(_b, _s, _c):
            _ax.add_patch(patches.Rectangle((_x1 - 0.5, _y1 - 0.5), _x2 - _x1, _y2 - _y1, fill=False, ec="red", lw=1.2))
            _ax.text(_x1, _y1 - 1, f"{_ci}:{_si:.2f}", color="red", fontsize=6.5, va="bottom")
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("*Green dashed: ground truth. Red: predictions (digit:score) after NMS. "
                           "Lower the score threshold to see the low-confidence boxes that the "
                           "evaluation in Section 5 must also account for.*")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Evaluating a detector: precision, recall and mAP

    Detection has no single "accuracy": a detector can miss objects (false
    negatives), report boxes where there is nothing (false positives), and
    report the right object with a sloppy box. The standard metric handles
    all three.

    1. Fix an IoU threshold (say 0.5). Sort all detections of one class,
       across the whole test set, by score.
    2. Walk down the list. A detection is a **true positive** if it overlaps
       a still-unmatched ground-truth box of its class with IoU at or above
       the threshold; otherwise (no overlap, or a duplicate of an already
       matched object) it is a **false positive**.
    3. After each detection, record **precision** (the fraction of detections
       so far that are correct) and **recall** (the fraction of all
       ground-truth objects found so far). This traces the precision–recall
       curve as the score threshold is lowered.
    4. **Average precision (AP)** is the area under that curve (after making
       precision non-increasing). **mAP** is the mean of AP over classes.

    PASCAL VOC reports mAP at IoU 0.5. COCO reports the average of mAP over
    the ten IoU thresholds 0.50, 0.55, …, 0.95, which rewards precise boxes;
    that is the "box mAP" in the table of Section 3.
    """)
    return


@app.cell
def _(box_iou, np):
    def average_precision(detections, ground_truth, cls, iou_threshold=0.5):
        """All-point interpolated AP for one class. detections: per image (boxes, scores, classes);
        ground_truth: per image (boxes, labels). Returns (AP, recall curve, precision curve)."""
        records, n_gt = [], 0
        for (b, s, c), (gb, gl) in zip(detections, ground_truth):
            g = gb[gl == cls]
            n_gt += len(g)
            matched = np.zeros(len(g), bool)
            for k in np.flatnonzero(c == cls)[np.argsort(-s[c == cls])]:
                tp = 0
                if len(g):
                    ious = box_iou(b[k], g)[0]
                    ious[matched] = -1
                    best = ious.argmax()
                    if ious[best] >= iou_threshold:
                        matched[best], tp = True, 1
                records.append((s[k], tp))
        if n_gt == 0:
            return float("nan"), np.array([]), np.array([])
        records.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in records])
        fp = np.cumsum([1 - r[1] for r in records])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-12)
        r = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
        p = np.concatenate([[1.0], precision, [0.0]])
        p_env = np.maximum.accumulate(p[::-1])[::-1]        # precision envelope (non-increasing)
        ap = float(np.sum((r[1:] - r[:-1]) * p_env[1:]))
        return ap, recall, precision

    def operating_point(detections, ground_truth, score_threshold, iou_threshold=0.5):
        """Precision and recall over all classes for the detections a user would actually see."""
        tp = n_det = n_gt = 0
        for (b, s, c), (gb, gl) in zip(detections, ground_truth):
            n_gt += len(gb)
            matched = np.zeros(len(gb), bool)
            for k in np.flatnonzero(s >= score_threshold)[np.argsort(-s[s >= score_threshold])]:
                n_det += 1
                ious = np.where((gl == c[k]) & ~matched, box_iou(b[k], gb)[0], -1) if len(gb) else np.array([-1.0])
                if ious.max() >= iou_threshold:
                    matched[ious.argmax()] = True
                    tp += 1
        return tp / max(n_det, 1), tp / n_gt, n_det
    return average_precision, operating_point


@app.cell
def _(
    N_CLASSES,
    average_precision,
    np,
    operating_point,
    yolo_decode,
    yolo_model,
    yolo_predict,
    yolo_test_set,
):
    _pred = yolo_predict(yolo_model, yolo_test_set)
    yolo_gt = [(s[1], s[2]) for s in yolo_test_set]
    yolo_dets = yolo_decode(_pred, score_threshold=0.01)
    _dets_no_nms = yolo_decode(_pred, score_threshold=0.01, use_nms=False)
    _thresholds = np.round(np.arange(0.5, 0.96, 0.05), 2)
    yolo_ap = {t: [average_precision(yolo_dets, yolo_gt, c, t)[0] for c in range(N_CLASSES)] for t in _thresholds}
    yolo_map = {
        "mAP@0.5": float(np.nanmean(yolo_ap[0.5])),
        "mAP@0.75": float(np.nanmean(yolo_ap[0.75])),
        "mAP@[.5:.95] (COCO)": float(np.mean([np.nanmean(v) for v in yolo_ap.values()])),
        "mAP@0.5 without NMS": float(np.nanmean([average_precision(_dets_no_nms, yolo_gt, c, 0.5)[0]
                                                 for c in range(N_CLASSES)])),
    }
    yolo_operating = {name: operating_point(d, yolo_gt, 0.5) for name, d in
                      [("with NMS", yolo_dets), ("without NMS", _dets_no_nms)]}
    return yolo_ap, yolo_dets, yolo_gt, yolo_map, yolo_operating


@app.cell
def _(mo):
    pr_class = mo.ui.slider(0, 9, value=8, label="Digit class for the PR curve")
    pr_class
    return (pr_class,)


@app.cell
def _(
    average_precision,
    mo,
    plt,
    pr_class,
    yolo_ap,
    yolo_dets,
    yolo_gt,
    yolo_map,
    yolo_operating,
):
    _fig, _ax = plt.subplots(figsize=(5, 3.6))
    for _t, _style in [(0.5, "-"), (0.75, "--"), (0.9, ":")]:
        _ap, _r, _p = average_precision(yolo_dets, yolo_gt, pr_class.value, _t)
        _ax.plot(_r, _p, _style, label=f"IoU ≥ {_t}: AP = {_ap:.3f}")
    _ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="recall", ylabel="precision",
            title=f"precision–recall for digit {pr_class.value}")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    _ap_row = " | ".join(f"{v:.2f}" for v in yolo_ap[0.5])
    _ap_row75 = " | ".join(f"{v:.2f}" for v in yolo_ap[0.75])
    _summary = "\n".join(f"| {k} | {v:.3f} |" for k, v in yolo_map.items())
    _op = "\n".join(f"| {k} | {n} | {p:.1%} | {r:.1%} |" for k, (p, r, n) in yolo_operating.items())
    mo.vstack([
        mo.hstack([_fig, mo.md(f"""
    | Metric (500 test scenes) | Value |
    |---|---|
    {_summary}
    """)], justify="start"),
        mo.md(f"""
    | digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | AP@0.5 | {_ap_row} |
    | AP@0.75 | {_ap_row75} |

    Three things to read from these numbers:

    - **Finding the digits is easy; drawing tight boxes is harder.** AP drops
      as the IoU threshold rises: many detections are correct at 0.5 but not
      precise enough for 0.75 or 0.9. COCO's mAP@[.5:.95] rewards that precision.
    - **NMS matters most for the boxes a user actually sees.** mAP barely changes
      without it: duplicates score lower than the box they duplicate, so they enter
      the ranked list late and cost precision only at the end of the curve. At a
      fixed score threshold of 0.5, though, every surviving duplicate is a false
      positive:

    | score ≥ 0.5, IoU ≥ 0.5 | Detections shown | Precision | Recall |
    |---|---|---|---|
    {_op}

    - A real benchmark is much harder than this toy (80 classes, clutter,
      occlusion, objects of a few pixels). The best COCO mAP@[.5:.95] in the
      table of Section 3 is below 0.5.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    - A detector outputs a variable-length list of (box, class, score).
      **IoU** measures box agreement; it defines a correct detection and drives
      **NMS**, which removes duplicates.
    - **Two-stage** detectors (Faster R-CNN) propose regions, then classify
      each; **one-stage** detectors (YOLO, SSD, RetinaNet) predict boxes and
      classes from a grid in one pass. One-stage is faster; two-stage has
      traditionally been more accurate, a gap that modern YOLOs have closed.
    - The YOLO recipe is a grid of cells, each responsible for the objects
      centered in it, predicting objectness, a box and class scores, trained
      with one combined loss and decoded with a score threshold and NMS.
    - Detectors are compared by **mAP**: the area under the precision–recall
      curve, averaged over classes, at one IoU threshold (VOC: 0.5) or many
      (COCO: 0.5–0.95).

    **Next (Dr. Wong):** segmentation, which labels every pixel instead of
    drawing boxes.

    **Reading:** Szeliski, *Computer Vision: Algorithms and Applications* (2nd
    ed.), §6.3.
    """)
    return


if __name__ == "__main__":
    app.run()
