# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import load_digits

    return load_digits, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Lecture 21: Machine Learning -- Introduction and Neural Networks

    Every detector and descriptor so far in this course was **designed by
    hand**: Harris corners come from an eigenvalue test someone derived
    (notebook 13), SIFT and HOG from histograms of gradient orientations
    someone engineered (Lecture 11, notebook 14), Canny from an optimality
    argument about step edges (notebook 15). Each one encodes a human's
    guess about *which numbers in an image matter*.

    This notebook starts the last part of the course, where that guess is
    **learned from examples** instead. We follow Szeliski's Chapter 5
    (§5.1 supervised learning, §5.3 deep neural networks) and build
    everything from scratch in NumPy -- no deep learning library -- so that
    every gradient is a line of code you can read:

    1. the learning recipe: data, model, loss, optimizer;
    2. **logistic regression**, a linear classifier trained by gradient descent,
       and where it breaks;
    3. a **multilayer perceptron (MLP)** with hand-written **backpropagation**,
       checked against numerical gradients;
    4. the same MLP on **real images** (8x8 handwritten digits), and one
       experiment that shows what an MLP *cannot* see -- the motivation for
       convolutional networks in Lecture 22.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The learning recipe

    Supervised learning (§5.1) is given $N$ labeled examples
    $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ -- an image (or feature vector)
    $\mathbf{x}_i$ and its class $y_i$ -- and asked for a function
    $f_\theta$ that predicts $y$ for images it has **never seen**. Every
    method in the rest of this course is the same four choices:

    | Choice | Question it answers | In this notebook |
    |---|---|---|
    | **Data** | What examples do we learn from? | 2D toy sets, then 8x8 digits |
    | **Model** $f_\theta$ | What family of functions? | linear $\to$ MLP |
    | **Loss** $\mathcal{L}(\theta)$ | How wrong is $f_\theta$ on the data? | cross-entropy |
    | **Optimizer** | How do we make it less wrong? | gradient descent |

    Gradient descent repeats one update until the loss stops decreasing:

    $$\theta \leftarrow \theta - \alpha\,\nabla_\theta \mathcal{L}(\theta),$$

    with **learning rate** (step size) $\alpha$. The whole difficulty of deep
    learning -- and the reason backpropagation exists -- is computing
    $\nabla_\theta \mathcal{L}$ efficiently when $\theta$ has millions of
    entries.

    We start in 2D, where we can *see* the function being learned.
    """)
    return


@app.cell
def _(np):
    def make_dataset(name, n_per_class=150, noise=0.1, seed=0):
        """Toy 2-class datasets in roughly [-1.5, 1.5]^2. Returns X (N,2), y (N,) in {0,1}."""
        rng = np.random.default_rng(seed)
        n = n_per_class
        if name == "Two blobs":
            X0 = rng.normal([-0.7, -0.5], 0.35 + noise, (n, 2))
            X1 = rng.normal([0.7, 0.5], 0.35 + noise, (n, 2))
        elif name == "XOR":
            centers = np.array([[-0.7, -0.7], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7]])
            k = n // 2
            X0 = np.vstack([rng.normal(c, 0.2 + noise, (k, 2)) for c in centers[:2]])
            X1 = np.vstack([rng.normal(c, 0.2 + noise, (k, 2)) for c in centers[2:]])
        elif name == "Circles":
            a0, a1 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
            X0 = 0.45 * np.c_[np.cos(a0), np.sin(a0)] + rng.normal(0, noise, (n, 2))
            X1 = 1.10 * np.c_[np.cos(a1), np.sin(a1)] + rng.normal(0, noise, (n, 2))
        elif name == "Two spirals":
            t = np.sqrt(rng.uniform(0.02, 1, n)) * 3 * np.pi
            arm = np.c_[t * np.cos(t), t * np.sin(t)] / (2 * np.pi)
            X0 = arm + rng.normal(0, noise, (n, 2))
            X1 = -arm + rng.normal(0, noise, (n, 2))
        else:
            raise ValueError(name)
        X = np.vstack([X0, X1])
        y = np.r_[np.zeros(len(X0), int), np.ones(len(X1), int)]
        return X, y

    def plot_boundary(ax, prob_fn, X, y, title="", lim=1.8):
        """Shade P(y=1|x) over a grid, draw the 0.5 contour, scatter the data."""
        g = np.linspace(-lim, lim, 200)
        gx, gy = np.meshgrid(g, g)
        p = prob_fn(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
        ax.contour(gx, gy, p, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(*X[y == 0].T, s=10, c="tab:blue", edgecolors="none", label="class 0")
        ax.scatter(*X[y == 1].T, s=10, c="tab:red", edgecolors="none", label="class 1")
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", title=title)
        ax.set_xticks([])
        ax.set_yticks([])

    return make_dataset, plot_boundary


@app.cell
def _(mo):
    dataset_choice = mo.ui.dropdown(
        ["Two blobs", "XOR", "Circles", "Two spirals"], value="Circles", label="Dataset"
    )
    noise_slider = mo.ui.slider(0.0, 0.3, value=0.08, step=0.02, label="Noise")
    mo.hstack([dataset_choice, noise_slider], justify="start")
    return dataset_choice, noise_slider


@app.cell
def _(dataset_choice, make_dataset, mo, noise_slider, plt):
    X2d, y2d = make_dataset(dataset_choice.value, noise=noise_slider.value)
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.scatter(*X2d[y2d == 0].T, s=10, c="tab:blue", label="class 0")
    _ax.scatter(*X2d[y2d == 1].T, s=10, c="tab:red", label="class 1")
    _ax.set(aspect="equal", title=f"{dataset_choice.value}: {len(y2d)} points")
    _ax.legend(loc="upper right", fontsize=8)
    mo.vstack([
        _fig,
        mo.md("*Each point is one training example $\\mathbf{x}_i\\in\\mathbb{R}^2$ "
              "with label $y_i\\in\\{0,1\\}$. In an image classifier, $\\mathbf{x}_i$ "
              "would be thousands of pixels; the math below does not change.*"),
    ])
    return X2d, y2d


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Logistic regression: a linear classifier you train by descent

    The simplest model scores each input with a **linear function** and
    squashes the score into a probability with the **sigmoid**
    $\sigma(z)=1/(1+e^{-z})$:

    $$p_i = P(y_i=1\mid\mathbf{x}_i) = \sigma(\mathbf{w}^\top\mathbf{x}_i + b).$$

    The decision boundary $p=0.5$ is the line $\mathbf{w}^\top\mathbf{x}+b=0$.
    We fit $(\mathbf{w}, b)$ by **maximum likelihood**: make the observed
    labels as probable as possible, which (after a log, a minus sign, and
    averaging) is the **binary cross-entropy** loss

    $$\mathcal{L}(\mathbf{w},b) = -\frac{1}{N}\sum_{i=1}^N \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big].$$

    Its gradient has a strikingly simple form -- *prediction minus truth*,
    weighted by the input:

    $$\nabla_{\mathbf{w}}\mathcal{L} = \frac{1}{N}\sum_i (p_i - y_i)\,\mathbf{x}_i,
      \qquad \frac{\partial\mathcal{L}}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i).$$

    This loss is **convex** in $(\mathbf{w},b)$, so gradient descent finds the
    global minimum. Try it below on each dataset, then tick the box to add one
    **hand-crafted feature**, $x_1^2+x_2^2$ (the squared radius).
    """)
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def radial_features(X, use_radius):
        return np.c_[X, (X**2).sum(1)] if use_radius else X

    def train_logistic(X, y, lr, steps):
        """Full-batch gradient descent on binary cross-entropy. Returns w, b, loss history."""
        w, b, losses = np.zeros(X.shape[1]), 0.0, []
        for _ in range(steps):
            p = sigmoid(X @ w + b)
            losses.append(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            w -= lr * (X.T @ (p - y)) / len(y)
            b -= lr * np.mean(p - y)
        return w, b, losses

    return radial_features, sigmoid, train_logistic


@app.cell
def _(mo):
    logreg_lr = mo.ui.slider(0.01, 2.0, value=0.5, step=0.01, label="Learning rate α")
    logreg_steps = mo.ui.slider(10, 2000, value=500, step=10, label="Steps")
    logreg_radius = mo.ui.checkbox(label="Add feature x₁² + x₂²", value=False)
    mo.hstack([logreg_lr, logreg_steps, logreg_radius], justify="start")
    return logreg_lr, logreg_radius, logreg_steps


@app.cell
def _(
    X2d,
    dataset_choice,
    logreg_lr,
    logreg_radius,
    logreg_steps,
    mo,
    np,
    plot_boundary,
    plt,
    radial_features,
    sigmoid,
    train_logistic,
    y2d,
):
    _use_r = logreg_radius.value
    _w, _b, _losses = train_logistic(radial_features(X2d, _use_r), y2d, logreg_lr.value, logreg_steps.value)
    _prob = lambda Z: sigmoid(radial_features(Z, _use_r) @ _w + _b)
    _acc = np.mean((_prob(X2d) > 0.5) == y2d)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    plot_boundary(_a1, _prob, X2d, y2d,
                  f"Logistic regression{' + radius feature' if _use_r else ''}\ntrain acc = {_acc:.1%}")
    _a2.plot(_losses)
    _a2.set(xlabel="gradient step", ylabel="cross-entropy loss", title="Training loss")
    _a2.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"*Learned weights $\\mathbf{{w}}$ = {np.round(_w, 2)}, $b$ = {_b:.2f}. "
              f"On **{dataset_choice.value}** without the extra feature, the boundary can only "
              "ever be a straight line -- no learning rate or step count fixes that.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What just happened

    - On **Two blobs** the linear model works, and the loss curve shows
      gradient descent at work: a fast drop, then a slow approach to the
      minimum. Push the learning rate up and watch the curve; the convex
      bowl makes this forgiving, but a deep network's loss is not.
    - On **XOR, Circles, and Two spirals** it fails, and not because of the
      optimizer: *no straight line separates the classes*. This is the
      limitation that stalled neural network research after Minsky and
      Papert's *Perceptrons* (1969).
    - Adding the single feature $x_1^2+x_2^2$ makes **Circles** linearly
      separable -- in the 3D space $(x_1, x_2, r^2)$ a flat plane cuts the
      inner ring from the outer one. That is exactly what HOG and SIFT did
      for images: a human chose a transformation $\phi(\mathbf{x})$ that makes
      the classes separable, and a linear classifier finished the job.

    The radius feature does nothing for **Two spirals**, and for natural
    images nobody knows what the right $\phi$ is. The idea of the rest of
    this course: **learn $\phi$ too.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Multilayer perceptron: learning the features

    Stack a learned feature map in front of the linear classifier. A
    one-hidden-layer MLP computes

    $$\underbrace{\mathbf{a} = W_1\mathbf{x} + \mathbf{b}_1,\quad
      \mathbf{h} = \phi(\mathbf{a})}_{\text{learned features (hidden layer)}},\qquad
      \underbrace{\mathbf{s} = W_2\mathbf{h} + \mathbf{b}_2,\quad
      \mathbf{p} = \mathrm{softmax}(\mathbf{s})}_{\text{linear classifier on top}},$$

    where $\phi$ is a **nonlinear activation** applied elementwise -- ReLU
    $\max(0,a)$, $\tanh(a)$, or the sigmoid -- and
    $\mathrm{softmax}(\mathbf{s})_c = e^{s_c}/\sum_k e^{s_k}$ generalizes the
    sigmoid to $C$ classes (§5.3.4). Without $\phi$ the two layers collapse to
    one linear map $W_2W_1$, and we are back to a straight-line boundary.

    The loss is the multi-class cross-entropy,
    $\mathcal{L} = -\frac1N\sum_i \log p_{i,y_i}$, and its gradient with respect
    to the scores is again *prediction minus truth*:
    $\partial\mathcal{L}/\partial\mathbf{s}_i = (\mathbf{p}_i - \mathbf{e}_{y_i})/N$,
    with $\mathbf{e}_{y_i}$ the one-hot label.

    ### Backpropagation = the chain rule, run backward, reusing the forward pass

    To get the gradient for **every** weight, start from the loss and walk the
    computation backward one stage at a time (§5.3.5). Each stage multiplies the
    incoming (upstream) gradient by its own local derivative. Writing a batch
    as rows ($X$ is $N\times d$):

    | Forward (cache these) | Backward (in reverse order) |
    |---|---|
    | $A = XW_1^\top + \mathbf{b}_1$ | $\delta_S = (P - Y)/N$ |
    | $H = \phi(A)$ | $\nabla W_2 = \delta_S^\top H,\ \ \nabla\mathbf{b}_2 = \textstyle\sum_i \delta_{S,i}$ |
    | $S = HW_2^\top + \mathbf{b}_2$ | $\delta_H = \delta_S W_2$ |
    | $P = \mathrm{softmax}(S)$ | $\delta_A = \delta_H \odot \phi'(A)$ |
    | $\mathcal{L} = -\frac1N\sum\log P_{i,y_i}$ | $\nabla W_1 = \delta_A^\top X,\ \ \nabla\mathbf{b}_1 = \textstyle\sum_i \delta_{A,i}$ |

    The backward pass costs about the same as the forward pass -- that one
    fact is what makes training networks with millions of weights possible.
    The code below is this table, line for line.
    """)
    return


@app.cell
def _(np):
    ACTIVATIONS = {
        "ReLU": (lambda a: np.maximum(0, a), lambda a: (a > 0).astype(float)),
        "tanh": (np.tanh, lambda a: 1 - np.tanh(a) ** 2),
        "sigmoid": (lambda a: 1 / (1 + np.exp(-a)), lambda a: (s := 1 / (1 + np.exp(-a))) * (1 - s)),
    }

    def softmax(S):
        E = np.exp(S - S.max(axis=1, keepdims=True))  # subtract the max for numerical stability
        return E / E.sum(axis=1, keepdims=True)

    def mlp_init(d_in, d_hidden, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "W1": rng.normal(0, np.sqrt(2.0 / d_in), (d_hidden, d_in)),
            "b1": np.zeros(d_hidden),
            "W2": rng.normal(0, np.sqrt(1.0 / d_hidden), (n_classes, d_hidden)),
            "b2": np.zeros(n_classes),
        }

    def mlp_forward(params, X, act="ReLU"):
        phi, _ = ACTIVATIONS[act]
        A = X @ params["W1"].T + params["b1"]
        H = phi(A)
        S = H @ params["W2"].T + params["b2"]
        return softmax(S), {"X": X, "A": A, "H": H}

    def cross_entropy(P, y):
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))

    def mlp_backward(params, P, cache, y, act="ReLU"):
        _, dphi = ACTIVATIONS[act]
        N = len(y)
        dS = P.copy()
        dS[np.arange(N), y] -= 1.0
        dS /= N  # (P - Y) / N
        grads = {"W2": dS.T @ cache["H"], "b2": dS.sum(0)}
        dH = dS @ params["W2"]
        dA = dH * dphi(cache["A"])
        grads["W1"] = dA.T @ cache["X"]
        grads["b1"] = dA.sum(0)
        return grads

    def train_mlp(X, y, n_classes, d_hidden, act, lr, epochs, seed=0, X_val=None, y_val=None):
        """Full-batch gradient descent. d_hidden=0 means no hidden layer (softmax regression)."""
        if d_hidden == 0:
            rng = np.random.default_rng(seed)
            params = {"W": rng.normal(0, 0.01, (n_classes, X.shape[1])), "b": np.zeros(n_classes)}
        else:
            params = mlp_init(X.shape[1], d_hidden, n_classes, seed)
        hist = {"loss": [], "val_acc": []}
        for _ in range(epochs):
            if d_hidden == 0:
                P = softmax(X @ params["W"].T + params["b"])
                dS = P.copy()
                dS[np.arange(len(y)), y] -= 1.0
                dS /= len(y)
                grads = {"W": dS.T @ X, "b": dS.sum(0)}
            else:
                P, cache = mlp_forward(params, X, act)
                grads = mlp_backward(params, P, cache, y, act)
            hist["loss"].append(cross_entropy(P, y))
            for k in params:
                params[k] -= lr * grads[k]  # the update rule: theta <- theta - alpha * grad
            if X_val is not None:
                hist["val_acc"].append(np.mean(predict(params, X_val, act) == y_val))
        return params, hist

    def predict_proba(params, X, act="ReLU"):
        if "W" in params:
            return softmax(X @ params["W"].T + params["b"])
        return mlp_forward(params, X, act)[0]

    def predict(params, X, act="ReLU"):
        return predict_proba(params, X, act).argmax(1)

    return (
        ACTIVATIONS,
        cross_entropy,
        mlp_backward,
        mlp_forward,
        mlp_init,
        predict,
        predict_proba,
        train_mlp,
    )


@app.cell
def _(mo):
    mlp_hidden = mo.ui.slider(1, 64, value=16, step=1, label="Hidden units")
    mlp_act = mo.ui.dropdown(["ReLU", "tanh", "sigmoid"], value="tanh", label="Activation φ")
    mlp_lr = mo.ui.slider(0.05, 3.0, value=1.0, step=0.05, label="Learning rate α")
    mlp_epochs = mo.ui.slider(100, 5000, value=3000, step=100, label="Steps")
    mlp_show_units = mo.ui.checkbox(label="Draw each hidden unit's line", value=True)
    mo.vstack([
        mo.hstack([mlp_hidden, mlp_act, mlp_lr], justify="start"),
        mo.hstack([mlp_epochs, mlp_show_units], justify="start"),
    ])
    return mlp_act, mlp_epochs, mlp_hidden, mlp_lr, mlp_show_units


@app.cell
def _(
    X2d,
    dataset_choice,
    mlp_act,
    mlp_epochs,
    mlp_hidden,
    mlp_lr,
    mlp_show_units,
    mo,
    np,
    plot_boundary,
    plt,
    predict,
    predict_proba,
    train_mlp,
    y2d,
):
    mlp2d_params, _hist = train_mlp(X2d, y2d, 2, mlp_hidden.value, mlp_act.value, mlp_lr.value, mlp_epochs.value)
    _prob = lambda Z: predict_proba(mlp2d_params, Z, mlp_act.value)[:, 1]
    _acc = np.mean(predict(mlp2d_params, X2d, mlp_act.value) == y2d)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    plot_boundary(_a1, _prob, X2d, y2d,
                  f"MLP, {mlp_hidden.value} {mlp_act.value} units\ntrain acc = {_acc:.1%}")
    if mlp_show_units.value:
        _xs = np.linspace(-1.8, 1.8, 2)
        for (_w1, _w2), _b in zip(mlp2d_params["W1"], mlp2d_params["b1"]):
            if abs(_w2) > 1e-6:
                _a1.plot(_xs, -(_w1 * _xs + _b) / _w2, color="0.3", lw=0.6, alpha=0.6)
        _a1.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    _a2.plot(_hist["loss"])
    _a2.set(xlabel="gradient step", ylabel="cross-entropy loss", title="Training loss", yscale="log")
    _a2.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"*{dataset_choice.value}. Gray lines: each hidden unit's own linear boundary "
              "$\\mathbf{w}_{1,j}^\\top\\mathbf{x}+b_{1,j}=0$. Every hidden unit is a logistic-"
              "regression-like detector of one straight cut; the output layer combines those "
              "cuts into a curved boundary.*"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Things to try

    - **Width.** On *Circles*, 1 hidden unit is one straight cut and fails;
      3 units can already wrap the inner ring in a triangle. On *Two spirals*
      you need many more. More units, more cuts, more complex boundaries --
      the 2D picture of the **universal approximation theorem** (Hornik,
      1991): one hidden layer with enough units can approximate any
      continuous function.
    - **Activation.** Switch $\phi$ to ReLU and the boundary becomes
      piecewise linear (kinks where a unit switches on); tanh gives smooth
      curves. The sigmoid trains noticeably slower -- its derivative is at
      most 0.25, so gradients shrink as they flow backward, a first glimpse
      of the **vanishing gradient** problem that kept deep networks
      untrainable until ReLU (§5.3.2).
    - **Learning rate.** Too large and the loss curve oscillates or blows
      up; too small and it crawls. Unlike logistic regression, the MLP loss
      is **not convex**: change the seed (in the code) and you can land in a
      different solution.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Never trust a hand-written gradient: the numerical check

    A sign error in `mlp_backward` would not crash anything -- the network
    would just train badly, and you might blame the learning rate for a week.
    The standard defense is to compare against a **centered finite
    difference** for each parameter $\theta_k$:

    $$\frac{\partial\mathcal{L}}{\partial\theta_k} \approx
      \frac{\mathcal{L}(\theta + h\mathbf{e}_k) - \mathcal{L}(\theta - h\mathbf{e}_k)}{2h},
      \qquad h \approx 10^{-5}.$$

    This costs two full forward passes *per parameter*, hopeless for training
    but perfect for a test on a tiny network. We report the relative error
    $\lVert g_{\text{bp}} - g_{\text{num}}\rVert / (\lVert g_{\text{bp}}\rVert + \lVert g_{\text{num}}\rVert)$;
    values near $10^{-7}$ or smaller mean the backward pass is right.
    (ReLU's kink at 0 can produce slightly larger errors, which is why this
    check uses tanh.)
    """)
    return


@app.cell
def _(cross_entropy, mlp_backward, mlp_forward, mlp_init, mo, np):
    _rng = np.random.default_rng(1)
    _Xc, _yc = _rng.normal(size=(8, 3)), _rng.integers(0, 4, 8)
    _params = mlp_init(3, 5, 4, seed=2)
    _P, _cache = mlp_forward(_params, _Xc, "tanh")
    _g_bp = mlp_backward(_params, _P, _cache, _yc, "tanh")

    _rows = []
    for _name, _theta in _params.items():
        _g_num = np.zeros_like(_theta)
        _it = np.nditer(_theta, flags=["multi_index"])
        for _ in _it:
            _ix = _it.multi_index
            _old = _theta[_ix]
            _theta[_ix] = _old + 1e-5
            _lp = cross_entropy(mlp_forward(_params, _Xc, "tanh")[0], _yc)
            _theta[_ix] = _old - 1e-5
            _lm = cross_entropy(mlp_forward(_params, _Xc, "tanh")[0], _yc)
            _theta[_ix] = _old
            _g_num[_ix] = (_lp - _lm) / 2e-5
        _rel = np.linalg.norm(_g_bp[_name] - _g_num) / (np.linalg.norm(_g_bp[_name]) + np.linalg.norm(_g_num))
        _rows.append(f"| `{_name}` | {_theta.shape} | {_rel:.1e} |")
    mo.md("| parameter | shape | relative error (backprop vs. numerical) |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Backprop on a graph you can do by hand

    The table in §3 is the matrix version of a rule you can apply to any
    expression drawn as a **computational graph**. Take
    $f(x,y,z) = (x+y)\,z$, with intermediate node $q = x+y$:

    - **Forward:** compute $q$, then $f = qz$, caching both.
    - **Backward:** start at $\partial f/\partial f = 1$. The multiply gate
      sends each input the *other* input times the upstream gradient:
      $\partial f/\partial z = q$, $\partial f/\partial q = z$. The add gate
      copies its upstream gradient to both inputs:
      $\partial f/\partial x = \partial f/\partial y = \partial f/\partial q$.

    So **add distributes, multiply swaps, max routes** -- the gate
    "personalities" from the slides. Change the inputs and check the numbers
    against a numerical derivative.
    """)
    return


@app.cell
def _(mo):
    gx_in = mo.ui.number(value=3.0, step=0.5, label="x")
    gy_in = mo.ui.number(value=-2.0, step=0.5, label="y")
    gz_in = mo.ui.number(value=6.0, step=0.5, label="z")
    mo.hstack([gx_in, gy_in, gz_in], justify="start")
    return gx_in, gy_in, gz_in


@app.cell
def _(gx_in, gy_in, gz_in, mo):
    _x, _y, _z = gx_in.value, gy_in.value, gz_in.value
    _f = lambda a, b, c: (a + b) * c
    _q = _x + _y
    _fval = _q * _z
    _df = 1.0
    _dz, _dq = _q * _df, _z * _df
    _dx, _dy = _dq, _dq
    _h = 1e-6
    _num = [(_f(_x + _h, _y, _z) - _f(_x - _h, _y, _z)) / (2 * _h),
            (_f(_x, _y + _h, _z) - _f(_x, _y - _h, _z)) / (2 * _h),
            (_f(_x, _y, _z + _h) - _f(_x, _y, _z - _h)) / (2 * _h)]
    mo.md(f"""
    | | forward value | backprop gradient $\\partial f/\\partial\\,\\cdot$ | numerical check |
    |---|---|---|---|
    | $x$ | {_x:g} | $\\partial f/\\partial q \\cdot 1$ = **{_dx:g}** | {_num[0]:.4f} |
    | $y$ | {_y:g} | $\\partial f/\\partial q \\cdot 1$ = **{_dy:g}** | {_num[1]:.4f} |
    | $z$ | {_z:g} | $q$ = **{_dz:g}** | {_num[2]:.4f} |
    | $q = x+y$ | {_q:g} | $z$ = **{_dq:g}** | |
    | $f = qz$ | **{_fval:g}** | 1 | |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Real images: handwritten digits

    Now the same code on images. The UCI handwritten digits set (bundled
    with scikit-learn, so no download) has 1,797 grayscale images of size
    $8\times8$, ten classes. Each image is flattened into a vector
    $\mathbf{x}\in\mathbb{R}^{64}$ -- the MLP sees **64 numbers, not a
    picture**. We hold out 20% as a **test set** the model never trains on,
    because accuracy on the training data says nothing about images the
    model has not seen.
    """)
    return


@app.cell
def _(load_digits, mo, np, plt):
    _digits = load_digits()
    _rng = np.random.default_rng(0)
    _order = _rng.permutation(len(_digits.target))
    _n_train = int(0.8 * len(_order))
    digits_X = _digits.data / 16.0  # pixel values 0..16 -> 0..1
    digits_y = _digits.target
    digits_train_idx, digits_test_idx = _order[:_n_train], _order[_n_train:]

    _fig, _axes = plt.subplots(2, 10, figsize=(9, 2.1))
    for _k, _ax in enumerate(_axes.ravel()):
        _i = digits_train_idx[_k]
        _ax.imshow(digits_X[_i].reshape(8, 8), cmap="gray_r")
        _ax.set_title(str(digits_y[_i]), fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"*{_n_train} training and {len(_order) - _n_train} test images. "
              "Source: UCI ML hand-written digits (Alpaydin & Kaynak, 1998), via "
              "`sklearn.datasets.load_digits`.*"),
    ])
    return digits_X, digits_test_idx, digits_train_idx, digits_y


@app.cell
def _(mo):
    digits_hidden = mo.ui.slider(0, 128, value=64, step=8,
                                 label="Hidden units (0 = softmax regression, no hidden layer)")
    digits_epochs = mo.ui.slider(100, 2000, value=1000, step=100, label="Steps")
    mo.hstack([digits_hidden, digits_epochs], justify="start")
    return digits_epochs, digits_hidden


@app.cell
def _(
    digits_X,
    digits_epochs,
    digits_hidden,
    digits_test_idx,
    digits_train_idx,
    digits_y,
    mo,
    np,
    plt,
    predict,
    train_mlp,
):
    _Xtr, _ytr = digits_X[digits_train_idx], digits_y[digits_train_idx]
    _Xte, _yte = digits_X[digits_test_idx], digits_y[digits_test_idx]
    digits_params, _hist = train_mlp(_Xtr, _ytr, 10, digits_hidden.value, "ReLU", 0.5,
                                     digits_epochs.value, X_val=_Xte, y_val=_yte)
    digits_pred = predict(digits_params, _Xte, "ReLU")
    _train_acc = np.mean(predict(digits_params, _Xtr, "ReLU") == _ytr)
    _test_acc = np.mean(digits_pred == _yte)

    _conf = np.zeros((10, 10), int)
    np.add.at(_conf, (_yte, digits_pred), 1)
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(11, 3.4))
    _a1.plot(_hist["loss"])
    _a1.set(xlabel="gradient step", ylabel="training loss", yscale="log", title="Loss")
    _a1.grid(alpha=0.3)
    _a2.plot(_hist["val_acc"])
    _a2.set(xlabel="gradient step", ylabel="test accuracy", title="Test accuracy", ylim=(0.5, 1))
    _a2.grid(alpha=0.3)
    _a3.imshow(_conf, cmap="Blues")
    for (_r, _c), _v in np.ndenumerate(_conf):
        if _v:
            _a3.text(_c, _r, _v, ha="center", va="center", fontsize=7,
                     color="white" if _v > _conf.max() / 2 else "black")
    _a3.set(xlabel="predicted", ylabel="true", title="Test confusion matrix",
            xticks=range(10), yticks=range(10))
    _fig.tight_layout()
    _model = "softmax regression" if digits_hidden.value == 0 else f"MLP with {digits_hidden.value} ReLU units"
    mo.vstack([
        _fig,
        mo.md(f"**{_model}:** train accuracy {_train_acc:.1%}, **test accuracy {_test_acc:.1%}**. "
              "Set the slider to 0 to compare against a purely linear classifier on raw pixels. "
              "Expect a small gap here (about 98% vs. 99%): these digits are tiny, centered, and "
              "size-normalized, so most of the hard variation from the slides -- viewpoint, "
              "illumination, clutter, scale -- was removed before the data reached us. On "
              "natural photos that variation is still in the pixels, and the learned features "
              "are what close the gap."),
    ])
    return digits_params, digits_pred


@app.cell
def _(digits_X, digits_hidden, digits_params, digits_pred, digits_test_idx, digits_y, mo, np, plt):
    _W = digits_params["W"] if digits_hidden.value == 0 else digits_params["W1"]
    _n_show = min(16, _W.shape[0])
    _fig, _axes = plt.subplots(2, 8, figsize=(9, 2.4))
    for _k, _ax in enumerate(_axes.ravel()):
        _ax.axis("off")
        if _k < _n_show:
            _v = np.abs(_W[_k]).max()
            _ax.imshow(_W[_k].reshape(8, 8), cmap="RdBu_r", vmin=-_v, vmax=_v)
            _ax.set_title(f"unit {_k}" if digits_hidden.value else f"class {_k}", fontsize=8)

    _wrong = np.flatnonzero(digits_pred != digits_y[digits_test_idx])[:8]
    _fig2, _axes2 = plt.subplots(1, 8, figsize=(9, 1.5))
    for _ax in _axes2:
        _ax.axis("off")
    for _ax, _j in zip(_axes2, _wrong):
        _ax.imshow(digits_X[digits_test_idx[_j]].reshape(8, 8), cmap="gray_r")
        _ax.set_title(f"true {digits_y[digits_test_idx[_j]]} → {digits_pred[_j]}", fontsize=8)
    mo.vstack([
        mo.md("**First-layer weights, reshaped to 8x8** (red = positive, blue = negative). "
              "Each hidden unit computes $\\mathbf{w}_j^\\top\\mathbf{x}$, a dot product of the "
              "image with this pattern. With many hidden units the patterns are hard to read: "
              "each is one partial detector, and only the next layer's combination of them "
              "means \"a 3\". Set the slider to 0 to see the ten class templates of softmax "
              "regression: red where that digit's ink tends to be, blue where other digits' ink "
              "tends to be (look for the ring of the 0). One template per class is all a linear "
              "classifier can learn."),
        _fig,
        mo.md("**Test mistakes** (true → predicted):"),
        _fig2,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. What an MLP cannot see

    Look again at the first-layer weights: each one is a full $8\times8$
    template, and nothing in the model knows that pixel $(2,3)$ is next to
    pixel $(2,4)$. To the MLP, the image is an unordered list of 64 numbers.

    Test that claim directly. Pick **one** fixed random permutation of the
    64 pixel positions and apply it to every image, training and test alike.
    The images become unrecognizable to us. If the MLP really ignores
    spatial layout, its accuracy should not change.
    """)
    return


@app.cell
def _(mo):
    shuffle_run = mo.ui.run_button(label="Train on original vs. pixel-shuffled digits")
    shuffle_run
    return (shuffle_run,)


@app.cell
def _(
    digits_X,
    digits_test_idx,
    digits_train_idx,
    digits_y,
    mo,
    np,
    plt,
    predict,
    shuffle_run,
    train_mlp,
):
    mo.stop(not shuffle_run.value, mo.md("*Click the button to run (about two seconds).*"))
    _perm = np.random.default_rng(123).permutation(64)
    _ytr, _yte = digits_y[digits_train_idx], digits_y[digits_test_idx]
    _results = {}
    for _label, _X in [("original", digits_X), ("pixel-shuffled", digits_X[:, _perm])]:
        _p, _ = train_mlp(_X[digits_train_idx], _ytr, 10, 64, "ReLU", 0.5, 1000)
        _results[_label] = np.mean(predict(_p, _X[digits_test_idx], "ReLU") == _yte)

    _fig, _axes = plt.subplots(2, 8, figsize=(9, 2.4))
    for _k in range(8):
        _i = digits_train_idx[_k]
        _axes[0, _k].imshow(digits_X[_i].reshape(8, 8), cmap="gray_r")
        _axes[1, _k].imshow(digits_X[_i, _perm].reshape(8, 8), cmap="gray_r")
        _axes[0, _k].axis("off")
        _axes[1, _k].axis("off")
    _axes[0, 0].set_title("original", fontsize=8, loc="left")
    _axes[1, 0].set_title("same pixels, shuffled", fontsize=8, loc="left")
    mo.vstack([
        _fig,
        mo.md(f"| input | MLP test accuracy |\n|---|---|\n"
              f"| original | {_results['original']:.1%} |\n"
              f"| pixel-shuffled | {_results['pixel-shuffled']:.1%} |"),
        mo.md("The two accuracies match to within run-to-run noise: a permutation of the "
              "inputs is just a permutation of the columns of $W_1$, which gradient descent "
              "learns equally well. An MLP throws away the one thing every earlier notebook "
              "relied on -- **locality** -- and must relearn every pattern at every position "
              "separately. On a $224\\times224$ color image, one fully connected hidden layer "
              "of 1,000 units would already need about 150 million weights."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary and what comes next

    - **Learning** replaces a hand-designed function with one fit to examples:
      choose data, a model, a loss, and an optimizer.
    - **Logistic / softmax regression** is a linear classifier on top of
      fixed features; its cross-entropy gradient is *prediction minus truth*.
      It is only as good as the features it is given.
    - An **MLP** learns the features too: each hidden unit is a learned
      linear detector followed by a nonlinearity, and stacking them bends the
      decision boundary.
    - **Backpropagation** is the chain rule run backward through the
      computational graph, reusing the forward pass; always verify it with a
      numerical gradient check. Frameworks such as PyTorch do this
      automatically (*autograd*), which is what we use from Lecture 22 on.
    - An MLP on raw pixels ignores spatial structure. **Lecture 22** puts the
      structure back: a **convolutional** layer is a small template slid over
      every position (the linear filtering of notebook 8, with *learned*
      kernels), so one set of weights serves the whole image. We then look at
      Transformers and at reusing networks pre-trained on millions of images
      (transfer learning).

    **Reading:** Szeliski, *Computer Vision: Algorithms and Applications*
    (2nd ed.), §5.1 and §5.3.
    """)
    return


if __name__ == "__main__":
    app.run()
