"""model.py: the two fitting stages shared by make_vectors.py and backtest.py.

Stage 1, collaborative filtering on the cosine itself (warm stocks = stocks with 13F history):
    z_ms = a + b * cos(u_m, v_s)
    minimise  sum_ms w_ms * BCE(y_ms, sigmoid(z_ms))  over u_m, v_s, a   (b fixed: the temperature
    bounds how sharply the model can fit any one cell, which is the regulariser of a cosine model)

Stage 2, side information for cold-start stocks (investor directions u_m held fixed):
    z_ms = c_s + <unit(u_m), W x_s>
    minimise  sum_ms w_ms * BCE(y_ms, sigmoid(z_ms)) + lam * |W|^2   over W and the intercepts c_s
    This is a convex (ridge-logistic) problem: W maps public features x_s to the direction in
    investor space that best separates a stock's holders from its non-holders; c_s absorbs each
    training stock's popularity so W learns only *who* holds, not *how many*.  A cold stock gets
    v_s = W x_s; only its direction matters for the cosine.

Everything is float64, single-threaded and seeded, so the result is reproducible.
"""
import numpy as np
import torch
from scipy.optimize import minimize

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
K = 16


def _lbfgs(f, theta0, maxiter):
    def fg(th_np):
        th = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        loss = f(th)
        loss.backward()
        return loss.item(), th.grad.numpy().copy()
    res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=0.0, gtol=1e-11, maxcor=50))
    return res


def _newton_polish(f, theta, steps=8, cg_iters=300, damping=1e-9):
    """Gradient-driven polish after L-BFGS.  L-BFGS stops when function values stop changing, which
    leaves the solution accurate only to about sqrt(machine epsilon) along flat directions.  Damped
    Newton steps, solved by conjugate gradients on exact Hessian-vector products, need only gradients,
    so they converge much further and make the result insensitive to tiny numerical differences."""
    th = torch.tensor(theta, dtype=torch.float64)

    def grad_at(x):
        x = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(f(x), x)
        return g

    g = grad_at(th)
    for _ in range(steps):
        x = th.clone().requires_grad_(True)
        gx, = torch.autograd.grad(f(x), x, create_graph=True)

        def hvp(v):
            hv, = torch.autograd.grad(gx, x, grad_outputs=v, retain_graph=True)
            return hv.detach() + damping * v

        # conjugate gradients for (H + damping I) d = -g
        d = torch.zeros_like(th)
        r = -g.clone()
        p = r.clone()
        rs = r @ r
        for _ in range(cg_iters):
            Hp = hvp(p)
            curv = p @ Hp
            if curv <= 0:
                break
            alpha = rs / curv
            d = d + alpha * p
            r = r - alpha * Hp
            rs_new = r @ r
            if rs_new.sqrt() < 1e-12 * g.norm():
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        g_new = grad_at(th + d)
        if g_new.norm() >= g.norm():
            break
        th, g = th + d, g_new
    return th.numpy()


def cf_fit(Y, Wt, b=6.0, seed=0, maxiter=5000, init=None, seed_scale=1e-10, k=K,
           Yn=None, Wn=None, lam_new=0.0, new_bias=False, polish=True):
    """Stage 1. Y, Wt: (n_inv, n_stock) soft targets and weights.
    Optional second term (lam_new > 0): Yn, Wn are labels/weights of observed *new-holding* transitions
    (stock not held at a base quarter, held or not at a later one), fitted with the same cosine and its
    own intercept (plus a per-stock intercept if new_bias), so the geometry also ranks future additions.
    k < K fits a k-dimensional model padded with zeros (used only for the dimension experiment).
    Returns unit U (n_inv, K), unit V (n_stock, K), info."""
    n_m, n_s = Y.shape
    if init is None:
        # deterministic spectral start: top-k singular vectors of the centred ownership matrix,
        # signs fixed, so the optimiser starts (and ends) at the same point on any machine
        A = Wt * (Y - (Y * Wt).sum() / Wt.sum())
        P, sv, Qt = np.linalg.svd(A, full_matrices=False)
        sgn = np.sign(P[np.argmax(np.abs(P[:, :k]), axis=0), np.arange(k)])
        U0 = P[:, :k] * sgn * np.sqrt(sv[:k])
        V0 = Qt[:k].T * sgn * np.sqrt(sv[:k])
        # rows with no observed entry (e.g. an investor with no 13F yet) start on the first axis
        U0[np.linalg.norm(U0, axis=1) < 1e-12] = np.eye(k)[0]
        V0[np.linalg.norm(V0, axis=1) < 1e-12] = np.eye(k)[0]
        U0 = U0 / np.linalg.norm(U0, axis=1, keepdims=True)
        V0 = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
        if seed:                                   # only used to test sensitivity to the start
            rng = np.random.RandomState(seed)
            U0, V0 = U0 + seed_scale * rng.normal(size=U0.shape), V0 + seed_scale * rng.normal(size=V0.shape)
    else:
        U0, V0 = init
    a0 = np.log(max(Y.mean(), 1e-3) / (1 - max(Y.mean(), 1e-3)))
    Yt, Wtt = torch.tensor(Y), torch.tensor(Wt)
    tot = float(Wt.sum())
    use_new = lam_new > 0 and Wn is not None and Wn.sum() > 0
    if use_new:
        Ynt, Wnt = torch.tensor(Yn), torch.tensor(Wn)
        totn = float(Wn.sum())
    n_extra = 2 + (n_s if (use_new and new_bias) else 0)

    def f(th):
        U = th[:n_m * k].reshape(n_m, k)
        V = th[n_m * k:(n_m + n_s) * k].reshape(n_s, k)
        e = th[(n_m + n_s) * k:]
        Un = U / U.norm(dim=1, keepdim=True)
        Vn = V / V.norm(dim=1, keepdim=True)
        cos = Un @ Vn.T
        z = e[0] + b * cos
        # keep raw norms near 1 so the optimisation stays well conditioned (does not change cosines)
        tie = 1e-3 * (((U.norm(dim=1) - 1) ** 2).sum() + ((V.norm(dim=1) - 1) ** 2).sum())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt, weight=Wtt, reduction="sum") / tot + tie
        if use_new:
            zn = e[1] + b * cos
            if new_bias:
                zn = zn + e[2:][None, :]
            loss = loss + lam_new * torch.nn.functional.binary_cross_entropy_with_logits(
                zn, Ynt, weight=Wnt, reduction="sum") / totn
        return loss

    res = _lbfgs(f, np.concatenate([U0.ravel(), V0.ravel(), [a0, a0], np.zeros(n_extra - 2)]), maxiter)
    th = _newton_polish(f, res.x) if polish else res.x
    U = th[:n_m * k].reshape(n_m, k)
    V = th[n_m * k:(n_m + n_s) * k].reshape(n_s, k)
    U = np.hstack([U / np.linalg.norm(U, axis=1, keepdims=True), np.zeros((n_m, K - k))])
    V = np.hstack([V / np.linalg.norm(V, axis=1, keepdims=True), np.zeros((n_s, K - k))])
    return U, V, dict(a=th[(n_m + n_s) * k], loss=res.fun, nit=res.nit, msg=str(res.message))


def side_fit(Un, Y, Wt, X, lam=1.0, maxiter=5000):
    """Stage 2. Un: (n_inv, K) unit investor vectors; Y, Wt: (n_inv, n_train) for training stocks;
    X: (n_train, p) features. Returns W (K, p)."""
    n_m, n_t = Y.shape
    p = X.shape[1]
    Ut, Yt, Wtt, Xt = (torch.tensor(a) for a in (Un, Y, Wt, X))
    tot = float(Wt.sum())

    def f(th):
        W = th[:K * p].reshape(K, p)
        c = th[K * p:]
        z = c[None, :] + Ut @ (Xt @ W.T).T
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt, weight=Wtt, reduction="sum") / tot
        return bce + lam * (W ** 2).sum() / n_t

    res = _lbfgs(f, np.zeros(K * p + n_t), maxiter)
    x = _newton_polish(f, res.x)
    return x[:K * p].reshape(K, p), dict(loss=res.fun, nit=res.nit)
