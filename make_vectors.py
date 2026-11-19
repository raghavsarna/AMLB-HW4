"""make_vectors.py: reproduces stocks.csv and managers.csv offline from data/ and external.parquet.

    python make_vectors.py            # writes stocks.csv and managers.csv next to this file

See design.md for the logic and model.py for the objective.
"""
import os
import warnings

import numpy as np
import pandas as pd

import model

# numpy on Apple Accelerate emits spurious floating-point warnings from matmul; results are checked below
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
HERE = os.path.dirname(os.path.abspath(__file__))
CUTOFF = "2026-03-02"        # only holdings public by this date are used
COLS = [f"d{j}" for j in range(1, model.K + 1)]

# chosen on the back-tests (see design.md and backtest.py)
DEFAULT = dict(gamma=0.8, b=6.0, lam=1.0, ipo_w=3.0, seed=0,
               emb="e5", k_text=8, text_mode="shared", text_scale=0.3, meta=1,
               knn=0, knn_mode="shared", tau=0.05, blend=0.0,
               k=16, lam_new=0.2, max_h=2, new_bias=0, inv_norm=0, sic=0.0)
US = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
         "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split())


def ownership(comp, hold, fil, mgr, cutoff, gamma):
    """Soft ownership target y_ms from the 13Fs public at `cutoff`: the recency-weighted share of the
    quarters (investor filed, stock already listed) in which the investor reported the stock."""
    h = hold[hold.filing_date <= cutoff]
    f = fil[fil.filing_date <= cutoff]
    quarters = sorted(f.quarter_end.unique())
    mids, sids = mgr.manager_id.tolist(), comp.stock_id.tolist()
    mi = {m: i for i, m in enumerate(mids)}
    si = {s: j for j, s in enumerate(sids)}
    listing = comp.listing_date.fillna("1900-01-01").to_numpy()
    num = np.zeros((len(mids), len(sids)))
    den = np.zeros((len(mids), len(sids)))
    for k, q in enumerate(quarters):
        w = gamma ** (len(quarters) - 1 - k)
        filed = np.zeros(len(mids), bool)
        filed[[mi[m] for m in f.loc[f.quarter_end == q, "manager_id"]]] = True
        obs = filed[:, None] & (listing <= q)[None, :]
        held = np.zeros_like(obs)
        hq = h[h.quarter_end == q]
        held[[mi[m] for m in hq.manager_id], [si[s] for s in hq.stock_id]] = True
        num += w * (held & obs)
        den += w * obs
    Y = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return Y, (den > 0).astype(float)


def canonical_rotation(U, V):
    """The objective depends on cosines only, so any rotation of the joint space is an equally good
    solution.  Rotate onto the principal axes of all 397 unit vectors (signs fixed by the third
    moment) so that the tables are unique and reproducible across machines; cosines are unchanged."""
    A = np.vstack([U, V])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    P = A @ Vt.T
    sign = np.sign((P ** 3).sum(axis=0))
    sign[sign == 0] = 1
    R = Vt.T * sign
    return U @ R, V @ R


def transitions(comp, hold, fil, mgr, cutoff, max_h):
    """New-holding transitions observable at `cutoff`: for every pair of quarters 1..max_h apart, the
    (investor, stock) cells where the investor filed both times, the stock was listed at the first
    quarter and not held then; the label is whether it was held at the second quarter.  This is the
    event Test 3 scores.  Returns the mean label and the number of observations per cell."""
    h = hold[hold.filing_date <= cutoff]
    f = fil[fil.filing_date <= cutoff]
    quarters = sorted(f.quarter_end.unique())
    mids, sids = mgr.manager_id.tolist(), comp.stock_id.tolist()
    mi = {m: i for i, m in enumerate(mids)}
    si = {s: j for j, s in enumerate(sids)}
    listing = comp.listing_date.fillna("1900-01-01").to_numpy()
    held, filed = {}, {}
    for q in quarters:
        H = np.zeros((len(mids), len(sids)), bool)
        hq = h[h.quarter_end == q]
        H[[mi[m] for m in hq.manager_id], [si[s] for s in hq.stock_id]] = True
        held[q] = H
        F = np.zeros(len(mids), bool)
        F[[mi[m] for m in f.loc[f.quarter_end == q, "manager_id"]]] = True
        filed[q] = F
    num = np.zeros((len(mids), len(sids)))
    den = np.zeros((len(mids), len(sids)))
    for i, q0 in enumerate(quarters):
        for q1 in quarters[i + 1:i + 1 + max_h]:
            obs = (filed[q0] & filed[q1])[:, None] & (listing <= q0)[None, :] & ~held[q0]
            num += obs & held[q1]
            den += obs
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0), den


def build(comp, hold, fil, mgr, ext, cutoff=CUTOFF, cfg=DEFAULT, hide=()):
    """hide: stock ids treated as cold (used only for cross-validation)."""
    cfg = {**DEFAULT, **cfg}
    Y, Wt = ownership(comp, hold, fil, mgr, cutoff, cfg["gamma"])
    Wt[:, comp.stock_id.isin(list(hide)).to_numpy()] = 0
    warm = Wt.sum(axis=0) > 0                       # stocks with at least one observed quarter
    # Stage 1: investors and warm stocks from holdings
    W1 = Wt.copy()
    Yn, Wn = (transitions(comp, hold, fil, mgr, cutoff, cfg["max_h"]) if cfg["lam_new"] > 0 else (None, None))
    if cfg["inv_norm"]:                             # every investor counts equally, as in Test 3
        W1 = W1 * (W1.sum() / len(W1)) / np.clip(W1.sum(axis=1, keepdims=True), 1e-12, None)
        if Wn is not None:
            Wn = Wn * (Wn.sum() / len(Wn)) / np.clip(Wn.sum(axis=1, keepdims=True), 1e-12, None)
    U, Vw, info = model.cf_fit(Y[:, warm], W1[:, warm], b=cfg["b"], seed=cfg["seed"], k=cfg["k"],
                               Yn=None if Yn is None else Yn[:, warm], Wn=None if Wn is None else Wn[:, warm],
                               lam_new=cfg["lam_new"], new_bias=bool(cfg["new_bias"]))
    # Stage 2: map public features to directions, trained on the warm stocks
    Vw_full = np.zeros((len(comp), model.K))
    Vw_full[warm] = Vw
    X = features(comp, ext, warm, cfg, Vw_full)
    new = (comp.status != "established").to_numpy()
    w2 = Wt[:, warm] * np.where(new[warm], cfg["ipo_w"], 1.0)[None, :]
    Wmap, _ = model.side_fit(U, Y[:, warm], w2, X[warm], lam=cfg["lam"])
    V = X @ Wmap.T
    V = V + 1e-6 * Vw.mean(axis=0)                  # guard against an all-zero vector
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    if cfg["blend"] > 0:                            # optional shrinkage of warm stocks toward their feature direction
        Vw = Vw + cfg["blend"] * V[warm]
        Vw = Vw / np.linalg.norm(Vw, axis=1, keepdims=True)
    V[warm] = Vw
    U, V = canonical_rotation(U, V)
    S = pd.DataFrame(V, index=pd.Index(comp.stock_id, name="stock_id"), columns=COLS)
    M = pd.DataFrame(U, index=pd.Index(mgr.manager_id, name="manager_id"), columns=COLS)
    return S, M


def pca_fit(E, train, k):
    """PCA fitted on the training (warm) stocks only; the sign of each axis is fixed so the result does
    not depend on the linear-algebra library."""
    mu = E[train].mean(axis=0)
    _, sv, Vt = np.linalg.svd(E[train] - mu, full_matrices=False)
    Vt = Vt[:k] * np.sign(Vt[:k, np.argmax(np.abs(Vt[:k]), axis=1)][np.arange(k), np.arange(k)])[:, None]
    Z = (E - mu) @ Vt.T
    return Z / Z[train].std(axis=0)


SIC_GROUPS = [  # (name, test on the 4-digit SIC code)
    ("pharma_biotech", lambda c: 2830 <= c <= 2836 or c == 8731),
    ("medical_devices", lambda c: c in (3826, 3841, 3842, 3843, 3844, 3845, 3851, 8071, 8090, 8093)),
    ("software_internet", lambda c: 7370 <= c <= 7379),
    ("semis_hardware", lambda c: 3570 <= c <= 3579 or 3660 <= c <= 3699),
    ("banks", lambda c: 6020 <= c <= 6036 or c == 6712),
    ("other_finance", lambda c: 6100 <= c <= 6299 or 6300 <= c <= 6411 or c == 6199),
    ("energy_mining_utilities", lambda c: 1000 <= c <= 1499 or 2900 <= c <= 2999 or 4900 <= c <= 4991),
    ("industrial", lambda c: 3300 <= c <= 3569 or 3580 <= c <= 3659 or 3700 <= c <= 3799 or 1500 <= c <= 1799),
    ("consumer", lambda c: 2000 <= c <= 2799 or 5000 <= c <= 5999 or 7000 <= c <= 7099 or 7800 <= c <= 7999),
]


def sic_matrix(ext, comp):
    """Dummies for coarse SIC sector groups (the code in the filing's own header)."""
    sic = pd.to_numeric(ext.set_index("stock_id").reindex(comp.stock_id).sic, errors="coerce").fillna(0).astype(int)
    return np.column_stack([[float(f(c)) for c in sic] for _, f in SIC_GROUPS])


def text_matrix(ext, comp, emb):
    if "+" in emb:                                   # concatenation of two models' embeddings
        return np.hstack([text_matrix(ext, comp, e) for e in emb.split("+")]) / np.sqrt(len(emb.split("+")))
    e = ext.set_index("stock_id").reindex(comp.stock_id)
    cols = [c for c in e.columns if c.startswith(emb + "_")]
    E = e[cols].to_numpy(float)
    E = np.nan_to_num(E)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


def meta_features(ext, comp, warm):
    """Offering terms of a listing (0 for established stocks), standardised on warm listings."""
    e = ext.set_index("stock_id").reindex(comp.stock_id)
    new = (comp.status != "established").to_numpy()
    size = np.log(e.deal_usd.astype(float).to_numpy())
    tr = new & warm & np.isfinite(size)
    if tr.sum() < 5:                      # no listing with 13F history yet: offering terms unusable
        return np.zeros((len(comp), 4))
    size = np.where(np.isfinite(size), size, np.median(size[tr]))
    size = (size - size[tr].mean()) / size[tr].std()
    small_tier = e.exchange.isin(["nasdaq_cm", "nyse_american"]).to_numpy(float)
    top_bank = (e.n_top_banks.fillna(0).to_numpy(float) > 0).astype(float)
    foreign = (~e.biz_state.isin(US)).to_numpy(float)
    M = np.column_stack([size, small_tier, top_bank, foreign]) * new[:, None]
    return M


def knn_feature(E, Vw_full, warm, k, tau):
    """Text-similarity-weighted mean of the CF directions of the k most similar warm stocks
    (a stock is never its own neighbour)."""
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    S[:, ~warm] = -np.inf
    out = np.zeros((len(E), Vw_full.shape[1]))
    for i in range(len(E)):
        nb = np.argsort(-S[i])[:k]
        w = np.exp((S[i, nb] - S[i, nb].max()) / tau)
        out[i] = (w[:, None] * Vw_full[nb]).sum(axis=0) / w.sum()
    return out


def features(comp, ext, warm, cfg, Vw_full=None):
    new = (comp.status != "established").to_numpy(float)
    cols = [1 - new, new]
    if ext is not None and (cfg["text_mode"] != "none" or cfg["knn"] > 0):
        E = text_matrix(ext, comp, cfg["emb"])
    if ext is not None and cfg["text_mode"] != "none":
        Z = cfg["text_scale"] * pca_fit(E, warm, cfg["k_text"])
        if cfg["text_mode"] in ("shared", "both"):
            cols += list(Z.T)
        if cfg["text_mode"] in ("new", "both"):
            cols += list((Z * new[:, None]).T)
    if ext is not None and cfg["sic"] > 0:
        G = sic_matrix(ext, comp)
        G = (G - G[warm].mean(axis=0)) / np.clip(G[warm].std(axis=0), 1e-9, None)
        cols += list((cfg["sic"] * G).T)
    if ext is not None and cfg["meta"]:
        cols += list(meta_features(ext, comp, warm).T)
    if ext is not None and cfg["knn"] > 0:
        Nb = knn_feature(E, Vw_full, warm, cfg["knn"], cfg["tau"])
        if cfg["knn_mode"] in ("shared", "both"):
            cols += list(Nb.T)
        if cfg["knn_mode"] in ("new", "both"):
            cols += list((Nb * new[:, None]).T)
    return np.column_stack(cols)


def main():
    d = os.path.join(HERE, "data")
    comp = pd.read_csv(os.path.join(d, "companies.csv"), dtype={"listing_date": str})
    hold = pd.read_csv(os.path.join(d, "holdings.csv"), dtype=str)
    fil = pd.read_csv(os.path.join(d, "filings.csv"), dtype=str)
    mgr = pd.read_csv(os.path.join(d, "manager_info.csv"))
    ext = pd.read_parquet(os.path.join(HERE, "external.parquet"))
    S, M = build(comp, hold, fil, mgr, ext)
    for T in (S, M):
        assert np.isfinite(T.to_numpy()).all() and (np.abs(T.to_numpy()).sum(axis=1) > 0).all()
    S.round(10).to_csv(os.path.join(HERE, "stocks.csv"))
    M.round(10).to_csv(os.path.join(HERE, "managers.csv"))
    print("wrote stocks.csv", S.shape, "and managers.csv", M.shape)


if __name__ == "__main__":
    main()
