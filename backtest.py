"""backtest.py: time-respecting historical tests used to make every design choice.

Back-test A (decision date 2025-08-15, i.e. after the June-2025 13Fs were due):
    train on holdings filed by 2025-08-15 (March and June 2025);
    Test 1/2 on the 2025H2 listings that listed after 2025-08-15 (owners at their first quarter-end);
    Test 3: stocks trading at June 2025, new holdings at December 2025 (same 6-month horizon as the real test).
Back-test B (decision date 2025-11-17, after the September-2025 13Fs):
    train on holdings filed by 2025-11-17, which includes the July-September listings' first holders;
    Test 1/2 on the listings after 2025-11-17 (owners at December 2025);
    Test 3: stocks trading at September 2025, new holdings at December 2025 (3-month horizon).
Features of a listing come only from its pre-listing registration statement; any transformation of
features (PCA, scaling) is fitted only on stocks that were trading at the decision date.
"""
import os
import sys

import numpy as np
import pandas as pd

import evaluate as ev
import make_vectors as mv

HERE = os.path.dirname(os.path.abspath(__file__))

BACKTESTS = {
    "A": dict(cutoff="2025-08-15", base="2025-06-30", target="2025-12-31"),
    "B": dict(cutoff="2025-11-17", base="2025-09-30", target="2025-12-31"),
}


def run(cfg, name, data, verbose=False):
    bt = BACKTESTS[name]
    comp, hold, fil, mgr, ext = data
    S, M = mv.build(comp, hold, fil, mgr, ext, cutoff=bt["cutoff"], cfg=cfg)
    C = ev.cosine(S, M)
    listed_after = comp[(comp.status != "established") & (comp.listing_date > bt["cutoff"])].stock_id
    outc = ev.listing_outcomes(listed_after, comp, hold, fil)
    t1, per = ev.test1(C, outc)
    pairs, invs = {}, {}
    t2, n2 = ev.test2(C, outc, per_pair=pairs)
    trading = comp[(comp.status == "established") | (comp.listing_date <= bt["base"])].stock_id.tolist()
    t3, n3 = ev.test3(C, hold, fil, bt["base"], bt["target"], trading, per_investor=invs)
    if verbose:
        print({k: round(v, 3) for k, v in per.items()})
    return dict(bt=name, t1=t1, n1=len(outc), t2=t2, n2=n2, t3=t3, n3=n3, u1=per, u2=pairs, u3=invs)


def cv(cfg, data, n_folds=4, verbose=False):
    """Cross-validation at the real decision date (2026-03-02): the 2025H2 listings are split into
    folds by listing date order; each fold's holdings are hidden and its owners at the first
    quarter-end after listing are predicted from features alone.  Not a historical test (it uses
    other stocks' holdings reported after the hidden listing's IPO); used only as extra evidence."""
    comp, hold, fil, mgr, ext = data
    ipos = comp[comp.status == "ipo_2025H2"].sort_values("listing_date").stock_id.tolist()
    outc_all = ev.listing_outcomes(ipos, comp, hold, fil)
    cols, t2s, n2, pairs = {}, [], 0, {}
    for k in range(n_folds):
        fold = ipos[k::n_folds]
        S, M = mv.build(comp, hold, fil, mgr, ext, cfg=cfg, hide=fold)
        C = ev.cosine(S, M)
        for s in fold:
            cols[s] = C[s]
        t2k, n2k = ev.test2(C, {s: o for s, o in outc_all.items() if s in fold}, per_pair=pairs)  # within a fold
        if n2k:
            t2s.append(t2k * n2k)
            n2 += n2k
    C = pd.DataFrame(cols)
    t1, per = ev.test1(C, outc_all)
    t2 = sum(t2s) / n2
    if verbose:
        print({k: round(v, 3) for k, v in per.items()})
    return dict(bt="CV", t1=t1, n1=len(outc_all), t2=t2, n2=n2, u1=per, u2=pairs)


def load():
    d = os.path.join(HERE, "data")
    comp = pd.read_csv(os.path.join(d, "companies.csv"), dtype={"listing_date": str})
    hold = pd.read_csv(os.path.join(d, "holdings.csv"), dtype=str)
    fil = pd.read_csv(os.path.join(d, "filings.csv"), dtype=str)
    mgr = pd.read_csv(os.path.join(d, "manager_info.csv"))
    ext_path = os.path.join(HERE, "external.parquet")
    ext = pd.read_parquet(ext_path) if os.path.exists(ext_path) else None
    return comp, hold, fil, mgr, ext


if __name__ == "__main__":
    data = load()
    cfg = mv.DEFAULT.copy()
    for kv in sys.argv[1:]:
        k, v = kv.split("=")
        cfg[k] = type(cfg[k])(v) if k in cfg else v
    verbose = cfg.pop("verbose", "0") == "1"
    for r in [run(cfg, name, data, verbose=verbose) for name in BACKTESTS] + [cv(cfg, data, verbose=verbose)]:
        out = {k: v for k, v in r.items() if not k.startswith("u")}
        for t, kind in (("1", "plain"), ("2", "pairs"), ("3", "plain")):
            if "u" + t in r and r["u" + t]:
                out["se" + t] = round(ev.bootstrap(r["u" + t], kind=kind)[1], 3)
        print(out)
