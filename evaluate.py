"""evaluate.py: re-implementation of the three tests in HW4_handout.md section 3, for back-testing.

All functions take a stock table S and a manager table M (DataFrames indexed by id, 16 columns)
and score them by cosine similarity only, as the bank does.
"""
import itertools

import numpy as np
import pandas as pd


def unit(df):
    a = df.to_numpy(dtype=float)
    return pd.DataFrame(a / np.linalg.norm(a, axis=1, keepdims=True), index=df.index)


def cosine(S, M):
    """DataFrame of cosines, rows = managers, columns = stocks."""
    s, m = unit(S), unit(M)
    return pd.DataFrame(m.to_numpy() @ s.to_numpy().T, index=m.index, columns=s.index)


def auc(scores, labels):
    """Probability that a random positive outranks a random negative, ties count one half."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(scores).rank(method="average").to_numpy()
    return (r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def held_sets(holdings, quarter):
    h = holdings[holdings.quarter_end == quarter]
    return h.groupby("stock_id").manager_id.apply(set).to_dict()


def filers(filings, quarter):
    return sorted(filings.loc[filings.quarter_end == quarter, "manager_id"].unique())


def first_quarter_end(date):
    d = pd.Timestamp(date)
    q = pd.Timestamp(year=d.year, month=((d.month - 1) // 3 + 1) * 3, day=1) + pd.offsets.MonthEnd(0)
    return q.strftime("%Y-%m-%d")


def listing_outcomes(listings, companies, holdings, filings, min_holders=3):
    """For each listing: (investor universe, set of holders) at its first quarter-end on/after listing."""
    out = {}
    comp = companies.set_index("stock_id")
    for sid in listings:
        q = first_quarter_end(comp.loc[sid, "listing_date"])
        uni = filers(filings, q)
        hs = held_sets(holdings, q).get(sid, set()) & set(uni)
        if len(hs) >= min_holders:
            out[sid] = (uni, hs)
    return out


def test1(C, outcomes):
    res = {sid: auc(C.loc[uni, sid], [m in hs for m in uni]) for sid, (uni, hs) in outcomes.items()}
    return float(np.nanmean(list(res.values()))), res


def test2(C, outcomes, per_pair=None):
    scores = []
    for a, b in itertools.combinations(sorted(outcomes), 2):
        ua, ha = outcomes[a]
        ub, hb = outcomes[b]
        only_a = sorted((ha - hb) & set(ub))   # held A, did not hold B (and filed for B's quarter)
        only_b = sorted((hb - ha) & set(ua))
        if not only_a or not only_b:
            continue
        ia, jb = C.loc[only_a, a].to_numpy(), C.loc[only_b, b].to_numpy()
        ib, ja = C.loc[only_a, b].to_numpy(), C.loc[only_b, a].to_numpy()
        diff = (ia[:, None] + jb[None, :]) - (ib[:, None] + ja[None, :])
        scores.append(((diff > 0) + 0.5 * (diff == 0)).mean())
        if per_pair is not None:
            per_pair[(a, b)] = scores[-1]
    return float(np.mean(scores)) if scores else np.nan, len(scores)


def test3(C, holdings, filings, base_q, target_q, trading, exclude=(), min_new=3, n_groups=4, per_investor=None):
    """New holdings at target_q among stocks trading at base_q that an investor did not hold at base_q."""
    hb = holdings[holdings.quarter_end == base_q]
    ht = holdings[holdings.quarter_end == target_q]
    base = set(zip(hb.manager_id, hb.stock_id))
    targ = set(zip(ht.manager_id, ht.stock_id))
    inv = filers(filings, target_q)
    trading = [s for s in trading if s not in set(exclude)]
    new_holders = pd.Series({s: sum((m, s) in targ and (m, s) not in base for m in inv) for s in trading})
    elig = new_holders[new_holders >= min_new].index.tolist()
    pop = hb[hb.stock_id.isin(elig)].groupby("stock_id").manager_id.nunique().reindex(elig).fillna(0)
    grp = pd.qcut(pop.rank(method="first"), n_groups, labels=False)
    per_inv = []
    for m in inv:
        aucs = []
        for g in range(n_groups):
            cand = [s for s in grp.index[grp == g] if (m, s) not in base]
            if len(cand) < 3:
                continue
            y = [(m, s) in targ for s in cand]
            a = auc(C.loc[m, cand], y)
            if not np.isnan(a):
                aucs.append(a)
        if aucs:
            per_inv.append(np.mean(aucs))
            if per_investor is not None:
                per_investor[m] = per_inv[-1]
    return float(np.mean(per_inv)), len(per_inv)


def bootstrap(units_a, units_b=None, kind="plain", n_boot=2000, seed=0):
    """Bootstrap standard error of a test score (units_b None) or of the paired difference a - b.
    kind="plain": units are independent (listings for Test 1, investors for Test 3).
    kind="pairs": units are listing pairs (Test 2); listings are resampled and a pair counts with the
    product of its two listings' multiplicities, because pairs sharing a listing are not independent."""
    rng = np.random.RandomState(seed)
    keys = sorted(units_a)
    a = np.array([units_a[k] for k in keys])
    d = a - (np.array([units_b[k] for k in keys]) if units_b is not None else 0)
    stats = []
    if kind == "plain":
        for _ in range(n_boot):
            stats.append(d[rng.randint(0, len(d), len(d))].mean())
    else:
        listings = sorted({x for k in keys for x in k})
        idx = {l: i for i, l in enumerate(listings)}
        pa = np.array([idx[k[0]] for k in keys])
        pb = np.array([idx[k[1]] for k in keys])
        for _ in range(n_boot):
            cnt = np.bincount(rng.randint(0, len(listings), len(listings)), minlength=len(listings))
            w = cnt[pa] * cnt[pb]
            if w.sum() > 0:
                stats.append((w * d).sum() / w.sum())
    return float(d.mean()), float(np.std(stats))
