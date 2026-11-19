# Design

## Data and cutoffs
**Holdings:** 13F rows filed by 2026-03-02. Target: the recency-weighted (0.8^age) share of quarters in which investor m reported stock s, over quarters where m filed and s was listed.

**External data** (`external.parquet`, from EDGAR via `fetch_edgar.py` and `build_external.py`):
- **One filing per company:** the latest 10-K/20-F before 2025-07-01 (valid in back-tests too) or, for listings, the latest S-1/F-1 before listing; never 424Bs or confidential drafts.
- **Text:** the first 1,500 words of *Business* or the *Prospectus Summary*, with the company's name masked, embedded locally with pinned `e5-large-v2`.
- **Offering terms:** deal size, small-cap exchange, top-five bookrunner, foreign address.
- **SIC code:** stored, unused (it hurt the back-tests; see below).

## Model (`model.py`)
**Stage 1 (stocks with 13F history).** Unit vectors fitted to the quantity the bank ranks by: a logistic loss on `a + 6·cos(u_m, v_s)` for holdings, plus 0.2 × the same loss (own intercept) on the Test 3 event, i.e. whether m held s one or two quarters after not holding it. The fixed temperature regularises (a cosine has no norm to penalise); L-BFGS starts from the top-16 SVD.

**Stage 2 (listings).** With the investor vectors fixed, a convex ridge-logistic map `W` sends features `x_s` to a direction, with logit `c_s + u_m·W x_s`; a 2026 listing gets `W x_s`. The free intercept `c_s` absorbs popularity, so `W` learns *who* holds a stock, not *how many*. `x_s` has separate intercepts for established stocks and listings, 8 text principal components (PCA fitted on stocks with history only) and, for listings only, the offering terms.

**Why a listing intercept.** IPO buyers are a distinct group (Soros, Gilder Gagnon, Tiger Global, HBK); from holdings alone listings sit near the mega-caps (Test 1: 0.34 in A). The intercept learns that direction; text adds the sector tilt (healthcare specialists for biotechs, small-cap value for banks).

**Reproducibility:** each L-BFGS fit is finished with gradient-based Newton steps, and the tables are rotated onto fixed-sign principal axes (cosines unchanged); perturbing the start moves them by under 1e-10.

## Choosing (`backtest.py`, `evaluate.py`)
`evaluate.py` re-implements the tests; features are refitted within each split.
- **Back-test A** (decision date 2025-08-15): train on March–June holdings and score the later 2025 listings; Test 3 runs June→December, the real horizon.
- **Back-test B** (decision date 2025-11-17): adds September holdings, including the first holders of July–September listings; scores December listings.
- **CV:** 2025H2 listings hidden in four folds at the real decision date (extra evidence, not a historical test).

| T1, T2, T3 | back-test A | back-test B | CV (T1, T2) |
|---|---|---|---|
| holdings only | .34 .50 .645 | .91 .50 .662 | .87 .50 |
| + text, terms | .74 .74 .645 | .92 .76 .662 | .88 .67 |
| **final** | **.74 .73 .650** | **.92 .75 .670** | **.87 .67** |

**Uncertainty.** Bootstrap standard errors resample listings (Tests 1–2) or investors (Test 3).
- **Levels are imprecise:** Test 1 ±0.04 (A, 14 listings), ±0.02 (CV, 22); Test 2 ±0.05–0.06; Test 3 ±0.02. B's Test 2 (6 pairs, ±0.24) is ignored. Choices therefore rest on much tighter paired differences.
- **New-holding term:** Test 3 +0.006 ± 0.003 (pooled A, B); CV Test 1 −0.005 ± 0.002, no detectable change in A or B. Weight 0.3 cost more (−0.008 ± 0.002) for little extra gain (+0.007).

**Rejected options** (paired differences):
- **SIC sector dummies:** A Test 1 −0.044 ± 0.030, Test 2 −0.059 ± 0.045; text already captures sector.
- **Unshrunk text:** CV Test 1 fell to 0.85 and Test 2 to 0.63. The 2026 listings share the IPO-buyer component Test 1 needs (mean cosine 0.865 between them); scaling text by 0.3 balances this against Test 2.
- **bge-base:** worse on A alone; combined with e5, CV Test 2 +0.014 ± 0.036 but A −0.017.
- **Text-neighbour directions, equal investor weights:** no gain. (On Test 3, Stage 1 also beats item-kNN, 0.56/0.58.)

## Sixteen dimensions
Refitting with k dimensions:

| k | A: T1, T2, T3 | B: T1, T3 | CV: T1, T2 |
|---|---|---|---|
| 2 | .61 .62 .521 | .80 .553 | .66 .56 |
| 4 | .71 .73 .598 | .88 .621 | .86 .57 |
| 8 | .72 .70 .616 | .91 .653 | .87 .67 |
| 12 | .75 .74 .639 | .91 .666 | .88 .67 |
| 16 | .74 .73 .650 | .92 .670 | .87 .67 |

Tests 1–2 level off by k = 8–12 (the IPO-buyer factor is one direction); Test 3 keeps rising to 16, as finer tastes use the extra axes. Text gets only 8 components so the map, trained on about 150 stocks, does not overfit.
