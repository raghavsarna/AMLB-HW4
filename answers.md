# HW4 answers

**1. AUC and ranks.**
AUC is the Mann–Whitney statistic: the share of (holder, non-holder) pairs in which the holder has the higher cosine. Only each pair's order matters, so any strictly increasing transformation of the cosines leaves AUC unchanged. A non-rank alternative is the log-loss of `p = σ(a + b·cos)`. It shows whether a score of 0.4 means a 40 % chance, which tells the bank how many calls are worthwhile. It misleads when calibration fitted on 2025 deals transfers poorly: base rates vary by deal, so a perfect ranker can score badly.

**2. Objective, missing holdings, 16 dimensions.**
Stage 1, over unit vectors: `Σ w·BCE(y_ms, σ(a + 6·cos(u_m, v_s))) + 0.2·Σ BCE(n_ms, σ(a' + 6·cos))`. `y_ms` is the recency-weighted share of quarters m reported s, and `n_ms` the share of observed transitions where m newly added s. Stage 2, for listings, with `u_m` fixed: `Σ w·BCE(y_ms, σ(c_s + u_m·W x_s)) + λ‖W‖²`, giving `v_s = W x_s`. A missing holding is a negative if m filed and s was listed, else weight 0. Sixteen dimensions lose niche tastes, multi-sector investors, position size.

**3. Identical stock vectors.**
Yes. With one shared stock vector, every listing ranks investors in the same order. If that order tracks appetite for new listings, Test 1 is high: my "new-listing" direction alone scored about 0.87 in cross-validation. Test 2 scores exactly 0.5, because `cos(i,A)+cos(j,B)` equals `cos(i,B)+cos(j,A)` term by term, so every comparison ties. A high Test 1 can therefore measure who buys IPOs at all, not who suits this company. Read it alongside Test 2, which isolates the match.

**4. Two other decisions.**
(a) *Block trades and follow-on offerings*: finding buyers for a large block of an existing stock. The vectors encode revealed preferences for similar companies, beyond current holders. I would first want back-tests on past block allocations, plus position-size data, since cosine ignores capacity. (b) *Peer groups* for valuation or investor-relations benchmarking: companies close in investor space share a shareholder base that industry codes can miss. I would check that these peers co-move in returns and multiples better than SIC peers, and stay stable across quarters.

**5. Reported positions vs. interest.**
(i) A 13F is a quarter-end snapshot filed up to 45 days later. Trades within the quarter are invisible, and window dressing distorts it. (ii) A position may not express interest. It may be an IPO allocation about to be flipped, a pre-IPO stake under lock-up, spin-off shares, an index sleeve, or one leg of a hedge (shorts are not reported). A better test would use the bank's book-building records (indications of interest and allocations). Alternatively, count only holdings that persist or grow over two consecutive quarters.

**6. If every bank used such tools.**
Mostly no. The tools would become table stakes rather than an advantage. All banks read the same public 13Fs, so they would call the same "IPO-buyer" investors, and the edge would move to private data: order books, roadshow feedback, relationships. The biggest effect would be on investors. Those on every list gain allocation power and better terms, while smaller investors are overlooked, so shareholder bases narrow. Second-order effects matter less: issuers tailoring prospectus language, and models trained on outcomes they helped create becoming self-fulfilling.
