"""fetch_edgar.py: step 1 of building external.parquet (needs internet; not needed by make_vectors.py).

For each of the 197 companies it downloads, from SEC EDGAR, the filing index (submissions JSON),
picks one document that was public before the relevant cutoff, and saves the raw document and
its SGML header (which carries the SIC code assigned at that filing) to a local cache.

Document choice (all dates are EDGAR filing dates):
  * established stocks: latest annual report (10-K / 20-F / 40-F) filed before 2025-07-01, so the
    same text is valid for the 2025 back-tests and for the final tables (cutoff 2026-03-02).
  * 2025H2 and 2026 listings: registration statements (S-1, F-1, S-11, F-10, N-2, 10-12B and their
    amendments) filed strictly before the listing date. 424B prospectuses (filed at or after pricing)
    and confidential DRS drafts are never used. We keep the latest full-length one for the text and
    the latest few pre-listing ones for the offering terms.

Usage: python fetch_edgar.py CACHE_DIR "Your Name your@email"
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANNUAL = {"10-K", "10-K405", "20-F", "40-F", "10-KT"}
REG = {"S-1", "S-1/A", "F-1", "F-1/A", "S-11", "S-11/A", "F-10", "F-10/A", "N-2", "N-2/A",
       "10-12B", "10-12B/A", "S-4", "S-4/A", "F-4", "F-4/A", "1-A", "1-A/A", "20FR12B", "20FR12B/A"}
ESTABLISHED_CUTOFF = "2025-07-01"

_last = [0.0]


def get(url, ua, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    wait = 0.15 - (time.time() - _last[0])  # stay well below 10 requests per second
    if wait > 0:
        time.sleep(wait)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _last[0] = time.time()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return data
        except Exception as e:  # retry politely
            print("  retry", url, e)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(url)


def filings_table(cik, ua, cache):
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua,
                         os.path.join(cache, "sub", f"{cik}.json")))
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        name = extra["name"]
        frames.append(pd.DataFrame(json.loads(get(f"https://data.sec.gov/submissions/{name}", ua,
                                                  os.path.join(cache, "sub", name)))))
    f = pd.concat(frames, ignore_index=True)
    return sub, f[["form", "filingDate", "accessionNumber", "primaryDocument", "size"]]


def main():
    cache, ua = sys.argv[1], sys.argv[2]
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype={"listing_date": str})
    rows = []
    for _, c in comp.iterrows():
        cik = int(c.sec_cik)
        sub, f = filings_table(cik, ua, cache)
        if c.status == "established":
            cand = f[f.form.isin(ANNUAL) & (f.filingDate < ESTABLISHED_CUTOFF)]
        else:
            cand = f[f.form.isin(REG) & (f.filingDate < c.listing_date)]
        cand = cand.sort_values(["filingDate", "accessionNumber"], ascending=False)
        cand = cand.head(1 if c.status == "established" else 4)
        if cand.empty:
            print("NO DOCUMENT", c.stock_id, c.company_name)
            rows.append(dict(stock_id=c.stock_id, cik=cik, form=None))
            continue
        for rank, (_, d) in enumerate(cand.iterrows()):
            acc = d.accessionNumber
            nod = acc.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nod}"
            doc_path = os.path.join(cache, "doc", f"{c.stock_id}_{acc}_{d.primaryDocument}")
            get(f"{base}/{d.primaryDocument}", ua, doc_path)
            hdr_path = os.path.join(cache, "hdr", f"{c.stock_id}_{acc}.hdr.sgml")
            if rank == 0:
                get(f"{base}/{acc}.hdr.sgml", ua, hdr_path)
            rows.append(dict(stock_id=c.stock_id, cik=cik, form=d.form, filing_date=d.filingDate,
                             accession=acc, document=d.primaryDocument, size=int(d["size"]), rank=rank,
                             doc_path=doc_path, hdr_path=hdr_path if rank == 0 else None,
                             sub_sic=sub.get("sic"), sub_state=sub.get("stateOfIncorporation")))
        print(c.stock_id, c.ticker, c.status, cand.iloc[0].form, cand.iloc[0].filingDate, len(cand))
    pd.DataFrame(rows).to_csv(os.path.join(cache, "documents.csv"), index=False)


if __name__ == "__main__":
    main()
