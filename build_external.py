"""build_external.py: step 2 of building external.parquet (runs locally on the cache from fetch_edgar.py).

For every stock it reads the one chosen EDGAR document and produces:
  * sic, sic2, state/country code of the business address (from the filing's own SGML header);
  * a business-description excerpt: Item 1 "Business" of an annual report (Item 4 of a 20-F), or the
    "Prospectus Summary" of a registration statement, first 1,500 words, with the company's own
    name and ticker replaced by "the Company" (the language model only ever sees the business text,
    never a question about a named company);
  * for listings only, the offering terms on the cover page of the latest pre-listing registration
    statement: shares offered, price range, implied deal size, exchange tier and lead underwriters;
  * sentence embeddings of the excerpt from two of the approved models, at the pinned revisions,
    run locally on CPU: e5-large-v2 and bge-base-en-v1.5 (mean of 250-word chunks, L2-normalised).

Usage: python build_external.py CACHE_DIR      -> writes external.parquet next to this script
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "e5": ("intfloat/e5-large-v2", "b322e09026e4ea05f42beadf4d661fb4e101d311", "query: "),
    "bge": ("BAAI/bge-base-en-v1.5", "70e8c52057794ed37381f7da00a03097927500a7", ""),
}
N_WORDS, CHUNK = 1500, 250
TOP_BANKS = ["Goldman Sachs", "Morgan Stanley", "J.P. Morgan", "BofA Securities", "Citigroup"]
OTHER_MAJOR = ["Barclays", "Jefferies", "Evercore", "UBS", "Deutsche Bank", "Wells Fargo", "RBC Capital",
               "Piper Sandler", "TD Cowen", "Leerink", "William Blair", "Stifel", "Raymond James", "Mizuho",
               "Keefe, Bruyette", "KBW", "Truist", "BMO Capital", "Guggenheim", "Cantor", "Needham",
               "Baird", "Oppenheimer", "Nomura", "HSBC", "Santander", "Mizuho", "SMBC", "Moelis", "Allen & Company"]
SUFFIX = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|l\.p|n\.v|s\.a|ag|se|" \
         r"holdings?|group|the|class [a-z]|/[a-z]{2}|new)\b\.?"
COMMON = {"first", "central", "commercial", "american", "general", "united", "national", "international",
          "applied", "rare", "silver", "smart", "black", "park", "space", "exploration", "technologies",
          "energy", "medical", "health", "life", "global", "capital", "financial", "pattern", "legence",
          "figure", "gemini", "neptune", "alliance", "phoenix", "andersen", "cardinal", "sunshine", "pershing",
          "guardian", "madison", "seaport", "odyssey", "liftoff", "applied", "home", "best", "service"}


def html_text(raw, limit=None):
    if limit:
        raw = raw[:limit]
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    txt = soup.get_text("\n")
    txt = txt.replace("\xa0", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    return re.sub(r"\n\s*\n+", "\n", txt)


TOC_WORDS = r"item\s*\d+\s*[a-d]?\b|risk factors|use of proceeds|dividend policy|capitalization|dilution|" \
            r"properties|legal proceedings|management's discussion|unresolved staff|selected financial"


def is_toc(txt, i):
    """A heading in a table of contents is followed by more headings, not by prose."""
    return len(re.findall(TOC_WORDS, txt[i + 30:i + 1500].lower())) >= 4


def section(txt, form):
    """Start of the business section: the first heading occurrence that is not in a table of contents."""
    low = txt.lower()
    if form.startswith("10-K"):
        pat = r"item\s*1\s*[\.:\-–—]?\s*business"
    elif form == "20-F":
        pat = r"item\s*4\s*[\.:\-–—]?\s*information on the company"
    else:
        # registration statement: the summary always opens with "This summary highlights ...";
        # the business overview follows the boilerplate paragraph
        m = re.search(r"this summary highlights", low)
        if m:
            o = re.search(r"\n\s*(business overview|overview|our company|our business|company overview|who we are)\s*\n",
                          low[m.start():m.start() + 8000])
            return txt[m.start() + (o.start() if o else 800):]
        pat = r"prospectus summary|\n\s*summary\s*\n"
    for pats in (pat, r"\n\s*(business overview|overview|our business|our company|business)\s*\n"):
        for m in re.finditer(pats, low):
            xref = re.match(r"[\s\.]*(and|or|of|in|to|for|,|\))\b", low[m.end():m.end() + 12])
            if m.start() > 2000 and not xref and not is_toc(low, m.start()):
                return txt[m.start():]
    return txt[len(txt) // 10:]


def mask(text, name, ticker):
    core = re.sub(SUFFIX, " ", name.lower().replace(",", " "))
    core = re.sub(r"\s+", " ", core).strip(" .&-")
    pats = []
    if len(core) >= 3:
        pats.append(re.escape(core).replace(r"\ ", r"[\s\-]+"))
    for p in pats:
        text = re.sub(r"\b" + p + r"(?:'s)?\b", "the Company", text, flags=re.I)
    # the first word of the name alone (e.g. "Cerebras"), only when it is used as a proper noun:
    # capitalised in the text and (almost) never written in lower case
    first = core.split(" ")[0] if core else ""
    if len(first) >= 4 and first not in COMMON:
        cap = len(re.findall(r"\b" + re.escape(first.capitalize()) + r"\b|\b" + re.escape(first.upper()) + r"\b", text))
        lower = len(re.findall(r"\b" + re.escape(first) + r"\b", text))
        if cap > 0 and lower <= 0.1 * cap:
            text = re.sub(r"\b(?:" + re.escape(first.capitalize()) + "|" + re.escape(first.upper()) + r")(?:'s)?\b",
                          "the Company", text)
    if isinstance(ticker, str) and len(ticker) >= 3:
        text = re.sub(r"\b" + re.escape(ticker) + r"\b", "the Company", text)
    return text


def words(text, n):
    w = re.findall(r"[A-Za-z][A-Za-z0-9'\-\.&%$,]*|\d[\d,\.]*%?", text)
    return w[:n]


def num(s):
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def offering_terms(cover):
    """Shares offered, price range and exchange from a registration statement's cover page."""
    c = re.sub(r"\s+", " ", cover)
    out = {}
    m = re.search(r"between \$\s?([\d\.,]+) and \$\s?([\d\.,]+)", c)
    if m:
        out["price_low"], out["price_high"] = num(m.group(1)), num(m.group(2))
    else:
        m = re.search(r"(?:initial public )?offering price (?:per [\w ]{0,40}?)?(?:is|of|will be)? ?(?:expected to be )?"
                      r"\$\s?([\d\.,]+)", c, re.I) or re.search(r"(?:assumed|fixed) (?:public )?(?:offering )?price of \$\s?([\d\.,]+)", c, re.I)
        if m:
            try:
                out["price_low"] = out["price_high"] = num(m.group(1))
            except ValueError:
                pass
    m = re.search(r"(?:offering|offer|selling)\s+([\d,]{5,})\s+(?:shares|ordinary shares|American Depositary|ADSs|units|"
                  r"Class [A-Z] (?:common|ordinary)|common)", c, re.I) or \
        re.search(r"([\d,]{6,})\s+(?:shares of (?:our )?(?:Class [A-Z] )?(?:common stock|ordinary shares)|"
                  r"ordinary shares|American Depositary Shares|Class [A-Z] (?:common|ordinary) shares)", c, re.I)
    if m:
        out["shares_offered"] = num(m.group(1))
    low = c.lower()
    if "nasdaq global select" in low:
        out["exchange"] = "nasdaq_gs"
    elif "nasdaq global market" in low:
        out["exchange"] = "nasdaq_gm"
    elif "nasdaq capital market" in low:
        out["exchange"] = "nasdaq_cm"
    elif "nyse american" in low:
        out["exchange"] = "nyse_american"
    elif "new york stock exchange" in low or "nyse" in low:
        out["exchange"] = "nyse"
    elif "nasdaq" in low:
        out["exchange"] = "nasdaq_other"
    out["n_top_banks"] = sum(b.lower() in low for b in TOP_BANKS)
    out["n_major_banks"] = sum(b.lower() in low for b in TOP_BANKS + OTHER_MAJOR)
    return out


def header(path):
    h = open(path, encoding="latin-1").read()
    g = lambda tag: (re.search(rf"<{tag}>(.*)", h) or [None, None])[1]
    return dict(sic=g("ASSIGNED-SIC"), state_inc=g("STATE-OF-INCORPORATION"),
                biz_state=(re.search(r"<BUSINESS-ADDRESS>.*?<STATE>(\w+)", h, re.S) or [None, None])[1])


def main():
    cache = sys.argv[1]
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype={"listing_date": str})
    docs = pd.read_csv(os.path.join(cache, "documents.csv"))
    recs = []
    for _, c in comp.iterrows():
        d = docs[docs.stock_id == c.stock_id].sort_values("rank")
        rec = dict(stock_id=c.stock_id)
        if d.empty or pd.isna(d.iloc[0].form):
            recs.append(rec)
            continue
        # text: the latest full-length document (an amendment that only adds exhibits is skipped)
        full = d[d["size"] >= 0.5 * d["size"].max()].iloc[0]
        rec.update(form=full.form, filing_date=full.filing_date, accession=full.accession)
        rec.update(header(d.iloc[0].hdr_path))
        raw = open(full.doc_path, "rb").read().decode("utf-8", "ignore")
        txt = html_text(raw)
        sec = section(txt, full.form)
        rec["text"] = " ".join(words(mask(sec, c.company_name, c.ticker), N_WORDS))
        if c.status != "established":
            for _, dd in d.iterrows():   # latest pre-listing statement whose cover states the terms
                cov = html_text(open(dd.doc_path, "rb").read().decode("utf-8", "ignore"), limit=400000)[:30000]
                t = offering_terms(cov)
                if "price_low" in t and "shares_offered" in t:
                    rec.update(t, terms_accession=dd.accession, terms_form=dd.form)
                    break
                if dd["rank"] == 0:
                    rec.update(t, terms_accession=dd.accession, terms_form=dd.form)
        print(c.stock_id, c.ticker, rec.get("form"), rec.get("sic"), len(rec.get("text", "").split()),
              {k: rec.get(k) for k in ("shares_offered", "price_low", "price_high", "exchange", "n_top_banks")})
        recs.append(rec)
    ext = pd.DataFrame(recs)
    ext["deal_usd"] = ext.shares_offered * (ext.price_low + ext.price_high) / 2

    from sentence_transformers import SentenceTransformer
    for key, (repo, rev, prefix) in MODELS.items():
        m = SentenceTransformer(repo, revision=rev, device="cpu")
        embs = []
        for t in ext.text.fillna(""):
            w = t.split()
            chunks = [prefix + " ".join(w[i:i + CHUNK]) for i in range(0, max(len(w), 1), CHUNK)] or [prefix]
            e = m.encode(chunks, normalize_embeddings=True, batch_size=8).mean(axis=0)
            embs.append(e / np.linalg.norm(e))
        E = pd.DataFrame(np.array(embs), columns=[f"{key}_{i}" for i in range(len(embs[0]))])
        ext = pd.concat([ext, E], axis=1)
    ext.to_parquet(os.path.join(HERE, "external.parquet"), index=False)
    print("wrote external.parquet", ext.shape)


if __name__ == "__main__":
    main()
