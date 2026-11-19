# HW4: Who Should Be in the Room — submission

| File | Role |
|---|---|
| `stocks.csv`, `managers.csv` | the two 16-dimensional tables (pass `check_tables.py`) |
| `make_vectors.py` | reproduces both tables **offline** from `data/` and `external.parquet` |
| `model.py` | the two fitting stages (cosine-logistic factorisation; ridge-logistic side-information map) |
| `external.parquet` | everything brought in from outside: SIC codes, filing metadata, masked business-description excerpts, offering terms and sentence embeddings |
| `fetch_edgar.py`, `build_external.py` | how `external.parquet` was built (online; not needed to reproduce the tables) |
| `evaluate.py`, `backtest.py` | re-implementation of the three tests and the time-respecting back-tests used for every design choice |
| `design.md`, `answers.md` | write-ups |

Put the course's `data/` folder (the four supplied CSVs) next to these scripts; it is not committed, because `external.parquet` and the two tables are the only data files allowed in the repository.

Reproduce the tables (CPU, under a minute; needs numpy, pandas, scipy, pyarrow and torch):

```bash
python make_vectors.py
python check_tables.py stocks.csv managers.csv
```

Rebuild `external.parquet` from EDGAR (needs internet plus `sentence-transformers`, `beautifulsoup4` and `lxml`; the two approved models are loaded at their pinned revisions):

```bash
python fetch_edgar.py ../cache "Your Name your@email"
python build_external.py ../cache
```

Run the back-tests: `python backtest.py` (optional `key=value` overrides of `make_vectors.DEFAULT`).
