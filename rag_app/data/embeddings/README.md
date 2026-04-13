# Vector Databases

This directory holds `.db` files (SQLite vector databases) built by the indexing pipeline.

These files are **gitignored** because they're too large for GitHub (>100MB) and are
derived from the source data in `data/corpus/`.

## Build the index

```bash
cd vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
```

This creates `wattbot_jinav4.db` in this directory.
