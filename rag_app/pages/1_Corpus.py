"""Corpus Browser — browse all documents in the WattBot knowledge base."""

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Corpus Browser", page_icon="\U0001F4DA", layout="wide")
st.title("\U0001F4DA Corpus Browser")

_METADATA_CSV = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"

if not _METADATA_CSV.exists():
    st.error("metadata.csv not found.")
    st.stop()

df = pd.read_csv(_METADATA_CSV, encoding="utf-8", encoding_errors="replace")
df = df.dropna(subset=["title"])

# Summary header
n_docs = len(df)
types = df["type"].value_counts()
type_parts = [f"**{count}** {t}{'s' if count != 1 else ''}" for t, count in types.items()]
year_min, year_max = int(df["year"].min()), int(df["year"].max())

st.markdown(
    f"The knowledge base contains **{n_docs} documents** "
    f"({' and '.join(type_parts)}) spanning **{year_min}\u2013{year_max}**."
)

st.divider()

# Interactive table with clickable links
display_df = df[["title", "type", "year", "citation", "url"]].copy()
display_df.columns = ["Title", "Type", "Year", "Citation", "URL"]

st.dataframe(
    display_df,
    column_config={
        "URL": st.column_config.LinkColumn("URL", display_text="Open"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
    },
    width="stretch",
    hide_index=True,
    height=(len(display_df) + 1) * 35 + 3,
)
