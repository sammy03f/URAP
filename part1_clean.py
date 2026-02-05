import pandas as pd
import re

# Part 1: Data ingestion and preprocessing

# 1.1 Load the raw tab-delimited file
path = "/Users/samfsmacbookpro/Personal Projects/URAP/org_role_100K.txt"
df = pd.read_csv(path, sep="\t")
df.columns = ["company", "title", "count"]  # standardize column names

# 1.2 inspection 
print("Raw shape:", df.shape)
df.tail(10)
df.info()
print("Missing values:\n", df.isna().sum())

# most frequent raw values 
print("\nTop raw titles:\n", df["title"].astype(str).str.strip().value_counts().head(20))
print("\nTop raw companies:\n", df["company"].astype(str).str.strip().value_counts().head(20))

# 1.3 text normalization 
def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\u2013\u2014]", "-", s)          # normalize unicode dashes
    s = re.sub(r"[^a-z0-9\s&/\-]", " ", s)         # keep letters/numbers/space/&//-
    s = re.sub(r"\s+", " ", s).strip()             # collapse whitespace
    return s

df["company_norm"] = df["company"].apply(norm_text)
df["title_norm"] = df["title"].apply(norm_text)

# 1.4 remove clearly corrupted rows only
bad_titles_exact = {
    "?", "-", "", "company title", "company", "title", "retired"
}
df = df[~df["title_norm"].isin(bad_titles_exact)].copy()

# remove rows with missing strings
df = df[df["title_norm"].str.len() >= 2].copy()
df = df[df["company_norm"].str.len() >= 2].copy()

print("Cleaned shape:", df.shape)

# 1.5 quick exploratory overview 
print("\nTop normalized titles:\n", df["title_norm"].value_counts().head(40))

print("\nRandom sample (human inspection):")
print(df.sample(30, random_state=7)[["company", "title", "count"]].to_string(index=False))

# estimate prevalence of "seniority-only" titles
seniority_only = (
    r"^(analyst|associate|vice president|vp|director|managing director|intern|"
    r"summer analyst|summer intern|executive director|assistant vice president|avp)$"
)
print("\nSeniority-only rate:", round(df["title_norm"].str.match(seniority_only, na=False).mean(), 4))

clean_path = "/Users/samfsmacbookpro/Personal Projects/URAP/output/cleanRoles.csv"
df.to_csv(clean_path, index=False)
print("\nSaved:", clean_path)