import pandas as pd
from pathlib import Path


# Part 2: Manual training data and rules-based classification

CLEAN_PATH  = Path("output/cleanRoles.csv")
MANUAL_PATH = Path("output/manualLabels.csv")
OUT_PATH    = Path("output/rulesClassified.csv")


PHRASE_RULES = [
    ("investment banking", "Investment Banking"),
    ("capital markets", "Investment Banking"),
    ("leveraged finance", "Investment Banking"),
    ("m&a", "Investment Banking"),

    ("equity research", "Research"),
    ("credit research", "Research"),

    ("credit risk", "Risk"),
    ("market risk", "Risk"),
    ("model risk", "Risk"),
    ("operational risk", "Risk"),

    ("anti money laundering", "Compliance/Legal"),
    ("aml", "Compliance/Legal"),
    ("kyc", "Compliance/Legal"),
    ("compliance", "Compliance/Legal"),

    ("middle office", "Operations"),
    ("back office", "Operations"),
    ("trade support", "Operations"),
    ("settlements", "Operations"),
    ("reconciliation", "Operations"),

    ("software engineer", "Technology"),
    ("software developer", "Technology"),
    ("information security", "Technology"),

    ("financial advisor", "Wealth Management"),
    ("private banker", "Wealth Management"),
]

KEYWORD_RULES = {
    "Investment Banking": {"ibd", "banking", "underwriter", "underwriting", "advisory", "restructuring"},
    "Sales & Trading": {"trader", "trading", "sales", "equities", "options", "futures", "fx", "rates", "derivatives"},
    "Research": {"research"},
    "Risk": {"risk"},
    "Compliance/Legal": {"compliance", "aml", "kyc", "regulatory", "surveillance", "legal"},
    "Operations": {"operations", "settlement", "settlements", "clearing", "reconciliation", "processing"},
    "Technology": {"software", "developer", "engineer", "devops", "infrastructure", "platform", "systems", "security"},
    "Finance/Accounting": {"finance", "accounting", "controller", "audit", "tax", "reporting"},
    "Wealth Management": {"advisor", "planner", "broker", "wealth"},
    "HR": {"recruiter", "recruiting", "talent", "human", "resources"},
    "Treasury": {"treasury", "liquidity", "cash"},
    "Product/Strategy": {"product", "strategy", "strategic"},
}

# classifying function for the most frequent uses
def classify_rules(title_norm: str):
    t = str(title_norm)

    # phrase rules: first hit wins 
    for phrase, dept in PHRASE_RULES:
        if phrase in t:
            return dept, "phrase", phrase

    # keyword scoring: most freq. dept.
    words = set(t.split())
    best_dept, best_score, best_hits = "Unknown", 0, []

    for dept, kws in KEYWORD_RULES.items():
        hits = words.intersection(kws)
        score = len(hits)
        if score > best_score:
            best_dept, best_score, best_hits = dept, score, sorted(hits)

    if best_score > 0:
        return best_dept, "keyword", ",".join(best_hits)

    # incase of anything
    return "Unknown", "none", ""


def main():
    df = pd.read_csv(CLEAN_PATH)
    print("Loaded clean data:", df.shape)

    # run 1: create manualLabels.csv if doesnt exist
    if not MANUAL_PATH.exists():
        sample = df.sample(800, random_state=42)[["company", "title", "count", "title_norm"]].copy()
        sample["manual_department"] = ""  # YOU fill this in
        sample.to_csv(MANUAL_PATH, index=False)
        print(f"Created {MANUAL_PATH}. Label 'manual_department' and re-run part2.py.")
        return

    # run 2: ensure manualLabels.csv is real and has the labels
    labels = pd.read_csv(MANUAL_PATH)

    required_cols = {"title_norm", "manual_department"}
    if not required_cols.issubset(labels.columns):
        raise ValueError(f"{MANUAL_PATH} must contain columns {required_cols}. Found: {labels.columns.tolist()}")

    labeled_count = (labels["manual_department"].astype(str).str.strip() != "").sum()
    if labeled_count < 50:
        raise ValueError(f"Not enough manual labels yet ({labeled_count}). Label at least ~200+ then re-run.")

    # make sure to apply classifier to full dataset
    df[["pred_dept", "pred_method", "pred_evidence"]] = df["title_norm"].apply(
        lambda x: pd.Series(classify_rules(x))
    )

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # output
    print("Saved predictions:", OUT_PATH)
    print("Unknown rate:", round((df["pred_dept"] == "Unknown").mean(), 4))
    print(df["pred_dept"].value_counts().head(15))


if __name__ == "__main__":
    main()
