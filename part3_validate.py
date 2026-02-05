import pandas as pd
from pathlib import Path


PRED_PATH = Path("output/rulesClassified.csv")      
VAL_PATH  = Path("output/validationSample.csv")     
REPORT_PATH = Path("output/validationReport.txt")   


#  Build a stratified-ish validation set
def make_validation_sample(df, n_per_dept=15, n_unknown=40, seed=7):
    pieces = []

    # get list of predicted departments we actually have
    depts = df["pred_dept"].dropna().unique().tolist()

    # sample roughly the same number from each dept (except Unknown)
    for dept in sorted(depts):
        if dept == "Unknown":
            continue  # handle Unknown separately below

        sub = df[df["pred_dept"] == dept]
        if len(sub) == 0:
            continue

        # don't request more rows than exist in that dept
        k = min(n_per_dept, len(sub))
        pieces.append(sub.sample(k, random_state=seed))

    # add extra Unknown rows (these are typically the hardest / most ambiguous cases)
    unk = df[df["pred_dept"] == "Unknown"]
    if len(unk) > 0:
        k = min(n_unknown, len(unk))
        pieces.append(unk.sample(k, random_state=seed))

    # combine everything and shuffle so it's not grouped by department
    val = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed)
    return val

# load p2 predictions
def main():
    df = pd.read_csv(PRED_PATH)

    # quick safety check: make sure Part 2 produced the columns we expect
    required = {"company", "title", "title_norm", "pred_dept", "pred_method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{PRED_PATH} is missing columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )
    

    # run 1: create a validation sample for labeling
    if not VAL_PATH.exists():
        val = make_validation_sample(df, n_per_dept=15, n_unknown=40, seed=7)

        # keep only the columns that are helpful for manual labeling + later evaluation
        val = val[[
            "company", "title", "count", "title_norm",
            "pred_dept", "pred_method", "pred_evidence"
        ]].copy()

        # blank label column (you fill this in manually)
        val["manual_department"] = ""

        # make sure output folder exists
        VAL_PATH.parent.mkdir(exist_ok=True)

        # save for labeling
        val.to_csv(VAL_PATH, index=False)

        print(f"Created {VAL_PATH}.")
        print("Next step: fill in manual_department for each row, then re-run this script.")
        print("Rows in validation sample:", len(val))
        return