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
        print("Rows in validation sample:", len(val))
        return
    
    # fun 2: evaluate once I've labeled the validation file
    val = pd.read_csv(VAL_PATH)

    if "manual_department" not in val.columns:
        raise ValueError("validationSample.csv must contain a manual_department column.")

    # treat empty strings as missing labels (common if you haven't finished labeling)
    val["manual_department"] = val["manual_department"].astype(str).str.strip()
    val = val[val["manual_department"] != ""].copy()

    # you want enough labeled rows for the accuracy to mean something
    if len(val) < 50:
        raise ValueError(
            f"Only {len(val)} labeled validation rows found. "
            "Label at least ~100 for a meaningful check."
        )

    # correctness = manual label matches model prediction
    val["correct"] = (val["manual_department"] == val["pred_dept"])
    acc = val["correct"].mean()

    # confusion matrix (counts) for where the system mixes up departments
    confusion = pd.crosstab(val["manual_department"], val["pred_dept"])

    # collect misclassifications so we can inspect patterns and improve rules
    errors = val[~val["correct"]].copy()

    # Write a simple report we can paste into the write-up
    lines = []
    lines.append(f"Validation rows labeled: {len(val)}")
    lines.append(f"Overall accuracy: {acc:.4f}")
    lines.append("")
    lines.append("Confusion matrix (counts):")
    lines.append(confusion.to_string())
    lines.append("")
    lines.append("Sample misclassifications (up to 25):")

    if len(errors) == 0:
        lines.append("None (no errors in labeled sample).")
    else:
        show = errors[[
            "title", "pred_dept", "manual_department", "pred_method", "pred_evidence"
        ]].head(25)
        lines.append(show.to_string(index=False))

    REPORT_PATH.write_text("\n".join(lines))

    print(f"Wrote report to {REPORT_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Errors: {len(errors)} / {len(val)}")


if __name__ == "__main__":
    main()