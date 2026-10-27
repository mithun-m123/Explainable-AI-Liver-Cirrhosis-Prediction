"""
Dataset preparation pipeline - v2.
Adds N_Days as a feature (strong predictor of stage).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DST    = ROOT / "datasets"
OUTPUT = DST / "final_train.csv"

# 21 features now — added N_Days
SCHEMA = ["Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
          "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
          "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
          "N_Days", "Stage"]

CAT_COLS_IDX = [1, 2, 3, 4, 5]  # Sex, Ascites, Hepatomegaly, Spiders, Edema


def load_mayo(path):
    df = pd.read_csv(path).dropna(subset=["Stage"]).copy()
    df = df[df["Stage"].isin([1, 2, 3, 4])]
    df["Age"]   = df["Age"] / 365.25
    enc = {"F": 0, "M": 1}
    df["Sex"] = df["Sex"].map(enc).fillna(0)
    for col, mapping in [("Ascites",      {"N": 0, "Y": 1}),
                         ("Hepatomegaly", {"N": 0, "Y": 1}),
                         ("Spiders",      {"N": 0, "Y": 1}),
                         ("Edema",        {"N": 0, "S": 1, "Y": 2})]:
        df[col] = df[col].map(mapping).fillna(0)
    df["Stage"] = df["Stage"].astype(int) - 1
    # N_Days already present in Mayo dataset
    cols = [c for c in SCHEMA if c in df.columns]
    result = df[cols].copy()
    print(f"  Mayo PBC:       {len(result)} rows  stage dist: {result['Stage'].value_counts().sort_index().to_dict()}")
    return result


def load_merged(path):
    df = pd.read_csv(path).copy()
    df.columns = [c.lower() for c in df.columns]
    rename = {
        "age": "Age", "sex": "Sex", "ascites": "Ascites",
        "hepatomegaly": "Hepatomegaly", "spiders": "Spiders", "edema": "Edema",
        "bilirubin": "Bilirubin", "cholesterol": "Cholesterol", "albumin": "Albumin",
        "copper": "Copper", "alk_phos": "Alk_Phos", "ast": "SGOT",
        "triglycerides": "Tryglicerides", "platelets": "Platelets",
        "prothrombin": "Prothrombin", "stage": "Stage"
    }
    df.rename(columns=rename, inplace=True)
    df = df[[c for c in SCHEMA if c in df.columns]].copy()
    df = df.dropna(subset=["Stage"])
    df["Stage"] = df["Stage"].astype(int)
    df = df[df["Stage"].isin([0, 1, 2, 3])]
    if "Platelets" in df.columns and df["Platelets"].median() > 1000:
        df["Platelets"] = df["Platelets"] / 1000
    # No N_Days — set to median of Mayo data (~1900)
    if "N_Days" not in df.columns:
        df["N_Days"] = 1900.0
    print(f"  Merged dataset: {len(df)} rows  stage dist: {df['Stage'].value_counts().sort_index().to_dict()}")
    return df


def load_synthetic_deduped(path):
    df = pd.read_csv(path).dropna(subset=["Stage"]).copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"  Synthetic deduped: {before} -> {after} rows (removed {before-after} dupes)")
    if df["Age"].median() > 1000:
        df["Age"] = df["Age"] / 365.25
    enc = {"F": 0, "M": 1}
    df["Sex"] = df["Sex"].map(enc).fillna(0)
    for col, mapping in [("Ascites",      {"N": 0, "Y": 1}),
                         ("Hepatomegaly", {"N": 0, "Y": 1}),
                         ("Spiders",      {"N": 0, "Y": 1}),
                         ("Edema",        {"N": 0, "S": 1, "Y": 2})]:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    df["Stage"] = df["Stage"].astype(int) - 1
    df = df[df["Stage"].isin([0, 1, 2, 3])]
    cols = [c for c in SCHEMA if c in df.columns]
    df = df[cols].copy()
    if "N_Days" not in df.columns:
        df["N_Days"] = 1900.0
    print(f"  Synthetic unique: {len(df)} rows  stage dist: {df['Stage'].value_counts().sort_index().to_dict()}")
    return df


def add_clinical_features(df):
    df = df.copy()
    def safe_div(a, b, default=0.0):
        return np.where(b != 0, a / b, default)

    sgot      = df["SGOT"].clip(lower=1)
    platelets = df["Platelets"].clip(lower=1)
    albumin   = df["Albumin"].clip(lower=0.1)
    bilirubin = df["Bilirubin"].clip(lower=0.1)
    age       = df["Age"].clip(lower=1)

    df["FIB4"]     = (age * sgot) / (platelets * np.sqrt(sgot))
    df["APRI"]     = safe_div(sgot / 40.0, platelets / 100.0)
    df["ALBI"]     = np.log10(bilirubin * 17.1) * 0.66 + albumin * (-0.085)
    df["Bili_Alb"] = safe_div(bilirubin, albumin)
    df["PT_proxy"] = df["Prothrombin"] / 11.0

    # Log-transform N_Days to reduce skew
    if "N_Days" in df.columns:
        df["Log_NDays"] = np.log1p(df["N_Days"].clip(lower=0))

    for feat in ["FIB4", "APRI", "ALBI", "Bili_Alb", "PT_proxy"]:
        q99 = df[feat].quantile(0.99)
        q01 = df[feat].quantile(0.01)
        df[feat] = df[feat].clip(lower=q01, upper=q99)

    print(f"  Clinical features added: FIB4, APRI, ALBI, Bili_Alb, PT_proxy, Log_NDays")
    return df


def apply_smotenc(X, y, cat_indices):
    from imblearn.over_sampling import SMOTENC
    print(f"\n  Before SMOTENC: {len(y)} samples  dist: {pd.Series(y).value_counts().sort_index().to_dict()}")
    min_count = pd.Series(y).value_counts().min()
    k = min(5, min_count - 1)
    if k < 1:
        print("  Skipping SMOTENC")
        return X, y
    smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  After  SMOTENC: {len(y_res)} samples  dist: {pd.Series(y_res).value_counts().sort_index().to_dict()}")
    return X_res, y_res


def main():
    print("=" * 60)
    print("DATASET PREPARATION PIPELINE v2 (with N_Days)")
    print("=" * 60)

    frames = []

    mayo_path = DST / "cirrhosis.csv"
    if mayo_path.exists():
        print("\n[1] Loading Mayo Clinic PBC dataset...")
        frames.append(load_mayo(mayo_path))

    merged_path = DST / "merged_liver_dataset.csv"
    if merged_path.exists():
        print("\n[2] Loading merged liver dataset...")
        try:
            frames.append(load_merged(merged_path))
        except Exception as e:
            print(f"  SKIP: {e}")

    synth_path = DST / "liver-cirrhosis-stage" / "liver_cirrhosis.csv"
    if synth_path.exists():
        print("\n[3] Loading deduplicated synthetic dataset...")
        try:
            frames.append(load_synthetic_deduped(synth_path))
        except Exception as e:
            print(f"  SKIP: {e}")

    print("\n[4] Combining datasets...")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates()

    base_cols = [c for c in SCHEMA if c != "Stage"]
    for col in base_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["Stage"])
    df["Stage"] = df["Stage"].astype(int)
    df = df[df["Stage"].isin([0, 1, 2, 3])]
    print(f"  After cleanup: {len(df)} rows  dist: {df['Stage'].value_counts().sort_index().to_dict()}")

    print("\n[5] Adding clinical features...")
    df = add_clinical_features(df)

    print("\n[6] Applying SMOTENC...")
    feature_cols = [c for c in df.columns if c != "Stage"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Stage"].values.astype(int)

    cat_names = ["Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
    cat_idx   = [feature_cols.index(c) for c in cat_names if c in feature_cols]
    X_res, y_res = apply_smotenc(X, y, cat_idx)

    print("\n[7] Saving final training dataset...")
    final_df = pd.DataFrame(X_res, columns=feature_cols)
    final_df["Stage"] = y_res
    final_df.to_csv(OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — Saved: {OUTPUT}")
    print(f"Total rows    : {len(final_df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    print(f"Stage dist: {pd.Series(y_res).value_counts().sort_index().to_dict()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
