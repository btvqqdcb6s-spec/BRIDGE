"""
TCGA Gene Expression Data Download via UCSC Xena
=================================================
Downloads RNA-seq (log2 RSEM+1) expression data + clinical variables for:
  - BRCA  (Breast Invasive Carcinoma,    ~1100 samples)
  - LUAD  (Lung Adenocarcinoma,          ~500 samples)
  - KIRC  (Kidney Renal Clear Cell Ca.,  ~530 samples)

Shared clinical variables (global confounders across all cancer types):
  - age_at_initial_pathologic_diagnosis
  - gender (binary)
  - pathologic_stage (ordinal I–IV)

These are appended to X so the DGP can use them as shared confounders,
mirroring the simulation structure where features 1 & 2 affect all contexts.

Output files (in ./tcga_data/):
  expression_{cohort}.parquet  — samples × genes, log2(RSEM+1)
  clinical_{cohort}.parquet    — clinical variables
  tcga_X.npz                   — (X, S, feature_names) ready for semi-synthetic DGP
  selected_genes.csv           — gene names + variance stats for prior construction

Usage:
  pip install requests pandas pyarrow tqdm
  python download_tcga.py
"""

import gzip
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

XENA_HOST = "https://tcga.xenahubs.net"
OUT_DIR = Path("tcga_data")
OUT_DIR.mkdir(exist_ok=True)

EXPRESSION_DATASETS = {
    "BRCA": "TCGA.BRCA.sampleMap/HiSeqV2",
    "LUAD": "TCGA.LUAD.sampleMap/HiSeqV2",
    "KIRC": "TCGA.KIRC.sampleMap/HiSeqV2",
}

CLINICAL_DATASETS = {
    "BRCA": "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
    "LUAD": "TCGA.LUAD.sampleMap/LUAD_clinicalMatrix",
    "KIRC": "TCGA.KIRC.sampleMap/KIRC_clinicalMatrix",
}

# Shared clinical columns to use as global confounders
# These are standard TCGA fields present in all cohorts
CLINICAL_COLS = {
    "age":    "age_at_initial_pathologic_diagnosis",
    "gender": "gender",
    "stage":  "pathologic_stage",
}

# Pathway genes required by the DGP — verified present in common_genes at runtime
REQUIRED_GENES = [
    # Confounders + effect modifiers
    "ESR1", "PIK3CA", "ERBB2", "BRCA1",   # BRCA
    "EGFR", "KRAS",   "STK11", "KEAP1",   # LUAD
    "VHL",  "PBRM1",  "BAP1",  "SETD2",   # KIRC
    # Instruments (affect treatment decision, not outcome)
    "MKI67", "TP53", "CDH1",              # BRCA: proliferation, tumor suppressor
    "ALK", "ROS1", "MET",                 # LUAD: tested in treatment workup
    "MTOR", "VEGFA", "KDM5C",            # KIRC: targeted therapy pathway genes
]

GENE_SELECTIONS = {
    "all": {"top_var": None, "out_npz": "tcga_X_all.npz",
            "out_csv": "selected_genes_all.csv"},
}


# ============================================================
# Helpers
# ============================================================

def xena_download_url(host, dataset):
    dataset_encoded = requests.utils.quote(dataset, safe="")
    return f"{host}/download/{dataset_encoded}"


def download_tsv(url, local_path, desc="downloading"):
    if local_path.exists():
        print(f"  [cache] {local_path.name}")
        return
    print(f"  {desc} → {local_path.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(local_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=local_path.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def read_xena_tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    try:
        df = pd.read_csv(path, sep=sep, index_col=0,
                         compression="gzip" if str(path).endswith(".gz") else None)
    except Exception:
        df = pd.read_csv(path, sep=sep, index_col=0)
    return df


# ============================================================
# Step 1: Download
# ============================================================

def download_all():
    for cohort, dataset in EXPRESSION_DATASETS.items():
        url = xena_download_url(XENA_HOST, dataset)
        for suffix in [".gz", ""]:
            dest = OUT_DIR / f"expression_{cohort}.tsv{suffix}"
            try:
                download_tsv(url + suffix if suffix else url,
                             dest, desc=f"{cohort} expression")
                break
            except requests.HTTPError:
                dest.unlink(missing_ok=True)

    for cohort, dataset in CLINICAL_DATASETS.items():
        url = xena_download_url(XENA_HOST, dataset)
        for suffix in [".gz", ""]:
            dest = OUT_DIR / f"clinical_{cohort}.tsv{suffix}"
            try:
                download_tsv(url + suffix if suffix else url,
                             dest, desc=f"{cohort} clinical")
                break
            except requests.HTTPError:
                dest.unlink(missing_ok=True)


# ============================================================
# Step 2: Parse
# ============================================================

def parse_expression():
    for cohort in EXPRESSION_DATASETS:
        out_path = OUT_DIR / f"expression_{cohort}.parquet"
        if out_path.exists():
            print(f"  [cache] expression_{cohort}.parquet")
            continue
        candidates = sorted(OUT_DIR.glob(f"expression_{cohort}.tsv*"))
        if not candidates:
            raise FileNotFoundError(f"No expression file for {cohort}")
        raw = read_xena_tsv(candidates[0])
        df = raw.T
        df.index.name = "sample_id"
        print(f"  {cohort}: {df.shape[0]} samples × {df.shape[1]} genes (raw)")
        df.to_parquet(out_path)
        print(f"  Saved {out_path.name}")


def parse_clinical():
    for cohort in CLINICAL_DATASETS:
        out_path = OUT_DIR / f"clinical_{cohort}.parquet"
        if out_path.exists():
            print(f"  [cache] clinical_{cohort}.parquet")
            continue
        candidates = sorted(OUT_DIR.glob(f"clinical_{cohort}.tsv*"))
        if not candidates:
            raise FileNotFoundError(f"No clinical file for {cohort}")
        df = read_xena_tsv(candidates[0])
        df.index.name = "sample_id"
        print(f"  {cohort}: {df.shape[0]} samples × {df.shape[1]} columns")
        df.to_parquet(out_path)
        print(f"  Saved {out_path.name}")


# ============================================================
# Step 3: Preprocess
# ============================================================

def encode_clinical(clin, cohort):
    """
    Extract and encode the three shared clinical variables.
    Returns a DataFrame with columns [age, gender, stage], index = sample_id.
    Missing values are imputed with the column median/mode.
    """
    out = pd.DataFrame(index=clin.index)

    # Age — continuous, standardize later with the rest of X
    age_col = CLINICAL_COLS["age"]
    if age_col in clin.columns:
        age = pd.to_numeric(clin[age_col], errors="coerce")
        out["age"] = age.fillna(age.median())
    else:
        print(f"  WARNING: '{age_col}' not found in {cohort}, filling with 0")
        out["age"] = 0.0

    # Gender — binary (female=1, male=0)
    gender_col = CLINICAL_COLS["gender"]
    if gender_col in clin.columns:
        g = clin[gender_col].astype(str).str.lower().str.strip()
        out["gender"] = (g == "female").astype(float)
        out["gender"] = out["gender"].where(g.isin(["female", "male"]),
                                             other=out["gender"].median())
    else:
        print(f"  WARNING: '{gender_col}' not found in {cohort}, filling with 0")
        out["gender"] = 0.0

    # Pathologic stage — ordinal: I=1, II=2, III=3, IV=4
    stage_col = CLINICAL_COLS["stage"]
    if stage_col in clin.columns:
        def parse_stage(s):
            s = str(s).upper()
            if "IV" in s:  return 4.0
            if "III" in s: return 3.0
            if "II" in s:  return 2.0
            if "I" in s:   return 1.0
            return np.nan
        stage = clin[stage_col].apply(parse_stage)
        out["stage"] = stage.fillna(stage.median())
    else:
        print(f"  WARNING: '{stage_col}' not found in {cohort}, filling with 2")
        out["stage"] = 2.0

    return out.astype(np.float32)


def preprocess(top_var, out_npz, out_csv):
    """
    Build X = [gene expression | age | gender | stage] and S per cohort.
    The 3 clinical variables sit at the end of X and serve as shared confounders
    across all cancer types in the semi-synthetic DGP.

    top_var : int or None — keep top-N genes by cross-cohort variance;
              None means use all common genes.
    out_npz : filename for the compressed numpy archive (inside OUT_DIR).
    out_csv : filename for the gene stats CSV (inside OUT_DIR).
    """
    cohorts = list(EXPRESSION_DATASETS.keys())
    expr_dfs, clin_dfs = {}, {}

    for cohort in cohorts:
        expr_dfs[cohort] = pd.read_parquet(OUT_DIR / f"expression_{cohort}.parquet")
        clin_dfs[cohort] = pd.read_parquet(OUT_DIR / f"clinical_{cohort}.parquet")

    # ---- Common genes ----
    common_genes = set(expr_dfs[cohorts[0]].columns)
    for cohort in cohorts[1:]:
        common_genes &= set(expr_dfs[cohort].columns)
    common_genes = sorted(common_genes)
    print(f"\nCommon genes: {len(common_genes)}")

    all_expr = pd.concat([expr_dfs[c][common_genes] for c in cohorts], axis=0)
    gene_means = all_expr.mean(axis=0)
    gene_vars  = all_expr.var(axis=0)

    # Select genes; force-include required pathway genes
    if top_var is None:
        top_genes = set(common_genes)
    else:
        top_genes = set(gene_vars.nlargest(top_var).index.tolist())
    not_in_data      = [g for g in REQUIRED_GENES if g not in set(common_genes)]
    if not_in_data:
        print(f"  WARNING: required genes missing from common gene set: {not_in_data}")
    missing_required = [g for g in REQUIRED_GENES
                        if g in set(common_genes) and g not in top_genes]
    if missing_required and top_var is not None:
        print(f"  Force-including {len(missing_required)} required genes "
              f"outside top-{top_var}: {missing_required}")
    selected_genes = sorted(top_genes | (set(REQUIRED_GENES) - set(not_in_data)))
    if top_var is None:
        print(f"Selected {len(selected_genes)} genes (all common genes)")
    else:
        print(f"Selected {len(selected_genes)} genes "
              f"({top_var} top-var + {len(missing_required)} force-included)")

    # ---- Build per-cohort arrays ----
    X_list, S_list = [], []

    for s_idx, cohort in enumerate(cohorts):
        expr = expr_dfs[cohort][selected_genes]
        clin = clin_dfs[cohort]

        # Align on shared sample IDs
        shared = expr.index.intersection(clin.index)
        expr = expr.loc[shared]
        clin = clin.loc[shared]

        clinical_df = encode_clinical(clin, cohort)

        # Concatenate: gene expression + clinical variables
        X_cohort = np.concatenate(
            [expr.values.astype(np.float32),
             clinical_df.values.astype(np.float32)],
            axis=1
        )
        print(f"  {cohort}: {X_cohort.shape[0]} samples  "
              f"(age mean={clinical_df['age'].mean():.1f}, "
              f"female={clinical_df['gender'].mean():.2f}, "
              f"stage mean={clinical_df['stage'].mean():.2f})")

        X_list.append(X_cohort)
        S_list.append(np.full(X_cohort.shape[0], s_idx, dtype=np.int32))

    X = np.concatenate(X_list, axis=0)
    S = np.concatenate(S_list, axis=0)

    # Standardize all columns (genes + clinical) to zero mean, unit variance
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd

    n_genes = len(selected_genes)
    # Feature names: gene names + clinical variable names
    feature_names = np.array(selected_genes + list(CLINICAL_COLS.keys()))
    # Indices of the clinical variables (last 3 columns, after all gene columns)
    clinical_indices = {name: n_genes + i
                        for i, name in enumerate(CLINICAL_COLS.keys())}

    print(f"\nFinal: {X.shape[0]} samples × {X.shape[1]} features "
          f"({n_genes} genes + {len(CLINICAL_COLS)} clinical)")
    print(f"Context counts: {dict(zip(cohorts, np.bincount(S)))}")
    print(f"Clinical variable indices: {clinical_indices}")

    out_path = OUT_DIR / out_npz
    np.savez_compressed(
        out_path,
        X=X, S=S,
        feature_names=feature_names,
        cohort_names=np.array(cohorts),
        clinical_indices=np.array(list(clinical_indices.values())),
        gene_mu=mu.squeeze(),
        gene_sd=sd.squeeze(),
    )
    print(f"Saved {out_path}")

    gene_df = pd.DataFrame({
        "gene":      selected_genes,
        "mean_expr": gene_means[selected_genes].values,
        "variance":  gene_vars[selected_genes].values,
    })
    gene_df.to_csv(OUT_DIR / out_csv, index=False)
    print(f"Saved {out_csv}")


# ============================================================
# Main
# ============================================================

def main():
    print("Step 1: Downloading data from UCSC Xena...")
    download_all()

    print("\nStep 2: Parsing expression matrices...")
    parse_expression()

    print("\nStep 3: Parsing clinical matrices...")
    parse_clinical()

    print("\nStep 4: Preprocessing...")
    preprocess(**GENE_SELECTIONS["all"])

    print("\nDone. Output: tcga_data/tcga_X_all.npz")


if __name__ == "__main__":
    main()
