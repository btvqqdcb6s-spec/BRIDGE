"""
Semi-synthetic TCGA experiment — context-aware CFRNet + baselines (multi-seed average)
========================================================================================
DGP:
  • Stronger confounding: larger propensity coefficients + age×stage interaction
  • More clinical signals: gender added to propensity and baseline
  • Shared baseline: age×stage×gender component identical across all cancer contexts
  • Nonlinear interactions: C×C and C×clinical terms in baseline

Methods:
  CFRNet    : context-aware via one-hot S appended to X (p_feat = original_p + 3)
  Baselines : t_learner, dr_learner, causal_forest (also context-aware via X_aug)

Reports mean ± std across n_seeds seeds.
"""

import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from econml.dml import CausalForestDML
from pathlib import Path

# ============================================================
# 0. Configuration
# ============================================================

DATA_PATH = Path("tcga_data/tcga_X_all.npz")

CONFIG = {
    "seed": 42,
    "n_seeds": 10,
    "base_seed": 42,
    # CFRNet
    "epochs": 150,
    "batch_size": 256,
    "lr": 1e-3,
    "repr_dim": 20,
    "hidden_dims": [256, 128],
    "head_hidden_dim": 32,
    "ipm_weight": 1.0,
    "prop_clip": 0.05,
    # DGP
    "noise_std": 1.0,
    "test_frac": 0.2,
    # Tree-based
    "n_estimators": 1000,
    "min_samples_leaf": 5,
    "lasso_cv": 3,
    "lasso_max_iter": 5000,
    "logistic_C": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CONTEXT_NAMES = {0: "BRCA", 1: "LUAD", 2: "KIRC"}
N_CONTEXTS = 3

# ============================================================
# 1. Load data and resolve gene indices
# ============================================================

def load_tcga():
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    S = data["S"]
    feature_names = list(data["feature_names"])
    clinical_idx  = data["clinical_indices"]   # [age_idx, gender_idx, stage_idx]
    return X, S, feature_names, clinical_idx


def resolve_gene(name, feature_names):
    try:
        return feature_names.index(name)
    except ValueError:
        print(f"  WARNING: gene '{name}' not in selected features")
        return None


def resolve_all_genes(feature_names, clinical_idx):
    age_idx    = int(clinical_idx[0])
    gender_idx = int(clinical_idx[1])
    stage_idx  = int(clinical_idx[2])

    genes = {
        "ESR1":  resolve_gene("ESR1",  feature_names),
        "PIK3CA":resolve_gene("PIK3CA",feature_names),
        "ERBB2": resolve_gene("ERBB2", feature_names),
        "BRCA1": resolve_gene("BRCA1", feature_names),
        "EGFR":  resolve_gene("EGFR",  feature_names),
        "KRAS":  resolve_gene("KRAS",  feature_names),
        "STK11": resolve_gene("STK11", feature_names),
        "KEAP1": resolve_gene("KEAP1", feature_names),
        "VHL":   resolve_gene("VHL",   feature_names),
        "PBRM1": resolve_gene("PBRM1", feature_names),
        "BAP1":  resolve_gene("BAP1",  feature_names),
        "SETD2": resolve_gene("SETD2", feature_names),
        "MKI67": resolve_gene("MKI67", feature_names),
        "TP53":  resolve_gene("TP53",  feature_names),
        "CDH1":  resolve_gene("CDH1",  feature_names),
        "ALK":   resolve_gene("ALK",   feature_names),
        "ROS1":  resolve_gene("ROS1",  feature_names),
        "MET":   resolve_gene("MET",   feature_names),
        "MTOR":  resolve_gene("MTOR",  feature_names),
        "VEGFA": resolve_gene("VEGFA", feature_names),
        "KDM5C": resolve_gene("KDM5C", feature_names),
    }
    missing = [g for g, idx in genes.items() if idx is None]
    if missing:
        raise RuntimeError(f"Required genes missing: {missing}")

    g = genes
    context_features = {
        0: {"C":  [age_idx, stage_idx, gender_idx, g["ESR1"], g["PIK3CA"]],
            "EM": [g["ERBB2"], g["BRCA1"]],
            "I":  [g["MKI67"], g["TP53"], g["CDH1"]],
            "C_names":  ["age", "stage", "gender", "ESR1", "PIK3CA"],
            "EM_names": ["ERBB2", "BRCA1"],
            "I_names":  ["MKI67", "TP53", "CDH1"]},
        1: {"C":  [age_idx, stage_idx, gender_idx, g["EGFR"], g["KRAS"]],
            "EM": [g["STK11"], g["KEAP1"]],
            "I":  [g["ALK"], g["ROS1"], g["MET"]],
            "C_names":  ["age", "stage", "gender", "EGFR", "KRAS"],
            "EM_names": ["STK11", "KEAP1"],
            "I_names":  ["ALK", "ROS1", "MET"]},
        2: {"C":  [age_idx, stage_idx, gender_idx, g["VHL"], g["PBRM1"]],
            "EM": [g["BAP1"], g["SETD2"]],
            "I":  [g["MTOR"], g["VEGFA"], g["KDM5C"]],
            "C_names":  ["age", "stage", "gender", "VHL", "PBRM1"],
            "EM_names": ["BAP1", "SETD2"],
            "I_names":  ["MTOR", "VEGFA", "KDM5C"]},
    }
    return g, context_features, age_idx, stage_idx, gender_idx


def get_relevant(s, context_features):
    return context_features[s]["C"] + context_features[s]["EM"]


# ============================================================
# 2. Synthetic DGP on real X
# ============================================================

def generate_outcomes(X, S, g, age_idx, stage_idx, gender_idx, rng, config):
    """
    Enhanced DGP:
    - Propensity : stronger confounders + instruments + age×stage interaction + gender
    - Baseline   : shared clinical component + C + C×C/C×clinical interactions
                   + EMs contribute (breaking C/EM separation)
    - CATE       : main EMs + confounders (partial) + instruments (weak)
    """
    n  = X.shape[0]
    m0 = (S == 0)
    m1 = (S == 1)
    m2 = (S == 2)

    # --- Propensity: strong confounding + instruments + clinical interactions ---
    logit_e = np.zeros(n)


    # All coefficients scaled ×0.7 vs previous; two instruments removed per context
    logit_e[m0] = (
        0.525 * X[m0, age_idx] + 0.42 * X[m0, stage_idx] + 0.28 * X[m0, gender_idx]
        + 0.7   * X[m0, g["ESR1"]]  + 0.525 * X[m0, g["PIK3CA"]]
        + 0.525 * X[m0, g["MKI67"]]                                # TP53, CDH1 removed
    )
    logit_e[m1] = (
        0.525 * X[m1, age_idx] + 0.42 * X[m1, stage_idx] + 0.28 * X[m1, gender_idx]
        + 0.7   * X[m1, g["EGFR"]]  + 0.525 * X[m1, g["KRAS"]]
        + 0.525 * X[m1, g["ALK"]]                                  # ROS1, MET removed
    )
    logit_e[m2] = (
        0.4   * X[m2, age_idx] + 0.42 * X[m2, stage_idx] + 0.28 * X[m2, gender_idx]
        + 0.525 * X[m2, g["VHL"]]   + 0.42  * X[m2, g["PBRM1"]]
        + 0.42  * X[m2, g["MTOR"]]                                 # VEGFA, KDM5C removed
    )                                                               # age uses 0.4 (not 0.525)

    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, config["prop_clip"], 1.0 - config["prop_clip"])
    T = rng.binomial(1, e)


    # --- Shared baseline component (identical across contexts) ---
    shared = (
        0.6 * X[:, age_idx]
        + 0.5 * X[:, stage_idx]
        + 0.3 * X[:, gender_idx]
        + 0.35 * X[:, age_idx] * X[:, stage_idx]
    )

    # --- Baseline outcome mu0: shared + confounders + C×C/C×clinical ---
    base = np.zeros(n)

    # BRCA: sin nonlinearity on PIK3CA, C×C and C×clinical interactions
    base[m0] = (
        shared[m0]
        + 0.6 * X[m0, g["ESR1"]]
        + 0.45 * np.sin(np.pi * X[m0, g["PIK3CA"]])
        + 0.15 * X[m0, g["ESR1"]] * X[m0, age_idx]            # C × clinical
        + 0.1  * X[m0, g["ESR1"]] * X[m0, g["PIK3CA"]]        # C × C
    )
    # LUAD: exp decay on KRAS, C×C and C×clinical
    base[m1] = (
        shared[m1]
        + 0.6 * X[m1, g["EGFR"]]
        + 0.3 * np.exp(-np.abs(X[m1, g["KRAS"]]))
        + 0.15 * X[m1, g["EGFR"]] * X[m1, age_idx]
        + 0.4  * X[m1, g["EGFR"]] * X[m1, g["KRAS"]]
    )
    # KIRC: ReLU on PBRM1, C×C and C×clinical
    base[m2] = (
        shared[m2]
        + 0.6  * X[m2, g["VHL"]]
        + 0.45 * np.maximum(X[m2, g["PBRM1"]], 0)
        + 0.35 * X[m2, g["VHL"]] * X[m2, age_idx]
        + 0.4  * X[m2, g["VHL"]] * X[m2, g["PBRM1"]]
    )

# --- CATE: EMs (primary) + confounders modulating EM effects ---
    # Confounders enter tau through interactions with EMs and clinical vars,
    # reflecting that disease severity (stage), patient profile (age, gender)
    # and oncogenic drivers (C) modulate treatment response, not just baseline risk.
    tau = np.zeros(n)

    tau[m0] = (
    0.7 * np.tanh(X[m0, g["ERBB2"]])
    + 0.6 * X[m0, g["BRCA1"]]
    + 0.3 * np.tanh(X[m0, g["ESR1"]])
    )

    tau[m1] = (
    0.5 * X[m1, g["STK11"]]
    + 0.4 * np.tanh(2.0 * X[m1, g["KEAP1"]])
    + 0.25 * np.tanh(X[m1, g["KRAS"]])
    )

    tau[m2] = (
    -0.5 * np.tanh(X[m2, g["BAP1"]])
    - 0.6 * np.tanh(X[m2,g["SETD2"]])           #np.sin(np.pi * X[m2, g["SETD2"]] / 2)
    - 0.5 * np.tanh(X[m2, g["VHL"]])
    )

    eps = rng.standard_normal(n) * config["noise_std"]
    Y0  = base + eps
    Y1  = base + tau + eps
    Y   = T * Y1 + (1 - T) * Y0

    return T, Y, tau, base


# ============================================================
# 3. CFRNet Model Components
# ============================================================

class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, repr_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ELU()])
            prev_dim = h
        layers.extend([nn.Linear(prev_dim, repr_dim), nn.ELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class HypothesisHead(nn.Module):
    def __init__(self, repr_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, phi):
        return self.net(phi).squeeze(-1)


class CFRNet(nn.Module):
    def __init__(self, p, hidden_dims, repr_dim, head_hidden_dim):
        super().__init__()
        self.phi = RepresentationNetwork(p, hidden_dims, repr_dim)
        self.h0  = HypothesisHead(repr_dim, head_hidden_dim)
        self.h1  = HypothesisHead(repr_dim, head_hidden_dim)

    def forward(self, X, T):
        phi    = self.phi(X)
        y0     = self.h0(phi)
        y1     = self.h1(phi)
        y_pred = T * y1 + (1 - T) * y0
        return y_pred, y0, y1, phi


# ============================================================
# 5. MMD
# ============================================================

def mmd_rbf(phi_t, phi_c, sigma=1.0):
    if phi_t.shape[0] == 0 or phi_c.shape[0] == 0:
        return torch.tensor(0.0, device=phi_t.device)
    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b, p=2).pow(2) / (2.0 * sigma ** 2))
    return (kernel(phi_t, phi_t).mean() + kernel(phi_c, phi_c).mean()
            - 2.0 * kernel(phi_t, phi_c).mean())


# ============================================================
# 4. CFRNet Training
# ============================================================

def train_cfrnet(model, X_train, T_train, Y_train, config):
    device = config["device"]
    model  = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_t, T_t, Y_t),
                        batch_size=config["batch_size"], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    model.train()
    for _ in range(config["epochs"]):
        for X_b, T_b, Y_b in loader:
            optimizer.zero_grad()
            y_pred, y0, y1, phi = model(X_b, T_b)
            mse   = nn.functional.mse_loss(y_pred, Y_b)
            idx_t = (T_b == 1).nonzero(as_tuple=True)[0]
            idx_c = (T_b == 0).nonzero(as_tuple=True)[0]
            ipm   = mmd_rbf(phi[idx_t], phi[idx_c])
            loss  = mse + config["ipm_weight"] * ipm
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


# ============================================================
# 5. Evaluation helpers
# ============================================================

def compute_metrics(tau_pred, tau_test, S_test):
    pehe = np.sqrt(np.mean((tau_pred - tau_test) ** 2))
    ctx = {}
    for s in range(N_CONTEXTS):
        mask = (S_test == s)
        if mask.sum() > 0:
            ctx[s] = {
                "pehe": np.sqrt(np.mean((tau_pred[mask] - tau_test[mask]) ** 2)),
            }
    return pehe, ctx


def eval_cfrnet(model, X_test, tau_test, S_test, config):
    device = config["device"]
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    T1  = torch.ones(X_t.shape[0], device=device)
    with torch.no_grad():
        _, y0_t, y1_t, _ = model(X_t, T1)
        tau_pred = (y1_t - y0_t).cpu().numpy()
    return compute_metrics(tau_pred, tau_test, S_test)


# ============================================================
# Baselines (context-aware via X_aug)
# ============================================================

def run_t_learner(X_train, T_train, Y_train, X_test, S_test, tau_test, config):
    mu0 = LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                  random_state=config["seed"], n_jobs=-1)
    mu1 = LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                  random_state=config["seed"], n_jobs=-1)
    mu0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    mu1.fit(X_train[T_train == 1], Y_train[T_train == 1])
    tau_pred = mu1.predict(X_test) - mu0.predict(X_test)
    pehe, ctx = compute_metrics(tau_pred, tau_test, S_test)
    print(f"  Overall PEHE: {pehe:.4f}")
    for s in range(N_CONTEXTS):
        print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {ctx[s]['pehe']:.4f}")
    print()
    return {"pehe": pehe, "ctx_metrics": ctx}


def run_dr_learner(X_train, T_train, Y_train, S_train, X_test, S_test, tau_test, config):
    skf  = StratifiedKFold(n_splits=2, shuffle=True, random_state=config["seed"])
    psi  = np.zeros(len(Y_train), dtype=float)

    for idx_a, idx_b in skf.split(X_train, S_train):
        X_a, X_b = X_train[idx_a], X_train[idx_b]
        T_a       = T_train[idx_a]
        Y_a       = Y_train[idx_a]
        T_b       = T_train[idx_b]
        Y_b       = Y_train[idx_b]

        mu0_a = LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                        random_state=config["seed"], n_jobs=-1)
        mu1_a = LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                        random_state=config["seed"], n_jobs=-1)
        e_a   = LogisticRegressionCV(Cs=10, cv=config["lasso_cv"],
                                     max_iter=config["lasso_max_iter"],
                                     penalty="l2", scoring="neg_log_loss",
                                     random_state=config["seed"], n_jobs=-1)
        mu0_a.fit(X_a[T_a == 0], Y_a[T_a == 0])
        mu1_a.fit(X_a[T_a == 1], Y_a[T_a == 1])
        e_a.fit(X_a, T_a)

        mu0_hat = mu0_a.predict(X_b)
        mu1_hat = mu1_a.predict(X_b)
        e_hat   = e_a.predict_proba(X_b)[:, 1]
        e_hat   = np.clip(e_hat, config["prop_clip"], 1.0 - config["prop_clip"])

        psi[idx_b] = (
            mu1_hat - mu0_hat
            + T_b * (Y_b - mu1_hat) / e_hat
            - (1 - T_b) * (Y_b - mu0_hat) / (1.0 - e_hat)
        )

    tau_model = LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                        random_state=config["seed"], n_jobs=-1)
    tau_model.fit(X_train, psi)
    tau_pred = tau_model.predict(X_test)
    pehe, ctx = compute_metrics(tau_pred, tau_test, S_test)
    print(f"  Overall PEHE: {pehe:.4f}")
    for s in range(N_CONTEXTS):
        print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {ctx[s]['pehe']:.4f}")
    print()
    return {"pehe": pehe, "ctx_metrics": ctx}


def run_causal_forest(X_train, T_train, Y_train, S_train, X_test, S_test, tau_test, config):
    cv_folds = list(StratifiedKFold(n_splits=2, shuffle=True,
                                    random_state=config["seed"]).split(X_train, S_train))
    cf = CausalForestDML(
        model_y=LassoCV(cv=config["lasso_cv"], max_iter=config["lasso_max_iter"],
                        random_state=config["seed"], n_jobs=-1),
        model_t=LogisticRegressionCV(Cs=10, cv=config["lasso_cv"],
                                     max_iter=config["lasso_max_iter"],
                                     penalty="l2", scoring="neg_log_loss",
                                     random_state=config["seed"], n_jobs=-1),
        discrete_treatment=True,
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features="sqrt",
        cv=cv_folds,
        random_state=config["seed"],
        n_jobs=-1,
    )
    cf.fit(Y_train, T_train, X=X_train)
    tau_pred = cf.effect(X_test).ravel()
    pehe, ctx = compute_metrics(tau_pred, tau_test, S_test)
    print(f"  Overall PEHE: {pehe:.4f}")
    for s in range(N_CONTEXTS):
        print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {ctx[s]['pehe']:.4f}")
    print()
    return {"pehe": pehe, "ctx_metrics": ctx}


# ============================================================
# 9. Aggregated reporting
# ============================================================

ALL_CONDITIONS = ["cfrnet", "t_learner", "dr_learner", "causal_forest"]


def print_table_avg(all_results):
    """Print mean±std PEHE table across seeds."""
    def agg(lst, key):
        vals = [d[key] for d in lst]
        return np.mean(vals), np.std(vals)

    def agg_ctx(lst, s, key):
        vals = [d["ctx_metrics"][s][key] for d in lst]
        return np.mean(vals), np.std(vals)

    cell  = 15
    ncols = 1 + N_CONTEXTS
    sep   = "=" * (22 + ncols * (cell + 1))
    print("\n" + sep)
    print(f"{'Condition':<20s}  {'Overall PEHE':^{cell}s} " +
          "".join(f"| {CONTEXT_NAMES[s]+' PEHE':^{cell}s} "
                  for s in range(N_CONTEXTS)))
    print("-" * (22 + ncols * (cell + 1)))

    for cond in ALL_CONDITIONS:
        lst = all_results[cond]
        m_p, s_p = agg(lst, "pehe")
        line = f"{cond:<20s}  {m_p:.4f}±{s_p:.4f} "
        for s in range(N_CONTEXTS):
            m_cp, s_cp = agg_ctx(lst, s, "pehe")
            line += f"  {m_cp:.4f}±{s_cp:.4f} "
        print(line)
    print(sep)


# ============================================================
# 10. Main
# ============================================================

def main():
    t_start = time.time()

    print(f"Loading TCGA data from {DATA_PATH}...")
    X, S, feature_names, clinical_idx = load_tcga()
    g, _, age_idx, stage_idx, gender_idx = resolve_all_genes(
        feature_names, clinical_idx)

    S_onehot = np.eye(N_CONTEXTS, dtype=X.dtype)[S]
    X_aug    = np.hstack([X, S_onehot])
    p_feat   = X_aug.shape[1]

    print(f"Device: {CONFIG['device']}")
    print(f"Full dataset: {X.shape[0]} samples × {X.shape[1]} features "
          f"(+{N_CONTEXTS} one-hot → p_feat={p_feat})")
    print(f"Architecture: [...] -> {CONFIG['hidden_dims']} -> {CONFIG['repr_dim']}")
    print(f"Noise std: {CONFIG['noise_std']}   Epochs: {CONFIG['epochs']}")
    print(f"Running {CONFIG['n_seeds']} seeds (base_seed={CONFIG['base_seed']})\n")

    all_results = {cond: [] for cond in ALL_CONDITIONS}

    for seed_idx in range(CONFIG["n_seeds"]):
        seed = CONFIG["base_seed"] + seed_idx
        cfg  = {**CONFIG, "seed": seed}
        t_seed = time.time()

        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{CONFIG['n_seeds']}  (seed={seed})")
        print(f"{'='*60}")

        rng_split = np.random.default_rng(seed)
        train_idx, test_idx = [], []
        for s in range(N_CONTEXTS):
            s_idx = np.where(S == s)[0]
            s_idx = s_idx[rng_split.permutation(len(s_idx))]
            split = int(len(s_idx) * (1 - cfg["test_frac"]))
            train_idx.extend(s_idx[:split])
            test_idx.extend(s_idx[split:])
        train_idx = np.array(train_idx)
        test_idx  = np.array(test_idx)

        rng = np.random.default_rng(seed)
        T, Y, tau, base = generate_outcomes(
            X, S, g, age_idx, stage_idx, gender_idx, rng, cfg)

        if seed_idx == 0:
            sigma2 = cfg["noise_std"] ** 2
            print("DGP diagnostics (seed 0):")
            for s in range(N_CONTEXTS):
                mask   = (S == s)
                v_tau  = np.var(tau[mask])
                v_base = np.var(base[mask])
                frac_s = v_tau / (v_tau + v_base) if (v_tau + v_base) > 0 else 0.0
                frac_t = v_tau / (v_tau + v_base + sigma2)
                print(f"  {CONTEXT_NAMES[s]} (n={mask.sum()}):")
                print(f"    Var[tau]: {v_tau:.3f}  Var[base]: {v_base:.3f}  "
                      f"Var[eps]: {sigma2:.3f}")
                print(f"    tau/(tau+base): {frac_s:.3f}   "
                      f"tau/(tau+base+eps): {frac_t:.3f}")
                print(f"    Mean[tau]: {np.mean(tau[mask]):.3f}   "
                      f"Treat frac: {T[mask].mean():.3f}")
            print()

        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        T_train  = T[train_idx]
        Y_train  = Y[train_idx]
        tau_test = tau[test_idx]
        S_train  = S[train_idx]
        S_test   = S[test_idx]

        # --- CFRNet (context-aware via one-hot augmentation) ---
        print(f"--- cfrnet (seed {seed_idx+1}/{CONFIG['n_seeds']}) ---")
        torch.manual_seed(seed)
        model = CFRNet(
            p=p_feat,
            hidden_dims=cfg["hidden_dims"],
            repr_dim=cfg["repr_dim"],
            head_hidden_dim=cfg["head_hidden_dim"],
        )
        model = train_cfrnet(model, X_train, T_train, Y_train, cfg)
        pehe, ctx = eval_cfrnet(model, X_test, tau_test, S_test, cfg)
        all_results["cfrnet"].append({"pehe": pehe, "ctx_metrics": ctx})
        print(f"  Overall PEHE: {pehe:.4f}")
        for s in range(N_CONTEXTS):
            print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {ctx[s]['pehe']:.4f}")
        print()

        # --- Context-aware baselines (X_aug passed as X_train / X_test) ---
        print(f"--- t_learner (seed {seed_idx+1}/{CONFIG['n_seeds']}) ---")
        res = run_t_learner(X_train, T_train, Y_train, X_test, S_test, tau_test, cfg)
        all_results["t_learner"].append(
            {"pehe": res["pehe"], "ctx_metrics": res["ctx_metrics"]})

        print(f"--- dr_learner (seed {seed_idx+1}/{CONFIG['n_seeds']}) ---")
        res = run_dr_learner(X_train, T_train, Y_train, S_train,
                             X_test, S_test, tau_test, cfg)
        all_results["dr_learner"].append(
            {"pehe": res["pehe"], "ctx_metrics": res["ctx_metrics"]})

        print(f"--- causal_forest (seed {seed_idx+1}/{CONFIG['n_seeds']}) ---")
        res = run_causal_forest(X_train, T_train, Y_train, S_train,
                                X_test, S_test, tau_test, cfg)
        all_results["causal_forest"].append(
            {"pehe": res["pehe"], "ctx_metrics": res["ctx_metrics"]})

        elapsed      = time.time() - t_seed
        total_so_far = time.time() - t_start
        print(f"\nSeed {seed_idx+1}/{CONFIG['n_seeds']} done in {elapsed/60:.1f} min  "
              f"(total so far: {total_so_far/60:.1f} min)")

    # --- Save results ---
    plot_dir = Path("bridge_plots")
    plot_dir.mkdir(exist_ok=True)
    pkl_path = plot_dir / "seed_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {pkl_path}")

    print_table_avg(all_results)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
