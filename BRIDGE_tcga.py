"""
Semi-synthetic TCGA experiment — BRIDGE weakest-feature misspecification (multi-seed average)
==============================================================================================
DGP:
  • Stronger confounding: larger propensity coefficients + age×stage interaction
  • More clinical signals: gender added to propensity and baseline
  • Shared baseline: age×stage×gender component identical across all cancer contexts
  • Nonlinear interactions: C×C and C×clinical terms in baseline

Methods:
  BRIDGE/CFRNet : informative, missing_weakest_1, missing_weakest_2

Workflow:
  1. python BRIDGE_tests_weakest_missing.py --rank-only
       Prints per-context leave-one-out Δvar rankings; no training.
  2. Fill in WEAKEST_GENES in the source file based on the output.
  3. python BRIDGE_tests_weakest_missing.py
       Runs the full multi-seed experiment.

Reports mean ± std across n_seeds seeds.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ============================================================
# 0. Configuration
# ============================================================

DATA_PATH = Path("tcga_data/tcga_X_all.npz")

CONFIG = {
    "seed": 42,
    "n_seeds": 10,
    "base_seed": 42,
    # BRIDGE
    "epochs": 150,
    "batch_size": 256,
    "lr": 1e-3,
    "lr_theta_multiplier": 35,
    "repr_dim": 20,
    "hidden_dims": [256, 128],
    "head_hidden_dim": 32,
    "ipm_weight": 1.0,
    "prop_clip": 0.05,
    # DGP
    "noise_std": 1.0,
    "test_frac": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CONTEXT_NAMES = {0: "BRCA", 1: "LUAD", 2: "KIRC"}
N_CONTEXTS = 3

SPURIOUS_IDX = 100   # non-relevant feature endorsed by the "noisy" prior condition

# ----------------------------------------------------------------
# Run `python BRIDGE_tests_weakest_missing.py --rank-only` once to
# obtain the per-context Δvar rankings, then fill in the two
# weakest genes per context (weakest first) and re-run normally.
# ----------------------------------------------------------------
WEAKEST_GENES = {
    0: ("PIK3CA", "ERBB2"),  # BRCA: Δvar 0.201, 0.357
    1: ("KEAP1",  "KRAS"),   # LUAD: Δvar 0.076, 0.275
    2: ("BAP1",   "VHL"),    # KIRC: Δvar 0.272, 0.286
}

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
    - 0.6 * np.tanh(X[m2,g["SETD2"]])           
    - 0.5 * np.tanh(X[m2, g["VHL"]])
    )

    eps = rng.standard_normal(n) * config["noise_std"]
    Y0  = base + eps
    Y1  = base + tau + eps
    Y   = T * Y1 + (1 - T) * Y0

    return T, Y, tau, base



# ============================================================
# 2b. Variance ranking utility
# ============================================================

def rank_features_by_variance(X, S, g, context_features,
                               age_idx, stage_idx, gender_idx, config):
    """
    Per-context leave-one-out Δvar ranking of candidate genes (C ∪ EM,
    excluding clinical variables age/stage/gender).

    Δvar_j = Var[base + tau] - Var[(base + tau) with X[:,j] = 0]  per context.
    Noise variance is identical in both terms and cancels, so we compare
    the noiseless outcomes.  Printed ascending (weakest contributor first).
    """
    rng_full = np.random.default_rng(config["base_seed"])
    _, _, tau_full, base_full = generate_outcomes(
        X, S, g, age_idx, stage_idx, gender_idx, rng_full, config)
    Y_full = base_full + tau_full

    clinical_set = {age_idx, stage_idx, gender_idx}

    print("\nVariance ranking (leave-one-out Δvar, ascending):")
    for s in range(N_CONTEXTS):
        mask       = (S == s)
        var_full_s = np.var(Y_full[mask])

        c_genes  = [(idx, name)
                    for idx, name in zip(context_features[s]["C"],
                                         context_features[s]["C_names"])
                    if idx not in clinical_set]
        em_genes = list(zip(context_features[s]["EM"],
                            context_features[s]["EM_names"]))
        candidates = c_genes + em_genes

        deltas = {}
        for idx, name in candidates:
            X_drop         = X.copy()
            X_drop[:, idx] = 0.0          # standardised mean = 0
            rng_drop       = np.random.default_rng(config["base_seed"])
            _, _, tau_drop, base_drop = generate_outcomes(
                X_drop, S, g, age_idx, stage_idx, gender_idx, rng_drop, config)
            Y_drop         = base_drop + tau_drop
            deltas[name]   = var_full_s - np.var(Y_drop[mask])

        ranked = sorted(deltas.items(), key=lambda kv: kv[1])
        print(f"Variance ranking — {CONTEXT_NAMES[s]}:")
        for name, dv in ranked:
            print(f"  {name:<8s}  Δvar = {dv:.3f}")
    print()


# ============================================================
# 3. Expert Prior Specifications (context-specific, BRIDGE)
# ============================================================

def get_prior_config(condition, p, n_contexts, context_features, g):
    """Return list of (alpha, prior_weight) per context.

      informative       — endorse all C + EM features (sanity check)
      noisy             — informative + one spurious feature at alpha=2.0
      missing_weakest_1 — endorse all relevant except WEAKEST_GENES[s][0]
      missing_weakest_2 — endorse all relevant except both WEAKEST_GENES[s]
    """
    configs = []
    for s in range(n_contexts):
        relevant = get_relevant(s, context_features)

        if condition == "informative":
            alpha = np.full(p, 1.0)
            alpha[relevant] = 5.0
            configs.append((alpha, 0.01))

        elif condition == "noisy":
            alpha = np.full(p, 1.0)
            alpha[relevant] = 5.0
            alpha[SPURIOUS_IDX] = 2.0
            configs.append((alpha, 0.01))

        elif condition == "missing_weakest_1":
            drop     = {g[WEAKEST_GENES[s][0]]}
            endorsed = [i for i in relevant if i not in drop]
            alpha    = np.full(p, 1.0)
            alpha[endorsed] = 5.0
            configs.append((alpha, 0.01))

        elif condition == "missing_weakest_2":
            drop     = {g[name] for name in WEAKEST_GENES[s]}
            endorsed = [i for i in relevant if i not in drop]
            alpha    = np.full(p, 1.0)
            alpha[endorsed] = 5.0
            configs.append((alpha, 0.01))

        else:
            raise ValueError(f"Unknown condition: {condition}")

    return configs


# ============================================================
# 4. BRIDGE Model Components
# ============================================================

class ImportanceWeightModule(nn.Module):
    def __init__(self, p, alpha=None):
        super().__init__()
        if alpha is not None:
            init = torch.log(torch.tensor(alpha, dtype=torch.float32))
        else:
            init = torch.zeros(p)
        self.theta = nn.Parameter(init)
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def get_weights(self):
        return torch.softmax(self.theta, dim=0)

    def log_prior(self):
        if self.alpha is None:
            return torch.tensor(0.0)
        w = self.get_weights()
        return torch.sum((self.alpha - 1.0) * torch.log(w + 1e-12))


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


class MultiContextCFRNet(nn.Module):
    def __init__(self, p, hidden_dims, repr_dim, head_hidden_dim,
                 n_contexts, alphas=None):
        super().__init__()
        self.n_contexts  = n_contexts
        self.use_weights = alphas is not None
        if self.use_weights:
            self.importance = nn.ModuleList([
                ImportanceWeightModule(p, alphas[s]) for s in range(n_contexts)
            ])
        self.phi = RepresentationNetwork(p, hidden_dims, repr_dim)
        self.h0  = HypothesisHead(repr_dim, head_hidden_dim)
        self.h1  = HypothesisHead(repr_dim, head_hidden_dim)

    def forward(self, X, T, S):
        if self.use_weights:
            all_w        = torch.stack([self.importance[s].get_weights()
                                        for s in range(self.n_contexts)])
            w_per_sample = all_w[S.long()]
            X_w          = X * w_per_sample
        else:
            X_w = X
        phi    = self.phi(X_w)
        y0     = self.h0(phi)
        y1     = self.h1(phi)
        y_pred = T * y1 + (1 - T) * y0
        return y_pred, y0, y1, phi

    def log_prior(self):
        if self.use_weights:
            return sum(self.importance[s].log_prior() for s in range(self.n_contexts))
        return torch.tensor(0.0)

    def get_context_weights(self):
        if not self.use_weights:
            return None
        with torch.no_grad():
            return {s: self.importance[s].get_weights().cpu().numpy()
                    for s in range(self.n_contexts)}


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
# 6. BRIDGE Training
# ============================================================

def train_bridge(model, X_train, T_train, Y_train, S_train, config, prior_weight):
    device = config["device"]
    model  = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    S_t = torch.tensor(S_train, dtype=torch.long,    device=device)

    loader = DataLoader(TensorDataset(X_t, T_t, Y_t, S_t),
                        batch_size=config["batch_size"], shuffle=True)

    if model.use_weights:
        imp_ids    = {id(p) for s in range(model.n_contexts)
                      for p in model.importance[s].parameters()}
        imp_params = [p for p in model.parameters() if id(p) in imp_ids]
        oth_params = [p for p in model.parameters() if id(p) not in imp_ids]
        param_groups = [
            {"params": oth_params, "lr": config["lr"]},
            {"params": imp_params, "lr": config["lr"] * config["lr_theta_multiplier"]},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": config["lr"]}]

    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    model.train()
    trajectory = ({s: [] for s in range(model.n_contexts)}
                  if model.use_weights else None)

    for _ in range(config["epochs"]):
        for X_b, T_b, Y_b, S_b in loader:
            optimizer.zero_grad()
            y_pred, y0, y1, phi = model(X_b, T_b, S_b)
            mse   = nn.functional.mse_loss(y_pred, Y_b)
            idx_t = (T_b == 1).nonzero(as_tuple=True)[0]
            idx_c = (T_b == 0).nonzero(as_tuple=True)[0]
            ipm   = mmd_rbf(phi[idx_t], phi[idx_c])
            loss  = mse + config["ipm_weight"] * ipm - prior_weight * model.log_prior()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if model.use_weights:
            w = model.get_context_weights()
            for s in range(model.n_contexts):
                trajectory[s].append(w[s].copy())

    if model.use_weights:
        trajectory = {s: np.stack(trajectory[s], axis=0)
                      for s in range(model.n_contexts)}

    return model, trajectory


# ============================================================
# 7. Evaluation helpers
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


def eval_bridge(model, X_test, tau_test, S_test, config):
    device = config["device"]
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    S_t = torch.tensor(S_test, dtype=torch.long,    device=device)
    T1  = torch.ones(X_t.shape[0], device=device)
    with torch.no_grad():
        _, y0_t, y1_t, _ = model(X_t, T1, S_t)
        tau_pred = (y1_t - y0_t).cpu().numpy()
    return compute_metrics(tau_pred, tau_test, S_test)


# ============================================================
# 9. Aggregated reporting
# ============================================================

BRIDGE_CONDITIONS = ["informative", "noisy", "missing_weakest_1", "missing_weakest_2"]
ALL_CONDITIONS    = BRIDGE_CONDITIONS


def print_table_avg(all_results):
    """Print mean±std table across seeds."""
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
    parser = argparse.ArgumentParser(
        description="BRIDGE weakest-feature misspecification experiment.")
    parser.add_argument("--rank-only", action="store_true",
                        help="Print per-context Δvar rankings and exit (no training).")
    args = parser.parse_args()

    t_start = time.time()

    print(f"Loading TCGA data from {DATA_PATH}...")
    X, S, feature_names, clinical_idx = load_tcga()
    g, context_features, age_idx, stage_idx, gender_idx = resolve_all_genes(
        feature_names, clinical_idx)

    spurious_name = feature_names[SPURIOUS_IDX]
    relevant_features = {
        "ESR1", "PIK3CA", "ERBB2", "BRCA1", "EGFR", "KRAS", "STK11",
        "KEAP1", "VHL", "PBRM1", "BAP1", "SETD2", "MKI67", "TP53",
        "CDH1", "ALK", "ROS1", "MET", "MTOR", "VEGFA", "KDM5C",
        "age", "gender", "stage",
    }
    if spurious_name in relevant_features:
        raise RuntimeError(
            f"SPURIOUS_IDX={SPURIOUS_IDX} lands on relevant feature '{spurious_name}'.")

    p_feat = X.shape[1]

    rank_features_by_variance(X, S, g, context_features,
                               age_idx, stage_idx, gender_idx, CONFIG)
    if args.rank_only:
        return

    print(f"Device: {CONFIG['device']}")
    print(f"Full dataset: {X.shape[0]} samples × {p_feat} features")
    print(f"Architecture: [...] -> {CONFIG['hidden_dims']} -> {CONFIG['repr_dim']}")
    print(f"Noise std: {CONFIG['noise_std']}   Epochs: {CONFIG['epochs']}")
    print(f"Running {CONFIG['n_seeds']} seeds (base_seed={CONFIG['base_seed']})\n")

    # Aggregators
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

        X_train, X_test = X[train_idx], X[test_idx]
        T_train  = T[train_idx]
        Y_train  = Y[train_idx]
        tau_test = tau[test_idx]
        S_train  = S[train_idx]
        S_test   = S[test_idx]

        # --- BRIDGE / CFRNet ---
        for cond in BRIDGE_CONDITIONS:
            print(f"--- {cond} (seed {seed_idx+1}/{CONFIG['n_seeds']}) ---")
            torch.manual_seed(seed)

            ctx_configs  = get_prior_config(cond, p_feat, N_CONTEXTS,
                                            context_features, g)
            alphas       = [c[0] for c in ctx_configs]
            prior_weight = ctx_configs[0][1]

            model = MultiContextCFRNet(
                p=p_feat,
                hidden_dims=cfg["hidden_dims"],
                repr_dim=cfg["repr_dim"],
                head_hidden_dim=cfg["head_hidden_dim"],
                n_contexts=N_CONTEXTS,
                alphas=alphas,
            )
            model, _ = train_bridge(
                model, X_train, T_train, Y_train, S_train, cfg, prior_weight)
            pehe, ctx = eval_bridge(model, X_test, tau_test, S_test, cfg)

            all_results[cond].append({"pehe": pehe, "ctx_metrics": ctx})

            print(f"  Overall PEHE: {pehe:.4f}")
            for s in range(N_CONTEXTS):
                cm = ctx[s]
                print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {cm['pehe']:.4f}")
            print()

        elapsed     = time.time() - t_seed
        total_so_far = time.time() - t_start
        print(f"\nSeed {seed_idx+1}/{CONFIG['n_seeds']} done in {elapsed/60:.1f} min  "
              f"(total so far: {total_so_far/60:.1f} min)")

    # --- Aggregated table ---
    print_table_avg(all_results)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
