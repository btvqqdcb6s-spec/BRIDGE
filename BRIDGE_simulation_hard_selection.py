"""
Simulation experiment — BRIDGE on synthetic block-correlated data
==================================================================
Same DGP as BRIDGE_tests_benchmarks_average.py but with synthetic
block-diagonal MVN features instead of real TCGA expression data.

Block layout (block_size=8, p=200):
  Block 0      : clinical   (age=0, stage=1, gender=2, rest corr. noise)
  Blocks 1–6   : context-specific C   (ESR1/PIK3CA/EGFR/KRAS/VHL/PBRM1 at block[0])
  Blocks 7–12  : EM                   (ERBB2/BRCA1/STK11/KEAP1/BAP1/SETD2 at block[0])
  Blocks 13–21 : instruments          (first index of each block)
  Blocks 22–24 : pure noise           (indices 176–199)

Features within each role block are correlated (rho_within=0.6) — the
"correlated noise neighbors" challenge for BRIDGE weight recovery.

DGP coefficients and functional forms are identical to the TCGA file.
Conditions: informative, uniform, missing_1, spurious, no_reweighting.
no_reweighting = context-blind CFRNet baseline.
"""

import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. Configuration
# ============================================================

SPURIOUS_IDX = None   # set in main() via global after build_role_indices

SIM_CONFIG = {
    "n_seeds": 5,
    "base_seed": 42,
    # Synthetic data
    "n_per_context": 2000,           # 6000 total
    "p":             200,
    "block_size":    8,
    "rho_within":    0.6,
    "spurious_idx":  None,           # set in main()
    # BRIDGE
    "epochs":               150,
    "batch_size":           256,
    "lr":                   1e-3,
    "lr_theta_multiplier":  35,
    "repr_dim":             20,
    "hidden_dims":          [256, 128],
    "head_hidden_dim":      32,
    "ipm_weight":           1.0,
    "prop_clip":            0.05,
    # DGP
    "noise_std": 1.0,
    "test_frac": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CONTEXT_NAMES = {0: "BRCA", 1: "LUAD", 2: "KIRC"}
N_CONTEXTS = 3

# Hyperparameters for the BRCA-only single-context CFRNet sweep.
# Kept separate from SIM_CONFIG so they can be tuned independently.
CFRNET_SINGLE_CONFIG = {
    "hidden_dims":     [64, 32],
    "repr_dim":        16,
    "head_hidden_dim": 16,
    "ipm_weight":      0.1,
}

# ============================================================
# 1. Synthetic data generation
# ============================================================

def generate_synthetic_X(n_per_context, p, block_size, rho_within, rng,
                         identity_block_idx=None):
    """
    Draw block-correlated MVN data.
    Each block of `block_size` features has within-block correlation `rho_within`;
    cross-block entries are zero. If `identity_block_idx` is set, that block uses
    an identity covariance (independent features) instead. Standardised to zero
    mean / unit variance column-wise (consistent with TCGA preprocessing).
    Returns X (n_total × p, float32) and S (n_total,).
    """
    n_total   = 3 * n_per_context
    n_blocks  = p // block_size
    remainder = p % block_size

    Sigma_corr = rho_within * np.ones((block_size, block_size))
    np.fill_diagonal(Sigma_corr, 1.0)
    L_corr = np.linalg.cholesky(Sigma_corr)
    L_eye  = np.eye(block_size)

    X_blocks = []
    for b in range(n_blocks):
        Z = rng.standard_normal((n_total, block_size))
        L = L_eye if b == identity_block_idx else L_corr
        X_blocks.append(Z @ L.T)
    if remainder > 0:
        X_blocks.append(rng.standard_normal((n_total, remainder)))

    X = np.concatenate(X_blocks, axis=1)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X  = ((X - mu) / sd).astype(np.float32)

    S = np.concatenate([
        np.zeros(n_per_context, dtype=int),
        np.ones(n_per_context,  dtype=int),
        2 * np.ones(n_per_context, dtype=int),
    ])
    return X, S


# ============================================================
# 2. Role-to-index mapping
# ============================================================

def build_role_indices(p, block_size):
    """
    Assign feature indices to biological roles based on block position.
    Role-feature names match the TCGA keys so generate_outcomes works unchanged.

    With block_size=8, p=200:
      22 role blocks × 8 = 176  →  pure noise: indices 176–199

    Returns:
      g               : dict of role-name → index  (same keys as TCGA resolve_all_genes)
      context_features: same nested dict structure as TCGA
      age_idx, stage_idx, gender_idx
      noise_indices   : list of all indices not assigned to any C/EM/I/clinical role
    """
    bs = block_size

    age_idx    = 0
    stage_idx  = 1
    gender_idx = 2

    g = {
        # Context-specific C (first index of each C block)
        "ESR1":  1  * bs,
        "PIK3CA":2  * bs,
        "EGFR":  3  * bs,
        "KRAS":  4  * bs,
        "VHL":   5  * bs,
        "PBRM1": 6  * bs,
        # EM (first index of each EM block)
        "ERBB2": 7  * bs,
        "BRCA1": 8  * bs,
        "STK11": 9  * bs,
        "KEAP1": 10 * bs,
        "BAP1":  11 * bs,
        "SETD2": 12 * bs,
        # Instruments (first index of each I block)
        "MKI67": 13 * bs,
        "TP53":  14 * bs,
        "CDH1":  15 * bs,
        "ALK":   16 * bs,
        "ROS1":  17 * bs,
        "MET":   18 * bs,
        "MTOR":  19 * bs,
        "VEGFA": 20 * bs,
        "KDM5C": 21 * bs,
    }

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

    n_role_blocks = 22          # 1 clinical + 6 C + 6 EM + 9 I
    first_noise   = n_role_blocks * bs
    noise_indices = list(range(first_noise, p))

    return g, context_features, age_idx, stage_idx, gender_idx, noise_indices


def get_relevant(s, context_features):
    return context_features[s]["C"] + context_features[s]["EM"]


# ============================================================
# 3. DGP — copied verbatim from BRIDGE_tests_benchmarks_average.py
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

# --- CATE: confounders + true EMs, equal-weight for monotone sweep recovery ---
    # Each relevant feature contributes ~0.5 * standardized_feature to tau,
    # giving equal Var[tau] share per feature.  Confounders (C) enter propensity
    # AND baseline AND tau; EMs are true EMs (propensity=0, baseline=0, tau≠0).
    tau = np.zeros(n)

    # BRCA: C1=ESR1 (linear), C2=PIK3CA (linear), EM1=ERBB2 (linear), EM2=BRCA1 (tanh)
    tau[m0] = (
        0.5 * X[m0, g["ESR1"]]
        + 0.5 * X[m0, g["PIK3CA"]]
        + 0.5 * X[m0, g["ERBB2"]]
        + 0.5 * np.tanh(X[m0, g["BRCA1"]])
    )

    # LUAD: C1=EGFR (linear), C2=KRAS (linear), EM1=STK11 (linear), EM2=KEAP1 (tanh)
    tau[m1] = (
        0.5 * X[m1, g["EGFR"]]
        + 0.5 * X[m1, g["KRAS"]]
        + 0.5 * X[m1, g["STK11"]]
        + 0.5 * np.tanh(X[m1, g["KEAP1"]])
    )

    # KIRC: C1=VHL (linear), C2=PBRM1 (linear), EM1=BAP1 (linear), EM2=SETD2 (tanh)
    tau[m2] = (
        -0.5 * X[m2, g["VHL"]]
        - 0.5 * X[m2, g["PBRM1"]]
        - 0.5 * X[m2, g["BAP1"]]
        - 0.5 * np.tanh(X[m2, g["SETD2"]])
    )

    eps = rng.standard_normal(n) * config["noise_std"]
    Y0  = base + eps
    Y1  = base + tau + eps
    Y   = T * Y1 + (1 - T) * Y0

    return T, Y, tau, base


# ============================================================
# 5. BRIDGE Model Components (copied from BRIDGE_tests_benchmarks_average.py)
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


class CFRNetSingle(nn.Module):
    """Single-context CFRNet for the BRCA-only sweep: no S-conditioning, no reweighting.
    Takes X[:, subset_indices] as input (pre-sliced outside the model).
    Architecture hyperparameters come from CFRNET_SINGLE_CONFIG.
    """

    def __init__(self, input_dim, hidden_dims, repr_dim, head_hidden_dim):
        super().__init__()
        self.phi = RepresentationNetwork(input_dim, hidden_dims, repr_dim)
        self.h0  = HypothesisHead(repr_dim, head_hidden_dim)
        self.h1  = HypothesisHead(repr_dim, head_hidden_dim)

    def forward(self, X, T):
        phi    = self.phi(X)
        y0     = self.h0(phi)
        y1     = self.h1(phi)
        y_pred = T * y1 + (1 - T) * y0
        return y_pred, y0, y1, phi


# ============================================================
# 6. MMD (copied from BRIDGE_tests_benchmarks_average.py)
# ============================================================

def mmd_rbf(phi_t, phi_c, sigma=1.0):
    if phi_t.shape[0] == 0 or phi_c.shape[0] == 0:
        return torch.tensor(0.0, device=phi_t.device)
    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b, p=2).pow(2) / (2.0 * sigma ** 2))
    return (kernel(phi_t, phi_t).mean() + kernel(phi_c, phi_c).mean()
            - 2.0 * kernel(phi_t, phi_c).mean())


# ============================================================
# 7. BRIDGE Training (copied from BRIDGE_tests_benchmarks_average.py)
# ============================================================

def train_bridge(model, X_train, T_train, Y_train, S_train, config, prior_weight,
                 ipm_weight=None):
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

    _ipm_w = config["ipm_weight"] if ipm_weight is None else ipm_weight

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
            loss  = mse + _ipm_w * ipm - prior_weight * model.log_prior()
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
# 8. Evaluation helpers (copied from BRIDGE_tests_benchmarks_average.py)
# ============================================================

def compute_metrics(tau_pred, tau_test, S_test):
    pehe     = np.sqrt(np.mean((tau_pred - tau_test) ** 2))
    ate_bias = np.abs(np.mean(tau_pred) - np.mean(tau_test))
    ctx = {}
    for s in range(N_CONTEXTS):
        mask = (S_test == s)
        if mask.sum() > 0:
            ctx[s] = {
                "pehe":     np.sqrt(np.mean((tau_pred[mask] - tau_test[mask]) ** 2)),
                "ate_bias": np.abs(np.mean(tau_pred[mask]) - np.mean(tau_test[mask])),
            }
    return pehe, ate_bias, ctx


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
# 9. BRCA-only single-context sweep helpers
# ============================================================

K_LEVELS = [1, 2, 3, 4]


def _brca_endorsed_set(k, context_features, age_idx, stage_idx, gender_idx):
    """Feature indices endorsed by the level-k BRCA Dirichlet prior."""
    clinical = {age_idx, stage_idx, gender_idx}
    endorsed = set(clinical)
    c_extra  = [c for c in context_features[0]["C"] if c not in clinical]
    em       = context_features[0]["EM"]
    if k >= 1: endorsed.add(c_extra[0])   # ESR1
    if k >= 2: endorsed.add(c_extra[1])   # PIK3CA
    if k >= 3: endorsed.add(em[0])        # ERBB2
    if k >= 4: endorsed.add(em[1])        # BRCA1
    return endorsed


def build_brca_feature_subsets(context_features, age_idx, stage_idx, gender_idx):
    """
    Return {k: sorted_index_list} for k=1..4, BRCA-specific:
      k=1: clinical (age, stage, gender) + ESR1
      k=2: + PIK3CA
      k=3: + ERBB2
      k=4: + BRCA1
    """
    return {k: sorted(_brca_endorsed_set(k, context_features, age_idx, stage_idx, gender_idx))
            for k in K_LEVELS}


def get_prior_config_matched_brca(k, p, context_features, age_idx, stage_idx, gender_idx):
    """
    Single (alpha, prior_weight) tuple for BRIDGE n_contexts=1 on BRCA,
    matched to level-k feature subset.  alpha[j]=2.0 for endorsed features, 1.0 elsewhere.
    """
    alpha = np.full(p, 1.0)
    alpha[list(_brca_endorsed_set(k, context_features, age_idx, stage_idx, gender_idx))] = 2.0
    return alpha, 0.005


def train_cfrnet_single(X_train_brca, T_train_brca, Y_train_brca,
                        subset_indices, config):
    """Train CFRNetSingle on BRCA-only data using CFRNET_SINGLE_CONFIG."""
    device    = config["device"]
    input_dim = len(subset_indices)
    model     = CFRNetSingle(
        input_dim,
        CFRNET_SINGLE_CONFIG["hidden_dims"],
        CFRNET_SINGLE_CONFIG["repr_dim"],
        CFRNET_SINGLE_CONFIG["head_hidden_dim"],
    ).to(device)

    X_sub = X_train_brca[:, subset_indices].astype(np.float32)
    X_t   = torch.tensor(X_sub,         dtype=torch.float32, device=device)
    T_t   = torch.tensor(T_train_brca,  dtype=torch.float32, device=device)
    Y_t   = torch.tensor(Y_train_brca,  dtype=torch.float32, device=device)

    loader    = DataLoader(TensorDataset(X_t, T_t, Y_t),
                           batch_size=config["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    ipm_w = CFRNET_SINGLE_CONFIG["ipm_weight"]
    model.train()
    for _ in range(config["epochs"]):
        for X_b, T_b, Y_b in loader:
            optimizer.zero_grad()
            y_pred, _, _, phi = model(X_b, T_b)
            mse   = nn.functional.mse_loss(y_pred, Y_b)
            idx_t = (T_b == 1).nonzero(as_tuple=True)[0]
            idx_c = (T_b == 0).nonzero(as_tuple=True)[0]
            ipm   = mmd_rbf(phi[idx_t], phi[idx_c])
            (mse + ipm_w * ipm).backward()
            optimizer.step()
        scheduler.step()
    return model


def eval_cfrnet_single(model, X_test_brca, subset_indices, tau_test_brca, config):
    """Evaluate CFRNetSingle on BRCA-only test set; returns (pehe, ate_bias)."""
    device = config["device"]
    model.eval()
    X_sub    = X_test_brca[:, subset_indices].astype(np.float32)
    X_t      = torch.tensor(X_sub, dtype=torch.float32, device=device)
    T1       = torch.ones(X_t.shape[0], device=device)
    with torch.no_grad():
        _, y0_t, y1_t, _ = model(X_t, T1)
        tau_pred = (y1_t - y0_t).cpu().numpy()
    pehe     = float(np.sqrt(np.mean((tau_pred - tau_test_brca) ** 2)))
    ate_bias = float(np.abs(np.mean(tau_pred) - np.mean(tau_test_brca)))
    return pehe, ate_bias


# ============================================================
# 10. Plots
# ============================================================

def plot_pehe_vs_k_brca(sweep_results_cfrnet_brca, sweep_results_bridge_brca,
                        sweep_results_bridge_reduced_uniform_brca, save_dir):
    """
    Single-panel PEHE vs. k line plot for the BRCA-only sweep.
    Two configurations: CFRNet reduced, BRIDGE full p=200 matched prior.
    Saved to save_dir/pehe_vs_k_brca.png.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    type1 = (80/255, 120/255, 200/255)   # blue  — BRIDGE
    type2 = (210/255, 90/255, 80/255)    # red   — CFRNet

    def get_mean_std(results_by_k):
        means, stds = [], []
        for k in K_LEVELS:
            vals = [d["pehe"] for d in results_by_k[k]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    series = [
        ("BRIDGE (matched prior, full p)",   sweep_results_bridge_brca,   type1, "--"),
        ("CFRNet single (reduced)",          sweep_results_cfrnet_brca,   type2, "-"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, res, color, ls in series:
        m, s = get_mean_std(res)
        ax.plot(K_LEVELS, m, label=label, color=color, linestyle=ls,
                linewidth=2, marker="o")
        ax.fill_between(K_LEVELS, m - s, m + s, alpha=0.15, color=color)
    ax.set_xlabel("Feature subset level k")
    ax.set_ylabel("PEHE (BRCA test set)")
    ax.set_xticks(K_LEVELS)
    ax.legend(fontsize=9)
    ax.set_title("BRCA-only: PEHE vs. feature subset level\n"
                 "(CFRNet reduced, BRIDGE full p=200 matched prior)", fontsize=11)
    plt.tight_layout()
    fname = save_dir / "pehe_vs_k_brca.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def print_table_sweep_brca(sweep_results_cfrnet_brca, sweep_results_bridge_brca,
                            sweep_results_bridge_reduced_uniform_brca):
    """Print mean±std PEHE table for the BRCA-only sweep."""
    def agg(results_by_k, k):
        lst      = results_by_k[k]
        pehe_all = [d["pehe"] for d in lst]
        return np.mean(pehe_all), np.std(pehe_all)

    cell = 18
    sep  = "=" * (38 + cell + 3)
    print("\n--- BRCA-Only Feature-Subset Sweep ---")
    print(sep)
    print(f"{'Method/k':<36s}  {'BRCA PEHE':^{cell}s}")
    print("-" * (38 + cell + 3))
    for k in K_LEVELS:
        for label, results in [
            (f"CFRNet_single_k{k}",               sweep_results_cfrnet_brca),
            (f"BRIDGE_brca_k{k}",                 sweep_results_bridge_brca),
            (f"BRIDGE_brca_reduced_uniform_k{k}", sweep_results_bridge_reduced_uniform_brca),
        ]:
            m_p, s_p = agg(results, k)
            print(f"{label:<36s}  {m_p:.4f}±{s_p:.4f}")
        print()
    print(sep)


# ============================================================
# 11. Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Override SIM_CONFIG['n_seeds'] for a quick smoke test")
    args = parser.parse_args()
    if args.n_seeds is not None:
        SIM_CONFIG["n_seeds"] = args.n_seeds

    global SPURIOUS_IDX

    t_start = time.time()

    p          = SIM_CONFIG["p"]
    block_size = SIM_CONFIG["block_size"]

    g, context_features, age_idx, stage_idx, gender_idx, noise_indices = \
        build_role_indices(p, block_size)

    # Pick SPURIOUS_IDX from the pure-noise tail; verify no collision with role indices
    SPURIOUS_IDX = noise_indices[10]
    SIM_CONFIG["spurious_idx"] = SPURIOUS_IDX
    spurious_block = SPURIOUS_IDX // block_size

    all_role_indices = set()
    for s in range(N_CONTEXTS):
        all_role_indices.update(context_features[s]["C"])
        all_role_indices.update(context_features[s]["EM"])
        all_role_indices.update(context_features[s]["I"])
    assert SPURIOUS_IDX not in all_role_indices, \
        f"SPURIOUS_IDX={SPURIOUS_IDX} collides with a role index"

    print("Synthetic simulation experiment")
    print(f"p={p}, block_size={block_size}, rho_within={SIM_CONFIG['rho_within']}")
    print(f"n_per_context={SIM_CONFIG['n_per_context']}, "
          f"total={3*SIM_CONFIG['n_per_context']}")
    print(f"Role blocks: 1 clinical + 6 C + 6 EM + 9 I = 22 × {block_size} = "
          f"{22*block_size}; noise: {len(noise_indices)} features "
          f"[{noise_indices[0]}–{noise_indices[-1]}]")
    print(f"SPURIOUS_IDX={SPURIOUS_IDX}  (noise_indices[10])")
    print(f"Spurious block index: {spurious_block} "
          f"(features {spurious_block*block_size}–{(spurious_block+1)*block_size-1} "
          f"set to identity covariance)")
    print(f"Device: {SIM_CONFIG['device']}")
    print(f"Architecture: [...] -> {SIM_CONFIG['hidden_dims']} -> {SIM_CONFIG['repr_dim']}")
    print(f"Noise std: {SIM_CONFIG['noise_std']}   Epochs: {SIM_CONFIG['epochs']}")
    print(f"Running {SIM_CONFIG['n_seeds']} seeds "
          f"(base_seed={SIM_CONFIG['base_seed']})\n")

    # Aggregators — BRCA-only single-context sweep
    sweep_results_cfrnet_brca                  = {k: [] for k in K_LEVELS}
    sweep_results_bridge_brca                  = {k: [] for k in K_LEVELS}
    sweep_results_bridge_reduced_uniform_brca  = {k: [] for k in K_LEVELS}
    sweep_trajectories_bridge_brca             = {k: [] for k in K_LEVELS}

    for seed_idx in range(SIM_CONFIG["n_seeds"]):
        seed = SIM_CONFIG["base_seed"] + seed_idx
        cfg  = {**SIM_CONFIG, "seed": seed}
        t_seed = time.time()

        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{SIM_CONFIG['n_seeds']}  (seed={seed})")
        print(f"{'='*60}")

        # Generate fresh synthetic X for this seed
        rng_X    = np.random.default_rng(seed)
        X, S     = generate_synthetic_X(
            SIM_CONFIG["n_per_context"], p,
            SIM_CONFIG["block_size"], SIM_CONFIG["rho_within"], rng_X,
            identity_block_idx=spurious_block)

        # Stratified train/test split by S
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

        # --- BRCA-only single-context sweep ---
        brca_train_mask  = (S_train == 0)
        brca_test_mask   = (S_test  == 0)
        X_train_brca     = X_train[brca_train_mask]
        T_train_brca     = T_train[brca_train_mask]
        Y_train_brca     = Y_train[brca_train_mask]
        S_train_brca     = np.zeros(brca_train_mask.sum(), dtype=int)
        X_test_brca      = X_test[brca_test_mask]
        tau_test_brca    = tau_test[brca_test_mask]
        S_test_brca      = np.zeros(brca_test_mask.sum(), dtype=int)

        brca_subsets = build_brca_feature_subsets(
            context_features, age_idx, stage_idx, gender_idx)

        print(f"\n--- BRCA-only sweep (seed {seed_idx+1}/{SIM_CONFIG['n_seeds']}) ---")
        print(f"  BRCA train: {len(X_train_brca)}  test: {len(X_test_brca)}")

        for k in K_LEVELS:
            subset = brca_subsets[k]
            print(f"  k={k}: {len(subset)} features  {subset}")

            # CFRNetSingle on BRCA-filtered reduced subset
            torch.manual_seed(seed)
            cfrnet_single = train_cfrnet_single(
                X_train_brca, T_train_brca, Y_train_brca, subset, cfg)
            pehe_c, ate_c = eval_cfrnet_single(
                cfrnet_single, X_test_brca, subset, tau_test_brca, cfg)
            sweep_results_cfrnet_brca[k].append(
                {"pehe": pehe_c, "brca_metrics": {"pehe": pehe_c}})
            print(f"    CFRNet   PEHE: {pehe_c:.4f}")

            # BRIDGE n_contexts=1 on BRCA only, full p=200, matched prior
            torch.manual_seed(seed + 1)
            alpha_brca, pw_brca = get_prior_config_matched_brca(
                k, p, context_features, age_idx, stage_idx, gender_idx)
            bridge_brca = MultiContextCFRNet(
                p=p,
                hidden_dims=cfg["hidden_dims"],
                repr_dim=cfg["repr_dim"],
                head_hidden_dim=cfg["head_hidden_dim"],
                n_contexts=1,
                alphas=[alpha_brca],
            )
            bridge_brca, traj_brca = train_bridge(
                bridge_brca, X_train_brca, T_train_brca, Y_train_brca,
                S_train_brca, cfg, pw_brca)
            pehe_b, ate_b, ctx_b = eval_bridge(
                bridge_brca, X_test_brca, tau_test_brca, S_test_brca, cfg)
            sweep_results_bridge_brca[k].append(
                {"pehe": pehe_b, "brca_metrics": {"pehe": pehe_b}})
            sweep_trajectories_bridge_brca[k].append(traj_brca[0])
            print(f"    BRIDGE   PEHE: {pehe_b:.4f}")

            # BRIDGE n_contexts=1 on BRCA only, reduced subset, uniform prior
            # Architecture matches CFRNetSingle exactly (CFRNET_SINGLE_CONFIG).
            # The only difference from CFRNet-single is the ImportanceWeightModule.
            torch.manual_seed(seed + 2)
            p_sub     = len(subset)
            alpha_uni = np.full(p_sub, 1.0)
            bridge_ru = MultiContextCFRNet(
                p=p_sub,
                hidden_dims=CFRNET_SINGLE_CONFIG["hidden_dims"],
                repr_dim=CFRNET_SINGLE_CONFIG["repr_dim"],
                head_hidden_dim=CFRNET_SINGLE_CONFIG["head_hidden_dim"],
                n_contexts=1,
                alphas=[alpha_uni],
            )
            bridge_ru, _ = train_bridge(
                bridge_ru, X_train_brca[:, subset], T_train_brca, Y_train_brca,
                S_train_brca, cfg, 0.005,
                ipm_weight=CFRNET_SINGLE_CONFIG["ipm_weight"])
            pehe_ru, ate_ru, ctx_ru = eval_bridge(
                bridge_ru, X_test_brca[:, subset], tau_test_brca, S_test_brca, cfg)
            sweep_results_bridge_reduced_uniform_brca[k].append(
                {"pehe": pehe_ru, "brca_metrics": {"pehe": pehe_ru}})
            print(f"    BRIDGE_RU PEHE: {pehe_ru:.4f}")

        elapsed      = time.time() - t_seed
        total_so_far = time.time() - t_start
        print(f"\nSeed {seed_idx+1}/{SIM_CONFIG['n_seeds']} done in {elapsed/60:.1f} min  "
              f"(total so far: {total_so_far/60:.1f} min)")

    # --- Save results ---
    plot_dir = Path("bridge_plots_sim")
    plot_dir.mkdir(exist_ok=True)
    pkl_path = plot_dir / "seed_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"sweep_cfrnet_brca":                 sweep_results_cfrnet_brca,
                     "sweep_bridge_brca":                  sweep_results_bridge_brca,
                     "sweep_bridge_reduced_uniform_brca":  sweep_results_bridge_reduced_uniform_brca,
                     "sweep_trajectories_bridge_brca":     sweep_trajectories_bridge_brca}, f)
    print(f"\nSaved {pkl_path}")

    # --- BRCA-only sweep reporting ---
    print_table_sweep_brca(sweep_results_cfrnet_brca, sweep_results_bridge_brca,
                           sweep_results_bridge_reduced_uniform_brca)
    plot_pehe_vs_k_brca(sweep_results_cfrnet_brca, sweep_results_bridge_brca,
                        sweep_results_bridge_reduced_uniform_brca,
                        save_dir=plot_dir)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
