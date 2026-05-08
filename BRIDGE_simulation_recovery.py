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

# --- CATE: EMs (primary) + confounders modulating EM effects ---
    # Confounders enter tau through interactions with EMs and clinical vars,
    # reflecting that disease severity (stage), patient profile (age, gender)
    # and oncogenic drivers (C) modulate treatment response, not just baseline risk.
    tau = np.zeros(n)

    tau[m0] = (
    0.7 * np.tanh(X[m0, g["ERBB2"]])
    + 0.9 * X[m0, g["BRCA1"]]
    + 0.3 * np.tanh(X[m0, g["ESR1"]])
    )

    tau[m1] = (
    0.8 * X[m1, g["STK11"]]
    + 0.6 * np.tanh(2.0 * X[m1, g["KEAP1"]])
    + 0.65 * np.tanh(X[m1, g["KRAS"]])
    )

    tau[m2] = (
    -0.7 * np.tanh(X[m2, g["BAP1"]])
    - 0.9 * np.tanh(X[m2,g["SETD2"]])           #np.sin(np.pi * X[m2, g["SETD2"]] / 2)
    - 0.4 * np.tanh(X[m2, g["VHL"]])
    )

    eps = rng.standard_normal(n) * config["noise_std"]
    Y0  = base + eps
    Y1  = base + tau + eps
    Y   = T * Y1 + (1 - T) * Y0

    return T, Y, tau, base


# ============================================================
# 4. Expert Prior Specifications (copied from BRIDGE_tests_benchmarks_average.py)
# ============================================================

def get_prior_config(condition, p, n_contexts, context_features,
                     spurious_idx=SPURIOUS_IDX):
    configs = []
    for s in range(n_contexts):
        relevant = get_relevant(s, context_features)

        if condition == "informative":
            alpha = np.full(p, 1.0)
            alpha[relevant] = 2.0
            configs.append((alpha, 0.005))

        elif condition == "uniform":
            configs.append((np.ones(p), 0.005))

        elif condition == "missing_1":
            alpha = np.full(p, 1.0)
            # Endorse all confounders + first EM only; second EM missed.
            # Missed: BRCA1 (BRCA), KEAP1 (LUAD), SETD2 (KIRC)
            partial = context_features[s]["C"] + context_features[s]["EM"][:-1]
            alpha[partial] = 2.0
            configs.append((alpha, 0.005))

        elif condition == "spurious":
            alpha = np.full(p, 1.0)
            alpha[relevant] = 2.0
            alpha[spurious_idx] = 2.0
            configs.append((alpha, 0.005))

        elif condition == "no_reweighting":
            configs.append((None, 0.0))

        else:
            raise ValueError(f"Unknown condition: {condition}")

    return configs


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
# 9. Plots
# ============================================================

def plot_recovery_summary_avg_contexts(all_trajectories, context_features, p,
                                       spurious_idx, noise_indices, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    def lighten(rgb, amount=0.2):
        if isinstance(rgb, str):
            import matplotlib.colors as mc
            rgb = mc.to_rgb(rgb)
        return tuple(c + (1 - c) * amount for c in rgb)

    color_em_kept  = lighten((80/255, 180/255, 80/255))
    color_em_miss  = lighten((80/255, 120/255, 200/255))
    color_spurious = lighten("#8e44ad")
    color_noise    = lighten("#95a5a6")

    weighted_conds = ["missing_1", "uniform"]

    for cond in weighted_conds:
        traj_by_ctx = all_trajectories[cond]
        if not traj_by_ctx[0]:
            continue

        n_seeds = len(traj_by_ctx[0])

        c_vals         = []
        em_kept_vals   = []
        em_missed_vals = []
        spurious_vals  = []
        noise_vals     = []

        for s in range(N_CONTEXTS):
            stacked = np.stack(traj_by_ctx[s], axis=0)
            final   = stacked[:, -1, :]
            c_idx   = context_features[s]["C"]
            em_idx  = context_features[s]["EM"]
            em1_idx, em2_idx = em_idx[0], em_idx[1]

            all_role_s   = set(c_idx + em_idx + context_features[s]["I"] + [spurious_idx])
            noise_pool_s = [j for j in range(p) if j not in all_role_s]

            c_vals.append(final[:, c_idx].mean(axis=1))
            noise_vals.append(final[:, noise_pool_s].mean(axis=1))

            if cond == "missing_1":
                em_kept_vals.append(final[:, em1_idx])
                em_missed_vals.append(final[:, em2_idx])
            else:
                em_kept_vals.append(
                    final[:, [em1_idx, em2_idx]].mean(axis=1))

            if cond == "spurious":
                spurious_vals.append(final[:, spurious_idx])

        def pool(vals_list):
            arr = np.concatenate(vals_list)
            return arr.mean(), arr.std()

        if cond == "missing_1":
            _color_c       = lighten((80/255, 120/255, 200/255))
            _color_em_kept = lighten((80/255, 180/255, 80/255))
            _color_em_miss = lighten((180/255, 90/255, 80/255))
        else:
            _color_c       = lighten((80/255, 120/255, 200/255))
            _color_em_kept = color_em_kept
            _color_em_miss = color_em_miss

        groups = [("C (mean)", c_vals, _color_c)]
        if cond == "missing_1":
            groups.append(("EM (endorsed)", em_kept_vals,  _color_em_kept))
            groups.append(("EM (missed)",   em_missed_vals, _color_em_miss))
        else:
            groups.append(("EM (mean)", em_kept_vals, _color_em_kept))
        if cond == "spurious":
            groups.append((f"Spurious ({spurious_idx})", spurious_vals, color_spurious))
        groups.append(("Noise (mean)", noise_vals, color_noise))

        labels = [grp[0] for grp in groups]
        means  = [pool(grp[1])[0] for grp in groups]
        stds   = [pool(grp[1])[1] for grp in groups]
        colors = [grp[2]          for grp in groups]

        display_name = "mild" if cond == "missing_1" else cond

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        ax.axhline(y=1.0 / p, color="black", linestyle=":", alpha=0.4,
                   label=f"1/p = {1/p:.5f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Mean final weight")
        ax.legend(fontsize=8)
        ax.set_title(
            f"Final weight recovery — {display_name} (averaged across contexts, n={n_seeds})",
            fontsize=12)
        plt.tight_layout()
        fname = save_dir / f"{cond}_recovery_avg_contexts.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


# ============================================================
# 10. Aggregated reporting (copied from BRIDGE_tests_benchmarks_average.py)
# ============================================================

BRIDGE_CONDITIONS = ["informative", "uniform", "missing_1", "spurious", "no_reweighting"]
ALL_CONDITIONS    = BRIDGE_CONDITIONS   # no external baselines in this file


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
        label = "mild" if cond == "missing_1" else cond
        line = f"{label:<20s}  {m_p:.4f}±{s_p:.4f} "
        for s in range(N_CONTEXTS):
            m_cp, s_cp = agg_ctx(lst, s, "pehe")
            line += f"  {m_cp:.4f}±{s_cp:.4f} "
        print(line)
    print(sep)


# ============================================================
# 11. Main
# ============================================================

def main():
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

    # Aggregators
    all_results      = {cond: [] for cond in ALL_CONDITIONS}
    all_trajectories = {cond: {s: [] for s in range(N_CONTEXTS)}
                        for cond in BRIDGE_CONDITIONS}

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

        # --- BRIDGE / CFRNet ---
        for cond in BRIDGE_CONDITIONS:
            print(f"--- {cond} (seed {seed_idx+1}/{SIM_CONFIG['n_seeds']}) ---")
            torch.manual_seed(seed)

            ctx_configs  = get_prior_config(cond, p, N_CONTEXTS,
                                            context_features, SPURIOUS_IDX)
            alphas       = [c[0] for c in ctx_configs]
            prior_weight = ctx_configs[0][1]
            alphas_arg   = None if cond == "no_reweighting" else alphas

            model = MultiContextCFRNet(
                p=p,
                hidden_dims=cfg["hidden_dims"],
                repr_dim=cfg["repr_dim"],
                head_hidden_dim=cfg["head_hidden_dim"],
                n_contexts=N_CONTEXTS,
                alphas=alphas_arg,
            )
            model, trajectory = train_bridge(
                model, X_train, T_train, Y_train, S_train, cfg, prior_weight)
            pehe, ate_bias, ctx = eval_bridge(model, X_test, tau_test, S_test, cfg)

            all_results[cond].append(
                {"pehe": pehe, "ate_bias": ate_bias, "ctx_metrics": ctx})
            if trajectory is not None:
                for s in range(N_CONTEXTS):
                    all_trajectories[cond][s].append(trajectory[s])

            print(f"  Overall PEHE: {pehe:.4f}")
            for s in range(N_CONTEXTS):
                cm = ctx[s]
                print(f"  {CONTEXT_NAMES[s]:<6s} PEHE: {cm['pehe']:.4f}")
            print()

        elapsed      = time.time() - t_seed
        total_so_far = time.time() - t_start
        print(f"\nSeed {seed_idx+1}/{SIM_CONFIG['n_seeds']} done in {elapsed/60:.1f} min  "
              f"(total so far: {total_so_far/60:.1f} min)")

    # --- Save results ---
    plot_dir = Path("bridge_plots_sim")
    plot_dir.mkdir(exist_ok=True)
    pkl_path = plot_dir / "seed_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {pkl_path}")

    # --- Aggregated table and plots ---
    print_table_avg(all_results)
    plot_recovery_summary_avg_contexts(all_trajectories, context_features, p,
                                       SPURIOUS_IDX, noise_indices, save_dir=plot_dir)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
