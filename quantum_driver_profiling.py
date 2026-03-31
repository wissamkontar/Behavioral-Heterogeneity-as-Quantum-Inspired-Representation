"""
Quantum-Inspired Driver Behavioral Profiling (PyTorch)
======================================================
Implements the density-matrix-based learning framework for driver state
estimation and behavioral profiling from trajectory data.

Model components (identical to methodology):
  - Random Fourier Feature (RFF) mapping with L2 normalization  (§3.4)
  - Density-matrix behavioral profiles ρ_k                      (§3.5)
  - Context-dependent softmax profile activation π_k(c)          (§3.6)
  - Driver-specific state evolution with α blending              (§3.7)
  - Born-rule behavioral likelihood                              (§3.8)
  - Behavior-driven state adaptation with η                      (§3.9)
  - Negative log-likelihood estimation of all parameters         (§3.10)

Optimization: PyTorch autodiff replaces finite differences.
  - Exact gradients via backpropagation (faster + more accurate)
  - Truncated backpropagation through time for memory efficiency
  - All model equations are unchanged
"""

import numpy as np
import pandas as pd
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


# ============================================================
# Utility: Flushed print for SLURM logs
# ============================================================

def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. Data Loading and Standardization (§3.2, §3.3)
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    behavior_cols = ["delta_v", "a_hdv", "headway_s"]
    context_cols  = ["d_ped", "d_stop", "density", "avg_v"]

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(df[behavior_cols].values)

    scaler_c = StandardScaler()
    C = scaler_c.fit_transform(df[context_cols].values)

    ids   = df["hdv_id"].values
    times = df["time"].values

    log(f"Loaded {len(df):,} samples | d={X.shape[1]} behavioral | q={C.shape[1]} contextual")
    return X, C, ids, times, scaler_x, scaler_c

# ============================================================
# 2. Random Fourier Feature Map (§3.4)
# ============================================================

def compute_rff(X, D=100, gamma=1.0):
    rff = RBFSampler(gamma=gamma, n_components=D, random_state=42)
    Phi = rff.fit_transform(X)

    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Phi /= norms

    log(f"RFF dimension: D={Phi.shape[1]}")
    return Phi, rff


# ============================================================
# 3. Model Definition (§3.5–§3.9)
# ============================================================

class QuantumDriverModel(nn.Module):
    """
    Quantum-inspired driver behavioral profiling model.

    Learnable parameters:
      - V_k (K matrices, D×rank): profile factors, ρ_k = V_k V_k^T / Tr(V_k V_k^T)
      - beta (K×q): context-to-profile activation weights
      - logit_alpha: controls state evolution blending (sigmoid → α)
      - logit_eta: controls behavioral adaptation strength (sigmoid → η)

    All density-matrix constraints (symmetric, PSD, Tr=1) are enforced
    by construction through the V_k V_k^T parameterization.
    """

    def __init__(self, K, D, q, rank=100, alpha_init=0.2, eta_init=0.1):
        super().__init__()

        self.K = K
        self.D = D
        self.rank = rank
        #self.temperature = 3.0

        # Profile factors V_k (§3.5)
        self.Vs = nn.ParameterList([
            nn.Parameter(torch.randn(D, rank) * 0.1) for _ in range(K)
        ])

        # Context activation weights β_k (§3.6)
        self.beta = nn.Parameter(torch.randn(K, q) * 0.05)

        # α and η in logit space for unconstrained optimization
        self.fixed_alpha = alpha_init
        
        self.logit_eta = nn.Parameter(
            torch.tensor(np.log(eta_init / (1.0 - eta_init)),
                         dtype=torch.float32)
        )

    def get_alpha(self):
        return self.fixed_alpha

    def get_eta(self):
        return torch.sigmoid(self.logit_eta)

    def build_profiles(self):
        """Build density matrices ρ_k = V_k V_k^T / Tr(V_k V_k^T) (§3.5)."""
        profiles = []
        for V in self.Vs:
            M = V @ V.T
            tr = torch.trace(M)
            if tr < 1e-12:
                profiles.append(torch.eye(self.D) / self.D)
            else:
                profiles.append(M / tr)
        return profiles

    def softmax_activation(self, c):
        """π_k(c) = softmax(β^T c) (§3.6)."""
        logits = (self.beta @ c) #/ self.temperature
        return torch.softmax(logits, dim=0)

    def forward_chunk(self, Phi_chunk, C_chunk, ids_chunk, driver_states):
        """
        Process a chunk of sequential observations.

        For each observation:
          1. Context activation π_k(c)              (§3.6)
          2. State evolution with α                  (§3.7)
          3. Born-rule likelihood                    (§3.8)
          4. Behavior-driven adaptation with η       (§3.9)
        """
        alpha = self.get_alpha()
        eta = self.get_eta()
        profiles = self.build_profiles()

        chunk_size = Phi_chunk.shape[0]
        total_negloglik = torch.tensor(0.0)

        identity = torch.eye(self.D) / self.D

        for i in range(chunk_size):
            driver = ids_chunk[i]
            phi = Phi_chunk[i]
            c = C_chunk[i]

            if driver not in driver_states:
                driver_states[driver] = identity.clone()

            rho_prev = driver_states[driver]

            # Step 1: Context-dependent activation (§3.6)
            pi = self.softmax_activation(c)

            # Step 2: State evolution (§3.7)
            mixture = torch.zeros(self.D, self.D)
            for k in range(self.K):
                mixture = mixture + pi[k] * profiles[k]
            rho_t = (1.0 - alpha) * rho_prev + alpha * mixture

            # Step 3: Born-rule likelihood (§3.8)
            p = phi @ rho_t @ phi
            p = torch.clamp(p, min=1e-12)
            total_negloglik = total_negloglik - torch.log(p)

            # Step 4: Behavior-driven adaptation (§3.9)
            outer = torch.outer(phi, phi)
            rho_t = (1.0 - eta) * rho_t + eta * outer

            driver_states[driver] = rho_t.detach()

        return total_negloglik, driver_states


# ============================================================
# 4. Density Matrix Enforcement
# ============================================================

def enforce_density_matrix_np(rho):
    rho = (rho + rho.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    total = np.sum(eigvals)
    if total < 1e-12:
        return np.eye(rho.shape[0]) / rho.shape[0]
    eigvals /= total
    return (eigvecs * eigvals) @ eigvecs.T


# ============================================================
# 5. Checkpointing
# ============================================================

def save_checkpoint(path, epoch, model, optimizer, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }, path)
    log(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log(f"  Resumed from epoch {ckpt['epoch']+1}, best_loss={ckpt['best_loss']:,.2f}")
    return ckpt["epoch"], ckpt["best_loss"]


# ============================================================
# 6. Training Loop (§3.10)
# ============================================================

def train_model(Phi, C, ids, K=3, D=100, rank=100, epochs=15,
                lr=0.005, alpha_init=0.2, eta_init=0.1,
                chunk_size=5000, checkpoint_dir="checkpoints",
                resume=True, seed=42):
                
    lambda_entropy = 1e-4   # start small: 1e-4 to 1e-3
    
    q = C.shape[1]
    N = len(Phi)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "training_checkpoint.pt")

    Phi_t = torch.tensor(Phi, dtype=torch.float32)
    C_t   = torch.tensor(C, dtype=torch.float32)

    torch.manual_seed(seed)
    log(f"Random seed: {seed}")

    model = QuantumDriverModel(K, D, q, rank=rank,
                               alpha_init=alpha_init, eta_init=eta_init)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_loss = float("inf")

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch += 1

    log(f"\n{'='*60}")
    log(f"Training | K={K}, D={D}, rank={rank}, epochs={epochs}")
    log(f"Chunk size: {chunk_size} | Optimizer: Adam (lr={lr})")
    log(f"Starting from epoch {start_epoch+1}")
    log(f"{'='*60}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        alpha_val = model.get_alpha()
        eta_val = model.get_eta().item()

        log(f"\n--- Epoch {epoch+1}/{epochs} | α={alpha_val:.4f}, η={eta_val:.4f} ---")

        driver_states = {}
        epoch_loss = 0.0
        n_chunks = (N + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)

            Phi_chunk = Phi_t[start:end]
            C_chunk   = C_t[start:end]
            ids_chunk = ids[start:end]

            optimizer.zero_grad()
            chunk_loss, driver_states = model.forward_chunk(
                Phi_chunk, C_chunk, ids_chunk, driver_states
            )
            
            # ---- Entropy regularization on profiles ----
            profiles = model.build_profiles()
            entropy_penalty = torch.tensor(0.0, device=Phi_chunk.device)
            
            for rho in profiles:
                eigvals = torch.linalg.eigvalsh(rho)
                eigvals = torch.clamp(eigvals, min=1e-12)
                entropy = -(eigvals * torch.log(eigvals)).sum()
                entropy_penalty += entropy
            
            # subtract because we minimize loss
            chunk_loss = chunk_loss - lambda_entropy * entropy_penalty            

            chunk_loss.backward()
            optimizer.step()

            epoch_loss += chunk_loss.item()

            if (chunk_idx + 1) % 50 == 0 or (chunk_idx + 1) == n_chunks:
                processed = end
                avg_so_far = epoch_loss / processed
                log(f"  Chunk {chunk_idx+1}/{n_chunks} | "
                    f"Processed: {processed:,} | "
                    f"Running avg NLL: {avg_so_far:.6f}")

        avg_nll = epoch_loss / N
        elapsed = time.time() - epoch_start

        log(f"  Epoch {epoch+1} complete | "
            f"Total loss: {epoch_loss:,.2f} | "
            f"Avg NLL: {avg_nll:.6f} | "
            f"Time: {elapsed/60:.1f} min")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            log(f"  >> New best loss! Saved best_model.pt")

        save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)

    # Load best
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
        log("Loaded best model parameters.")

    alpha_final = model.get_alpha()
    eta_final = model.get_eta().item()
    beta_final = model.beta.detach().numpy()
    Vs_final = [V.detach().numpy() for V in model.Vs]
    rho_profiles = []
    for V_np in Vs_final:
        M = V_np @ V_np.T
        tr = np.trace(M)
        rho = M / tr if tr > 1e-12 else np.eye(D) / D
        rho = enforce_density_matrix_np(rho)
        rho_profiles.append(rho)

    log(f"\n{'='*60}")
    log(f"Training complete.")
    log(f"  Best loss: {best_loss:,.2f}")
    log(f"  Final α={alpha_final:.4f}, η={eta_final:.4f}")
    log(f"  Final β:\n{beta_final}")
    log(f"{'='*60}")

    return rho_profiles, Vs_final, beta_final, alpha_final, eta_final


# ============================================================
# 7. Interpretation (§3.11)
# ============================================================

def interpret_profiles(rho_profiles):
    log(f"\n{'='*60}")
    log("Profile Interpretation (Eigenanalysis)")
    log(f"{'='*60}")
    for k, rho in enumerate(rho_profiles):
        eigvals, eigvecs = np.linalg.eigh(rho)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]

        top_n = 10
        log(f"\n  Profile {k+1}:")
        log(f"    Top {top_n} eigenvalues: {np.array2string(eigvals[:top_n], precision=6)}")
        log(f"    Cumulative weight (top 5):  {np.sum(eigvals[:5]):.4f}")
        log(f"    Cumulative weight (top 10): {np.sum(eigvals[:10]):.4f}")
        log(f"    Trace: {np.sum(eigvals):.6f}")


# ============================================================
# 8. Save Results
# ============================================================

def save_results(rho_profiles, Vs, beta, alpha, eta, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    for k, rho in enumerate(rho_profiles):
        np.save(os.path.join(output_dir, f"rho_profile_{k+1}.npy"), rho)

    np.savez(os.path.join(output_dir, "model_params.npz"),
             beta=beta, alpha=alpha, eta=eta,
             **{f"V_{k}": V for k, V in enumerate(Vs)})

    log(f"Results saved to {output_dir}/")


def back_project_modes(rho_profiles, rff_sampler, scaler_x, behavior_cols, output_dir="results"):
    """
    Back-project top eigenvectors of each profile to behavioral space.
    For each eigenmode, find the behavioral signature by evaluating
    φ(x)^T v across a grid of behavioral values.
    """
    from itertools import product as iterproduct
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    # Save RFF sampler and scaler for reproducibility
    with open(os.path.join(output_dir, "rff_sampler.pkl"), "wb") as f:
        pickle.dump(rff_sampler, f)
    with open(os.path.join(output_dir, "scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)

    n_grid = 30  # per dimension
    grid_1d = np.linspace(-3, 3, n_grid)
    grid = np.array(list(iterproduct(grid_1d, grid_1d, grid_1d)))  # shape (n_grid^3, 3)

    # Map grid to RFF space and normalize
    Phi_grid = rff_sampler.transform(grid)
    norms = np.linalg.norm(Phi_grid, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Phi_grid /= norms

    log(f"\n{'='*60}")
    log("Back-Projection to Behavioral Space")
    log(f"{'='*60}")
    log(f"  Grid: {n_grid}^3 = {len(grid):,} points in standardized space")

    for k, rho in enumerate(rho_profiles):
        eigvals, eigvecs = np.linalg.eigh(rho)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        log(f"\n  Profile {k+1}:")

        n_modes = min(5, np.sum(eigvals > 0.001))  # only meaningful modes
        if n_modes == 0:
            n_modes = 1

        for m in range(n_modes):
            v = eigvecs[:, m]
            activations = Phi_grid @ v

            # Peak activation
            peak_idx = np.argmax(activations**2)
            peak_std = grid[peak_idx]
            peak_orig = scaler_x.inverse_transform(peak_std.reshape(1, -1))[0]

            # Weighted mean (using activation^2 as weight)
            weights = activations**2
            weights /= weights.sum()
            mean_std = (weights[:, None] * grid).sum(axis=0)
            mean_orig = scaler_x.inverse_transform(mean_std.reshape(1, -1))[0]

            # Weighted std
            var_std = (weights[:, None] * (grid - mean_std)**2).sum(axis=0)
            std_orig = np.sqrt(var_std) * scaler_x.scale_

            log(f"    Mode {m+1} (λ={eigvals[m]:.4f}):")
            for j, col in enumerate(behavior_cols):
                log(f"      {col}: peak={peak_orig[j]:.3f}, "
                    f"mean={mean_orig[j]:.3f}, std={std_orig[j]:.3f}")

        # Save eigenvectors and activations for this profile
        np.savez(os.path.join(output_dir, f"profile_{k+1}_modes.npz"),
                 eigvals=eigvals[:10],
                 eigvecs=eigvecs[:, :10],
                 grid_std=grid,
                 behavior_cols=behavior_cols)

    log(f"\n  Back-projection data saved to {output_dir}/")

# ============================================================
# 9. Main Pipeline
# ============================================================

if __name__ == "__main__":

    log(f"Python:  {sys.version}")
    log(f"NumPy:   {np.__version__}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA:    {torch.cuda.is_available()}")
    log(f"Start:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Configuration ---
    DATA_PATH  = "TGSIM_Combined_30s_Cleaned_3000_0.999.csv"
    D          = 100
    K          = 4
    RANK       = D       # full-rank
    EPOCHS     = 15
    CHUNK_SIZE = 5000
    LR         = 0.005

    # --- Load data ---
    X, C, ids, times, scaler_x, scaler_c = load_data(DATA_PATH)

    # --- Compute normalized RFF features ---
    Phi, rff_sampler = compute_rff(X, D=D, gamma=1.0)

    # --- Train model ---
    SEED = 100  # change for subsequent runs
    rho_profiles, Vs, beta, alpha, eta = train_model(
        Phi, C, ids,
        K=K, D=D, rank=RANK, epochs=EPOCHS,
        lr=LR, alpha_init=0.2, eta_init=0.1,
        chunk_size=CHUNK_SIZE,
        checkpoint_dir=f"checkpoints_seed{SEED}",
        resume=True,
        seed=SEED,
    )

    # --- Interpret ---
    interpret_profiles(rho_profiles)

    # --- Save ---
    save_results(rho_profiles, Vs, beta, alpha, eta, output_dir=f"results_seed{SEED}")

    # --- Back-project modes to behavioral space ---
    behavior_cols = ["delta_v", "a_hdv", "headway_s"]
    back_project_modes(rho_profiles, rff_sampler, scaler_x, behavior_cols,
                       output_dir=f"results_seed{SEED}")

    log(f"\nEnd:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("Done.")
