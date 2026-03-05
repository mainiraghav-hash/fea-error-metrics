"""
Generate sample data for FEA vs Experiment dashboard testing.
Simulates a nonlinear force-displacement response (tensile coupon with plasticity).

FEA result    : smooth bilinear curve (elastic → plastic with hardening)
Experiment 1  : similar response with realistic noise + slight stiffness variation
Experiment 2  : second test run, slightly different yield point and scatter
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ── Material / model parameters ───────────────────────────────────────────────
E        = 210_000   # MPa  (elastic modulus, steel-ish)
sigma_y  = 350       # MPa  (yield stress)
H        = 2_100     # MPa  (isotropic hardening modulus, ~1% of E)
L        = 50.0      # mm   (gauge length)
A        = 50.0      # mm²  (cross-section area)

eps_y    = sigma_y / E          # yield strain
disp_y   = eps_y * L            # yield displacement  ≈ 0.083 mm

# Displacement array: 0 → 0.6 mm (beyond yield, into plastic range)
disp = np.linspace(0, 0.60, 120)

# ── FEA: clean bilinear ────────────────────────────────────────────────────────
def bilinear_force(d, E_, sigma_y_, H_, L_, A_):
    eps    = d / L_
    eps_y_ = sigma_y_ / E_
    sigma  = np.where(
        eps <= eps_y_,
        E_ * eps,
        sigma_y_ + H_ * (eps - eps_y_),
    )
    return sigma * A_

fea_force = bilinear_force(disp, E, sigma_y, H, L, A)

fea_df = pd.DataFrame({
    "displacement_mm": np.round(disp, 5),
    "force_N":         np.round(fea_force, 3),
})
fea_df.to_csv("sample_fea.csv", index=False)
print("Saved sample_fea.csv")

# ── Experiment 1: noise + ~3% stiffer, yield 2% lower ─────────────────────────
E1       = E       * 1.03
sigma_y1 = sigma_y * 0.98
noise1   = rng.normal(0, 12, size=len(disp))   # ±12 N scatter

exp1_force = bilinear_force(disp, E1, sigma_y1, H, L, A) + noise1

exp1_df = pd.DataFrame({
    "displacement_mm": np.round(disp, 5),
    "force_N":         np.round(exp1_force, 3),
})
exp1_df.to_csv("sample_experiment_1.csv", index=False)
print("Saved sample_experiment_1.csv")

# ── Experiment 2: noise + ~1% softer, yield 1.5% higher ───────────────────────
E2       = E       * 0.99
sigma_y2 = sigma_y * 1.015
noise2   = rng.normal(0, 15, size=len(disp))   # slightly more scatter

exp2_force = bilinear_force(disp, E2, sigma_y2, H, L, A) + noise2

exp2_df = pd.DataFrame({
    "displacement_mm": np.round(disp, 5),
    "force_N":         np.round(exp2_force, 3),
})
exp2_df.to_csv("sample_experiment_2.csv", index=False)
print("Saved sample_experiment_2.csv")

# ── Quick summary ──────────────────────────────────────────────────────────────
print(f"\nFEA peak force     : {fea_force[-1]:.1f} N  at {disp[-1]:.3f} mm")
print(f"Exp 1 peak force   : {exp1_force[-1]:.1f} N  at {disp[-1]:.3f} mm")
print(f"Exp 2 peak force   : {exp2_force[-1]:.1f} N  at {disp[-1]:.3f} mm")
print(f"Yield displacement : {disp_y:.4f} mm")
