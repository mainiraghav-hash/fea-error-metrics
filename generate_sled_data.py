"""
Generate aircraft seat sled test data (FAR 25.562 Condition 1 - forward-facing).
Simulates occupant lumbar acceleration response from a 16g haversine input pulse.

FEA result    : clean simulation, slight over-prediction of peak
Experiment 1  : test run with realistic noise + minor timing offset
Experiment 2  : repeat test run, slightly different peak and scatter
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# ── Time axis ─────────────────────────────────────────────────────────────────
dt   = 0.001          # 1 ms sample interval (1000 Hz)
t    = np.arange(0, 0.250 + dt, dt)   # 0 → 250 ms

g    = 9.81           # m/s²

# ── Sled input: haversine pulse (FAR 25.562-ish) ─────────────────────────────
A_peak  = 16.0        # g  (peak sled deceleration)
t_start = 0.010       # pulse onset (10 ms)
T_pulse = 0.090       # pulse duration (90 ms)

def haversine(t, A, t0, T):
    """Half-versed-sine pulse, zero outside [t0, t0+T]."""
    phase = (t - t0) / T
    inside = (t >= t0) & (t <= t0 + T)
    return np.where(inside, A * 0.5 * (1 - np.cos(2 * np.pi * phase)), 0.0)

sled_input = haversine(t, A_peak, t_start, T_pulse)

# ── Seat/occupant response model ─────────────────────────────────────────────
# Dynamic amplification + phase lag + post-pulse rebound oscillation
def seat_response(t, A_peak, t_start, T_pulse,
                  daf=1.08, lag=0.006, damp=25.0, freq=18.0):
    """
    Seat lumbar response:
      - dynamic amplification factor (DAF) on the haversine
      - small time lag (seat structure compliance)
      - decaying oscillation after pulse ends
    """
    t_end   = t_start + T_pulse
    response = haversine(t, A_peak * daf, t_start + lag, T_pulse)

    # Rebound oscillation (seat springs/energy absorber)
    rebound_t0 = t_end + lag
    decay  = np.where(t > rebound_t0,
                      -0.15 * A_peak * np.exp(-damp * (t - rebound_t0))
                      * np.sin(2 * np.pi * freq * (t - rebound_t0)), 0.0)
    return response + decay

# ── FEA: clean model output ───────────────────────────────────────────────────
fea_accel = seat_response(t, A_peak, t_start, T_pulse,
                          daf=1.10, lag=0.005, damp=22.0, freq=19.0)

fea_df = pd.DataFrame({
    "time_s":     np.round(t, 4),
    "accel_g":    np.round(fea_accel, 4),
})
fea_df.to_csv("sled_fea.csv", index=False)
print("Saved sled_fea.csv")

# ── Experiment 1: standard test run ──────────────────────────────────────────
noise1   = rng.normal(0, 0.18, size=len(t))   # ±0.18 g sensor noise
exp1_accel = seat_response(t, A_peak * 0.97, t_start, T_pulse,
                           daf=1.07, lag=0.007, damp=24.0, freq=17.5) + noise1

exp1_df = pd.DataFrame({
    "time_s":     np.round(t, 4),
    "accel_g":    np.round(exp1_accel, 4),
})
exp1_df.to_csv("sled_experiment_1.csv", index=False)
print("Saved sled_experiment_1.csv")

# ── Experiment 2: repeat test run ────────────────────────────────────────────
noise2   = rng.normal(0, 0.22, size=len(t))   # slightly more scatter
exp2_accel = seat_response(t, A_peak * 1.02, t_start, T_pulse,
                           daf=1.06, lag=0.008, damp=26.0, freq=18.5) + noise2

exp2_df = pd.DataFrame({
    "time_s":     np.round(t, 4),
    "accel_g":    np.round(exp2_accel, 4),
})
exp2_df.to_csv("sled_experiment_2.csv", index=False)
print("Saved sled_experiment_2.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFEA peak      : {fea_accel.max():.2f} g  at t={t[fea_accel.argmax()]*1000:.1f} ms")
print(f"Exp 1 peak    : {exp1_accel.max():.2f} g  at t={t[exp1_accel.argmax()]*1000:.1f} ms")
print(f"Exp 2 peak    : {exp2_accel.max():.2f} g  at t={t[exp2_accel.argmax()]*1000:.1f} ms")
print(f"Pulse duration: {T_pulse*1000:.0f} ms,  Sample rate: {1/dt:.0f} Hz,  Points: {len(t)}")
