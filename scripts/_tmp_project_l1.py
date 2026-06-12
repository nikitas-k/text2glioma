#!/usr/bin/env python3
"""Project val L1 convergence to ep1000."""
import numpy as np
from scipy.optimize import curve_fit

# Val L1 progression from TensorBoard (excluding ep0 warm-up)
epochs = np.array([
    4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79,
    84, 89, 94, 99, 104, 109, 114, 119, 124, 129, 134, 139, 144, 149, 154, 159,
])
l1 = np.array([
    0.052174, 0.040538, 0.038730, 0.038056, 0.033547, 0.033958, 0.036690,
    0.032118, 0.030638, 0.036001, 0.031213, 0.030237, 0.028461, 0.029309,
    0.027255, 0.024090, 0.028688, 0.025874, 0.027903, 0.023454,
    0.024999, 0.023266, 0.022143, 0.023096, 0.025785, 0.023911,
    0.025289, 0.021117, 0.026071, 0.021111, 0.020680, 0.021199,
])


def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c


def power_law(t, a, b, c):
    return a * t**(-b) + c


def log_decay(t, a, b, c):
    return a / np.log(t + b) + c


targets = [200, 300, 500, 750, 1000]

print("=== Val L1 Projection to ep1000 ===\n")

for name, func, p0 in [
    ("Exp decay", exp_decay, [0.03, 0.01, 0.018]),
    ("Power law", power_law, [0.1, 0.3, 0.015]),
    ("Log decay", log_decay, [0.1, 2.0, 0.01]),
]:
    try:
        popt, pcov = curve_fit(func, epochs, l1, p0=p0, maxfev=10000)
        residuals = l1 - func(epochs, *popt)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"{name}:  RMSE={rmse:.6f}  params={np.round(popt, 6)}")
        for ep in targets:
            pred = func(ep, *popt)
            print(f"  ep{ep:>4d}: {pred:.6f}")
        print()
    except Exception as e:
        print(f"{name}: FAILED ({e})\n")

# Running-min envelope (best-so-far)
print("--- Running min envelope (power law fit) ---")
min_l1 = np.minimum.accumulate(l1)
# Find indices where new minima occur
new_min_mask = np.concatenate(([True], min_l1[1:] < min_l1[:-1]))
min_eps = epochs[new_min_mask]
min_vals = min_l1[new_min_mask]
print(f"  {len(min_eps)} new-minimum points")

if len(min_eps) > 3:
    try:
        popt, _ = curve_fit(power_law, min_eps, min_vals, p0=[0.1, 0.3, 0.015], maxfev=10000)
        residuals = min_vals - power_law(min_eps, *popt)
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"  Power (min): RMSE={rmse:.6f}  params={np.round(popt, 6)}")
        for ep in targets:
            print(f"    ep{ep:>4d}: {power_law(ep, *popt):.6f}")
    except Exception as e:
        print(f"  Power (min): FAILED ({e})")

# Summary table
print("\n=== Summary: Projected Val L1 ===")
print(f"{'Epoch':>6s}  {'Exp':>8s}  {'Power':>8s}  {'Log':>8s}")
print(f"{'-----':>6s}  {'-----':>8s}  {'-----':>8s}  {'-----':>8s}")
models = []
for name, func, p0 in [
    ("Exp", exp_decay, [0.03, 0.01, 0.018]),
    ("Power", power_law, [0.1, 0.3, 0.015]),
    ("Log", log_decay, [0.1, 2.0, 0.01]),
]:
    try:
        popt, _ = curve_fit(func, epochs, l1, p0=p0, maxfev=10000)
        models.append((name, func, popt))
    except:
        models.append((name, None, None))

for ep in [159] + targets:
    row = f"{ep:>6d}"
    for name, func, popt in models:
        if func is not None:
            row += f"  {func(ep, *popt):>8.5f}"
        else:
            row += f"  {'N/A':>8s}"
    row += f"  {'<-- current' if ep == 159 else ''}"
    print(row)
