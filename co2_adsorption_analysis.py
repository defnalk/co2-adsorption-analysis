"""
co2_adsorption_analysis.py
--------------------------
Analysis of CO₂ adsorption data from a solid sorbent system.

Covers:
  1. Adsorption isotherm fitting (Langmuir & Freundlich models)
  2. Breakthrough curve simulation
  3. Regeneration cycle analysis
  4. Visualisation of all results

Context: Relevant to post-combustion carbon capture using solid sorbents
(e.g. amine-functionalised silica, zeolite 13X, MOFs).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLOUR_CO2  = "#e63946"
COLOUR_FIT  = "#1d3557"
COLOUR_LANG = "#457b9d"
COLOUR_FRND = "#a8dadc"


# ════════════════════════════════════════════════════════════════════════════
# 1.  ADSORPTION ISOTHERM DATA & FITTING
# ════════════════════════════════════════════════════════════════════════════

def langmuir(P, q_max, K_L):
    """
    Langmuir adsorption isotherm.

    q = q_max * K_L * P / (1 + K_L * P)

    Parameters
    ----------
    P : array-like
        Equilibrium pressure (bar).
    q_max : float
        Maximum adsorption capacity (mol kg⁻¹).
    K_L : float
        Langmuir affinity constant (bar⁻¹).

    Returns
    -------
    np.ndarray
        Adsorbed amount q (mol kg⁻¹).
    """
    return q_max * K_L * P / (1 + K_L * P)


def freundlich(P, K_F, n):
    """
    Freundlich adsorption isotherm.

    q = K_F * P^(1/n)

    Parameters
    ----------
    P : array-like
        Equilibrium pressure (bar).
    K_F : float
        Freundlich capacity constant (mol kg⁻¹ bar^(-1/n)).
    n : float
        Freundlich intensity parameter (dimensionless, n > 1 → favourable).

    Returns
    -------
    np.ndarray
        Adsorbed amount q (mol kg⁻¹).
    """
    return K_F * np.array(P) ** (1 / n)


# Simulated experimental isotherm data at 40 °C (representative of
# flue-gas post-combustion conditions) for a zeolite 13X sorbent.
# Values are consistent with published literature ranges.
P_exp = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00])
q_exp = np.array([0.42, 0.68, 1.05, 1.42, 1.65, 1.83, 2.05, 2.30, 2.48, 2.65])
q_err = np.array([0.03, 0.04, 0.04, 0.05, 0.04, 0.05, 0.06, 0.05, 0.06, 0.07])

# Fit models
popt_L, pcov_L = curve_fit(
    langmuir, P_exp, q_exp, p0=[3.0, 5.0], sigma=q_err,
    bounds=([0.0, 0.0], [np.inf, np.inf]),
)
popt_F, pcov_F = curve_fit(
    freundlich, P_exp, q_exp, p0=[3.0, 2.0], sigma=q_err,
    bounds=([0.0, 1e-6], [np.inf, np.inf]),
)

P_fit = np.linspace(0, 1.0, 300)
q_lang = langmuir(P_fit, *popt_L)
q_frnd = freundlich(P_fit, *popt_F)

print("─" * 55)
print("ISOTHERM FIT PARAMETERS")
print(f"  Langmuir:   q_max = {popt_L[0]:.3f} mol/kg,  K_L = {popt_L[1]:.3f} bar⁻¹")
print(f"  Freundlich: K_F   = {popt_F[0]:.3f},         n   = {popt_F[1]:.3f}")
print("─" * 55)


# ════════════════════════════════════════════════════════════════════════════
# 2.  BREAKTHROUGH CURVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def breakthrough_model(t, c, params):
    """
    Simple linear driving-force (LDF) model for a fixed-bed adsorber.

    dc/dt = (u/L) * (c_in - c) - rho_b/epsilon * k_LDF * (q* - q_bar)

    Simplified to a single ODE for the exit concentration assuming
    an average solid loading q_bar evolves with a characteristic time τ.

    Parameters
    ----------
    t : float
        Time (s).
    c : float
        Gas-phase CO₂ mole fraction at column exit.
    params : dict
        Model parameters.
    """
    c_in    = params["c_in"]
    tau_col = params["tau_col"]   # Column residence time (s)
    tau_ads = params["tau_ads"]   # Adsorption characteristic time (s)

    # Simplified: approach to inlet as column saturates
    dcdt = (c_in - c) / tau_col * np.exp(-t / tau_ads)
    return dcdt


# Column parameters — representative of a lab-scale fixed-bed unit
params = {
    "c_in":    0.15,   # Feed CO₂ mole fraction (15 vol%, typical flue gas)
    "tau_col": 60,     # Residence time (s)
    "tau_ads": 500,    # Adsorption time constant (s)
}

t_span = (0, 2500)
t_eval = np.linspace(0, 2500, 500)

# Use a simple analytical approximation for cleaner output
def breakthrough_curve(t, c_in, tau):
    """Sigmoidal breakthrough: c/c_in = 1 - exp(-t/tau) ** k"""
    return c_in * (1 - np.exp(-(t / tau) ** 2.5))

t_bt = t_eval
c_bt = breakthrough_curve(t_bt, params["c_in"], tau=900)

def _first_crossing(t, c, threshold):
    """Return the first time at which c >= threshold, or NaN if never."""
    idx = np.searchsorted(c, threshold, side="left")
    if idx >= len(t):
        return float("nan")
    return float(t[idx])

t_break = _first_crossing(t_bt, c_bt, 0.05 * params["c_in"])
t_sat   = _first_crossing(t_bt, c_bt, 0.95 * params["c_in"])

print(f"\nBREAKTHROUGH ANALYSIS")
print(f"  Breakthrough time (5% of c_in):    {t_break:.0f} s  ({t_break/60:.1f} min)")
print(f"  Saturation time (95% of c_in):     {t_sat:.0f} s  ({t_sat/60:.1f} min)")
print(f"  Usable bed fraction:               {t_break/t_sat:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# 3.  CYCLIC REGENERATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def working_capacity(T_ads, T_regen, q_max, K_L):
    """
    Calculate cyclic working capacity from Langmuir isotherm.

    Assumes adsorption at low T (P_CO2 = 0.15 bar, flue gas)
    and regeneration at elevated T (P_CO2 = 0.01 bar, N2 purge).

    Parameters
    ----------
    T_ads, T_regen : float
        Adsorption and regeneration temperatures (°C).
    q_max, K_L : float
        Langmuir parameters at T_ads.

    Returns
    -------
    float
        Working capacity Δq (mol kg⁻¹).
    """
    # Approximate temperature dependence via van't Hoff
    dH_ads = -35e3   # J/mol (exothermic adsorption)
    R = 8.314
    T_ads_K   = T_ads + 273.15
    T_regen_K = T_regen + 273.15
    K_regen = K_L * np.exp(-dH_ads / R * (1 / T_regen_K - 1 / T_ads_K))

    q_ads   = langmuir(0.15, q_max, K_L)
    q_regen = langmuir(0.01, q_max, K_regen)
    return q_ads - q_regen

T_regen_range = np.linspace(60, 200, 60)
wc_values = [working_capacity(40, T, *popt_L) for T in T_regen_range]

optimal_T = T_regen_range[np.argmax(wc_values)]
print(f"\nREGENERATION ANALYSIS")
print(f"  Max working capacity:  {max(wc_values):.3f} mol/kg at T_regen = {optimal_T:.0f} °C")


# ════════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# — Panel A: Isotherm ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(P_exp, q_exp, yerr=q_err, fmt="o", color=COLOUR_CO2,
             label="Experimental data", capsize=4, ms=6, zorder=5)
ax1.plot(P_fit, q_lang, "-",  color=COLOUR_LANG, lw=2,
         label=f"Langmuir  (q_max={popt_L[0]:.2f}, K={popt_L[1]:.2f})")
ax1.plot(P_fit, q_frnd, "--", color=COLOUR_FRND, lw=2,
         label=f"Freundlich (K_F={popt_F[0]:.2f}, n={popt_F[1]:.2f})")
ax1.set_xlabel("CO₂ partial pressure (bar)")
ax1.set_ylabel("Adsorbed amount (mol kg⁻¹)")
ax1.set_title("A  |  CO₂ Adsorption Isotherm (40 °C)", fontweight="bold")
ax1.legend(fontsize=8)

# — Panel B: Breakthrough curve ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_bt / 60, c_bt / params["c_in"], color=COLOUR_CO2, lw=2.5)
ax2.axvline(t_break / 60, ls=":", color="grey",  lw=1.5, label=f"Breakthrough ({t_break/60:.0f} min)")
ax2.axvline(t_sat   / 60, ls="--", color="black", lw=1.5, label=f"Saturation ({t_sat/60:.0f} min)")
ax2.fill_betweenx([0, 1], 0, t_break / 60, alpha=0.08, color="green",  label="Usable zone")
ax2.fill_betweenx([0, 1], t_break / 60, t_sat / 60, alpha=0.08, color="orange", label="Mass-transfer zone")
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("c / c₀")
ax2.set_title("B  |  CO₂ Breakthrough Curve", fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.05)

# — Panel C: Working capacity vs regeneration temperature ──────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(T_regen_range, wc_values, color=COLOUR_FIT, lw=2.5)
ax3.axvline(optimal_T, ls="--", color=COLOUR_CO2, lw=1.5,
            label=f"Optimal T = {optimal_T:.0f} °C")
ax3.fill_between(T_regen_range, 0, wc_values, alpha=0.12, color=COLOUR_FIT)
ax3.set_xlabel("Regeneration temperature (°C)")
ax3.set_ylabel("Working capacity (mol kg⁻¹)")
ax3.set_title("C  |  Cyclic Working Capacity", fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_ylim(bottom=0)

# — Panel D: Parity plot (model vs experiment) ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
q_pred_L = langmuir(P_exp, *popt_L)
q_pred_F = freundlich(P_exp, *popt_F)
ax4.scatter(q_exp, q_pred_L, color=COLOUR_LANG, label="Langmuir",    s=60, zorder=5)
ax4.scatter(q_exp, q_pred_F, color=COLOUR_CO2,  label="Freundlich",  s=60, marker="^", zorder=5)
lims = [min(q_exp) - 0.1, max(q_exp) + 0.1]
ax4.plot(lims, lims, "k--", lw=1.2, label="Perfect fit")
ax4.set_xlabel("Experimental q (mol kg⁻¹)")
ax4.set_ylabel("Predicted q (mol kg⁻¹)")
ax4.set_title("D  |  Parity Plot", fontweight="bold")
ax4.legend(fontsize=9)
ax4.set_xlim(lims); ax4.set_ylim(lims)

fig.suptitle(
    "CO₂ Adsorption Analysis — Zeolite 13X Sorbent",
    fontsize=14, fontweight="bold", y=1.01
)

plt.savefig("co2_adsorption_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved → co2_adsorption_results.png")
