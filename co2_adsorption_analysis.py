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
from scipy.optimize import curve_fit

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
    """
    return q_max * K_L * P / (1 + K_L * P)


def freundlich(P, K_F, n):
    """
    Freundlich adsorption isotherm.

    q = K_F * P^(1/n)
    """
    return K_F * np.array(P) ** (1 / n)


# Simulated experimental isotherm data at 40 °C (representative of
# flue-gas post-combustion conditions) for a zeolite 13X sorbent.
P_exp = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00])
q_exp = np.array([0.42, 0.68, 1.05, 1.42, 1.65, 1.83, 2.05, 2.30, 2.48, 2.65])
q_err = np.array([0.03, 0.04, 0.04, 0.05, 0.04, 0.05, 0.06, 0.05, 0.06, 0.07])


def fit_isotherms(P=P_exp, q=q_exp, sigma=q_err):
    """Fit Langmuir and Freundlich isotherms to (P, q) data.

    Returns
    -------
    dict with keys 'langmuir' and 'freundlich', each mapping to the
    fitted parameter array.
    """
    popt_L, _ = curve_fit(langmuir,   P, q, p0=[3.0, 5.0], sigma=sigma)
    popt_F, _ = curve_fit(freundlich, P, q, p0=[3.0, 2.0], sigma=sigma)
    return {"langmuir": popt_L, "freundlich": popt_F}


# ════════════════════════════════════════════════════════════════════════════
# 2.  BREAKTHROUGH CURVE SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def breakthrough_curve(t, c_in, tau):
    """Sigmoidal breakthrough: c/c_in = 1 - exp(-(t/tau)^2.5)."""
    return c_in * (1 - np.exp(-(t / tau) ** 2.5))


def breakthrough_times(t, c, c_in, low_frac=0.05, high_frac=0.95):
    """Return (t_break, t_sat) where c first crosses each fraction of c_in.

    Raises
    ------
    ValueError
        If the curve never reaches either threshold.
    """
    low_idx  = np.where(c >= low_frac  * c_in)[0]
    high_idx = np.where(c >= high_frac * c_in)[0]
    if low_idx.size == 0 or high_idx.size == 0:
        raise ValueError("Breakthrough curve never reaches threshold.")
    return float(t[low_idx[0]]), float(t[high_idx[0]])


# ════════════════════════════════════════════════════════════════════════════
# 3.  CYCLIC REGENERATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def working_capacity(T_ads, T_regen, q_max, K_L, P_ads=0.15, P_regen=0.01):
    """
    Calculate cyclic working capacity from Langmuir isotherm.

    Assumes adsorption at low T (P_ads bar, flue gas)
    and regeneration at elevated T (P_regen bar, N2 purge).
    Temperature dependence of K via van't Hoff with ΔH_ads = -35 kJ/mol.
    """
    dH_ads = -35e3   # J/mol (exothermic adsorption)
    R = 8.314
    T_ads_K   = T_ads + 273.15
    T_regen_K = T_regen + 273.15
    K_regen = K_L * np.exp(-dH_ads / R * (1 / T_regen_K - 1 / T_ads_K))

    q_ads   = langmuir(P_ads,   q_max, K_L)
    q_regen = langmuir(P_regen, q_max, K_regen)
    return q_ads - q_regen


# ════════════════════════════════════════════════════════════════════════════
# 4.  DRIVER (script entrypoint)
# ════════════════════════════════════════════════════════════════════════════

def main():  # pragma: no cover
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fits = fit_isotherms()
    popt_L = fits["langmuir"]
    popt_F = fits["freundlich"]

    P_fit  = np.linspace(0, 1.0, 300)
    q_lang = langmuir(P_fit, *popt_L)
    q_frnd = freundlich(P_fit, *popt_F)

    print("─" * 55)
    print("ISOTHERM FIT PARAMETERS")
    print(f"  Langmuir:   q_max = {popt_L[0]:.3f} mol/kg,  K_L = {popt_L[1]:.3f} bar⁻¹")
    print(f"  Freundlich: K_F   = {popt_F[0]:.3f},         n   = {popt_F[1]:.3f}")
    print("─" * 55)

    # Breakthrough
    c_in = 0.15
    t_bt = np.linspace(0, 2500, 500)
    c_bt = breakthrough_curve(t_bt, c_in, tau=900)
    t_break, t_sat = breakthrough_times(t_bt, c_bt, c_in)

    print(f"\nBREAKTHROUGH ANALYSIS")
    print(f"  Breakthrough time (5% of c_in):    {t_break:.0f} s  ({t_break/60:.1f} min)")
    print(f"  Saturation time (95% of c_in):     {t_sat:.0f} s  ({t_sat/60:.1f} min)")
    print(f"  Usable bed fraction:               {t_break/t_sat:.2f}")

    # Regeneration
    T_regen_range = np.linspace(60, 200, 60)
    wc_values = [working_capacity(40, T, *popt_L) for T in T_regen_range]
    optimal_T = T_regen_range[np.argmax(wc_values)]
    print(f"\nREGENERATION ANALYSIS")
    print(f"  Max working capacity:  {max(wc_values):.3f} mol/kg at T_regen = {optimal_T:.0f} °C")

    # Plot
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

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

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_bt / 60, c_bt / c_in, color=COLOUR_CO2, lw=2.5)
    ax2.axvline(t_break / 60, ls=":",  color="grey",  lw=1.5,
                label=f"Breakthrough ({t_break/60:.0f} min)")
    ax2.axvline(t_sat   / 60, ls="--", color="black", lw=1.5,
                label=f"Saturation ({t_sat/60:.0f} min)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("c / c₀")
    ax2.set_title("B  |  CO₂ Breakthrough Curve", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)

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

    ax4 = fig.add_subplot(gs[1, 1])
    q_pred_L = langmuir(P_exp, *popt_L)
    q_pred_F = freundlich(P_exp, *popt_F)
    ax4.scatter(q_exp, q_pred_L, color=COLOUR_LANG, label="Langmuir",   s=60, zorder=5)
    ax4.scatter(q_exp, q_pred_F, color=COLOUR_CO2,  label="Freundlich", s=60, marker="^", zorder=5)
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


if __name__ == "__main__":
    main()
