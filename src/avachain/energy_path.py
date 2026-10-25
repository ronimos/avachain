"""Per-cell quantities for path-energy crack arrest (hypothesis, not validated).

A crack carries slab tension N along its path; it advances into a cell while
the energy that tension can release, G_avail, covers the weak layer's G_c.
Port of ~/avalanche-release-areas energy_path.py; see its
docs/release_area_methods.md §3.

Units: SI (Pa, m, N/m, J/m2); angles in degrees.
"""
import numpy as np

from arrest_indices import (
    G_ACC,
    R_FIT,
    plane_stress_modulus,
    residual_shear,
    slab_energy_cap,
    weak_layer_fracture_energy,
)

# Copies of constants that live elsewhere in avachain (pinned equal by
# tests/test_energy_path.py): snowpack_analysis.G_WL / NU / E_RELATION,
# release_geometry.FRICTION_DEG, and the 0.04 m fallback in compute_meloche_features.
G_WL          = 0.2e6               # weak-layer shear modulus (Pa), Meloche et al. (2025) Table 1
NU            = 0.3                 # slab Poisson's ratio, Meloche et al. (2025) Table 1
PHI_DEG       = 27.0                # snow friction angle (deg), Meloche et al. (2025) Table 1
E_RELATION    = 'vanherwijnen2016'  # relation behind the plain E_eff column
D_WL_FALLBACK = 0.04                # m; weak-layer thickness when wl_thickness is NaN (Meloche default)

# Energy-mode switches. ENERGY_DELTA is separate from the Eq. 20 delta (1.0):
# R = 0.48 was fitted to the JGR runs at delta = 0, so the R-based test uses 0.
ENERGY_DELTA        = 0.0
SIGMA_T_COLUMN      = 'sigma_t_mean'  # thickness-weighted slab σt; 'sigma_t_wl' = weakest-link bound
ENERGY_USE_RESIDUAL = False           # τr = 0 in G_c, as arrest_indices.evaluate(use_residual=False)
ENERGY_R            = None            # energy ratio R; None = arrest_indices.R_FIT. Sensitivity runs only
# Weak-layer thickness D_wl behind K_wl = G_wl / D_wl, and so G_c ∝ D_wl:
# 'sequence' = wl_thickness, the whole basal FC/DH sequence (as Eq. 20 uses);
# 'weakest_element' = wl_weakest_thickness, the lowest-tau_p WL element;
# a float = that thickness in m (the JGR calibration used 0.04).
KWL_THICKNESS       = 'sequence'
ENERGY_E_RELATION   = None            # E(rho) for the energy mode; None = E_RELATION (plain E_eff)


def tau_net(rho, h, psi_deg, phi_deg=PHI_DEG):
    """Net driving shear rho g h (sin psi - cos psi tan phi) (Pa); <= 0 at psi <= phi.
    Written as rho g h sin(psi - phi) / cos(phi), which is exactly 0 at psi = phi."""
    return (rho * G_ACC * h * np.sin(np.radians(psi_deg - phi_deg))
            / np.cos(np.radians(phi_deg)))


def g_c(tau_p, K_wl, delta=ENERGY_DELTA, tau_r=0.0):
    """Weak-layer fracture energy (tau_p - tau_r)^2 (1 + delta) / (2 K_wl) (J/m2)."""
    return weak_layer_fracture_energy(tau_p, K_wl, delta, tau_r)


def g_slab(sigma_t, h, E, nu=NU):
    """Slab energy cap sigma_t^2 h / (2 E') (J/m2)."""
    return slab_energy_cap(sigma_t, h, E, nu)


def n_max(sigma_t, h):
    """Slab tension cap sigma_t h (N/m)."""
    return sigma_t * h


def g_avail(N, E, h, nu=NU, R=R_FIT):
    """Energy available from slab tension N: R N^2 / (2 E' h) (J/m2)."""
    return R * np.asarray(N, float)**2 / (2.0 * plane_stress_modulus(E, nu) * h)


def energy_r():
    """R in use: ENERGY_R when set, else the Meloche Fig. 8 fit."""
    return R_FIT if ENERGY_R is None else ENERGY_R


def e_column(relation=None):
    """Feature column holding E_eff under `relation`."""
    relation = E_RELATION if relation is None else relation
    return 'E_eff' if relation == E_RELATION else f'E_eff__{relation}'


def wl_thickness(row, choice=None):
    """D_wl (m) under KWL_THICKNESS; D_WL_FALLBACK where the chosen column is missing."""
    choice = KWL_THICKNESS if choice is None else choice
    if not isinstance(choice, str):
        return float(choice)
    col = {'sequence': 'wl_thickness', 'weakest_element': 'wl_weakest_thickness'}.get(choice)
    if col is None:
        raise ValueError(f"unknown KWL_THICKNESS {choice!r}")
    try:
        D = float(row.get(col, np.nan))
    except (TypeError, ValueError):
        D = np.nan
    return D if np.isfinite(D) and D > 0 else D_WL_FALLBACK


def cell_energy(row, psi_deg, sigma_t_col=None, e_relation=None, delta=None,
                G_wl=None, use_residual=None, kwl_thickness=None, R=None):
    """tau_net, G_c, G_slab_path, N_max and ln_Pi = ln(R G_slab / G_c) for one cluster row.

    NaN where inputs are missing. Defaults come from the module constants.
    """
    sigma_t_col = SIGMA_T_COLUMN if sigma_t_col is None else sigma_t_col
    delta = ENERGY_DELTA if delta is None else delta
    G_wl = G_WL if G_wl is None else G_wl
    use_residual = ENERGY_USE_RESIDUAL if use_residual is None else use_residual

    def get(k):
        try:
            return float(row.get(k, np.nan))
        except (TypeError, ValueError):
            return np.nan

    rho, h = get('slab_density'), get('slab_thickness')
    e_relation = ENERGY_E_RELATION if e_relation is None else e_relation
    E, sig = get(e_column(e_relation)), get(sigma_t_col)
    tau_p = get('wl_shear_strength')
    K_wl = G_wl / wl_thickness(row, kwl_thickness)
    psi = float(psi_deg)
    tr = float(residual_shear(rho, h, psi, PHI_DEG)) if use_residual else 0.0
    G_c, G_sl = float(g_c(tau_p, K_wl, delta, tr)), float(g_slab(sig, h, E))
    R = energy_r() if R is None else R
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_pi = float(np.log(R * G_sl / G_c)) if G_c > 0 and G_sl > 0 else np.nan
    return {
        'tau_net': float(tau_net(rho, h, psi)),
        'G_c': G_c,
        'G_slab_path': G_sl,
        'N_max': float(n_max(sig, h)),
        'ln_Pi': ln_pi,
    }
