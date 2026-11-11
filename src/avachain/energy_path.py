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
    plane_stress_modulus,
    residual_shear,
    slab_energy_cap,
    weak_layer_fracture_energy,
)
import model_params as mp



def tau_net(rho, h, psi_deg, phi_deg=None):
    """Net driving shear rho g h (sin psi - cos psi tan phi) (Pa); <= 0 at psi <= phi.
    Written as rho g h sin(psi - phi) / cos(phi), which is exactly 0 at psi = phi."""
    phi_deg = mp.PHI_DEG if phi_deg is None else phi_deg
    return (rho * G_ACC * h * np.sin(np.radians(psi_deg - phi_deg))
            / np.cos(np.radians(phi_deg)))


def g_c(tau_p, K_wl, delta=None, tau_r=0.0):
    """Weak-layer fracture energy (tau_p - tau_r)^2 (1 + delta) / (2 K_wl) (J/m2)."""
    delta = mp.ENERGY_DELTA if delta is None else delta
    return weak_layer_fracture_energy(tau_p, K_wl, delta, tau_r)


def g_slab(sigma_t, h, E, nu=None):
    """Slab energy cap sigma_t^2 h / (2 E') (J/m2)."""
    nu = mp.NU if nu is None else nu
    return slab_energy_cap(sigma_t, h, E, nu)


def n_max(sigma_t, h):
    """Slab tension cap sigma_t h (N/m)."""
    return sigma_t * h


def g_avail(N, E, h, nu=None, R=None):
    """Energy available from slab tension N: R N^2 / (2 E' h) (J/m2)."""
    nu = mp.NU if nu is None else nu
    R = mp.R_FIT if R is None else R
    return R * np.asarray(N, float)**2 / (2.0 * plane_stress_modulus(E, nu) * h)


def energy_r():
    """R in use: ENERGY_R when set, else the Meloche Fig. 8 fit."""
    return mp.R_FIT if mp.ENERGY_R is None else mp.ENERGY_R


def e_column(relation=None):
    """Feature column holding E_eff under `relation`."""
    relation = mp.E_RELATION if relation is None else relation
    return 'E_eff' if relation == mp.E_RELATION else f'E_eff__{relation}'


def wl_thickness(row, choice=None):
    """D_wl (m) under KWL_THICKNESS; D_WL_FALLBACK where the chosen column is missing."""
    choice = mp.KWL_THICKNESS if choice is None else choice
    if not isinstance(choice, str):
        return float(choice)
    col = {'sequence': 'wl_thickness', 'weakest_element': 'wl_weakest_thickness'}.get(choice)
    if col is None:
        raise ValueError(f"unknown KWL_THICKNESS {choice!r}")
    try:
        D = float(row.get(col, np.nan))
    except (TypeError, ValueError):
        D = np.nan
    return D if np.isfinite(D) and D > 0 else mp.D_WL_FALLBACK


def cell_energy(row, psi_deg, sigma_t_col=None, e_relation=None, delta=None,
                G_wl=None, use_residual=None, kwl_thickness=None, R=None):
    """tau_net, G_c, G_slab_path, N_max and ln_Pi = ln(R G_slab / G_c) for one cluster row.

    NaN where inputs are missing. Defaults come from the module constants.
    """
    sigma_t_col = mp.SIGMA_T_COLUMN if sigma_t_col is None else sigma_t_col
    delta = mp.ENERGY_DELTA if delta is None else delta
    G_wl = mp.G_WL if G_wl is None else G_wl
    use_residual = mp.ENERGY_USE_RESIDUAL if use_residual is None else use_residual

    def get(k):
        try:
            return float(row.get(k, np.nan))
        except (TypeError, ValueError):
            return np.nan

    rho, h = get('slab_density'), get('slab_thickness')
    e_relation = mp.ENERGY_E_RELATION if e_relation is None else e_relation
    E, sig = get(e_column(e_relation)), get(sigma_t_col)
    tau_p = get('wl_shear_strength')
    K_wl = G_wl / wl_thickness(row, kwl_thickness)
    psi = float(psi_deg)
    tr = float(residual_shear(rho, h, psi, mp.PHI_DEG)) if use_residual else 0.0
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
