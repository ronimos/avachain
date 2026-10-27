"""Crack-arrest indices for dry-slab avalanches.

Sources: Meloche et al. (2025, JGR Earth Surface), Meloche et al. (ISSW 2026),
Meloche et al. (2026, TARP report). The energy-cap functions are a hypothesis
(see docs/arrest_indices.md), not published results. The full derivation and the
Jan 18 2026 validation live in ~/avalanche-release-areas/docs/release_area_methods.md.

Units: SI (Pa, m, kg/m3, J/m2); angles in degrees.
"""
import numpy as np

G_ACC = 9.81


# ---------------------------------------------------------------------------
# Material and loading helpers
# ---------------------------------------------------------------------------
def plane_stress_modulus(E, nu=0.3):
    """E' = E / (1 - nu^2)."""
    return E / (1.0 - nu**2)


def shear_modulus(E, nu=0.3):
    """G = E / (2 (1 + nu))."""
    return E / (2.0 * (1.0 + nu))


def weak_layer_stiffness(G_wl, D_wl):
    """K_wl = G_wl / D_wl (Pa/m)."""
    return G_wl / D_wl


def elastic_length(E, h, D_wl, G_wl, nu=0.3):
    """Upslope (mode II) Lambda = sqrt(E h D_wl / ((1 - nu^2) G_wl))."""
    return np.sqrt(E * h * D_wl / ((1.0 - nu**2) * G_wl))


def elastic_length_cross(E, h, D_wl, G_wl, nu=0.3):
    """Cross-slope (mode III) Lambda using slab shear modulus G instead of E'.
    Lambda_III / Lambda_II = sqrt((1 - nu) / 2). Confirmed against Johan
    Gaume's own antiplane-shear derivation (personal communication,
    2026-10-18; not yet published) -- see ~/avalanche-release-areas/docs/release_area_methods.md §3.
    Whether antiplane shear (vs. bending) is the right cross-slope regime is
    still open; this formula is correct *for* the antiplane case."""
    return np.sqrt(shear_modulus(E, nu) * h * D_wl / G_wl)


def gravitational_shear(rho, h, psi_deg):
    """tau_g = rho g h sin(psi)."""
    return rho * G_ACC * h * np.sin(np.radians(psi_deg))


def residual_shear(rho, h, psi_deg, phi_deg=27.0):
    """tau_r = rho g h cos(psi) tan(phi)."""
    return rho * G_ACC * h * np.cos(np.radians(psi_deg)) * np.tan(np.radians(phi_deg))


def wave_speeds(E, rho, nu=0.3):
    """(c_s, c_p) with c_p = sqrt(E'/rho)."""
    return np.sqrt(shear_modulus(E, nu) / rho), np.sqrt(plane_stress_modulus(E, nu) / rho)


# ---------------------------------------------------------------------------
# Strength-based indices (published scalings)
# ---------------------------------------------------------------------------
def tension_gradient(rho, psi_deg, phi_deg=27.0):
    """k_f = rho g sin(psi) (1 - tan(phi)/tan(psi)); 0 when psi <= phi."""
    psi, phi = np.radians(psi_deg), np.radians(phi_deg)
    k = rho * G_ACC * np.sin(psi) * (1.0 - np.tan(phi) / np.tan(psi))
    return np.maximum(k, 0.0)


# Crack-speed regimes. Upslope (mode II) cracks run supershear; mode III
# (antiplane) cracks cannot exceed the shear wave speed.
#
# On the 1.6: Meloche et al. (2025) write the supershear speed as 1.6 c_s but
# identify it with c_p -- "A horizontal dashed line is set at 1.6 c_s ~= sqrt(E'/rho),
# which is the supershear speed (Trottet et al., 2022)" -- and plot "until
# c_p ~= 1.6 c_s". So 1.6 is their rounding of c_p/c_s = sqrt(2/(1-nu)) = 1.690
# at nu = 0.3. Taking adot = c_p exactly gives k_x/k_f = 1/2, which is what the
# paper's own Fig. 3 caption states; the literal 1.6 gives 0.527.
# We keep the literal 1.6 so published figures reproduce; cp_over_cs() below
# gives the exact alternative. The choice moves mode3_length_ratio by ~5%
# (0.712 -> 0.675).
MODE2_SPEED_RATIO = 1.6   # adot / c_s upslope, Meloche et al. (2025) as written
MODE3_SPEED_RATIO = 1.0   # adot / c_s cross-slope cap, Broberg (1989)


def dynamic_factor(E, rho, nu=0.3, speed_ratio=MODE2_SPEED_RATIO):
    """k_x / k_f = c_p^2 / (c_p^2 + adot^2), adot = speed_ratio * c_s."""
    c_s, c_p = wave_speeds(E, rho, nu)
    adot = speed_ratio * c_s
    return c_p**2 / (c_p**2 + adot**2)


def dynamic_gradient(k_f, E, rho, nu=0.3, speed_ratio=MODE2_SPEED_RATIO):
    """k_x = k_f c_p^2 / (c_p^2 + adot^2)."""
    return k_f * dynamic_factor(E, rho, nu, speed_ratio)


def cp_over_cs(nu=0.3):
    """c_p / c_s = sqrt(2/(1-nu)); 1.690 at nu=0.3.

    The exact supershear bound that Meloche et al. round to 1.6. Pass as
    speed_ratio_along to evaluate() for the self-consistent k_x/k_f = 1/2.
    """
    return np.sqrt(2.0 / (1.0 - nu))


def mode3_length_ratio(E, rho, nu=0.3,
                       speed_ratio_along=MODE2_SPEED_RATIO,
                       speed_ratio_cross=MODE3_SPEED_RATIO):
    """L_dyn,III / L_dyn,II = k_x,II / k_x,III.

    A slower crack builds slab tension faster per unit advance, so the mode III
    speed cap shortens the distance to first slab fracture. Since
    c_s^2/c_p^2 = (1-nu)/2, E and rho cancel and this reduces to a function of
    nu and the two speed ratios alone: 0.712 at nu=0.3.
    """
    return (dynamic_factor(E, rho, nu, speed_ratio_along)
            / dynamic_factor(E, rho, nu, speed_ratio_cross))


def tensile_length(sigma_t, k):
    """sigma_t / k; inf if k <= 0."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(k > 0, sigma_t / k, np.inf)


def slab_tensile_stress_cantilever(rho, h, l):
    """sigma_max = 3 rho g l^2 / h (Pa): bending stress at the surface of an
    unsupported slab of thickness h over a crack of length l.

    SOURCE: Schweizer, Reuter, van Herwijnen, Gauthier & Jamieson (2014), "On
    how the tensile strength of the slab affects crack propagation
    propensity", ISSW 2014, Eq. 1: a horizontal cantilever under its own
    weight, evaluated at l = r_c. Flat: no slope term, no E, no nu. Their check:
    rho = 100, h = 0.3 m, r_c = 0.2 m gives ~400 Pa. Reuter & Schweizer (2018,
    GRL 45, 7019) replace it with FE stresses and a layered index whose exact
    form we could not read; this is the closed form, not that index.
    """
    return 3.0 * np.asarray(rho, dtype=float) * G_ACC * np.asarray(l, dtype=float) ** 2 \
        / np.asarray(h, dtype=float)


def slab_tensile_support(sigma_t, rho, h, r_c):
    """S_t = sigma_t / sigma_max(l = r_c). S_t < 1: the slab breaks in tension
    before the weak-layer crack reaches critical length, so it cannot carry
    propagation (Schweizer et al. 2014). The threshold 1 is stress = strength,
    not a fit."""
    return np.asarray(sigma_t, dtype=float) / slab_tensile_stress_cantilever(rho, h, r_c)


C_FIT = 0.045  # fitted from Meloche et al. 2025 Fig. 8 (two runs); see docs/arrest_indices.md


def arrest_length(tau_g, theta, Lam, sigma_t, delta=1.0, L_t=None, C=C_FIT):
    """JGR Eq. 20: A_ca / L_t = C * X * sqrt(sigma_t / tau_g),
    X = tau_g / (theta Lam sqrt(1 + delta)). C = 0.045 is a two-run fit (Fig. 8).

    The only implementation of Eq. 20 in the package — call this rather than
    re-deriving the scaling, so the C prefactor cannot be dropped.

    Returns (A_ca or A_ca/L_t, X)."""
    X = tau_g / (theta * Lam * np.sqrt(1.0 + delta))
    rel = C * X * np.sqrt(sigma_t / tau_g)
    return (rel * L_t if L_t is not None else rel), X


def release_size_index(a_t, a_c, a_sc, arrest_ratio=6.5):
    """ISSW 2026 preliminary RSI: 1 arrest, 2 sub-Rayleigh en-echelon, 3 supershear."""
    if a_t / a_sc >= 1.0:
        return 3
    return 1 if a_t / a_c < arrest_ratio else 2


def shear_gradient(x, tau_p):
    """theta (Pa/m): linear-fit slope of tau_p along an upslope transect."""
    return float(np.polyfit(np.asarray(x, float), np.asarray(tau_p, float), 1)[0])


# ---------------------------------------------------------------------------
# Slab material parameterizations by density
# ---------------------------------------------------------------------------
RHO_REF      = 300.0
E_RHO_SCALE  = 4.0e6
E_RHO_EXP    = 2.5
ST_RHO_SCALE = 5.0e3
ST_RHO_EXP   = 1.4
VH2016_SCALE = 0.93    # van Herwijnen et al. (2016) J. Glaciol. 62(236), Eq. 8
VH2016_EXP   = 2.8


def slab_modulus_project_fit(rho, scale=E_RHO_SCALE, exp=E_RHO_EXP,
                             rho_ref=RHO_REF):
    """E(rho) = scale (rho/rho_ref)^exp; 4 MPa at rho = 300, exp 2.5.

    SOURCE: this project, not a publication. A hand-fit chosen to pass through
    ~2/4/6 MPa at rho = 200/300/350, i.e. through the *range* van Herwijnen
    et al. (2016) report, not through their regression. It sits a factor
    1.6-2.2 below that regression over rho = 150-400, and gives 2.54 MPa at
    rho = 250 where Meloche et al. (2025) ran their calibration at E = 4 MPa.
    """
    return (np.asarray(rho, dtype=float) / rho_ref) ** exp * scale


def slab_modulus_vanherwijnen2016(rho, scale=VH2016_SCALE, exp=VH2016_EXP):
    """E(rho) = 0.93 rho^2.8 Pa.

    SOURCE: van Herwijnen, Gaume, Bair, Reuter, Birkeland & Schweizer (2016),
    J. Glaciol. 62(236), 997-1007, Eq. 8; fitted to PST field measurements over
    roughly rho = 100-350 kg m-3 (Spearman r = 0.69, NRMSE 13%). Gives 4.82 MPa
    at rho = 250, close to the 4 MPa Meloche et al. (2025) used in the runs that
    C_FIT was calibrated against.
    """
    return scale * np.asarray(rho, dtype=float) ** exp


E_RELATIONS = {'vanherwijnen2016': slab_modulus_vanherwijnen2016,
               'project_fit': slab_modulus_project_fit}
E_DEFAULT = 'vanherwijnen2016'


def slab_modulus(rho, relation=E_DEFAULT):
    """E(rho) from a named relation; see E_RELATIONS."""
    try:
        return E_RELATIONS[relation](rho)
    except KeyError:
        raise ValueError(f"unknown E relation {relation!r}; "
                         f"available: {sorted(E_RELATIONS)}") from None


def slab_tensile_strength(rho, scale=ST_RHO_SCALE, exp=ST_RHO_EXP, rho_ref=RHO_REF):
    """sigma_t,RG(rho) = scale (rho/rho_ref)^exp, for rounded grains.

    SOURCE: this project, not a publication. ~5 kPa at rho = 300, inside the
    2-10 kPa swept by Meloche et al. (2025); the exponent 1.4 is a hand-fit.
    Neither paper publishes a sigma_t(rho) regression.
    """
    return (np.asarray(rho, dtype=float) / rho_ref) ** exp * scale


JJ1990_SCALE = 79.7e3   # Pa; Jamieson & Johnston (1990) J. Glaciol. 36(122), Table III, group I
JJ1990_EXP   = 2.39
JJ1990_RHO_ICE = 917.0  # kg m-3, as in their Eq. 4


def slab_tensile_strength_jamieson1990(rho, scale=JJ1990_SCALE, exp=JJ1990_EXP,
                                       rho_ice=JJ1990_RHO_ICE):
    """sigma_t(rho) = 79.7 (rho/rho_ice)^2.39 kPa, in-situ, rounded grains.

    SOURCE: Jamieson & Johnston (1990), J. Glaciol. 36(122), 102-106, Table III
    and Conclusions: group I (new, partly settled or rounded grains), 43 layers,
    rho = 100-345 kg m-3. Their faceted fit (58.3, 2.65; 23 layers, 190-260) is
    not used: facets are handled by FACET_STRENGTH_FACTOR, their own factor-2 finding.
    """
    return scale * (np.asarray(rho, dtype=float) / rho_ice) ** exp


# 'sigrist2006' (ETH Diss. 16736, lab) is named in the path-energy brief but not
# implemented: the thesis was not obtainable on 2026-10-25 and the regression is
# not to be coded from memory.
ST_RELATIONS = {'project_fit': slab_tensile_strength,
                'jamieson1990': slab_tensile_strength_jamieson1990}


# ---------------------------------------------------------------------------
# Layered-slab aggregation for Eq. 20
# ---------------------------------------------------------------------------
# a: faceted snow is about half as strong in tension as rounded snow at equal
# density (Jamieson & Johnston 1990). Applied continuously through f so that
# neighbouring grid cells cannot step across a grain-type boundary.
FACET_STRENGTH_FACTOR = 0.5


def facetedness(sphericity, dendricity, sp_ref=None):
    """f in [0, 1]: 1 - sphericity for non-dendritic layers, 0 otherwise.

    sp_ref anchors f to a reference sphericity, f = (sp_ref - sp)/sp_ref, so
    that snow at sp_ref keeps the unmodified sigma_t,RG. sp_ref=None is the
    unanchored form, under which rounded grains (sphericity ~0.86 in our
    profiles) still carry f ~0.14 and lose ~7% of sigma_t.
    """
    sp = np.asarray(sphericity, dtype=float)
    dd = np.asarray(dendricity, dtype=float)
    f = (1.0 - sp) if sp_ref is None else (sp_ref - sp) / sp_ref
    f = np.where(dd == 0.0, f, 0.0)
    return np.clip(np.where(np.isfinite(f), f, 0.0), 0.0, 1.0)


# Grain classes counted as faceted (SNOWPACK codes: 4 FC, 5 DH, 9 FCxr).
# Jamieson & Johnston (1990), Microstructure: group II is "layers of faceted
# grains and of layers of grains which showed both faceting and rounding"
# (8 of their 23 group II layers), group I "new snow, partly settled grains or
# rounded grains", both with no evidence of melting. So FCxr is faceted, and
# so is a group I primary (PP/DF/RG) with a faceted secondary form, e.g. 341.
FACETED_GRAIN_CLASSES = (4, 5, 9)
MIXED_PRIMARY_CLASSES = (1, 2, 3)


def facetedness_grain_class(grain_type, classes=FACETED_GRAIN_CLASSES,
                            mixed_primary=MIXED_PRIMARY_CLASSES,
                            secondary=True):
    """f in {0, 1}: 1 where the layer is faceted by Jamieson & Johnston's
    (1990) grouping, from the SNOWPACK code 100 F1 + 10 F2 + marker.

    Faceted: primary F1 in `classes`, or (secondary=True) a group I primary
    F1 in `mixed_primary` whose secondary F2 is in `classes` - J&J's grains
    "which showed both faceting and rounding". Melt forms (MF/IF) and SH
    stay 0 whatever F2 is: J&J tested no melted layers. secondary=False is
    the primary-only rule (2026-10-27, 163ea0f). 1 - sphericity under-counts
    both FCxr and the mixed forms. Missing or non-positive codes get f = 0.
    """
    gt = np.asarray(grain_type, dtype=float)
    ok = np.isfinite(gt) & (gt > 0)
    code = np.round(np.where(ok, gt, 0.0)).astype(int)
    f1, f2 = code // 100, (code // 10) % 10
    fac = np.isin(f1, classes)
    if secondary:
        fac |= np.isin(f1, mixed_primary) & np.isin(f2, classes)
    return np.where(ok & fac, 1.0, 0.0)


# How f is derived: 'grain_class' (primary or mixed, see
# facetedness_grain_class), 'grain_class_primary' (primary class only,
# 163ea0f) or 'sphericity' (1 - sphericity, the form before 2026-10-27).
FACET_SOURCES = ('grain_class', 'grain_class_primary', 'sphericity')


def layer_tensile_strength(rho, f, a=FACET_STRENGTH_FACTOR, **kw):
    """sigma_t,i = sigma_t,RG(rho_i) * (1 - a f_i)."""
    return slab_tensile_strength(rho, **kw) * (1.0 - a * np.asarray(f, dtype=float))


def _weights(dz):
    dz = np.asarray(dz, dtype=float)
    tot = np.sum(dz)
    if not np.isfinite(tot) or tot <= 0:
        return None, np.nan
    return dz, float(tot)


def effective_modulus(E_i, dz_i):
    """Iso-strain slope-parallel stiffness: E_eff = sum(E_i h_i) / h."""
    dz, tot = _weights(dz_i)
    if dz is None:
        return np.nan
    return float(np.sum(np.asarray(E_i, dtype=float) * dz) / tot)


def tensile_strength_mean(sigma_t_i, dz_i):
    """Load-redistribution bound: sum(sigma_t_i h_i) / h."""
    dz, tot = _weights(dz_i)
    if dz is None:
        return np.nan
    return float(np.sum(np.asarray(sigma_t_i, dtype=float) * dz) / tot)


def tensile_strength_weakest_link(sigma_t_i, E_i, dz_i):
    """Weakest-link bound: E_eff * min_i(sigma_t_i / E_i), and the argmin index.

    The slab is taken to fail when the lowest-failure-strain layer reaches its
    own strength, with every layer at the common iso-strain value. This is
    never above tensile_strength_mean: writing the mean as
    sum((sigma_t_i/E_i) E_i h_i)/h >= min(sigma_t_i/E_i) * E_eff, with equality
    only when sigma_t_i/E_i is the same in every layer.
    """
    st = np.asarray(sigma_t_i, dtype=float)
    E = np.asarray(E_i, dtype=float)
    dz, _ = _weights(dz_i)
    if dz is None:
        return np.nan, -1
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.where(E > 0, st / E, np.inf)
    if not np.any(np.isfinite(eps)):
        return np.nan, -1
    idx = int(np.nanargmin(np.where(np.isfinite(eps), eps, np.inf)))
    return float(effective_modulus(E, dz) * eps[idx]), idx


# --- Edge-crack (ligament) estimate: HYPOTHESIS, not a published criterion ---
# Treats the controlling layer's own thickness as a crack of length a in a slab
# of thickness h loaded in slope-parallel tension, and returns the nominal slab
# stress at which K_I reaches K_Ic. A faceted layer is not a traction-free
# crack, so this is an upper bound on its weakening effect at best.
#
# K_Ic(rho) now has two sourced options, K_IC_RELATIONS below. Neither is a
# slab-tension measurement: Schweizer et al. (2004) is a cantilever/beam fit and
# Kirchner et al. (2000) is apparent toughness from small beams, so both carry a
# specimen-size effect and kirchner2000 is a LOWER BOUND. The rest of the
# accessible literature (van Herwijnen et al. 2016; Sigrist & Schweizer 2007;
# Sigrist 2006) measures WEAK-LAYER energy release rates, not slab K_Ic, and
# Meloche et al. (2025, 2026) give neither. The bound has never been validated
# against an observed crown, and is not fed into the BFS.
#
# The on/off switch is snowpack_analysis.USE_LIGAMENT_BOUND (release_areas keeps
# it in config.py) -- do not shadow it here.

RHO_ICE = 917.0
DMAX_FACTOR = 1.0   # d_max = DMAX_FACTOR * grain size (both in m)

# Schweizer, Michot & Kirchner (2004) Ann. Glaciol. 38, Eq. 8. A4 is the
# paper's 0.35 kPa m.
#
# The exponent default changed to the paper's printed 1.9 on 2026-10-09. It had
# been 2.0, this project's rounding of the abstract's "an exponent of about 2",
# which runs 11% low at rho = 300 and 17% low at rho = 150. Nothing was
# measured to justify 2.0 and nothing argues for it over the printed
# regression, so the paper's figure is now the default and the rounding is kept
# only as a named constant for reproducing older output. Note this cannot move
# any release polygon: k_ic() reaches only the ligament columns, which the BFS
# never reads.
SCH2004_A4        = 350.0   # Pa m
SCH2004_EXP_PAPER = 1.9     # as printed in Eq. 8 (r = 0.98)
SCH2004_EXP_ROUND = 2.0     # the abstract's "about 2"; default before 2026-10-09
SCH2004_EXP       = SCH2004_EXP_PAPER   # used by default


def k_ic_kirchner2000(rho, d_max=None):
    """K_Ic = 7.84 (rho/917)^2.3 kPa m^0.5 -> Pa m^0.5. d_max unused.

    Kirchner et al. (2000), Phil. Mag. A 80(5). Valid 100-540 kg/m3.

    TREAT AS A LOWER BOUND ON K_Ic. These are *apparent* toughnesses from small
    cantilever beams, so they carry the small-specimen size effect: the
    measured value rises with specimen size, and a slab-scale crack is far
    larger than the beams. It gives 600 Pa m^0.5 at rho = 300, roughly half
    Schweizer et al. (2004) Eq. 8 at d_max = 1 mm and below the bottom of the
    1-4 kPa m^0.5 range Borstad & McClung (2013) report.
    """
    return 7.84e3 * (np.asarray(rho, dtype=float) / RHO_ICE) ** 2.3


def k_ic_schweizer2004(rho, d_max=None, A4=SCH2004_A4, exp=SCH2004_EXP):
    """K_Ic = A4 (rho/917)^exp / sqrt(d_max), A4 = 350 Pa m, d_max in m.

    Schweizer, Michot & Kirchner (2004), Ann. Glaciol. 38, 1-8, Eq. 8, which
    the paper gives as superseding its own Eqs. 5 and 6 (two (rho/rho_ice)^2.0
    fits with prefactors 13.0 and 21.6 kPa m^0.5 for different snow types) by
    folding in the d_max^(-1/2) dependence, reported at r = 0.98.

    DENSITY EXPONENT: the paper's printed Eq. 8 carries **1.9**; its abstract
    says "an exponent of about 2". Since 2026-10-09 we default to the printed
    exp = 1.9. The earlier default of 2.0 was this project's rounding and runs
    11% low at rho = 300 (1.18 vs 1.32 kPa m^0.5), diverging further below it;
    only the prefactor-and-exponent pair as printed reproduces the paper's fit.
    Pass exp=SCH2004_EXP_ROUND to reproduce output generated before the change.

    Valid 80-300 kg/m3: series C-F, chosen to resolve the density dependence,
    span 80-250, and the abstract quotes 100-300 across all series A-F.

    d_max is a LENGTH IN METRES. Note our zarr `grain_size` is already in m
    (median 5.9e-4), unlike raw SNOWPACK .pro output which is in mm.

    rho = 300, d_max = 1 mm -> 1.18 kPa m^0.5 at the default exp = 2.0.
    """
    rho = np.asarray(rho, dtype=float)
    if d_max is None:
        return np.full(rho.shape, np.nan)
    d = np.asarray(d_max, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, A4 * (rho / RHO_ICE) ** exp / np.sqrt(d), np.nan)


# Selectable K_Ic(rho) relations for intact slab snow in mode I, with the
# density range each was fitted over. borstad2013 stays gated: it is bronze OA
# at Wiley only, which refuses automated access, and OpenAlex reports no
# repository copy, so its regression has not been sourced.
K_IC_RELATIONS = {
    'kirchner2000':  {'func': k_ic_kirchner2000,  'rho': (100.0, 540.0)},
    'schweizer2004': {'func': k_ic_schweizer2004, 'rho': (80.0, 300.0)},
}
K_IC_PENDING = ('borstad2013',)
K_IC_DEFAULT = 'schweizer2004'


def _k_ic_spec(relation):
    if relation in K_IC_PENDING:
        raise NotImplementedError(
            f"K_Ic relation {relation!r} is not implemented: its published "
            "equation has not been sourced and quoted back for checking yet.")
    try:
        return K_IC_RELATIONS[relation]
    except KeyError:
        raise ValueError(
            f"unknown K_Ic relation {relation!r}; "
            f"available: {sorted(K_IC_RELATIONS)}") from None


def k_ic_valid_range(relation=K_IC_DEFAULT):
    """(rho_min, rho_max) the relation was fitted over."""
    return _k_ic_spec(relation)['rho']


def k_ic_n_clamped(rho, relation=K_IC_DEFAULT):
    """How many finite rho values fall outside the relation's valid range."""
    rho = np.asarray(rho, dtype=float)
    lo, hi = k_ic_valid_range(relation)
    return int(np.sum(np.isfinite(rho) & ((rho < lo) | (rho > hi))))


def k_ic(rho, relation=K_IC_DEFAULT, d_max=None, clamp=True):
    """K_Ic (Pa m^0.5) from a named density relation.

    clamp=True (default) clamps rho into the relation's fitted range *for this
    evaluation only*, rather than returning NaN, so that a single out-of-range
    layer cannot poison a thickness-weighted ligament. Density is not modified
    anywhere else. clamp=False restores the NaN-outside-range behaviour.
    Use k_ic_n_clamped() to report how many layers were clamped.
    """
    spec = _k_ic_spec(relation)
    rho = np.asarray(rho, dtype=float)
    lo, hi = spec['rho']
    if clamp:
        rho_eval = np.clip(rho, lo, hi)
        return spec['func'](rho_eval, d_max)
    K = spec['func'](rho, d_max)
    return np.where((rho >= lo) & (rho <= hi), K, np.nan)


def characteristic_length(K_Ic, sigma_t):
    """l_ch = (K_Ic / sigma_t)^2 (m)."""
    K_Ic = np.asarray(K_Ic, dtype=float)
    sigma_t = np.asarray(sigma_t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma_t > 0, (K_Ic / sigma_t) ** 2, np.nan)


def sent_geometry_factor(s):
    """F(a/h) for a single-edge-notch plate in tension (Tada et al.), s <= 0.6."""
    s = np.asarray(s, dtype=float)
    F = (1.122 - 0.231 * s + 10.550 * s**2
         - 21.710 * s**3 + 30.382 * s**4)
    return np.where((s >= 0.0) & (s <= 0.6), F, np.nan)


def embedded_geometry_factor(s):
    """F for a centre crack of length 2a in width 2b, s = a/b (Feddersen secant)."""
    s = np.asarray(s, dtype=float)
    F = np.sqrt(1.0 / np.cos(np.pi * s / 2.0))
    return np.where((s >= 0.0) & (s < 0.95), F, np.nan)


def _geometry_factor(a, h, embedded):
    return float(embedded_geometry_factor((a / 2.0) / (h / 2.0)) if embedded
                 else sent_geometry_factor(a / h))


def intrinsic_crack_length(K_Ic, sigma_t_lig, h, a=None, embedded=False,
                           self_consistent=False, tol=1e-12, max_iter=50):
    """El Haddad a0 = (1/pi) (K_Ic / (F sigma_t_lig))^2.

    Default: F is evaluated at the actual crack length, F(a/h), the same factor
    sigma_c uses. That makes the a -> 0 limit recover sigma_t_lig exactly and
    needs no iteration.

    self_consistent=True instead evaluates F at a0/h and solves by fixed-point
    iteration. Kept as an option; it makes a0 a property of the material and
    geometry alone, but then sigma_c(a -> 0) only tends to
    sigma_t_lig * F(a0/h)/F(0).
    """
    if not (np.isfinite(K_Ic) and np.isfinite(sigma_t_lig) and np.isfinite(h)) \
            or K_Ic <= 0 or sigma_t_lig <= 0 or h <= 0:
        return np.nan

    if not self_consistent:
        if a is None:
            raise ValueError("a is required unless self_consistent=True")
        F = _geometry_factor(float(a), h, embedded)
        if not np.isfinite(F) or F <= 0:
            return np.nan
        a0 = (K_Ic / (F * sigma_t_lig)) ** 2 / np.pi
        return float(a0) if np.isfinite(a0) and a0 < h else np.nan

    F = _geometry_factor(0.0, h, embedded)
    a0 = np.nan
    for _ in range(max_iter):
        if not np.isfinite(F) or F <= 0:
            return np.nan
        a0_new = (K_Ic / (F * sigma_t_lig)) ** 2 / np.pi
        if not np.isfinite(a0_new) or a0_new >= h:
            return np.nan
        if np.isfinite(a0) and abs(a0_new - a0) <= tol * max(a0_new, 1e-12):
            return float(a0_new)
        a0 = a0_new
        F = _geometry_factor(a0, h, embedded)
    return float(a0) if np.isfinite(a0) else np.nan


def edge_crack_strength(K_Ic, a, h, embedded=False, sigma_t_lig=None,
                        model='elhaddad', self_consistent_a0=False):
    """Nominal slab stress at which a crack of length a in thickness h runs.

    LEFM:        sigma_c = K_Ic / (F(a/h) sqrt(pi a))
    El Haddad:   sigma_c = K_Ic / (F(a/h) sqrt(pi (a + a0)))

    The El Haddad short-crack correction keeps sigma_c finite as a -> 0, where
    it tends to sigma_t_lig instead of diverging, and collapses onto LEFM for
    a >> a0. It needs the ligament tensile strength; without sigma_t_lig, or
    with model='lefm', the plain LEFM form is returned.
    """
    a = float(a)
    h = float(h)
    if not (np.isfinite(a) and np.isfinite(h)) or a <= 0 or h <= 0 or a >= h:
        return np.nan
    if embedded:
        F = float(embedded_geometry_factor((a / 2.0) / (h / 2.0)))
        a_eff = a / 2.0
    else:
        F = float(sent_geometry_factor(a / h))
        a_eff = a
    if not np.isfinite(F) or F <= 0:
        return np.nan

    a0 = 0.0
    if model == 'elhaddad' and sigma_t_lig is not None:
        a0 = intrinsic_crack_length(K_Ic, sigma_t_lig, h, a=a,
                                    embedded=embedded,
                                    self_consistent=self_consistent_a0)
        if not np.isfinite(a0):
            return np.nan
    elif model not in ('elhaddad', 'lefm'):
        raise ValueError(f"unknown model {model!r}; use 'elhaddad' or 'lefm'")
    return float(K_Ic / (np.sqrt(np.pi * (a_eff + a0)) * F))


def ligament_bound(rho_intact, dz_intact, st_intact, a, h, embedded,
                   relation=K_IC_DEFAULT, d_max_intact=None,
                   dmax_factor=DMAX_FACTOR, model='elhaddad',
                   self_consistent_a0=False):
    """Ligament edge-crack bound from the intact layers. Returns a dict.

    K_Ic and sigma_t are thickness-weighted over the intact layers (the slab
    minus the controlling layer, which is treated as the crack of length a).
    d_max_intact is a per-layer length in m; it is scaled by dmax_factor.
    """
    out = {'sigma_t_ligament': np.nan, 'K_Ic_ligament': np.nan,
           'sigma_t_lig_intact': np.nan, 'a0_ligament': np.nan,
           'l_ch_ligament': np.nan, 'n_rho_clamped': 0}
    if len(dz_intact) == 0:
        return out
    d_max = None if d_max_intact is None else \
        dmax_factor * np.asarray(d_max_intact, dtype=float)
    out['n_rho_clamped'] = k_ic_n_clamped(rho_intact, relation)
    K_lig = tensile_strength_mean(
        k_ic(rho_intact, relation, d_max=d_max, clamp=True), dz_intact)
    st_lig = tensile_strength_mean(st_intact, dz_intact)
    out['K_Ic_ligament'] = K_lig
    out['sigma_t_lig_intact'] = st_lig
    if not (np.isfinite(K_lig) and np.isfinite(st_lig)):
        return out
    out['l_ch_ligament'] = float(characteristic_length(K_lig, st_lig))
    out['a0_ligament'] = intrinsic_crack_length(
        K_lig, st_lig, h, a=a, embedded=embedded,
        self_consistent=self_consistent_a0)
    out['sigma_t_ligament'] = edge_crack_strength(
        K_lig, a, h, embedded=embedded, sigma_t_lig=st_lig, model=model,
        self_consistent_a0=self_consistent_a0)
    return out


def aggregate_slab(rho_i, dz_i, sphericity_i, dendricity_i,
                   a=FACET_STRENGTH_FACTOR, sp_ref=None,
                   depth_i=None, K_Ic=None, h=None,
                   k_ic_relation=None, ligament_model='elhaddad',
                   grain_size_i=None, dmax_factor=DMAX_FACTOR,
                   e_relation=E_DEFAULT, e_relations_extra=(),
                   k_ic_variants=(), self_consistent_a0=False,
                   st_relations_extra=(), grain_type_i=None,
                   facet_source='sphericity'):
    """Layered-slab aggregation for Eq. 20. Returns a dict of scalars.

    E_eff, sigma_t_mean, sigma_t_wl and the index/depth/thickness of the layer
    that controls the weakest-link bound.

    Ligament bound: pass k_ic_relation (named K_Ic(rho) relation) or K_Ic (a
    constant, Pa m^0.5). The controlling layer is treated as the crack; K_Ic
    and sigma_t for the ligament are thickness-weighted over the *intact*
    layers, i.e. the slab with the controlling layer removed.

    e_relation picks the E(rho) relation; e_relations_extra is a list of extra
    names emitted as E_eff__<name>. k_ic_variants is a list of
    (label, relation, dmax_factor) emitted as <key>__<label>, in addition to
    the unsuffixed bound from k_ic_relation. st_relations_extra names extra
    sigma_t(rho) relations (ST_RELATIONS), emitted as sigma_t_mean__<name> and
    sigma_t_wl__<name> with the same facet knockdown.

    facet_source picks f: 'sphericity' (facetedness, from sphericity_i and
    dendricity_i), or 'grain_class' / 'grain_class_primary'
    (facetedness_grain_class with / without mixed forms, from grain_type_i,
    which is then required).
    """
    if facet_source not in FACET_SOURCES:
        raise ValueError(f'facet_source {facet_source!r} not in {FACET_SOURCES}')
    if facet_source.startswith('grain_class') and grain_type_i is None:
        raise ValueError(f"facet_source={facet_source!r} needs grain_type_i")
    rho = np.asarray(rho_i, dtype=float)
    dz = np.asarray(dz_i, dtype=float)
    ok = np.isfinite(rho) & np.isfinite(dz) & (dz > 0) & (rho > 0)
    out = {k: np.nan for k in ('E_eff', 'sigma_t_mean', 'sigma_t_wl',
                               'f_mean_weighted', 'wl_ctrl_index',
                               'wl_ctrl_depth', 'wl_ctrl_thickness',
                               'sigma_t_ligament', 'K_Ic_ligament',
                               'sigma_t_lig_intact', 'a0_ligament',
                               'l_ch_ligament')}
    out['n_rho_clamped'] = 0
    out['n_slab_layers'] = int(ok.sum())
    if ok.sum() == 0:
        return out

    rho, dz = rho[ok], dz[ok]
    gsz = (None if grain_size_i is None
           else np.asarray(grain_size_i, dtype=float)[ok])
    if facet_source.startswith('grain_class'):
        f = facetedness_grain_class(np.asarray(grain_type_i, dtype=float)[ok],
                                    secondary=facet_source == 'grain_class')
    else:
        f = facetedness(np.asarray(sphericity_i, dtype=float)[ok],
                        np.asarray(dendricity_i, dtype=float)[ok], sp_ref)
    E = slab_modulus(rho, e_relation)
    st = layer_tensile_strength(rho, f, a)

    out['E_eff'] = effective_modulus(E, dz)
    for name in e_relations_extra:
        out[f'E_eff__{name}'] = effective_modulus(slab_modulus(rho, name), dz)
    out['sigma_t_mean'] = tensile_strength_mean(st, dz)
    out['f_mean_weighted'] = tensile_strength_mean(f, dz)
    st_wl, idx = tensile_strength_weakest_link(st, E, dz)
    out['sigma_t_wl'] = st_wl
    for name in st_relations_extra:
        st_x = ST_RELATIONS[name](rho) * (1.0 - a * f)
        out[f'sigma_t_mean__{name}'] = tensile_strength_mean(st_x, dz)
        out[f'sigma_t_wl__{name}'] = tensile_strength_weakest_link(st_x, E, dz)[0]
    if idx < 0:
        return out

    out['wl_ctrl_index'] = int(np.where(ok)[0][idx])
    out['wl_ctrl_thickness'] = float(dz[idx])
    if depth_i is not None:
        out['wl_ctrl_depth'] = float(np.asarray(depth_i, dtype=float)[ok][idx])

    if K_Ic is None and k_ic_relation is None and not k_ic_variants:
        return out

    h_slab = float(np.sum(dz)) if h is None else float(h)
    d = out['wl_ctrl_depth']
    # Surface-connected if the controlling layer touches either slab face.
    edge = (not np.isfinite(d)) or (d - dz[idx] <= 0.0) or (d >= h_slab - 1e-9)

    # Ligament = the slab minus the controlling layer, so the crack-tip
    # material is the surrounding snow rather than the faceted layer being
    # treated as the crack.
    intact = np.ones(len(dz), dtype=bool)
    intact[idx] = False
    if not intact.any():
        return out
    a_crack = float(dz[idx])
    gsz_in = None if gsz is None else gsz[intact]

    if K_Ic is not None and k_ic_relation is None:
        # Constant K_Ic override: no density relation, so nothing to clamp.
        st_lig = tensile_strength_mean(st[intact], dz[intact])
        out['K_Ic_ligament'] = float(K_Ic)
        out['sigma_t_lig_intact'] = st_lig
        if np.isfinite(st_lig):
            out['l_ch_ligament'] = float(characteristic_length(K_Ic, st_lig))
            out['a0_ligament'] = intrinsic_crack_length(
                K_Ic, st_lig, h_slab, a=a_crack, embedded=not edge,
                self_consistent=self_consistent_a0)
            out['sigma_t_ligament'] = edge_crack_strength(
                K_Ic, a_crack, h_slab, embedded=not edge, sigma_t_lig=st_lig,
                model=ligament_model, self_consistent_a0=self_consistent_a0)
    elif k_ic_relation is not None:
        out.update(ligament_bound(
            rho[intact], dz[intact], st[intact], a_crack, h_slab, not edge,
            relation=k_ic_relation, d_max_intact=gsz_in,
            dmax_factor=dmax_factor, model=ligament_model,
            self_consistent_a0=self_consistent_a0))

    for label, relation, dmf in k_ic_variants:
        v = ligament_bound(
            rho[intact], dz[intact], st[intact], a_crack, h_slab, not edge,
            relation=relation, d_max_intact=gsz_in, dmax_factor=dmf,
            model=ligament_model, self_consistent_a0=self_consistent_a0)
        for k, val in v.items():
            out[f'{k}__{label}'] = val
    return out


# ---------------------------------------------------------------------------
# Energy-cap framework (hypothesis)
# ---------------------------------------------------------------------------
R_FIT = 0.48  # fitted from Meloche et al. 2025 Fig. 8 (two runs); see docs/arrest_indices.md


def slab_energy_cap(sigma_t, h, E, nu=0.3):
    """Energy a slab segment loaded to sigma_t can deliver per unit crack advance:
    G_slab = sigma_t^2 h / (2 E')  (J/m2)."""
    return sigma_t**2 * h / (2.0 * plane_stress_modulus(E, nu))


def slab_energy_cap_cross(tau_flank, h, E, nu=0.3):
    """Cross-slope analogue: G_slab,III = tau_flank^2 h / (2 G).
    tau_flank is a slab flank (shear) strength; no published parameterization."""
    return tau_flank**2 * h / (2.0 * shear_modulus(E, nu))


def weak_layer_fracture_energy(tau_p, K_wl, delta=1.0, tau_r=0.0):
    """G_c = (tau_p - tau_r)^2 (1 + delta) / (2 K_wl)  (J/m2)."""
    return (np.asarray(tau_p, float) - tau_r)**2 * (1.0 + delta) / (2.0 * K_wl)


def energy_ratio(tau_p, K_wl, sigma_t, h, E, delta=1.0, tau_r=0.0, nu=0.3):
    """R = G_c(tau_p) / G_slab. Hypothesis: arrest when R >= R_crit."""
    return weak_layer_fracture_energy(tau_p, K_wl, delta, tau_r) / slab_energy_cap(sigma_t, h, E, nu)


def critical_strength(sigma_t, h, E, K_wl, delta=1.0, tau_r=0.0, R=R_FIT, nu=0.3):
    """tau_p* where G_c(tau_p*) = R * G_slab.
    R must be a number; pass R='dynamic' to evaluate() to resolve it via k_x/k_f first."""
    G_cap = R * slab_energy_cap(sigma_t, h, E, nu)
    return tau_r + np.sqrt(2.0 * K_wl * G_cap / (1.0 + delta))


def arrest_length_energy(tau_p0, tau_p_star, theta):
    """A_ca = (tau_p* - tau_p0) / theta for a linear strength ramp.
    0 if tau_p0 >= tau_p*; inf if theta <= 0."""
    if theta <= 0:
        return np.inf
    return max(tau_p_star - tau_p0, 0.0) / theta


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------
def evaluate(rho, h, psi_deg, E, sigma_t, D_wl, G_wl, theta, a_c=None, a_sc=None,
             nu=0.3, phi_deg=27.0, delta=1.0, C=C_FIT, tau_p0=None, R=R_FIT,
             use_residual=False, speed_ratio_along=MODE2_SPEED_RATIO,
             speed_ratio_cross=MODE3_SPEED_RATIO):
    """All indices for one slab / weak-layer configuration.
    R: critical energy ratio (number) or 'dynamic' to use k_x/k_f.
    speed_ratio_along / speed_ratio_cross: crack speed as a multiple of c_s for
    mode II (supershear) and mode III (capped at c_s) propagation."""
    Lam = elastic_length(E, h, D_wl, G_wl, nu)
    K_wl = weak_layer_stiffness(G_wl, D_wl)
    tau_g = gravitational_shear(rho, h, psi_deg)
    tau_r = residual_shear(rho, h, psi_deg, phi_deg)
    k_f = float(tension_gradient(rho, psi_deg, phi_deg))
    k_x = float(dynamic_gradient(k_f, E, rho, nu, speed_ratio_along))
    k_x_cross = float(dynamic_gradient(k_f, E, rho, nu, speed_ratio_cross))
    L_t = float(tensile_length(sigma_t, k_f))
    L_dyn = float(tensile_length(sigma_t, k_x))
    L_dyn_cross = float(tensile_length(sigma_t, k_x_cross))

    out = dict(Lambda=Lam, Lambda_cross=elastic_length_cross(E, h, D_wl, G_wl, nu),
               K_wl=K_wl, tau_g=tau_g, tau_r=tau_r, k_f=k_f, k_x=k_x,
               k_x_cross=k_x_cross, L_t=L_t, L_dyn=L_dyn,
               L_dyn_cross=L_dyn_cross,
               mode3_length_ratio=float(mode3_length_ratio(
                   E, rho, nu, speed_ratio_along, speed_ratio_cross)),
               sustained=psi_deg > phi_deg)

    if np.isfinite(L_t) and theta > 0:
        out["A_ca"], out["X"] = arrest_length(tau_g, theta, Lam, sigma_t, delta, L_t, C)

    R_val = dynamic_factor(E, rho, nu) if R == "dynamic" else R
    tr = tau_r if use_residual else 0.0
    out["G_slab"] = slab_energy_cap(sigma_t, h, E, nu)
    out["R_used"] = R_val
    out["tau_p_star"] = critical_strength(sigma_t, h, E, K_wl, delta, tr, R_val, nu)
    if tau_p0 is not None:
        out["G_c0"] = float(weak_layer_fracture_energy(tau_p0, K_wl, delta, tr))
        out["R0"] = out["G_c0"] / out["G_slab"]
        out["A_ca_energy"] = arrest_length_energy(tau_p0, out["tau_p_star"], theta)

    if a_c and a_sc and np.isfinite(L_dyn):
        out["RSI"] = release_size_index(L_dyn, a_c, a_sc)
    return out
