# Crack propagation into a neighbour cluster: the gates

Reference for the **legacy** arrest mode (`release_geometry.ARREST_MODE =
'legacy'`, the default), as the code stands on 2026-10-25. The code is
`release_geometry.propagate_release`; the per-cluster decision is its inner
function `_qualifies`. Read the feature flags in `CLAUDE.md` first — the list
below assumes `USE_MELOCHE_ARREST = True` and the mode III options off.

## How the flood-fill works

1. The release grows from the trigger cluster ring by ring (breadth-first).
2. Each cluster's neighbours are its **8 nearest clusters by centroid**.
3. A neighbour is tested **once**, against the cluster it was reached from (its
   *parent*). If it passes, it joins the release and its own neighbours are
   tested next ring. If it fails, it is never retried from another parent.
4. **Direction is measured from the trigger, not from the parent.** The fall
   line is the trigger's aspect; a candidate within ±45° of downslope is
   *downslope*, within ±45° of upslope is *upslope*, everything else is
   *cross-slope*.
5. Growth stops when a ring admits nothing, or at the safety cap of 2000
   clusters (`[CAP-BOUND]` is printed — that polygon is not a physical result).

## Fixed per trigger, before the flood-fill

| Quantity | Value | Used by |
|---|---|---|
| `d_up` | `max(A_ca, 20 m) × size_factor`. A_ca is the trigger's `A_ca_brittle`, capped at 300 m; 50 m when missing | upslope distance cap |
| `d_down` | `max(distance to the stauchwall along the trigger's fall line, 20 m) × size_factor`. Stauchwall = first cell below 28° | downslope distance cap |
| `d_lat` | `max(A_ca × min(θ_along/θ_cross, 2.5) × size_factor, 15 m)` — Gaume cross-slope width | cross-slope distance cap |

`size_factor` (0.70–1.30 in the scenario set) scales all three caps.

## The gates, in order

The first gate that fails decides the arrest reason (shown in brackets).

| # | Gate | Arrests when | Up | Down | Cross |
|---|---|---|---|---|---|
| 1 | Start-zone KML (`MASK_MODE = 'gate'` only) | centroid pixel outside the KML [`outside_start_zone`] | ✓ | ✓ | ✓ |
| 2 | Distance from the trigger | dist > `d_up` / `d_down` / `d_lat` [`*_distance_cap`] | `d_up` | `d_down` | `d_lat` |
| 3 | Terrain slope | cluster mean slope < 28° [`stauchwall_slope`] | — exempt | ✓ | ✓ |
| 4 | Driving shear stress | τ_g < 350 Pa [`tau_g_below_floor`] | ✓ | ✓ | ✓ |
| 5 | Slab thickness | slab < 0.5 m [`thin_slab`] | ✓ | ✓ | ✓ |
| 6 | Elastic length | Λ < 0.1 m [`low_lambda`] | ✓ | ✓ | ✓ |
| 7 | Meloche directional arrest | see below [`meloche_arrest`] | ✓ | ✓ | ✓ |

Gates 4–6 look only at the **candidate's own** values. Gate 7 is the only one
that compares the candidate with its parent. Cross-slope uses the same Λ as
along-slope (`USE_MODE3_LAMBDA` is off).

### Gate 7: Meloche directional arrest (Eq. 20)

With parent *c* and candidate *n* a distance *d* apart:

```
θ_dir = max(0, (τ_p,n − τ_p,c) / d)          # only a rise in WL strength counts
if θ_dir > 0:
    A_ca = C · X · sqrt(σ_t / τ_g) · L_t,     X = τ_g / (θ_dir · Λ · sqrt(1 + δ))
    arrest if A_ca < d
```

C = 0.045 and δ = 1.0. τ_g, Λ, σ_t and L_t are the **candidate's** values.
τ_p is SNOWPACK's weak-layer shear strength (`wl_shear_strength`). The
equation is `arrest_indices.arrest_length`, the only implementation in the
package.

- Propagation into an **equal or weaker** weak layer never arrests here.
- Propagation into a **stronger** weak layer arrests only if the arrest length
  is shorter than one cluster step (a few metres). On Jan 18, trigger 1772,
  it fired for 6 of ~1200 clusters, so in practice the polygon edge almost always
  comes from gates 1–5.

If every gate passes, the candidate propagates [`propagated`].

## Not active (skipped while `USE_MELOCHE_ARREST = True`)

The old heuristics, still in the code after gate 7. Each one compares the
candidate with its parent:

- Λ change: arrest on a drop > 20% or a rise > 30% (× size_factor).
- Slab-thickness change: arrest on a drop > 20% or a rise > 30% (× size_factor).

## After the flood-fill

1. The admitted clusters are rasterised and the largest piece is kept.
2. Interior holes are filled.
3. The polygon is clipped to the start zone and to terrain ≥ 28°.
4. If the result is under 200 m², the legacy mode falls back to an oriented
   rectangle built from A_ca, the stauchwall and `d_lat`.

## What this means against Jan 18 (2026-10-25 analysis)

Across the observed **crown line and left (north) flank**, slab thickness,
τ_g, Λ and τ_p all rise going outward (slab ~1.2–1.4 m → ~2 m or more). Across
the **right (south) flank** they fall. Gates 4–6 only ever stop propagation
into *thinner or less loaded* snow, so they can reproduce the right flank but
not the crown line or the left flank. Gate 7 is the only one that responds to
the outward rise in τ_p, and it rarely fires. On the left flank, legacy is
stopped by the start-zone KML (gate 1), not by any physics gate.

## Other arrest modes

`ARREST_MODE = 'energy'` (`energy_path.py`) replaces gates 3–7 with a
path-energy balance; see `docs/results_energy_mode.md`. `'slope_only'` and
`'stability_connected'` (`baselines.py`) are comparison baselines without
mechanics.
