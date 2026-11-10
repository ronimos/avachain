# Release-model calibration sweep, 2026-11-08

Objective (Ron, 11-08): improve the Jan 18 expected IoU while the expected
release area on the 10 no-event survey days does not grow. Sweep:
`scripts/calibration_sweep.py`, results in
`outputs/little_prof/calibration/20261108_{c_fit,size_scale}/` (198 jobs, all ok).
Baseline = c_fit 0.045, size_scale 1.0.

| c_fit | Jan 18 E[IoU] | Jan 18 E[area] m² | no-event median m² | no-event max m² | Jan 18 AUC |
|---|---|---|---|---|---|
| 0.0225 | 0.155 | 1066 | 1789 | 2940 | 0.76 |
| 0.03 | 0.167 | 1403 | 2237 | 3521 | 0.74 |
| 0.035 | 0.181 | 1763 | 2554 | 3843 | 0.72 |
| 0.04 | 0.192 | 2111 | 2840 | 4294 | 0.69 |
| **0.045** | **0.200** | **2417** | **3112** | **4676** | **0.68** |
| 0.05 | 0.207 | 2703 | 3410 | 5050 | 0.66 |
| 0.055 | 0.214 | 2966 | 3684 | 5368 | 0.65 |
| 0.0675 | 0.230 | 3491 | 4218 | 6081 | 0.64 |
| 0.09 | 0.259 | 4129 | 4792 | 7177 | 0.64 |

| size_scale | Jan 18 E[IoU] | Jan 18 E[area] m² | no-event median m² | no-event max m² | Jan 18 AUC |
|---|---|---|---|---|---|
| 0.6 | 0.144 | 1132 | 1637 | 2973 | 0.75 |
| 0.8 | 0.176 | 1791 | 2283 | 3792 | 0.71 |
| **1.0** | **0.200** | **2417** | **3112** | **4676** | **0.68** |
| 1.25 | 0.224 | 3103 | 4177 | 5730 | 0.66 |
| 1.6 | 0.260 | 3891 | 5271 | 7177 | 0.67 |

## Findings

1. **No value of either knob meets the objective.** Both responses are smooth
   and monotonic with no plateau: every IoU gain comes with larger no-event
   areas, on every one of the 10 days (no day moves against the trend).
2. **c_fit is the more selective knob.** Jan 18 area / no-event median area
   rises from 0.78 to 0.86 between c_fit 0.045 and 0.09, whereas size_scale
   holds it flat or lowers it (0.74 at 1.25 and 1.6). Arrest strictness
   reads the snowpack; the BFS cap does not.
3. **The model under-sizes Jan 18 and over-sizes the no-event days.** At
   baseline the Jan 18 expected area (2417 m²) is 0.42 of the 5810 m² polygon,
   yet smaller than the median no-event day (3112 m²). A single size knob
   cannot fix an ordering problem. The discrimination has to come from what
   separates the days (trigger gates, which decide the 10/10 false alarms),
   not from how far a release grows.
4. Pixel AUC on Jan 18 falls as either knob grows: larger polygons add
   area outside the crown faster than inside it.

## Recommendation

Do not write a `[calibration]` block now; c_fit 0.045 stays, and is recorded
as checked against Jan 18 + 10 no-event days. If a trade is acceptable,
c_fit (not size_scale) is the knob to use, e.g. 0.055: IoU +0.014 for
+18 % no-event median area.
