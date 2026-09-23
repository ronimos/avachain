# Testing guide

## Why we test

The dominant failure mode in this pipeline is **silent numerical corruption**: SNOWPACK accepts
badly-formatted input without error and runs at the wrong temperature, wrong HS, or with the
wrong layer structure.  By the time the stability output looks wrong (or never does), the root
cause is buried three pipeline steps back.

Tests in this repo are targeted at the exact functions where a silent mistake is most likely
and most consequential.  We do not aim for broad line-coverage; we aim to trap the specific
bugs that have burned us or would be hardest to trace.

## What we explicitly do not test

- The SNOWPACK binary itself
- AvaFrame / com1DFA
- Full end-to-end pipeline runs with real UAS or weather data
- The database layer (`sql_util`, `aws_ingest`) — these hit live SQL
- Anything in `windninja/`

## How to run

```bash
# Python unit/integration tests
pytest tests/ -v

# Shell tests (anchor .sno functions)
bash tests/test_anchor_sno.sh
```

All pytest tests use only `tmp_path` fixtures and synthetic data — no real survey files,
no SNOWPACK binary, no network.

## Coverage map

| Module | Test file | Status | Priority | Why it matters |
|---|---|---|---|---|
| `nwp_ingest.py` — all functions | `test_nwp_ingest.py` | ✅ 46 tests | — | Written 2026-09-23; SMET truncate/append logic is stateful; circular wind interpolation has a wrap-around edge case |
| `run_little_prof.sh` anchor fns | `test_anchor_sno.sh` | ✅ 10 tests | — | Shell logic for saving/restoring SNOWPACK restart state between NWP cycles |
| `smet_append._format_smet_row` | `test_smet_append.py` | ✅ | 1 | **Opposite unit conventions from nwp_ingest**: TA stored as K (+273.15), RH as fraction (/100); wrong merge silently runs SNOWPACK at wrong temperature |
| `smet_append._read_smet_last_timestamp` | `test_smet_append.py` | ✅ | 1 | Controls where the append starts; off-by-one duplicates or drops an hour |
| `snowpack_analysis.split_wl_slab` | `test_snowpack_analysis.py` | ✅ | 3 | Pivot of the stability pipeline; near-surface facets must not be picked up; wrong interface shifts every Sk38 and Meloche length |
| `snowpack_analysis.assign_cluster_groups` | `test_snowpack_analysis.py` | ✅ | 6 | 30% overlap threshold and terrain-matching criteria determine what feeds the stability signal |
| `config.ProjectConfig.from_toml` | `test_config.py` | ✅ | 5 | Every pipeline step resolves paths through this; a renamed TOML key surfaces deep in a run |
| `config.ProjectConfig.release_geojsons_for_date` | `test_config.py` | ✅ | 5 | Date-filtering logic for multi-event geojson discovery |
| `reinitialize_snowpack.scour_sno` | `test_reinitialize_snowpack.py` | ✅ | 2 | Partial-layer trimming + header recomputation; corrupt .sno propagates through the rest of the season |
| `reinitialize_snowpack.read_sno/write_sno` | `test_reinitialize_snowpack.py` | ✅ | 2 | Round-trip fidelity; field order must be preserved |
| `reinitialize_snowpack.sanitize_sno_dir` | `test_reinitialize_snowpack.py` | ✅ | 2 | Zero-thickness layer removal; triggers SIGFPE in SNOWPACK if missed |
| `scenario_writer.write_asc` | `test_scenario_writer.py` | ✅ | 4 | `yllcorner` computed from negative transform.e; wrong sign flips the ASC grid origin |
| `scenario_writer` weight normalisation + summary CSV | `test_scenario_writer.py` | ✅ | 4 | Weights must sum to 1.0 for com1DFA relative probabilities |
| `forcing_pipeline.step_reinit` argument routing | `test_forcing_pipeline.py` | ✅ | 7 | Multi-arg pass-through; wrong argument name silently skips the reinit |
| `.pro` → zarr format contract | *(not yet written)* | ⬜ TODO | 8 | See backlog |

## Backlog (not yet implemented)

### Priority 8 — `.pro` → zarr format contract
Write when a new SNOWPACK flavor is adopted.  This is **not** a test of the parser staying the same — it is a format-contract alarm that fires if the new SNOWPACK output breaks the parser.

Steps:
1. Capture a small real `.pro` snippet from the current SNOWPACK binary (a single cluster, a handful of timesteps) and commit it as `tests/fixtures/sample_cluster.pro`.
2. Write a pytest that calls `xsnow.read()` on that fixture and asserts:
   - `HS`, `grain_type`, `hand_hardness`, `critical_cut_length` are present in the dataset
   - Values at a known timestep match expected (hardcode from the fixture)
3. When switching SNOWPACK flavors, run this test against a sample `.pro` from the new flavor.  A failure tells you exactly what variable or unit changed.

Effort: **S** (once the fixture file exists)

## Test conventions

- All pytest tests use `tmp_path` (pytest built-in); no real data files required.
- Helper factories (`_make_*`) live at the top of each test file, not in `conftest.py`, so each file is self-contained.
- Shell tests use a subshell with a temp `SLOPE_DIR`; no side-effects outside `/tmp`.
- Never mock file I/O — write real synthetic files to `tmp_path` and let the production code read them.  Mocking the file system masks format bugs.
