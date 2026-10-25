"""Build the table sections of docs/results_energy_mode.md from the experiment CSVs.

    python src/avachain/energy_results.py > /tmp/tables.md
    python src/avachain/energy_results.py --tag fit_gwl modes envelope
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / 'outputs/little_prof_energy/experiments'


def md(df: pd.DataFrame, floatfmt='{:.2f}') -> str:
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append('' if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def modes_table():
    d = pd.read_csv(OUT / 'fixed_triggers.csv')
    g = d.groupby(['mask_mode', 'mode'], sort=False).agg(
        n=('iou', 'size'), area_m2=('area_m2', 'median'), iou_median=('iou', 'median'),
        iou_max=('iou', 'max'), over_m2=('over_m2', 'median'), under_m2=('under_m2', 'median'),
        crown_offset_m=('crown_offset_median_m', 'median'),
        flank_left_m=('flank_left_m', 'median'), flank_right_m=('flank_right_m', 'median'),
        frac_outside_mask=('frac_outside_mask', 'median'), cap_bound=('cap_bound', 'sum'))
    return md(g.reset_index())


def shipped_table():
    d = pd.read_csv(OUT / 'fixed_triggers.csv')
    d = d[d.trigger_set == 'shipped']
    p = d.pivot_table(index='trigger', columns=['mask_mode', 'mode'], values=['area_m2', 'iou'])
    rows = []
    for t in p.index:
        for mm in ('gate', 'diagnostic'):
            for mode in ('legacy', 'energy', 'slope_only', 'stability_connected'):
                rows.append({'trigger': t, 'mask': mm, 'mode': mode,
                             'area_m2': p.loc[t, ('area_m2', mm, mode)],
                             'iou': p.loc[t, ('iou', mm, mode)]})
    return md(pd.DataFrame(rows), '{:.3f}')


def mechanism_table():
    m = pd.read_csv(OUT / 'fixed_mechanisms.csv')
    t = m.groupby(['mask_mode', 'mode', 'segment', 'reason']).n.sum().reset_index()
    t['share'] = t.n / t.groupby(['mask_mode', 'mode', 'segment']).n.transform('sum')
    t = t.sort_values(['mask_mode', 'mode', 'segment', 'share'], ascending=[True, True, True, False])
    t['reasons'] = t.apply(lambda r: f"{r.reason} {r.share:.0%}", axis=1)
    w = t.groupby(['mask_mode', 'mode', 'segment']).agg(
        n=('n', 'sum'), reasons=('reasons', lambda s: ', '.join(s.head(4)))).reset_index()
    return md(w)


def params_tables():
    p = pd.read_csv(OUT / 'params_lnpi.csv')
    p['value'] = p['value'].fillna('None')
    cols = ['param', 'value', 'lnPi_in_p10', 'lnPi_in_p50', 'lnPi_in_p90', 'lnPi_out_p50',
            'frac_Pi_ge1_in', 'frac_Pi_ge1_out', 'band_frac_Pi_ge1_in', 'band_frac_Pi_ge1_out',
            'crosses_at_boundary']
    r = pd.read_csv(OUT / 'params_runs.csv')
    r['value'] = r['value'].fillna('None')
    g = r.groupby(['param', 'value'], sort=False).agg(
        area_median=('area_m2', 'median'), area_max=('area_m2', 'max'),
        iou_median=('iou', 'median'), iou_max=('iou', 'max'),
        crown_offset_median=('crown_offset_median_m', 'median'),
        crown_offset_min=('crown_offset_median_m', 'min'),
        crown_offset_max=('crown_offset_median_m', 'max')).reset_index()
    return md(p[cols]), md(g)


def envelope_table():
    f = OUT / 'envelope_runs.csv'
    if not f.exists():
        return '(envelope not run)'
    d = pd.read_csv(f)
    g = d.groupby(['mode', 'mask_mode']).agg(
        n=('iou', 'size'), with_polygon=('area_m2', lambda s: int((s > 0).sum())),
        area_median=('area_m2', 'median'), iou_median=('iou', 'median'),
        iou_max=('iou', 'max'), cap_bound=('cap_bound', 'sum')).reset_index()
    return md(g)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--tag':      # read experiments/<tag>/
        OUT = OUT / sys.argv[2]
        sys.argv = sys.argv[:1] + sys.argv[3:]
    def _check():
        f = OUT / 'check.json'
        return json.dumps(json.loads(f.read_text()), indent=1) if f.exists() else '{}'
    sections = {
        'modes': modes_table, 'shipped': shipped_table, 'mechanisms': mechanism_table,
        'lnpi': lambda: params_tables()[0], 'param_runs': lambda: params_tables()[1],
        'envelope': envelope_table, 'check': _check}
    which = sys.argv[1:] or list(sections)
    for k in which:
        print(f'<!-- {k} -->\n{sections[k]()}\n')
