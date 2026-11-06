"""Diagnose why our SMAP/MSL numbers differ from the paper (Table 3/5) using existing results (no training).

Compares, averaged over seeds:
  protocol   : aggregate (test_all) | per-channel mean (test_each) | concatenated per-channel scores (test_each)
  metric impl: TSB-AD as in evaluate.py (VUS/Range-AUC window 100, RF oracle and RF mean+3std)
               | original VUS package (TheDatumOrg/VUS, pip `vus`) with slidingWindow = find_length / 100 / 50,
                 RF from its metric_new (threshold mean+3std)
  alignment  : window score assigned to the window's end / centre / start timestep
Optionally (--baseline) per-channel IsolationForest and |x - train mean| baselines under the same evaluator.

    python diag_msl_gap.py --run_name paper_repro --dataset msl [--seeds 0 1 2 3 4] [--alignments end centre start] [--baseline]
Writes result/<run_name>/diag/<dataset>_gap.csv
"""
import argparse
import contextlib
import glob
import io
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import evaluate as ev
from loaders.load import load_data, msl_data_set_number, smap_data_set_number

W = 100
METRICS = ['VUS-ROC', 'VUS-PR', 'RF', 'R-AUC-ROC', 'R-AUC-PR']


def import_vus(vus_dir=None):
    if vus_dir:
        sys.path.insert(0, vus_dir)
    try:
        import vus.utils.metrics  # noqa: F401
    except ImportError:
        # the package's own requirements (tensorflow, tsfresh, ...) are only needed by its models, not the metrics
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'vus'], check=True)
    from vus.utils.metrics import metricor
    from vus.basic_metrics import basic_metricor
    from vus.utils.slidingWindows import find_length
    return metricor, basic_metricor, find_length


def load(dataset, entity, group='test'):
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data(dataset, group, entity, downsampling=1, root_dir='./dataset', verbose=False)
    return [e for e in data.entities if e.Y.shape[1] >= W]


def aligned_labels(entities, how):
    off = {'end': W - 1, 'centre': W // 2, 'start': 0}[how]
    return np.concatenate([e.labels[0][off:off + e.labels.shape[1] - W + 1] for e in entities]).astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default='paper_repro')
    p.add_argument('--dataset', default='msl', choices=['msl', 'smap'])
    p.add_argument('--seeds', nargs='+', default=None, help='default: every seed found')
    p.add_argument('--alignments', nargs='+', default=['end', 'centre', 'start'])
    p.add_argument('--baseline', action='store_true', help='also score IsolationForest and |x - train mean| baselines')
    p.add_argument('--vus_dir', default=None, help='folder containing the vus package (otherwise pip install vus)')
    p.add_argument('--result_dir', default=None, help='run folder holding <dataset>/<dataset>/*/<seed>/ (default: result/<run_name>)')
    p.add_argument('--out', default=None, help='output csv (default: result/<run_name>/diag/<dataset>_gap.csv)')
    args = p.parse_args()
    t0 = time.time()

    metricor, basic_metricor, find_length = import_vus(args.vus_dir)
    ds = args.dataset
    order = msl_data_set_number if ds == 'msl' else smap_data_set_number  # aggregate order used by main.py (SMAP lists P-2 twice)
    channels = list(dict.fromkeys(order))
    paper = ev.PAPER[ds]

    result_dir = args.result_dir or f'result/{args.run_name}'
    seed_dirs = {os.path.basename(d): d for d in glob.glob(f'{result_dir}/{ds}/{ds}/*/*')
                 if os.path.isfile(f'{d}/test_all/anomaly_score.npy')}
    seeds = [s for s in (args.seeds or sorted(seed_dirs)) if s in seed_dirs]
    if not seeds:
        sys.exit(f'no results under {result_dir}/{ds}/{ds}/*/<seed>/test_all/anomaly_score.npy')
    print(f'{ds}: seeds {seeds}', flush=True)

    agg_ents = load(ds, ds)
    ch_ents = {c: load(ds, c) for c in channels}
    raw = lambda es: np.concatenate([e.Y[0] for e in es])  # first feature, as TSB-UAD/VUS use for find_length

    def tsbad(s, l, x):
        m = ev.compute_metrics(s, l, ds)
        return {'VUS-ROC': m['VUS-ROC'], 'VUS-PR': m['VUS-PR'], 'RF': m['RF (oracle)'], 'RF (mean+3std)': m['RF (mean+3std)'],
                'R-AUC-ROC': m['R-AUC-ROC'], 'R-AUC-PR': m['R-AUC-PR']}

    def vus(s, l, sw):
        g = metricor()
        r_roc, r_pr, _, _, _ = g.RangeAUC(labels=l, score=s, window=sw, plot_ROC=True)
        _, _, _, _, auc3d, ap3d = g.RangeAUC_volume_opt(labels_original=l, score=s, windowSize=sw, thre=250)
        rf = basic_metricor().metric_new(l, s, plot_ROC=False)[8]  # Range-F at threshold mean+3*std
        return {'VUS-ROC': float(np.mean(auc3d)), 'VUS-PR': float(np.mean(ap3d)), 'RF': rf, 'R-AUC-ROC': r_roc, 'R-AUC-PR': r_pr}

    impls = {
        'TSB-AD (evaluate.py), RF oracle': tsbad,
        'VUS pkg, sw=find_length': lambda s, l, x: vus(s, l, find_length(x)),
        'VUS pkg, sw=100': lambda s, l, x: vus(s, l, 100),
        'VUS pkg, sw=50': lambda s, l, x: vus(s, l, 50),
    }

    def score_series(series):
        out = {}
        for name, fn in impls.items():
            ms = [fn(s, l, x) for s, l, x in series if l.sum() > 0]
            out[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        return out

    rows = []
    for seed in seeds:
        sd = seed_dirs[seed]
        agg = np.load(f'{sd}/test_all/anomaly_score.npy')
        each = {c: np.load(f'{sd}/test_each/{c}/test_all/anomaly_score.npy') for c in channels}
        concat = np.concatenate([each[c] for c in order])
        for how in args.alignments:
            variants = {
                'aggregate (test_all)': [(agg, aligned_labels(agg_ents, how), raw(agg_ents))],
                'per-channel mean (test_each)': [(each[c], aligned_labels(ch_ents[c], how), raw(ch_ents[c])) for c in channels],
                'concat per-channel (test_each)': [(concat, np.concatenate([aligned_labels(ch_ents[c], how) for c in order]),
                                                    np.concatenate([raw(ch_ents[c]) for c in order]))],
            }
            for proto, series in variants.items():
                for impl, m in score_series(series).items():
                    rows.append({'protocol': proto, 'metric impl': impl, 'alignment': how, 'seed': seed, **m})
            print(f'  seed {seed} alignment {how} done ({time.time() - t0:.0f}s)', flush=True)

    if args.baseline:
        from sklearn.ensemble import IsolationForest
        base = {'IsolationForest': [], '|x - train mean|': []}
        for c in channels:
            tr, te = load(ds, c, 'train')[0].Y.T, ch_ents[c][0].Y.T
            lab = ch_ents[c][0].labels[0][W - 1:].astype(int)
            base['IsolationForest'].append((-IsolationForest(random_state=0).fit(tr).score_samples(te)[W - 1:], lab, te[:, 0]))
            base['|x - train mean|'].append((np.abs(te - tr.mean(0)).mean(1)[W - 1:], lab, te[:, 0]))
        for k, series in base.items():
            for impl, m in score_series(series).items():
                rows.append({'protocol': f'BASELINE {k} (per-channel mean)', 'metric impl': impl, 'alignment': 'end', 'seed': '-', **m})

    df = pd.DataFrame(rows)
    keys = ['protocol', 'metric impl', 'alignment']
    cols = [c for c in METRICS + ['RF (mean+3std)'] if c in df]
    mean = df.groupby(keys, sort=False)[cols].mean()
    std = df.groupby(keys, sort=False)[METRICS].std().add_suffix(' std')
    out = pd.concat([mean, std], axis=1).reset_index()
    out['#seeds'] = df.groupby(keys, sort=False)['seed'].nunique().values
    out['mean |diff| vs paper'] = out[METRICS].sub(pd.Series({m: paper[m] for m in METRICS})).abs().mean(1)
    path = args.out or f'result/{args.run_name}/diag/{ds}_gap.csv'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.round(4).to_csv(path, index=False)
    df.to_csv(path.replace('.csv', '_per_seed.csv'), index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 300)
    show = out[keys + METRICS + ['RF (mean+3std)', 'mean |diff| vs paper']] if 'RF (mean+3std)' in out else out
    print(show.round(3).to_string())
    print(f'paper RedLamp {ds.upper()}:', {m: paper[m] for m in METRICS})
    best = out[~out.protocol.str.startswith('BASELINE')].sort_values('mean |diff| vs paper').iloc[0]
    print('closest to paper:', ' | '.join(str(best[k]) for k in keys), f"(mean |diff| {best['mean |diff| vs paper']:.3f})")
    print(f'saved {path} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
