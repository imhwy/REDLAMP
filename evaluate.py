"""Evaluate RedLamp outputs (result/<run_name>/...) with the metrics used in the paper.

Two protocols:

  'paper' (default) - the setting that reproduces the paper's MSL numbers within ~0.01 (diag_msl_gap.py):
    * SMAP/MSL: metrics on every channel (test_each), averaged over channels. The aggregated test_all series is
      still reported as 'aggregate (reference)' for reference only.
    * score alignment: each window score is assigned to the window's centre timestep (half a window of padding
      on each side), the convention TSB-UAD/VUS use for window-based detectors.
    * RF: Range-F1 (Tatbul et al., alpha=0.2) at threshold mean(score) + 3*std(score), as in the original VUS /
      TSB-UAD metric_new; the oracle-threshold RF is kept as the extra column 'RF (oracle)'.
  'legacy' - the earlier setting: window score at its last timestep, RF at the oracle threshold
    (extra column 'RF (mean+3std)'), SMAP/MSL settings named 'aggregate' and 'per-channel'.

Metrics shared by both (TSB-AD implementation, identical to the original VUS package with slidingWindow=100):
    VUS-ROC, VUS-PR   -> generate_curve(slidingWindow=100): volume over buffers 0..50 per side
    Range-AUC-ROC/PR  -> RangeAUC(window=100); the library extends each anomaly by window//2 = 50 on each side,
                         i.e. the paper's "buffer = half the window size"
    Acc (UCR only)    -> 1 if the highest anomaly score falls inside the labeled anomaly range (with the
                         protocol's alignment applied to the labels)

    python evaluate.py --run_name paper_repro --dataset msl [--protocol legacy] [--result_dir result]
"""
import argparse
import contextlib
import glob
import io
import os

import numpy as np
import pandas as pd

from loaders.load import load_data
from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve

WINDOW = 100
VUS_WINDOW = WINDOW
RANGE_AUC_WINDOW = WINDOW  # buffer = RANGE_AUC_WINDOW // 2 = 50 on each side of an anomaly
PROTOCOLS = ['paper', 'legacy']
DEFAULT_PROTOCOL = 'paper'

DATASETS = ['anomaly_archive', 'iops', 'smd', 'smap', 'msl']
PAPER_NAME = {'anomaly_archive': 'UCR', 'iops': 'AIOps', 'smd': 'SMD', 'smap': 'SMAP', 'msl': 'MSL'}

# RedLamp row of Table 3 (V-R, V-P, RF, Acc) and Table 5 (R-R, R-P)
PAPER = {
    'anomaly_archive': {'VUS-ROC': 0.897, 'VUS-PR': 0.492, 'RF': 0.234, 'Acc': 0.620, 'R-AUC-ROC': 0.902, 'R-AUC-PR': 0.517},
    'iops':            {'VUS-ROC': 0.911, 'VUS-PR': 0.448, 'RF': 0.179, 'R-AUC-ROC': 0.940, 'R-AUC-PR': 0.636},
    'smd':             {'VUS-ROC': 0.813, 'VUS-PR': 0.366, 'RF': 0.235, 'R-AUC-ROC': 0.837, 'R-AUC-PR': 0.429},
    'smap':            {'VUS-ROC': 0.642, 'VUS-PR': 0.360, 'RF': 0.196, 'R-AUC-ROC': 0.658, 'R-AUC-PR': 0.374},
    'msl':             {'VUS-ROC': 0.753, 'VUS-PR': 0.482, 'RF': 0.254, 'R-AUC-ROC': 0.766, 'R-AUC-PR': 0.507},
}
MAIN_METRICS = ['VUS-ROC', 'VUS-PR', 'RF', 'R-AUC-ROC', 'R-AUC-PR', 'Acc']
EXTRA_RF = {'paper': 'RF (oracle)', 'legacy': 'RF (mean+3std)'}
ALIGNMENT = {'paper': 'centre', 'legacy': 'end'}
SETTINGS = {  # (main / reference setting names) per protocol
    'paper': {'channel': 'per-channel', 'aggregate': 'aggregate (reference)', 'entity': 'per-entity'},
    'legacy': {'channel': 'per-channel', 'aggregate': 'aggregate', 'entity': 'per-entity'},
}
# number of entities the paper averages over (smap per-channel: P-2 is listed twice in main.py, so 49 folders)
EXPECTED_ENTITIES = {('anomaly_archive', 'per-entity'): 250, ('iops', 'per-entity'): 29, ('smd', 'per-entity'): 28,
                     ('smap', 'per-channel'): 49, ('msl', 'per-channel'): 23,
                     ('smap', 'aggregate'): 1, ('msl', 'aggregate'): 1,
                     ('smap', 'aggregate (reference)'): 1, ('msl', 'aggregate (reference)'): 1}
PAPER_SEEDS = 5


def metrics_for(protocol):
    return MAIN_METRICS + [EXTRA_RF[protocol]]


class Metricor(basic_metricor):
    # TSB-AD 1.5 finds segments with np.diff(label) == 1, which fails on the soft (extended) labels used by
    # RangeAUC and raises ZeroDivisionError; treat any non-zero value as anomalous like the original TSB-UAD code.
    def range_convers_new(self, label):
        return super().range_convers_new((np.asarray(label) > 0).astype(int))


def test_labels(dataset, entity, root_dir='./dataset', downsampling=1, window=WINDOW, alignment='centre'):
    """Point labels aligned with the window scores produced by main.py (same loader call as datautils).
    Window i covers timesteps [i, i+window); alignment picks which of them its score is compared with."""
    with contextlib.redirect_stdout(io.StringIO()):  # the NASA loader prints channel lists unconditionally
        data = load_data(dataset=dataset, group='test', entities=entity, downsampling=downsampling,
                         min_length=None, root_dir=root_dir, normalize=True, verbose=False, validation=False)
    off = {'end': window - 1, 'centre': window // 2, 'start': 0}[alignment]
    labels = [e.labels[0, off:off + e.labels.shape[1] - window + 1] for e in data.entities if e.Y.shape[1] >= window]
    return np.concatenate(labels).astype(int)


def compute_metrics(score, label, dataset, protocol=DEFAULT_PROTOCOL):
    """Returns the main metrics plus both RF variants ('RF' is the protocol's main one)."""
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=int).ravel()
    if len(score) != len(label):
        raise ValueError(f'score length {len(score)} != label length {len(label)}')
    keys = MAIN_METRICS + ['RF (oracle)', 'RF (mean+3std)']
    if label.sum() == 0:
        return {m: np.nan for m in keys}

    grader = Metricor()
    r_roc, r_pr, _, _, _ = grader.RangeAUC(labels=label, score=score, window=RANGE_AUC_WINDOW, plot_ROC=True)
    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(label, score, VUS_WINDOW)
    pred_3std = (score > score.mean() + 3 * score.std()).astype(int)
    rf_3std = grader.metric_RF1(label, score, preds=pred_3std) if pred_3std.any() else 0.0
    rf_oracle = grader.metric_RF1(label, score)
    return {
        'VUS-ROC': vus_roc,
        'VUS-PR': vus_pr,
        'RF': rf_3std if protocol == 'paper' else rf_oracle,
        'R-AUC-ROC': r_roc,
        'R-AUC-PR': r_pr,
        'Acc': float(label[np.argmax(score)] == 1) if dataset == 'anomaly_archive' else np.nan,
        'RF (oracle)': rf_oracle,
        'RF (mean+3std)': rf_3std,
    }


def _seed_dirs(run_name, dataset, entity, result_dir='./result'):
    pattern = os.path.join(result_dir, run_name, dataset, entity, '*', '*')
    return sorted(d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, 'test_all', 'anomaly_score.npy')))


def evaluate_run(run_name, dataset, entities=None, result_dir='./result', root_dir='./dataset', verbose=True,
                 protocol=DEFAULT_PROTOCOL):
    """Return one row per (setting, entity, seed). For smap/msl both the per-channel rows (test_each) and the
    aggregated test_all row are produced; under the 'paper' protocol the latter is labelled as reference."""
    assert protocol in PROTOCOLS, protocol
    align, names = ALIGNMENT[protocol], SETTINGS[protocol]
    keep = MAIN_METRICS + [EXTRA_RF[protocol]]
    base = os.path.join(result_dir, run_name, dataset)
    if entities is None:
        entities = sorted(os.listdir(base)) if os.path.isdir(base) else []
    rows = []
    for entity in entities:
        for seed_dir in _seed_dirs(run_name, dataset, entity, result_dir):
            seed = os.path.basename(seed_dir)
            score = np.load(os.path.join(seed_dir, 'test_all', 'anomaly_score.npy'))
            m = compute_metrics(score, test_labels(dataset, entity, root_dir, alignment=align), dataset, protocol)
            setting = names['aggregate'] if dataset in ['smap', 'msl'] else names['entity']
            rows.append({'dataset': dataset, 'setting': setting, 'entity': entity, 'seed': seed, **{k: m[k] for k in keep}})
            if verbose:
                print(f'{dataset} {entity} seed={seed}: ' + ', '.join(f'{k}={m[k]:.3f}' for k in keep if not np.isnan(m[k])))

            each_dir = os.path.join(seed_dir, 'test_each')
            if dataset in ['smap', 'msl'] and os.path.isdir(each_dir):
                for ch in sorted(os.listdir(each_dir)):
                    f = os.path.join(each_dir, ch, 'test_all', 'anomaly_score.npy')
                    if not os.path.isfile(f):
                        continue
                    m = compute_metrics(np.load(f), test_labels(dataset, ch, root_dir, alignment=align), dataset, protocol)
                    rows.append({'dataset': dataset, 'setting': names['channel'], 'entity': ch, 'seed': seed,
                                 **{k: m[k] for k in keep}})
    df = pd.DataFrame(rows)
    df.attrs['protocol'] = protocol
    return df


def summarize(df, protocol=None):
    """Mean over entities within each seed, then mean/std over seeds, next to the paper's numbers."""
    if df.empty:
        return df
    protocol = protocol or df.attrs.get('protocol') or ('legacy' if 'RF (mean+3std)' in df else 'paper')
    metrics = [m for m in metrics_for(protocol) if m in df]
    per_seed = df.groupby(['dataset', 'setting', 'seed'])[metrics].mean()
    mean = per_seed.groupby(['dataset', 'setting']).mean()
    std = per_seed.groupby(['dataset', 'setting']).std()
    n_ent = df.groupby(['dataset', 'setting', 'seed'])['entity'].nunique().groupby(['dataset', 'setting']).min()
    n_seed = per_seed.groupby(['dataset', 'setting']).size()

    rows = []
    for (dataset, setting), r in mean.iterrows():
        n_e, n_s = int(n_ent.loc[(dataset, setting)]), int(n_seed.loc[(dataset, setting)])
        expected = EXPECTED_ENTITIES.get((dataset, setting))
        if '(reference)' in setting:
            status = 'reference (not compared with the paper)'
        elif expected is not None and n_e < expected:
            status = f'PARTIAL ({n_e}/{expected} entities) - not comparable'
        elif n_s < PAPER_SEEDS:
            status = f'{n_s}/{PAPER_SEEDS} seeds'
        else:
            status = 'complete (all entities x 5 seeds)' + (', protocol paper' if protocol == 'paper' else ', protocol legacy')
        for m in metrics:
            if np.isnan(r[m]):
                continue
            paper = PAPER.get(dataset, {}).get(m, np.nan)
            rows.append({
                'dataset': PAPER_NAME.get(dataset, dataset), 'setting': setting, 'metric': m,
                'ours (mean)': round(r[m], 3), 'std (seeds)': round(std.loc[(dataset, setting), m], 3),
                'paper': paper, 'diff': round(r[m] - paper, 3) if not np.isnan(paper) else np.nan,
                '#entities': n_e, '#seeds': n_s, 'status': status,
            })
    out = pd.DataFrame(rows)
    # main rows first, reference rows last
    return out.sort_values('setting', key=lambda s: s.str.contains('(reference)', regex=False), kind='stable').reset_index(drop=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--protocol', default=DEFAULT_PROTOCOL, choices=PROTOCOLS)
    p.add_argument('--result_dir', default='./result')
    a = p.parse_args()
    pd.set_option('display.width', 250)
    print(summarize(evaluate_run(a.run_name, a.dataset, result_dir=a.result_dir, verbose=False, protocol=a.protocol)).to_string())
