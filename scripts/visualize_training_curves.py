#!/usr/bin/env python3
import os
import re
import argparse
import csv
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DATASET_TOKENS = ['trinity', 'beat', 'ted_expressive', 'ted', 'expressive']
METHOD_TOKENS = ['foundation', 'ae', 'vae', 'diffusion', 'train_diffusion', 'train_ae', 'train_vae']


def find_runs(base_dir: str, include: List[str], exclude: List[str]) -> Dict[str, str]:
    runs = {}
    for root, dirs, files in os.walk(base_dir):
        rel = os.path.relpath(root, base_dir)
        if include and not any(s in rel for s in include):
            continue
        if exclude and any(s in rel for s in exclude):
            continue
        has_events = any(f.startswith('events.') for f in files)
        if has_events:
            runs[rel] = root
    return runs


def infer_metadata(run_rel: str) -> Tuple[str, str]:
    p = run_rel.lower()
    dataset = 'unknown'
    for tok in DATASET_TOKENS:
        if tok in p:
            if tok in ('ted', 'expressive'):
                dataset = 'ted_expressive'
            else:
                dataset = tok
            break
    method = 'unknown'
    for tok in METHOD_TOKENS:
        if tok in p:
            # prefer exact model label
            if tok in ('train_diffusion', 'train_ae', 'train_vae'):
                method = tok
            elif tok == 'diffusion':
                # avoid catching random folder names; keep if nothing else
                method = method if method != 'unknown' else 'diffusion'
            else:
                method = tok
    return dataset, method


def load_scalars(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    acc = EventAccumulator(run_dir)
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        series = [(ev.step, float(ev.value)) for ev in events]
        out[tag] = series
    return out


def dedup_by_step(series: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not series:
        return series
    last = {}
    for s, v in series:
        last[s] = v
    steps = sorted(last.keys())
    return [(s, last[s]) for s in steps]


def moving_average(series: List[Tuple[int, float]], window: int) -> List[Tuple[int, float]]:
    if window <= 1 or len(series) == 0:
        return series
    values = [v for _, v in series]
    steps = [s for s, _ in series]
    smoothed = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window:
            cumsum -= values[i - window]
        denom = min(i + 1, window)
        smoothed.append((steps[i], cumsum / denom))
    return smoothed


def ema(series: List[Tuple[int, float]], alpha: float) -> List[Tuple[int, float]]:
    if alpha <= 0.0 or alpha >= 1.0 or len(series) == 0:
        return series
    out = []
    m = series[0][1]
    for step, val in series:
        m = alpha * val + (1.0 - alpha) * m
        out.append((step, m))
    return out


def sanitize(text: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9_.\-]+', '_', text)
    return s.strip('_') or 'x'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize TensorBoard scalar curves under a base directory')
    parser.add_argument('--base', type=str, default='/home/bosong/workplace/cospeech/custom_diffusiongesture/scripts/model/AEs/ckpt', help='Base ckpt directory to scan')
    parser.add_argument('--out', type=str, default=None, help='Output directory for plots/CSV (default: base/plots)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help='Only include paths containing these substrings')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='Exclude paths containing these substrings')
    parser.add_argument('--tags', type=str, nargs='*', default=[], help='Only plot these scalar tags (e.g., val/loss train/loss)')
    parser.add_argument('--smooth', type=int, default=1, help='Moving average window size (>=1)')
    parser.add_argument('--ema', type=float, default=0.0, help='EMA alpha (0..1). Used if smooth<=1 and 0<alpha<1')
    parser.add_argument('--dpi', type=int, default=120, help='Figure DPI')
    parser.add_argument('--max_runs_per_plot', type=int, default=100, help='Limit lines per plot')
    parser.add_argument('--legend', type=str, choices=['auto', 'always', 'none'], default='always', help='Legend behavior')
    parser.add_argument('--label', type=str, choices=['leaf', 'path'], default='leaf', help='Run label style')
    parser.add_argument('--align', type=str, choices=['step', 'index'], default='step', help='X-axis alignment: TB step or index')
    parser.add_argument('--group_by', type=str, choices=['none', 'dataset', 'method', 'dataset_method'], default='dataset_method', help='How to group plots')

    args = parser.parse_args()

    base = os.path.abspath(args.base)
    out_dir = os.path.abspath(args.out or os.path.join(base, 'plots'))
    ensure_dir(out_dir)

    runs = find_runs(base, args.include, args.exclude)
    if not runs:
        print(f"No event files found under: {base}")
        return
    print(f"Found {len(runs)} runs")

    # Aggregate: tag -> group -> list of (run_label, series)
    tag_to_group_series: DefaultDict[str, DefaultDict[str, List[Tuple[str, List[Tuple[int, float]]]]]] = defaultdict(lambda: defaultdict(list))
    csv_rows: List[Tuple[str, str, str, int, float]] = []

    for run_rel, run_dir in runs.items():
        try:
            scalars = load_scalars(run_dir)
        except Exception as e:
            print(f"[WARN] Failed to load {run_dir}: {e}")
            continue
        label = os.path.basename(run_rel) if args.label == 'leaf' else run_rel
        dataset, method = infer_metadata(run_rel)
        if args.group_by == 'dataset':
            group_key = dataset
        elif args.group_by == 'method':
            group_key = method
        elif args.group_by == 'dataset_method':
            group_key = f"{dataset}|{method}"
        else:
            group_key = 'all'

        for tag, series in scalars.items():
            if args.tags and tag not in args.tags:
                continue
            seq = dedup_by_step(series)
            if args.smooth and args.smooth > 1:
                seq = moving_average(seq, args.smooth)
            elif args.ema and 0.0 < args.ema < 1.0:
                seq = ema(seq, args.ema)
            tag_to_group_series[tag][group_key].append((label, seq))
            for i, (step, val) in enumerate(seq):
                xval = step if args.align == 'step' else i
                csv_rows.append((group_key, label, tag, xval, val))

    # Write CSV
    csv_path = os.path.join(out_dir, 'all_scalars_grouped.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'run', 'tag', 'x', 'value'])
        writer.writerows(csv_rows)
    print(f"Saved CSV: {csv_path}")

    if not HAS_MPL:
        print("matplotlib is not available. Skipping plot generation.")
        return

    # Plot per tag per group
    for tag, groups in tag_to_group_series.items():
        for group_key, series_list in groups.items():
            if len(series_list) == 0:
                continue
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)
            plotted = 0
            for run_label, seq in sorted(series_list, key=lambda x: x[0]):
                if len(seq) == 0:
                    continue
                xs = [s for s, _ in seq]
                ys = [v for _, v in seq]
                if args.align == 'index':
                    xs = list(range(len(ys)))
                ax.plot(xs, ys, label=run_label, linewidth=1.5)
                plotted += 1
                if plotted >= args.max_runs_per_plot:
                    break
            title = f'{tag} [{group_key}]'
            ax.set_title(title)
            ax.set_xlabel('step' if args.align == 'step' else 'index')
            ax.set_ylabel('value')
            ax.grid(True, alpha=0.3)
            if args.legend == 'always':
                ax.legend(fontsize=7, ncol=2, frameon=True)
            elif args.legend == 'auto' and plotted <= 12:
                ax.legend(fontsize=8)
            fname = f'{sanitize(tag)}__{sanitize(group_key)}.png'
            fpath = os.path.join(out_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath)
            plt.close(fig)
            print(f"Saved plot: {fpath}")


if __name__ == '__main__':
    main() 