#!/usr/bin/env python3
"""Utility script to visualize PPO training metrics produced by learning_agent_ppo."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _flatten_record(record: Dict, prefix: str = "", out: Dict | None = None) -> Dict:
    if out is None:
        out = {}
    for key, value in record.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_record(value, f"{full_key}.", out)
        elif isinstance(value, (int, float)):
            out[full_key] = value
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, (int, float)):
                    out[f"{full_key}.{idx}"] = item
    return out


def flatten_records(records: List[Dict]) -> List[Dict]:
    return [_flatten_record(record) for record in records]


def collect_series(records: List[Dict], field: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for record in records:
        if "update" not in record or field not in record:
            continue
        value = record[field]
        if value is None:
            continue
        xs.append(record["update"])
        ys.append(value)
    return xs, ys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics logged to metrics.jsonl.")
    parser.add_argument(
        "--metrics-file",
        "--metrics-path",
        dest="metrics_file",
        type=Path,
        required=True,
        help="Path to the metrics.jsonl generated during PPO training.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["policy_loss", "value_loss", "ep_rew_mean", "success_rate", "gail.expert_loss", "gail.policy_loss"],
        help=(
            "Metric fields to plot. Supports nested names such as gail.expert_loss or ep_rew_mean_separate.0 "
            "(default: policy_loss value_loss ep_rew_mean success_rate gail.expert_loss gail.policy_loss)."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of points (useful for large logs).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Enable watch mode that periodically reloads metrics and refreshes the plots.",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        help="Seconds between refreshes when --watch is enabled (default: 5).",
    )
    return parser.parse_args()


def render(records: List[Dict], fields: List[str], fig=None, axes=None):
    num_fields = len(fields)
    if fig is None or axes is None or (isinstance(axes, list) and len(axes) != num_fields):
        fig, axes = plt.subplots(num_fields, 1, sharex=True, figsize=(8, max(2.5 * num_fields, 4)))
    if num_fields == 1:
        axes = [axes]
    for ax, field in zip(axes, fields):
        ax.clear()
        xs, ys = collect_series(records, field)
        if xs:
            ax.plot(xs, ys, label=field)
        ax.set_ylabel(field)
        ax.grid(True, linestyle="--", alpha=0.4)
        if xs:
            ax.legend(loc="best")
        else:
            ax.set_title(f"{field} (no data)")
    axes[-1].set_xlabel("PPO Update")
    fig.tight_layout()
    return fig, axes


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8")
    fig = axes = None
    try:
        while True:
            raw_records = load_metrics(args.metrics_file)
            flattened = flatten_records(raw_records)
            if args.max_points is not None and args.max_points > 0:
                flattened = flattened[-args.max_points :]
            if not flattened:
                if not args.watch:
                    raise SystemExit(f"No records found in {args.metrics_file}")
                time.sleep(args.watch_interval)
                continue
            fig, axes = render(flattened, args.fields, fig, axes)
            if not args.watch:
                plt.show()
                break
            plt.pause(0.001)
            time.sleep(max(args.watch_interval, 0.1))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
