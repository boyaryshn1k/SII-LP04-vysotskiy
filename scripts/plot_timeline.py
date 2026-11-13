"""Графики по результатам прогона: сколько объектов найдено по кадрам.

Читает out/timeline.csv и out/report.json, сохраняет out/timeline_plot.png.
Если есть out/stride_exp/stride_comparison.json — дополнительно рисует
сравнение stride.

Пример:
    python -m scripts.plot_timeline
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI, чтобы работало и в VS Code, и на сервере
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_timeline(path: Path) -> tuple[list[int], list[int]]:
    frames, counts = [], []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            frames.append(int(row["frame_index"]))
            counts.append(int(row["count"]))
    return frames, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out")
    parser.add_argument(
        "--save", type=Path, default=ROOT / "out" / "timeline_plot.png"
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    frames, counts = read_timeline(out_dir / "timeline.csv")

    fps = report.get("video_fps") or 30.0
    stride = report["stride"]
    times = [f / fps for f in frames]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [3, 2]}
    )

    ax1.step(times, counts, where="post", color="#1f77b4", linewidth=1.2)
    ax1.fill_between(times, counts, step="post", color="#1f77b4", alpha=0.25)
    ax1.axhline(
        report["detections_per_frame_mean"],
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"среднее = {report['detections_per_frame_mean']:.2f}",
    )
    ax1.set_title(
        f"Класс «{report['class_name']}», stride = {stride}, "
        f"обработано {report['frames_processed']} из {report['frames_total']} кадров"
    )
    ax1.set_xlabel("время в исходном видео, с")
    ax1.set_ylabel("объектов в кадре")
    ax1.set_ylim(0, max(counts) + 1 if max(counts) else 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    unique = sorted(set(counts))
    hist = [counts.count(u) for u in unique]
    bars = ax2.bar([str(u) for u in unique], hist, color="#2ca02c", alpha=0.85)
    ax2.bar_label(bars, padding=2)
    ax2.set_title("Распределение числа найденных объектов по обработанным кадрам")
    ax2.set_xlabel("объектов в кадре")
    ax2.set_ylabel("кадров")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=140)
    print(f"Сохранено: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
