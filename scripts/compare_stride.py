"""Сравнение разных stride на одном и том же видео.

Запускает `python -m src.count_video` для каждого значения stride, собирает
агрегаты из report.json и складывает сводку в out/stride_comparison.md и .json.

Пример:
    python -m scripts.compare_stride --source data/sample.mp4 --class "stop sign" --strides 5 10 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(
    source: Path, class_name: str, stride: int, out_dir: Path, extra: list[str]
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "src.count_video",
        "--source",
        str(source),
        "--class",
        class_name,
        "--stride",
        str(stride),
        "--out",
        str(out_dir),
        "--no-save-video",
        "--max-frame-saves",
        "0",
        *extra,
    ]
    print(f"\n=== stride={stride} ===", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - start
    if completed.returncode != 0:
        print(completed.stdout[-2000:], file=sys.stderr)
        print(completed.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"Команда упала на stride={stride}")
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report["wall_clock_sec"] = round(wall, 2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/sample_stop_signs.mp4"))
    parser.add_argument("--class", dest="class_name", default="stop sign")
    parser.add_argument("--strides", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--base-out", type=Path, default=Path("out/stride_exp"))
    parser.add_argument("--model", default="")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args(argv)

    extra: list[str] = []
    if args.model:
        extra += ["--model", args.model]
    if args.conf is not None:
        extra += ["--conf", str(args.conf)]
    if args.imgsz is not None:
        extra += ["--imgsz", str(args.imgsz)]

    rows = []
    for stride in args.strides:
        report = run_one(
            args.source, args.class_name, stride, args.base_out / f"stride_{stride}", extra
        )
        rows.append(
            {
                "stride": stride,
                "frames_processed": report["frames_processed"],
                "frames_skipped": report["frames_skipped"],
                "mean": report["detections_per_frame_mean"],
                "max": report["detections_per_frame_max"],
                "total": report["detections_total"],
                "frames_with_detections": report["frames_with_detections"],
                "elapsed_sec": report["elapsed_sec"],
                "wall_clock_sec": report["wall_clock_sec"],
                "processed_fps": report["processed_fps"],
            }
        )

    best_total = max(r["total"] for r in rows) or 1
    header = (
        "| stride | обработано кадров | среднее | максимум | детекций всего | "
        "кадров с детекциями | время, с | кадр/с |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| {s} | {p} | {m:.2f} | {mx} | {t} | {w} | {e:.1f} | {f:.2f} |".format(
                s=r["stride"],
                p=r["frames_processed"],
                m=r["mean"],
                mx=r["max"],
                t=r["total"],
                w=r["frames_with_detections"],
                e=r["elapsed_sec"],
                f=r["processed_fps"],
            )
        )

    args.base_out.mkdir(parents=True, exist_ok=True)
    (args.base_out / "stride_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.base_out / "stride_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nСохранено: {args.base_out / 'stride_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
