"""Подсчёт объектов выбранного класса в видео (лабораторная работа №4, вариант 7).

Конвейер: видео -> отбор каждого N-го кадра -> детекция YOLO -> подсчёт -> report.json.

Пример запуска (из корня проекта):

    python -m src.count_video --source data/sample.mp4 --class "stop sign" --stride 5

По умолчанию результат складывается в out/: report.json, annotated.mp4 и кадры
с рамками в out/frames/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

LOGGER = logging.getLogger("count_video")

REPORT_VERSION = "v1.0"
DEFAULT_MODEL = "yolo11n.pt"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.count_video",
        description="Подсчёт объектов заданного класса в видео с шагом кадров (stride).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="путь к видеофайлу (mp4/avi/mov/...)",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        required=True,
        help='имя класса COCO, например "stop sign" или "person"',
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="обрабатывать каждый N-й кадр (1 = каждый кадр)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="веса YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="порог уверенности")
    parser.add_argument("--iou", type=float, default=0.45, help="порог NMS (IoU)")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="размер входа модели (сторона, px); 960 выбрано ради мелких знаков",
    )
    parser.add_argument(
        "--device", default="", help="устройство для torch: cpu, cuda:0, mps (пусто = авто)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("out"), help="каталог для результатов"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="обработать только первые N кадров источника (0 = всё видео)",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="сохранять видео с рамками (--no-save-video чтобы отключить)",
    )
    parser.add_argument(
        "--max-frame-saves",
        type=int,
        default=30,
        help="сколько кадров с детекциями сохранить в out/frames/",
    )
    parser.add_argument(
        "--video-fps",
        choices=("source", "stride"),
        default="source",
        help=(
            "частота кадров annotated.mp4: source — как в исходнике (каждый "
            "обработанный кадр повторяется stride раз); stride — fps/stride"
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="подробный лог (DEBUG)"
    )
    return parser


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def resolve_class_id(model_names: dict[int, str], class_name: str) -> int:
    """Ищет id класса по имени без учёта регистра и лишних пробелов."""
    wanted = " ".join(class_name.strip().lower().split())
    for idx, name in model_names.items():
        if name.lower() == wanted:
            return int(idx)
    available = ", ".join(sorted(model_names.values()))
    raise SystemExit(
        f"Класс {class_name!r} не найден в модели. Доступные классы: {available}"
    )


def video_meta(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return fps, frames, width, height


def draw_detection(frame, box, conf: float, class_name: str, color=(0, 0, 255)):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(12, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, frame_idx: int, stride: int, count: int, class_name: str) -> None:
    lines = [
        f"frame {frame_idx} (stride {stride})",
        f"{class_name}: {count}",
    ]
    y = 30
    for text in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 34


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    # mp4v есть в любой сборке OpenCV; avc1 требует внешний openh264.dll,
    # поэтому идёт вторым — но даёт заметно меньший файл, если доступен.
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), (width, height)
        )
        if writer.isOpened():
            LOGGER.debug("Видео с рамками: кодек %s", codec)
            return writer
        writer.release()
    LOGGER.warning("Не удалось открыть VideoWriter для %s — видео не будет сохранено", path)
    return None


# --------------------------------------------------------------------------- #
# Основной конвейер
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO  # импорт здесь, чтобы --help работал без torch

    source = args.source.expanduser()
    if not source.is_file():
        raise SystemExit(f"Файл не найден: {source}")
    if args.stride < 1:
        raise SystemExit("--stride должен быть >= 1")

    out_dir: Path = args.out.expanduser()
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {source}")
    fps, frames_total_hdr, width, height = video_meta(cap)
    if frames_total_hdr <= 0:
        # некоторые контейнеры не сообщают длину — досчитаем перебором
        LOGGER.warning("Длительность в заголовке неизвестна, считаю кадры перебором")
        frames_total_hdr = 0

    LOGGER.info("Источник: %s", source)
    LOGGER.info(
        "Видео: %dx%d, %.2f fps, ~%d кадров, ~%.1f c",
        width,
        height,
        fps,
        frames_total_hdr,
        (frames_total_hdr / fps) if fps else 0.0,
    )

    LOGGER.info("Загрузка модели %s ...", args.model)
    model = YOLO(args.model)
    class_id = resolve_class_id(dict(model.names), args.class_name)
    class_name = dict(model.names)[class_id]
    LOGGER.info("Класс %r -> id %d", class_name, class_id)

    writer = None
    repeat = args.stride if args.video_fps == "source" else 1
    video_fps = fps if args.video_fps == "source" else fps / args.stride
    if args.save_video:
        # обработанные кадры идут с шагом stride. Режим "stride": пишем их с
        # fps/stride — темп совпадает с источником, но файл «рваный».
        # Режим "source": каждый обработанный кадр повторяется stride раз,
        # поэтому частота кадров как в исходнике (плавнее для плеера).
        writer = open_writer(out_dir / "annotated.mp4", video_fps, width, height)
        LOGGER.info(
            "Видео с рамками: %.2f fps, повтор каждого кадра x%d", video_fps, repeat
        )

    detect_kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
        classes=[class_id],
    )
    if args.device:
        detect_kwargs["device"] = args.device

    counts: list[int] = []
    timeline: list[tuple[int, int]] = []
    frames_with_detections = 0
    detections_total = 0
    saved_frames = 0
    frames_read = 0
    frame_idx = -1
    started = time.perf_counter()
    progress_step = max(1, (args.limit or frames_total_hdr or 3000) // 10)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frames_read += 1
            if args.limit and frames_read > args.limit:
                break
            if frame_idx % args.stride != 0:
                continue  # кадр пропускается: детекция не запускается

            result = model.predict(frame, **detect_kwargs)[0]
            boxes = result.boxes
            n = 0 if boxes is None else len(boxes)
            counts.append(n)
            timeline.append((frame_idx, n))
            detections_total += n

            for box in ([] if boxes is None else boxes):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                draw_detection(frame, xyxy, conf, class_name)

            draw_hud(frame, frame_idx, args.stride, n, class_name)

            if n:
                frames_with_detections += 1
                if saved_frames < args.max_frame_saves:
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    name = f"frame_{frame_idx:06d}_n{n:02d}.jpg"
                    cv2.imwrite(str(frames_dir / name), frame)
                    saved_frames += 1

            if writer is not None:
                for _ in range(repeat):
                    writer.write(frame)

            if len(counts) % progress_step == 0:
                LOGGER.info(
                    "обработано кадров: %d, пропущено: %d, найдено объектов: %d",
                    len(counts),
                    frames_read - len(counts),
                    detections_total,
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    if not counts:
        raise SystemExit(
            "Ни один кадр не обработан — проверьте --stride, --limit и целостность видео."
        )

    frames_total = args.limit if args.limit else (frames_total_hdr or frames_read)
    processed = len(counts)
    skipped = frames_read - processed

    LOGGER.info(
        "Готово: кадров всего %d, обработано %d, пропущено %d, за %.1f c (%.2f кадра/с)",
        frames_total,
        processed,
        skipped,
        elapsed,
        processed / elapsed,
    )

    timeline_path = out_dir / "timeline.csv"
    with timeline_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("frame_index,count\n")
        for idx, n in timeline:
            fh.write(f"{idx},{n}\n")

    report = {
        "source": source.as_posix(),
        "class_name": class_name,
        "stride": args.stride,
        "frames_total": frames_total,
        "frames_processed": processed,
        "detections_per_frame_mean": round(detections_total / processed, 3),
        "detections_per_frame_max": max(counts),
        "unique_tracks": None,
        "model_version": f"{Path(args.model).name} (ultralytics {_ultralytics_version()})",
        "elapsed_sec": round(elapsed, 2),
        # --- дополнительные поля, полезные для отчёта ---
        "report_version": REPORT_VERSION,
        "detections_total": detections_total,
        "frames_with_detections": frames_with_detections,
        "frames_skipped": skipped,
        "processed_fps": round(processed / elapsed, 2),
        "video_fps": round(fps, 2),
        "video_duration_sec": round(frames_total / fps, 2) if fps else None,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "annotated_video_fps": round(video_fps, 2),
        "annotated_frame_repeat": repeat,
        "tracker": "none",
        "notes": (
            "Трекер не используется, поэтому unique_tracks = null: счётчик считает "
            "детекции на каждом обработанном кадре, а не уникальные объекты. "
            "Один и тот же знак STOP учитывается многократно, пока остаётся в кадре; "
            "пропущенные между обработанными кадрами объекты не учитываются вовсе."
        ),
        "artifacts": {
            "annotated_video": f"{(out_dir / 'annotated.mp4').as_posix()}"
            if args.save_video
            else None,
            "frames_dir": frames_dir.as_posix() if saved_frames else None,
            "frames_saved": saved_frames,
            "timeline_csv": timeline_path.as_posix(),
        },
    }

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("report.json: %s", report_path)
    return report


def _ultralytics_version() -> str:
    try:
        import ultralytics

        return getattr(ultralytics, "__version__", "unknown")
    except Exception:  # pragma: no cover - на всякий случай
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    report = run(args)
    print(
        "\nИтог: класс {c}, stride {s}, среднее {m:.2f}, максимум {mx}, "
        "кадров {p}/{t} за {e:.1f} c".format(
            c=report["class_name"],
            s=report["stride"],
            m=report["detections_per_frame_mean"],
            mx=report["detections_per_frame_max"],
            p=report["frames_processed"],
            t=report["frames_total"],
            e=report["elapsed_sec"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
