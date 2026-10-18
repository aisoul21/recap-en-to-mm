"""Auto-detect Chinese (CJK) on-screen text and build time-ranged blur boxes.

Used by the recap pipeline (core.py) for Rednote / Chinese-source videos:
text that is burned into the picture (titles, comments, watermarks) is found
with OCR and blurred only while it is actually visible.

Pipeline wiring (core.py):
    boxes = detect_cjk_boxes(ffmpeg, final_video, work_dir, log=log)
    # → [{"x":…, "y":…, "w":…, "h":…, "t0":…, "t1":…}, …]  (x/y/w/h are 0..1)
    # core.apply_production_overlays() understands the optional t0/t1 via
    # overlay enable='between(t,t0,t1)'.

Detection runs on the FINAL (Smart Sync) video, so t0/t1 are already on the
output timeline — no time-mapping needed.

OCR engine: EasyOCR (optional dependency). Install once with:
    py -3.12 -m pip install easyocr
The OCR models (~100-150 MB) download automatically on first use.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# CJK Unified Ideographs (+ Extension A, Compatibility, Ext. B plane)
CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002a6df]")


def has_cjk(text: Any) -> bool:
    """True if the string contains at least one CJK character."""
    return bool(text) and bool(CJK_RE.search(str(text)))


def _easyocr_reader() -> Any:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "easyocr is not installed — run: py -3.12 -m pip install easyocr"
        ) from exc
    # First call downloads the detection + recognition models (~100-150 MB).
    return easyocr.Reader(["ch_sim", "en"], gpu=False)


def _extract_frames(
    ffmpeg: str,
    video: Path,
    interval_s: float,
    width: int,
    out_dir: Path,
    log: LogFn,
) -> list[Path]:
    """Sample one frame every interval_s seconds, downscaled for speed.

    Returns the frame paths sorted by time; frame k ≈ t = k * interval_s.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={1.0 / interval_s:.6f},scale={int(width)}:-2"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vf", vf, str(out_dir / "frm_%05d.jpg"),
    ]
    log(f"🔍 Sampling frames every {interval_s:g}s for OCR…")
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frm_*.jpg"))


def _box_area(b: dict[str, float]) -> float:
    return max(0.0, b["w"]) * max(0.0, b["h"])


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _union(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    out = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    for k in ("text", "conf"):
        if k in a or k in b:
            out[k] = a.get(k, b.get(k))
    return out


def _merge_boxes(
    boxes: list[dict[str, float]],
    pad: float = 0.008,
    iou_thresh: float = 0.02,
) -> list[dict[str, float]]:
    """Merge overlapping / touching boxes within one frame (greedy union-find)."""
    if len(boxes) < 2:
        return [dict(b) for b in boxes]
    expanded = []
    for b in boxes:
        e = dict(b)
        e["x"] = max(0.0, b["x"] - pad)
        e["y"] = max(0.0, b["y"] - pad)
        e["w"] = min(1.0 - e["x"], b["w"] + 2 * pad)
        e["h"] = min(1.0 - e["y"], b["h"] + 2 * pad)
        expanded.append(e)
    parent = list(range(len(expanded)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(expanded)):
        for j in range(i + 1, len(expanded)):
            if _iou(expanded[i], expanded[j]) > iou_thresh:
                parent[find(i)] = find(j)
    groups: dict[int, dict[str, float]] = {}
    for i, b in enumerate(expanded):
        r = find(i)
        groups[r] = _union(groups[r], b) if r in groups else dict(b)
    return list(groups.values())


def _detect_frame(
    reader: Any,
    img_path: Path,
    img_w: int,
    img_h: int,
    min_conf: float,
) -> list[dict[str, float]]:
    """OCR one frame → merged CJK boxes in normalized (0..1) coordinates."""
    boxes: list[dict[str, float]] = []
    for bbox, text, conf in reader.readtext(str(img_path)):
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f < min_conf or not has_cjk(text):
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x0 = max(0.0, min(xs)) / img_w
        x1 = min(float(img_w), max(xs)) / img_w
        y0 = max(0.0, min(ys)) / img_h
        y1 = min(float(img_h), max(ys)) / img_h
        if x1 - x0 < 0.004 or y1 - y0 < 0.004:
            continue
        boxes.append(
            {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
             "text": str(text), "conf": round(conf_f, 3)}
        )
    return _merge_boxes(boxes)


def _track_boxes(
    boxes_by_time: list[tuple[float, list[dict[str, float]]]],
    interval_s: float,
    max_gap_s: float = 8.0,
    min_dur_s: float = 1.0,
    match_iou: float = 0.20,
    bridge_iou: float = 0.25,
) -> list[dict[str, Any]]:
    """Link per-frame boxes into tracks {box, t0, t1}; bridge short gaps."""
    tracks: list[dict[str, Any]] = []
    for t, boxes in boxes_by_time:
        for b in boxes:
            best = None
            best_iou = 0.0
            for tr in tracks:
                if t - tr["t1"] > interval_s * 2.0 + 0.5:
                    continue  # too old to be the same text
                iou = _iou(b, tr["box"])
                if iou > best_iou:
                    best, best_iou = tr, iou
            if best is not None and best_iou >= match_iou:
                best["box"] = _union(best["box"], b)
                best["t1"] = t
            else:
                tracks.append({"box": dict(b), "t0": t, "t1": t})
    # Bridge gaps: same text zone that blinked for a few seconds.
    tracks.sort(key=lambda tr: tr["t0"])
    merged = True
    while merged:
        merged = False
        out: list[dict[str, Any]] = []
        for tr in tracks:
            if out:
                prev = out[-1]
                gap = tr["t0"] - prev["t1"]
                if 0 <= gap <= max_gap_s and _iou(prev["box"], tr["box"]) >= bridge_iou:
                    prev["box"] = _union(prev["box"], tr["box"])
                    prev["t1"] = tr["t1"]
                    merged = True
                    continue
            out.append(tr)
        tracks = out
    return [tr for tr in tracks if tr["t1"] - tr["t0"] >= min_dur_s - 1e-9]


def _to_output_box(
    track: dict[str, Any],
    pad_xy: float = 0.015,
    pad_t: float = 1.0,
) -> dict[str, Any]:
    b = track["box"]
    x = max(0.0, b["x"] - pad_xy)
    y = max(0.0, b["y"] - pad_xy)
    w = min(1.0 - x, b["w"] + 2 * pad_xy)
    h = min(1.0 - y, b["h"] + 2 * pad_xy)
    box = {
        "x": round(x, 4), "y": round(y, 4),
        "w": round(w, 4), "h": round(h, 4),
        "t0": round(max(0.0, track["t0"] - pad_t), 3),
        "t1": round(track["t1"] + pad_t, 3),
    }
    if "text" in b:
        box["text"] = b["text"]
    return box


def detect_cjk_boxes(
    ffmpeg: str,
    video_path: str | Path,
    work_dir: str | Path,
    *,
    log: LogFn = _noop,
    interval_s: float = 2.0,
    min_conf: float = 0.35,
    sample_width: int = 960,
    max_gap_s: float = 8.0,
    min_dur_s: float = 1.0,
    max_boxes: int = 80,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Find Chinese on-screen text in *video_path*.

    Returns time-ranged blur boxes ready for core.apply_production_overlays().
    Raises RuntimeError if easyocr is missing; never raises on empty video.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    frames_dir = work_dir / "cjk_frames"
    try:
        frames = _extract_frames(
            ffmpeg, video_path, interval_s, sample_width, frames_dir, log)
        if not frames:
            return []
        log("🔍 Loading OCR model (first run downloads ~100-150 MB)…")
        reader = _easyocr_reader()
        from PIL import Image

        boxes_by_time: list[tuple[float, list[dict[str, float]]]] = []
        total = len(frames)
        for i, fp in enumerate(frames):
            if should_stop and should_stop():
                log("🔍 OCR stopped by user.")
                return []
            t = i * interval_s
            with Image.open(fp) as im:
                img_w, img_h = im.size
            boxes = _detect_frame(reader, fp, img_w, img_h, min_conf)
            boxes_by_time.append((t, boxes))
            if (i + 1) % 20 == 0 or i + 1 == total:
                log(f"🔍 OCR {i + 1}/{total} frames…")
        tracks = _track_boxes(
            boxes_by_time, interval_s,
            max_gap_s=max_gap_s, min_dur_s=min_dur_s)
        # Longest-lived zones first; cap the count so the ffmpeg graph stays small.
        tracks.sort(key=lambda tr: tr["t1"] - tr["t0"], reverse=True)
        boxes_out = [_to_output_box(tr, pad_t=interval_s * 0.75)
                     for tr in tracks[:max_boxes]]
        boxes_out.sort(key=lambda b: b["t0"])
        log(f"🔍 CJK text zones: {len(boxes_out)}")
        return boxes_out
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def boxes_to_json(boxes: list[dict[str, Any]]) -> str:
    return json.dumps(boxes, ensure_ascii=False, indent=1)
