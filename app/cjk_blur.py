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
    # Some networks/PCs fail SSL verification for that download
    # (CERTIFICATE_VERIFY_FAILED) — bypass verification for the model
    # download only. These are public ML weights, not sensitive data.
    try:
        import ssl as _ssl

        _ssl._create_default_https_context = _ssl._create_unverified_context
    except Exception:
        pass
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
    out["lc"] = bool(a.get("lc", False) and b.get("lc", False))
    out["ss"] = bool(a.get("ss", False) or b.get("ss", False))
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


LOW_CONF = 0.01  # boxes this weak may extend a track, never start one


def _load_pixels(img_path: Path) -> tuple[Any, int, int]:
    """Load an image as an RGB numpy array; returns (pixels, w, h).

    Uses PIL — never cv2.imread — because OpenCV returns None for
    non-ASCII paths on Windows (e.g. a job folder named after a Chinese
    video). EasyOCR's internal cv2.imread() hit exactly that and crashed
    with "'NoneType' object has no attribute 'shape'", which skipped the
    entire blur pass for the video.
    """
    from PIL import Image
    import numpy as np

    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        return np.array(rgb), rgb.width, rgb.height


def _detect_frame(
    reader: Any,
    img_path: Path,
    img_w: int,
    img_h: int,
    min_conf: float,
) -> list[dict[str, float]]:
    """OCR one frame → merged CJK boxes in normalized (0..1) coordinates.

    Boxes weaker than LOW_CONF are dropped — UNLESS their shape screams
    "text": very large (disclaimers/titles) or tall-narrow (vertical text).
    Weak boxes (conf < min_conf) without a strong shape are flagged ("lc"):
    they may extend an existing nearby track but never start a new one.
    """
    pixels, _, _ = _load_pixels(img_path)
    return _detect_pixels(reader, pixels, img_w, img_h, min_conf)


def _detect_pixels(
    reader: Any,
    pixels: Any,
    img_w: int,
    img_h: int,
    min_conf: float,
) -> list[dict[str, float]]:
    """Same as _detect_frame but takes image pixels (numpy array).

    EasyOCR receives the array directly — no file path ever reaches its
    internal cv2.imread(), so non-ASCII (Chinese) paths can't break it.
    """
    boxes: list[dict[str, float]] = []
    for bbox, text, conf in reader.readtext(pixels):
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x0 = max(0.0, min(xs)) / img_w
        x1 = min(float(img_w), max(xs)) / img_w
        y0 = max(0.0, min(ys)) / img_h
        y1 = min(float(img_h), max(ys)) / img_h
        if x1 - x0 < 0.004 or y1 - y0 < 0.004:
            continue
        w, h = x1 - x0, y1 - y0
        strong_shape = (w * h > 0.03) or (h > 2.5 * w and h > 0.08)
        # Strong-shape regions are kept even when recognition fails
        # completely (stylized vertical text) — but only when the region
        # does NOT bleed off a frame edge. Real overlays always have
        # padding; branches/window-frames often touch the edge.
        touches_edge = (x0 < 0.01 or y0 < 0.02 or x1 > 0.99 or y1 > 0.98)
        if not has_cjk(text) and not (strong_shape and not touches_edge):
            continue
        if conf_f < LOW_CONF and not strong_shape:
            continue
        boxes.append(
            {"x": x0, "y": y0, "w": w, "h": h,
             "text": str(text), "conf": round(conf_f, 3),
             "lc": conf_f < min_conf and not strong_shape,
             "ss": strong_shape}
        )
    return _merge_boxes(boxes)


def _track_boxes(
    boxes_by_time: list[tuple[float, list[dict[str, float]]]],
    interval_s: float,
    max_gap_s: float = 20.0,
    min_dur_s: float = 1.0,
    match_iou: float = 0.20,
    bridge_iou: float = 0.10,
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
                # Match against the most recent box, not the whole-track
                # union: a short subtitle line still matches the previous
                # short line even after the union grew wide.
                iou = _iou(b, tr["last"])
                if iou > best_iou:
                    best, best_iou = tr, iou
            if best is not None and best_iou >= match_iou:
                best["box"] = _union(best["box"], b)
                best["last"] = dict(b)
                best["ss"] = best.get("ss", False) or b.get("ss", False)
                best["t1"] = t
            elif not b.get("lc"):
                tracks.append({"box": dict(b), "last": dict(b),
                               "t0": t, "t1": t,
                               "ss": b.get("ss", False)})
            # else: weak box with no nearby track — likely noise, drop it
    # Bridge gaps: same text zone that blinked for a few seconds.
    # Every earlier track is a merge candidate (not just the previous one):
    # different text elements (subtitles + side name cards) often interleave
    # in time, so the matching track may not be the immediate predecessor.
    tracks.sort(key=lambda tr: tr["t0"])
    merged = True
    while merged:
        merged = False
        out: list[dict[str, Any]] = []
        for tr in tracks:
            best = None
            best_iou = 0.0
            for prev in out:
                gap = tr["t0"] - prev["t1"]
                if 0 <= gap <= max_gap_s:
                    iou = _iou(prev["box"], tr["box"])
                    if iou >= bridge_iou and iou > best_iou:
                        best, best_iou = prev, iou
            if best is not None:
                best["box"] = _union(best["box"], tr["box"])
                best["t1"] = tr["t1"]
                best["ss"] = best.get("ss", False) or tr.get("ss", False)
                merged = True
            else:
                out.append(tr)
        tracks = out
    # Strong-shape tracks (big titles, vertical text) survive even from a
    # single frame — brief name cards appear for ~2s and must not be lost.
    return [tr for tr in tracks
            if tr["t1"] - tr["t0"] >= min_dur_s - 1e-9 or tr.get("ss")]


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


def _ocr_one_frame(
    reader: Any,
    img_path: Path,
    min_conf: float,
    log: LogFn,
    frame_no: int = 0,
    total: int = 0,
) -> list[dict[str, float]]:
    """OCR a single frame; never raises.

    One unreadable/corrupt frame must not kill the whole blur pass —
    log it and move on (gap-bridging in _track_boxes covers the hole).
    """
    try:
        pixels, img_w, img_h = _load_pixels(img_path)
        return _detect_pixels(reader, pixels, img_w, img_h, min_conf)
    except Exception as exc:
        tag = f" {frame_no}/{total}" if total else ""
        log(f"⚠️ OCR frame{tag} failed ({exc}); continuing with the rest.")
        return []


def detect_cjk_boxes(
    ffmpeg: str,
    video_path: str | Path,
    work_dir: str | Path,
    *,
    log: LogFn = _noop,
    interval_s: float = 2.0,
    min_conf: float = 0.25,
    sample_width: int = 960,
    max_gap_s: float = 20.0,
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

        boxes_by_time: list[tuple[float, list[dict[str, float]]]] = []
        total = len(frames)
        for i, fp in enumerate(frames):
            if should_stop and should_stop():
                log("🔍 OCR stopped by user.")
                return []
            t = i * interval_s
            boxes = _ocr_one_frame(reader, fp, min_conf, log, i + 1, total)
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
