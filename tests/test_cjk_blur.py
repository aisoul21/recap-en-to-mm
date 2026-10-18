"""Chinese text auto-blur (cjk_blur.py + core._blur_enable_expr): CJK filter,
box merging, temporal tracking, gap bridging, and the time-ranged ffmpeg
enable expression. Needs no network and no OCR models (reader is stubbed)."""
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from cjk_blur import (
    _detect_frame,
    _iou,
    _merge_boxes,
    _to_output_box,
    _track_boxes,
    has_cjk,
)
from core import _blur_enable_expr


def _b(x, y, w, h, **kw):
    d = {"x": x, "y": y, "w": w, "h": h}
    d.update(kw)
    return d


class FakeReader:
    """Stub EasyOCR reader: scripted detections per image path."""

    def __init__(self, script):
        self.script = script  # {basename: [(bbox4, text, conf), ...]}

    def readtext(self, img_path):
        return self.script.get(Path(img_path).name, [])


def _quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class TestHasCjk(unittest.TestCase):
    def test_chinese(self):
        self.assertTrue(has_cjk("你好世界"))

    def test_mixed(self):
        self.assertTrue(has_cjk("hello 世界"))

    def test_english_only(self):
        self.assertFalse(has_cjk("hello world"))

    def test_empty(self):
        self.assertFalse(has_cjk(""))
        self.assertFalse(has_cjk(None))


class TestMergeBoxes(unittest.TestCase):
    def test_overlapping_merges(self):
        boxes = [_b(0.10, 0.10, 0.10, 0.05), _b(0.15, 0.10, 0.10, 0.05)]
        merged = _merge_boxes(boxes)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["x"], 0.10 - 0.008, places=4)

    def test_distant_stays_separate(self):
        boxes = [_b(0.10, 0.10, 0.05, 0.05), _b(0.80, 0.80, 0.05, 0.05)]
        self.assertEqual(len(_merge_boxes(boxes)), 2)

    def test_single_passthrough(self):
        boxes = [_b(0.10, 0.10, 0.05, 0.05)]
        self.assertEqual(len(_merge_boxes(boxes)), 1)


class TestDetectFrame(unittest.TestCase):
    def test_cjk_kept_english_dropped(self):
        reader = FakeReader({
            "frm_00001.jpg": [
                (_quad(10, 10, 100, 30), "中文字幕", 0.9),
                (_quad(200, 10, 100, 30), "hello", 0.9),
                (_quad(10, 200, 100, 30), "你好", 0.1),  # below min_conf
            ],
        })
        boxes = _detect_frame(reader, Path("frm_00001.jpg"), 960, 540, 0.35)
        # high-conf CJK kept as normal; weak CJK kept flagged (extends only)
        self.assertEqual(len(boxes), 2)
        by_text = {b["text"]: b for b in boxes}
        self.assertFalse(by_text["中文字幕"]["lc"])
        self.assertTrue(by_text["你好"]["lc"])
        # merge pads boxes by 0.008
        self.assertAlmostEqual(by_text["中文字幕"]["x"], 10 / 960 - 0.008, places=4)


class TestTrackBoxes(unittest.TestCase):
    def test_continuous_track(self):
        seq = [(t, [_b(0.2, 0.2, 0.3, 0.1)]) for t in (0.0, 2.0, 4.0, 6.0)]
        tracks = _track_boxes(seq, 2.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t0"], 0.0)
        self.assertEqual(tracks[0]["t1"], 6.0)

    def test_gap_bridged(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (2.0, []),  # blinked for one sample
            (4.0, [_b(0.2, 0.2, 0.3, 0.1)]),
        ]
        tracks = _track_boxes(seq, 2.0, max_gap_s=8.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)

    def test_long_gap_stays_split(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (2.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (20.0, [_b(0.2, 0.2, 0.3, 0.1)]),
            (22.0, [_b(0.2, 0.2, 0.3, 0.1)]),
        ]
        tracks = _track_boxes(seq, 2.0, max_gap_s=8.0)
        self.assertEqual(len(tracks), 2)

    def test_single_frame_dropped_as_noise(self):
        seq = [(0.0, []), (2.0, [_b(0.5, 0.5, 0.2, 0.1)]), (4.0, [])]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(tracks, [])

    def test_two_zones_tracked_separately(self):
        seq = [(t, [_b(0.1, 0.1, 0.2, 0.1), _b(0.7, 0.7, 0.2, 0.1)])
               for t in (0.0, 2.0, 4.0)]
        tracks = _track_boxes(seq, 2.0)
        self.assertEqual(len(tracks), 2)


class TestOutputBox(unittest.TestCase):
    def test_padding_and_clamp(self):
        tr = {"box": _b(0.0, 0.0, 0.5, 0.5), "t0": 0.5, "t1": 4.0}
        out = _to_output_box(tr, pad_xy=0.015, pad_t=1.0)
        self.assertEqual(out["x"], 0.0)  # clamped, not negative
        self.assertEqual(out["t0"], 0.0)  # clamped, not negative
        self.assertAlmostEqual(out["w"], 0.53, places=3)
        self.assertAlmostEqual(out["t1"], 5.0, places=3)


class TestBlurEnableExpr(unittest.TestCase):
    def test_ranged(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2,
                               "t0": 1.5, "t1": 4.0}),
            ":enable='between(t,1.500,4.000)'",
        )

    def test_legacy_no_time(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}), "")

    def test_bad_range(self):
        self.assertEqual(
            _blur_enable_expr({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2,
                               "t0": 5.0, "t1": 5.0}), "")


class TestIou(unittest.TestCase):
    def test_identical(self):
        b = _b(0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(_iou(b, b), 1.0)

    def test_disjoint(self):
        self.assertEqual(_iou(_b(0.0, 0.0, 0.1, 0.1), _b(0.9, 0.9, 0.1, 0.1)), 0.0)


if __name__ == "__main__":
    unittest.main()

class TestLowConfExtension(unittest.TestCase):
    def test_weak_box_extends_track(self):
        seq = [
            (0.0, [_b(0.2, 0.2, 0.3, 0.1, conf=0.9, lc=False)]),
            (2.0, [_b(0.2, 0.2, 0.3, 0.1, conf=0.9, lc=False)]),
            # weak box (bad recognition) at the same spot extends the track
            (4.0, [_b(0.21, 0.21, 0.28, 0.09, conf=0.08, lc=True)]),
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)

    def test_weak_box_alone_starts_nothing(self):
        seq = [
            (0.0, []),
            (2.0, [_b(0.5, 0.5, 0.2, 0.1, conf=0.08, lc=True)]),
            (4.0, []),
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(tracks, [])

    def test_merged_high_plus_weak_can_start_track(self):
        from cjk_blur import _merge_boxes
        merged = _merge_boxes([
            _b(0.20, 0.20, 0.10, 0.05, conf=0.9, lc=False),
            _b(0.22, 0.20, 0.10, 0.05, conf=0.08, lc=True),
        ])
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0]["lc"])

class TestLastBoxMatching(unittest.TestCase):
    def test_short_line_matches_previous_short_line(self):
        # union grows wide from an early long line; later short lines must
        # still match via the most recent box, not the wide union.
        seq = [
            (0.0, [_b(0.20, 0.80, 0.60, 0.10, conf=0.9)]),   # long line
            (2.0, [_b(0.40, 0.80, 0.10, 0.10, conf=0.9)]),   # short line
            (4.0, [_b(0.41, 0.80, 0.10, 0.10, conf=0.9)]),   # short line again
        ]
        tracks = _track_boxes(seq, 2.0, min_dur_s=1.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["t1"], 4.0)
